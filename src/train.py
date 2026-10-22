from __future__ import annotations
from ast import Tuple, List
import os, sys, time, argparse, csv
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback)
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment import A1SitToStandEnv

_ROOT       = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _ROOT / "output"
_LOG_DIR    = _OUTPUT_DIR / "logs"
_MODEL_DIR  = _OUTPUT_DIR / "models"
_STATS_DIR  = _OUTPUT_DIR / "vecnorm_stats"


class TrainingProgressCallback(BaseCallback):
    """Prints one-line summary every print_freq rollouts."""
    def __init__(self, print_freq=5, verbose=0):
        super().__init__(verbose)
        self.print_freq  = print_freq
        self._n_rollouts = 0
        self._t0         = None

    def _on_training_start(self):
        self._t0 = time.time()
        total = self.locals.get("total_timesteps", "?")
        print(f"\n{'='*65}")
        print(f"  PPO Training | Steps: {total} | Device: {self.model.device}")
        print(f"{'='*65}")

    def _on_rollout_end(self):
        self._n_rollouts += 1
        if self._n_rollouts % self.print_freq == 0:
            elapsed = time.time() - self._t0
            fps     = self.num_timesteps / max(elapsed, 1e-6)
            ep_rew  = (self.model.ep_info_buffer[-1]['r']
                       if len(self.model.ep_info_buffer) > 0 else float('nan'))
            print(f"[Rollout {self._n_rollouts:4d}] steps={self.num_timesteps:8d} "
                  f"fps={fps:6.0f} ep_rew={ep_rew:8.2f} elapsed={elapsed:6.0f}s")
        return True

    def _on_step(self): return True


class TerrainSuccessEvalCallback(BaseCallback):
    """Evaluate task success on each terrain plus out-of-distribution difficulty.

    The policy receives only normalized proprioceptive observations. Terrain type,
    rise time and success labels are used by this evaluator only, never as policy input.
    """
    def __init__(self, eval_freq, n_episodes, output_dir, seed=10000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = max(1, int(eval_freq))
        self.n_episodes = int(n_episodes)
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.last_eval_step = 0
        self.best_success_rate = -1.0
        self.cases = [
            ("flat", "flat", 1.0),
            ("slope", "slope", 1.0),
            ("stairs", "stairs", 1.0),
            ("rough", "rough", 1.0),
            ("slope_ood", "slope", 1.2),
            ("stairs_ood", "stairs", 1.2),
            ("rough_ood", "rough", 1.2),
        ]

    def _on_training_start(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "terrain_success_eval.csv"
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timesteps", "terrain_case", "terrain_type", "curriculum_level",
                    "episodes", "successes", "success_rate", "mean_rise_time_sec"])

    def _evaluate_case(self, case_idx, label, terrain_type, level):
        successes, rise_times = 0, []
        was_training = getattr(self.training_env, "training", True)
        was_norm_reward = getattr(self.training_env, "norm_reward", True)
        try:
            self.training_env.training = False
            self.training_env.norm_reward = False
            for ep in range(self.n_episodes):
                env = A1SitToStandEnv(
                    terrain_type=terrain_type, curriculum_level=level,
                    enable_domain_randomization=False,
                    enable_inertia_randomization=False,
                    use_discrete_scenes=True,
                    render_mode=None,
                    seed=self.seed + self.num_timesteps + case_idx * 100 + ep)
                try:
                    obs, info = env.reset(seed=self.seed + self.num_timesteps + case_idx * 100 + ep)
                    done = False
                    while not done:
                        obs_input = np.asarray(obs, dtype=np.float32).reshape(1, -1)
                        normalized_obs = self.training_env.normalize_obs(obs_input)
                        action, _ = self.model.predict(normalized_obs, deterministic=True)
                        act = action[0] if action.ndim > 1 else action
                        obs, reward, terminated, truncated, info = env.step(act)
                        done = bool(terminated or truncated)
                    if bool(info.get("success", False)):
                        successes += 1
                        rt = info.get("rise_time_sec")
                        if rt is not None:
                            rise_times.append(float(rt))
                finally:
                    env.close()
        finally:
            self.training_env.training = was_training
            self.training_env.norm_reward = was_norm_reward
        return {
            "timesteps": self.num_timesteps,
            "terrain_case": label,
            "terrain_type": terrain_type,
            "curriculum_level": level,
            "episodes": self.n_episodes,
            "successes": successes,
            "success_rate": successes / max(1, self.n_episodes),
            "mean_rise_time_sec": (float(np.mean(rise_times)) if rise_times else ""),
        }

    def _on_step(self):
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = self.num_timesteps
        rows = [self._evaluate_case(i, *case) for i, case in enumerate(self.cases)]
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timesteps", "terrain_case", "terrain_type", "curriculum_level",
                "episodes", "successes", "success_rate", "mean_rise_time_sec"])
            writer.writerows(rows)
        mean_success = float(np.mean([r["success_rate"] for r in rows]))
        if self.verbose:
            print(f"\n[Terrain Eval @ {self.num_timesteps:,} steps] mean success={mean_success:.1%}")
            for row in rows:
                rt = row["mean_rise_time_sec"]
                rt_str = f"{rt:.2f}s" if isinstance(rt, (float, int)) else "n/a"
                print(f"  {row['terrain_case']:<12} success={row['success_rate']:.1%} "
                      f"({row['successes']}/{row['episodes']}), mean rise={rt_str}")
        if mean_success >= self.best_success_rate:
            self.best_success_rate = mean_success
            best_path = self.output_dir / "best_by_terrain_success"
            self.model.save(str(best_path))
            # Keep matching observation normalization statistics with this policy.
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(str(self.output_dir / "best_by_terrain_success_vecnorm.pkl"))
            if self.verbose:
                print(f"  Saved best multi-terrain policy: {best_path}.zip")
        return True


class CurriculumCallback(BaseCallback):
    """Keep short smoke runs easy; ramp terrain only after a useful warm-up."""
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        # Short diagnostic runs (e.g. 10k steps) stay at level 0 rather than
        # reaching maximum terrain difficulty before PPO has learned anything.
        self.warmup_steps = max(50_000, int(0.10 * total_timesteps))
        self.ramp_steps = max(500_000, int(0.65 * total_timesteps))
        self._last_level = -1.0

    def _on_step(self) -> bool:
        level = float(np.clip((self.num_timesteps - self.warmup_steps) / self.ramp_steps, 0.0, 1.0))
        if abs(level - self._last_level) > 0.05:
            self._last_level = level
            self.training_env.env_method('set_curriculum_level', level)
            if self.verbose > 0:
                print(f"[Curriculum] Advanced difficulty level to {level:.2f}")
        return True


CURRICULUM_STAGES = ["flat", "slope", "rough", "stairs"]


def evaluate_stage_success(model: PPO, training_vec_env: VecNormalize, terrain_type: str,
                           n_episodes: int = 50, seed: int = 20000) -> Tuple[float, List[float]]:
    """Evaluate current policy on target stage terrain without altering training env stats.

    Guarantees:
      - Uses existing training observation-normalization statistics without updating them.
      - Training state of training_vec_env (training flag, norm_obs) is preserved.
      - Observations are reshaped to (1, obs_dim) for single-step prediction.
      - Success requires valid rise in [2.0, 5.0] s and 1.0 s stable hold.
    """
    successes = 0
    rise_times = []

    # Preserve training_vec_env mode
    was_training = getattr(training_vec_env, "training", True)
    was_norm_reward = getattr(training_vec_env, "norm_reward", True)
    try:
        training_vec_env.training = False
        training_vec_env.norm_reward = False

        for ep in range(n_episodes):
            test_seed = seed + ep * 137
            env = A1SitToStandEnv(
                terrain_type=terrain_type,
                curriculum_level=1.0,
                enable_domain_randomization=False,
                enable_inertia_randomization=False,
                use_discrete_scenes=True,
                render_mode=None,
                seed=test_seed,
            )
            try:
                obs, info = env.reset(seed=test_seed)
                done = False
                while not done:
                    # Ensure observation has batch dimension (1, 27) for vectorized normalization
                    obs_input = np.asarray(obs, dtype=np.float32).reshape(1, -1)
                    norm_obs = training_vec_env.normalize_obs(obs_input)
                    action, _ = model.predict(norm_obs, deterministic=True)
                    # Unpack 1D action for single environment
                    act = action[0] if action.ndim > 1 else action
                    obs, reward, terminated, truncated, info = env.step(act)
                    done = bool(terminated or truncated)

                if bool(info.get("success", False)):
                    successes += 1
                    rt = info.get("rise_time_sec")
                    if rt is not None:
                        rise_times.append(float(rt))
            finally:
                env.close()
    finally:
        training_vec_env.training = was_training
        training_vec_env.norm_reward = was_norm_reward

    success_rate = successes / max(1, n_episodes)
    return success_rate, rise_times


def create_stage_env(terrain_type: str, n_envs: int, seed: int, log_dir: Path, run_name: str):
    """Build vectorized environment for a dedicated discrete XML terrain stage."""
    vec_env = make_vec_env(
        env_id=lambda: A1SitToStandEnv(
            terrain_type=terrain_type,
            curriculum_level=1.0,
            enable_domain_randomization=True,
            enable_inertia_randomization=True,
            use_discrete_scenes=True,
        ),
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(log_dir / run_name / terrain_type),
    )
    # Confirm exact XML scene loaded
    sample_env = A1SitToStandEnv(terrain_type=terrain_type, use_discrete_scenes=True)
    xml_basename = os.path.basename(sample_env._model_path)
    print(f"  [create_stage_env] Active Stage: {terrain_type.upper()} -> Loaded XML: {xml_basename} ({sample_env._model_path})")
    sample_env.close()
    return vec_env


def train(total_timesteps=5_000_000, n_envs=4, seed=42,
          run_name="ppo_a1_sit_to_stand",
          success_threshold=0.90,
          eval_window=50,
          eval_freq=25_000,
          max_stage_steps=2_000_000,
          start_stage=0,
          checkpoint_model=None,
          checkpoint_vecnorm=None,
          check_only=False):
    """Train PPO with genuine performance-based curriculum progression:
       Stage 1 (Flat) -> Stage 2 (Slope) -> Stage 3 (Rough) -> Stage 4 (Stairs).

    Transitions occur ONLY when the policy achieves >= success_threshold over
    eval_window rolling evaluation episodes on the active XML stage.
    Policy weights and observation normalization stats are preserved seamlessly across stages.
    """
    for d in (_LOG_DIR, _MODEL_DIR, _STATS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    run_model_dir = _MODEL_DIR / run_name
    run_model_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*75}")
    print(f"  PPO Genuine Staged Curriculum Training Orchestrator")
    print(f"  Stages: {' -> '.join(s.upper() for s in CURRICULUM_STAGES)}")
    print(f"  Device: {device} | Parallel Envs: {n_envs} | RNG Seed: {seed}")
    print(f"  Transition Gate: Success >= {success_threshold*100:.1f}% over {eval_window} evaluation episodes")
    print(f"{'='*75}\n")

    hparams = dict(
        learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5)

    print("PPO Hyperparameters:")
    for k, v in hparams.items():
        print(f"  {k:<20s}: {v}")

    if check_only:
        print("\n[--check-only flag provided] Verifying curriculum setup without training...")
        for idx, stage_name in enumerate(CURRICULUM_STAGES):
            test_env = A1SitToStandEnv(terrain_type=stage_name, use_discrete_scenes=True)
            print(f"  Verified Stage {idx+1}: {stage_name.upper():<7} -> XML: {os.path.basename(test_env._model_path)} | Obs: {test_env.observation_space.shape} | Act: {test_env.action_space.shape}")
            test_env.close()
        print("Curriculum stage definition: VALID & READY.\n")
        return None, None, None

    model = None
    vec_env = None
    current_model_path = checkpoint_model
    current_vecnorm_path = checkpoint_vecnorm

    for stage_idx in range(start_stage, len(CURRICULUM_STAGES)):
        stage_name = CURRICULUM_STAGES[stage_idx]
        print(f"\n{'#'*75}")
        print(f"  ENTERING STAGE {stage_idx+1}/{len(CURRICULUM_STAGES)}: {stage_name.upper()} TERRAIN")
        print(f"{'#'*75}\n")

        raw_vec_env = create_stage_env(stage_name, n_envs, seed + stage_idx * 1000, _LOG_DIR, run_name)
        if current_vecnorm_path and os.path.exists(current_vecnorm_path):
            print(f"  Restoring VecNormalize stats from: {current_vecnorm_path}")
            vec_env = VecNormalize.load(current_vecnorm_path, raw_vec_env)
            vec_env.training = True
            vec_env.norm_reward = True
        else:
            vec_env = VecNormalize(
                raw_vec_env, norm_obs=True, norm_reward=True,
                clip_obs=10.0, clip_reward=10.0, gamma=0.99)

        if model is None:
            if current_model_path and os.path.exists(current_model_path):
                print(f"  Loading pre-existing PPO model from: {current_model_path}")
                model = PPO.load(
                    current_model_path, env=vec_env, device=device,
                    tensorboard_log=str(_LOG_DIR / run_name),
                    **hparams)
            else:
                model = PPO(
                    policy="MlpPolicy", env=vec_env, verbose=1,
                    seed=seed, device=device,
                    tensorboard_log=str(_LOG_DIR / run_name),
                    policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.Tanh),
                    **hparams)
        else:
            print(f"  Preserving learned policy & optimizer across stage transition to {stage_name.upper()}")
            model.set_env(vec_env)

        stage_steps = 0
        stage_passed = False
        stage_iter = 0

        while stage_steps < max_stage_steps and not stage_passed:
            steps_this_cycle = min(eval_freq, max_stage_steps - stage_steps)
            stage_iter += 1
            print(f"\n[Stage {stage_idx+1}: {stage_name.upper()}] Training cycle {stage_iter} ({steps_this_cycle:,} steps)...")

            model.learn(
                total_timesteps=steps_this_cycle,
                reset_num_timesteps=False,
                tb_log_name=f"{run_name}_{stage_name}",
                progress_bar=False,
            )
            stage_steps += steps_this_cycle

            # Rolling evaluation on active stage terrain
            print(f"  Evaluating performance on {stage_name.upper()} ({eval_window} episodes)...")
            succ_rate, r_times = evaluate_stage_success(
                model=model,
                training_vec_env=vec_env,
                terrain_type=stage_name,
                n_episodes=eval_window,
                seed=seed + 50000 + stage_steps,
            )
            mean_rt = f"{np.mean(r_times):.2f}s" if r_times else "N/A"
            print(f"  --> Result: Success Rate = {succ_rate*100:.1f}% (Required: {success_threshold*100:.1f}%) | Mean Rise Time = {mean_rt}")

            # Save stage checkpoint
            stage_ckpt = run_model_dir / f"stage_{stage_idx+1}_{stage_name}_step_{stage_steps}"
            model.save(str(stage_ckpt))
            vec_env.save(str(stage_ckpt) + "_vecnorm.pkl")

            if succ_rate >= success_threshold:
                print(f"\n>>> STAGE {stage_idx+1} ({stage_name.upper()}) MASTERED! "
                      f"Success Rate {succ_rate*100:.1f}% >= Threshold {success_threshold*100:.1f}%. Advancing to next stage.")
                stage_passed = True
            else:
                print(f"  Stage criteria not yet satisfied ({succ_rate*100:.1f}% < {success_threshold*100:.1f}%). Continuing on {stage_name.upper()}...")

        # Save definitive stage checkpoint
        passed_ckpt = run_model_dir / f"stage_{stage_idx+1}_{stage_name}_final"
        model.save(str(passed_ckpt))
        vecnorm_final = str(run_model_dir / f"stage_{stage_idx+1}_{stage_name}_vecnorm.pkl")
        vec_env.save(vecnorm_final)
        current_model_path = str(passed_ckpt)
        current_vecnorm_path = vecnorm_final

        vec_env.close()

        if not stage_passed:
            print(f"\n[CURRICULUM HALTED] Did not master Stage {stage_idx+1} ({stage_name.upper()}) "
                  f"within {max_stage_steps:,} steps. Preserving stage checkpoint for resumption.")
            return model, current_model_path, current_vecnorm_path

    print(f"\n{'='*75}")
    print(f"  ALL CURRICULUM STAGES SUCCESSFULLY MASTERED!")
    print(f"  Final policy: {current_model_path}.zip")
    print(f"  VecNormalize stats: {current_vecnorm_path}")
    print(f"{'='*75}\n")

    return model, current_model_path, current_vecnorm_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train A1 Sit-to-Stand with Staged Curriculum PPO")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs",    type=int, default=4)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--run-name",  type=str, default="ppo_a1_sit_to_stand")
    parser.add_argument("--success-threshold", type=float, default=0.90,
                        help="Required success rate to advance to next stage (default: 0.90)")
    parser.add_argument("--eval-window", type=int, default=50,
                        help="Number of episodes in rolling evaluation window (default: 50)")
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="Number of environment steps between curriculum evaluations (default: 25000)")
    parser.add_argument("--max-stage-steps", type=int, default=2_000_000,
                        help="Maximum training steps per stage before stopping (default: 2000000)")
    parser.add_argument("--start-stage", type=int, default=0,
                        help="Stage index to start from (0: Flat, 1: Slope, 2: Rough, 3: Stairs)")
    parser.add_argument("--checkpoint-model", type=str, default=None,
                        help="Path to pre-existing PPO model zip to resume training")
    parser.add_argument("--checkpoint-vecnorm", type=str, default=None,
                        help="Path to pre-existing VecNormalize pickle to resume training")
    parser.add_argument("--check-only", action="store_true",
                        help="Lightweight verification of curriculum stages without running training")
    args = parser.parse_args()
    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        run_name=args.run_name,
        success_threshold=args.success_threshold,
        eval_window=args.eval_window,
        eval_freq=args.eval_freq,
        max_stage_steps=args.max_stage_steps,
        start_stage=args.start_stage,
        checkpoint_model=args.checkpoint_model,
        checkpoint_vecnorm=args.checkpoint_vecnorm,
        check_only=args.check_only,
    )

