# # Standard evaluation: all four terrains
# python src/evaluate.py --terrain all --episodes 5 --seed 42

# # Held-out robustness tests
# python src/evaluate.py --robustness --episodes 5 --seed 42

# # Both suites
# python src/evaluate.py --all-eval --episodes 5 --seed 42

# # Render the standard tests
# python src/evaluate.py --terrain all --render
from __future__ import annotations
import argparse
import csv
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment import A1SitToStandEnv
from terrain import TerrainGenerator

STAND_HEIGHT_MIN = 0.21
STAND_TILT_MAX = 25.0
STABLE_VEL_LIN = 0.25
STABLE_VEL_ANG = 0.60
STABLE_HEIGHT_VAR = 0.025
STABILITY_WINDOW = 2.0
STAND_SUCCESS_MIN_TIME = 2.0
STAND_SUCCESS_TIME = 5.0

_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _ROOT / "output"
_CSV_DIR = _OUTPUT_DIR / "eval_results"
_BEST_MODEL_DIR = _OUTPUT_DIR / "best_model"
_DEFAULT_MODEL_PATH = _BEST_MODEL_DIR / "best_model.zip"
_DEFAULT_VECNORM_PATH = _BEST_MODEL_DIR / "best_model_vecnorm.pkl"

ROBUSTNESS_CONDITIONS: List[Dict[str, Any]] = [
    {
        "name": "Steeper_Slope_18deg",
        "category": "Slope (Steeper)",
        "terrain_type": "slope",
        "pitch_deg": 18.0,
        "roll_deg": 0.0,
        "description": "Planar slope at 18.0 deg pitch",
    },
    {
        "name": "Steeper_Slope_20deg",
        "category": "Slope (Steeper)",
        "terrain_type": "slope",
        "pitch_deg": 20.0,
        "roll_deg": 0.0,
        "description": "Planar slope at 20.0 deg pitch",
    },
    {
        "name": "Severe_Rough_50mm",
        "category": "Rough (Height Variation)",
        "terrain_type": "rough",
        "roughness": 0.050,
        "smoothness": 9,
        "description": "Rough terrain with 50mm height amplitude",
    },
    {
        "name": "Tall_Stairs_10cm",
        "category": "Stairs (Altered Height)",
        "terrain_type": "stairs",
        "step_height": 0.10,
        "step_width": 0.45,
        "direction": "up",
        "description": "Stairs with 10cm step height spanning steps",
    },
    {
        "name": "Narrow_Stairs_38cm",
        "category": "Stairs (Altered Tread)",
        "terrain_type": "stairs",
        "step_height": 0.06,
        "step_width": 0.38,
        "direction": "up",
        "description": "Stairs with 38cm tread depth spanning steps",
    },
    {
        "name": "Low_Friction_0.20",
        "category": "Friction Variation",
        "terrain_type": "flat",
        "override_friction": 0.20,
        "description": "Slick surface with friction mu=0.20",
    },
    {
        "name": "High_Payload_Mass_30pct",
        "category": "Mass Variation",
        "terrain_type": "flat",
        "override_trunk_mass": 6.123,
        "description": "Trunk payload mass +30% at 6.12kg",
    },
]


class RobustnessA1Env(A1SitToStandEnv):
    """Subclass of A1SitToStandEnv for testing held-out out-of-distribution conditions."""

    def __init__(self, condition_name: str, condition_config: Dict[str, Any], **kwargs):
        self.condition_name = condition_name
        self.condition_config = condition_config
        kwargs["use_discrete_scenes"] = condition_config.get("use_discrete_scenes", False)
        if "override_friction" in condition_config or "override_trunk_mass" in condition_config:
            kwargs["enable_domain_randomization"] = False
            kwargs["enable_inertia_randomization"] = False
        super().__init__(**kwargs)

    def _setup_terrain(self):
        c_type = self.condition_config.get("terrain_type", self.terrain_type)
        if c_type == "slope":
            pitch = float(self.condition_config.get("pitch_deg", 18.0))
            roll = float(self.condition_config.get("roll_deg", 0.0))
            self.terrain.generate_slope(pitch_deg=pitch, roll_deg=roll)
        elif c_type == "rough":
            roughness = float(self.condition_config.get("roughness", 0.050))
            smoothness = int(self.condition_config.get("smoothness", 9))
            self.terrain.generate_rough(roughness=roughness, smoothness=smoothness, rng=self._rng)
        elif c_type == "stairs":
            step_h = float(self.condition_config.get("step_height", 0.10))
            step_w = float(self.condition_config.get("step_width", 0.45))
            direction = str(self.condition_config.get("direction", "up"))
            self.terrain.generate_stairs(step_height=step_h, step_width=step_w, direction=direction)
        elif c_type == "flat":
            self.terrain.generate_flat()
        else:
            super()._setup_terrain()

        if hasattr(self.model, "hfield_data") and self.model.nhfield > 0:
            self.model.hfield_data[:] = self.terrain._to_hfield_data()

    def _apply_domain_randomization(self):
        super()._apply_domain_randomization()
        if "override_friction" in self.condition_config:
            mu = float(self.condition_config["override_friction"])
            self.model.geom_friction[self._terrain_geom_id, 0] = mu
            for gid in self._terrain_geom_ids:
                self.model.geom_friction[gid, 0] = mu
            mujoco.mj_setConst(self.model, self.data)
        if "override_trunk_mass" in self.condition_config:
            mass = float(self.condition_config["override_trunk_mass"])
            self.model.body_mass[self._trunk_id] = mass
            mujoco.mj_setConst(self.model, self.data)


def evaluate_episode(raw_env: A1SitToStandEnv, model: PPO, vec_env: VecNormalize,
                     condition_type: str = "standard", condition_name: str = "unknown",
                     episode_num: int = 1, seed: int = 0,
                     render: bool = False) -> Dict[str, Any]:
    """Execute one deterministic evaluation episode and collect performance metrics."""
    obs = vec_env.reset()
    done = False
    step = 0
    ctrl_dt = raw_env.control_dt
    stable_window_steps = max(1, int(STABILITY_WINDOW / ctrl_dt))

    sit_joints_err = float(np.max(np.abs(raw_env._get_joint_pos() - raw_env._sit_joints)))
    base_pos = raw_env._get_base_pos()
    ground_z = raw_env._get_terrain_ground_z(base_pos)
    h_rel_init = float(base_pos[2] - ground_z)
    sitting_pose_verified = bool(sit_joints_err <= 0.30 and h_rel_init < 0.23 and not raw_env._standing)

    if render:
        raw_env.render()
        time.sleep(0.5)

    heights, tilts_deg, lin_vels, ang_vels = [], [], [], []
    first_stand_step = None
    first_stable_step = None
    consecutive_stable = 0
    max_consec_stable = 0
    stable_achieved = False
    termination_reason = "unknown"
    env_success = False
    last_info = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated_arr, info_list = vec_env.step(action)
        info = info_list[0] if info_list else {}
        last_info = info
        step += 1

        if render:
            time.sleep(ctrl_dt)

        height = float(info.get("h_rel", 0.0) if info.get("h_rel") is not None else 0.0)
        tilt_deg = float(info.get("tilt_deg", 0.0) if info.get("tilt_deg") is not None else 0.0)
        is_standing = bool(info.get("is_standing", False))
        is_stable = bool(info.get("is_stable", False))

        v = np.array(raw_env.data.qvel[0:3]) if hasattr(raw_env, "data") else np.zeros(3)
        w = np.array(raw_env.data.qvel[3:6]) if hasattr(raw_env, "data") else np.zeros(3)
        lin_vel = float(np.linalg.norm(v))
        ang_vel = float(np.linalg.norm(w))

        heights.append(height)
        tilts_deg.append(tilt_deg)
        lin_vels.append(lin_vel)
        ang_vels.append(ang_vel)

        if is_standing and first_stand_step is None:
            first_stand_step = step

        if is_stable:
            consecutive_stable += 1
            if first_stable_step is None:
                first_stable_step = step
        else:
            max_consec_stable = max(max_consec_stable, consecutive_stable)
            consecutive_stable = 0

        if consecutive_stable >= stable_window_steps and not stable_achieved:
            stable_achieved = True

        if bool(info.get("success", False)):
            env_success = True

        done_flag = bool(terminated_arr[0])
        truncated = bool(info.get("truncated", False) or info.get("TimeLimit.truncated", False))
        terminated = bool(info.get("terminated", False)) or (done_flag and not truncated)
        done = done_flag or terminated or truncated

        if terminated:
            reason = info.get("termination_reason")
            if reason:
                termination_reason = str(reason)
            elif info.get("success", False):
                termination_reason = "success"
            else:
                termination_reason = "fall"
        elif truncated:
            termination_reason = "timeout"

    max_consec_stable = max(max_consec_stable, consecutive_stable)
    heights = np.array(heights)
    tilts_deg = np.array(tilts_deg)
    lin_vels = np.array(lin_vels)
    ang_vels = np.array(ang_vels)
    total_time = step * ctrl_dt

    env_rise_time = last_info.get("rise_time_sec")
    if env_rise_time is not None and not (isinstance(env_rise_time, float) and math.isnan(env_rise_time)):
        t2s = float(env_rise_time)
    elif first_stand_step is not None:
        t2s = float(first_stand_step * ctrl_dt)
    else:
        t2s = None

    t2_stable = float(first_stable_step * ctrl_dt) if first_stable_step is not None else None
    stand_success = (t2s is not None and STAND_SUCCESS_MIN_TIME <= t2s <= STAND_SUCCESS_TIME)

    window = max(1, stable_window_steps)
    fh = heights[-window:] if len(heights) > 0 else np.array([0.0])
    ft = tilts_deg[-window:] if len(tilts_deg) > 0 else np.array([0.0])
    fv = lin_vels[-window:] if len(lin_vels) > 0 else np.array([0.0])
    fw = ang_vels[-window:] if len(ang_vels) > 0 else np.array([0.0])

    balance_maintained = (
        stand_success and stable_achieved
        and float(np.mean(fh)) >= STAND_HEIGHT_MIN
        and float(np.mean(ft)) < STAND_TILT_MAX
        and float(np.mean(fv)) < STABLE_VEL_LIN
        and float(np.mean(fw)) < STABLE_VEL_ANG
        and float(np.std(fh)) < STABLE_HEIGHT_VAR
    )

    failure_reason = "none" if env_success else (termination_reason if termination_reason != "unknown" else "timeout")
    terrain_label = last_info.get("terrain_type", getattr(getattr(raw_env, "terrain", None), "current_type", "unknown"))

    return {
        "condition_type": condition_type,
        "condition_name": condition_name,
        "episode": episode_num,
        "seed": seed,
        "terrain_type": terrain_label,
        "sitting_pose_verified": int(sitting_pose_verified),
        "stand_success": int(stand_success),
        "stable_achieved": int(stable_achieved),
        "environment_success": int(env_success),
        "balance_maintained": int(balance_maintained),
        "time_to_stand_s": t2s if t2s is not None else float("nan"),
        "time_to_stabilize_s": t2_stable if t2_stable is not None else float("nan"),
        "max_consec_stable_s": max_consec_stable * ctrl_dt,
        "final_height_m": float(np.mean(fh)),
        "final_tilt_deg": float(np.mean(ft)),
        "final_ang_vel_rads": float(np.mean(fw)),
        "height_var_m": float(np.std(fh)),
        "total_steps": step,
        "total_time_s": total_time,
        "termination": termination_reason,
        "failure_reason": failure_reason,
    }


def compute_condition_summary(metrics: List[Dict[str, Any]], condition_type: str,
                              condition_name: str, category: str) -> Dict[str, Any]:
    """Calculate aggregated metrics for a completed evaluation condition."""
    total = len(metrics)
    if total == 0:
        return {
            "condition_type": condition_type,
            "condition_name": condition_name,
            "category": category,
            "episodes": 0,
            "success_count": 0,
            "success_rate_pct": 0.0,
            "stand_rate_pct": 0.0,
            "stable_rate_pct": 0.0,
            "mean_rise_time_s": float("nan"),
            "std_rise_time_s": float("nan"),
            "mean_stabilize_time_s": float("nan"),
            "std_stabilize_time_s": float("nan"),
            "mean_max_hold_s": 0.0,
            "fall_count": 0,
            "timeout_count": 0,
            "other_fail_count": 0,
            "sitting_pose_verified_pct": 0.0,
        }

    succ = sum(m["environment_success"] for m in metrics)
    stand = sum(m["stand_success"] for m in metrics)
    stable = sum(m["stable_achieved"] for m in metrics)
    t2s = [m["time_to_stand_s"] for m in metrics if not math.isnan(m["time_to_stand_s"]) and m["stand_success"]]
    t2stab = [m["time_to_stabilize_s"] for m in metrics if not math.isnan(m["time_to_stabilize_s"]) and m["stable_achieved"]]
    holds = [m["max_consec_stable_s"] for m in metrics]
    falls = sum(1 for m in metrics if m["termination"] == "fall" or (not m["environment_success"] and m["termination"] != "timeout"))
    timeouts = sum(1 for m in metrics if m["termination"] == "timeout")
    other_fail = total - succ - falls - timeouts
    sitting_v = sum(m["sitting_pose_verified"] for m in metrics)

    return {
        "condition_type": condition_type,
        "condition_name": condition_name,
        "category": category,
        "episodes": total,
        "success_count": succ,
        "success_rate_pct": round(100.0 * succ / max(1, total), 2),
        "stand_rate_pct": round(100.0 * stand / max(1, total), 2),
        "stable_rate_pct": round(100.0 * stable / max(1, total), 2),
        "mean_rise_time_s": round(float(np.mean(t2s)), 3) if t2s else float("nan"),
        "std_rise_time_s": round(float(np.std(t2s)), 3) if t2s else float("nan"),
        "mean_stabilize_time_s": round(float(np.mean(t2stab)), 3) if t2stab else float("nan"),
        "std_stabilize_time_s": round(float(np.std(t2stab)), 3) if t2stab else float("nan"),
        "mean_max_hold_s": round(float(np.mean(holds)), 3) if holds else 0.0,
        "fall_count": falls,
        "timeout_count": timeouts,
        "other_fail_count": max(0, other_fail),
        "sitting_pose_verified_pct": round(100.0 * sitting_v / max(1, total), 2),
    }


def evaluate(model_path: str, vecnorm_path: Optional[str] = None,
             terrain_type: str = "random", curriculum_level: float = 1.0,
             n_episodes: int = 5, seed: int = 0, csv_path: Optional[str] = None,
             render: bool = False, condition_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Evaluate a trained PPO policy across a specified terrain condition."""
    c_name = condition_config["name"] if condition_config else terrain_type.upper()
    c_type = "robustness" if condition_config else "standard"

    print('=' * 75)
    print(f"  POLICY EVALUATION: [{c_type.upper()}] {c_name}")
    print('=' * 75)
    print(f"  Model       : {model_path}")
    print(f"  VecNorm     : {vecnorm_path}")
    if condition_config:
        print(f"  Description : {condition_config.get('description', 'N/A')}")
    else:
        print(f"  Curriculum  : level={curriculum_level:.2f}")
    print(f"  Episodes    : {n_episodes}  Seed: {seed}  Render: {render}")
    print('=' * 75)

    def make_single_env():
        if condition_config:
            return RobustnessA1Env(
                condition_name=condition_config["name"],
                condition_config=condition_config,
                seed=seed,
                curriculum_level=curriculum_level,
                render_mode="human" if render else None,
            )
        return A1SitToStandEnv(
            seed=seed,
            terrain_type=terrain_type,
            curriculum_level=curriculum_level,
            enable_domain_randomization=False,
            enable_inertia_randomization=False,
            render_mode="human" if render else None,
        )

    dummy_env = DummyVecEnv([lambda: Monitor(make_single_env())])
    raw_env = dummy_env.envs[0].env

    try:
        if vecnorm_path and os.path.exists(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, dummy_env)
            vec_env.training = False
            vec_env.norm_reward = False
        else:
            print("Warning: VecNormalize stats not found - obs not normalized.")
            vec_env = VecNormalize(dummy_env, norm_obs=False, norm_reward=False, training=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PPO.load(model_path, env=vec_env, device=device)

        all_metrics = []
        for ep in range(n_episodes):
            ep_seed = seed + ep * 100
            m = evaluate_episode(raw_env, model, vec_env, condition_type=c_type,
                                 condition_name=c_name, episode_num=ep + 1,
                                 seed=ep_seed, render=render)
            all_metrics.append(m)
            s_str = "STAND " if m["stand_success"] else "FAIL  "
            st_str = "STABLE" if m["stable_achieved"] else "      "
            env_str = "ENV_OK" if m["environment_success"] else "      "
            b_str = "BAL   " if m["balance_maintained"] else "      "
            t2s = f'{m["time_to_stand_s"]:.2f}s' if not math.isnan(m["time_to_stand_s"]) else "  N/A"
            t2stab = f'{m["time_to_stabilize_s"]:.2f}s' if not math.isnan(m["time_to_stabilize_s"]) else "  N/A"
            print(f"  Ep {ep+1:3d} | {s_str} {st_str} {env_str} {b_str} | rise={t2s:>6} | stab={t2stab:>6} | "
                  f"hold={m['max_consec_stable_s']:.2f}s | h={m['final_height_m']:.3f}m | tilt={m['final_tilt_deg']:.1f}d | {m['termination']}")
            if render:
                time.sleep(0.5)

        s = compute_condition_summary(all_metrics, c_type, c_name, c_name)
        rise_str = f"{s['mean_rise_time_s']:.2f} +/- {s['std_rise_time_s']:.2f} s" if not math.isnan(s['mean_rise_time_s']) else "N/A"
        stab_str = f"{s['mean_stabilize_time_s']:.2f} +/- {s['std_stabilize_time_s']:.2f} s" if not math.isnan(s['mean_stabilize_time_s']) else "N/A"

        print(f"\n{'-'*75}")
        print(f"  SUMMARY: {c_name} ({s['success_count']}/{n_episodes} Success - {s['success_rate_pct']:.1f}%)")
        print(f"  Rise time        : {rise_str}")
        print(f"  Time to stabilize: {stab_str}")
        print(f"  Mean max hold    : {s['mean_max_hold_s']:.2f} s")
        print(f"  Failures         : {s['fall_count']} falls, {s['timeout_count']} timeouts")
        print(f"{'-'*75}\n")

        if csv_path:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
                writer.writeheader()
                writer.writerows(all_metrics)

        return all_metrics
    finally:
        vec_env.close()


def resolve_path(path_str: Optional[str]) -> Optional[str]:
    """Resolve path relative to project root (_ROOT) if relative, or return None if None."""
    if path_str is None:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = _ROOT / p
    return str(p.resolve())


def run_standard_evaluation(model_path: str, vecnorm_path: str, n_episodes: int = 5,
                            seed: int = 42, render: bool = False,
                            csv_dir: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate Flat -> Slope -> Stairs -> Rough sequentially."""
    sequence = ["flat", "slope", "stairs", "rough"]
    all_episodes: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    print("\n" + "#" * 80)
    print("  STANDARD EVALUATION SUITE: FLAT -> SLOPE -> STAIRS -> ROUGH")
    print(f"  Episodes per terrain: {n_episodes} | Base Seed: {seed} | Render: {render}")
    print("#" * 80 + "\n")

    for t in sequence:
        metrics = evaluate(
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            terrain_type=t,
            curriculum_level=1.0,
            n_episodes=n_episodes,
            seed=seed,
            render=render,
        )
        all_episodes.extend(metrics)
        summaries.append(compute_condition_summary(metrics, "standard", t.upper(), f"Standard_{t.capitalize()}"))

    return all_episodes, summaries


def run_robustness_evaluation(model_path: str, vecnorm_path: str, n_episodes: int = 5,
                              seed: int = 42, render: bool = False,
                              csv_dir: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate explicit held-out conditions without retraining."""
    all_episodes: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    print("\n" + "#" * 80)
    print("  HELD-OUT ROBUSTNESS EVALUATION SUITE (OUT-OF-DISTRIBUTION TESTING)")
    print(f"  Episodes per test: {n_episodes} | Base Seed: {seed} | Render: {render}")
    print("#" * 80 + "\n")

    for cond in ROBUSTNESS_CONDITIONS:
        try:
            metrics = evaluate(
                model_path=model_path,
                vecnorm_path=vecnorm_path,
                terrain_type=cond["terrain_type"],
                n_episodes=n_episodes,
                seed=seed,
                render=render,
                condition_config=cond,
            )
            all_episodes.extend(metrics)
            summaries.append(compute_condition_summary(metrics, "robustness", cond["name"], cond["category"]))
        except Exception as e:
            print(f"[REPORT: UNSUPPORTED CONDITION] {cond['name']} encountered error: {e}")
            summaries.append(compute_condition_summary([], "robustness", cond["name"], cond["category"]))

    return all_episodes, summaries


def print_formatted_summary_table(summaries: List[Dict[str, Any]], title: str):
    """Print clean summary table to standard output."""
    print("\n" + "=" * 115)
    print(f"  {title}")
    print("=" * 115)
    hdr = (f"{'Condition':<24} | {'Category':<22} | {'Episodes':<8} | {'Success':<8} | "
           f"{'EnvSucc %':<9} | {'Rise Time (Mean +/- Std)':<24} | {'Stabilize Time (M +/- S)':<24} | {'Hold':<7} | {'Falls':<5} | {'Timeouts':<8}")
    print(hdr)
    print("-" * 115)
    for s in summaries:
        succ_str = f"{s['success_count']}/{s['episodes']}"
        rise_str = f"{s['mean_rise_time_s']:.2f} +/- {s['std_rise_time_s']:.2f} s" if not math.isnan(s['mean_rise_time_s']) else "N/A"
        stab_str = f"{s['mean_stabilize_time_s']:.2f} +/- {s['std_stabilize_time_s']:.2f} s" if not math.isnan(s['mean_stabilize_time_s']) else "N/A"
        print(f"{s['condition_name']:<24} | {s['category']:<22} | {s['episodes']:<8} | {succ_str:<8} | "
              f"{s['success_rate_pct']:>8.1f}% | {rise_str:>24} | {stab_str:>24} | {s['mean_max_hold_s']:>5.2f}s | {s['fall_count']:<5} | {s['timeout_count']:<8}")
    print("=" * 115 + "\n")


def save_evaluation_csvs(episodes: List[Dict[str, Any]], summaries: List[Dict[str, Any]], csv_dir: Path):
    """Save episode-level CSV and aggregate summary CSV."""
    csv_dir.mkdir(parents=True, exist_ok=True)

    ep_path = csv_dir / "evaluation_episodes.csv"
    if episodes:
        with open(ep_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
            writer.writeheader()
            writer.writerows(episodes)
        print(f"[CSV EXPORT] Episode-level results saved: {ep_path}")

    sum_path = csv_dir / "evaluation_summary.csv"
    if summaries:
        with open(sum_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"[CSV EXPORT] Aggregate summary saved    : {sum_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Unitree A1 Sit-to-Stand Policy")
    parser.add_argument("--model", type=str, default=None,
        help=f"Path to trained PPO model zip (default: {_DEFAULT_MODEL_PATH.relative_to(_ROOT)})")
    parser.add_argument("--vecnorm", type=str, default=None,
        help=f"Path to matching VecNormalize pkl (default: {_DEFAULT_VECNORM_PATH.relative_to(_ROOT)})")
    parser.add_argument("--terrain", type=str, default="all",
        choices=["all", "flat", "slope", "stairs", "rough", "random"],
        help="Terrain type to evaluate (default: all -> Flat, Slope, Stairs, Rough sequentially)")
    parser.add_argument("--robustness", action="store_true",
        help="Run explicit held-out robustness evaluation suite")
    parser.add_argument("--all-eval", action="store_true",
        help="Run both Standard Evaluation suite and Held-out Robustness suite")
    parser.add_argument("--curriculum", type=float, default=1.0,
        help="Curriculum difficulty level (default: 1.0)")
    parser.add_argument("--episodes", type=int, default=5,
        help="Number of episodes to evaluate per condition (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
        help="RNG seed for evaluation (default: 42)")
    parser.add_argument("--render", action="store_true",
        help="Launch live MuJoCo passive viewer at real-time 50 FPS")
    parser.add_argument("--csv", type=str, default=None,
        help="Path to output directory for CSV exports (default: output/eval_results/)")
    args = parser.parse_args()

    model_resolved = resolve_path(args.model) if args.model else str(_DEFAULT_MODEL_PATH)
    vecnorm_resolved = resolve_path(args.vecnorm) if args.vecnorm else str(_DEFAULT_VECNORM_PATH)

    if not os.path.exists(model_resolved):
        print(f"\n[ERROR] Model file not found: {model_resolved}\n")
        sys.exit(1)

    if not os.path.exists(vecnorm_resolved):
        alt_vecnorm = str(Path(vecnorm_resolved).parent / "best_model.pkl")
        if os.path.exists(alt_vecnorm):
            vecnorm_resolved = alt_vecnorm
        else:
            print(f"\n[ERROR] VecNormalize statistics file not found: {vecnorm_resolved}\n")
            sys.exit(1)

    csv_dir = Path(args.csv) if args.csv else _CSV_DIR
    all_episodes: List[Dict[str, Any]] = []
    all_summaries: List[Dict[str, Any]] = []

    run_std = (args.terrain == "all" or args.all_eval or not args.robustness)
    run_rob = (args.robustness or args.all_eval)

    if args.terrain != "all" and not run_rob:
        metrics = evaluate(
            model_path=model_resolved,
            vecnorm_path=vecnorm_resolved,
            terrain_type=args.terrain,
            curriculum_level=args.curriculum,
            n_episodes=args.episodes,
            seed=args.seed,
            csv_path=str(csv_dir / f"eval_{args.terrain}.csv"),
            render=args.render,
        )
        sys.exit(0)

    if run_std:
        std_episodes, std_summaries = run_standard_evaluation(
            model_path=model_resolved,
            vecnorm_path=vecnorm_resolved,
            n_episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            csv_dir=str(csv_dir),
        )
        all_episodes.extend(std_episodes)
        all_summaries.extend(std_summaries)
        print_formatted_summary_table(std_summaries, "STANDARD EVALUATION SUMMARY (Flat -> Slope -> Stairs -> Rough)")

    if run_rob:
        rob_episodes, rob_summaries = run_robustness_evaluation(
            model_path=model_resolved,
            vecnorm_path=vecnorm_resolved,
            n_episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            csv_dir=str(csv_dir),
        )
        all_episodes.extend(rob_episodes)
        all_summaries.extend(rob_summaries)
        print_formatted_summary_table(rob_summaries, "HELD-OUT ROBUSTNESS EVALUATION SUMMARY (Out-of-Distribution)")

    save_evaluation_csvs(all_episodes, all_summaries, csv_dir)
