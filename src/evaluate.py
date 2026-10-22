from __future__ import annotations
import time
import sys, csv, math, os, argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment import A1SitToStandEnv, _tilt_angle
from terrain import TerrainGenerator

STAND_HEIGHT_MIN   = 0.21   # Must be strictly above the 0.20m calf length
STAND_TILT_MAX     = 25.0   # Maximum tilt relative to terrain normal
STABLE_VEL_LIN     = 0.25   # Max linear velocity for stable stand [m/s]
STABLE_VEL_ANG     = 0.60   # Max angular velocity for stable stand [rad/s]
STABLE_HEIGHT_VAR  = 0.025  # Max height standard deviation in stability window [m]
STABILITY_WINDOW   = 1.0    # Required stable standing duration [s]
STAND_SUCCESS_MIN_TIME = 2.0  # Task specifies a 2–5 second rise
STAND_SUCCESS_TIME = 5.0       # Required maximum time to stand up [s]

_ROOT       = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _ROOT / "output"
_MODEL_DIR  = _OUTPUT_DIR / "models"
_STATS_DIR  = _OUTPUT_DIR / "vecnorm_stats"
_CSV_DIR    = _OUTPUT_DIR / "eval_results"


def evaluate_episode(raw_env: A1SitToStandEnv, model: PPO, vec_env: VecNormalize,
                     render: bool = False) -> Dict[str, Any]:
    """Run one deterministic episode and return metrics dict."""
    obs  = vec_env.reset()
    done = False
    step = 0
    ctrl_dt = raw_env.control_dt
    stable_window_steps = max(1, int(STABILITY_WINDOW / ctrl_dt))

    if render:
        raw_env.render()
        time.sleep(0.5)

    heights, tilts_deg, lin_vels, ang_vels = [], [], [], []
    first_stand_step   = None
    consecutive_stand  = 0
    max_consec_stand   = 0
    stable_achieved    = False
    termination_reason = "unknown"

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated_arr, info_list = vec_env.step(action)
        info = info_list[0]
        step += 1

        if render:
            time.sleep(ctrl_dt)

        height      = float(info.get("h_rel", 0.0))
        tilt_deg    = float(info.get("tilt_deg", 0.0))
        is_standing = bool(info.get("is_standing", False))

        v = np.array(raw_env.data.qvel[0:3])
        w = np.array(raw_env.data.qvel[3:6])
        lin_vel = float(np.linalg.norm(v))
        ang_vel = float(np.linalg.norm(w))

        heights.append(height)
        tilts_deg.append(tilt_deg)
        lin_vels.append(lin_vel)
        ang_vels.append(ang_vel)

        if is_standing and first_stand_step is None:
            first_stand_step = step
        if is_standing:
            consecutive_stand += 1
        else:
            max_consec_stand  = max(max_consec_stand, consecutive_stand)
            consecutive_stand = 0
        if consecutive_stand >= stable_window_steps and not stable_achieved:
            stable_achieved = True

        done_flag = bool(terminated_arr[0])
        truncated = bool(info.get("truncated", False) or info.get("TimeLimit.truncated", False))
        terminated = bool(info.get("terminated", False)) or (done_flag and not truncated)
        done = done_flag or terminated or truncated

        if terminated:
            termination_reason = info.get("termination_reason", "fall" if not info.get("success", False) else "success")
        elif truncated:
            termination_reason = "timeout"

    max_consec_stand = max(max_consec_stand, consecutive_stand)
    heights   = np.array(heights)
    tilts_deg = np.array(tilts_deg)
    lin_vels  = np.array(lin_vels)
    ang_vels  = np.array(ang_vels)
    total_time = step * ctrl_dt

    t2s = (first_stand_step * ctrl_dt) if first_stand_step else None
    stand_success = (t2s is not None and STAND_SUCCESS_MIN_TIME <= t2s <= STAND_SUCCESS_TIME)

    window = max(1, stable_window_steps)
    fh = heights[-window:];   ft = tilts_deg[-window:]
    fv = lin_vels[-window:];  fw = ang_vels[-window:]

    balance_maintained = (
        stand_success and stable_achieved
        and float(np.mean(fh)) >= STAND_HEIGHT_MIN
        and float(np.mean(ft)) < STAND_TILT_MAX
        and float(np.mean(fv)) < STABLE_VEL_LIN
        and float(np.mean(fw)) < STABLE_VEL_ANG
        and float(np.std(fh))  < STABLE_HEIGHT_VAR
    )
    return {
        "terrain_type":        info.get("terrain_type", raw_env.terrain.current_type),
        "stand_success":       int(stand_success),
        "time_to_stand_s":     t2s if t2s is not None else float("nan"),
        "stable_achieved":     int(stable_achieved),
        "balance_maintained":  int(balance_maintained),
        "final_height_m":      float(np.mean(fh)),
        "final_tilt_deg":      float(np.mean(ft)),
        "final_ang_vel_rads":  float(np.mean(fw)),
        "height_var_m":        float(np.std(fh)),
        "total_steps":         step,
        "total_time_s":        total_time,
        "termination":         termination_reason,
        "max_consec_stand_s":  max_consec_stand * ctrl_dt,
    }


def evaluate(model_path: str, vecnorm_path: Optional[str] = None,
             terrain_type: str = "random", curriculum_level: float = 1.0,
             n_episodes: int = 50, seed: int = 0, csv_path: Optional[str] = None,
             render: bool = False):
    """Load and evaluate a trained PPO model across specified terrain settings."""
    print('=' * 68)
    print(f'  A1 Sit-to-Stand Policy Evaluation | Terrain: {terrain_type.upper()}')
    print('=' * 68)
    print(f"  Model       : {model_path}")
    print(f"  VecNorm     : {vecnorm_path}")
    print(f"  Curriculum  : level={curriculum_level:.2f}")
    print(f"  Episodes    : {n_episodes}  Seed: {seed}  Render: {render}")
    print('=' * 68)

    # Build unwrapped and monitored env
    def make_single_env():
        return A1SitToStandEnv(
            seed=seed,
            terrain_type=terrain_type,
            curriculum_level=curriculum_level,
            enable_domain_randomization=False,
            enable_inertia_randomization=False,
            render_mode="human" if render else None
        )

    dummy_env = DummyVecEnv([lambda: Monitor(make_single_env())])
    raw_env = dummy_env.envs[0].env  # unwrapped underlying env

    if vecnorm_path and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, dummy_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print(f"VecNormalize loaded: {vecnorm_path}")
    else:
        print('Warning: VecNormalize stats not found - obs not normalized.')
        vec_env = VecNormalize(
            dummy_env, norm_obs=False, norm_reward=False, training=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = PPO.load(model_path, env=vec_env, device=device)
    print(f"Model loaded (device={device})\n")

    all_metrics = []
    for ep in range(n_episodes):
        m = evaluate_episode(raw_env, model, vec_env, render=render)
        all_metrics.append(m)
        s_str = "STAND " if m["stand_success"]     else "FAIL  "
        b_str = "STABLE" if m["balance_maintained"] else "      "
        t2s   = f'{m["time_to_stand_s"]:.2f}s' if not math.isnan(m["time_to_stand_s"]) else "  N/A"
        print(f"  Ep {ep+1:3d} [{m['terrain_type']:<6}] | {s_str} {b_str} | t2stand={t2s} | "
              f"h={m['final_height_m']:.3f}m | tilt={m['final_tilt_deg']:.1f}d | {m['termination']}")
        if render:
            time.sleep(0.5)

    successes    = [m["stand_success"]      for m in all_metrics]
    t2s_list     = [m["time_to_stand_s"]    for m in all_metrics if not math.isnan(m["time_to_stand_s"])]
    stable_list  = [m["stable_achieved"]    for m in all_metrics]
    balance_list = [m["balance_maintained"] for m in all_metrics]
    heights_fin  = [m["final_height_m"]     for m in all_metrics]
    tilts_fin    = [m["final_tilt_deg"]     for m in all_metrics]
    avels_fin    = [m["final_ang_vel_rads"] for m in all_metrics]
    term_counts  = {}
    for m in all_metrics:
        t = m["termination"]; term_counts[t] = term_counts.get(t, 0) + 1

    def pct(lst): return 100.0 * sum(lst) / n_episodes

    print(f"\n{'='*68}")
    print(f"  EVALUATION SUMMARY: {terrain_type.upper()}")
    print(f"{'='*68}")
    print(f"  Episodes                        : {n_episodes}")
    print(f"  Sit-to-stand success rate       : {pct(successes):.1f}%")
    if t2s_list:
        print(f"  Mean time to stand              : {np.mean(t2s_list):.2f}s")
        print(f"  Median time to stand            : {np.median(t2s_list):.2f}s")
    else:
        print("  Time to stand                   : N/A (no successful stands)")
    print(f"  Stability window achieved       : {pct(stable_list):.1f}%")
    print(f"  Balance maintained (all crit.)  : {pct(balance_list):.1f}%")
    print(f"  Mean final base height          : {np.mean(heights_fin):.3f}m")
    print(f"  Mean final torso tilt           : {np.mean(tilts_fin):.2f}deg")
    print(f"  Mean final ang velocity         : {np.mean(avels_fin):.3f}rad/s")
    print(f"  Termination reasons             : {term_counts}")
    print(f"{'='*68}")

    _CSV_DIR.mkdir(parents=True, exist_ok=True)
    if csv_path is None:
        csv_path = str(_CSV_DIR / f"eval_{terrain_type}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"CSV saved: {csv_path}\n")

    vec_env.close()
    return all_metrics


def run_full_benchmark(model_path: str, vecnorm_path: Optional[str] = None,
                       episodes_per_terrain: int = 25):
    """Run full benchmark across Flat, Slope, Stairs, Rough, and Unseen Challenge terrains."""
    terrains = ["flat", "slope", "stairs", "rough"]
    summary = {}

    print("\n" + "#"*70)
    print("  STARTING COMPREHENSIVE MULTI-TERRAIN BENCHMARK")
    print("#"*70)

    benchmark_cases = [(t, 1.0, f"benchmark_{t}.csv") for t in terrains]
    benchmark_cases += [(t, 1.2, f"benchmark_ood_{t}.csv") for t in ["slope", "stairs", "rough"]]
    for t, level, csv_name in benchmark_cases:
        csv_p = str(_CSV_DIR / csv_name)
        metrics = evaluate(model_path, vecnorm_path, terrain_type=t,
                           curriculum_level=level, n_episodes=episodes_per_terrain,
                           seed=100, csv_path=csv_p)
        succ = 100.0 * sum(m["stand_success"] for m in metrics) / len(metrics)
        bal  = 100.0 * sum(m["balance_maintained"] for m in metrics) / len(metrics)
        t2s  = [m["time_to_stand_s"] for m in metrics if not math.isnan(m["time_to_stand_s"])]
        mean_t = np.mean(t2s) if t2s else float("nan")
        mean_h = np.mean([m["final_height_m"] for m in metrics])
        mean_tilt = np.mean([m["final_tilt_deg"] for m in metrics])
        summary[f"{t}_L{level:.1f}"] = {
            "success_rate": succ,
            "balance_rate": bal,
            "mean_time_s": mean_t,
            "mean_height_m": mean_h,
            "mean_tilt_deg": mean_tilt
        }

    # Consolidated summary table
    print("\n" + "="*75)
    print("  CONSOLIDATED MULTI-TERRAIN BENCHMARK RESULTS")
    print("="*75)
    print(f"{'Case':<16} | {'Success Rate':<14} | {'Balance Rate':<14} | {'Time to Stand':<14} | {'Base Height':<12}")
    print("-"*75)
    for t, s in summary.items():
        t_str = f"{s['mean_time_s']:.2f}s" if not math.isnan(s['mean_time_s']) else "N/A"
        print(f"{t:<16} | {s['success_rate']:>12.1f}% | {s['balance_rate']:>12.1f}% | {t_str:>14} | {s['mean_height_m']:>10.3f}m")
    print("="*75 + "\n")

    summary_csv = _CSV_DIR / "benchmark_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Case", "SuccessRatePct", "BalanceRatePct", "MeanTimeToStandS", "MeanBaseHeightM", "MeanTiltDeg"])
        for t, s in summary.items():
            writer.writerow([t, s["success_rate"], s["balance_rate"], s["mean_time_s"], s["mean_height_m"], s["mean_tilt_deg"]])
    print(f"Summary CSV saved to: {summary_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multi-Terrain A1 Policy")
    parser.add_argument("--model", type=str,
        default=str(_MODEL_DIR / "ppo_a1_sit_to_stand" / "best_model"))
    parser.add_argument("--vecnorm", type=str,
        default=str(_STATS_DIR / "ppo_a1_sit_to_stand_vecnorm_eval.pkl"))
    parser.add_argument("--terrain", type=str, default="flat",
        choices=["benchmark", "flat", "slope", "stairs", "rough", "random"],
        help="Terrain type to evaluate (default: flat, or benchmark for headless all-terrain test)")
    parser.add_argument("--curriculum", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=5,
        help="Number of episodes to evaluate (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true", help="Launch live MuJoCo passive viewer at real-time 50 FPS")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if args.terrain == "benchmark" and not args.render:
        run_full_benchmark(args.model, args.vecnorm, episodes_per_terrain=args.episodes)
    else:
        evaluate(args.model, args.vecnorm, terrain_type=args.terrain,
                 curriculum_level=args.curriculum, n_episodes=args.episodes,
                 seed=args.seed, csv_path=args.csv, render=args.render)
