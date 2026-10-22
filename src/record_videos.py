from __future__ import annotations
import os, sys, math, time, argparse
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import cv2
import imageio
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment import A1SitToStandEnv
from terrain import TerrainGenerator

_ROOT       = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _ROOT / "output"
_MODEL_DIR  = _OUTPUT_DIR / "models"
_STATS_DIR  = _OUTPUT_DIR / "vecnorm_stats"
_VIDEO_DIR  = _OUTPUT_DIR / "videos"


def draw_hud(frame: np.ndarray, time_s: float, height: float,
             tilt_deg: float, state_str: str, terrain_name: str) -> np.ndarray:
    """Draw telemetry overlay on RGB frame."""
    img = frame.copy()
    h, w, _ = img.shape

    # Semi-transparent top bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (18, 24, 32), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # Color coded state badge
    if state_str == "STANDING STABLE":
        badge_color = (40, 200, 80)
    elif state_str == "RISING":
        badge_color = (30, 180, 240)
    else:
        badge_color = (200, 120, 40)

    # Text lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    line1 = f"Terrain: {terrain_name.upper()} | t={time_s:4.2f}s | h={height:.3f}m | tilt={tilt_deg:4.1f} deg"
    cv2.putText(img, line1, (14, 24), font, 0.52, (240, 240, 240), 1, cv2.LINE_AA)

    cv2.putText(img, f"Status: {state_str}", (14, 44), font, 0.55, badge_color, 2, cv2.LINE_AA)
    return img


def record_episode(model: PPO, vec_env: VecNormalize, raw_env: A1SitToStandEnv,
                   terrain_name: str, out_prefix: str, max_steps: int = 250):
    """Run one episode, render frames, and save MP4 & GIF."""
    _VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    obs = vec_env.reset()
    frames = []

    ctrl_dt = raw_env.control_dt
    consecutive_stand = 0
    stable_thresh_steps = int(1.0 / ctrl_dt)

    for step in range(max_steps):
        # Render offscreen frame
        rgb = raw_env.render()
        if rgb is not None:
            # Determine state
            pos = raw_env._get_base_pos()
            z_floor = raw_env.terrain.get_height(pos[0], pos[1])
            info_h = float(pos[2] - z_floor)
            info_normal = raw_env.terrain.get_normal(pos[0], pos[1])
            z_axis = raw_env.data.xmat[raw_env._trunk_id].reshape(3, 3)[:, 2]
            cos_tilt = float(np.clip(np.dot(z_axis, info_normal), -1.0, 1.0))
            tilt_d = float(np.degrees(np.arccos(cos_tilt)))

            is_standing = (info_h >= 0.20 and tilt_d <= 25.0)
            if is_standing:
                consecutive_stand += 1
            else:
                consecutive_stand = 0

            if consecutive_stand >= stable_thresh_steps:
                status = "STANDING STABLE"
            elif is_standing:
                status = "RISING"
            elif step < 25:
                status = "SITTING (INIT)"
            else:
                status = "BALANCING"

            t_sec = step * ctrl_dt
            hud_frame = draw_hud(rgb, t_sec, info_h, tilt_d, status, terrain_name)
            frames.append(hud_frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated_arr, info_list = vec_env.step(action)
        if terminated_arr[0]:
            break

    if not frames:
        print(f"Warning: No frames captured for {terrain_name}")
        return

    mp4_path = _VIDEO_DIR / f"{out_prefix}.mp4"
    gif_path = _VIDEO_DIR / f"{out_prefix}.gif"

    # 1. Write MP4 (50 FPS)
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(mp4_path), fourcc, 50.0, (w, h))
    for f in frames:
        # cv2 uses BGR
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"  Saved MP4: {mp4_path} ({len(frames)} frames)")

    # 2. Write GIF (subsampled every 2nd frame = 25 FPS)
    gif_frames = frames[::2]
    imageio.mimsave(str(gif_path), gif_frames, fps=25, loop=0)
    print(f"  Saved GIF: {gif_path} ({len(gif_frames)} frames)")


def record_all_terrains(model_path: str, vecnorm_path: Optional[str] = None):
    """Record video across standard terrains and out-of-distribution challenges."""
    print("=" * 68)
    print("  RECORDING MULTI-TERRAIN DEMONSTRATIONS (MP4 & GIF)")
    print("=" * 68)
    print(f"  Model  : {model_path}")
    print(f"  VecNorm: {vecnorm_path}")
    print(f"  Output : {_VIDEO_DIR}")
    print("=" * 68)

    scenarios = [
        {"name": "Flat Ground",       "type": "flat",   "curriculum": 1.0, "file": "stand_flat_ground"},
        {"name": "Slope (up to 10 deg)", "type": "slope",  "curriculum": 0.56, "file": "stand_slope_10deg"},
        {"name": "OOD slope (up to 18 deg)", "type": "slope", "curriculum": 1.2, "file": "stand_slope_18deg_unseen"},
        {"name": "Stairs (up to 4 cm)", "type": "stairs", "curriculum": 0.30, "file": "stand_stairs_4cm"},
        {"name": "OOD stairs (up to 7 cm)", "type": "stairs", "curriculum": 1.2, "file": "stand_stairs_7cm_unseen"},
        {"name": "Rocky / Rough (5 cm scale)", "type": "rough", "curriculum": 1.0, "file": "stand_rough_rocky"},
    ]

    for sc in scenarios:
        print(f"\nRecording scenario: {sc['name']}...")
        def make_env():
            return A1SitToStandEnv(
                render_mode="rgb_array",
                seed=42,
                terrain_type=sc["type"],
                curriculum_level=sc["curriculum"],
                enable_domain_randomization=True,
                enable_inertia_randomization=True
            )

        dummy_env = DummyVecEnv([lambda: Monitor(make_env())])
        raw_env = dummy_env.envs[0].env

        if vecnorm_path and os.path.exists(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, dummy_env)
            vec_env.training = False
            vec_env.norm_reward = False
        else:
            vec_env = VecNormalize(dummy_env, norm_obs=False, norm_reward=False, training=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PPO.load(model_path, env=vec_env, device=device)

        record_episode(model, vec_env, raw_env, sc["name"], sc["file"], max_steps=250)
        vec_env.close()

    print("\nAll videos successfully recorded!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record A1 Demonstration Videos")
    parser.add_argument("--model", type=str,
        default=str(_MODEL_DIR / "ppo_a1_sit_to_stand" / "best_model"))
    parser.add_argument("--vecnorm", type=str,
        default=str(_STATS_DIR / "ppo_a1_sit_to_stand_vecnorm_eval.pkl"))
    args = parser.parse_args()

    record_all_terrains(args.model, args.vecnorm)
