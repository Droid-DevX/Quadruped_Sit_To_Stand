from __future__ import annotations
import sys, time, argparse, math
from pathlib import Path
import numpy as np

# Ensure src is in import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass

from environment import A1SitToStandEnv


def print_contact_diagnostics(env: A1SitToStandEnv):
    model = env.model
    data  = env.data
    ncon  = data.ncon

    print("\n" + "=" * 60)
    print("  INITIAL POSE & COLLISION DIAGNOSTICS")
    print("=" * 60)
    pos = env._get_base_pos()
    z_floor = env.terrain.get_height(pos[0], pos[1])
    h_rel = pos[2] - z_floor
    normal = env.terrain.get_normal(pos[0], pos[1])

    print(f"  Terrain Type       : {env.terrain.current_type.upper()}")
    print(f"  Base Position (XYZ): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
    print(f"  Terrain Elevation  : {z_floor:.4f} m")
    print(f"  Base Height (h_rel): {h_rel:.4f} m (Crouch target: {env.crouch_height_init:.3f} m)")
    print(f"  Terrain Normal     : [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")

    q = env._get_joint_pos()
    print(f"  Joint Positions [rad]:")
    leg_names = ["FR", "FL", "RR", "RL"]
    for i, leg in enumerate(leg_names):
        hip, thigh, calf = q[i*3], q[i*3+1], q[i*3+2]
        print(f"    {leg}: hip={hip:+.3f}, thigh={thigh:+.3f}, calf={calf:+.3f}")

    print(f"\n  Active Contacts ({ncon} total):")
    min_dist = float("inf")
    feet_contact = []
    torso_contact_count = 0
    for i in range(ncon):
        con = data.contact[i]
        g1_id, g2_id = int(con.geom1), int(con.geom2)
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom_{con.geom1}"
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom_{con.geom2}"
        dist  = con.dist
        if g1_id in env._terrain_geom_ids or g2_id in env._terrain_geom_ids:
            min_dist = min(min_dist, dist)
            if g1_id in env._feet_geom_ids or g2_id in env._feet_geom_ids:
                feet_contact.append((geom1, geom2, dist))
            if g1_id == env._trunk_contact_geom_id or g2_id == env._trunk_contact_geom_id:
                torso_contact_count += 1
        if i < 8:
            print(f"    [{i+1:2d}] {geom1:<18} <-> {geom2:<18} | dist = {dist*1000:+.2f} mm")

    if ncon > 8:
        print(f"    ... and {ncon - 8} more contacts")

    if min_dist == float("inf"):
        print("\n  WARNING: MuJoCo reports no active contacts.")
    elif min_dist < -0.002:
        print(f"\n  WARNING: Significant penetration detected: {abs(min_dist)*1000:.2f} mm")
    else:
        print(f"\n  Minimum contact distance: {min_dist*1000:+.2f} mm")

    print(f"  Torso/terrain contacts : {torso_contact_count}")
    print(f"  Foot ground contacts   : {len(feet_contact)} contact pairs")
    print(f"  Torso touch requirement: {'PASS' if torso_contact_count else 'FAIL'}")
    print("=" * 60 + "\n")


def get_runtime_contact_diagnostics(env: A1SitToStandEnv):
    model, data = env.model, env.data
    terrain_ids = env._terrain_geom_ids
    trunk_id = int(getattr(env, "_trunk_contact_geom_id", -1))
    feet_ids = getattr(env, "_feet_geom_ids", set())
    min_dist = float("inf")
    terrain_pairs = []
    foot_pairs = []
    trunk_pairs = []

    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 not in terrain_ids and g2 not in terrain_ids:
            continue
        terrain_id = g1 if g1 in terrain_ids else g2
        robot_geom_id = g2 if g1 in terrain_ids else g1
        robot_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom_id) or f"geom_{robot_geom_id}"
        terrain_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, terrain_id) or "terrain"
        dist = float(c.dist)
        min_dist = min(min_dist, dist)
        pair = (robot_name, terrain_name, dist)
        terrain_pairs.append(pair)
        if robot_geom_id in feet_ids:
            foot_pairs.append(pair)
        if robot_geom_id == trunk_id:
            trunk_pairs.append(pair)

    pos = env._get_base_pos()
    z_sample = float(env.terrain.get_height(float(pos[0]), float(pos[1])))
    h_rel = float(pos[2] - z_sample)
    return {
        "h_rel": h_rel,
        "base_z": float(pos[2]),
        "terrain_z_sample": z_sample,
        "n_terrain_contacts": len(terrain_pairs),
        "n_foot_contacts": len(foot_pairs),
        "n_trunk_contacts": len(trunk_pairs),
        "min_terrain_dist": min_dist,
        "terrain_pairs": terrain_pairs,
        "foot_pairs": foot_pairs,
        "trunk_pairs": trunk_pairs,
    }


def print_runtime_contact_diagnostics(step: int, diag: dict):
    min_dist = diag["min_terrain_dist"]
    min_text = "none" if not np.isfinite(min_dist) else f"{min_dist * 1000:+.2f}mm"
    print(
        f"  Step {step:4d} | h_rel={diag['h_rel']:.4f}m "
        f"base_z={diag['base_z']:.4f}m terrain_z={diag['terrain_z_sample']:.4f}m "
        f"contacts terrain/feet/trunk={diag['n_terrain_contacts']}/"
        f"{diag['n_foot_contacts']}/{diag['n_trunk_contacts']} "
        f"min_dist={min_text}"
    )
    pairs = diag["terrain_pairs"]
    if pairs:
        details = ", ".join(f"{name}:{dist*1000:+.2f}mm" for name, _, dist in pairs[:8])
        print(f"           terrain pairs: {details}")
    else:
        print("           terrain pairs: NONE (robot has no active terrain contacts)")

def main():
    parser = argparse.ArgumentParser(description="Inspect A1 Initial Reset Pose & Test Sit-to-Stand Actions")
    parser.add_argument("--terrain", type=str, default="flat",
                        choices=["flat", "slope", "stairs", "rough", "all"],
                        help="Terrain type to inspect, or all four terrain generators")
    parser.add_argument("--mode", type=str, default="random",
                        choices=["random", "stand", "hold-sit", "zero"],
                        help="Action mode: 'random' (random actions from sit), 'stand' (target standing pose), 'hold-sit' (hold sit pose), 'zero' (zero action)")
    parser.add_argument("--render", action="store_true",
                        help="Opt in to opening the passive MuJoCo viewer; without this, only print diagnostics")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch full interactive MuJoCo GUI viewer (pan/zoom/contact view); implies rendering")
    parser.add_argument("--step", action="store_true",
                        help="Alias for legacy step mode")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reset")
    parser.add_argument("--check-only", action="store_true",
                        help="Headless reset validation; do not open a viewer")
    args = parser.parse_args()

    if args.check_only or args.terrain == "all":
        terrains = ["flat", "slope", "stairs", "rough"] if args.terrain == "all" else [args.terrain]
        failed = []
        for terrain_name in terrains:
            print(f"\nChecking {terrain_name.upper()} resets ({args.episodes} episodes)...")
            check_env = A1SitToStandEnv(terrain_type=terrain_name, curriculum_level=1.0,
                enable_domain_randomization=False, render_mode=None, seed=args.seed)
            try:
                for episode in range(args.episodes):
                    _, info = check_env.reset(seed=args.seed + episode)
                    pos = check_env._get_base_pos()
                    spawn_ok = bool(info.get("spawn_valid", False))
                    min_dist = float(info.get("min_terrain_contact_distance", float("nan")))
                    pen_mm = float(info.get("terrain_penetration_mm", 0.0))
                    finite_obs = bool(np.all(np.isfinite(check_env._get_obs())))
                    yaw_deg = math.degrees(info.get('initial_yaw_rad', 0.0))
                    
                    # Verify physical contact: must have finite contact distance and negligible penetration (<0.1mm)
                    valid = spawn_ok and (pen_mm <= 0.10) and finite_obs
                    print(f"  ep={episode+1:02d} valid={valid} "
                          f"min_dist={min_dist*1000:+.2f}mm pen={pen_mm:.2f}mm "
                          f"feet_con={info.get('n_foot_contacts', 0)} "
                          f"h_rel={info.get('h_rel', float('nan')):.3f}m "
                          f"yaw={yaw_deg:+.1f}deg "
                          f"obs_finite={finite_obs}")
                    if not valid:
                        failed.append(f"{terrain_name} episode {episode+1}")
            finally:
                check_env.close()
        print("\nRESET CHECK:", "PASS" if not failed else "FAIL: " + ", ".join(failed))
        if args.check_only or args.terrain == "all":
            return 0 if not failed else 2

    mode = args.mode
    if args.step and mode == "random":
        mode = "hold-sit"

    if not args.render and not args.interactive:
        print(f"Initializing A1 environment on '{args.terrain}' terrain (headless diagnostics, seed={args.seed})...")
        env = A1SitToStandEnv(
            terrain_type=args.terrain,
            curriculum_level=1.0,
            enable_domain_randomization=False,
            render_mode=None,
            seed=args.seed,
        )
        try:
            _, info = env.reset(seed=args.seed)
            print_contact_diagnostics(env)
            print(f"  Torso-ground contact at reset: {info.get('torso_ground_contact', False)}")
            print(f"  Initial position: {np.round(info.get('initial_base_position', []), 3)}")
            env.print_diagnostics()
            print("Viewer not opened. Add --render to visualize the initial pose.")
        finally:
            env.close()
        return 0

    print(f"Initializing A1 environment on '{args.terrain}' terrain (seed={args.seed}, mode={mode})...")

    render_mode = "human" if args.render and not args.interactive else None
    env = A1SitToStandEnv(
        terrain_type=args.terrain,
        curriculum_level=1.0,
        enable_domain_randomization=False,
        render_mode=render_mode,
        seed=args.seed
    )

    try:
        obs, info = env.reset(seed=args.seed)
        print_contact_diagnostics(env)
        print(f"  Torso-ground contact at reset: {info.get('torso_ground_contact', False)}")
        print(f"  Initial position: {np.round(info.get('initial_base_position', []), 3)}")
        print(f"  Initial orientation perturbation: roll={info.get('initial_roll_perturb_deg', 0.0):+.2f} deg, "
              f"pitch={info.get('initial_pitch_perturb_deg', 0.0):+.2f} deg, "
              f"yaw={math.degrees(info.get('initial_yaw_rad', 0.0)):+.1f} deg")
        env.print_diagnostics()

        if args.interactive:
            print("Launching full interactive MuJoCo GUI Viewer...")
            print("  Controls:")
            print("    - Left Click + Drag : Rotate camera")
            print("    - Right Click + Drag: Translate camera")
            print("    - Scroll Wheel      : Zoom in/out")
            print("    - Spacebar          : Pause / Resume simulation")
            print("    - Backspace         : Reset simulation")
            print("    - Tab key           : Toggle collision contact & geom visualization")
            print("Close viewer window to exit.\n")
            mujoco.viewer.launch(env.model, env.data)
        else:
            print(f"Launching passive viewer window (Mode: {mode.upper()})...")
            env.render()

            if env._viewer is None:
                print("Note: Passive viewer could not be initialized (headless or missing display).")
                print("Falling back to full interactive viewer...")
                mujoco.viewer.launch(env.model, env.data)
                return

            ctrl_dt = env.control_dt
            crouch_q = env._crouch_q.copy()
            print("\nHOLD-SIT target joint angles [rad]:")
            for i, leg in enumerate(["FR", "FL", "RR", "RL"]):
                hip, thigh, calf = crouch_q[i*3:i*3+3]
                print(
                    f"  {leg}: hip={hip:+.3f}, "
                    f"thigh={thigh:+.3f}, calf={calf:+.3f}"
                )

            for ep in range(args.episodes):
                if env._viewer is None or not env._viewer.is_running():
                    break

                obs, info = env.reset()
                print(f"\n--- Episode {ep + 1}/{args.episodes} Started (Terrain: {env.terrain.current_type.upper()}) ---")
                pos0 = env._get_base_pos()
                z0_floor = env.terrain.get_height(pos0[0], pos0[1])
                print(f"  Spawned at: Base Height = {pos0[2]:.4f} m (h_rel = {pos0[2] - z0_floor:.4f} m) [SITTING]")

                env.render()
                time.sleep(0.5)

                step_count = 0
                ep_reward = 0.0
                done = False

                while env._viewer is not None and env._viewer.is_running() and not done:
                    if mode == "random":
                        action = env.action_space.sample()
                        obs, reward, terminated, truncated, step_info = env.step(action)
                    elif mode == "stand":
                        action = np.zeros(12, dtype=np.float32)
                        obs, reward, terminated, truncated, step_info = env.step(action)
                    elif mode == "zero":
                        action = np.zeros(12, dtype=np.float32)
                        obs, reward, terminated, truncated, step_info = env.step(action)
                    elif mode == "hold-sit":
                        q = env._get_joint_pos()
                        qdot = env._get_joint_vel()
                        tau = env._pd.compute(crouch_q, q, qdot)
                        env.data.ctrl[env._actuator_ids] = tau
                        for _ in range(env.n_substeps):
                            mujoco.mj_step(env.model, env.data)
                        reward = 0.0
                        terminated = False
                        truncated = (step_count + 1 >= 500)
                        diag = get_runtime_contact_diagnostics(env)
                        step_info = {"h_rel": diag["h_rel"], "is_standing": False, "contact_diag": diag}

                    step_count += 1
                    ep_reward += reward
                    done = terminated or truncated

                    if mode == "hold-sit" and (step_count % 25 == 0 or done):
                        print_runtime_contact_diagnostics(step_count, step_info["contact_diag"])
                    elif step_count % 50 == 0 or done:
                        h = step_info.get("h_rel", 0.0)
                        standing_str = "STAND" if step_info.get("is_standing", False) else "SIT/LOW"
                        print(f"  Step {step_count:4d} | h_rel = {h:.4f} m | {standing_str} | reward = {reward:+.3f}")

                    env.render()
                    time.sleep(ctrl_dt)

                term_str = "Fell/Terminated" if terminated else ("Timeout" if truncated else "User Stopped")
                print(f"--- Episode {ep + 1} Ended: {term_str} at step {step_count} (Total Reward: {ep_reward:.2f}) ---")
                time.sleep(0.5)

    finally:
        env.close()
        print("Viewer closed cleanly.")


if __name__ == "__main__":
    main()
