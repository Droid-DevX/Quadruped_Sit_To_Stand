# Unitree A1 Quadruped Sit-to-Stand RL (MuJoCo + PPO)

Reinforcement learning framework for training a Unitree A1 quadruped to rise from an initial ground-seated position (torso resting on or near ground) to a stable upright standing posture across multiple terrains using PPO (Proximal Policy Optimization).

---

## Terrains & Curriculum

Training follows a performance-gated curriculum across dedicated discrete MuJoCo XML scenes:

1. **Stage 1: Flat** (`a1/scene_flat.xml`) — Standard horizontal ground plane.
2. **Stage 2: Slope** (`a1/scene_slope.xml`) — Rigid 10° inclined plane.
3. **Stage 3: Rough** (`a1/scene_rough.xml`) — Filtered multi-scale continuous heightfield.
4. **Stage 4: Stairs** (`a1/scene_stairs.xml`) — Discrete rigid stair risers and treads (0.35m tread, 0.08m rise).

The policy advances to the next stage only when achieving $\ge 90\%$ success across rolling evaluation episodes on the active terrain.

---

## State & Action Spaces

- **Observation Space (27 dimensions, Blind Proprioception):**
  - Joint positions (12 dim, rad)
  - Joint velocities (12 dim, rad/s)
  - Projected gravity vector in body frame: $R^T [0, 0, -1]$ (3 dim)
  - *Strictly no contact sensors, camera feeds, heightmaps, or terrain normal information are provided to the policy.*
- **Action Space (12 dimensions in $[-1, 1]$):**
  - Residual target joint offsets scaled by $\pm 0.25$ rad ($\approx \pm 14.3^\circ$).
  - Target angles: $q_{\text{target}} = \text{clip}(q_{\text{nominal}}(t) + 0.25 \cdot a, q_{\text{lo}}, q_{\text{hi}})$.
- **Low-Level PD Controller:**
  - $K_p = [40, 80, 80]$ N·m/rad, $K_d = [1, 2, 2]$ N·m·s/rad (abduction, hip, knee).
  - Torque saturation: $\pm 33.5$ N·m.

---

## Key Criteria for Success

- **Rise Window:** The first stand detection must occur between 2.0 and 5.0 seconds after episode reset.
- **Standing Height:** Base clearance $h_{\text{rel}} \ge 0.21$ m relative to local terrain surface.
- **Torso Alignment:** Trunk $z$-axis aligned with terrain normal ($\text{tilt} \le 25^\circ$).
- **Stability Hold:** Linear velocity $\|v\| \le 0.15$ m/s and angular velocity $\|\omega\| \le 0.50$ rad/s maintained continuously for at least 1.0 s (50 control steps at 50 Hz) within the 5.0 s deadline.

---

## Project Structure

```
├── a1/                     # Unitree A1 MuJoCo models and scene XMLs
│   ├── a1.xml              # Robot definition, joints, actuators, and 'sit' keyframe
│   ├── scene_flat.xml      # Stage 1 scene
│   ├── scene_slope.xml     # Stage 2 scene
│   ├── scene_rough.xml     # Stage 3 scene
│   ├── scene_stairs.xml    # Stage 4 scene
│   └── assets/             # STL meshes
├── src/
│   ├── environment.py      # Gymnasium environment with MuJoCo physics and rewards
│   ├── train.py            # PPO staged curriculum training orchestrator
│   ├── evaluate.py         # Multi-terrain policy benchmark evaluation
│   ├── test_env.py         # Diagnostic reset and contact verification tool
│   ├── terrain.py          # Procedural heightfield and discrete terrain querying
│   └── record_videos.py    # Policy rollout video & GIF rendering with HUD telemetry
├── output/                 # Checkpoints, logs, and evaluation metrics (ignored in git)
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Verification & Diagnostics

Validate terrain loading, contact points, and initial sitting posture without rendering:
```bash
python src/test_env.py --terrain all --check-only
```

Launch an interactive viewer on stairs:
```bash
python src/test_env.py --terrain stairs --interactive
```

Verify curriculum stages and configurations:
```bash
python src/train.py --check-only
```

### 2. Training

Start staged curriculum training:
```bash
python src/train.py --n-envs 4 --run-name ppo_a1_curriculum
```

Resume training from a specific stage or checkpoint:
```bash
python src/train.py --start-stage 1 \
  --checkpoint-model output/models/ppo_a1_curriculum/stage_1_flat_final.zip \
  --checkpoint-vecnorm output/models/ppo_a1_curriculum/stage_1_flat_vecnorm.pkl
```

### 3. Evaluation & Video Recording

Evaluate a trained model across all benchmark terrains:
```bash
python src/evaluate.py --model output/models/ppo_a1_curriculum/stage_4_stairs_final \
  --vecnorm output/models/ppo_a1_curriculum/stage_4_stairs_vecnorm.pkl \
  --terrain benchmark --episodes 25
```

Record demonstration videos with HUD telemetry:
```bash
python src/record_videos.py --model output/models/ppo_a1_curriculum/stage_4_stairs_final \
  --vecnorm output/models/ppo_a1_curriculum/stage_4_stairs_vecnorm.pkl \
  --terrain all --episodes 1
```
