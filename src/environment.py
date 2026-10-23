from __future__ import annotations
import os, math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import mujoco
try:
    import mujoco.viewer
except Exception:
    pass
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from terrain import TerrainGenerator, DiscreteTerrain

_THIS_DIR       = Path(__file__).resolve().parent
_MODEL_PATH     = str(_THIS_DIR.parent / "a1" / "scene.xml")
_SCENE_FLAT     = str(_THIS_DIR.parent / "a1" / "scene_flat.xml")
_SCENE_STAIRS   = str(_THIS_DIR.parent / "a1" / "scene_stairs.xml")
_SCENE_SLOPE    = str(_THIS_DIR.parent / "a1" / "scene_slope.xml")
_SCENE_ROUGH    = str(_THIS_DIR.parent / "a1" / "scene_rough.xml")


class PDController:
    """Explicit PD controller: tau = Kp*(q_target-q) - Kd*qdot, clamped to torque_limit."""
    _KP = {"abduction": 40.0, "hip": 80.0, "knee": 80.0}
    _KD = {"abduction":  1.0, "hip":  2.0, "knee":  2.0}

    def __init__(self, joint_type_list, torque_limit=33.5):
        self.kp = np.array([self._KP[t] for t in joint_type_list], dtype=np.float64)
        self.kd = np.array([self._KD[t] for t in joint_type_list], dtype=np.float64)
        self.torque_limit = torque_limit

    def compute(self, q_target, q, qdot):
        tau = self.kp * (q_target - q) - self.kd * qdot
        return np.clip(tau, -self.torque_limit, self.torque_limit)


def _quat_to_rot(q):
    """MuJoCo quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _gravity_in_body(quat_world):
    """Project world gravity [0, 0, -1] into robot body frame."""
    return _quat_to_rot(quat_world).T @ np.array([0.0, 0.0, -1.0])


def _tilt_angle(quat_world):
    """Angle [rad] between trunk z-axis and world z."""
    return math.acos(float(np.clip(_quat_to_rot(quat_world)[2, 2], -1.0, 1.0)))


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for MuJoCo-order quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    q = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)


def _euler_quat(roll: float, pitch: float, yaw: float = 0.0) -> np.ndarray:
    """Convert XYZ Euler angles to a MuJoCo-order quaternion."""
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ], dtype=np.float64)


def align_quat_to_normal(normal: np.ndarray, yaw: float) -> np.ndarray:
    """Compute MuJoCo quaternion (w,x,y,z) aligning body z with normal, with given yaw."""
    z_axis = np.array([0.0, 0.0, 1.0])
    normal = normal / np.linalg.norm(normal)
    v = np.cross(z_axis, normal)
    c = np.dot(z_axis, normal)
    if c < -0.99999:
        R_align = np.diag([1.0, -1.0, -1.0])
    else:
        s = np.linalg.norm(v)
        if s < 1e-6:
            R_align = np.eye(3)
        else:
            k = v / s
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R_align = np.eye(3) + s * K + (1.0 - c) * (K @ K)

    cy, sy = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_total = R_align @ R_yaw

    tr = R_total[0, 0] + R_total[1, 1] + R_total[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R_total[2, 1] - R_total[1, 2]) / S
        qy = (R_total[0, 2] - R_total[2, 0]) / S
        qz = (R_total[1, 0] - R_total[0, 1]) / S
    elif (R_total[0, 0] > R_total[1, 1]) and (R_total[0, 0] > R_total[2, 2]):
        S = np.sqrt(1.0 + R_total[0, 0] - R_total[1, 1] - R_total[2, 2]) * 2
        qw = (R_total[2, 1] - R_total[1, 2]) / S
        qx = 0.25 * S
        qy = (R_total[0, 1] + R_total[1, 0]) / S
        qz = (R_total[0, 2] + R_total[2, 0]) / S
    elif R_total[1, 1] > R_total[2, 2]:
        S = np.sqrt(1.0 + R_total[1, 1] - R_total[0, 0] - R_total[2, 2]) * 2
        qw = (R_total[0, 2] - R_total[2, 0]) / S
        qx = (R_total[0, 1] + R_total[1, 0]) / S
        qy = 0.25 * S
        qz = (R_total[1, 2] + R_total[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R_total[2, 2] - R_total[0, 0] - R_total[1, 1]) * 2
        qw = (R_total[1, 0] - R_total[0, 1]) / S
        qx = (R_total[0, 2] + R_total[2, 0]) / S
        qy = (R_total[1, 2] + R_total[2, 1]) / S
        qz = 0.25 * S
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    return quat / np.linalg.norm(quat)


# ---------------------------------------------------------------------------
# Constants derived from a1.xml
# ---------------------------------------------------------------------------
_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
_ACTUATOR_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]
_JOINT_TYPES = ["abduction", "hip", "knee"] * 4

# Joint limits from a1.xml [rad]
_JOINT_LO = np.array([-0.802851, -1.0472,  -2.69653 ] * 4, dtype=np.float64)
_JOINT_HI = np.array([ 0.802851,  4.18879, -0.916298] * 4, dtype=np.float64)

# Sit / crouch starting pose — loaded from the 'sit' keyframe in a1.xml at runtime.
# Kept as module-level fallback for environments that haven't called __init__ yet.
_SIT_BASE_HEIGHT = 0.134   # z-height of base in the 'sit' keyframe [m]
_SIT_JOINTS_DEFAULT = np.array([
    -0.10, 1.00, -2.50,
     0.10, 1.00, -2.50,
    -0.10, 1.00, -2.50,
     0.10, 1.00, -2.50,
], dtype=np.float64)
# Legacy aliases (still valid as fallbacks)
_CROUCH_JOINTS = _SIT_JOINTS_DEFAULT
_SIT_JOINTS    = _SIT_JOINTS_DEFAULT
_CROUCH_QPOS   = np.concatenate([[0.0, 0.0, _SIT_BASE_HEIGHT, 1.0, 0.0, 0.0, 0.0], _SIT_JOINTS_DEFAULT])
_SIT_QPOS      = _CROUCH_QPOS

# Standing target pose
_STAND_QPOS = np.array([
    0.0, 0.0, 0.27,
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
], dtype=np.float64)
_STAND_JOINTS = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64)

# Timing
_PHYSICS_DT = 0.002
_CONTROL_DT = 0.02
_N_SUBSTEPS = int(round(_CONTROL_DT / _PHYSICS_DT))  # 10

# Heights relative to terrain [m]
# Calf length is 0.20m. Specification requires base height above terrain > calf length.
_TRUNK_STAND_MIN    = 0.21    # > 0.20m calf length
_TRUNK_STAND_TARGET = 0.27    # nominal standing height
_TRUNK_CROUCH_INIT  = _SIT_BASE_HEIGHT   # crouched resting height
_TRUNK_SIT_INIT     = _TRUNK_CROUCH_INIT

# Horizon & Stability
_MAX_EPISODE_STEPS = 500      # 10 seconds max
_STABILITY_STEPS   = 100      # 2.0 second stability hold window at 50 Hz (configurable via hold_duration)
_STABLE_LIN_VEL_MAX = 0.25    # m/s, base translational speed for a stable stand
_STABLE_ANG_VEL_MAX = 0.60    # rad/s, base angular speed for a stable stand


class A1SitToStandEnv(gym.Env):
    """Gymnasium Environment: Unitree A1 Blind Sit-to-Stand on Multi-Terrain."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    ACTION_SCALE = 0.25  # [rad] — tuned to +/- 14.3 deg to eliminate twitching while ensuring full control authority

    def __init__(self, model_path=None, render_mode=None, seed=None,
                 init_noise_scale=0.03, terrain_type="random", curriculum_level=0.0,
                 enable_domain_randomization=True, enable_inertia_randomization=True,
                 inertia_randomization_range=(0.90, 1.10), debug=False, rise_duration=3.8,
                 use_discrete_scenes=True, hold_duration=2.0):
        super().__init__()
        self.use_discrete_scenes = bool(use_discrete_scenes)
        self.terrain_type = terrain_type
        self.hold_duration = float(hold_duration)
        self._stability_steps_target = max(1, int(round(self.hold_duration / _CONTROL_DT)))

        # Select model path based on terrain_type if using discrete rigid scenes
        if model_path is None:
            if self.use_discrete_scenes:
                if terrain_type == "flat":
                    model_path = _SCENE_FLAT
                elif terrain_type == "stairs":
                    model_path = _SCENE_STAIRS
                elif terrain_type == "slope":
                    model_path = _SCENE_SLOPE
                elif terrain_type == "rough":
                    model_path = _SCENE_ROUGH
                else:
                    model_path = _MODEL_PATH
            else:
                model_path = _MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        self._model_path = model_path
        if debug:
            print(f"[A1SitToStandEnv] Loaded XML: {os.path.basename(model_path)} ({model_path}) | Terrain Mode: {self.terrain_type}")
        self.rise_duration = float(rise_duration)
        if not 2.0 <= self.rise_duration <= 5.0:
            raise ValueError("rise_duration must be between 2.0 and 5.0 seconds")

        # Domain randomization flags and configuration
        self.enable_domain_randomization = bool(enable_domain_randomization)
        self.enable_inertia_randomization = bool(enable_inertia_randomization)
        self.inertia_randomization_range = tuple(inertia_randomization_range)
        # Diagnostics distinguish requested configuration from what was actually
        # applied on the most recent reset.
        self._last_domain_randomization_applied = False
        self._last_inertia_randomization_applied = False

        # Lookups
        self._joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in _JOINT_NAMES], dtype=np.int32)
        self._qpos_ids  = np.array(
            [self.model.jnt_qposadr[j] for j in self._joint_ids], dtype=np.int32)
        self._qvel_ids  = np.array(
            [self.model.jnt_dofadr[j]  for j in self._joint_ids], dtype=np.int32)
        self._actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in _ACTUATOR_NAMES], dtype=np.int32)
        self._trunk_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self._trunk_contact_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "trunk_main_collision")
        self._terrain_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        # Identify all terrain / obstacle geoms belonging to the static world body (body 0)
        self._terrain_geom_ids = set(
            gid for gid in range(self.model.ngeom)
            if self.model.geom_bodyid[gid] == 0
        )
        # Identify all 4 foot sphere collision geoms (type 2 attached to calf bodies)
        calf_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            for b in ("FR_calf", "FL_calf", "RR_calf", "RL_calf")
        }
        self._feet_geom_ids = set(
            gid for gid in range(self.model.ngeom)
            if self.model.geom_bodyid[gid] in calf_body_ids and self.model.geom_type[gid] == 2
        )
        if self._trunk_contact_geom_id < 0:
            raise ValueError(
                "a1.xml must name the main trunk collision geom 'trunk_main_collision' "
                "so reset can guarantee main-torso contact."
            )

        # Nominal dynamics parameters for domain randomization
        self._nominal_trunk_mass = float(self.model.body_mass[self._trunk_id])
        self._nominal_body_masses = self.model.body_mass.copy()
        self._nominal_body_inertia = self.model.body_inertia.copy()
        self._nominal_geom_friction = self.model.geom_friction.copy()
        self._nominal_friction = float(self.model.geom_friction[self._terrain_geom_id, 0])

        self._pd = PDController(_JOINT_TYPES, torque_limit=33.5)
        if self.use_discrete_scenes and self.terrain_type in ("flat", "stairs", "slope"):
            self.terrain = DiscreteTerrain(terrain_type=self.terrain_type)
        else:
            self.terrain = TerrainGenerator(
                nrow=401,
                ncol=401,
                size_x=4.0,
                size_y=4.0,
                max_height=0.40,
                depth=0.10,
            )

        # ----------------------------------------------------------------
        # Load the 'sit' keyframe from a1.xml as the canonical start pose.
        # This makes a1.xml the single source of truth — editing the
        # keyframe there propagates to every reset automatically.
        # ----------------------------------------------------------------
        sit_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "sit")
        if sit_key_id >= 0:
            # key_qpos is a 2D array: shape (nkeys, nq) — index directly by key id row
            self._sit_qpos   = self.model.key_qpos[sit_key_id].copy()   # shape (nq,) = (19,)
            self._sit_joints = self._sit_qpos[7:].copy()                 # 12 joint angles
            self._sit_base_z = float(self._sit_qpos[2])                  # z-height in keyframe
        else:
            # Fallback: use module-level defaults if keyframe not found
            import warnings
            warnings.warn("'sit' keyframe not found in model — using default pose constants.")
            self._sit_qpos   = _CROUCH_QPOS.copy()
            self._sit_joints = _SIT_JOINTS_DEFAULT.copy()
            self._sit_base_z = _SIT_BASE_HEIGHT
        self._sit_base_height_rel = self._sit_base_z

        # Alias for inspect_reset.py compatibility
        self._crouch_q = self._sit_joints.copy()

        # 27-dim Strictly Compliant Blind Observation Space:
        # [0:12] joint positions (12 dim)
        # [12:24] joint velocities (12 dim)
        # [24:27] projected gravity in body frame representing base orientation (3 dim)
        # NO contact sensors, terrain heightmaps, terrain normals, base angvel, or prev actions.
        obs_dim = 27
        self.observation_space = spaces.Box(
            np.full(obs_dim, -np.inf, dtype=np.float32),
            np.full(obs_dim,  np.inf, dtype=np.float32), dtype=np.float32)
        self.action_space = spaces.Box(
            np.full(12, -1.0, dtype=np.float32),
            np.full(12,  1.0, dtype=np.float32), dtype=np.float32)

        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)
        self._init_noise = init_noise_scale
        self.terrain_type = terrain_type
        # Training uses [0, 1]. Values up to 1.2 are reserved for explicit OOD tests.
        self.curriculum_level = float(np.clip(curriculum_level, 0.0, 1.2))
        self._debug = debug

        # Episode state & anti-jerk filters
        self._prev_action = np.zeros(12, dtype=np.float64)
        self._applied_action = np.zeros(12, dtype=np.float64)
        self._delayed_action = np.zeros(12, dtype=np.float64)
        self._prev_qdot = np.zeros(12, dtype=np.float64)
        self._step_count = 0
        self._stable_steps = 0
        self._standing = False
        self._first_stand_step = None
        self._success = False
        self._viewer = None

    def get_nominal_pose(self, t_sec: float):
        """Compute smooth S-curve reference trajectory over the configured 2–5s rise."""
        s = float(np.clip(t_sec / max(self.rise_duration, 1e-4), 0.0, 1.0))
        # Minimum-jerk cosine S-curve with zero initial & final velocity/acceleration
        alpha = 0.5 * (1.0 - math.cos(math.pi * s))
        q_nominal = (1.0 - alpha) * self._sit_joints + alpha * _STAND_JOINTS
        h_nominal = (1.0 - alpha) * self._sit_base_height_rel + alpha * _TRUNK_STAND_TARGET
        return q_nominal, h_nominal, alpha

    def set_curriculum_level(self, level: float):
        """Set curriculum level; training uses [0, 1], OOD evaluation may use up to 1.2."""
        self.curriculum_level = float(np.clip(level, 0.0, 1.2))

    def _get_joint_pos(self):          return self.data.qpos[self._qpos_ids].copy()
    def _get_joint_vel(self):          return self.data.qvel[self._qvel_ids].copy()
    def _get_base_pos(self):           return self.data.qpos[0:3].copy()

    def _get_terrain_ground_z(self, pos):
        """Compute the effective terrain support elevation under the robot.
        On flat or planar slopes, this matches terrain.get_height(pos[0], pos[1]).
        On discrete stairs, where the front feet rest on step k+1 and rear feet on step k,
        the effective terrain ground reference is the mean elevation of the step treads
        supporting the feet, avoiding arbitrary step-edge discretization offsets.
        """
        if self.terrain.current_type == "stairs" and hasattr(self, "_feet_geom_ids"):
            foot_tzs = [self.terrain.get_height(self.data.geom_xpos[fid][0], self.data.geom_xpos[fid][1])
                        for fid in self._feet_geom_ids]
            return float(np.mean(foot_tzs))
        return float(self.terrain.get_height(pos[0], pos[1]))
    def _get_base_quat(self):          return self.data.qpos[3:7].copy()
    def _get_base_ang_vel_world(self): return self.data.qvel[3:6].copy()
    def _get_base_lin_vel_world(self): return self.data.qvel[0:3].copy()
    def _ang_vel_in_body(self):
        return _quat_to_rot(self._get_base_quat()).T @ self._get_base_ang_vel_world()

    def _get_obs(self):
        """Blind proprioceptive observation strictly compliant with assignment (27-dim).
        [ 0:12] joint positions (12 dim)
        [12:24] joint velocities (12 dim)
        [24:27] projected gravity in body frame representing base orientation (3 dim)
        """
        obs = np.concatenate([
            self._get_joint_pos(),
            self._get_joint_vel(),
            _gravity_in_body(self._get_base_quat()),
        ]).astype(np.float32)
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        return obs

    def _apply_domain_randomization(self):
        """Randomize friction, masses, and inertia with physical validity guarantees."""
        # 1. Always restore nominal parameters before applying randomization
        # This prevents cumulative drift across repeated resets.
        self.model.body_mass[:] = self._nominal_body_masses
        self.model.body_inertia[:] = self._nominal_body_inertia
        self.model.geom_friction[:] = self._nominal_geom_friction
        self._last_domain_randomization_applied = False
        self._last_inertia_randomization_applied = False

        if not self.enable_domain_randomization:
            mujoco.mj_setConst(self.model, self.data)
            return

        self._last_domain_randomization_applied = True

        # 2. Friction randomization: mu in [0.4, 1.2]
        mu = float(self._rng.uniform(0.4, 1.2))
        self.model.geom_friction[self._terrain_geom_id, 0] = mu

        # 3. Trunk mass randomization: +-15% (range [0.85, 1.15])
        m_scale_trunk = float(self._rng.uniform(0.85, 1.15))
        self.model.body_mass[self._trunk_id] = self._nominal_trunk_mass * m_scale_trunk

        # 4. Leg links mass randomization: +-10% (range [0.90, 1.10])
        for bid in range(self.model.nbody):
            if bid != self._trunk_id and bid != 0:
                scale = float(self._rng.uniform(0.90, 1.10))
                self.model.body_mass[bid] = self._nominal_body_masses[bid] * scale

        # 5. Explicit, Physically Valid Inertia Tensor Randomization
        # Randomizes principal moments of inertia on trunk and leg links (bodies 1..13)
        if self.enable_inertia_randomization:
            lo, hi = self.inertia_randomization_range
            for bid in range(1, self.model.nbody):
                nominal_mass = float(self._nominal_body_masses[bid])
                nominal_I = self._nominal_body_inertia[bid]
                # Skip world/fixed/massless bodies: their nominal inertias may be zero,
                # which would make mass-ratio scaling undefined or physically invalid.
                if nominal_mass <= 0.0 or not np.all(np.isfinite(nominal_I)) or np.any(nominal_I <= 0.0):
                    continue
                # Sample bounded inertia scaling factor (default: [0.90, 1.10])
                i_scale = float(self._rng.uniform(lo, hi))
                # Coordinate inertia with mass ratio to keep scaling physically sensible (I ~ m * r^2)
                mass_ratio = self.model.body_mass[bid] / nominal_mass
                new_I = nominal_I * mass_ratio * i_scale

                # Enforce physical validity:
                # 1. Strictly positive principal moments (I_x > 0, I_y > 0, I_z > 0)
                # 2. Strict triangle inequalities:
                #    I_x + I_y >= I_z, I_y + I_z >= I_x, I_z + I_x >= I_y
                Ix, Iy, Iz = float(new_I[0]), float(new_I[1]), float(new_I[2])
                eps = 1e-7
                if Ix + Iy < Iz:
                    Iz = Ix + Iy - eps
                if Iy + Iz < Ix:
                    Ix = Iy + Iz - eps
                if Iz + Ix < Iy:
                    Iy = Iz + Ix - eps

                Ix = max(Ix, 1e-7)
                Iy = max(Iy, 1e-7)
                Iz = max(Iz, 1e-7)
                self.model.body_inertia[bid] = np.array([Ix, Iy, Iz], dtype=np.float64)
            self._last_inertia_randomization_applied = True

        # 6. Recompute all derived inertial & dynamic quantities in MuJoCo
        mujoco.mj_setConst(self.model, self.data)

    def _setup_terrain(self):
        """Generate procedural terrain based on curriculum level."""
        if isinstance(self.terrain, DiscreteTerrain):
            return

        selected = self.terrain_type
        if selected == "random":
            # Curriculum-weighted distribution
            p_flat = max(0.1, 1.0 - 0.9 * self.curriculum_level)
            p_other = (1.0 - p_flat) / 3.0
            selected = self._rng.choice(["flat", "slope", "stairs", "rough"],
                                        p=[p_flat, p_other, p_other, p_other])

        lvl = self.curriculum_level
        if selected == "flat":
            self.terrain.generate_flat()
        elif selected == "slope":
            # In-distribution training reaches 15 deg pitch at level 1.0;
            # explicit OOD level 1.2 reaches 18 deg.
            max_pitch = 3.0 + 12.5 * lvl
            max_roll  = 2.0 + (8.0 / 1.2) * lvl
            pitch = float(self._rng.uniform(-max_pitch, max_pitch))
            roll  = float(self._rng.uniform(-max_roll, max_roll))
            self.terrain.generate_slope(pitch_deg=pitch, roll_deg=roll)
        elif selected == "stairs":
            step_h = float(np.clip(0.04 + 0.04 * lvl, 0.04, 0.08))
            step_w = 0.45
            direction = self._rng.choice(["up", "down"])
            self.terrain.generate_stairs(step_height=step_h, step_width=step_w, direction=direction)
        elif selected == "rough":
            # Map curriculum level in [0.0, 1.0] to 4 progressive difficulty stages:
            # Level 1 (lvl <= 0.25): gentle rolling waves, roughness ~0.010, smoothness 21
            # Level 2 (lvl <= 0.50): mild undulations, roughness ~0.018, smoothness 17
            # Level 3 (lvl <= 0.75): moderate roughness, roughness ~0.026, smoothness 13
            # Level 4 (lvl > 0.75): harder uneven terrain, roughness ~0.035, smoothness 11
            diff_level = int(np.clip(math.floor(lvl * 4) + 1, 1, 4))
            self.terrain.generate_rough(level=diff_level, rng=self._rng)

        # Update MuJoCo heightfield data in place if model has hfield
        if hasattr(self.model, "hfield_data") and self.model.nhfield > 0:
            self.model.hfield_data[:] = self.terrain._to_hfield_data()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._apply_domain_randomization()
        self._setup_terrain()

        # Randomized start position.
        # Staircase travels/ascends along +X.
        # Requirement: Front two legs rest on the upper step, rear two legs rest on the lower step.
        # Base is placed at tread_index * step_w + 0.05m so that rear paws rest securely on step k
        # and front paws reach across to step k+1.
        if self.terrain.current_type == "stairs":
            step_w = float(self.terrain.current_params.get("step_width", 0.45))
            tread_index = int(self._rng.integers(-1, 2))
            x0 = float(tread_index * step_w + 0.050 + self._rng.uniform(-0.005, 0.005))
            y0 = float(self._rng.uniform(-0.08, 0.08))
            # On stairs spanning two steps, trunk has natural pitch ~10 deg up to bridge the 8cm step height
            stair_pitch_rad = float(-math.radians(10.0))
        else:
            x0 = float(self._rng.uniform(-0.12, 0.12))
            y0 = float(self._rng.uniform(-0.12, 0.12))
            stair_pitch_rad = 0.0
        z_floor = self.terrain.get_height(x0, y0)
        normal = self.terrain.get_normal(x0, y0)

        # Orientation:
        # On stairs, align the robot's forward axis (+X) directly with the staircase direction
        # of travel (yaw = 0.0 deg with small jitter) and pitch tilted up to bridge the step.
        # On flat, slope, and rough terrain, yaw is randomized uniformly across full [-pi, pi].
        if self.terrain.current_type == "stairs":
            yaw = float(self._rng.uniform(-math.radians(2.0), math.radians(2.0)))
            base_quat = align_quat_to_normal(normal, yaw)
            base_quat = _quat_multiply(base_quat, _euler_quat(0.0, stair_pitch_rad))
            roll_perturb = 0.0
            pitch_perturb = stair_pitch_rad
        else:
            yaw = float(self._rng.uniform(-math.pi, math.pi))
            tilt_deg = 2.0
            base_quat = align_quat_to_normal(normal, yaw)
            roll_perturb = float(self._rng.uniform(math.radians(-tilt_deg), math.radians(tilt_deg)))
            pitch_perturb = float(self._rng.uniform(math.radians(-tilt_deg), math.radians(tilt_deg)))
            base_quat = _quat_multiply(base_quat, _euler_quat(roll_perturb, pitch_perturb))

        # ---------------------------------------------------------------
        # Geometry-informed placement followed by physics-level verification:
        # 1. Start safely above terrain accounting for full collision geometry.
        # 2. Descend iteratively using mj_forward to detect true physical contacts.
        # 3. Fine-adjust to eliminate penetration while ensuring intended contact.
        # ---------------------------------------------------------------
        sit_qpos = self._sit_qpos.copy()
        sit_qpos[0], sit_qpos[1] = x0, y0
        sit_qpos[2] = z_floor + 0.50
        sit_qpos[3:7] = base_quat
        if self.terrain.current_type == "stairs":
            # Adjust front leg joint targets in sitting pose so front paws rest on the upper step
            # FR leg (indices 7, 8): thigh +1.07 rad, calf -2.60 rad
            # FL leg (indices 10, 11): thigh +1.07 rad, calf -2.60 rad
            sit_qpos[7 + 1] = 1.07
            sit_qpos[7 + 2] = -2.60
            sit_qpos[7 + 4] = 1.07
            sit_qpos[7 + 5] = -2.60
        sit_qpos[7:] = np.clip(sit_qpos[7:], _JOINT_LO, _JOINT_HI)
        self.data.qpos[:] = sit_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_kinematics(self.model, self.data)

        # Upper estimate for clearance across all robot geoms
        max_required_base_z = z_floor + float(self._sit_base_z) + 0.05
        for gid in range(self.model.ngeom):
            bid = self.model.geom_bodyid[gid]
            if bid == 0:
                continue
            g_mat = self.data.geom_xmat[gid].reshape(3, 3)
            g_pos = self.data.geom_xpos[gid]
            g_size = self.model.geom_size[gid]
            g_type = self.model.geom_type[gid]
            
            if g_type == 2:
                dz = float(g_size[0])
            elif g_type in (3, 5):
                r, h = float(g_size[0]), float(g_size[1])
                axis_z = float(g_mat[2, 2])
                dz = abs(axis_z) * h + math.sqrt(max(0.0, 1.0 - axis_z * axis_z)) * r
            elif g_type == 6:
                dz = abs(float(g_mat[2, 0])) * float(g_size[0]) + \
                     abs(float(g_mat[2, 1])) * float(g_size[1]) + \
                     abs(float(g_mat[2, 2])) * float(g_size[2])
            else:
                dz = float(g_size[0])

            tz = self.terrain.get_height(float(g_pos[0]), float(g_pos[1]))
            offset_below_base = float(sit_qpos[2]) - (float(g_pos[2]) - dz)
            req = tz + 0.010 + offset_below_base
            max_required_base_z = max(max_required_base_z, req)

        # Physics-based verified descent:
        # Step down in fine increments until the robot establishes valid ground contact
        # without penetrating.
        sit_qpos[2] = max_required_base_z + 0.020
        sit_qpos[3:7] = base_quat
        self.data.qpos[:] = sit_qpos
        mujoco.mj_forward(self.model, self.data)

        # Ensure starting pose has strictly zero collisions across all robot geoms
        while True:
            world_contacts = [
                self.data.contact[i] for i in range(self.data.ncon)
                if (self.data.contact[i].geom1 in self._terrain_geom_ids or
                    self.data.contact[i].geom2 in self._terrain_geom_ids)
            ]
            if not world_contacts:
                break
            sit_qpos[2] += 0.010
            self.data.qpos[:] = sit_qpos
            mujoco.mj_forward(self.model, self.data)

        # Fine-tune base_z with iterative forward collision checks
        # Stage 1: Coarse descent (1mm) until contact is detected or near (within 1.5mm)
        for _ in range(150):
            world_contacts = [
                self.data.contact[i] for i in range(self.data.ncon)
                if (self.data.contact[i].geom1 in self._terrain_geom_ids or
                    self.data.contact[i].geom2 in self._terrain_geom_ids)
            ]
            min_dist = min([c.dist for c in world_contacts], default=1.0)
            if min_dist <= 0.0015:
                break
            sit_qpos[2] -= 0.001
            self.data.qpos[:] = sit_qpos
            mujoco.mj_forward(self.model, self.data)

        # Stage 2: Fine descent (0.2mm) to establish exact intended contact
        for _ in range(60):
            world_contacts = [
                self.data.contact[i] for i in range(self.data.ncon)
                if (self.data.contact[i].geom1 in self._terrain_geom_ids or
                    self.data.contact[i].geom2 in self._terrain_geom_ids)
            ]
            min_dist = min([c.dist for c in world_contacts], default=1.0)
            if min_dist <= 0.0005:
                # Iteratively back off if there is penetration until min_dist >= -0.00005
                while min_dist < -0.00005:
                    sit_qpos[2] -= min_dist
                    self.data.qpos[:] = sit_qpos
                    mujoco.mj_forward(self.model, self.data)
                    world_contacts = [
                        self.data.contact[i] for i in range(self.data.ncon)
                        if (self.data.contact[i].geom1 in self._terrain_geom_ids or
                            self.data.contact[i].geom2 in self._terrain_geom_ids)
                    ]
                    min_dist = min([c.dist for c in world_contacts], default=0.0)
                break
            sit_qpos[2] -= 0.0002
            self.data.qpos[:] = sit_qpos
            mujoco.mj_forward(self.model, self.data)

        # Stage 3: Micro-adjustment if back-off left the robot just outside contact detection
        active_world_contacts = [
            self.data.contact[i] for i in range(self.data.ncon)
            if (self.data.contact[i].geom1 in self._terrain_geom_ids or
                self.data.contact[i].geom2 in self._terrain_geom_ids)
        ]
        if not active_world_contacts:
            for _ in range(25):
                sit_qpos[2] -= 0.00005
                self.data.qpos[:] = sit_qpos
                mujoco.mj_forward(self.model, self.data)
                active_world_contacts = [
                    self.data.contact[i] for i in range(self.data.ncon)
                    if (self.data.contact[i].geom1 in self._terrain_geom_ids or
                        self.data.contact[i].geom2 in self._terrain_geom_ids)
                ]
                if active_world_contacts:
                    break

        # Check torso contact only as a diagnostic/requirement. Never change the
        # base height in response to contacts: reset placement is fixed above.
        torso_ground_contact = False
        for ci in range(self.data.ncon):
            contact = self.data.contact[ci]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if ((g1 == self._terrain_geom_id and g2 == self._trunk_contact_geom_id)
                or (g2 == self._terrain_geom_id and g1 == self._trunk_contact_geom_id)):
                torso_ground_contact = True
                break

        self._prev_action.fill(0.0)
        self._applied_action.fill(0.0)
        self._delayed_action.fill(0.0)
        self._prev_qdot = self._get_joint_vel()
        self._step_count = 0
        self._stable_steps = 0
        self._standing = False
        self._first_stand_step = None
        self._success = False

        pos = self._get_base_pos()
        z_curr_floor = self._get_terrain_ground_z(pos)
        h_rel = pos[2] - z_curr_floor
        self._sit_base_height_rel = float(h_rel)
        # Re-check after all reset bookkeeping, without moving the robot.
        torso_ground_contact = any(
            ((int(self.data.contact[i].geom1) in self._terrain_geom_ids and int(self.data.contact[i].geom2) == self._trunk_contact_geom_id)
             or (int(self.data.contact[i].geom2) in self._terrain_geom_ids and int(self.data.contact[i].geom1) == self._trunk_contact_geom_id))
            for i in range(self.data.ncon)
        )
        # Record contact metrics across full collision geometry
        terrain_dists = [float(self.data.contact[i].dist) for i in range(self.data.ncon)
                         if (int(self.data.contact[i].geom1) in self._terrain_geom_ids
                             or int(self.data.contact[i].geom2) in self._terrain_geom_ids)]
        min_terrain_dist = min(terrain_dists, default=float("inf"))
        
        # Verify intended physical contacts (all feet and low geoms)
        n_foot_contacts = sum(
            1 for i in range(self.data.ncon)
            if (((int(self.data.contact[i].geom1) in self._terrain_geom_ids and int(self.data.contact[i].geom2) in self._feet_geom_ids)
                 or (int(self.data.contact[i].geom2) in self._terrain_geom_ids and int(self.data.contact[i].geom1) in self._feet_geom_ids))
                and self.data.contact[i].dist <= 0.003)
        )
        has_ground_contact = len(terrain_dists) > 0 and min_terrain_dist <= 0.003
        no_penetration = (min_terrain_dist >= -0.001) if np.isfinite(min_terrain_dist) else True
        spawn_valid = bool(has_ground_contact and no_penetration)

        obs = self._get_obs()
        info = {
            "base_height": pos[2],
            "h_rel": h_rel,
            "terrain_type": self.terrain.current_type,
            "curriculum_level": self.curriculum_level,
            "friction": float(self.model.geom_friction[self._terrain_geom_id, 0]),
            "trunk_mass": float(self.model.body_mass[self._trunk_id]),
            "sit_joints_keyframe": self._sit_joints.tolist(),
            "torso_ground_contact": bool(torso_ground_contact),
            "n_foot_contacts": int(n_foot_contacts),
            "spawn_valid": spawn_valid,
            "min_terrain_contact_distance": min_terrain_dist,
            "terrain_penetration_mm": max(0.0, -min_terrain_dist * 1000.0) if np.isfinite(min_terrain_dist) else 0.0,
            "initial_base_position": pos.tolist(),
            "initial_yaw_rad": yaw,
            "initial_roll_perturb_deg": math.degrees(roll_perturb),
            "initial_pitch_perturb_deg": math.degrees(pitch_perturb),
        }
        if self.render_mode == 'human':
            self.render()
        return obs, info

    def _is_fallen(self, h_rel, cos_align):
        """Fall detection relative to terrain surface normal."""
        # Misaligned more than 60 degrees relative to slope normal
        too_tilted = cos_align < 0.50
        # Start safely above the terrain in the configured sitting joint pose.
        # Allow a 1.5 s sit-to-rise grace period; after that, remaining collapsed
        # is a fall. Once standing has been reached, fall_after_stand catches collapse.
        elapsed_sec = self._step_count * _CONTROL_DT
        too_low = (h_rel < 0.075 and elapsed_sec > 1.5 and not self._standing)
        # Fall after having achieved standing
        fall_after_stand = self._standing and (h_rel < 0.16 or cos_align < 0.707)
        return bool(too_tilted or too_low or fall_after_stand)

    def _compute_reward(self, q, qdot, quat, h_rel, cos_align, v_world, w_world, action,
                        q_nominal, h_nominal, t_sec, fallen, is_standing):
        """Reward formulation targeting smooth 2-5s rise, terrain alignment, anti-jerk, and steady balance."""

        # 1. Height tracking along smooth S-curve rise trajectory
        # Actively tracks h_nominal(t) so the robot does NOT jerk or snap up prematurely
        h_err = h_rel - h_nominal
        r_height = float(np.exp(-40.0 * (h_err ** 2)))

        # 2. Terrain alignment: torso axis parallel to terrain normal
        r_align = float(np.exp(-5.0 * (1.0 - np.clip(cos_align, -1.0, 1.0))))

        # 3. Posture tracking along smooth nominal joint progression
        pose_err = float(np.dot(q - q_nominal, q - q_nominal))
        r_pose = float(np.exp(-0.4 * pose_err))

        # 4. Objectives: 2 to 5 second rise and stable balance.
        # Explicitly discourage standing before the required 2.0s lower bound.
        # Strengthened to -4.0 to firmly penalize standing before 2.0s.
        r_early_stand_pen = -4.0 if (is_standing and t_sec < 2.0) else 0.0
        if is_standing and t_sec >= 2.0:
            r_stand_bonus = 1.5
            r_stable_bonus = 1.5 if self._stable_steps >= self._stability_steps_target else 0.0
        else:
            r_stand_bonus = 0.0
            r_stable_bonus = 0.0

        # Regularization penalties
        act_diff = action - self._prev_action
        r_act_smooth = -0.05 * float(np.dot(act_diff, act_diff))
        r_act_mag = -0.01 * float(np.dot(action, action))
        r_jvel_pen = -0.005 * float(np.dot(qdot, qdot))

        qacc = (qdot - self._prev_qdot) / _CONTROL_DT
        mean_sq_acc = float(np.mean(np.square(qacc)))
        r_jacc_pen = -1e-5 * float(np.clip(mean_sq_acc, 0.0, 1e4))

        r_vel_pen    = -0.05  * float(np.dot(v_world, v_world))
        r_angvel_pen = -0.05  * float(np.dot(w_world, w_world))
        r_fall_pen   = -10.0 if fallen else 0.0

        total = (2.0 * r_height + r_align + r_pose + r_stand_bonus + r_stable_bonus
                 + r_early_stand_pen + r_act_smooth + r_act_mag + r_jvel_pen + r_jacc_pen
                 + r_vel_pen + r_angvel_pen + r_fall_pen)

        return float(total), {
            "r_height": r_height, "r_align": r_align, "r_pose": r_pose,
            "r_stand_bonus": r_stand_bonus, "r_stable_bonus": r_stable_bonus,
            "r_early_stand_pen": r_early_stand_pen,
            "r_act_smooth": r_act_smooth, "r_act_mag": r_act_mag,
            "r_jvel_pen": r_jvel_pen, "r_jacc_pen": r_jacc_pen,
            "r_vel_pen": r_vel_pen, "r_angvel_pen": r_angvel_pen,
            "r_fall_pen": r_fall_pen,
            "h_rel": h_rel, "cos_align": cos_align, "is_standing": float(is_standing),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)
        t_sec = self._step_count * _CONTROL_DT

        self._applied_action = 0.70 * action + 0.30 * self._applied_action
        act_to_apply = self._delayed_action if self.enable_domain_randomization else self._applied_action
        self._delayed_action = self._applied_action.copy()

        q_nominal, h_nominal, alpha = self.get_nominal_pose(t_sec)

        # Joint position target: nominal trajectory + policy residual
        q_target = np.clip(q_nominal + act_to_apply * self.ACTION_SCALE, _JOINT_LO, _JOINT_HI)
        tau = self._pd.compute(q_target, self._get_joint_pos(), self._get_joint_vel())

        # Actuator noise
        if self.enable_domain_randomization:
            tau += self._rng.normal(0.0, 0.5, size=12)
            tau = np.clip(tau, -33.5, 33.5)

        self.data.ctrl[self._actuator_ids] = tau

        for _ in range(_N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        q       = self._get_joint_pos()
        qdot    = self._get_joint_vel()
        quat    = self._get_base_quat()
        pos     = self._get_base_pos()
        v_world = self._get_base_lin_vel_world()
        w_world = self._get_base_ang_vel_world()

        # Local terrain queries (Privileged in reward/critic only)
        z_floor = self._get_terrain_ground_z(pos)
        h_rel   = float(pos[2] - z_floor)
        normal  = self.terrain.get_normal(pos[0], pos[1])

        R = _quat_to_rot(quat)
        z_trunk = R[:, 2]
        cos_align = float(np.dot(z_trunk, normal))

        # Standing definition:
        # Base height above terrain > 0.20m (calf length) and aligned with terrain normal
        # Armed only after the sitting grace period (t >= 1.0s) so sitting poses on uneven
        # terrain do not trigger premature standing at step 1.
        elapsed_sec = self._step_count * _CONTROL_DT
        is_standing = (elapsed_sec >= 1.0) and (h_rel >= _TRUNK_STAND_MIN) and (cos_align >= math.cos(math.radians(25.0)))
        lin_speed = float(np.linalg.norm(v_world))
        ang_speed = float(np.linalg.norm(w_world))
        is_stable = (is_standing and lin_speed <= _STABLE_LIN_VEL_MAX
                     and ang_speed <= _STABLE_ANG_VEL_MAX)
        if is_standing:
            self._standing = True
            if self._first_stand_step is None:
                self._first_stand_step = self._step_count
        if is_stable:
            self._stable_steps += 1
        else:
            self._stable_steps = 0

        rise_time_sec = (self._first_stand_step * _CONTROL_DT
                         if self._first_stand_step is not None else None)
        stability_sec = self._stable_steps * _CONTROL_DT
        rise_in_spec = (rise_time_sec is not None and 2.0 <= rise_time_sec <= 5.0)
        # Success criteria: Valid rise initiated between 2.0s and 5.0s, followed by
        # active continuous hold for the required duration (default 2.0s / 100 steps).
        # Standing alone does NOT terminate the episode; only completing the full hold terminates with success.
        success = bool(rise_in_spec and self._stable_steps >= self._stability_steps_target)
        self._success = success

        fallen = self._is_fallen(h_rel, cos_align)
        reward, components = self._compute_reward(
            q, qdot, quat, h_rel, cos_align, v_world, w_world, action,
            q_nominal, h_nominal, t_sec,
            fallen=fallen, is_standing=is_standing)

        terminated = bool(fallen or success)
        truncated  = bool(self._step_count >= _MAX_EPISODE_STEPS and not terminated)

        self._prev_action = action.copy()
        self._prev_qdot = qdot.copy()
        obs = self._get_obs()

        info = {
            "base_height": pos[2],
            "h_rel": h_rel,
            "cos_align": cos_align,
            "tilt_deg": math.degrees(math.acos(np.clip(cos_align, -1.0, 1.0))),
            "is_standing": is_standing,
            "is_stable": is_stable,
            "linear_speed": lin_speed,
            "angular_speed": ang_speed,
            "stable_steps": self._stable_steps,
            "stability_duration_sec": stability_sec,
            "rise_time_sec": rise_time_sec,
            "success": success,
            "termination_reason": "success" if success else ("fall" if fallen else None),
            "step_count": self._step_count,
            "terminated": terminated,
            "truncated": truncated,
            "terrain_type": self.terrain.current_type,
            **{k: v for k, v in components.items() if k != "total"},
        }

        if self._debug and (self._step_count % 50 == 0 or terminated or truncated):
            status = "TERM" if terminated else ("TRUNC" if truncated else "")
            print(f"[{self.terrain.current_type:<6}] Step {self._step_count:4d} | "
                  f"h_rel={h_rel:.3f}m align={cos_align:.3f} stand={int(is_standing)} "
                  f"stable={self._stable_steps} v={lin_speed:.2f} w={ang_speed:.2f} "
                  f"success={int(success)} R={reward:+.3f} ({status})")

        if self.render_mode == 'human':
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            if self._viewer is None and hasattr(mujoco, 'viewer'):
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self._viewer is not None:
                self._viewer.sync()
        elif self.render_mode == 'rgb_array':
            if not hasattr(self, '_renderer') or self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if hasattr(self, '_renderer') and self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    @property
    def model_path(self):               return self._model_path
    @property
    def n_substeps(self):               return _N_SUBSTEPS
    @property
    def physics_dt(self):               return _PHYSICS_DT
    @property
    def control_dt(self):               return _CONTROL_DT
    @property
    def joint_names(self):              return list(_JOINT_NAMES)
    @property
    def actuator_names(self):           return list(_ACTUATOR_NAMES)
    @property
    def joint_limits_lo(self):          return _JOINT_LO.copy()
    @property
    def joint_limits_hi(self):          return _JOINT_HI.copy()
    @property
    def stand_height_target(self):      return _TRUNK_STAND_TARGET
    @property
    def stand_height_min(self):         return _TRUNK_STAND_MIN
    @property
    def crouch_height_init(self):       return _TRUNK_CROUCH_INIT
    @property
    def sit_height_init(self):          return _TRUNK_SIT_INIT
    @property
    def stability_steps_required(self): return self._stability_steps_target
    @property
    def stable_steps(self):             return self._stable_steps

    def print_diagnostics(self):
        """Print clear diagnostics on observation compliance, domain randomization, and link parameters."""
        obs = self._get_obs()
        print("=" * 68)
        print("  A1 SIT-TO-STAND ENVIRONMENT COMPLIANCE & DR DIAGNOSTICS")
        print("=" * 68)
        print(f"  Observation Dimension       : {obs.shape[0]} (Strictly compliant: 27)")
        print(f"  Observation Signals         : [ 0:12] joint positions (12 dim)")
        print(f"                                [12:24] joint velocities (12 dim)")
        print(f"                                [24:27] projected gravity / orientation (3 dim)")
        print(f"  Strict Proprioception Only  : True (zero contacts, zero terrain heightmaps/normals)")
        print(f"  All Observation Finite      : {bool(np.all(np.isfinite(obs)))}")
        print(f"  Domain Randomization Configured: {self.enable_domain_randomization}")
        print(f"  Domain Randomization Applied   : {self._last_domain_randomization_applied}")
        print(f"  Inertia Randomization Configured: {self.enable_inertia_randomization}")
        print(f"  Inertia Randomization Applied   : {self._last_inertia_randomization_applied}")
        print(f"  Inertia Randomization Range : {self.inertia_randomization_range}")
        print(f"  Current Terrain Type        : {self.terrain.current_type.upper()}")
        print(f"  Stable Stand Criteria       : linear speed <= {_STABLE_LIN_VEL_MAX:.2f} m/s; "
              f"angular speed <= {_STABLE_ANG_VEL_MAX:.2f} rad/s for {self._stability_steps_target * _CONTROL_DT:.1f}s hold")
        print("-" * 68)
        print("  SELECTED LINK MASS & INERTIA DIAGNOSTICS:")
        selected_links = ["trunk", "FR_thigh", "FR_calf"]
        for name in selected_links:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            m_nom = self._nominal_body_masses[bid]
            m_cur = float(self.model.body_mass[bid])
            I_nom = self._nominal_body_inertia[bid]
            I_cur = self.model.body_inertia[bid]
            m_pct = (m_cur / m_nom - 1.0) * 100.0
            print(f"  - {name:<10}: mass = {m_cur:.4f} kg (nom: {m_nom:.4f} kg, delta: {m_pct:+.1f}%)")
            print(f"               inertia = [{I_cur[0]:.6f}, {I_cur[1]:.6f}, {I_cur[2]:.6f}]")
            print(f"               nominal = [{I_nom[0]:.6f}, {I_nom[1]:.6f}, {I_nom[2]:.6f}]")
        print("=" * 68)
