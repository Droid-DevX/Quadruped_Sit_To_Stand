from __future__ import annotations

import math
from typing import Tuple, Optional, Dict, Any

import numpy as np


class TerrainGenerator:
    """Generate a MuJoCo-compatible heightfield and query its sampled surface."""

    def __init__(
        self,
        nrow: int = 401,
        ncol: int = 401,
        size_x: float = 4.0,
        size_y: float = 4.0,
        max_height: float = 0.40,
        depth: float = 0.10,
    ):
        if nrow < 2 or ncol < 2:
            raise ValueError("nrow and ncol must both be >= 2")
        if size_x <= 0 or size_y <= 0 or max_height <= 0 or depth < 0:
            raise ValueError("heightfield sizes must be positive (depth may be zero)")

        self.nrow = int(nrow)
        self.ncol = int(ncol)
        self.size_x = float(size_x)
        self.size_y = float(size_y)
        self.max_height = float(max_height)
        self.depth = float(depth)

        # MuJoCo hfield size[0:2] are half-extents, so the surface spans
        # [-size_x, size_x] x [-size_y, size_y].
        self.x_coords = np.linspace(-self.size_x, self.size_x, self.ncol, dtype=np.float64)
        self.y_coords = np.linspace(-self.size_y, self.size_y, self.nrow, dtype=np.float64)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords, indexing="xy")

        self.current_type: str = "flat"
        self.current_params: Dict[str, Any] = {}
        self._elevation_meters = np.zeros((self.nrow, self.ncol), dtype=np.float32)

    @property
    def hfield_size(self) -> Tuple[float, float, float, float]:
        return (self.size_x, self.size_y, self.max_height, self.depth)

    @property
    def sample_spacing(self) -> Tuple[float, float]:
        """Distance between adjacent samples in x and y, in metres."""
        return (
            2.0 * self.size_x / (self.ncol - 1),
            2.0 * self.size_y / (self.nrow - 1),
        )

    def _finish(self) -> np.ndarray:
        """Validate elevations and return row-major normalized MuJoCo data."""
        z = np.asarray(self._elevation_meters, dtype=np.float32)
        if z.shape != (self.nrow, self.ncol):
            raise RuntimeError(
                f"Bad elevation shape {z.shape}; expected {(self.nrow, self.ncol)}"
            )
        if not np.isfinite(z).all():
            raise ValueError("Terrain contains NaN or infinite elevations")
        np.clip(z, 0.0, self.max_height, out=z)
        self._elevation_meters = z

        # MuJoCo hfield_data is normalized to [0, 1], not metres.
        data = (z / self.max_height).astype(np.float32, copy=False).ravel(order="C")
        if data.size != self.nrow * self.ncol:
            raise RuntimeError("Incorrect number of heightfield samples")
        return data.copy()

    def _to_hfield_data(self) -> np.ndarray:
        """Backward-compatible method used by environment.py during reset.

        Returns the current terrain as MuJoCo-normalized samples in row-major
        order. This method intentionally does not regenerate or alter terrain.
        """
        return self._finish()

    def generate_flat(self) -> np.ndarray:
        self.current_type = "flat"
        self.current_params = {}
        self._elevation_meters.fill(0.0)
        return self._finish()

    def generate_slope(self, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
        """Generate a planar slope near the centre, clipped only at far boundaries."""
        if not (-35.0 <= pitch_deg <= 35.0):
            raise ValueError("pitch_deg must be between -35 and 35 degrees")
        if not (-25.0 <= roll_deg <= 25.0):
            raise ValueError("roll_deg must be between -25 and 25 degrees")

        self.current_type = "slope"
        self.current_params = {"pitch_deg": float(pitch_deg), "roll_deg": float(roll_deg)}
        z_center = 0.20
        z = (
            z_center
            + self.X * math.tan(math.radians(pitch_deg))
            + self.Y * math.tan(math.radians(roll_deg))
        )
        self._elevation_meters = np.clip(z, 0.0, self.max_height).astype(np.float32)
        return self._finish()

    def generate_stairs(
        self,
        step_height: float = 0.04,
        step_width: float = 0.28,
        direction: str = "up",
        landing_half_width: Optional[float] = None,
        riser_ramp_width: Optional[float] = None,
    ) -> np.ndarray:
        """Generate continuous stair treads across the whole robot area.

        No flat central landing is created: stairs continue under the spawn area.
        Heightfield risers are narrow smooth ramps because a MuJoCo heightfield
        cannot represent perfectly vertical steps. The tread centred near x=0
        is the reference level at z=0.20 m.
        """
        if step_height <= 0 or step_width <= 0:
            raise ValueError("step_height and step_width must be positive")
        if direction not in ("up", "down"):
            raise ValueError("direction must be 'up' or 'down'")

        dx, _ = self.sample_spacing
        if riser_ramp_width is None:
            # Use 8 sample intervals for the riser ramp — wide enough for MuJoCo's
            # triangle mesh to form a smooth slope rather than a near-vertical wall.
            # Narrow ramps (< 3 samples) cause the collision solver to tunnel.
            riser_ramp_width = 8.0 * dx
        riser_ramp_width = float(riser_ramp_width)
        if riser_ramp_width < dx:
            raise ValueError(
                f"riser_ramp_width must be at least one sample interval ({dx:.4f} m)"
            )
        # Clamp riser ramp to at most 60% of step width to preserve a flat tread surface.
        riser_ramp_width = min(riser_ramp_width, 0.60 * float(step_width))

        self.current_type = "stairs"
        self.current_params = {
            "step_height": float(step_height),
            "step_width": float(step_width),
            "direction": direction,
            "riser_ramp_width": riser_ramp_width,
        }

        sign = 1.0 if direction == "up" else -1.0
        z_reference = 0.20

        # Centre a flat tread at x=0 so the robot does not spawn on a riser.
        # Steps extend continuously across the complete terrain in X and Y.
        distance = self.X + 0.5 * float(step_width)
        level_index = np.floor(distance / float(step_width)).astype(np.int32)
        within_step = np.mod(distance, float(step_width))
        ramp_start = float(step_width) - riser_ramp_width

        idx = level_index.astype(np.float64)
        ramp_mask = within_step >= ramp_start
        t = np.clip((within_step - ramp_start) / riser_ramp_width, 0.0, 1.0)
        smooth_t = t * t * (3.0 - 2.0 * t)

        z = z_reference + sign * idx * float(step_height)
        z = np.where(
            ramp_mask,
            z_reference + sign * (idx + smooth_t) * float(step_height),
            z,
        )
        self._elevation_meters = np.clip(z, 0.0, self.max_height).astype(np.float32)
        return self._finish()

    def generate_rough(
        self,
        roughness: float = 0.03,
        smoothness: int = 5,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if roughness < 0:
            raise ValueError("roughness must be non-negative")
        if smoothness < 1:
            raise ValueError("smoothness must be >= 1")
        if rng is None:
            rng = np.random.default_rng()

        self.current_type = "rough"
        self.current_params = {"roughness": float(roughness), "smoothness": int(smoothness)}
        z_center = 0.08

        broad = rng.normal(0.0, 1.0, size=(self.nrow, self.ncol)).astype(np.float32)
        fine = rng.normal(0.0, 1.0, size=(self.nrow, self.ncol)).astype(np.float32)

        def smooth_noise(field: np.ndarray, width: int) -> np.ndarray:
            width = max(3, int(width) | 1)
            kernel = np.ones(width, dtype=np.float32) / width
            out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, field)
            out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, out)
            out_std = float(np.std(out))
            return out / max(out_std, 1e-6)

        broad = smooth_noise(broad, max(9, smoothness * 3))
        fine = smooth_noise(fine, max(3, smoothness))
        filtered = roughness * (0.70 * broad + 0.30 * fine)

        # Apply continuous roughness across the entire terrain including spawn area
        z = z_center + filtered

        self._elevation_meters = np.clip(z, 0.0, self.max_height).astype(np.float32)
        return self._finish()

    def get_height(self, x: float, y: float) -> float:
        """Sample the piecewise-triangle heightfield surface at world (x, y).

        Each grid cell is split into triangles. This avoids bilinear interpolation,
        which can report a height different from the collision mesh between samples.
        """
        x = float(np.clip(x, -self.size_x, self.size_x))
        y = float(np.clip(y, -self.size_y, self.size_y))
        u = (x + self.size_x) / (2.0 * self.size_x) * (self.ncol - 1)
        v = (y + self.size_y) / (2.0 * self.size_y) * (self.nrow - 1)

        c0 = min(int(math.floor(u)), self.ncol - 2)
        r0 = min(int(math.floor(v)), self.nrow - 2)
        du = float(np.clip(u - c0, 0.0, 1.0))
        dv = float(np.clip(v - r0, 0.0, 1.0))

        z00 = float(self._elevation_meters[r0, c0])
        z01 = float(self._elevation_meters[r0, c0 + 1])
        z10 = float(self._elevation_meters[r0 + 1, c0])
        z11 = float(self._elevation_meters[r0 + 1, c0 + 1])

        # Piecewise linear interpolation over the cell diagonal (top-left to
        # bottom-right). If the MuJoCo version uses the opposite diagonal, this
        # remains a close height estimate; the physical contacts are authoritative.
        if dv <= du:
            # Triangle z00, z01, z11
            return z00 + du * (z01 - z00) + dv * (z11 - z01)
        # Triangle z00, z10, z11
        return z00 + dv * (z10 - z00) + du * (z11 - z10)

    def get_normal(self, x: float, y: float, eps: Optional[float] = None) -> np.ndarray:
        """Estimate the local normal from the sampled surface."""
        if eps is None:
            dx, dy = self.sample_spacing
            eps = max(min(dx, dy), 1e-3)
        eps = float(max(eps, 1e-4))

        # A heightfield stair transition is deliberately a ramp; avoid a large
        # finite-difference radius that averages several different treads.
        dx = (self.get_height(x + eps, y) - self.get_height(x - eps, y)) / (2.0 * eps)
        dy = (self.get_height(x, y + eps) - self.get_height(x, y - eps)) / (2.0 * eps)
        normal = np.array([-dx, -dy, 1.0], dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        return normal / max(norm, 1e-12)


class DiscreteTerrain:
    """Analytical queries for discrete rigid MuJoCo scenes (flat, slope, stairs)."""

    def __init__(self, terrain_type: str = "flat", step_width: float = 0.35, step_height: float = 0.08, pitch_deg: float = 10.0):
        self.current_type = terrain_type
        self.step_width = float(step_width)
        self.step_height = float(step_height)
        self.pitch_deg = float(pitch_deg)
        self.current_params = {
            "step_width": self.step_width,
            "step_height": self.step_height,
            "pitch_deg": self.pitch_deg,
        }

    def _to_hfield_data(self) -> np.ndarray:
        return np.zeros(401 * 401, dtype=np.float32)

    def get_height(self, x: float, y: float) -> float:
        if self.current_type == "flat":
            return 0.0
        elif self.current_type == "slope":
            # For plane with normal [nx, ny, nz] where nx*x + ny*y + nz*z = 0:
            # z(x) = -(nx/nz) * x
            theta = math.radians(self.pitch_deg)
            # Plane normal from euler="0 0.174533 0" is [sin(theta), 0, cos(theta)]
            nx = math.sin(theta)
            nz = math.cos(theta)
            return -(nx / nz) * float(x)
        elif self.current_type == "stairs":
            # Discrete steps along X: step_width=0.35m, step_height=0.08m
            # Ground: x < -0.525 -> z = 0.0
            # step_n1: [-0.525, -0.175], top = 0.080
            # step_0:  [-0.175,  0.175], top = 0.160
            # step_1:  [ 0.175,  0.525], top = 0.240
            # step_2:  [ 0.525,  0.875], top = 0.320
            # step_3:  [ 0.875,  1.225], top = 0.400
            # step_4:  [ 1.225,  1.575], top = 0.480
            # step_5:  [ 1.575,  3.275], top = 0.560
            x_f = float(x)
            if x_f < -0.525:
                return 0.0
            elif x_f < -0.175:
                return 0.080
            elif x_f < 0.175:
                return 0.160
            elif x_f < 0.525:
                return 0.240
            elif x_f < 0.875:
                return 0.320
            elif x_f < 1.225:
                return 0.400
            elif x_f < 1.575:
                return 0.480
            else:
                return 0.560
        return 0.0

    def get_normal(self, x: float, y: float, eps: Optional[float] = None) -> np.ndarray:
        if self.current_type == "flat" or self.current_type == "stairs":
            # Each step tread is perfectly horizontal
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        elif self.current_type == "slope":
            theta = math.radians(self.pitch_deg)
            # Plane normal from euler="0 0.174533 0": [sin(theta), 0, cos(theta)]
            n = np.array([math.sin(theta), 0.0, math.cos(theta)], dtype=np.float64)
            return n / np.linalg.norm(n)
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

