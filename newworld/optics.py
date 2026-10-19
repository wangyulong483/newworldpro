"""Physical light model for the 2D side-view world.

The world is a 2-D plane (x to the right, y downwards) lit by a perfectly
uniform white illumination.  The environment is an infinite white
Lambertian/emissive background with radiance 1.0 in every direction; a future
object is a 2-D shape whose surface reflects a fraction (albedo) of that
uniform illumination.  Under uniform illumination and pure diffuse reflection,
radiance is constant along a ray, so the radiance seen in a direction is simply
the albedo of the first surface hit (or the environment radiance).

That gives a *physically consistent but cheap* light field for vision: rays,
shapes, occlusion, and no dependence on distance.  Once objects are added to
the world (food, other creatures, ...) the same ray caster feeds the eye.

Optics (blur from the pupil / defocus / diffraction) lives in
:mod:`newworld.eye`; the photoreceptor layer lives in :mod:`newworld.retina`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

LinearRGB = tuple[float, float, float]

#: radiance of the infinite white environment (all channels)
WHITE: LinearRGB = (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# 2-D shapes
# ---------------------------------------------------------------------------
class Shape:
    """Base class for things that can be hit by a ray in the 2-D world."""

    #: diffuse albedo (linear RGB)
    albedo: LinearRGB = (0.0, 0.0, 0.0)

    def intersect(
        self, ox: float, oy: float, dx: float, dy: float
    ) -> float | None:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True)
class Disc(Shape):
    """A filled circle in the world plane."""

    cx: float
    cy: float
    radius: float
    albedo: LinearRGB = (0.0, 0.0, 0.0)

    def intersect(self, ox: float, oy: float, dx: float, dy: float) -> float | None:
        fx = ox - self.cx
        fy = oy - self.cy
        b = fx * dx + fy * dy
        c = fx * fx + fy * fy - self.radius * self.radius
        discriminant = b * b - c
        if discriminant < 0.0:
            return None
        root = math.sqrt(discriminant)
        t_near = -b - root
        t_far = -b + root
        if t_near > 0.0:
            return t_near
        if t_far > 0.0:
            return t_far
        return None

    def intersect_many(self, ox, oy, dx, dy):  # type: ignore[override]
        fx = ox - self.cx
        fy = oy - self.cy
        b = fx * dx + fy * dy
        c = fx * fx + fy * fy - self.radius * self.radius
        discriminant = b * b - c
        t = np.full(dx.shape, np.inf, dtype=np.float64)
        valid = discriminant >= 0.0
        root = np.sqrt(np.where(valid, discriminant, 0.0))
        t_near = -b - root
        t_far = -b + root
        t = np.where(valid & (t_near > 0.0), t_near, t)
        t = np.where(valid & (t_near <= 0.0) & (t_far > 0.0), t_far, t)
        return t


@dataclass(frozen=True)
class Box(Shape):
    """An axis-aligned rectangle in the world plane."""

    x0: float
    y0: float
    x1: float
    y1: float
    albedo: LinearRGB = (0.0, 0.0, 0.0)

    def intersect(self, ox: float, oy: float, dx: float, dy: float) -> float | None:
        t_min = -math.inf
        t_max = math.inf
        for origin, direction, low, high in (
            (ox, dx, self.x0, self.x1),
            (oy, dy, self.y0, self.y1),
        ):
            if abs(direction) < 1e-12:
                if origin < low or origin > high:
                    return None
                continue
            inv = 1.0 / direction
            t1 = (low - origin) * inv
            t2 = (high - origin) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None
        if t_max < 0.0:
            return None
        return t_min if t_min > 0.0 else t_max

    def intersect_many(self, ox, oy, dx, dy):  # type: ignore[override]
        t_min = np.full(dx.shape, -np.inf, dtype=np.float64)
        t_max = np.full(dx.shape, np.inf, dtype=np.float64)
        for origin, direction, low, high in (
            (ox, dx, self.x0, self.x1),
            (oy, dy, self.y0, self.y1),
        ):
            parallel = np.abs(direction) < 1e-12
            inside = (origin >= low) & (origin <= high)
            inv = 1.0 / np.where(parallel, 1.0, direction)
            t1 = (low - origin) * inv
            t2 = (high - origin) * inv
            lo = np.minimum(t1, t2)
            hi = np.maximum(t1, t2)
            t_min = np.where(parallel & inside, t_min, np.maximum(t_min, lo))
            t_max = np.where(parallel & inside, t_max, np.minimum(t_max, hi))
        t = np.where((t_min > 0.0) & (t_min <= t_max), t_min, np.inf)
        t = np.where((t_min <= 0.0) & (t_max > 0.0), t_max, t)
        return t


# ---------------------------------------------------------------------------
# scene + panorama
# ---------------------------------------------------------------------------
@dataclass
class Panorama:
    """A horizontal light-field slice as seen by an eye.

    ``rel_angles_deg`` are angles *relative to the gaze direction* (0 = fovea),
    ``radiance`` is (N, 3) linear RGB, ``distance_px`` is the hit distance
    (``inf`` for the white environment).
    """

    rel_angles_deg: np.ndarray
    radiance: np.ndarray
    distance_px: np.ndarray

    def __len__(self) -> int:
        return len(self.rel_angles_deg)


@dataclass
class Scene:
    """The visible world: uniform white environment plus 2-D objects."""

    objects: tuple[Shape, ...] = ()
    environment: LinearRGB = WHITE

    # -- single-ray query -------------------------------------------------
    def trace(self, x: float, y: float, angle_rad: float) -> tuple[LinearRGB, float]:
        """Return ``(radiance, distance)`` for the ray leaving ``(x, y)``.

        The distance is ``math.inf`` when the ray hits the white environment.
        """
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        nearest = math.inf
        hit: LinearRGB | None = None
        for obj in self.objects:
            distance = obj.intersect(x, y, dx, dy)
            if distance is not None and distance < nearest:
                nearest = distance
                hit = obj.albedo
        if hit is None:
            return self.environment, math.inf
        return hit, nearest

    # -- full panorama ----------------------------------------------------
    def panorama(
        self,
        x: float,
        y: float,
        gaze_deg: float,
        fov_deg: float,
        samples_per_deg: float = 120.0,
    ) -> Panorama:
        """Render the light field over ``fov_deg`` around ``gaze_deg``.

        The result is expressed relative to the gaze direction, which is what
        the retina samples.  If the scene has no objects, the whole panorama is
        the constant environment radiance (cheap path).
        """
        samples = max(2, int(round(fov_deg * samples_per_deg)))
        rel = np.linspace(-fov_deg / 2.0, fov_deg / 2.0, samples)
        if not self.objects:
            radiance = np.empty((samples, 3), dtype=np.float32)
            radiance[:, :] = self.environment
            distances = np.full(samples, np.inf, dtype=np.float64)
            return Panorama(rel, radiance, distances)

        angles = math.radians(gaze_deg) + np.radians(rel)
        dx = np.cos(angles)
        dy = np.sin(angles)
        distances = np.full(samples, np.inf, dtype=np.float64)
        radiance = np.empty((samples, 3), dtype=np.float32)
        radiance[:, :] = self.environment
        for obj in self.objects:
            if not hasattr(obj, "intersect_many"):
                continue
            hit = obj.intersect_many(x, y, dx, dy)
            closer = hit < distances
            if closer.any():
                distances = np.where(closer, hit, distances)
                radiance[closer] = obj.albedo
        return Panorama(rel, radiance, distances)


def mean_luminance(radiance: np.ndarray) -> float:
    """Photopic luminance (Rec. 709) of an (N, 3) radiance array."""
    if radiance.size == 0:
        return 0.0
    weights = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
    return float(np.mean(radiance @ weights))
