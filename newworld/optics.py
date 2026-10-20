"""2-D 侧视世界的物理光照模型。

世界是一个 2-D 平面（x 向右、y 向下），被完全均匀的白光照明。
环境是一个无限大的白色朗伯/发光背景，向每个方向的辐射亮度都是
1.0；物体是 2-D 形状，其表面反射该均匀照明的一部分（反照率）。
在均匀照明与纯漫反射下，辐射亮度沿光线不变，因此某一方向看到的
亮度就 simply 是第一个被命中表面的反照率（或环境亮度）。

这给视觉提供了一个**物理自洽但便宜**的光场：光线、形状、遮挡，
且与距离无关。一旦世界里加了物体（食物、其他生物……），同一个
光线投射器就喂给眼睛。

光学（瞳孔/失焦/衍射造成的模糊）在 :mod:`newworld.eye`；
感光细胞层在 :mod:`newworld.retina`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

LinearRGB = tuple[float, float, float]

#: 无限白色环境的辐射亮度（所有通道）
WHITE: LinearRGB = (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# 2-D 形状
# ---------------------------------------------------------------------------
class Shape:
    """2-D 世界中能被光线命中的东西的基类。"""

    #: 漫反射反照率（线性 RGB）
    albedo: LinearRGB = (0.0, 0.0, 0.0)

    def intersect(
        self, ox: float, oy: float, dx: float, dy: float
    ) -> float | None:  # pragma: no cover - 抽象方法
        """与射线求交（子类实现）。"""
        raise NotImplementedError


@dataclass(frozen=True)
class Disc(Shape):
    """世界平面上的一个实心圆。"""

    cx: float
    cy: float
    radius: float
    albedo: LinearRGB = (0.0, 0.0, 0.0)

    def intersect(self, ox: float, oy: float, dx: float, dy: float) -> float | None:
        """单射线求交：解二次方程，返回命中距离（未命中 None）。"""
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
        """批量射线求交（numpy 向量化）：返回每条射线的命中距离。"""
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
    """世界平面上的一个轴对齐矩形。"""

    x0: float
    y0: float
    x1: float
    y1: float
    albedo: LinearRGB = (0.0, 0.0, 0.0)

    def intersect(self, ox: float, oy: float, dx: float, dy: float) -> float | None:
        """单射线与 AABB 求交（经典的 slab 法）。"""
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
        """批量射线与 AABB 求交（numpy 向量化）。"""
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
# 场景 + 全景
# ---------------------------------------------------------------------------
@dataclass
class Panorama:
    """眼睛看到的一个水平光场切片。

    ``rel_angles_deg`` 是**相对注视方向**的角度（0 = 中央凹），
    ``radiance`` 是 (N, 3) 线性 RGB，``distance_px`` 是命中距离
    （命中白色环境时为 inf）。
    """

    rel_angles_deg: np.ndarray
    radiance: np.ndarray
    distance_px: np.ndarray

    def __len__(self) -> int:
        """采样点数。"""
        return len(self.rel_angles_deg)


@dataclass
class Scene:
    """可见世界：均匀白色环境 + 若干 2-D 物体。"""

    objects: tuple[Shape, ...] = ()
    environment: LinearRGB = WHITE

    # -- 单射线查询 ----------------------------------------------------
    def trace(self, x: float, y: float, angle_rad: float) -> tuple[LinearRGB, float]:
        """从 (x, y) 沿 ``angle_rad`` 方向投射一条光线，返回 (亮度, 距离)。

        光线命中白色环境时距离为 ``math.inf``。
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

    # -- 完整全景 -------------------------------------------------------
    def panorama(
        self,
        x: float,
        y: float,
        gaze_deg: float,
        fov_deg: float,
        samples_per_deg: float = 120.0,
    ) -> Panorama:
        """渲染 ``gaze_deg`` 周围 ``fov_deg`` 范围内的光场。

        结果以相对注视方向的角度表示（视网膜采样要的就是这个）。
        场景没有物体时整幅全景都是常数环境亮度（快速路径）。
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
        # 逐物体批量求交，保留最近命中（遮挡）
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
    """(N, 3) 辐射亮度阵列的光度学亮度（Rec. 709 加权）。"""
    if radiance.size == 0:
        return 0.0
    weights = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
    return float(np.mean(radiance @ weights))
