"""昼夜循环：时刻、太阳高度、天空颜色与日照强度。

世界有一个虚拟时钟。一整天 = ``day_length_s`` 秒仿真时间（默认 10 分钟）。
太阳 06:00 升起、12:00 达到最大高度角 ``max_elevation_deg``、18:00 落下；
夜晚高度角为负。太阳高度角驱动光照、阴影、天空颜色，并（经由环境模块）
影响气温与角色的昼夜生理（体温/睡眠压力）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _lerp(a: float, b: float, t: float) -> float:
    """线性插值。"""
    return a + (b - a) * t


def _lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """RGB 三通道各自线性插值。"""
    return (
        int(round(_lerp(a[0], b[0], t))),
        int(round(_lerp(a[1], b[1], t))),
        int(round(_lerp(a[2], b[2], t))),
    )


#: 天空颜色关键帧：(小时, 颜色)。黎明/黄昏是暖色，正午是亮蓝，深夜近黑。
_SKY_KEYS = (
    (0.0, (8, 12, 26)),
    (4.5, (10, 16, 34)),
    (6.0, (240, 148, 104)),
    (7.5, (196, 224, 240)),
    (12.0, (210, 235, 246)),
    (16.5, (222, 231, 236)),
    (18.0, (246, 150, 102)),
    (19.5, (18, 22, 44)),
    (24.0, (8, 12, 26)),
)


def sky_colour_at(hour: float) -> tuple[int, int, int]:
    """按小时（0..24）在关键帧之间插值出天空颜色。"""
    hour = hour % 24.0
    for index in range(len(_SKY_KEYS) - 1):
        h0, c0 = _SKY_KEYS[index]
        h1, c1 = _SKY_KEYS[index + 1]
        if h0 <= hour <= h1:
            t = 0.0 if h1 == h0 else (hour - h0) / (h1 - h0)
            return _lerp_rgb(c0, c1, t)
    return _SKY_KEYS[-1][1]


@dataclass
class DayCycle:
    """虚拟时钟与太阳几何（一天的长短、日出日落实由常数决定）。"""

    day_length_s: float = 600.0
    start_hour: float = 8.0
    max_elevation_deg: float = 65.0

    # ------------------------------------------------------------------
    def hour(self, time_s: float) -> float:
        """仿真秒数 → 一天中的小时（0..24，含 start_hour 偏移）。"""
        return (self.start_hour + time_s / self.day_length_s * 24.0) % 24.0

    def day_index(self, time_s: float) -> int:
        """仿真秒数 → 第几天（从 0 开始）。"""
        return int((self.start_hour + time_s / self.day_length_s * 24.0) // 24.0)

    def sun_elevation_deg(self, hour: float) -> float:
        """太阳高度角：白天（06-18 点）是正弦，夜晚在地平线下为负。"""
        if 6.0 <= hour <= 18.0:
            return self.max_elevation_deg * math.sin(math.pi * (hour - 6.0) / 12.0)
        # 夜晚在地平线之下
        if hour < 6.0:
            night_phase = (hour + 18.0) / 12.0  # 18:00 为 0，06:00 为 1
        else:
            night_phase = (hour - 18.0) / 12.0
        return -self.max_elevation_deg * math.sin(math.pi * night_phase)

    def daylight(self, hour: float) -> float:
        """直射阳光强度 0..1（高度角正弦，夜间为 0）。"""
        elevation = self.sun_elevation_deg(hour)
        if elevation <= 0.0:
            return 0.0
        return math.sin(math.radians(min(90.0, elevation)))

    def sun_horizontal_frac(self, hour: float) -> float:
        """太阳在天幕上的水平位置：0 = 最左，1 = 最右。"""
        if 6.0 <= hour <= 18.0:
            return 0.5 - 0.5 * math.cos(math.pi * (hour - 6.0) / 12.0)
        return 0.5

    def ambient_light(self, hour: float) -> float:
        """环境光 0..1（夜晚 ~0.08 的月光，白天 ~1）。"""
        daylight = self.daylight(hour)
        return 0.08 + 0.92 * daylight

    def is_day(self, hour: float) -> bool:
        """现在是否白天（太阳在地平线上）。"""
        return self.sun_elevation_deg(hour) > 0.0

    def phase_name_zh(self, hour: float) -> str:
        """按时段返回中文名（黎明/上午/正午/下午/黄昏/夜晚）。"""
        if 5.0 <= hour < 7.0:
            return "黎明"
        if 7.0 <= hour < 11.0:
            return "上午"
        if 11.0 <= hour < 14.0:
            return "正午"
        if 14.0 <= hour < 17.5:
            return "下午"
        if 17.5 <= hour < 19.5:
            return "黄昏"
        return "夜晚"

    def sky_colour(self, hour: float) -> tuple[int, int, int]:
        """当前天空颜色（关键帧插值）。"""
        return sky_colour_at(hour)

    # ------------------------------------------------------------------
    def sun_screen_fraction(self, hour: float) -> tuple[float, float]:
        """太阳在屏幕上的近似位置 (x_frac, y_frac)，两者都在 0..1。"""
        x = self.sun_horizontal_frac(hour)
        elevation = max(0.0, self.sun_elevation_deg(hour))
        y = 0.42 - 0.30 * math.sin(math.radians(elevation))
        return x, y

    def sun_direction(self, hour: float) -> int:
        """太阳在屏幕中线哪一侧：左 -1 / 右 +1（阴影方向用）。"""
        return -1 if self.sun_horizontal_frac(hour) < 0.5 else 1
