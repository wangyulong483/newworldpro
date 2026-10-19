"""Day/night cycle: time of day, solar elevation, sky colour and daylight.

The world has a virtual clock.  A full day lasts ``day_length_s`` real seconds
(default 10 minutes).  The sun rises at 06:00, culminates at 12:00 with
``max_elevation_deg`` and sets at 18:00; at night the elevation is negative.
Solar elevation drives illumination, shadows, sky colour and (through the
environment) air temperature and the inhabitant's circadian physiology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(round(_lerp(a[0], b[0], t))),
        int(round(_lerp(a[1], b[1], t))),
        int(round(_lerp(a[2], b[2], t))),
    )


#: sky colour keyframes: (hour, colour)
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
    """Interpolate the sky colour for a given hour (0..24)."""
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
    """Virtual clock and solar geometry."""

    day_length_s: float = 600.0
    start_hour: float = 8.0
    max_elevation_deg: float = 65.0

    # ------------------------------------------------------------------
    def hour(self, time_s: float) -> float:
        return (self.start_hour + time_s / self.day_length_s * 24.0) % 24.0

    def day_index(self, time_s: float) -> int:
        return int((self.start_hour + time_s / self.day_length_s * 24.0) // 24.0)

    def sun_elevation_deg(self, hour: float) -> float:
        """Solar elevation: sinusoid between sunrise (06) and sunset (18)."""
        if 6.0 <= hour <= 18.0:
            return self.max_elevation_deg * math.sin(math.pi * (hour - 6.0) / 12.0)
        # below the horizon at night
        if hour < 6.0:
            night_phase = (hour + 18.0) / 12.0  # 0 at 18:00, 1 at 06:00
        else:
            night_phase = (hour - 18.0) / 12.0
        return -self.max_elevation_deg * math.sin(math.pi * night_phase)

    def daylight(self, hour: float) -> float:
        """Direct sunlight strength 0..1 (sin of elevation, clamped)."""
        elevation = self.sun_elevation_deg(hour)
        if elevation <= 0.0:
            return 0.0
        return math.sin(math.radians(min(90.0, elevation)))

    def sun_horizontal_frac(self, hour: float) -> float:
        """Where the sun sits across the sky: 0 = left, 1 = right."""
        if 6.0 <= hour <= 18.0:
            return 0.5 - 0.5 * math.cos(math.pi * (hour - 6.0) / 12.0)
        return 0.5

    def ambient_light(self, hour: float) -> float:
        """Ambient illumination 0..1 (night ~0.08 moonlight, day ~1)."""
        daylight = self.daylight(hour)
        return 0.08 + 0.92 * daylight

    def is_day(self, hour: float) -> bool:
        return self.sun_elevation_deg(hour) > 0.0

    def phase_name_zh(self, hour: float) -> str:
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
        return sky_colour_at(hour)

    # ------------------------------------------------------------------
    def sun_screen_fraction(self, hour: float) -> tuple[float, float]:
        """Approximate screen-space sun position (x_frac, y_frac)."""
        x = self.sun_horizontal_frac(hour)
        elevation = max(0.0, self.sun_elevation_deg(hour))
        y = 0.42 - 0.30 * math.sin(math.radians(elevation))
        return x, y

    def sun_direction(self, hour: float) -> int:
        """-1 if the sun is left of centre, +1 if right (for shadows)."""
        return -1 if self.sun_horizontal_frac(hour) < 0.5 else 1
