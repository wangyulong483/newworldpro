"""The inhabitant's physiology: energy, hydration, sleep, circadian rhythm.

Homeostatic drives give the free creature its own reasons to act (it is not
controlled from outside): energy drops with activity and is restored by eating
berries; hydration drops and is restored by drinking; sleep pressure rises
during the day and is discharged by sleeping at night.  A circadian component
comes from the solar clock (alertness high around midday, low at night), and
body temperature / heart rate / breathing respond to activity, time of day and
the physical heat balance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import physics


@dataclass
class Physiology:
    """Internal state of the creature (all values 0..100 unless noted)."""

    energy: float = 78.0
    hydration: float = 80.0
    sleep_pressure: float = 12.0
    alertness: float = 55.0
    arousal: float = 35.0
    heart_rate: float = 66.0
    breathing_rate: float = 14.0
    state: str = "awake"  # "awake" | "sleep"
    circadian_phase: float = 0.0

    # ------------------------------------------------------------------
    def update(
        self,
        dt: float,
        *,
        hour: float,
        daylight: float,
        moving: bool,
        jumped: bool,
        body_temperature_c: float,
    ) -> None:
        activity = 1.6 if moving else 1.0
        # energy: metabolism + activity; jumping costs a burst
        self.energy -= (0.045 + 0.10 * activity) * dt
        if jumped:
            self.energy -= 1.8
        self.energy = min(100.0, max(0.0, self.energy))

        # hydration: more loss in sunlight / when active
        self.hydration -= (0.030 + 0.020 * activity + 0.020 * daylight) * dt
        self.hydration = min(100.0, max(0.0, self.hydration))

        # circadian rhythm: alertness peaks around 14:00, dips at night
        self.circadian_phase = 2.0 * math.pi * (hour - 6.0) / 24.0
        circadian = 0.5 + 0.5 * math.cos(2.0 * math.pi * (hour - 14.0) / 24.0)
        night = 1.0 - daylight
        if self.state == "sleep":
            self.sleep_pressure -= 0.42 * dt
        else:
            self.sleep_pressure += (0.075 + 0.060 * night) * dt
        self.sleep_pressure = min(100.0, max(0.0, self.sleep_pressure))
        self.alertness = max(0.0, min(100.0, 100.0 * circadian - 0.55 * self.sleep_pressure + 15.0))

        # physiology responds to activity and temperature
        target_heart = 58.0 + 30.0 * (activity - 1.0) / 0.6 + 0.25 * (body_temperature_c - 37.0) * 10.0
        target_heart += 8.0 if self.state == "sleep" else 0.0
        self.heart_rate += (max(50.0, min(170.0, target_heart)) - self.heart_rate) * min(1.0, dt / 2.0)
        target_breath = 11.0 + 6.0 * (activity - 1.0) / 0.6
        self.breathing_rate += (target_breath - self.breathing_rate) * min(1.0, dt / 2.0)
        self.arousal = max(
            0.0,
            min(100.0, 0.6 * self.alertness + 0.4 * (100.0 - self.sleep_pressure) * 0.6 + (10.0 if moving else 0.0)),
        )

    # ------------------------------------------------------------------
    @property
    def hungry(self) -> bool:
        return self.energy < 35.0

    @property
    def thirsty(self) -> bool:
        return self.hydration < 38.0

    @property
    def sleepy(self) -> bool:
        return self.sleep_pressure > 62.0

    @property
    def sleeping(self) -> bool:
        return self.state == "sleep"

    def eat(self) -> None:
        self.energy = min(100.0, self.energy + 26.0)
        self.arousal = min(100.0, self.arousal + 12.0)

    def drink(self) -> None:
        self.hydration = min(100.0, self.hydration + 30.0)

    def sleep(self) -> None:
        self.state = "sleep"

    def wake(self) -> None:
        self.state = "awake"

    def summary_zh(self) -> str:
        state = "睡眠" if self.sleeping else "清醒"
        return (
            f"{state} 能量{self.energy:4.0f} 水分{self.hydration:4.0f} "
            f"困意{self.sleep_pressure:4.0f} 警觉{self.alertness:4.0f} "
            f"心率{self.heart_rate:4.0f} 呼吸{self.breathing_rate:4.1f}"
        )
