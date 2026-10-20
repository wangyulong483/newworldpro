"""居民的生理：能量、水分、睡眠与昼夜节律。

稳态驱动力让自由生物有**自己的**行动理由（而不是被外部控制）：
能量随活动消耗、吃浆果恢复；水分下降、喝水恢复；睡眠压力白天
累积、夜里睡觉释放。昼夜节律分量来自太阳钟（警觉性正午高、
夜间低）；体温、心率、呼吸则对活动、时刻与物理热平衡做出响应。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import physics


@dataclass
class Physiology:
    """生物的内在状态（除注明外全部取值 0..100）。"""

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
        """按 ``dt`` 秒推进内在状态（能量/水分/昼夜节律/心率/呼吸）。"""
        activity = 1.6 if moving else 1.0
        # 能量：基础代谢 + 活动；跳跃一次性消耗
        self.energy -= (0.045 + 0.10 * activity) * dt
        if jumped:
            self.energy -= 1.8
        self.energy = min(100.0, max(0.0, self.energy))

        # 水分：日晒/活动时流失更快
        self.hydration -= (0.030 + 0.020 * activity + 0.020 * daylight) * dt
        self.hydration = min(100.0, max(0.0, self.hydration))

        # 昼夜节律：警觉性 14:00 达峰、夜间低迷；睡眠压力夜间积累更快
        self.circadian_phase = 2.0 * math.pi * (hour - 6.0) / 24.0
        circadian = 0.5 + 0.5 * math.cos(2.0 * math.pi * (hour - 14.0) / 24.0)
        night = 1.0 - daylight
        if self.state == "sleep":
            self.sleep_pressure -= 0.42 * dt
        else:
            self.sleep_pressure += (0.075 + 0.060 * night) * dt
        self.sleep_pressure = min(100.0, max(0.0, self.sleep_pressure))
        self.alertness = max(0.0, min(100.0, 100.0 * circadian - 0.55 * self.sleep_pressure + 15.0))

        # 生理指标对活动与体温的响应（平滑趋近目标值）
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
        """是否饥饿（能量 < 35）。"""
        return self.energy < 35.0

    @property
    def thirsty(self) -> bool:
        """是否口渴（水分 < 38）。"""
        return self.hydration < 38.0

    @property
    def sleepy(self) -> bool:
        """是否困倦（睡眠压力 > 62）。"""
        return self.sleep_pressure > 62.0

    @property
    def sleeping(self) -> bool:
        """是否正在睡觉。"""
        return self.state == "sleep"

    def eat(self) -> None:
        """吃一个浆果：能量 +26、唤醒度 +12。"""
        self.energy = min(100.0, self.energy + 26.0)
        self.arousal = min(100.0, self.arousal + 12.0)

    def drink(self) -> None:
        """喝水：水分 +30。"""
        self.hydration = min(100.0, self.hydration + 30.0)

    def sleep(self) -> None:
        """进入睡眠状态。"""
        self.state = "sleep"

    def wake(self) -> None:
        """醒来。"""
        self.state = "awake"

    def summary_zh(self) -> str:
        """生理状态摘要（中文，UI 调试面板展示用）。"""
        state = "睡眠" if self.sleeping else "清醒"
        return (
            f"{state} 能量{self.energy:4.0f} 水分{self.hydration:4.0f} "
            f"困意{self.sleep_pressure:4.0f} 警觉{self.alertness:4.0f} "
            f"心率{self.heart_rate:4.0f} 呼吸{self.breathing_rate:4.1f}"
        )
