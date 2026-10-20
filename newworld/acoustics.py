"""声学：空气中的压力波、语音生成与听觉。

用到的物理（全部是标准结论）：

* 声音是纵向压力波；理想气体中声速 ``c = sqrt(gamma R T)``
  （20 C 干燥空气中为 343.1 m/s，
  https://www.engineeringtoolbox.com/speed-sound-d_82.html）；
* 2-D 平面上的点源按柱面扩散，因此声压幅值随距离按 ``1/sqrt(r)``
  衰减（以 1 m 为参考）；
* ``t0`` 时刻从声源发出的波，要经过 ``r / c`` 秒才到达距离 ``r`` 处；
* 声压级 = ``20 log10(p_rms / 20 µPa)``（听阈，
  https://www.nidcd.nih.gov/health/how-do-we-hear）。

语音由一个简化的"声道"产生：一串有声"音节"（基频约 110-190 Hz，
外加一个更弱的二次谐波作为音色），从嘴的位置发出。居民通过自己的
耳朵（空气传导）听到自己的声音；未来的其他听者也会听到同样的波。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import physics

#: 用来拼"词"的音节表（音系的替代品）
SYLLABLES = ("ba", "da", "gu", "mi", "no", "su", "la", "ti", "po", "ke", "ra", "sho", "hu", "ya")


def random_word(rng: random.Random, syllables: int | None = None) -> str:
    """随机拼一个"词"：1-3 个音节用连字符连接。"""
    count = syllables if syllables is not None else rng.choice((1, 2, 2, 3))
    return "-".join(rng.choice(SYLLABLES) for _ in range(count))


@dataclass
class SoundWave:
    """点源发出的一个有限长波包。"""

    origin_x: float
    origin_y: float
    start_time_s: float
    duration_s: float
    frequency_hz: float
    pressure_pa_1m: float
    phase: float = 0.0

    def pressure(self, x: float, y: float, t: float) -> float:
        """时刻 ``t``、位置 (x, y) 处的瞬时声压（帕斯卡）。

        包含传播延迟、柱面扩散衰减、正弦开关包络与载波振荡。
        """
        if t < self.start_time_s or self.duration_s <= 0.0:
            return 0.0
        dx = x - self.origin_x
        dy = y - self.origin_y
        distance_px = math.hypot(dx, dy)
        delay = distance_px / physics.SPEED_OF_SOUND_PX
        local = t - self.start_time_s - delay
        if local < 0.0 or local > self.duration_s:
            return 0.0
        envelope = math.sin(math.pi * local / self.duration_s)  # 平滑开/关
        distance_m = max(0.02, distance_px / physics.PIXELS_PER_METER)
        spreading = 1.0 / math.sqrt(distance_m)  # 柱面扩散（参考 1 m）
        return (
            self.pressure_pa_1m
            * spreading
            * envelope
            * math.sin(2.0 * math.pi * self.frequency_hz * local + self.phase)
        )

    def expires(self, t: float) -> bool:
        """该波包是否已完全过去（可从声场里清掉）。"""
        return t > self.start_time_s + self.duration_s + 2.0


@dataclass
class SoundField:
    """当前在空气中传播的全部声波。"""

    waves: list[SoundWave] = field(default_factory=list)
    time: float = 0.0

    def add(self, wave: SoundWave) -> None:
        """向声场加入一个新波包。"""
        self.waves.append(wave)

    def advance(self, dt: float) -> None:
        """推进声场时钟并清理过期的波包。"""
        self.time += dt
        if self.waves:
            self.waves = [wave for wave in self.waves if not wave.expires(self.time)]

    def pressure_at(self, x: float, y: float, t: float | None = None) -> float:
        """位置 (x, y) 的瞬时总声压（所有波叠加）。"""
        moment = self.time if t is None else t
        return sum(wave.pressure(x, y, moment) for wave in self.waves)

    def rms_pressure_at(
        self,
        x: float,
        y: float,
        t: float | None = None,
        *,
        window_s: float = 0.05,
        samples: int = 12,
    ) -> float:
        """位置 (x, y) 的 RMS 声压（听觉/分贝计算用，窗长默认 50 ms）。"""
        moment = self.time if t is None else t
        if not self.waves:
            return 0.0
        start = moment - window_s
        values = [
            self.pressure_at(x, y, start + index * window_s / max(1, samples - 1))
            for index in range(samples)
        ]
        return math.sqrt(sum(value * value for value in values) / len(values))


@dataclass
class SpeechEvent:
    """一次说话事件：谁、说了什么、何时、产生了哪些波。"""

    speaker_id: int
    text: str
    start_time_s: float
    duration_s: float
    waves: list[SoundWave]


def synthesize_speech(
    origin_x: float,
    origin_y: float,
    start_time_s: float,
    text: str,
    rng: random.Random,
    *,
    speaker_id: int = 0,
    base_f0_hz: float | None = None,
    pressure_pa_1m: float = 0.02,  # ~60 dB SPL @ 1 m，正常说话音量
) -> SpeechEvent:
    """把一个"词"合成为一串有声音节波包（基频 + 弱二次谐波）。"""
    syllables = [part for part in text.split("-") if part] or ["ba"]
    f0 = base_f0_hz if base_f0_hz is not None else rng.uniform(110.0, 190.0)
    waves: list[SoundWave] = []
    t = start_time_s
    for syllable in syllables:
        duration = rng.uniform(0.12, 0.28)
        frequency = f0 * rng.uniform(0.92, 1.08)
        # 基频（嗓音）+ 更弱的二次谐波（音色）
        waves.append(
            SoundWave(origin_x, origin_y, t, duration, frequency, pressure_pa_1m)
        )
        waves.append(
            SoundWave(
                origin_x,
                origin_y,
                t,
                duration,
                frequency * 2.0,
                pressure_pa_1m * 0.28,
            )
        )
        t += duration + 0.02
    return SpeechEvent(
        speaker_id=speaker_id,
        text="-".join(syllables),
        start_time_s=start_time_s,
        duration_s=max(0.05, t - start_time_s),
        waves=waves,
    )
