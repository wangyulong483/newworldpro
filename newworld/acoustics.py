"""Acoustics: pressure waves in air, speech production and hearing.

Physics used (all standard):

* sound is a longitudinal pressure wave; the speed of sound in an ideal gas is
  ``c = sqrt(gamma R T)`` (343.1 m/s in dry air at 20 C,
  https://www.engineeringtoolbox.com/speed-sound-d_82.html);
* a point source in a 2-D plane spreads cylindrically, so the pressure
  amplitude falls as ``1/sqrt(r)`` with distance (relative to 1 m);
* a wave emitted at ``t0`` from a source reaches a point at distance ``r``
  after ``r / c`` seconds;
* sound pressure level is ``20 log10(p_rms / 20 uPa)`` (threshold of hearing,
  https://www.nidcd.nih.gov/health/how-do-we-hear).

Speech is produced by a simplified vocal tract: a sequence of voiced
"syllables" (fundamental frequency around 110-190 Hz plus a weaker second
harmonic for timbre), emitted from the mouth.  The inhabitant hears his own
voice through his ears (air conduction); other future listeners will hear the
same waves.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import physics

#: syllable inventory used to make up "words" (a stand-in for phonology)
SYLLABLES = ("ba", "da", "gu", "mi", "no", "su", "la", "ti", "po", "ke", "ra", "sho", "hu", "ya")


def random_word(rng: random.Random, syllables: int | None = None) -> str:
    count = syllables if syllables is not None else rng.choice((1, 2, 2, 3))
    return "-".join(rng.choice(SYLLABLES) for _ in range(count))


@dataclass
class SoundWave:
    """One finite wave packet emitted by a point source."""

    origin_x: float
    origin_y: float
    start_time_s: float
    duration_s: float
    frequency_hz: float
    pressure_pa_1m: float
    phase: float = 0.0

    def pressure(self, x: float, y: float, t: float) -> float:
        if t < self.start_time_s or self.duration_s <= 0.0:
            return 0.0
        dx = x - self.origin_x
        dy = y - self.origin_y
        distance_px = math.hypot(dx, dy)
        delay = distance_px / physics.SPEED_OF_SOUND_PX
        local = t - self.start_time_s - delay
        if local < 0.0 or local > self.duration_s:
            return 0.0
        envelope = math.sin(math.pi * local / self.duration_s)  # smooth on/off
        distance_m = max(0.02, distance_px / physics.PIXELS_PER_METER)
        spreading = 1.0 / math.sqrt(distance_m)  # cylindrical spreading (ref. 1 m)
        return (
            self.pressure_pa_1m
            * spreading
            * envelope
            * math.sin(2.0 * math.pi * self.frequency_hz * local + self.phase)
        )

    def expires(self, t: float) -> bool:
        return t > self.start_time_s + self.duration_s + 2.0


@dataclass
class SoundField:
    """All sound waves currently travelling through the air."""

    waves: list[SoundWave] = field(default_factory=list)
    time: float = 0.0

    def add(self, wave: SoundWave) -> None:
        self.waves.append(wave)

    def advance(self, dt: float) -> None:
        self.time += dt
        if self.waves:
            self.waves = [wave for wave in self.waves if not wave.expires(self.time)]

    def pressure_at(self, x: float, y: float, t: float | None = None) -> float:
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
    pressure_pa_1m: float = 0.02,  # ~60 dB SPL at 1 m, normal speech level
) -> SpeechEvent:
    """Turn a made-up word into a sequence of voiced syllable wave packets."""
    syllables = [part for part in text.split("-") if part] or ["ba"]
    f0 = base_f0_hz if base_f0_hz is not None else rng.uniform(110.0, 190.0)
    waves: list[SoundWave] = []
    t = start_time_s
    for syllable in syllables:
        duration = rng.uniform(0.12, 0.28)
        frequency = f0 * rng.uniform(0.92, 1.08)
        # fundamental (voice) plus a weaker second harmonic (timbre)
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
