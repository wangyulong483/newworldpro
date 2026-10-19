"""Physical constants and unit system for newworld.

World coordinates are screen pixels.  Following the eye/camera scale used by
the visual system, **1 pixel = 1 cm** (``PIXELS_PER_METER = 100``), so the
2560x1600 screen is a 25.6 m x 16 m patch of world and the 1.38 m tall
inhabitant is physically plausible.

Constants are the SI values from the authoritative sources:

* standard gravity ``g = 9.806 65 m/s^2`` (exact, NIST/CODATA)
  https://physics.nist.gov/cgi-bin/cuu/Value?gn
* Stefan-Boltzmann constant ``sigma = 5.670 374 419...e-8 W m^-2 K^-4``
  (exact, NIST/CODATA) https://physics.nist.gov/cgi-bin/cuu/Value?sigma
* speed of sound in dry air at 20 C: ``343.1 m/s``
  https://www.engineeringtoolbox.com/speed-sound-d_82.html
* solar constant (total solar irradiance at 1 AU): ``~1361 W/m^2``
  (NASA/TSI; see :mod:`newworld.electromagnetism`)
* human body temperature ~310.15 K (37 C), hearing range 20 Hz - 20 kHz,
  reference sound pressure 20 uPa (threshold of hearing)
  https://www.nidcd.nih.gov/health/how-do-we-hear
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------
PIXELS_PER_METER = 100.0


def m_to_px(metres: float) -> float:
    return metres * PIXELS_PER_METER


def px_to_m(pixels: float) -> float:
    return pixels / PIXELS_PER_METER


# ---------------------------------------------------------------------------
# mechanics
# ---------------------------------------------------------------------------
#: standard acceleration of gravity (NIST CODATA, exact)
STANDARD_GRAVITY = 9.80665  # m/s^2
GRAVITY_PX = STANDARD_GRAVITY * PIXELS_PER_METER  # px/s^2

#: total solar irradiance at 1 AU (~1361 W/m^2, NASA/TSI)
SOLAR_CONSTANT = 1361.0  # W/m^2

#: speed of the inhabitant's jump (energetic, but still a ballistic arc)
JUMP_SPEED = 5.0  # m/s
JUMP_SPEED_PX = JUMP_SPEED * PIXELS_PER_METER

#: air drag (quadratic): F = 0.5 * rho * Cd * A * v^2
DRAG_COEFFICIENT = 1.0
BODY_FRONTAL_AREA_M2 = 0.45


def air_drag_force(velocity_m_s: float, area_m2: float = BODY_FRONTAL_AREA_M2) -> float:
    """Quadratic drag force in newtons (opposes velocity)."""
    return 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * area_m2 * velocity_m_s * abs(velocity_m_s)


# ---------------------------------------------------------------------------
# air / acoustics
# ---------------------------------------------------------------------------
#: dry air at 20 C, sea level
AIR_DENSITY = 1.225  # kg/m^3
AIR_GAS_CONSTANT = 287.05  # J/(kg K)
AIR_HEAT_CAPACITY_RATIO = 1.4
AIR_TEMPERATURE_K = 293.15  # 20 C
AIR_PRESSURE_PA = 101325.0

#: speed of sound in dry air at 20 C (Engineering ToolBox)
SPEED_OF_SOUND_20C = 343.1  # m/s
SPEED_OF_SOUND_PX = SPEED_OF_SOUND_20C * PIXELS_PER_METER  # px/s

#: human hearing
HEARING_MIN_HZ = 20.0
HEARING_MAX_HZ = 20000.0
REFERENCE_SOUND_PRESSURE_PA = 20e-6  # 0 dB SPL


def sound_speed(temperature_k: float = AIR_TEMPERATURE_K) -> float:
    """Ideal-gas speed of sound ``c = sqrt(gamma R T)`` (m/s)."""
    return math.sqrt(AIR_HEAT_CAPACITY_RATIO * AIR_GAS_CONSTANT * temperature_k)


def sound_speed_px(temperature_k: float = AIR_TEMPERATURE_K) -> float:
    return sound_speed(temperature_k) * PIXELS_PER_METER


def spl_db(pressure_rms_pa: float) -> float:
    """Sound pressure level in dB re 20 uPa."""
    if pressure_rms_pa <= 0.0:
        return 0.0
    return 20.0 * math.log10(pressure_rms_pa / REFERENCE_SOUND_PRESSURE_PA)


# ---------------------------------------------------------------------------
# thermodynamics
# ---------------------------------------------------------------------------
#: Stefan-Boltzmann constant (NIST CODATA, exact)
STEFAN_BOLTZMANN = 5.670374419e-8  # W m^-2 K^-4

HUMAN_BODY_TEMPERATURE_K = 310.15  # 37 C
BODY_MASS_KG = 35.0
BODY_SURFACE_AREA_M2 = 1.2
BODY_SPECIFIC_HEAT = 3500.0  # J/(kg K)
METABOLIC_RATE_W = 100.0  # resting heat production
SKIN_EMISSIVITY = 0.98
SKIN_SOLAR_ABSORPTIVITY = 0.50
CONVECTIVE_COEFFICIENT = 5.0  # W/(m^2 K), natural convection
SOLAR_ABSORBING_AREA_M2 = 0.20
#: simplified physiological thermoregulation (vasomotion/sweating): a
#: proportional term that pulls the core temperature back towards 37 C
REGULATION_GAIN_W_PER_K = 120.0


def jump_height_m(speed_m_s: float = JUMP_SPEED) -> float:
    """Ballistic peak height ``v^2 / (2 g)`` in metres (no drag)."""
    return speed_m_s * speed_m_s / (2.0 * STANDARD_GRAVITY)
