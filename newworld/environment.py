"""World environment: sun, air and the body's heat balance.

The world has physical *fields* rather than painted decorations:

* a **sun** (black-body radiator at ~5772 K) above the world that supplies both
  visible light (used by the eyes, see :mod:`newworld.electromagnetism`) and
  radiative heat (used here);
* **air** at a temperature and pressure, which carries sound
  (:mod:`newworld.acoustics`) and applies drag;
* a lumped **body heat balance** for the inhabitant: metabolic heat, solar
  absorption, radiative exchange (Stefan-Boltzmann), convection, and a simple
  physiological regulation term that pulls the core temperature back to 37 C.
  This is a physical-thermoregulation hybrid (real homeostasis involves
  vasomotion and sweating); the physical terms use the standard constants.

Nothing here is painted: the world stays visually pure white.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import electromagnetism as em
from . import physics

#: effective surface temperature of the Sun (K); Wien peak ~502 nm
SUN_SURFACE_TEMPERATURE_K = 5772.0


@dataclass
class Sun:
    """A black-body sun above the world."""

    elevation_deg: float = 55.0
    azimuth_deg: float = 20.0
    irradiance: float = physics.SOLAR_CONSTANT  # W/m^2 at normal incidence
    temperature_k: float = SUN_SURFACE_TEMPERATURE_K

    @property
    def sin_elevation(self) -> float:
        return max(0.0, math.sin(math.radians(self.elevation_deg)))

    @property
    def horizontal_irradiance(self) -> float:
        """Irradiance on a horizontal surface (``E = S * sin(elevation)``)."""
        return self.irradiance * self.sin_elevation

    @property
    def visible_irradiance(self) -> float:
        """Visible part of the irradiance (W/m^2)."""
        return self.horizontal_irradiance * em.solar_visible_fraction(self.temperature_k)

    @property
    def peak_wavelength_nm(self) -> float:
        return em.wien_peak_wavelength_m(self.temperature_k) * 1e9

    def light_direction(self) -> tuple[float, float]:
        """Unit vector pointing *from the sun into the world* (screen coords)."""
        elevation = math.radians(self.elevation_deg)
        azimuth = math.radians(self.azimuth_deg)
        # 2-D projection: the sun is up and to one side; y axis points down
        horizontal = math.sin(azimuth) * math.cos(elevation)
        vertical = -math.sin(elevation)
        length = math.hypot(horizontal, vertical) or 1.0
        return horizontal / length, vertical / length


@dataclass
class Air:
    """The medium around the world (participates in sound and drag)."""

    temperature_k: float = physics.AIR_TEMPERATURE_K
    density_kg_m3: float = physics.AIR_DENSITY
    pressure_pa: float = physics.AIR_PRESSURE_PA

    @property
    def temperature_c(self) -> float:
        return self.temperature_k - 273.15

    @property
    def speed_of_sound_m_s(self) -> float:
        return physics.sound_speed(self.temperature_k)

    @property
    def speed_of_sound_px_s(self) -> float:
        return self.speed_of_sound_m_s * physics.PIXELS_PER_METER


@dataclass
class HeatBalance:
    """The terms of the lumped body heat-balance equation (all in watts)."""

    metabolic_w: float
    solar_w: float
    radiative_loss_w: float
    convective_loss_w: float
    regulation_w: float

    @property
    def net_w(self) -> float:
        return (
            self.metabolic_w
            + self.solar_w
            - self.radiative_loss_w
            - self.convective_loss_w
            + self.regulation_w
        )


def body_heat_balance(
    body_temperature_k: float,
    air_temperature_k: float,
    solar_irradiance: float,
    *,
    activity: float = 1.0,
) -> HeatBalance:
    """Heat balance of the lumped body model.

    * metabolism: ``Q_met = 100 W * activity``
    * solar absorption: ``Q_sun = E * absorptivity * absorbing area``
    * radiation: ``sigma * epsilon * A * (T_body^4 - T_env^4)``
    * convection: ``h * A * (T_body - T_air)``
    * regulation: ``-K * (T_body - 37 C)`` (simplified homeostasis)
    """
    temperature = max(100.0, body_temperature_k)
    environment = max(100.0, air_temperature_k)
    metabolic = physics.METABOLIC_RATE_W * activity
    solar = solar_irradiance * physics.SKIN_SOLAR_ABSORPTIVITY * physics.SOLAR_ABSORBING_AREA_M2
    radiative = (
        physics.SKIN_EMISSIVITY
        * physics.STEFAN_BOLTZMANN
        * physics.BODY_SURFACE_AREA_M2
        * (temperature**4 - environment**4)
    )
    convective = (
        physics.CONVECTIVE_COEFFICIENT
        * physics.BODY_SURFACE_AREA_M2
        * (temperature - environment)
    )
    regulation = -physics.REGULATION_GAIN_W_PER_K * (
        temperature - physics.HUMAN_BODY_TEMPERATURE_K
    )
    return HeatBalance(
        metabolic_w=metabolic,
        solar_w=solar,
        radiative_loss_w=radiative,
        convective_loss_w=convective,
        regulation_w=regulation,
    )


def step_body_temperature(
    character,
    air: Air,
    sun: Sun,
    dt: float,
    *,
    activity: float = 1.0,
) -> HeatBalance:
    """Integrate the lumped body temperature by one physical step."""
    balance = body_heat_balance(
        character.body_temperature_k,
        air.temperature_k,
        sun.horizontal_irradiance,
        activity=activity,
    )
    heat_capacity = physics.BODY_MASS_KG * physics.BODY_SPECIFIC_HEAT
    character.body_temperature_k = min(
        450.0,
        max(200.0, character.body_temperature_k + balance.net_w / heat_capacity * dt),
    )
    character.heat_balance = balance
    return balance
