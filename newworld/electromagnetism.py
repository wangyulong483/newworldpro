"""Electromagnetic radiation: the sun as a black-body light and heat source.

The visible light used by the inhabitant's eyes and the radiative heat that
warms his body are the same electromagnetic phenomenon, so the sun is modelled
once as a black body and reused by the optics and thermodynamics layers.

* Planck's law gives the spectral radiance of a black body at temperature T.
* Wien's displacement law gives the peak wavelength ``b/T`` with
  ``b = 2.897 771 955e-3 m K`` (about 502 nm for the Sun's ~5772 K).
* The visible band is 380-750 nm; the fraction of solar power that is visible
  is obtained by integrating Planck's law over that band.
"""

from __future__ import annotations

import math

import numpy as np

from .physics import SOLAR_CONSTANT  # noqa: F401  (re-exported for convenience)

#: Planck constant (exact, SI)
PLANCK_H = 6.62607015e-34  # J s
#: speed of light in vacuum (exact, SI)
LIGHT_SPEED = 299792458.0  # m/s
#: Boltzmann constant (exact, SI)
BOLTZMANN_K = 1.380649e-23  # J/K
#: Wien displacement constant (CODATA)
WIEN_B = 2.897771955e-3  # m K
#: visible band
VISIBLE_LOW_M = 380e-9
VISIBLE_HIGH_M = 750e-9


def planck_spectral_radiance(wavelength_m: float, temperature_k: float) -> float:
    """Spectral radiance of a black body (W sr^-1 m^-3)."""
    if wavelength_m <= 0.0 or temperature_k <= 0.0:
        return 0.0
    a = 2.0 * PLANCK_H * LIGHT_SPEED**2 / wavelength_m**5
    b = PLANCK_H * LIGHT_SPEED / (wavelength_m * BOLTZMANN_K * temperature_k)
    try:
        return a / (math.exp(b) - 1.0)
    except OverflowError:
        return 0.0


def spectral_irradiance(wavelength_m: float, temperature_k: float) -> float:
    """Spectral irradiance of a Lambertian black body (W m^-2 m^-1)."""
    return math.pi * planck_spectral_radiance(wavelength_m, temperature_k)


def wien_peak_wavelength_m(temperature_k: float) -> float:
    return WIEN_B / temperature_k


def band_fraction(temperature_k: float, low_m: float = VISIBLE_LOW_M, high_m: float = VISIBLE_HIGH_M) -> float:
    """Fraction of black-body radiant power inside ``[low, high]``."""
    if temperature_k <= 0.0:
        return 0.0
    wavelengths = np.linspace(low_m, high_m, 1200)
    radiance = np.array([spectral_irradiance(w, temperature_k) for w in wavelengths])
    whole = np.array([spectral_irradiance(w, temperature_k) for w in np.linspace(1e-9, 20e-6, 2400)])
    visible = float(np.trapezoid(radiance, wavelengths))
    total = float(np.trapezoid(whole, np.linspace(1e-9, 20e-6, 2400)))
    return visible / total if total > 0.0 else 0.0


def solar_visible_fraction(sun_temperature_k: float = 5772.0) -> float:
    """Visible fraction of the Sun's radiation (~0.37)."""
    return band_fraction(sun_temperature_k, VISIBLE_LOW_M, VISIBLE_HIGH_M)


def photon_energy_ev(wavelength_m: float) -> float:
    """Photon energy in electron-volts."""
    return PLANCK_H * LIGHT_SPEED / wavelength_m / 1.602176634e-19
