"""电磁辐射：把太阳建模为黑体的光与热源。

居民眼睛用的可见光与温暖身体的辐射热是同一种电磁现象，因此太阳
只按黑体建模一次，由光学层与热学层共用。

* 普朗克定律给出温度 T 的黑体光谱辐射亮度。
* 维恩位移定律给出峰值波长 ``b/T``，其中
  ``b = 2.897 771 955e-3 m·K``（太阳 ~5772 K 时约 502 nm）。
* 可见光波段是 380-750 nm；太阳功率中可见光占比通过在该波段上
  积分普朗克定律得到。
"""

from __future__ import annotations

import math

import numpy as np

from .physics import SOLAR_CONSTANT  # noqa: F401  (为方便使用而重导出)

#: 普朗克常数（SI 精确值）
PLANCK_H = 6.62607015e-34  # J s
#: 真空光速（SI 精确值）
LIGHT_SPEED = 299792458.0  # m/s
#: 玻尔兹曼常数（SI 精确值）
BOLTZMANN_K = 1.380649e-23  # J/K
#: 维恩位移常数（CODATA）
WIEN_B = 2.897771955e-3  # m K
#: 可见光波段
VISIBLE_LOW_M = 380e-9
VISIBLE_HIGH_M = 750e-9


def planck_spectral_radiance(wavelength_m: float, temperature_k: float) -> float:
    """黑体的光谱辐射亮度（W sr^-1 m^-3，普朗克定律）。"""
    if wavelength_m <= 0.0 or temperature_k <= 0.0:
        return 0.0
    a = 2.0 * PLANCK_H * LIGHT_SPEED**2 / wavelength_m**5
    b = PLANCK_H * LIGHT_SPEED / (wavelength_m * BOLTZMANN_K * temperature_k)
    try:
        return a / (math.exp(b) - 1.0)
    except OverflowError:
        return 0.0


def spectral_irradiance(wavelength_m: float, temperature_k: float) -> float:
    """朗伯黑体的光谱辐照度（W m^-2 m^-1，辐射亮度乘 π）。"""
    return math.pi * planck_spectral_radiance(wavelength_m, temperature_k)


def wien_peak_wavelength_m(temperature_k: float) -> float:
    """维恩位移定律：黑体辐射的峰值波长（米）。"""
    return WIEN_B / temperature_k


def band_fraction(temperature_k: float, low_m: float = VISIBLE_LOW_M, high_m: float = VISIBLE_HIGH_M) -> float:
    """黑体辐射功率落在 ``[low, high]`` 波段内的比例（数值积分）。"""
    if temperature_k <= 0.0:
        return 0.0
    wavelengths = np.linspace(low_m, high_m, 1200)
    radiance = np.array([spectral_irradiance(w, temperature_k) for w in wavelengths])
    whole = np.array([spectral_irradiance(w, temperature_k) for w in np.linspace(1e-9, 20e-6, 2400)])
    visible = float(np.trapezoid(radiance, wavelengths))
    total = float(np.trapezoid(whole, np.linspace(1e-9, 20e-6, 2400)))
    return visible / total if total > 0.0 else 0.0


def solar_visible_fraction(sun_temperature_k: float = 5772.0) -> float:
    """太阳辐射中可见光占比（~0.37）。"""
    return band_fraction(sun_temperature_k, VISIBLE_LOW_M, VISIBLE_HIGH_M)


def photon_energy_ev(wavelength_m: float) -> float:
    """单个光子的能量（电子伏特，E = hc/λ）。"""
    return PLANCK_H * LIGHT_SPEED / wavelength_m / 1.602176634e-19
