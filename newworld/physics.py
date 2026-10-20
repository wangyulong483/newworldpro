"""newworld 的物理常数与单位系统（全世界的度量衡）。

世界坐标用屏幕像素。沿用视觉系统的比例 **1 像素 = 1 cm**
（``PIXELS_PER_METER = 100``），因此 2560×1600 的屏幕是一块
25.6 m × 16 m 的世界，1.38 m 高的居民在物理上是合理的。

常数取自权威资料的 SI 数值：

* 标准重力 ``g = 9.806 65 m/s^2``（精确值，NIST/CODATA）
  https://physics.nist.gov/cgi-bin/cuu/Value?gn
* 斯特藩-玻尔兹曼常数 ``sigma = 5.670 374 419...e-8 W m^-2 K^-4``
  （精确值，NIST/CODATA）https://physics.nist.gov/cgi-bin/cuu/Value?sigma
* 20 C 干燥空气中的声速：``343.1 m/s``
  https://www.engineeringtoolbox.com/speed-sound-d_82.html
* 太阳常数（1 AU 处总太阳辐照度）：``~1361 W/m^2``
  （NASA/TSI；见 :mod:`newworld.electromagnetism`）
* 人体体温 ~310.15 K（37 C）、听觉范围 20 Hz - 20 kHz、
  参考声压 20 µPa（听阈）
  https://www.nidcd.nih.gov/health/how-do-we-hear
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# 单位换算：米 ↔ 像素
# ---------------------------------------------------------------------------
PIXELS_PER_METER = 100.0


def m_to_px(metres: float) -> float:
    """米 → 像素。"""
    return metres * PIXELS_PER_METER


def px_to_m(pixels: float) -> float:
    """像素 → 米。"""
    return pixels / PIXELS_PER_METER


# ---------------------------------------------------------------------------
# 力学
# ---------------------------------------------------------------------------
#: 标准重力加速度（NIST CODATA，精确值）
STANDARD_GRAVITY = 9.80665  # m/s^2
GRAVITY_PX = STANDARD_GRAVITY * PIXELS_PER_METER  # px/s^2

#: 1 AU 处总太阳辐照度（~1361 W/m^2，NASA/TSI）
SOLAR_CONSTANT = 1361.0  # W/m^2

#: 居民的跳跃初速度（有劲，但仍是一条弹道抛物线）
JUMP_SPEED = 5.0  # m/s
JUMP_SPEED_PX = JUMP_SPEED * PIXELS_PER_METER

#: 空气阻力（平方阻力）：F = 0.5 * rho * Cd * A * v^2
DRAG_COEFFICIENT = 1.0
BODY_FRONTAL_AREA_M2 = 0.45


def air_drag_force(velocity_m_s: float, area_m2: float = BODY_FRONTAL_AREA_M2) -> float:
    """平方阻力模型的阻力（牛顿），方向与速度相反（这里返回带符号大小）。"""
    return 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * area_m2 * velocity_m_s * abs(velocity_m_s)


# ---------------------------------------------------------------------------
# 空气 / 声学
# ---------------------------------------------------------------------------
#: 海平面 20 C 的干燥空气
AIR_DENSITY = 1.225  # kg/m^3
AIR_GAS_CONSTANT = 287.05  # J/(kg K)
AIR_HEAT_CAPACITY_RATIO = 1.4
AIR_TEMPERATURE_K = 293.15  # 20 C
AIR_PRESSURE_PA = 101325.0

#: 20 C 干燥空气中的声速（Engineering ToolBox）
SPEED_OF_SOUND_20C = 343.1  # m/s
SPEED_OF_SOUND_PX = SPEED_OF_SOUND_20C * PIXELS_PER_METER  # px/s

#: 人耳听觉
HEARING_MIN_HZ = 20.0
HEARING_MAX_HZ = 20000.0
REFERENCE_SOUND_PRESSURE_PA = 20e-6  # 0 dB SPL（听阈）


def sound_speed(temperature_k: float = AIR_TEMPERATURE_K) -> float:
    """理想气体声速 ``c = sqrt(gamma R T)``（m/s）。"""
    return math.sqrt(AIR_HEAT_CAPACITY_RATIO * AIR_GAS_CONSTANT * temperature_k)


def sound_speed_px(temperature_k: float = AIR_TEMPERATURE_K) -> float:
    """声速的像素单位版本（px/s）。"""
    return sound_speed(temperature_k) * PIXELS_PER_METER


def spl_db(pressure_rms_pa: float) -> float:
    """声压级 dB（参考 20 µPa）；非正值按 0 dB 处理。"""
    if pressure_rms_pa <= 0.0:
        return 0.0
    return 20.0 * math.log10(pressure_rms_pa / REFERENCE_SOUND_PRESSURE_PA)


# ---------------------------------------------------------------------------
# 热力学（人体热平衡用）
# ---------------------------------------------------------------------------
#: 斯特藩-玻尔兹曼常数（NIST CODATA，精确值）
STEFAN_BOLTZMANN = 5.670374419e-8  # W m^-2 K^-4

HUMAN_BODY_TEMPERATURE_K = 310.15  # 37 C
BODY_MASS_KG = 35.0
BODY_SURFACE_AREA_M2 = 1.2
BODY_SPECIFIC_HEAT = 3500.0  # J/(kg K)
METABOLIC_RATE_W = 100.0  # 静息产热
SKIN_EMISSIVITY = 0.98
SKIN_SOLAR_ABSORPTIVITY = 0.50
CONVECTIVE_COEFFICIENT = 5.0  # W/(m^2 K)，自然对流
SOLAR_ABSORBING_AREA_M2 = 0.20
#: 简化的生理体温调节（血管收缩/出汗）：一个把核心温度拉回 37 C 的比例项
REGULATION_GAIN_W_PER_K = 120.0


def jump_height_m(speed_m_s: float = JUMP_SPEED) -> float:
    """弹道跳跃的最大高度 ``v^2 / (2 g)``（米，忽略阻力）。"""
    return speed_m_s * speed_m_s / (2.0 * STANDARD_GRAVITY)
