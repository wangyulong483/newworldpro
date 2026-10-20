"""世界环境：太阳、空气与身体热平衡。

世界拥有的是物理**场**，而不是画上去的装饰：

* **太阳**：悬在世界上方的一个 ~5772 K 黑体辐射源，同时提供可见光
  （给眼睛用，见 :mod:`newworld.electromagnetism`）与辐射热（本模块）；
* **空气**：有一定温度与气压的介质，传声（:mod:`newworld.acoustics`）
  并施加空气阻力；
* **集总身体热平衡**：代谢产热、太阳吸收、辐射交换（斯特藩-玻尔兹曼）、
  对流，以及把核心温度拉回 37 C 的简化生理调节项。这是物理-体温调节
  混合模型（真实的稳态涉及血管收缩与出汗）；物理项都取标准常数。

这里没有任何绘制：世界保持视觉纯白。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import electromagnetism as em
from . import physics

#: 太阳的有效表面温度（K）；维恩峰值 ~502 nm（可见光绿段）
SUN_SURFACE_TEMPERATURE_K = 5772.0


@dataclass
class Sun:
    """世界上方的黑体太阳。"""

    elevation_deg: float = 55.0
    azimuth_deg: float = 20.0
    irradiance: float = physics.SOLAR_CONSTANT  # 垂直入射辐照度 W/m^2
    temperature_k: float = SUN_SURFACE_TEMPERATURE_K

    @property
    def sin_elevation(self) -> float:
        """高度角正弦（夜间按 0 处理）。"""
        return max(0.0, math.sin(math.radians(self.elevation_deg)))

    @property
    def horizontal_irradiance(self) -> float:
        """水平面上的辐照度（``E = S * sin(高度角)``）。"""
        return self.irradiance * self.sin_elevation

    @property
    def visible_irradiance(self) -> float:
        """辐照度中的可见光部分（W/m^2）。"""
        return self.horizontal_irradiance * em.solar_visible_fraction(self.temperature_k)

    @property
    def peak_wavelength_nm(self) -> float:
        """黑体辐射的维恩峰值波长（nm）。"""
        return em.wien_peak_wavelength_m(self.temperature_k) * 1e9

    def light_direction(self) -> tuple[float, float]:
        """从太阳指向世界的单位向量（屏幕坐标，y 轴向下）。"""
        elevation = math.radians(self.elevation_deg)
        azimuth = math.radians(self.azimuth_deg)
        # 2-D 投影：太阳在上方偏一侧
        horizontal = math.sin(azimuth) * math.cos(elevation)
        vertical = -math.sin(elevation)
        length = math.hypot(horizontal, vertical) or 1.0
        return horizontal / length, vertical / length


@dataclass
class Air:
    """世界周围的介质（参与传声与阻力）。"""

    temperature_k: float = physics.AIR_TEMPERATURE_K
    density_kg_m3: float = physics.AIR_DENSITY
    pressure_pa: float = physics.AIR_PRESSURE_PA

    @property
    def temperature_c(self) -> float:
        """气温（摄氏度）。"""
        return self.temperature_k - 273.15

    @property
    def speed_of_sound_m_s(self) -> float:
        """当前温度下的声速（m/s，理想气体公式）。"""
        return physics.sound_speed(self.temperature_k)

    @property
    def speed_of_sound_px_s(self) -> float:
        """声速（像素/秒）。"""
        return self.speed_of_sound_m_s * physics.PIXELS_PER_METER


@dataclass
class HeatBalance:
    """集总身体热平衡方程的各项（单位全是瓦特）。"""

    metabolic_w: float
    solar_w: float
    radiative_loss_w: float
    convective_loss_w: float
    regulation_w: float

    @property
    def net_w(self) -> float:
        """净热流（正 = 身体升温）。"""
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
    """集总身体模型的热平衡。

    * 代谢：``Q_met = 100 W * activity``
    * 太阳吸收：``Q_sun = E * 吸收率 * 受照面积``
    * 辐射：``sigma * epsilon * A * (T_body^4 - T_env^4)``
    * 对流：``h * A * (T_body - T_air)``
    * 调节：``-K * (T_body - 37 C)``（简化的体温稳态）
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
    """按一个物理步长积分身体温度（写入 character.body_temperature_k）。"""
    balance = body_heat_balance(
        character.body_temperature_k,
        air.temperature_k,
        sun.horizontal_irradiance,
        activity=activity,
    )
    heat_capacity = physics.BODY_MASS_KG * physics.BODY_SPECIFIC_HEAT
    # 夹在 200..450 K，避免数值发散
    character.body_temperature_k = min(
        450.0,
        max(200.0, character.body_temperature_k + balance.net_w / heat_capacity * dt),
    )
    character.heat_balance = balance
    return balance
