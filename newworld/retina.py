"""newworld 居民的类人视网膜。

按人眼视网膜的蓝图实现：

* 人眼约有 1.2 亿视杆细胞和 600 万视锥细胞；视杆集中在外围、
  对光敏感 500-1000 倍；视锥集中在中心、承担色觉
  （https://www.aao.org/eye-health/anatomy/rods，
   https://www.aao.org/eye-health/anatomy/cones）；
* 三种视锥：约 60% L（红）、30% M（绿）、10% S（蓝）
  （https://www.aao.org/eye-health/anatomy/cones）；
* 中央凹（黄斑中心）视觉最锐利，向外围快速下降
  （https://www.aao.org/eye-health/anatomy/fovea，
   https://www.aao.org/eye-health/anatomy/peripheral-retina）；
* 感光细胞随平均光强自适应，所以同一个物体在暗/亮适应后看起来
  不一样。

由于 newworld 是 2-D 侧视世界，光场只有一个角度维度，因此这里的
视网膜是沿离心角的一维镶嵌。完整的人类机制（视杆/视锥镶嵌、
中央凹锐度衰减、L/M/S 色彩采样、昼/夜视适应、中心-外周神经节
细胞）都在这一根轴上实现；将来世界在视觉上变成 2-D/3-D 时，
同一个类可以自然扩展。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .optics import Panorama

#: 视锥类型编码
L_CONE = 0
M_CONE = 1
S_CONE = 2

#: sRGB（线性）→ LMS（Hunt-Pointer-Estevez，D65 归一化；每行和为 1）
_RGB_TO_LMS = np.array(
    (
        (0.38971, 0.68898, -0.07868),
        (-0.22981, 1.18340, 0.04641),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)

#: 昼视亮度权重
_LUMINANCE = np.array((0.2126, 0.7152, 0.0722), dtype=np.float64)


@dataclass
class RetinaParameters:
    """镶嵌规模与适应常数（从约 1.26 亿细胞缩放到可计算规模）。"""

    #: 每套镶嵌的感光细胞总数（从 ~1.26 亿缩小！）
    n_cones: int = 1800
    n_rods: int = 3000
    #: 视锥 L/M/S 比例（AAO：60/30/10）
    cone_proportions: tuple[float, float, float] = (0.60, 0.30, 0.10)
    #: 中央凹（度）：此处无视杆，中央小凹抑制 S 锥
    fovea_radius_deg: float = 1.5
    foveola_radius_deg: float = 0.5
    #: 锐度衰减的离心角常数 E2（度）
    eccentricity_constant_deg: float = 2.5
    #: 视杆密度在约 18 度离心角处达峰
    rod_peak_deg: float = 18.0
    rod_width_deg: float = 9.0
    #: 昼视 / 夜视适应时间常数
    cone_adaptation_tau_s: float = 0.20
    rod_adaptation_tau_s: float = 8.0
    #: 感光响应的时间滤波
    response_tau_s: float = 0.05
    #: 视网膜拓扑特征向量的分箱数
    n_bins: int = 32


@dataclass
class RetinaObservation:
    """离开视网膜、送往大脑（以及 RL 策略）的东西。"""

    eccentricity_deg: np.ndarray       # (n_cones,) 视锥离心角
    cone_types: np.ndarray             # (n_cones,) int，L/M/S
    cone_responses: np.ndarray         # (n_cones, 3) L/M/S 响应
    rod_eccentricity_deg: np.ndarray   # (n_rods,) 视杆离心角
    rod_responses: np.ndarray          # (n_rods,) 视杆响应
    ganglion: np.ndarray               # (n_bins,) 中心-外周输出
    mean_luminance: float
    foveal_luminance: float
    adaptation: float

    def feature_vector(self) -> np.ndarray:
        """紧凑、定长的视网膜拓扑特征向量（供学习用）。"""
        bins = self.ganglion.size
        span = 160.0
        # 视锥响应按离心角分箱成 3 通道直方图
        cone_hist = np.zeros((bins, 3), dtype=np.float32)
        if bins and self.eccentricity_deg.size:
            indices = np.clip(
                ((self.eccentricity_deg + span / 2.0) / span * (bins - 1)).astype(int),
                0,
                bins - 1,
            )
            for channel in range(3):
                np.add.at(cone_hist[:, channel], indices, self.cone_responses[:, channel])
        # 视杆响应同样分箱
        rod_hist = np.zeros(bins, dtype=np.float32)
        if bins and self.rod_responses.size:
            indices = np.clip(
                ((self.rod_eccentricity_deg + span / 2.0) / span * (bins - 1)).astype(int),
                0,
                bins - 1,
            )
            np.add.at(rod_hist, indices, self.rod_responses)
        scalars = np.array(
            (
                self.mean_luminance,
                self.foveal_luminance,
                self.adaptation,
            ),
            dtype=np.float32,
        )
        return np.concatenate(
            (cone_hist.ravel(), rod_hist, self.ganglion, scalars)
        ).astype(np.float32)


class Retina:
    """感光细胞镶嵌 + 光适应 + 视网膜神经节细胞。"""

    def __init__(self, params: RetinaParameters, rng: np.random.Generator) -> None:
        """初始化增益并构建镶嵌。"""
        self.params = params
        self.rng = rng
        self.cone_gain = np.ones(3, dtype=np.float64)
        self.rod_gain = 1.0
        self._build_mosaic()

    # ------------------------------------------------------------------
    # 镶嵌
    # ------------------------------------------------------------------
    def _build_mosaic(self) -> None:
        """按人类密度分布放置视锥/视杆并分配 L/M/S 类型。"""
        params = self.params
        n_cones = params.n_cones
        fov = 160.0  # 与 EyeParameters.field_of_view_deg 一致
        # 视锥密度：中央密（1/(1+ecc/E2)），外围稀
        self.cone_ecc = self._sample_angles(
            lambda ecc: 1.0 / (1.0 + ecc / params.eccentricity_constant_deg),
            n_cones,
            fov,
        )
        # 视杆密度：中央凹没有，18 度处高斯达峰
        rod_density = lambda ecc: np.where(  # noqa: E731
            ecc < params.fovea_radius_deg,
            0.0,
            np.exp(-0.5 * ((ecc - params.rod_peak_deg) / params.rod_width_deg) ** 2),
        )
        self.rod_ecc = self._sample_angles(rod_density, params.n_rods, fov)

        # 按 60/30/10 比例随机分配视锥类型
        l_prob, m_prob, _s_prob = params.cone_proportions
        total = l_prob + m_prob + params.cone_proportions[2]
        l_threshold = l_prob / total
        m_threshold = l_threshold + m_prob / total
        u = self.rng.random(n_cones)
        types = np.where(u < l_threshold, L_CONE, np.where(u < m_threshold, M_CONE, S_CONE))
        # 中央小凹（几乎）没有 S 锥——著名的人类特性
        central = np.abs(self.cone_ecc) < params.foveola_radius_deg
        types[central & (types == S_CONE)] = M_CONE
        self.cone_types = types.astype(np.int8)

    @staticmethod
    def _sample_angles(density, n: int, fov: float) -> np.ndarray:
        """按密度函数分层采样 n 个离心角（确定性 + 微抖动防混叠）。"""
        grid = np.linspace(-fov / 2.0, fov / 2.0, 4001)
        weights = density(np.abs(grid)).astype(np.float64)
        weights[weights < 0] = 0.0
        if weights.sum() <= 0.0:
            return np.zeros(0, dtype=np.float64)
        cdf = np.cumsum(weights)
        cdf -= cdf[0]
        cdf /= cdf[-1]
        u = (np.arange(n) + 0.5) / n
        # 分层采样 → 确定且平滑；微抖动避免镶嵌出现完美规则的混叠
        jitter = (np.random.default_rng(12345).random(n) - 0.5) / n
        u = np.clip(u + 0.8 * jitter, 0.0, 1.0 - 1e-9)
        u.sort()
        return np.interp(u, cdf, grid)

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------
    def sample(
        self,
        panorama: Panorama,
        dt: float,
        *,
        lid_transmission: float = 1.0,
        neural_gain: float = 1.0,
    ) -> RetinaObservation:
        """把光学全景变成感光细胞 + 神经节细胞响应。"""
        params = self.params
        rel = panorama.rel_angles_deg
        radiance = panorama.radiance.astype(np.float64) * lid_transmission

        # 在每个视锥位置插值出 RGB，再变换到 LMS
        cone_rgb = np.stack(
            [np.interp(self.cone_ecc, rel, radiance[:, channel]) for channel in range(3)],
            axis=1,
        )
        lms = cone_rgb @ _RGB_TO_LMS.T
        lms = np.clip(lms, 0.0, None)

        panorama_luminance = radiance @ _LUMINANCE
        foveal = float(np.interp(0.0, rel, panorama_luminance))
        mean_lum = float(np.mean(panorama_luminance)) if panorama_luminance.size else 0.0

        # -- 适应：增益跟随平均光强（韦伯式） --------------------------
        if dt > 0.0:
            alpha_cone = 1.0 - math.exp(-dt / max(1e-3, params.cone_adaptation_tau_s))
            target_gain = 1.0 / (lms.mean(axis=0) + 0.05)
            self.cone_gain += (target_gain - self.cone_gain) * alpha_cone

            alpha_rod = 1.0 - math.exp(-dt / max(1e-3, params.rod_adaptation_tau_s))
            target_rod = 1.0 / (mean_lum + 0.02)
            self.rod_gain += (target_rod - self.rod_gain) * alpha_rod

        # -- 感光细胞响应（Naka-Rushton 软饱和） ----------------------
        activated = np.clip(lms * self.cone_gain, 0.0, None)
        cone_responses = activated / (1.0 + activated) * neural_gain

        # 视杆只看亮度；白天饱和（photopic 抑制）
        rod_rgb = np.stack(
            [np.interp(self.rod_ecc, rel, radiance[:, channel]) for channel in range(3)],
            axis=1,
        )
        rod_lum = np.clip(rod_rgb @ _LUMINANCE, 0.0, None)
        rod_activated = np.clip(rod_lum * self.rod_gain, 0.0, None)
        photopic = 1.0 / (1.0 + (mean_lum / 0.05) ** 2)  # 视杆在日光下饱和
        rod_responses = rod_activated / (1.0 + rod_activated) * photopic * neural_gain

        cone_luminance = cone_rgb @ _LUMINANCE
        ganglion = self._ganglion(cone_luminance, neural_gain)
        return RetinaObservation(
            eccentricity_deg=self.cone_ecc.copy(),
            cone_types=self.cone_types.copy(),
            cone_responses=cone_responses.astype(np.float32),
            rod_eccentricity_deg=self.rod_ecc.copy(),
            rod_responses=rod_responses.astype(np.float32),
            ganglion=ganglion,
            mean_luminance=mean_lum,
            foveal_luminance=foveal,
            adaptation=float(np.mean(self.cone_gain)),
        )

    # ------------------------------------------------------------------
    # 视网膜神经节细胞
    # ------------------------------------------------------------------
    def _ganglion(self, luminance: np.ndarray, neural_gain: float) -> np.ndarray:
        """规则网格上的中心-外周（高斯差分）输出。

        均匀光场（比如现在的全白世界）输出为零：神经节细胞报告
        的是**对比度**，不是绝对光强。
        """
        params = self.params
        bins = max(2, params.n_bins)
        if neural_gain <= 0.0:
            return np.zeros(bins, dtype=np.float32)
        profile = self.ganglion_profile(luminance, neural_gain=neural_gain)
        edges = np.linspace(0, profile.size, bins + 1).astype(int)
        return np.array(
            [profile[start:end].mean() for start, end in zip(edges[:-1], edges[1:])],
            dtype=np.float32,
        )

    def ganglion_profile(
        self,
        luminance: np.ndarray,
        *,
        neural_gain: float = 1.0,
        n_samples: int = 2048,
    ) -> np.ndarray:
        """细粒度的中心-外周响应（显示 / 分析用）。"""
        if neural_gain <= 0.0:
            return np.zeros(n_samples, dtype=np.float32)
        # 把不规则镶嵌重采样到细规则网格，小小的中心感受野才解析得开
        fine = np.linspace(-80.0, 80.0, n_samples)
        order = np.argsort(self.cone_ecc)
        sampled = np.interp(fine, self.cone_ecc[order], luminance[order])
        spacing = fine[1] - fine[0]
        centre = _gaussian_filter_1d(sampled, 0.10, spacing=spacing)
        surround = _gaussian_filter_1d(sampled, 0.35, spacing=spacing)
        return ((centre - surround) * neural_gain).astype(np.float32)


def _gaussian_filter_1d(values: np.ndarray, sigma: float, spacing: float) -> np.ndarray:
    """一维高斯滤波（边缘填充）。"""
    sigma_samples = max(1e-3, sigma / max(1e-9, spacing))
    radius = max(1, int(round(4.0 * sigma_samples)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values.astype(np.float64), (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")
