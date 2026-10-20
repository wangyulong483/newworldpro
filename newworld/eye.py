"""newworld 居民的类人眼睛光学。

模型遵循视觉光学与眼科学中标准的示意眼/相机镜头处理（角膜 +
晶状体 → 等效薄透镜，瞳孔 → 光圈，视网膜 → 像平面）：

* 角膜提供大部分聚焦力（约 70%），晶状体提供其余（约 30%）；
  晶状体可以变形以对近处物体调焦
  （https://www.aao.org/eye-health/anatomy/parts-of-eye）；
* 虹膜/瞳孔控制到达视网膜的光量，亮处直径约 2 mm，暗处可到 8 mm；
* 视网膜上的像受**失焦**（眼睛对焦的距离与物体距离不一致）和
  **衍射**（瞳孔是有限孔径：艾里斑，第一暗环在
  ``1.22 * lambda / D``）限制；
* 视锐度在中央凹最高、随离心角下降（视网膜一侧的实现见
  :mod:`newworld.retina`）。

所有角度量都以度为单位；``pupil_mm`` 是瞳孔直径。
"""

from __future__ import annotations

import math

import numpy as np

from .optics import Panorama

#: 衍射极限用的波长（昼视峰值，nm → m）
WAVELENGTH_M = 550e-9

#: 艾里斑半径（第一暗环）为 1.22 * lambda / D；衍射 PSF 的高斯
#: 近似标准差是这个系数
_AIRY_TO_SIGMA = 0.42


class EyeParameters:
    """一只眼的解剖 / 光学参数（人类数值）。"""

    __slots__ = (
        "field_of_view_deg",
        "max_rotation_deg",
        "focal_length_mm",
        "axial_length_mm",
        "pupil_min_mm",
        "pupil_max_mm",
        "resting_accommodation_dpt",
        "accommodation_range_dpt",
        "accommodation_tau_s",
        "world_scale_m_per_px",
        "panorama_samples_per_deg",
    )

    def __init__(self) -> None:
        """按人类解剖学数值初始化全部参数。"""
        #: 人眼的水平视场（单眼约 160 度）
        self.field_of_view_deg = 160.0
        #: 眼球在头内最多能旋转约 50 度
        self.max_rotation_deg = 50.0
        #: 人眼的有效焦距（约 17 mm）
        self.focal_length_mm = 17.0
        #: 透镜到视网膜的距离（约 24 mm）
        self.axial_length_mm = 24.0
        #: 亮光/黑暗中的瞳孔直径
        self.pupil_min_mm = 2.0
        self.pupil_max_mm = 8.0
        #: 没有对焦目标时的静息调节（约 1 屈光度）
        self.resting_accommodation_dpt = 1.0
        #: 年轻成年人的调节范围（屈光度）
        self.accommodation_range_dpt = 8.0
        #: 调节反射的时间常数
        self.accommodation_tau_s = 0.35
        #: 多少世界像素 = 1 米（把命中距离换算成屈光度用）。
        #: 1 px = 1 cm，居民身高约 1.4 m。
        self.world_scale_m_per_px = 0.01
        #: 光场的角分辨率（每度采样数）
        self.panorama_samples_per_deg = 120.0


# ---------------------------------------------------------------------------
# 瞳孔 + 模糊
# ---------------------------------------------------------------------------
def pupil_diameter_mm(mean_luminance: float, params: EyeParameters) -> float:
    """瞳孔直径随视网膜照度的变化（对数律）。

    亮白（亮度 1.0）→ 2 mm；黑暗（1e-3）→ 8 mm。映射是两个极端
    之间平滑的对数插值。
    """
    luminance = min(max(float(mean_luminance), 1e-4), 1.0)
    fraction = (math.log10(luminance) + 3.0) / 3.0  # 0（暗）.. 1（亮）
    fraction = min(1.0, max(0.0, fraction))
    return params.pupil_max_mm - (params.pupil_max_mm - params.pupil_min_mm) * fraction


def circle_of_confusion_deg(
    pupil_mm: float, defocus_dpt: float
) -> float:
    """失焦模糊盘的角直径（度）。

    薄透镜下角模糊近似为 ``瞳孔直径 × 失焦屈光度``（弧度）。
    """
    return math.degrees(pupil_mm * 1e-3 * abs(defocus_dpt))


def diffraction_limit_arcmin(pupil_mm: float) -> float:
    """艾里斑第一暗环的角直径（角分）。"""
    theta = 1.22 * WAVELENGTH_M / (pupil_mm * 1e-3)  # 弧度
    return math.degrees(theta) * 60.0


def blur_sigma_deg(pupil_mm: float, defocus_dpt: float) -> float:
    """衍射 + 失焦合并后的高斯等效模糊 sigma（度）。

    失焦 PSF 是角直径为 ``c`` 的圆盘，方差 ``c^2/16``（即
    sigma = c/4 的高斯）；衍射 PSF 近似为
    ``sigma = 0.42 * lambda / D`` 的高斯。方差相加。
    """
    coc = pupil_mm * 1e-3 * abs(defocus_dpt)          # 弧度
    sigma_defocus = coc / 4.0
    sigma_diffraction = _AIRY_TO_SIGMA * WAVELENGTH_M / (pupil_mm * 1e-3)
    return math.degrees(math.hypot(sigma_defocus, sigma_diffraction))


#: 人眼近点（米）：再近的物体无法对焦，失焦量在近点处截断。
#: 若不截断，嵌入几何体/贴脸物体（距离→0）会产生上万屈光度、数千度模糊的
#: 病理帧（FFT 把数组 padding 到数十万点，单帧 100ms+）。
NEAR_POINT_M = 0.1
#: 失焦量上限（屈光度）：人眼调节范围之外的值没有生理意义。
MAX_DEFOCUS_DPT = 20.0


def defocus_diopters(distance_px: float, accommodation_dpt: float, params: EyeParameters) -> float:
    """物体在 ``distance_px``、当前调节为 ``accommodation_dpt`` 时的失焦量（屈光度）。

    距离按人眼近点（0.1 m）截断、失焦量按 20 D 截断：这是真实的生理约束，
    同时避免贴脸几何体触发病理级模糊（见 :data:`NEAR_POINT_M`）。
    """
    if not math.isfinite(distance_px) or distance_px <= 0.0:
        return min(abs(accommodation_dpt), MAX_DEFOCUS_DPT)  # 看向无穷远
    distance_m = max(distance_px * params.world_scale_m_per_px, NEAR_POINT_M)
    vergence_dpt = 1.0 / distance_m
    return min(abs(accommodation_dpt - vergence_dpt), MAX_DEFOCUS_DPT)


def accommodation_for_distance(distance_px: float, params: EyeParameters) -> float:
    """把 ``distance_px`` 处的物体对焦所需的调节（屈光度）。"""
    if not math.isfinite(distance_px) or distance_px <= 0.0:
        return 0.0
    distance_m = distance_px * params.world_scale_m_per_px
    return min(params.accommodation_range_dpt, 1.0 / max(distance_m, 1e-6))


# ---------------------------------------------------------------------------
# 角度模糊
# ---------------------------------------------------------------------------
def angular_gaussian_blur(panorama: Panorama, sigma_deg: float) -> Panorama:
    """用 1-D 高斯模糊全景（边界用边缘填充）。

    sigma 小时直接卷积；sigma 很大（大扫视/大失焦）时切换到
    FFT 卷积（O(n log n)）。
    """
    if sigma_deg <= 1e-4 or len(panorama) < 3:
        return panorama
    step = float(panorama.rel_angles_deg[1] - panorama.rel_angles_deg[0])
    sigma_samples = max(1e-3, sigma_deg / abs(step))
    half_width = max(1, int(round(4.0 * sigma_samples)))
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    kernel /= kernel.sum()

    radiance = panorama.radiance.astype(np.float64)
    padded = np.pad(radiance, ((half_width, half_width), (0, 0)), mode="edge")
    if half_width > 24:
        # 大扫视/失焦模糊：FFT 卷积（O(n log n)）
        length = padded.shape[0] + kernel.size - 1
        fft_length = 1
        while fft_length < length:
            fft_length *= 2
        kernel_padded = np.zeros(fft_length, dtype=np.float64)
        kernel_padded[: kernel.size] = kernel
        spectrum = np.fft.rfft(padded, fft_length, axis=0) * np.fft.rfft(
            kernel_padded, fft_length
        )[:, None]
        convolved = np.fft.irfft(spectrum, fft_length, axis=0)
        filtered = convolved[2 * half_width : 2 * half_width + len(panorama)].astype(np.float32)
    else:
        filtered = np.empty_like(radiance, dtype=np.float32)
        for channel in range(radiance.shape[1]):
            filtered[:, channel] = np.convolve(
                padded[:, channel], kernel, mode="valid"
            ).astype(np.float32)
    return Panorama(panorama.rel_angles_deg, filtered, panorama.distance_px)


# ---------------------------------------------------------------------------
# 眼睛状态
# ---------------------------------------------------------------------------
class EyeState:
    """眼睛的慢状态：调节与瞳孔。"""

    __slots__ = ("accommodation_dpt", "pupil_mm", "mean_luminance")

    def __init__(self, params: EyeParameters) -> None:
        """初始化静息调节与瞳孔。"""
        self.accommodation_dpt = params.resting_accommodation_dpt
        self.pupil_mm = params.pupil_min_mm
        self.mean_luminance = 1.0

    def update(
        self,
        dt: float,
        fixation_distance_px: float,
        mean_luminance: float,
        params: EyeParameters,
    ) -> float:
        """反射式更新（调节趋向目标、瞳孔随亮度）；返回失焦屈光度。"""
        target = accommodation_for_distance(fixation_distance_px, params)
        alpha = min(1.0, dt / max(1e-3, params.accommodation_tau_s))
        self.accommodation_dpt += (target - self.accommodation_dpt) * alpha
        self.pupil_mm = pupil_diameter_mm(mean_luminance, params)
        self.mean_luminance = float(mean_luminance)
        return defocus_diopters(fixation_distance_px, self.accommodation_dpt, params)
