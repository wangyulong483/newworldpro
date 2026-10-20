"""newworld 居民的紧凑视觉特征：真实视网膜 → 决策观测。

把 :class:`~newworld.perception.VisualSystem` 产出的
:class:`~newworld.perception.PerceptionState`（眼动 → 全景渲染 → 光学模糊
→ 视网膜镶嵌 → 视网膜神经节细胞）压缩成 16 维定长特征，供
``neural.features()`` / RL 策略消费。

**这不是几何启发式**：所有数值都来自上面那条真实光学/视网膜管线
（:meth:`VisualSystem.perceive` + :class:`~newworld.perception.Audition`，
即 :class:`~newworld.perception.Perceiver` 组装的那两个真实感官部件），
本模块只做定标与降采样。

16 维组成（``VISUAL_FEATURE_DIM = 16``）
----------------------------------------
===  ==============================================================
维度  含义
===  ==============================================================
0-7  视网膜神经节细胞（ganglion）响应分箱降采样：把
     ``RetinaObservation.ganglion``（``n_bins`` 个中心-外周 DoG 响应；
     与 ``RetinaObservation.feature_vector()`` 里的 ganglion 段是**同一
     个数组**，这里直接取用避免脆弱的偏移切片）按均值聚成 8 组
     （``np.array_split`` → ``mean``），乘 :data:`GANGLION_GAIN` 定标后
     ``tanh`` 压缩；
8    平均亮度 ``PerceptionState.mean_luminance``，log10 压缩；
9    中央凹亮度 ``PerceptionState.foveal_luminance``，log10 压缩；
10   光适应状态 ``RetinaObservation.adaptation``（视锥平均增益），log 压缩；
11   注视偏移角 / 眼最大旋转角（``gaze_offset_deg / max_rotation_deg``）；
12   离焦模糊 σ ``PerceptionState.blur_sigma_deg``（失焦 + 衍射 + 扫视拖影）；
13   扫视抑制强度（替代量，见下）；
14   声压级 SPL ``sound_spl_db / 120``；
15   是否听到（0/1，``hearing``）。
===  ==============================================================

``PerceptionState`` **没有**显式"扫视抑制强度"字段：管线里扫视抑制是以
``retina.sample(neural_gain=0.25)`` 的常数增益施加的，无法反推强度。本模块
因此用两个真实可得的量替代（见 :func:`_saccade_suppression`）：眨眼时取
1.0（眼睑把光完全挡掉），否则取 ``|saccade_velocity_deg_s| / 600``
（扫视速度越快、抑制越强，600 deg/s 饱和）。这条替代关系是本模块唯一
"非直接读数"的维度，其余 15 维都是感知管线里的真实字段。

病理帧保护（近点下限）
----------------------
:class:`~newworld.perception.VisualSystem` 用"中央凹方向的光线命中距离"驱动
调节/失焦。当眼睛嵌进几何体（或光线擦着表面）时命中距离趋近 0，失焦屈光
度会爆到约 3 万 D、模糊 σ 到约 3400°，``angular_gaussian_blur`` 的 FFT 分支
把数组 padding 到 30 多万点，**单帧 130-170 ms**（实测）。本模块在把真实
场景交给 ``VisualSystem`` 之前包一层 :class:`_NearPointScene`，只把"有限且
小于人眼近点（0.1 m = 10 px）"的命中距离抬到近点。像素/距离仍来自真实
光线追踪，正常帧零拷贝直通；实测 3 个 seed × 200 帧最大模糊 σ 5.3°、
最慢单帧 4.7 ms（不再出现百毫秒级病理帧）。眼/视网膜本身没有为此改动
（不在本模块写权限内），这个病理成本同样存在于 GUI 高保真档。

学习用低分辨率档（默认参数）
----------------------------
默认是**学习 / RL 用的低分辨率档**，不是运行 GUI 的高保真档：

* 视网膜：``n_cones=96, n_rods=0, n_bins=16``（高保真档 1800/3000/32）；
* 全景角分辨率：12 samples/deg（高保真档 120，即 :class:`EyeParameters`
  的默认值）。160° × 12 = 1921 条光线，仍然比视网膜本身密 20 倍以上
  （96 视锥 / 160° ≈ 0.6 samples/deg）。

**降低全景分辨率不损失特征信息**：实测把全景从 12 提到 120 samples/deg
（视网膜保持 96 视锥档），16 维特征最大差异仅 0.015、平均 0.0004，而单次
``observe`` 从约 1.1-1.7 ms 涨到约 7.0-7.9 ms；特征的信息上限由视网膜镶嵌
（96 个视锥）决定，不是全景采样。高保真档随时可以注入：
``retina_parameters=RetinaParameters()`` + ``eye_parameters=EyeParameters()``
（实测约 9.7-11.8 ms/次）。

实测成本（``tools/bench_vision_features.py``，640×400 世界、100 次 observe、
每帧之间真实推进世界、seed=0；两次独立运行覆盖不同机器负载）：
默认档 **平均 1.08-1.75 ms、中位数 1.06-1.69 ms、p95 ≤ 2.11 ms、
最大 ≤ 2.61 ms**，两次运行都满足契约 §1.3 的 ≤ 2.5 ms 目标（连原始均值
也 ≤ 2.5 ms）；分阶段约：全景 0.68-0.69 ms、光学模糊 0.13 ms、视网膜采样
0.41-0.44 ms（其中神经节 DoG 0.14-0.15 ms）、听觉 0.007 ms、特征压缩
0.07 ms（其余是感知状态/Python 开销）。

**按 ``perception_stride`` 调用（推荐集成方式）**：``observe`` 是"一帧感知"，
不需要每仿真帧都跑。调用方可以每 ``stride`` 帧调用一次，并把 ``dt`` 传成
``stride × fixed_dt``（眼动/光适应按这个总时长积分，语义仍然正确）；这样
每仿真帧的摊薄成本 ≈ 1.1-1.75 ms / stride（stride=8 → 约 0.2 ms/帧）。

用法::

    from newworld.visual_features import VisualFeatureExtractor

    extractor = VisualFeatureExtractor(seed=0)
    state = extractor.observe(world, world.agent, world.config.fixed_dt)
    state.vector          # (16,) float64，全有限、约 [-1, 1]
    state.source          # "retina" | "cache" | "none"
    state.cost_ms         # 本次 observe 的真实耗时

异常 / 无场景时返回零向量并把 ``source`` 置为 ``"none"``，不抛错（感知
不应该让整条仿真链崩掉）；失败原因留在 :attr:`VisualFeatureExtractor.last_error`。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np

from .eye import EyeParameters
from .optics import Panorama
from .perception import Audition, PerceptionState, VisualSystem
from .retina import RetinaObservation, RetinaParameters

#: 视觉特征维度（冻结契约 V2 §1.2）
VISUAL_FEATURE_DIM = 16

#: 视网膜神经节细胞特征占用的维度数（8 组均值降采样）
GANGLION_FEATURES = 8

#: 默认低分辨率档的视网膜参数（学习用；不是 GUI 高保真档）
LEARNING_RETINA_CONES = 96
LEARNING_RETINA_RODS = 0
LEARNING_RETINA_BINS = 16

#: 默认低分辨率档的全景角分辨率（samples/deg；高保真档为 120）
#:
#: 160° 视场 × 12 = 1921 条光线，仍然比视网膜本身密 20 倍以上（默认档
#: 96 个视锥 / 160° ≈ 0.6 samples/deg），所以特征信息上限由视网膜镶嵌
#: 决定、不是全景采样；实测把全景提到 120 samples/deg 后 16 维特征差异
#: ≈ 0，而成本变成 6 倍（见 ``tools/bench_vision_features.py``）。
LEARNING_PANORAMA_SAMPLES_PER_DEG = 12.0

#: ganglion 箱值的定标增益。
#:
#: ``_ganglion`` 对每个箱取的是 2048 点 DoG 剖面的**均值**，而 DoG 响应
#: 集中在物体边缘的少数采样点上，所以箱值比剖面峰值小约两个数量级。
#: 实测（640×400 世界，seed 0/7/42，各 12 个位置/朝向，默认 96 视锥档）
#: 箱值量级 1e-4 ~ 1.3e-3；乘 500 后落在 tanh 的可用动态范围内。
#: 这只是一个固定定标常数（类似 ``SPL / 120``），不引入任何新信息。
GANGLION_GAIN = 500.0

#: 扫视抑制特征的分母（deg/s）：扫视角速度达到该值时抑制饱和为 1.0
SACCADE_SUPPRESSION_VELOCITY_DEG_S = 600.0

#: SPL 特征的定标分母（dB）
SPL_NORMALIZATION_DB = 120.0

#: 亮度 log 压缩的下限（线性 RGB 亮度 1e-3 对应特征 -1）
LUMINANCE_FLOOR = 1e-3

#: 人眼近点（米）：眼睛无法对焦到比这更近的物体（年轻成人约 0.1 m）。
#:
#: 用作 :class:`_NearPointScene` 的"对焦距离下限"：把它换算成像素后，
#: 交给 :class:`~newworld.perception.VisualSystem` 的光线命中距离不会小于
#: 该值。没有这个下限时会出现**病理帧**：眼睛嵌进几何体（或光线擦着表面
#: 穿过）→ 对焦距离趋近 0 → 失焦屈光度爆到约 3 万 D → 模糊 σ 约 3400° →
#: ``angular_gaussian_blur`` 走进 FFT 分支、padding 出 30 多万个采样点，
#: 单帧耗时 130-170 ms（实测见 ``tools/bench_vision_features.py`` 与
#: ``.tmp`` 探查记录）。这是真实生理约束（近点），不是几何启发式：
#: 光线命中距离仍然来自真实光学追踪。
NEAR_POINT_M = 0.1


@dataclass
class VisualFeatureState:
    """一次视觉特征提取的结果（契约 §1.2 冻结字段）。"""

    #: (16,) float64，全有限，约 [-1, 1]
    vector: np.ndarray
    #: "retina" = 本次真实采集；"cache" = 没有推进感知时间，返回上次结果；
    #: "none" = 降级零向量（世界/角色不可用或管线抛错）
    source: str
    #: 本次 :meth:`VisualFeatureExtractor.observe` 的真实耗时（毫秒）
    cost_ms: float = 0.0


# ---------------------------------------------------------------------------
# 近点保护
# ---------------------------------------------------------------------------
class _NearPointScene:
    """真实 :class:`~newworld.optics.Scene` 的薄代理：只把命中距离钳到人眼近点。

    :class:`~newworld.perception.VisualSystem` 用"中央凹方向的光线命中距离"
    驱动调节/失焦。当眼睛嵌进几何体（或光线擦着表面）时该距离趋近 0，
    失焦屈光度会爆到数万 D、模糊 σ 到数千度，``angular_gaussian_blur`` 的
    FFT 分支会把数组 padding 到几十万点，单帧 130-170 ms（实测）。

    本代理在调用真实 ``scene.panorama(...)`` 之后，只把**有限且小于近点的
    命中距离**抬到人眼近点（:data:`NEAR_POINT_M` 换算成像素）。像素/距离
    仍然来自真实光线追踪，不读物体列表、不做几何近似；正常帧原样返回
    （零拷贝，只在病理帧才重建 Panorama）。
    """

    __slots__ = ("_scene", "_panorama", "_floor_px")

    def __init__(self, scene, floor_px: float) -> None:
        """包一层真实场景，记录对焦距离下限（像素）。"""
        self._scene = scene
        #: 绑定的真实 panorama 方法（省一次属性查找）
        self._panorama = scene.panorama
        self._floor_px = max(1e-6, float(floor_px))

    def panorama(self, x, y, gaze_deg, fov_deg, samples_per_deg: float = 120.0):
        """渲染真实全景；仅在病理近距命中时抬到近点。"""
        panorama = self._panorama(x, y, gaze_deg, fov_deg, samples_per_deg)
        distances = panorama.distance_px
        if distances.size and float(np.min(distances)) < self._floor_px:
            clamped = np.where(
                np.isfinite(distances),
                np.maximum(distances, self._floor_px),
                distances,
            )
            return Panorama(panorama.rel_angles_deg, panorama.radiance, clamped)
        return panorama

    def __getattr__(self, name):
        """其余属性全部委托给真实场景。"""
        return getattr(self._scene, name)


class VisualFeatureExtractor:
    """从真实眼睛/视网膜/注视系统提取 16 维紧凑视觉特征。

    调用的是 :class:`~newworld.perception.VisualSystem`（眼 + 视网膜 +
    注视控制器）与 :class:`~newworld.perception.Audition`（听觉）——
    和 :class:`~newworld.perception.Perceiver` 内部用的是同一组真实感官
    部件；本类只是换成本模块注入的参数并把结果压成 16 维。

    Parameters
    ----------
    seed:
        视网膜镶嵌 / 注视随机的种子（同一 seed 的两条提取器行为一致）。
    retina_parameters:
        注入的 :class:`~newworld.retina.RetinaParameters`；默认是学习用的
        低分辨率档 ``n_cones=96, n_rods=0, n_bins=16``。高保真档可以传
        ``RetinaParameters()``。
    eye_parameters:
        注入的 :class:`~newworld.eye.EyeParameters`；默认只把全景角分辨率
        降到 20 samples/deg，其余解剖数值保持人类参数。
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        retina_parameters=None,
        eye_parameters=None,
    ) -> None:
        """组装真实视觉/听觉管线，并置好学习用低分辨率档。"""
        self._seed = 0 if seed is None else int(seed)
        self.retina_parameters: RetinaParameters = (
            retina_parameters
            if retina_parameters is not None
            else RetinaParameters(
                n_cones=LEARNING_RETINA_CONES,
                n_rods=LEARNING_RETINA_RODS,
                n_bins=LEARNING_RETINA_BINS,
            )
        )
        self.eye_parameters: EyeParameters = (
            eye_parameters if eye_parameters is not None else _learning_eye_parameters()
        )
        self.visual = VisualSystem(
            self._seed,
            eye_parameters=self.eye_parameters,
            retina_parameters=self.retina_parameters,
        )
        self.audition = Audition()
        #: 对焦距离下限（像素）：人眼近点换算；防"眼睛嵌入几何体"的病理帧
        self.near_point_px: float = NEAR_POINT_M / max(
            1e-9, float(getattr(self.eye_parameters, "world_scale_m_per_px", 0.01))
        )
        self._last_perception: PerceptionState | None = None
        self._last_state: VisualFeatureState | None = None
        #: 最近一次降级的原因（成功时为 None），供调试
        self.last_error: str | None = None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def observe(self, world, character=None, dt: float = 0.05) -> VisualFeatureState:
        """看一帧世界，返回 16 维视觉特征。

        Parameters
        ----------
        world:
            带 ``scene()`` / ``config`` / ``agent`` / ``sound`` 的世界
            （:class:`~newworld.world.World`）。任何缺失或异常都会被捕获
            并降级为零向量。
        character:
            要感知的居民；``None`` 时用 ``world.agent``。
        dt:
            本帧感知时间步（秒）。``dt <= 0`` 表示"不推进感知时间"，
            此时返回上一次真实结果并标记 ``source="cache"``（不会伪造新
            感知，也不会推进眼动状态）。

        Returns
        -------
        VisualFeatureState
            恒为 16 维有限向量；失败时全零 + ``source="none"``。
        """
        started = time.perf_counter()
        step_dt = float(dt)
        if step_dt <= 0.0:
            cached = self._last_state
            if cached is not None:
                return VisualFeatureState(
                    vector=cached.vector.copy(),
                    source="cache",
                    cost_ms=self._elapsed_ms(started),
                )
            return self._degraded(
                started, "dt<=0 but no cached perception yet"
            )
        try:
            agent = character if character is not None else world.agent
            # 真实场景外面包一层"对焦距离 ≥ 人眼近点"的保护（见 _NearPointScene）
            scene = _NearPointScene(world.scene(), self.near_point_px)
            perception = self.visual.perceive(
                scene, agent, world.config, step_dt
            )
            audio = self.audition.listen(world, agent)
            # 听觉字段来自 Audition（与 Perceiver.perceive 的合并方式一致）
            perception = replace(
                perception,
                sound_pressure_pa=audio.pressure_pa,
                sound_spl_db=audio.spl_db,
                hearing=audio.hearing,
            )
            vector = self._features(perception)
            state = VisualFeatureState(
                vector=vector,
                source="retina",
                cost_ms=self._elapsed_ms(started),
            )
            self._last_perception = perception
            self._last_state = state
            self.last_error = None
            return state
        except Exception as error:  # 感知失败不应让仿真/训练崩掉
            return self._degraded(started, f"{type(error).__name__}: {error}")

    # ------------------------------------------------------------------
    # 自省 / 维护
    # ------------------------------------------------------------------
    def last_perception(self) -> PerceptionState | None:
        """最近一次真实采集到的 :class:`PerceptionState`（调试用）。"""
        return self._last_perception

    def reset(self, seed: int | None = None) -> None:
        """重建眼/视网膜/注视状态（新 episode 时调用，保留注入的参数）。"""
        if seed is not None:
            self._seed = int(seed)
        self.visual = VisualSystem(
            self._seed,
            eye_parameters=self.eye_parameters,
            retina_parameters=self.retina_parameters,
        )
        self.audition = Audition()
        self._last_perception = None
        self._last_state = None
        self.last_error = None

    @property
    def zero_vector(self) -> np.ndarray:
        """降级时使用的零向量（副本，可安全修改）。"""
        return np.zeros(VISUAL_FEATURE_DIM, dtype=np.float64)

    # ------------------------------------------------------------------
    # 16 维特征
    # ------------------------------------------------------------------
    def _features(self, perception: PerceptionState) -> np.ndarray:
        """把真实 :class:`PerceptionState` 压成 16 维特征向量。"""
        values: list[float] = []
        # 0-7：神经节细胞分箱降采样
        values.extend(_ganglion_features(perception.retina))
        # 8-9：平均亮度 / 中央凹亮度（log 压缩）
        values.append(_squash_log_luminance(perception.mean_luminance))
        values.append(_squash_log_luminance(perception.foveal_luminance))
        # 10：光适应状态（视锥平均增益）
        values.append(_squash_adaptation(perception.retina.adaptation))
        # 11：注视偏移角 / 最大偏移
        rotation = float(getattr(self.eye_parameters, "max_rotation_deg", 50.0) or 50.0)
        values.append(_clip_unit(float(perception.gaze_offset_deg) / max(1e-6, rotation)))
        # 12：离焦模糊 σ
        values.append(_squash_blur(perception.blur_sigma_deg))
        # 13：扫视抑制强度（替代量）
        values.append(_saccade_suppression(perception))
        # 14：声压级 SPL
        values.append(_clip_unit(float(perception.sound_spl_db) / SPL_NORMALIZATION_DB))
        # 15：是否听到
        values.append(1.0 if perception.hearing else 0.0)
        return _finite_vector(values)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    @staticmethod
    def _elapsed_ms(started: float) -> float:
        """自 ``started`` 起的毫秒数（全有限）。"""
        return _finite_float((time.perf_counter() - started) * 1e3)

    def _degraded(self, started: float, reason: str) -> VisualFeatureState:
        """降级：零向量 + ``source="none"``，并记录原因。"""
        self.last_error = reason
        return VisualFeatureState(
            vector=self.zero_vector,
            source="none",
            cost_ms=self._elapsed_ms(started),
        )


# ---------------------------------------------------------------------------
# 参数构造
# ---------------------------------------------------------------------------
def _learning_eye_parameters() -> EyeParameters:
    """学习档眼参数：人类解剖值 + 更低的全景角分辨率。"""
    eye = EyeParameters()
    eye.panorama_samples_per_deg = LEARNING_PANORAMA_SAMPLES_PER_DEG
    return eye


# ---------------------------------------------------------------------------
# 单维特征（纯函数，便于测试）
# ---------------------------------------------------------------------------
def _ganglion_features(observation: RetinaObservation) -> list[float]:
    """前 8 维：神经节细胞响应按均值聚成 8 组后 tanh 定标。

    取的是 ``RetinaObservation.ganglion``——与
    ``RetinaObservation.feature_vector()`` 返回向量里的 ganglion 段是同一个
    数组（``feature_vector`` 只是把它拼接进定长向量），直接取数组避免了
    依赖 ``n_bins`` 推算切片偏移。箱数少于 8 时缺的组补零。
    """
    bins = np.asarray(getattr(observation, "ganglion", ()), dtype=np.float64).ravel()
    groups = np.zeros(GANGLION_FEATURES, dtype=np.float64)
    if bins.size:
        if bins.size % GANGLION_FEATURES == 0:
            # 常见情况（n_bins=16 → 每组 2 箱）：一次 reshape+mean，快且无循环
            groups = bins.reshape(GANGLION_FEATURES, -1).mean(axis=1)
        else:
            for index, chunk in enumerate(np.array_split(bins, GANGLION_FEATURES)):
                if chunk.size:
                    groups[index] = float(chunk.mean())
    return [math.tanh(float(value) * GANGLION_GAIN) for value in groups]


def _squash_log_luminance(value: float) -> float:
    """亮度 log10 压缩到 [-1, 1]：1.0 → +1，1e-3 → -1（更低截断）。"""
    level = _finite_float(value)
    clipped = min(max(level, LUMINANCE_FLOOR), 1.0)
    log_value = math.log10(clipped)
    top = math.log10(1.0)
    bottom = math.log10(LUMINANCE_FLOOR)
    return _clip_unit(2.0 * (log_value - bottom) / (top - bottom) - 1.0)


def _squash_adaptation(adaptation: float) -> float:
    """光适应（视锥平均增益）log 压缩：增益约 1 → 0.22，17 → 0.74。"""
    gain = max(0.0, _finite_float(adaptation))
    return float(math.tanh(math.log1p(gain) / 3.0))


def _squash_blur(sigma_deg: float) -> float:
    """离焦模糊 σ（度）压缩：0.01° → ≈0.02，0.5° → ≈0.76。"""
    sigma = max(0.0, _finite_float(sigma_deg))
    return float(math.tanh(sigma / 0.5))


def _saccade_suppression(perception: PerceptionState) -> float:
    """第 13 维：扫视抑制强度（替代量，0..1）。

    ``PerceptionState`` 没有显式抑制字段（管线内部以常数 ``neural_gain=0.25``
    施加），所以用两个真实可读的量替代：

    * 眨眼 → 1.0：眼睑完全挡光，是最强的"抑制"；
    * 扫视中 → ``|saccade_velocity_deg_s| / 600``，饱和到 1.0：扫视角速度
      越大，视网膜上图像移动越快、抑制越强（人类约 300-700 deg/s 达峰）。
    """
    if bool(getattr(perception, "blink", False)):
        return 1.0
    velocity = abs(_finite_float(getattr(perception, "saccade_velocity_deg_s", 0.0)))
    return _clip_unit(velocity / SACCADE_SUPPRESSION_VELOCITY_DEG_S)


# ---------------------------------------------------------------------------
# 数值工具
# ---------------------------------------------------------------------------
def _finite_float(value) -> float:
    """把任意输入变成有限 float（NaN/±inf → 0）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _clip_unit(value: float) -> float:
    """把值夹到 [-1, 1]（先过一遍有限性）。"""
    return min(1.0, max(-1.0, _finite_float(value)))


def _finite_vector(values) -> np.ndarray:
    """把候选特征变成 (16,) float64、全有限、约 [-1,1] 的向量。"""
    vector = np.asarray(values, dtype=np.float64).ravel()
    if vector.size != VISUAL_FEATURE_DIM:
        # 维度契约是硬的：不足补零、超出截断（而不是抛错拖垮仿真）
        fixed = np.zeros(VISUAL_FEATURE_DIM, dtype=np.float64)
        keep = min(vector.size, VISUAL_FEATURE_DIM)
        fixed[:keep] = vector[:keep]
        vector = fixed
    vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(vector, -1.0, 1.0)
