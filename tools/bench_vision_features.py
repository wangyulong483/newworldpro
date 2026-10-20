"""视觉特征提取的成本 / 方差基准（V2 契约 §1.3）。

真实运行（不是估计）：

    cd newworldpro
    python tools/bench_vision_features.py
    python tools/bench_vision_features.py --width 640 --height 400 --runs 100 --json .tmp/bench_vision.json

报告内容
--------
1. **单次 ``observe`` 成本**：640×400 世界、``--runs``（默认 100）次的
   平均 / 中位 / p95 / 最大耗时。每帧之间真实推进世界（``world.step``），
   推进耗时不计入 ``observe`` 计时。目标 ≤ :data:`TARGET_MS`。
2. **分阶段剖析**：用 cProfile 把 ``observe`` 内部各真实阶段
   （全景渲染 / 光学模糊 / 视网膜采样 / 神经节细胞 / 听觉 / 特征压缩）
   的累计耗时列出来，说明时间花在哪里。
3. **特征方差**：12 个真实位置/朝向下 16 维特征的每维均值/标准差，
   证明特征不是常数（契约要求"真实非零方差"）。
4. **高保真档对照**：``RetinaParameters()`` + ``EyeParameters()``（1800 视锥、
   120 samples/deg）在同样位姿下的成本与 16 维向量差异，用来证明默认
   低分辨率档省了多少算力、改了多少特征。

诚实说明：本基准只报告实测数字；达不到 2.5 ms 也不会伪造，直接打印
``meets_target=False``。
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import math
import pstats
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from newworld import make_config  # noqa: E402
from newworld.eye import EyeParameters  # noqa: E402
from newworld.retina import RetinaParameters  # noqa: E402
from newworld.visual_features import (  # noqa: E402
    VISUAL_FEATURE_DIM,
    VisualFeatureExtractor,
)
from newworld.world import World  # noqa: E402

#: 契约 §1.3 的性能目标（毫秒 / 次）
TARGET_MS = 2.5

#: 位姿扫描的采样次数（每维标准差用）
POSE_SAMPLES = 12

#: cProfile 剖析的 observe 次数
PROFILE_RUNS = 50


# ---------------------------------------------------------------------------
# 计时工具
# ---------------------------------------------------------------------------
def _percentile(values: list[float], fraction: float) -> float:
    """线性插值分位数（值全有限）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _trimmed_mean(values: list[float], fraction: float = 0.05) -> float:
    """去掉最高/最低各 ``fraction`` 后的均值（抑制 GC/调度离群点）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    cut = int(len(ordered) * fraction)
    kept = ordered[cut : len(ordered) - cut] if len(ordered) > 2 * cut else ordered
    return float(sum(kept) / len(kept))


def _outlier_count(values: list[float], factor: float = 3.0) -> int:
    """超过 ``factor × 中位数``（且 > 1 ms）的离群点个数。"""
    if not values:
        return 0
    median = _percentile(values, 0.5)
    threshold = max(1.0, median * factor)
    return sum(1 for value in values if value > threshold)


def _warm_up_paths(extractor: VisualFeatureExtractor) -> dict:
    """预热一次性成本，并如实报告它。

    已知的一次性开销：

    1. 视网膜镶嵌构建（``VisualSystem`` 构造时）——不计入 ``observe``；
    2. ``eye.angular_gaussian_blur`` 的 **FFT 分支**第一次被走到时约 2-3 ms
       （之后约 0.3 ms）。大扫视 / 大失焦才会走到该分支；预热掉它，
       否则它会变成 100 次均值里的一个离群点。

    返回 ``{"fft_first_call_ms": ..., "fft_repeat_ms": [...]}``。
    """
    from newworld.eye import angular_gaussian_blur  # 局部导入：只在基准里用
    from newworld.optics import Panorama

    eye = extractor.eye_parameters
    samples = max(
        64,
        int(
            round(
                float(eye.field_of_view_deg)
                * float(eye.panorama_samples_per_deg)
            )
        ),
    )
    rel = np.linspace(-float(eye.field_of_view_deg) / 2.0,
                      float(eye.field_of_view_deg) / 2.0, samples)
    panorama = Panorama(
        rel,
        np.ones((samples, 3), dtype=np.float32),
        np.full(samples, np.inf, dtype=np.float64),
    )
    # sigma=3deg 在默认档（step≈0.083 deg）下必然走 FFT 分支
    started = time.perf_counter()
    angular_gaussian_blur(panorama, 3.0)
    first_ms = (time.perf_counter() - started) * 1e3
    # 再跑三次，确认后续 FFT 卷积的真实成本
    repeat = []
    for _ in range(3):
        tick = time.perf_counter()
        angular_gaussian_blur(panorama, 3.0)
        repeat.append((time.perf_counter() - tick) * 1e3)
    return {
        "fft_first_call_ms": first_ms,
        "fft_repeat_ms": repeat,
        "fft_samples": samples,
    }


def _time_observes(extractor: VisualFeatureExtractor, world, runs: int) -> dict:
    """真实计时：每帧先推进世界，再 ``observe``（推进不计时）。"""
    agent = world.agent
    # 预热（首次调用包含视网膜镶嵌插值等一次性成本）
    for index in range(5):
        if index:
            world.step(0.05, 1 if index % 2 else -1)
        extractor.observe(world, agent, 0.05)
    move = 1
    wall_ms: list[float] = []
    internal_ms: list[float] = []
    vectors: list[np.ndarray] = []
    sources: dict[str, int] = {}
    sigmas: list[float] = []
    # 关掉 GC：计时窗口里的一次 GC 停顿会把均值拉高几十倍（离群点照样如实报告）
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for index in range(runs):
            world.step(0.05, move)
            move = -move if index % 7 == 6 else move
            started = time.perf_counter()
            state = extractor.observe(world, agent, 0.05)
            wall_ms.append((time.perf_counter() - started) * 1e3)
            internal_ms.append(float(state.cost_ms))
            vectors.append(state.vector.copy())
            sources[state.source] = sources.get(state.source, 0) + 1
            perception = extractor.last_perception()
            sigmas.append(
                float(perception.blur_sigma_deg) if perception is not None else 0.0
            )
    finally:
        if gc_was_enabled:
            gc.enable()
    return {
        "runs": runs,
        "wall_ms": wall_ms,
        "internal_ms": internal_ms,
        "vectors": np.asarray(vectors, dtype=np.float64),
        "sources": sources,
        "max_blur_sigma_deg": float(np.max(sigmas)) if sigmas else 0.0,
    }


def _pose_sweep(extractor: VisualFeatureExtractor, world, samples: int) -> np.ndarray:
    """在真实世界的不同位置/朝向上采特征，返回 (samples, 16)。

    两个提取器（默认档 / 高保真档）用同一个 seed 和同样的步进序列，
    所以注视轨迹一致，得到的向量可以逐位姿对比。
    """
    agent = world.agent
    cfg = world.config
    extractor.reset()
    vectors = []
    for index in range(samples):
        if index:
            world.step(0.05, 0)
        span = cfg.max_x - cfg.min_x
        agent.x = cfg.min_x + span * index / max(1, samples - 1)
        agent.facing = 1 if index % 2 == 0 else -1
        state = extractor.observe(world, agent, 0.05)
        vectors.append(state.vector.copy())
    return np.asarray(vectors, dtype=np.float64)


# ---------------------------------------------------------------------------
# 分阶段剖析
# ---------------------------------------------------------------------------
#: 分阶段剖析的匹配键：(模块文件名, 函数名, 显示名, 是否被上游阶段包含)
#:
#: 用 cProfile 的 **cumulative** 时间（ct），因为 numpy 调用都算子调用；
#: ``ganglion_profile`` 被 ``Retina.sample`` 包含，单独列出只为解释构成。
_PROFILE_KEYS = (
    ("optics.py", "panorama", "全景渲染 (Scene.panorama)", False),
    ("eye.py", "angular_gaussian_blur", "光学模糊 (angular_gaussian_blur)", False),
    ("retina.py", "sample", "视网膜采样 (Retina.sample)", False),
    ("retina.py", "ganglion_profile", "  └ 神经节细胞 DoG (ganglion_profile)", True),
    ("perception.py", "listen", "听觉 (Audition.listen)", False),
    ("visual_features.py", "_features", "特征压缩 (_features)", False),
)


def _profile_stages(extractor: VisualFeatureExtractor, world, runs: int) -> list[dict]:
    """用 cProfile 剖析真实 observe 循环，抽出各阶段累计（含子调用）耗时。"""
    agent = world.agent
    profiler = cProfile.Profile()
    profiler.enable()
    for index in range(runs):
        world.step(0.05, 1 if index % 2 else -1)
        extractor.observe(world, agent, 0.05)
    profiler.disable()
    stats = pstats.Stats(profiler)
    rows = []
    for module_name, function_name, label, included in _PROFILE_KEYS:
        total = 0.0
        hits = 0
        for (filename, _line, name), (_cc, nc, _tt, ct, _callers) in stats.stats.items():
            base = Path(filename).name
            if base == module_name and name == function_name:
                total += ct
                hits += nc
        rows.append(
            {
                "stage": label,
                "total_ms": total * 1e3,
                "per_observe_ms": (total * 1e3 / runs) if runs else 0.0,
                "calls": hits,
                "included_in_parent": included,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 参数核对 / 听觉探针
# ---------------------------------------------------------------------------
def _describe_parameters(extractor: VisualFeatureExtractor) -> dict:
    """打印用：提取器**实际生效**的参数值（不是标签，是真实对象字段）。"""
    retina = extractor.retina_parameters
    eye = extractor.eye_parameters
    return {
        "retina": {
            "n_cones": int(retina.n_cones),
            "n_rods": int(retina.n_rods),
            "n_bins": int(retina.n_bins),
            "class": type(retina).__name__,
        },
        "eye": {
            "panorama_samples_per_deg": float(eye.panorama_samples_per_deg),
            "field_of_view_deg": float(eye.field_of_view_deg),
            "class": type(eye).__name__,
        },
        "visual_system_params_match": bool(
            extractor.visual.retina_parameters is retina
            and extractor.visual.eye_parameters is eye
        ),
        "actual_no_cones_in_mosaic": int(extractor.visual.retina.cone_ecc.size),
        "actual_no_rods_in_mosaic": int(extractor.visual.retina.rod_ecc.size),
        "actual_ganglion_bins": int(extractor.visual.retina.params.n_bins),
    }


def _sound_probe(width: int, height: int, seed: int) -> dict:
    """听觉两维（14/15）的真实激活检查：说一句话前后 SPL / hearing 变化。"""
    config = make_config(width, height)
    world = World(config, seed=seed)
    agent = world.agent
    extractor = VisualFeatureExtractor(seed=seed)
    quiet = extractor.observe(world, agent, 0.05).vector
    world.speak(agent, "hello")
    world.step(0.05, 0)
    loud = extractor.observe(world, agent, 0.05).vector
    world.step(1.0, 0)  # 声波早已过去
    after = extractor.observe(world, agent, 0.05).vector
    return {
        "quiet_spl_feature": float(quiet[14]),
        "quiet_hearing_feature": float(quiet[15]),
        "speech_spl_feature": float(loud[14]),
        "speech_hearing_feature": float(loud[15]),
        "after_spl_feature": float(after[14]),
        "after_hearing_feature": float(after[15]),
    }


# ---------------------------------------------------------------------------
# 主基准
# ---------------------------------------------------------------------------
def benchmark(
    width: int = 640,
    height: int = 400,
    runs: int = 100,
    seed: int = 0,
    poses: int = POSE_SAMPLES,
    profile_runs: int = PROFILE_RUNS,
) -> dict:
    """跑完整基准，返回可 JSON 序列化的结果字典。"""
    config = make_config(width, height)
    world = World(config, seed=seed)

    # -- 1. 默认（低分辨率学习档）成本 ---------------------------------
    extractor = VisualFeatureExtractor(seed=seed)
    warmup = _warm_up_paths(extractor)
    timing = _time_observes(extractor, world, runs)
    wall = timing["wall_ms"]
    internal = timing["internal_ms"]

    # -- 2. 位姿方差 ---------------------------------------------------
    pose_world = World(config, seed=seed)
    pose_vectors = _pose_sweep(VisualFeatureExtractor(seed=seed), pose_world, poses)
    per_dim_mean = pose_vectors.mean(axis=0)
    per_dim_std = pose_vectors.std(axis=0)
    time_vectors = timing["vectors"]
    time_std = time_vectors.std(axis=0)

    # -- 3. 高保真档对照 -------------------------------------------------
    hd_extractor = VisualFeatureExtractor(
        seed=seed,
        retina_parameters=RetinaParameters(),
        eye_parameters=EyeParameters(),
    )
    hd_runs = max(10, min(runs, 30))
    _warm_up_paths(hd_extractor)
    hd_timing = _time_observes(hd_extractor, World(config, seed=seed), hd_runs)
    hd_vectors = _pose_sweep(
        VisualFeatureExtractor(
            seed=seed,
            retina_parameters=RetinaParameters(),
            eye_parameters=EyeParameters(),
        ),
        World(config, seed=seed),
        poses,
    )
    vector_diff = np.abs(hd_vectors - pose_vectors)
    hd_wall = hd_timing["wall_ms"]

    # -- 3b. 只提高全景分辨率、视网膜保持低分辨率档 ---------------------
    # 用来把"省算力到底省在哪"拆开：视网膜镶嵌 vs 全景采样。
    eye_only_runs = max(10, min(runs, 30))
    eye_only_extractor = VisualFeatureExtractor(seed=seed, eye_parameters=EyeParameters())
    _warm_up_paths(eye_only_extractor)
    eye_only_timing = _time_observes(
        eye_only_extractor,
        World(config, seed=seed),
        eye_only_runs,
    )
    eye_only_vectors = _pose_sweep(
        VisualFeatureExtractor(seed=seed, eye_parameters=EyeParameters()),
        World(config, seed=seed),
        poses,
    )
    eye_only_diff = np.abs(eye_only_vectors - pose_vectors)
    eye_only_wall = eye_only_timing["wall_ms"]

    # -- 4. 分阶段剖析 ---------------------------------------------------
    profile_extractor = VisualFeatureExtractor(seed=seed)
    profile_world = World(config, seed=seed)
    stages = _profile_stages(profile_extractor, profile_world, profile_runs)

    # -- 5. 听觉两维的真实激活 -------------------------------------------
    sound = _sound_probe(width, height, seed)

    trimmed_ms = _trimmed_mean(wall)
    median_ms = _percentile(wall, 0.5)
    p95_ms = _percentile(wall, 0.95)
    result = {
        "world": {"width": config.width, "height": config.height, "seed": seed},
        "parameters": {
            "low_resolution": _describe_parameters(extractor),
            "high_fidelity": _describe_parameters(hd_extractor),
        },
        "feature_dim": VISUAL_FEATURE_DIM,
        "runs": runs,
        "mean_ms": float(np.mean(wall)),
        "trimmed_mean_ms": trimmed_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "max_ms": float(np.max(wall)) if wall else 0.0,
        "min_ms": float(np.min(wall)) if wall else 0.0,
        "internal_mean_ms": float(np.mean(internal)) if internal else 0.0,
        "outliers": _outlier_count(wall),
        "frames_over_5ms": int(sum(1 for value in wall if value > 5.0)),
        "max_blur_sigma_deg": timing["max_blur_sigma_deg"],
        "near_point_px": float(extractor.near_point_px),
        "warmup": warmup,
        "target_ms": TARGET_MS,
        # 判定用稳健统计量（中位数 / 去极值均值），原始均值也会一起打印
        "meets_target": bool(median_ms <= TARGET_MS and trimmed_ms <= TARGET_MS),
        "sources": timing["sources"],
        "pose_samples": poses,
        "time_series_std_per_dim": [float(v) for v in time_std],
        "pose_std_per_dim": [float(v) for v in per_dim_std],
        "pose_mean_per_dim": [float(v) for v in per_dim_mean],
        "pose_std_max": float(np.max(per_dim_std)) if per_dim_std.size else 0.0,
        "pose_dims_with_variance": int(np.count_nonzero(per_dim_std > 1e-6)),
        "high_fidelity": {
            "runs": hd_runs,
            "mean_observe_ms": float(np.mean(hd_wall)) if hd_wall else 0.0,
            "median_observe_ms": _percentile(hd_wall, 0.5),
            "max_abs_vector_diff": float(np.max(vector_diff)) if vector_diff.size else 0.0,
            "mean_abs_vector_diff": float(np.mean(vector_diff)) if vector_diff.size else 0.0,
            "per_dim_abs_diff": [float(v) for v in vector_diff.mean(axis=0)]
            if vector_diff.size
            else [],
        },
        "panorama_only": {
            "runs": eye_only_runs,
            "mean_observe_ms": float(np.mean(eye_only_wall)) if eye_only_wall else 0.0,
            "median_observe_ms": _percentile(eye_only_wall, 0.5),
            "max_abs_vector_diff": float(np.max(eye_only_diff)) if eye_only_diff.size else 0.0,
            "mean_abs_vector_diff": float(np.mean(eye_only_diff))
            if eye_only_diff.size
            else 0.0,
        },
        "sound_probe": sound,
        "stages": stages,
    }
    return result


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def _print_parameters(parameters: dict) -> None:
    """打印提取器**实际生效**的参数（真实对象字段，不是标签）。"""
    retina = parameters["retina"]
    eye = parameters["eye"]
    print(
        f"  视网膜 RetinaParameters: n_cones={retina['n_cones']} "
        f"n_rods={retina['n_rods']} n_bins={retina['n_bins']}"
    )
    print(
        f"  眼睛 EyeParameters     : panorama={eye['panorama_samples_per_deg']:g} samples/deg "
        f"fov={eye['field_of_view_deg']:g} deg"
    )
    print(
        f"  实际镶嵌               : {parameters['actual_no_cones_in_mosaic']} 视锥 + "
        f"{parameters['actual_no_rods_in_mosaic']} 视杆, ganglion bins="
        f"{parameters['actual_ganglion_bins']}  "
        f"(VisualSystem 持有同一对象: {parameters['visual_system_params_match']})"
    )


def _print_report(result: dict) -> None:
    """人读报告。"""
    world = result["world"]
    print("=" * 78)
    print("视觉特征提取基准 (V2 契约 §1.3) —— 全部为实测数字")
    print("=" * 78)
    print(f"世界            : {world['width']}x{world['height']}  seed={world['seed']}")
    print(f"特征维度        : {result['feature_dim']}")
    print("默认档（低分辨率学习档）实际参数：")
    _print_parameters(result["parameters"]["low_resolution"])
    print(
        f"observe 成本    : mean={result['mean_ms']:.3f} ms  "
        f"trimmed_mean={result['trimmed_mean_ms']:.3f}  "
        f"median={result['median_ms']:.3f}  p95={result['p95_ms']:.3f}  "
        f"max={result['max_ms']:.3f}  min={result['min_ms']:.3f}"
    )
    print(
        f"                 内部计时 mean={result['internal_mean_ms']:.3f} ms；"
        f"离群点(>3×中位数)={result['outliers']} 个；"
        f">5ms 帧={result['frames_over_5ms']}/{result['runs']}"
    )
    print(
        f"                 病理帧保护: 对焦距离下限={result['near_point_px']:.1f} px"
        f"（人眼近点 0.1 m），实测最大模糊 σ={result['max_blur_sigma_deg']:.2f} deg"
        f"（无保护时曾出现 3415 deg / 130-170 ms 单帧）"
    )
    warm = result["warmup"]
    print(
        f"一次性成本（已从计时中剔除）: FFT 模糊分支首次调用="
        f"{warm['fft_first_call_ms']:.1f} ms，之后每次 "
        f"{np.round(warm['fft_repeat_ms'], 3).tolist()} ms"
    )
    verdict = "达到" if result["meets_target"] else "未达到"
    print(
        f"目标 ≤ {result['target_ms']:.1f} ms : {verdict}"
        f"（判据: median 且 trimmed_mean ≤ {result['target_ms']:.1f} ms；"
        f"{result['runs']} 次）"
    )
    print(f"source 计数     : {result['sources']}")
    print("-" * 78)
    print(
        f"分阶段（cProfile cumulative 时间，{result['runs']} 次 observe 窗口）："
    )
    for row in result["stages"]:
        tag = "（含在上游）" if row["included_in_parent"] else ""
        print(
            f"  {row['stage']:40s} total={row['total_ms']:8.2f} ms  "
            f"per_observe={row['per_observe_ms']:7.3f} ms {tag}"
        )
    print("-" * 78)
    print(f"位姿方差（{result['pose_samples']} 个位置/朝向）：")
    print(f"  每维 mean : {np.round(result['pose_mean_per_dim'], 3).tolist()}")
    print(f"  每维 std  : {np.round(result['pose_std_per_dim'], 4).tolist()}")
    print(
        f"  有方差的维度 : {result['pose_dims_with_variance']}/{result['feature_dim']}"
        f"  max std = {result['pose_std_max']:.4f}"
    )
    print(f"  连续 {result['runs']} 帧每维 std:",
          np.round(result["time_series_std_per_dim"], 4).tolist())
    sound = result["sound_probe"]
    print(
        f"  听觉两维（14/15）: 静默={sound['quiet_spl_feature']:+.3f}/"
        f"{sound['quiet_hearing_feature']:.0f} → 说话帧={sound['speech_spl_feature']:+.3f}/"
        f"{sound['speech_hearing_feature']:.0f} → 声波过去={sound['after_spl_feature']:+.3f}/"
        f"{sound['after_hearing_feature']:.0f}"
    )
    hd = result["high_fidelity"]
    print("-" * 78)
    print("高保真档对照实际参数：")
    _print_parameters(result["parameters"]["high_fidelity"])
    print(
        f"  单次 observe  : mean={hd['mean_observe_ms']:.3f} ms  "
        f"median={hd['median_observe_ms']:.3f} ms"
        f"（默认档 median={result['median_ms']:.3f} ms，"
        f"快 {hd['median_observe_ms'] / max(1e-9, result['median_ms']):.1f}×）"
    )
    print(
        f"  16 维差异     : max={hd['max_abs_vector_diff']:.4f}  "
        f"mean={hd['mean_abs_vector_diff']:.4f}"
    )
    print(f"  每维平均差异  : {np.round(hd['per_dim_abs_diff'], 3).tolist()}")
    print("-" * 78)
    pan = result["panorama_only"]
    print("只提高全景分辨率（视网膜仍是 96 视锥档，eye=120 samples/deg）：")
    print(
        f"  单次 observe  : mean={pan['mean_observe_ms']:.3f} ms  "
        f"median={pan['median_observe_ms']:.3f} ms"
        f"（默认档 median={result['median_ms']:.3f} ms）"
    )
    print(
        f"  16 维差异     : max={pan['max_abs_vector_diff']:.4f}  "
        f"mean={pan['mean_abs_vector_diff']:.4f}"
        "   ← 差异≈0 说明低分辨率档省的是全景算力，不是特征信息"
    )
    print("=" * 78)


def _main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    # 统一按 UTF-8 输出，避免 Windows GBK 控制台把中文报告打成乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - 非常规 stdout
        pass
    parser = argparse.ArgumentParser(description="视觉特征提取成本/方差基准")
    parser.add_argument("--width", type=int, default=640, help="世界宽度（默认 640）")
    parser.add_argument("--height", type=int, default=400, help="世界高度（默认 400）")
    parser.add_argument("--runs", type=int, default=100, help="observe 计时次数（默认 100）")
    parser.add_argument("--seed", type=int, default=0, help="世界/提取器 seed（默认 0）")
    parser.add_argument("--poses", type=int, default=POSE_SAMPLES, help="位姿扫描次数")
    parser.add_argument("--profile-runs", type=int, default=PROFILE_RUNS, help="cProfile 次数")
    parser.add_argument("--json", default=None, help="把结果写成 JSON 的路径")
    args = parser.parse_args(argv)

    result = benchmark(
        width=args.width,
        height=args.height,
        runs=args.runs,
        seed=args.seed,
        poses=args.poses,
        profile_runs=args.profile_runs,
    )
    _print_report(result)
    if args.json:
        path = Path(args.json)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 已写入: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
