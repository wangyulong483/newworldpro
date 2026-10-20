"""视觉特征测试（V2 契约 §1.2 / §1.4）。

覆盖：维度 / 有限性 / 归一化范围、真实世界非零方差、朝向与位置变化、场景
布局变化（真实光学）、异常输入降级为零向量、``source`` 字段、低分辨率参数
生效、ganglion 8 维确实来自 ``RetinaObservation``、听觉两维跟随真实声场、
``last_perception``、benchmark 脚本可运行。

所有特征都跑真实 ``World``（640×400）+ 真实 ``VisualSystem.perceive``，
没有 fake 感知数；只有"异常输入"用例故意喂坏输入来验证降级。
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:  # 沙箱环境下 tempfile.mkdtemp 不可写，用项目垫片（契约 §0）
    from tests._env_compat import patch_tempdir
except ImportError:  # pragma: no cover - 直接以 tests 目录为根运行时
    try:
        from _env_compat import patch_tempdir
    except ImportError:
        patch_tempdir = None  # type: ignore[assignment]
if patch_tempdir is not None:
    patch_tempdir()

from newworld import make_config  # noqa: E402
from newworld.eye import EyeParameters  # noqa: E402
from newworld.optics import Box  # noqa: E402
from newworld.perception import PerceptionState  # noqa: E402
from newworld.retina import RetinaParameters  # noqa: E402
from newworld.visual_features import (  # noqa: E402
    GANGLION_FEATURES,
    GANGLION_GAIN,
    VISUAL_FEATURE_DIM,
    VisualFeatureExtractor,
    VisualFeatureState,
)
from newworld.world import World  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SIZE = (640, 400)


def small_world(seed: int = 0) -> World:
    """640×400 真实世界（契约 §1.3 的基准尺寸）。"""
    return World(make_config(*SIZE), seed=seed)


def pose(extractor: VisualFeatureExtractor, world: World, x=None, facing=1, steps=1):
    """重置提取器 → 摆好位置/朝向 → 真实 observe（返回 (16,) 向量）。"""
    agent = world.agent
    if x is not None:
        agent.x = float(x)
    agent.facing = 1 if facing >= 0 else -1
    extractor.reset()
    for _ in range(steps):
        world.step(0.05, 0)
    return extractor.observe(world, agent, 0.05).vector


class TestVisualFeatureContract(unittest.TestCase):
    """接口 / 数值契约。"""

    def test_dimension_and_dtype(self) -> None:
        """VISUAL_FEATURE_DIM == 16，向量形状 (16,)、float64。"""
        self.assertEqual(VISUAL_FEATURE_DIM, 16)
        self.assertEqual(GANGLION_FEATURES, 8)
        world = small_world(1)
        extractor = VisualFeatureExtractor(seed=1)
        state = extractor.observe(world, world.agent, 0.05)
        self.assertIsInstance(state, VisualFeatureState)
        self.assertEqual(state.vector.shape, (VISUAL_FEATURE_DIM,))
        self.assertEqual(state.vector.dtype, np.float64)
        self.assertEqual(len(state.vector), 16)
        self.assertEqual(state.source, "retina")
        self.assertGreaterEqual(state.cost_ms, 0.0)

    def test_finite_and_normalized(self) -> None:
        """真实世界连续 24 帧：全有限，且落在约 [-1, 1]。"""
        world = small_world(2)
        extractor = VisualFeatureExtractor(seed=2)
        agent = world.agent
        for index in range(24):
            agent.x = 20.0 + 8.0 * index
            agent.facing = 1 if index % 2 == 0 else -1
            world.step(0.05, 1 if index % 3 else -1)
            state = extractor.observe(world, agent, 0.05)
            vector = state.vector
            self.assertTrue(np.all(np.isfinite(vector)), f"non-finite at frame {index}")
            self.assertLessEqual(float(np.abs(vector).max()), 1.0)
            self.assertGreaterEqual(float(np.abs(vector).max()), 0.0)

    def test_zero_vector_constant(self) -> None:
        """降级零向量是 16 维浮点零。"""
        extractor = VisualFeatureExtractor(seed=0)
        zero = extractor.zero_vector
        self.assertEqual(zero.shape, (VISUAL_FEATURE_DIM,))
        self.assertTrue(np.all(zero == 0.0))
        self.assertTrue(np.all(np.isfinite(zero)))


class TestRealPerceptionSignal(unittest.TestCase):
    """特征必须来自真实感知，并且有真实方差。"""

    def test_nonzero_variance_across_poses(self) -> None:
        """12 个真实位置/朝向下 16 维特征不是常数（契约 §1.3）。

        世界无语音时最后两维（SPL / 是否听到）恒为 0，这是**预期**的：
        ``world.speak`` 时它们会变化（见 ``test_audition_dims_follow_real_sound``）。
        """
        world = small_world(3)
        extractor = VisualFeatureExtractor(seed=3)
        config = world.config
        rows = []
        for index in range(12):
            x = config.min_x + (config.max_x - config.min_x) * index / 11.0
            rows.append(pose(extractor, world, x=x, facing=1 if index % 2 == 0 else -1))
        vectors = np.asarray(rows)
        std_per_dim = vectors.std(axis=0)
        active = int(np.count_nonzero(std_per_dim > 1e-3))
        self.assertGreaterEqual(active, 10, f"only {active} dims vary: {np.round(std_per_dim, 5)}")
        self.assertGreater(float(std_per_dim.max()), 0.02)
        # 最后两维在静默世界里恒为 0（预期行为）
        self.assertEqual(float(std_per_dim[14]), 0.0)
        self.assertEqual(float(std_per_dim[15]), 0.0)
        # 也不是"全零向量"
        self.assertGreater(float(np.abs(vectors).max()), 0.05)

    def test_variance_over_time(self) -> None:
        """同一位置连续 30 帧（眼动/适应持续进行）特征有非零方差。"""
        world = small_world(4)
        extractor = VisualFeatureExtractor(seed=4)
        agent = world.agent
        rows = []
        for _ in range(30):
            world.step(0.05, 0)
            rows.append(extractor.observe(world, agent, 0.05).vector)
        std_per_dim = np.asarray(rows).std(axis=0)
        self.assertGreater(float(std_per_dim.max()), 0.01)
        self.assertGreaterEqual(int(np.count_nonzero(std_per_dim > 1e-4)), 6)

    def test_facing_changes_features(self) -> None:
        """只改朝向（左右转 180°）就改变特征——真实光场方向变了。"""
        world = small_world(5)
        extractor = VisualFeatureExtractor(seed=5)
        x = world.config.width * 0.35
        right = pose(extractor, world, x=x, facing=1)
        left = pose(extractor, world, x=x, facing=-1)
        self.assertGreater(float(np.abs(right - left).max()), 0.01)

    def test_position_changes_ganglion_features(self) -> None:
        """换位置 → 前 8 维（视网膜神经节细胞）至少有一维变化。"""
        world = small_world(6)
        extractor = VisualFeatureExtractor(seed=6)
        config = world.config
        first = pose(extractor, world, x=config.min_x + 40.0)
        second = pose(extractor, world, x=config.width * 0.72)
        self.assertGreater(float(np.abs(first[:GANGLION_FEATURES] - second[:GANGLION_FEATURES]).max()), 1e-3)

    def test_scene_layout_changes_features(self) -> None:
        """在眼前加一块暗色物体（真实光学场景）→ 特征变化。"""
        world = small_world(7)
        extractor = VisualFeatureExtractor(seed=7)
        agent = world.agent
        agent.x = world.config.width * 0.5
        agent.facing = 1
        before = pose(extractor, world, x=agent.x, facing=1)
        world.add_object(
            Box(agent.x + 40.0, 200.0, agent.x + 140.0, 392.0, albedo=(0.02, 0.02, 0.02))
        )
        after = pose(extractor, world, x=agent.x, facing=1)
        self.assertGreater(float(np.abs(before - after).max()), 0.01)

    def test_ganglion_dims_come_from_retina_observation(self) -> None:
        """前 8 维 == tanh(mean(视网膜 ganglion 箱分组) × GANGLION_GAIN)。

        这条把"特征来自真实视网膜"钉死：拿 ``last_perception()`` 里的
        ``RetinaObservation.ganglion`` 手算一遍，必须与特征前 8 维逐位相等。
        """
        world = small_world(8)
        extractor = VisualFeatureExtractor(seed=8)
        state = extractor.observe(world, world.agent, 0.05)
        perception = extractor.last_perception()
        self.assertIsInstance(perception, PerceptionState)
        ganglion = np.asarray(perception.retina.ganglion, dtype=np.float64)
        self.assertEqual(ganglion.size, 16)
        # 契约 §1.2 说 ganglion 来自 retina.feature_vector() 的 n_bins 箱：
        # 布局 = cone_hist(n_bins*3) + rod_hist(n_bins) + ganglion(n_bins) + scalars(3)
        feature_vector = np.asarray(perception.retina.feature_vector(), dtype=np.float64)
        bins = ganglion.size
        offset = bins * 3 + bins
        np.testing.assert_allclose(
            feature_vector[offset : offset + bins], ganglion, atol=1e-6
        )
        groups = ganglion.reshape(GANGLION_FEATURES, -1).mean(axis=1)
        expected = np.tanh(groups * GANGLION_GAIN)
        np.testing.assert_allclose(state.vector[:GANGLION_FEATURES], expected, atol=1e-12)
        # 亮度 / 适应维度也直接来自 PerceptionState
        self.assertAlmostEqual(
            float(state.vector[8]),
            float(np.clip(2.0 * (math.log10(min(max(perception.mean_luminance, 1e-3), 1.0)) + 3.0) / 3.0 - 1.0, -1.0, 1.0)),
            places=6,
        )
        self.assertGreater(float(state.vector[10]), 0.0)  # 光适应增益 > 0

    def test_audition_dims_follow_real_sound(self) -> None:
        """第 14/15 维跟随真实声场：静默 0/0 → 说话帧 >0/1 → 声波过去回 0。"""
        world = small_world(9)
        extractor = VisualFeatureExtractor(seed=9)
        agent = world.agent
        quiet = extractor.observe(world, agent, 0.05).vector
        self.assertEqual(float(quiet[14]), 0.0)
        self.assertEqual(float(quiet[15]), 0.0)
        world.speak(agent, "hello")
        world.step(0.05, 0)
        loud = extractor.observe(world, agent, 0.05).vector
        self.assertGreater(float(loud[14]), 0.1)
        self.assertEqual(float(loud[15]), 1.0)
        world.step(1.0, 0)  # 声波早已过去
        after = extractor.observe(world, agent, 0.05).vector
        self.assertEqual(float(after[14]), 0.0)
        self.assertEqual(float(after[15]), 0.0)


class TestDegradationAndSource(unittest.TestCase):
    """异常降级 / source 字段。"""

    def test_none_world_degrades_to_zero(self) -> None:
        """world=None 不抛错：零向量 + source="none"。"""
        extractor = VisualFeatureExtractor(seed=0)
        state = extractor.observe(None, None, 0.05)
        self.assertEqual(state.source, "none")
        self.assertTrue(np.all(state.vector == 0.0))
        self.assertEqual(state.vector.shape, (VISUAL_FEATURE_DIM,))
        self.assertIsNotNone(extractor.last_error)

    def test_broken_scene_degrades_to_zero(self) -> None:
        """世界的 scene() 抛错 → 零向量，不抛出。"""

        class BrokenWorld:
            config = make_config(*SIZE)

            class _Agent:
                facing = 1
                x = 100.0
                y = None

            agent = _Agent()

            def scene(self):
                raise RuntimeError("no optics today")

        state = VisualFeatureExtractor(seed=0).observe(BrokenWorld(), None, 0.05)
        self.assertEqual(state.source, "none")
        self.assertTrue(np.all(state.vector == 0.0))
        # 失败原因被记录下来（类型 + 消息），方便集成时定位
        extractor = VisualFeatureExtractor(seed=0)
        extractor.observe(BrokenWorld(), None, 0.05)
        self.assertIn("RuntimeError", extractor.last_error or "")
        self.assertIn("no optics today", extractor.last_error or "")

    def test_world_without_agents_degrades(self) -> None:
        """世界没有居民（world.agent 抛 NoAgentsError）→ 零向量，不崩。"""
        world = small_world(10)
        world.agents.clear()
        state = VisualFeatureExtractor(seed=10).observe(world, None, 0.05)
        self.assertEqual(state.source, "none")
        self.assertTrue(np.all(state.vector == 0.0))

    def test_source_field_retina_cache_none(self) -> None:
        """source：真实采集 "retina"，dt<=0 复用 "cache"，失败 "none"。"""
        world = small_world(11)
        extractor = VisualFeatureExtractor(seed=11)
        fresh = extractor.observe(world, world.agent, 0.05)
        self.assertEqual(fresh.source, "retina")
        cached = extractor.observe(world, world.agent, 0.0)
        self.assertEqual(cached.source, "cache")
        np.testing.assert_allclose(cached.vector, fresh.vector)
        degraded = extractor.observe(None, None, 0.05)
        self.assertEqual(degraded.source, "none")
        # 失败后 cache 仍然可用（上一次成功的结果）
        again = extractor.observe(world, world.agent, 0.0)
        self.assertEqual(again.source, "cache")
        np.testing.assert_allclose(again.vector, fresh.vector)

    def test_last_perception_lifecycle(self) -> None:
        """last_perception()：观察前 None，观察后是真实 PerceptionState。"""
        world = small_world(12)
        extractor = VisualFeatureExtractor(seed=12)
        self.assertIsNone(extractor.last_perception())
        extractor.observe(world, world.agent, 0.05)
        perception = extractor.last_perception()
        self.assertIsInstance(perception, PerceptionState)
        self.assertGreater(perception.retina.ganglion.size, 0)
        self.assertGreater(perception.time, 0.0)
        # reset 后清空
        extractor.reset()
        self.assertIsNone(extractor.last_perception())


class TestParameters(unittest.TestCase):
    """低分辨率参数注入必须真正生效。"""

    def test_default_profile_is_low_resolution(self) -> None:
        """默认档就是契约规定的 96/0/16，并且真的被 VisualSystem 使用。"""
        extractor = VisualFeatureExtractor(seed=0)
        retina = extractor.retina_parameters
        self.assertEqual((retina.n_cones, retina.n_rods, retina.n_bins), (96, 0, 16))
        self.assertLessEqual(extractor.eye_parameters.panorama_samples_per_deg, 20.0)
        visual_retina = extractor.visual.retina
        self.assertEqual(visual_retina.cone_ecc.size, 96)
        self.assertEqual(visual_retina.rod_ecc.size, 0)
        self.assertEqual(int(visual_retina.params.n_bins), 16)
        # VisualSystem 持有的是同一个参数对象（不是复制一份默认值）
        self.assertIs(extractor.visual.retina_parameters, retina)
        self.assertIs(extractor.visual.eye_parameters, extractor.eye_parameters)

    def test_injected_parameters_take_effect(self) -> None:
        """注入自定义 RetinaParameters：镶嵌规模真的变，特征仍是 16 维。"""
        retina = RetinaParameters(n_cones=24, n_rods=0, n_bins=8)
        extractor = VisualFeatureExtractor(seed=0, retina_parameters=retina)
        self.assertIs(extractor.retina_parameters, retina)
        self.assertEqual(extractor.visual.retina.cone_ecc.size, 24)
        self.assertEqual(extractor.visual.retina.rod_ecc.size, 0)
        self.assertEqual(int(extractor.visual.retina.params.n_bins), 8)
        world = small_world(13)
        state = extractor.observe(world, world.agent, 0.05)
        self.assertEqual(state.vector.shape, (VISUAL_FEATURE_DIM,))
        self.assertTrue(np.all(np.isfinite(state.vector)))
        # n_bins=8 → 每维对应 1 个箱（而不是 96/16 档的 2 个箱）
        perception = extractor.last_perception()
        self.assertEqual(np.asarray(perception.retina.ganglion).size, 8)

    def test_high_fidelity_parameters_also_work(self) -> None:
        """高保真档（1800 视锥 / 32 箱）也能产出合法 16 维特征。"""
        extractor = VisualFeatureExtractor(
            seed=0,
            retina_parameters=RetinaParameters(),
            eye_parameters=EyeParameters(),
        )
        world = small_world(14)
        state = extractor.observe(world, world.agent, 0.05)
        self.assertEqual(state.vector.shape, (VISUAL_FEATURE_DIM,))
        self.assertTrue(np.all(np.isfinite(state.vector)))
        self.assertLessEqual(float(np.abs(state.vector).max()), 1.0)
        self.assertEqual(extractor.visual.retina.cone_ecc.size, 1800)

    def test_deterministic_for_same_seed_and_pose(self) -> None:
        """同一 seed + 同一世界 + 同一位姿 → 特征逐位一致。"""
        world = small_world(15)
        config = world.config
        x = config.width * 0.4
        first = pose(VisualFeatureExtractor(seed=15), world, x=x, facing=1)
        second = pose(VisualFeatureExtractor(seed=15), world, x=x, facing=1)
        np.testing.assert_allclose(first, second, atol=0.0)


class TestCost(unittest.TestCase):
    """成本可观测（不硬性断言 2.5 ms，避免 CI 抖动；bench 里有实测报告）。"""

    def test_cost_reported_and_sane(self) -> None:
        """每次 observe 都记录真实 cost_ms，且远低于"不可用"级别。"""
        world = small_world(16)
        extractor = VisualFeatureExtractor(seed=16)
        agent = world.agent
        costs = []
        for _ in range(20):
            world.step(0.05, 0)
            state = extractor.observe(world, agent, 0.05)
            self.assertTrue(math.isfinite(state.cost_ms))
            self.assertGreaterEqual(state.cost_ms, 0.0)
            costs.append(state.cost_ms)
        median = sorted(costs)[len(costs) // 2]
        # 极宽松上限（只防数量级回归；目标 2.5 ms 由 bench 实测报告）
        self.assertLess(median, 30.0, f"median observe cost {median:.2f} ms")


class TestNearPointGuard(unittest.TestCase):
    """眼睛嵌入几何体时的病理帧保护（对焦距离下限 = 人眼近点）。"""

    def test_near_point_scene_clamps_only_pathological_hits(self) -> None:
        """``_NearPointScene`` 只抬近距命中，远处与 inf 原样保留。"""
        from newworld.optics import Panorama
        from newworld.visual_features import NEAR_POINT_M, _NearPointScene

        class FakeScene:
            def panorama(self, x, y, gaze_deg, fov_deg, samples_per_deg=120.0):
                rel = np.linspace(-fov_deg / 2.0, fov_deg / 2.0, 5)
                radiance = np.ones((5, 3), dtype=np.float32)
                distances = np.array([0.001, 50.0, 500.0, np.inf, 0.02])
                return Panorama(rel, radiance, distances)

        floor = NEAR_POINT_M / 0.01  # 0.1 m @ 0.01 m/px = 10 px
        guard = _NearPointScene(FakeScene(), floor)
        guarded = guard.panorama(0.0, 0.0, 0.0, 160.0, 12.0)
        self.assertAlmostEqual(float(guarded.distance_px[0]), floor)
        self.assertAlmostEqual(float(guarded.distance_px[4]), floor)
        self.assertEqual(float(guarded.distance_px[1]), 50.0)
        self.assertEqual(float(guarded.distance_px[2]), 500.0)
        self.assertTrue(math.isinf(float(guarded.distance_px[3])))
        # 正常场景（全部 > 近点）直接返回原对象，零拷贝
        normal = _NearPointScene(
            type(
                "Normal",
                (),
                {
                    "panorama": lambda self, x, y, gaze_deg, fov_deg, samples_per_deg=120.0: (
                        Panorama(
                            np.linspace(-80, 80, 5),
                            np.ones((5, 3), np.float32),
                            np.full(5, 100.0),
                        )
                    )
                },
            )(),
            floor,
        ).panorama(0.0, 0.0, 0.0, 160.0, 12.0)
        self.assertTrue(np.all(normal.distance_px == 100.0))

    def test_eye_embedded_in_geometry_stays_bounded(self) -> None:
        """把眼睛包在一块大暗色几何体里：模糊 σ 有界、特征仍有限（不爆 FFT）。"""
        world = small_world(17)
        extractor = VisualFeatureExtractor(seed=17)
        agent = world.agent
        agent.x = world.config.width * 0.5
        eye_y = 322.0
        # 覆盖眼睛位置的大箱子（光线从内部出发 → 命中距离≈0 的病态几何）
        world.add_object(
            Box(agent.x - 200.0, eye_y - 120.0, agent.x + 200.0, eye_y + 120.0,
                albedo=(0.05, 0.05, 0.05))
        )
        state = extractor.observe(world, agent, 0.05)
        self.assertEqual(state.source, "retina")
        self.assertTrue(np.all(np.isfinite(state.vector)))
        self.assertLessEqual(float(np.abs(state.vector).max()), 1.0)
        perception = extractor.last_perception()
        # 无保护时实测出现过 3415 deg（单帧 130-170 ms）；有保护应远小于此
        self.assertLess(perception.blur_sigma_deg, 20.0)
        self.assertGreater(extractor.near_point_px, 1.0)


class TestBenchmarkScript(unittest.TestCase):
    """``tools/bench_vision_features.py`` 必须能真实跑起来并输出实测数字。"""

    def test_benchmark_runs_and_reports_real_parameters(self) -> None:
        """子进程真实运行 benchmark（小 runs）→ exit 0 + JSON 里参数/耗时真实。"""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bench.json"
            command = [
                sys.executable,
                str(REPO_ROOT / "tools" / "bench_vision_features.py"),
                "--width",
                "640",
                "--height",
                "400",
                "--runs",
                "6",
                "--poses",
                "8",
                "--profile-runs",
                "3",
                "--json",
                str(out),
            ]
            env = dict(os.environ)
            env.setdefault("NEWWORLD_SIZE", "640x400")
            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",       # 子进程 stdout 已按 UTF-8 输出
                errors="replace",
                timeout=600,
                env=env,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
            self.assertTrue(out.exists())
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["feature_dim"], VISUAL_FEATURE_DIM)
        self.assertGreater(payload["runs"], 0)
        self.assertGreaterEqual(payload["mean_ms"], 0.0)
        self.assertTrue(math.isfinite(payload["median_ms"]))
        low = payload["parameters"]["low_resolution"]
        self.assertEqual(low["retina"]["n_cones"], 96)
        self.assertEqual(low["retina"]["n_rods"], 0)
        self.assertEqual(low["retina"]["n_bins"], 16)
        self.assertTrue(low["visual_system_params_match"])
        self.assertGreaterEqual(payload["pose_dims_with_variance"], 6)
        self.assertGreater(payload["pose_std_max"], 0.01)
        self.assertIn("stages", payload)
        self.assertGreater(payload["sound_probe"]["speech_hearing_feature"], 0.5)


if __name__ == "__main__":
    unittest.main()
