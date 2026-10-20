"""V2 集成测试：无限世界、低层 RL 转移、视觉特征进入观测。

这些测试把三条并行工作流的产物接到主干上验证（不是模块单测）：
- ``InfiniteScene`` 通过 ``World`` 真正驱动仿真，窗口平移时动态坐标一起平移；
- ``Population`` 在 ``low_level_mode="rl"`` 下采样策略动作并记录真实奖励转移；
- 16 维视网膜特征经 ``Population`` 写入后确实进入 56 维神经观测。
"""

from __future__ import annotations

import unittest

import numpy as np

from newworld.config import CivilizationConfig, make_config
from newworld.neural import FEATURE_DIM, V2_FEATURE_DIM, features
from newworld.population import Population
from newworld.simulation import RunSpec, SocietyRunner
from newworld.world import World

try:  # "python -m unittest tests.test_v2_integration"
    from tests._env_compat import patch_tempdir
except ImportError:  # "python -m unittest discover -s tests"
    from _env_compat import patch_tempdir
patch_tempdir()


class _FakeSample:
    """伪造的 ActionSample（只含 Population 解码需要的字段）。"""

    def __init__(self, move: int = 2, jump: bool = False, speak: bool = False) -> None:
        self.mode = 0
        self.move = move
        self.jump = jump
        self.speak = speak
        self.logprob = 0.0


class _FakeLowLevelPolicy:
    """测试用的最小策略：固定采样给定动作，并记录输入特征维度。"""

    def __init__(self, move: int = 2) -> None:
        self.move = move
        self.calls = 0
        self.feature_dims: list[int] = []

    def sample(self, features: np.ndarray, rng=None) -> _FakeSample:  # noqa: ANN001
        self.calls += 1
        self.feature_dims.append(int(np.asarray(features).size))
        return _FakeSample(move=self.move)


class TestInfiniteWorldIntegration(unittest.TestCase):
    def test_infinite_world_uses_windowed_scene(self) -> None:
        config = make_config(480, 300, infinite=True, window_atoms=96)
        world = World(config, seed=0)
        self.assertEqual(type(world.pixel_scene).__name__, "InfiniteScene")
        self.assertEqual(world.infinite_fallback_reason, "")
        state = world.state()
        self.assertTrue(state["infinite"])
        self.assertEqual(state["window_origin_px"], 0.0)

    def test_finite_world_is_still_pixel_scene(self) -> None:
        world = World(make_config(480, 300), seed=0)
        self.assertEqual(type(world.pixel_scene).__name__, "PixelScene")
        self.assertFalse(world.state()["infinite"])

    def test_infinite_window_recenters_and_keeps_agents_inside(self) -> None:
        runner = SocietyRunner(
            RunSpec(
                group="C",
                seed=7,
                steps=60,
                dt=0.05,
                width=480,
                height=300,
                infinite=True,
                window_atoms=96,
            )
        )
        world = runner.world
        # 主动把人口推到窗口边缘，触发一次真实平移（不依赖随机漫步）
        for agent in world.agents:
            agent.x = 6.0
        before = {agent.agent_id: agent.x for agent in world.agents}
        for _ in range(5):
            runner.population.step(0.05)
        self.assertNotEqual(world.window_origin_px, 0.0)  # 可向负方向（左）延伸
        self.assertEqual(world.infinite_fallback_reason, "")
        for agent in world.agents:
            self.assertGreaterEqual(agent.x, 0.0)
            self.assertLessEqual(agent.x, world.config.width)
        # 窗口平移后，动态坐标应整体反向平移同一距离（局部坐标体系一致）
        expected_shift = -world.window_origin_px
        for agent in world.agents:
            if agent.agent_id in before:
                self.assertAlmostEqual(
                    agent.x, before[agent.agent_id] + expected_shift, delta=100.0
                )

    def test_infinite_render_matches_screen_size(self) -> None:
        from newworld.render import render_world

        runner = SocietyRunner(
            RunSpec(group="C", seed=3, steps=40, dt=0.05, width=480, height=300,
                    infinite=True, window_atoms=96)
        )
        runner.run()
        image = render_world(runner.world)
        self.assertEqual(image.size, (480, 300))

    def test_population_cap_override(self) -> None:
        runner = SocietyRunner(
            RunSpec(
                group="C",
                seed=11,
                steps=200,
                dt=0.05,
                width=480,
                height=300,
                max_agents_created=16,
            )
        )
        result = runner.run()
        self.assertEqual(runner.civ_config.max_agents_created, 16)
        self.assertLessEqual(result["civilization"]["total_agents_created"], 16)


class TestLowLevelRLIntegration(unittest.TestCase):
    def test_rl_mode_samples_policy_and_records_transitions(self) -> None:
        runner = SocietyRunner(
            RunSpec(
                group="C",
                seed=5,
                steps=240,
                dt=0.05,
                width=480,
                height=300,
                low_level_mode="rl",
                visual_features=False,
            )
        )
        policy = _FakeLowLevelPolicy(move=2)
        runner.population.attach_lowlevel_policy(policy)
        result = runner.run()
        transitions = runner.population.drain_lowlevel_transitions()
        self.assertGreater(policy.calls, 0)
        self.assertEqual(set(policy.feature_dims), {FEATURE_DIM})
        self.assertGreater(len(transitions), 0)
        self.assertEqual(result["lowlevel"]["mode"], "rl")
        for agent_id, feats, action, reward in transitions:
            self.assertIsInstance(agent_id, int)
            self.assertEqual(np.asarray(feats).shape, (FEATURE_DIM,))
            self.assertTrue(np.all(np.isfinite(feats)))
            self.assertIsInstance(float(reward), float)
            self.assertTrue(np.isfinite(float(reward)))
            self.assertEqual(int(action.move), 2)
        self.assertGreater(runner.population.decision_counts["lowlevel_rl"], 0)

    def test_rl_mode_without_policy_falls_back_to_bc(self) -> None:
        runner = SocietyRunner(
            RunSpec(
                group="C",
                seed=5,
                steps=120,
                dt=0.05,
                width=480,
                height=300,
                low_level_mode="rl",
                lowlevel_policy_path=".tmp/definitely_missing_checkpoint.npz",
                visual_features=False,
            )
        )
        runner.run()
        self.assertEqual(runner.population.drain_lowlevel_transitions(), [])
        self.assertEqual(runner.population.lowlevel_summary()["transitions"], 0.0)

    def test_rl_move_index_is_decoded_with_neural_semantics(self) -> None:
        """move 头下标 2 应解码为向左（-1）、0 应解码为向右（+1）。"""
        runner = SocietyRunner(
            RunSpec(group="C", seed=5, steps=10, dt=0.05, width=480, height=300,
                    low_level_mode="rl", visual_features=False)
        )
        first = runner.world.agents[0]
        mind = runner.population.minds[first.agent_id]
        for move_index, expected in ((2, -1), (0, 1), (1, 0)):
            runner.population.attach_lowlevel_policy(_FakeLowLevelPolicy(move=move_index))
            decision = runner.population._rl_low_level_decision(first, mind, 0.05)
            self.assertEqual(decision.move, expected)
        self.assertGreater(runner.population.decision_counts["lowlevel_rl"], 0)


class TestVisualFeaturesInObservation(unittest.TestCase):
    def test_population_refreshes_visual_features(self) -> None:
        config = make_config(480, 300)
        civ = CivilizationConfig(
            seed=0, enable_perception=False, enable_visual_features=True, perception_stride=2
        )
        world = World(config, seed=0, civ_config=civ)
        world.reset(seed=0)
        population = Population(world, seed=0)
        for _ in range(20):
            population.step(0.05)
        agents = [a for a in world.agents if a.alive]
        self.assertTrue(agents)
        self.assertIsNotNone(agents[0].visual_features)
        self.assertEqual(len(agents[0].visual_features), 16)
        for value in agents[0].visual_features:
            self.assertTrue(np.isfinite(value))
            self.assertLessEqual(abs(value), 1.0)

    def test_visual_features_reach_neural_observation(self) -> None:
        config = make_config(480, 300)
        civ = CivilizationConfig(
            seed=0, enable_perception=False, enable_visual_features=True, perception_stride=1
        )
        world = World(config, seed=0, civ_config=civ)
        world.reset(seed=0)
        agent = world.agent
        agent.visual_features = tuple([0.25] * 16)
        vector = features(world, character=agent)
        self.assertEqual(vector.shape, (FEATURE_DIM,))
        np.testing.assert_allclose(vector[V2_FEATURE_DIM:], 0.25)


if __name__ == "__main__":
    unittest.main()

class TestReproducibilityGuards(unittest.TestCase):
    """V2 修复：视觉列不被随机变异；社会网络默认不落盘（实验 run 独立）。"""

    def test_vision_toggle_is_behaviour_neutral_with_zero_visual_weights(self) -> None:
        def run(visual: bool) -> tuple:
            runner = SocietyRunner(
                RunSpec(
                    group="C",
                    seed=202,
                    steps=300,
                    dt=0.05,
                    width=640,
                    height=400,
                    perception=False,
                    visual_features=visual,
                    perception_stride=8,
                )
            )
            result = runner.run()
            return (
                result["metrics"]["tech_verified"],
                result["metrics"]["repeat_run_max"],
                tuple(sorted(result["decisions"].items())),
                result["civilization"]["alive_count"],
            )

        self.assertEqual(run(False), run(True))

    def test_inherited_policy_does_not_mutate_visual_columns(self) -> None:
        runner = SocietyRunner(
            RunSpec(group="C", seed=202, steps=20, dt=0.05, width=480, height=300)
        )
        population = runner.population
        parents = runner.world.agents[:2]
        child = type(parents[0])(
            x=100.0, agent_id=99, parents=tuple(agent.agent_id for agent in parents)
        )
        policy = population._inherit_policy(child)
        self.assertTrue(np.all(policy.w1[V2_FEATURE_DIM:] == 0.0))

    def test_society_network_is_not_persisted_by_default(self) -> None:
        runner = SocietyRunner(
            RunSpec(group="C", seed=202, steps=10, dt=0.05, width=480, height=300)
        )
        self.assertFalse(runner.population.persist_society)
        self.assertIsNotNone(runner.population.society.network)
