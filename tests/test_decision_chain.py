"""决策链路验证测试（本轮升级 P2）。

覆盖任务书 §4.5 的 5 项决策验证 + §7.1 决策指标：
1. 改变环境/初始资源后决策是否变化；
2. 关闭神经网络（低层 random/off）或把高层目标策略换成随机后，表现是否可测变化；
3. 关闭规划器后的变化（baseline vs planning）；
4. 未见过的初始布局/配置（不同 seed/world size）仍可运行；
5. 新增 16 维文明上下文特征是否真正进入训练数据（非零方差）、是否影响模型输出；
另外验证决策来源计量、目标策略挂载/转移、创新接口不绕过世界规则。
"""

from __future__ import annotations

import unittest

import numpy as np

from newworld.config import CivilizationConfig, make_config
from newworld.neural import FEATURE_DIM, LEGACY_FEATURE_DIM, NeuralPolicy, features
from newworld.population import Population
from newworld.simulation import RunSpec, SocietyRunner
from newworld.training import collect_dataset
from newworld.world import World


def make_population(**civ) -> tuple[World, Population]:
    config = make_config(480, 300)
    cfg = CivilizationConfig(seed=0, enable_perception=False, **civ)
    world = World(config, seed=0, civ_config=cfg)
    world.reset(seed=0)
    return world, Population(world, seed=0)


class TestFeaturePipeline(unittest.TestCase):
    def test_live_observations_have_varying_context(self) -> None:
        world, population = make_population()
        rows = []
        for _ in range(400):
            population.step(0.05)
            for agent in world.agents:
                rows.append(features(world, character=agent))
        array = np.asarray(rows)
        context = array[:, LEGACY_FEATURE_DIM:]
        varying = int((context.std(axis=0) > 1e-9).sum())
        self.assertGreaterEqual(varying, 8, "真实运行中文明上下文特征必须有变化")
        self.assertTrue(np.all(np.isfinite(array)))

    def test_training_data_context_is_not_dead(self) -> None:
        """修复前：BC 训练数据 16/16 维标准差为 0（死特征）。"""
        data = collect_dataset(seed=0, steps_per_hour=80, hours=(9.0, 21.0))
        context = data["x"][:, LEGACY_FEATURE_DIM:]
        varying = int((context.std(axis=0) > 1e-9).sum())
        self.assertGreaterEqual(varying, 6, "训练数据里必须有真实变化的上下文特征")
        self.assertEqual(data["x"].shape[1], FEATURE_DIM)

    def test_context_affects_policy_output_after_training(self) -> None:
        """训练后（上下文权重非零）改变上下文特征，输出概率必须变化。"""
        policy = NeuralPolicy(seed=3)
        base = np.zeros(FEATURE_DIM)
        prob_before = policy.forward(base.reshape(1, -1))[0]
        policy.w1[LEGACY_FEATURE_DIM:] += 0.5  # simulate learned context weights
        changed = base.copy()
        changed[LEGACY_FEATURE_DIM:] = 1.0
        prob_after = policy.forward(changed.reshape(1, -1))[0]
        self.assertGreater(float(np.abs(prob_after - prob_before).max()), 1e-6)


class TestDecisionSources(unittest.TestCase):
    def test_neural_policy_takes_meaningful_share(self) -> None:
        """修复前实测 neural 只占 0.3% 的逐帧决策。"""
        world, population = make_population()
        for _ in range(3000):
            population.step(0.05)
            if not world.agents:
                break
        share = population.decision_share()
        self.assertGreater(share.get("neural", 0.0), 0.05, f"决策来源：{share}")
        self.assertLessEqual(sum(share.values()), 1.0001)

    def test_low_level_mode_changes_behaviour(self) -> None:
        """关闭/随机化底层网络后，决策分布与结果必须出现可测差异。"""
        shares = {}
        techs = {}
        for mode in ("full", "off"):
            runner = SocietyRunner(
                RunSpec(group="C", seed=1, steps=800, dt=0.05, width=480, height=300,
                        low_level_mode=mode)
            )
            result = runner.run()
            shares[mode] = result["decisions"]
            techs[mode] = result["metrics"]["tech_verified"]
        self.assertIn("lowlevel_off", shares["off"])
        self.assertNotIn("lowlevel_off", shares["full"])
        self.assertNotEqual(shares["full"], shares["off"])
        self.assertNotEqual(techs["full"], techs["off"])

    def test_planning_off_changes_behaviour(self) -> None:
        baseline = SocietyRunner(
            RunSpec(group="A", seed=2, steps=600, dt=0.05, width=480, height=300)
        ).run()
        planning = SocietyRunner(
            RunSpec(group="C", seed=2, steps=600, dt=0.05, width=480, height=300)
        ).run()
        self.assertEqual(baseline["metrics"]["goal_completion_rate"], 0.0)
        self.assertGreater(planning["metrics"]["goal_completion_rate"], 0.5)
        self.assertNotEqual(baseline["decisions"], planning["decisions"])


class TestGoalPolicyIntegration(unittest.TestCase):
    class _RandomPolicy:
        def __init__(self) -> None:
            self.rng = np.random.default_rng(0)

        def sample(self, features, rng=None):  # noqa: ANN001
            return int(self.rng.integers(0, 8))

    def test_policy_attach_and_transitions(self) -> None:
        world, population = make_population()
        population.attach_goal_policy(self._RandomPolicy())
        for _ in range(1500):
            population.step(0.05)
            if not world.agents:
                break
        transitions = population.drain_policy_transitions()
        self.assertGreater(len(transitions), 0)
        for feats, action, reward in transitions:
            self.assertEqual(np.asarray(feats).shape, (30,))
            self.assertIn(int(action), range(8))
            self.assertTrue(np.isfinite(float(reward)))
        self.assertEqual(population.drain_policy_transitions(), [])
        # cap still enforced with a policy attached
        self.assertLessEqual(world.registry.total_created, 8)

    def test_attach_is_idempotent_and_detachable(self) -> None:
        world, population = make_population()
        policy = self._RandomPolicy()
        population.attach_goal_policy(policy)
        population.attach_goal_policy(policy)
        self.assertIs(population.goal_policy, policy)
        population.detach_goal_policy()
        self.assertIsNone(population.goal_policy)

    def test_neural_and_random_policy_modes_run(self) -> None:
        for mode in ("heuristic", "random"):
            result = SocietyRunner(
                RunSpec(group="C", seed=3, steps=500, dt=0.05, width=480, height=300,
                        goal_policy_mode=mode)
            ).run()
            self.assertLessEqual(result["metrics"]["total_agents_created"], 8)
            self.assertIn("decisions", result)


class TestUnseenLayoutsAndCap(unittest.TestCase):
    def test_unseen_seeds_and_sizes_run(self) -> None:
        for seed, size in ((101, (480, 300)), (202, (640, 400))):
            result = SocietyRunner(
                RunSpec(group="C", seed=seed, steps=500, dt=0.05,
                        width=size[0], height=size[1])
            ).run()
            self.assertLessEqual(result["metrics"]["total_agents_created"], 8)
            self.assertGreater(result["metrics"]["total_agents_created"], 0)

    def test_cap_never_exceeded_under_policy_and_innovation(self) -> None:
        result = SocietyRunner(
            RunSpec(group="C", seed=5, steps=1200, dt=0.05, width=480, height=300,
                    goal_policy_mode="random", enable_innovation=True)
        ).run()
        created = result["metrics"]["total_agents_created"]
        self.assertLessEqual(created, 8, "累计创建绝不能超过 8")
        self.assertGreaterEqual(created, 1)
        self.assertLessEqual(len(result["extra_events"]["creation_denied"]), 8)


if __name__ == "__main__":
    unittest.main()
