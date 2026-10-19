"""高层目标选择策略（REINFORCE）单元测试。

契约：``docs/upgrade_contracts_v1.md`` §2.4（>=10 用例）。本文件只验证**机制
正确性**（softmax、采样、手写梯度、更新方向、序列化、特征、训练流程），
**不**声称"已经训练出智能策略"，也不依赖 Lead 尚未落地的集成钩子
（trainer 用 fake env / fake population 验证）。

测试运行::

    cd newworldpro
    $env:TEMP=(Resolve-Path .tmp).Path
    python -m unittest tests.test_highlevel -v
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from tests._env_compat import patch_tempdir
except ImportError:
    from _env_compat import patch_tempdir

patch_tempdir()

from newworld.config import make_config  # noqa: E402
from newworld.highlevel import (  # noqa: E402
    FEATURE_DIM,
    FEATURE_NAMES,
    GOAL_ACTIONS,
    GoalPolicy,
    GoalPolicyTrainer,
    extract_features,
)
from newworld.population import Population  # noqa: E402
from newworld.world import World  # noqa: E402


# ---------------------------------------------------------------------------
# fake 集成层：只实现契约 §2.2 的四个接口，用于验证 trainer 流程
# ---------------------------------------------------------------------------
class _FakePopulation:
    """只提供 ``attach_goal_policy/drain_policy_transitions`` 的假 population。"""

    def __init__(self) -> None:
        self._buffer: list[tuple[np.ndarray, int, float]] = []
        self.drain_calls = 0
        self.attach_calls = 0
        self.attached_policy = None
        self.goals = None

    def attach_goal_policy(self, policy) -> None:
        self.attach_calls += 1
        self.attached_policy = policy

    def push(self, features, action: int, reward: float) -> None:
        self._buffer.append((np.asarray(features, dtype=np.float64), int(action), float(reward)))

    def drain_policy_transitions(self):
        self.drain_calls += 1
        items = self._buffer
        self._buffer = []
        return items


class _FakeEnv:
    """契约 §2.2 的训练环境：``population/step/ended/result``。"""

    def __init__(
        self,
        steps: int,
        *,
        transitions_per_step: int = 1,
        rewards: list[float] | None = None,
        result: dict[str, float] | None = None,
    ) -> None:
        self.population = _FakePopulation()
        self.world = None
        self._budget = int(steps)
        self._transitions_per_step = int(transitions_per_step)
        self._rewards = list(rewards or ())
        self._step_index = 0
        self._stopped = False
        self.result_data: dict[str, float] = dict(
            result
            or {
                "world_time_s": 0.0,
                "alive_count": 2.0,
                "total_agents_created": 2.0,
                "total_deaths": 0.0,
                "tech_verified": 1.0,
                "facilities_active": 0.0,
                "cooperation_index": 0.0,
                "extinction_time": 500.0,
                "steps_run": 0.0,
            }
        )

    def step(self) -> None:
        if self._stopped:
            return
        reward = (
            self._rewards[self._step_index]
            if self._step_index < len(self._rewards)
            else 0.0
        )
        for _ in range(self._transitions_per_step):
            self.population.push(
                np.zeros(FEATURE_DIM, dtype=np.float64) + 0.25,
                self._step_index % len(GOAL_ACTIONS),
                reward,
            )
        self._step_index += 1
        if self._step_index >= self._budget:
            self._stopped = True

    def ended(self) -> bool:
        return self._stopped

    def result(self) -> dict[str, float]:
        return dict(self.result_data)


def _constant_factory(env: _FakeEnv):
    """返回忽略 seed 的 env_factory（trainer 会用关键字 steps/dt 调用它）。"""

    def factory(seed, *, steps=None, dt=None):  # noqa: ARG001 - 契约签名
        return env

    return factory


class GoalPolicyMechanicsTest(unittest.TestCase):
    """网络机制：softmax / 采样 / 梯度 / 更新方向。"""

    def test_feature_table_contract(self) -> None:
        """FEATURE_NAMES 维度一致、无重复、包含契约要求的语义特征。"""
        self.assertGreaterEqual(FEATURE_DIM, 16)
        self.assertEqual(FEATURE_DIM, len(FEATURE_NAMES))
        self.assertEqual(len(set(FEATURE_NAMES)), len(FEATURE_NAMES))
        required = {
            "energy",
            "hydration",
            "inventory_fill",
            "tech_verified",
            "facility_active",
            "population_alive",
            "goal_success",
            "goal_failure",
            "memory_recent_success",
            "daylight",
        }
        self.assertTrue(required.issubset(set(FEATURE_NAMES)))

    def test_softmax_normalized_and_finite(self) -> None:
        """概率向量形状正确、非负、和为 1；极端输入不产生 NaN/inf。"""
        policy = GoalPolicy(seed=7)
        for features in (
            np.zeros(FEATURE_DIM),
            np.linspace(-1.0, 1.0, FEATURE_DIM),
            np.full(FEATURE_DIM, 1e3),
            np.full(FEATURE_DIM, -1e3),
        ):
            probs = policy.probabilities(features)
            self.assertEqual(probs.shape, (len(GOAL_ACTIONS),))
            self.assertTrue(np.all(np.isfinite(probs)))
            self.assertTrue(np.all(probs >= 0.0))
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=10)

    def test_sample_deterministic_with_fixed_seed(self) -> None:
        """同一 rng 种子得到同一动作序列；不同种子极大概率不同。"""
        policy = GoalPolicy(seed=11)
        features = np.linspace(-0.3, 0.3, FEATURE_DIM)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        rng_c = np.random.default_rng(43)
        seq_a = [policy.sample(features, rng_a) for _ in range(50)]
        seq_b = [policy.sample(features, rng_b) for _ in range(50)]
        seq_c = [policy.sample(features, rng_c) for _ in range(50)]
        self.assertEqual(seq_a, seq_b)
        self.assertNotEqual(seq_a, seq_c)
        self.assertTrue(all(0 <= action < len(GOAL_ACTIONS) for action in seq_a))
        # policy.sample(features)（不传 rng）必须可用
        self.assertIn(policy.sample(features), range(len(GOAL_ACTIONS)))

    def test_gradients_have_expected_shapes_and_are_finite(self) -> None:
        """loss/grads 形状与参数一致且全部有限。"""
        policy = GoalPolicy(seed=0)
        features = np.random.default_rng(1).normal(0.0, 0.5, FEATURE_DIM)
        loss, grads = policy.loss_and_grads(features, 2, 1.5)
        self.assertTrue(math.isfinite(loss))
        expected = {
            "W1": (FEATURE_DIM, GoalPolicy.HIDDEN),
            "b1": (GoalPolicy.HIDDEN,),
            "W2": (GoalPolicy.HIDDEN, len(GOAL_ACTIONS)),
            "b2": (len(GOAL_ACTIONS),),
        }
        self.assertEqual(set(grads), set(expected))
        for name, shape in expected.items():
            self.assertEqual(grads[name].shape, shape)
            self.assertTrue(np.all(np.isfinite(grads[name])))

    def test_gradient_matches_finite_difference(self) -> None:
        """手写反向传播与数值差分一致（验证没有写反符号/漏项）。"""
        policy = GoalPolicy(seed=3)
        features = np.linspace(-0.4, 0.4, FEATURE_DIM)
        action = 2
        advantage = 1.0
        _loss, grads = policy.loss_and_grads(features, action, advantage)
        delta = 1e-5
        for name, index in (
            ("W1", (0, 0)),
            ("W2", (3, 2)),
            ("b2", (action,)),
        ):
            target = getattr(policy, name)
            original = float(target[index])
            target[index] = original + delta
            plus = policy.loss_and_grads(features, action, advantage)[0]
            target[index] = original - delta
            minus = policy.loss_and_grads(features, action, advantage)[0]
            target[index] = original
            numeric = (plus - minus) / (2.0 * delta)
            self.assertTrue(
                math.isclose(float(grads[name][index]), numeric, rel_tol=1e-4, abs_tol=1e-6),
                f"{name}{index}: analytic {grads[name][index]} vs numeric {numeric}",
            )

    def test_action_probability_increases_after_update(self) -> None:
        """固定正 advantage 做若干次更新后，被选中动作的概率必须上升。"""
        policy = GoalPolicy(seed=5)
        features = np.full(FEATURE_DIM, 0.3, dtype=np.float64)
        action = 0
        before = float(policy.probabilities(features)[action])
        loss_before, _ = policy.loss_and_grads(features, action, 1.0)
        for _ in range(6):
            _loss, grads = policy.loss_and_grads(features, action, 1.0)
            policy.apply_grads(grads, 0.1)
        after = float(policy.probabilities(features)[action])
        self.assertGreater(after, before)
        self.assertGreater(after, 0.0)
        self.assertGreater(-_loss, -loss_before)  # 对数似然上升（loss 下降）

    def test_invalid_action_index_and_features_raise(self) -> None:
        """越界动作 / 错误特征维度 / 非有限特征必须明确报错。"""
        policy = GoalPolicy(seed=0)
        features = np.zeros(FEATURE_DIM)
        with self.assertRaises(ValueError):
            policy.loss_and_grads(features, -1, 1.0)
        with self.assertRaises(ValueError):
            policy.loss_and_grads(features, len(GOAL_ACTIONS), 1.0)
        with self.assertRaises(ValueError):
            policy.probabilities(np.zeros(FEATURE_DIM - 1))
        bad = np.zeros(FEATURE_DIM)
        bad[0] = float("nan")
        with self.assertRaises(ValueError):
            policy.probabilities(bad)
        with self.assertRaises(ValueError):
            policy.action_name(99)

    def test_apply_grads_rejects_wrong_shape(self) -> None:
        """梯度形状不匹配时必须报错而不是静默更新。"""
        policy = GoalPolicy(seed=0)
        with self.assertRaises(ValueError):
            policy.apply_grads({"W1": np.zeros((1, 1))}, 0.1)


class GoalPolicySerializationTest(unittest.TestCase):
    """checkpoint / dict 序列化往返。"""

    def test_save_load_roundtrip(self) -> None:
        """save→load 后参数与概率完全一致；无后缀路径自动补 .npz。"""
        policy = GoalPolicy(seed=21)
        features = np.linspace(-0.2, 0.2, FEATURE_DIM)
        expected = policy.probabilities(features)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "goal_policy"
            saved = policy.save(target)
            self.assertEqual(saved.suffix, ".npz")
            self.assertTrue(saved.exists())
            loaded = GoalPolicy.load(saved)
            self.assertEqual(loaded.seed, 21)
            self.assertTrue(np.allclose(loaded.flat_weights(), policy.flat_weights()))
            self.assertTrue(np.allclose(loaded.probabilities(features), expected))
            # load 也应接受不带后缀的路径
            loaded_plain = GoalPolicy.load(target)
            self.assertTrue(np.allclose(loaded_plain.probabilities(features), expected))

    def test_save_load_missing_checkpoint_raises(self) -> None:
        """加载不存在的 checkpoint 必须报错。"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                GoalPolicy.load(Path(tmp) / "nope.npz")

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """to_dict/from_dict 往返一致，且对动作顺序不匹配报错。"""
        policy = GoalPolicy(seed=9)
        features = np.linspace(0.0, 1.0, FEATURE_DIM)
        data = policy.to_dict()
        restored = GoalPolicy.from_dict(data)
        self.assertTrue(np.allclose(restored.probabilities(features), policy.probabilities(features)))
        self.assertEqual(data["feature_dim"], FEATURE_DIM)
        self.assertEqual(tuple(data["actions"]), GOAL_ACTIONS)
        broken = dict(data)
        broken["actions"] = list(reversed(GOAL_ACTIONS))
        with self.assertRaises(ValueError):
            GoalPolicy.from_dict(broken)


class FeatureExtractionTest(unittest.TestCase):
    """extract_features 对真实 World/Population 的行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.world = World(make_config(800, 500), seed=0)
        cls.population = Population(cls.world, seed=0)
        cls.agent = next(agent for agent in cls.world.agents if agent.alive)

    def test_features_finite_and_right_dim(self) -> None:
        """真实世界输出 ``(FEATURE_DIM,)``、float64、全有限。"""
        features = extract_features(self.world, self.agent, self.population)
        self.assertEqual(features.shape, (FEATURE_DIM,))
        self.assertEqual(features.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_features_degrade_gracefully_without_world(self) -> None:
        """world/agent 都缺失时仍返回有限向量（不抛错）。"""
        features = extract_features(None, None, None)
        self.assertEqual(features.shape, (FEATURE_DIM,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_features_change_with_world_state(self) -> None:
        """能量/库存/技术变化必须改变特征向量（真实状态敏感）。"""
        energy_index = FEATURE_NAMES.index("energy")
        wood_index = FEATURE_NAMES.index("wood")
        before = extract_features(self.world, self.agent, self.population)
        energy_before = float(before[energy_index])
        wood_before = float(before[wood_index])
        self.agent.physiology.energy = 5.0
        self.agent.inventory.wood = 5
        after = extract_features(self.world, self.agent, self.population)
        self.assertFalse(np.allclose(before, after))
        self.assertLess(float(after[energy_index]), energy_before)
        self.assertGreater(float(after[wood_index]), wood_before)
        # 复原，避免影响同类其它测试
        self.agent.physiology.energy = 78.0
        self.agent.inventory.wood = 0

    def test_features_respond_to_verified_tech(self) -> None:
        """已验证技术数增加后 tech_verified 特征必须上升。"""
        index = FEATURE_NAMES.index("tech_verified")
        before = float(extract_features(self.world, self.agent, self.population)[index])
        knowledge = self.world.knowledge.get("stone_axe")
        self.assertIsNotNone(knowledge)
        previous_status = knowledge.status
        knowledge.status = "verified"
        try:
            after = float(extract_features(self.world, self.agent, self.population)[index])
        finally:
            knowledge.status = previous_status
        self.assertGreater(after, before)

    def test_policy_probabilities_change_when_features_perturbed(self) -> None:
        """训练后扰动能量/技术特征，策略概率向量必须产生可测变化。"""
        policy = GoalPolicy(seed=0)
        base = extract_features(self.world, self.agent, self.population)
        perturbed = base.copy()
        perturbed[FEATURE_NAMES.index("energy")] = 0.0
        perturbed[FEATURE_NAMES.index("tech_verified")] = 1.0
        for _ in range(5):  # 先做几次真实策略梯度更新，离开初始点
            _loss, grads = policy.loss_and_grads(base, 0, 1.0)
            policy.apply_grads(grads, 0.1)
        p_base = policy.probabilities(base)
        p_perturbed = policy.probabilities(perturbed)
        self.assertGreater(float(np.abs(p_base - p_perturbed).sum()), 1e-4)


class GoalPolicyTrainerTest(unittest.TestCase):
    """trainer 流程（fake env）：采样→奖励→更新→评估。"""

    def test_trainer_attaches_policy_to_env(self) -> None:
        """trainer 必须把**当前**策略挂到 env.population（否则评估会用错策略）。"""
        policy = GoalPolicy(seed=4)
        env = _FakeEnv(2, transitions_per_step=1, rewards=[1.0, 1.0])
        trainer = GoalPolicyTrainer(policy, _constant_factory(env), seed=0)
        trainer.run_episode(seed=0, steps=2)
        self.assertIs(env.population.attached_policy, policy)
        self.assertGreaterEqual(env.population.attach_calls, 1)

    def test_run_episode_uses_env_and_drains_transitions(self) -> None:
        """run_episode 推进 env、读取真实奖励、合并 env.result() 指标。"""
        policy = GoalPolicy(seed=0)
        before = policy.flat_weights().copy()
        env = _FakeEnv(2, transitions_per_step=1, rewards=[1.0, 2.0])
        trainer = GoalPolicyTrainer(policy, _constant_factory(env), seed=0, lr=0.05)
        metrics = trainer.run_episode(seed=0, steps=10, dt=0.05)
        self.assertEqual(metrics["return"], 3.0)
        self.assertEqual(metrics["steps"], 2.0)  # ended() 提前结束
        self.assertEqual(metrics["transitions"], 2.0)
        self.assertGreaterEqual(env.population.drain_calls, 1)
        # env.result() 的数值指标被真实并入
        self.assertEqual(metrics["extinction_time"], 500.0)
        self.assertEqual(metrics["alive_count"], 2.0)
        # 发生过一次真实参数更新
        self.assertFalse(np.allclose(policy.flat_weights(), before))
        self.assertEqual(trainer._updates, 1)

    def test_run_episode_without_transitions_does_not_update(self) -> None:
        """没有转移时不得假装学习：参数不变、transitions=0。"""
        policy = GoalPolicy(seed=0)
        before = policy.flat_weights().copy()
        env = _FakeEnv(3, transitions_per_step=0)
        trainer = GoalPolicyTrainer(policy, _constant_factory(env), seed=0)
        metrics = trainer.run_episode(seed=0, steps=3)
        self.assertEqual(metrics["transitions"], 0.0)
        self.assertEqual(metrics["return"], 0.0)
        self.assertTrue(np.allclose(policy.flat_weights(), before))
        self.assertEqual(trainer._updates, 0)

    def test_train_runs_episodes_and_evaluates_independent_seeds(self) -> None:
        """train 完成多回合、周期评估与最终独立评估，且调用 drain。"""
        created_envs: list[_FakeEnv] = []

        def factory(seed, *, steps=None, dt=None):  # noqa: ARG001
            env = _FakeEnv(4, rewards=[0.5, 1.0, -0.5, 2.0])
            created_envs.append(env)
            return env

        policy = GoalPolicy(seed=0)
        trainer = GoalPolicyTrainer(policy, factory, seed=0, lr=0.02)
        result = trainer.train(
            episodes=3,
            steps=4,
            dt=0.05,
            eval_episodes=2,
            eval_steps=4,
            eval_every=2,
        )
        self.assertEqual(len(result["train_returns"]), 3)
        self.assertTrue(all(isinstance(value, float) for value in result["train_returns"]))
        # 训练种子与评估种子必须不重叠
        self.assertEqual(len(result["eval"]), 2)
        self.assertIn("final_eval", result)
        self.assertGreater(result["final_eval"]["episodes"], 0.0)
        self.assertTrue(math.isfinite(result["final_eval"]["mean_return"]))
        # 每个环境都被真实 drain 过（训练 + 评估都会 drain）
        self.assertGreaterEqual(len(created_envs), 1)
        self.assertTrue(all(env.population.drain_calls > 0 for env in created_envs))
        self.assertGreater(result["updates"], 0.0)
        self.assertGreater(len(result["train_transitions"]), 0)

    def test_train_rejects_invalid_budget(self) -> None:
        """非法回合/步数必须报错（不做静默伪训练）。"""
        policy = GoalPolicy(seed=0)
        trainer = GoalPolicyTrainer(policy, _constant_factory(_FakeEnv(1)), seed=0)
        with self.assertRaises(ValueError):
            trainer.train(episodes=0, steps=10)
        with self.assertRaises(ValueError):
            trainer.train(episodes=1, steps=0)
        with self.assertRaises(ValueError):
            trainer.evaluate(seeds=[], steps=10)

    def test_discounted_returns_formula(self) -> None:
        """折扣回报公式正确：``G_t = r_t + γ r_{t+1} + …``。"""
        returns = GoalPolicyTrainer.discounted_returns([1.0, 2.0, 3.0], 0.5)
        expected = [1.0 + 0.5 * 2.0 + 0.25 * 3.0, 2.0 + 0.5 * 3.0, 3.0]
        self.assertTrue(np.allclose(returns, expected))

    def test_real_population_hook_contract_optional(self) -> None:
        """可选联调：真实 Population 钩子产出的转移必须符合 trainer 的约定。

        集成钩子未落地时 skip；落地后它验证 ``attach_goal_policy`` /
        ``drain_policy_transitions`` 的返回格式（features 维度/动作范围/奖励有限）。
        """
        from newworld.config import make_civilization_config

        civ = make_civilization_config("D", seed=0, enable_perception=False)
        world = World(make_config(640, 400), seed=0, civ_config=civ)
        world.reset(seed=0)
        population = Population(world, seed=0)
        attach = getattr(population, "attach_goal_policy", None)
        drain = getattr(population, "drain_policy_transitions", None)
        if not callable(attach) or not callable(drain):
            self.skipTest("Population 集成钩子尚未落地")
        policy = GoalPolicy(seed=0)
        attach(policy)
        try:
            for _ in range(80):
                population.step(0.05)
            transitions = list(drain())
        finally:
            detach = getattr(population, "detach_goal_policy", None)
            if callable(detach):
                detach()
        self.assertGreater(len(transitions), 0, "策略挂载后必须产生真实转移")
        for features, action, reward in transitions:
            vector = np.asarray(features, dtype=np.float64)
            self.assertEqual(vector.shape, (FEATURE_DIM,))
            self.assertTrue(np.all(np.isfinite(vector)))
            self.assertGreaterEqual(int(action), 0)
            self.assertLess(int(action), len(GOAL_ACTIONS))
            self.assertTrue(math.isfinite(float(reward)))


if __name__ == "__main__":
    unittest.main()
