"""低层动作策略（REINFORCE）单元测试。

契约：``docs/upgrade_contracts_v2.md`` §2.4（≥10 用例）。本文件只验证**机制
正确性**（softmax/BCE 对数概率、采样、手写反向与有限差分一致、更新方向、
序列化、语言头不参与 RL、trainer 全流程），**不**声称"训练出了更好的策略"；
也不依赖 Lead 尚未落地的集成钩子（trainer 用 fake env 验证；真实钩子只在
存在时做可选契约检查，否则 skip）。

测试运行::

    cd newworldpro
    $env:TEMP=(Resolve-Path .tmp).Path
    python -m unittest tests.test_lowlevel_rl -v
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

from newworld.lowlevel_rl import (  # noqa: E402
    JUMP_INDEX,
    LANG_START,
    MODE_NAMES,
    MOVE_DIM,
    MOVE_VALUES,
    ActionSample,
    LowLevelPolicy,
    LowLevelTrainer,
)
from newworld.neural import FEATURE_DIM, MODES, NeuralPolicy  # noqa: E402


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------
def make_policy(seed: int = 0, *, zero: bool = False) -> LowLevelPolicy:
    """构造一个包装真实 ``NeuralPolicy`` 的低层策略。"""
    neural = NeuralPolicy(seed=seed)
    if zero:
        neural = NeuralPolicy(weights=np.zeros(neural.flat_weights().size))
    return LowLevelPolicy(neural, seed=seed)


def policy_feature_dim(policy: LowLevelPolicy) -> int:
    return int(policy.feature_dim)


# ---------------------------------------------------------------------------
# fake 集成层：只实现契约 §2 的 env 接口，用于验证 trainer 流程
# ---------------------------------------------------------------------------
class _FakePopulation:
    """只提供 ``attach_lowlevel_policy/drain_lowlevel_transitions`` 的假 population。"""

    def __init__(self) -> None:
        self.buffer: list[tuple] = []
        self.attach_calls = 0
        self.attached_policy = None
        self.drain_calls = 0

    def attach_lowlevel_policy(self, policy) -> None:
        self.attach_calls += 1
        self.attached_policy = policy

    def drain_lowlevel_transitions(self):
        self.drain_calls += 1
        items = self.buffer
        self.buffer = []
        return items


class _FakeEnv:
    """契约 §2 的低层训练环境：``population/step/ended/result``。"""

    def __init__(
        self,
        feature_dim: int,
        *,
        budget: int = 4,
        agents: tuple[int, ...] = (0,),
        reward_fn=None,
        result: dict[str, float] | None = None,
    ) -> None:
        self.population = _FakePopulation()
        self.world = None
        self._feature_dim = int(feature_dim)
        self._budget = int(budget)
        self._agents = tuple(int(agent) for agent in agents)
        self._reward_fn = reward_fn
        self._step_index = 0
        self._stopped = False
        self.result_data: dict[str, float] = dict(
            result
            or {
                "world_time_s": 0.0,
                "alive_count": 2.0,
                "tech_verified": 1.0,
                "facilities_active": 0.0,
            }
        )

    def step(self) -> None:
        if self._stopped:
            return
        step = self._step_index
        for agent_id in self._agents:
            features = np.full(
                self._feature_dim, 0.1 * step + 0.01 * agent_id, dtype=np.float64
            )
            action = ActionSample(
                mode=step % len(MODE_NAMES),
                move=step % MOVE_DIM,
                jump=bool(step % 2),
                speak=bool((step + 1) % 2),
            )
            if self._reward_fn is not None:
                reward = float(self._reward_fn(step, agent_id))
            else:
                reward = 0.1 * (step + 1)
            self.population.buffer.append((int(agent_id), features, action, reward))
        self._step_index += 1
        if self._step_index >= self._budget:
            self._stopped = True

    def ended(self) -> bool:
        return self._stopped

    def result(self) -> dict[str, float]:
        data = dict(self.result_data)
        data["steps_run"] = float(self._step_index)
        return data


def _constant_factory(env: _FakeEnv):
    def factory(seed, *, steps=None, dt=None):  # noqa: ARG001 - 契约签名
        return env

    return factory


class LowLevelPolicyMechanicsTest(unittest.TestCase):
    """采样 / 对数概率 / 手写梯度 / 更新方向。"""

    def test_feature_dim_and_sample_legality(self) -> None:
        """真实 NeuralPolicy 上采样动作合法、维度自适应、decode_move 正确。"""
        policy = make_policy(seed=0)
        features = np.zeros(policy_feature_dim(policy), dtype=np.float64)
        action = policy.sample(features)
        self.assertIsInstance(action, ActionSample)
        self.assertIn(action.mode, range(len(MODE_NAMES)))
        self.assertIn(action.move, range(MOVE_DIM))
        self.assertIsInstance(action.jump, bool)
        self.assertIsInstance(action.speak, bool)
        self.assertTrue(math.isfinite(action.logprob))
        self.assertEqual(policy.decode_move(0), 1)
        self.assertEqual(policy.decode_move(1), 0)
        self.assertEqual(policy.decode_move(2), -1)
        self.assertEqual(tuple(MOVE_VALUES), (1, 0, -1))
        with self.assertRaises(ValueError):
            policy.decode_move(3)

    def test_logprob_matches_manual_zero_weights(self) -> None:
        """零权重网络 logits=0 → 各头均匀；联合对数概率可手算验证。"""
        policy = make_policy(seed=0, zero=True)
        features = np.zeros(policy_feature_dim(policy), dtype=np.float64)
        expected = (
            math.log(1.0 / len(MODE_NAMES))
            + math.log(1.0 / MOVE_DIM)
            + math.log(0.5)
            + math.log(0.5)
        )
        for action in (
            ActionSample(mode=0, move=0, jump=False, speak=False),
            ActionSample(mode=2, move=1, jump=True, speak=False),
            ActionSample(mode=4, move=2, jump=False, speak=True),
        ):
            self.assertAlmostEqual(policy.logprob(features, action), expected, places=9)
            self.assertAlmostEqual(
                policy.probability(features, action), math.exp(expected), places=9
            )
        # 确定性解码在零权重下也必须给出合法动作与有限 logprob
        deterministic = policy.deterministic(features)
        self.assertTrue(math.isfinite(deterministic.logprob))

    def test_sample_deterministic_with_fixed_seed(self) -> None:
        """同一 seed 的采样序列一致；不同 seed 极大概率不同。"""
        features = np.full(FEATURE_DIM, 0.05, dtype=np.float64)
        policy_a = make_policy(seed=3)
        policy_b = make_policy(seed=3)
        policy_c = make_policy(seed=4)
        rng_a = np.random.default_rng(77)
        rng_b = np.random.default_rng(77)
        rng_c = np.random.default_rng(78)
        seq_a = [policy_a.sample(features, rng_a) for _ in range(30)]
        seq_b = [policy_b.sample(features, rng_b) for _ in range(30)]
        seq_c = [policy_c.sample(features, rng_c) for _ in range(30)]
        self.assertEqual(
            [(a.mode, a.move, a.jump, a.speak) for a in seq_a],
            [(a.mode, a.move, a.jump, a.speak) for a in seq_b],
        )
        self.assertNotEqual(
            [(a.mode, a.move, a.jump, a.speak) for a in seq_a],
            [(a.mode, a.move, a.jump, a.speak) for a in seq_c],
        )
        # 不传 rng（集成层用法）也必须可用
        self.assertIsInstance(policy_a.sample(features), ActionSample)

    def test_gradient_matches_finite_difference(self) -> None:
        """手写反向与数值差分一致（至少 W1/W3 若干元素，另含 W2/b3）。"""
        policy = make_policy(seed=3)
        neural = policy.neural_policy
        features = np.linspace(-0.3, 0.3, policy_feature_dim(policy))
        action = policy.sample(features, rng=np.random.default_rng(11))
        advantage = 1.3
        _loss, grads = policy.logprob_and_grads(features, action, advantage)
        delta = 1e-5
        checks = (
            ("w1", (0, 0)),
            ("w1", (5, 7)),
            ("w2", (1, 2)),
            ("w3", (3, 0)),
            ("w3", (4, 5)),
            ("w3", (2, JUMP_INDEX)),
            ("b3", (0,)),
        )
        for name, index in checks:
            target = getattr(neural, name)
            original = float(target[index])
            target[index] = original + delta
            plus = policy.logprob_and_grads(features, action, advantage)[0]
            target[index] = original - delta
            minus = policy.logprob_and_grads(features, action, advantage)[0]
            target[index] = original
            numeric = (plus - minus) / (2.0 * delta)
            self.assertTrue(
                math.isclose(
                    float(grads[name][index]), numeric, rel_tol=1e-4, abs_tol=1e-6
                ),
                f"{name}{index}: analytic {grads[name][index]} vs numeric {numeric}",
            )

    def test_probability_increases_after_positive_update(self) -> None:
        """固定正 advantage 做若干次更新后，该动作的联合概率必须上升。"""
        policy = make_policy(seed=5)
        features = np.full(policy_feature_dim(policy), 0.2, dtype=np.float64)
        action = policy.sample(features, rng=np.random.default_rng(1))
        before = policy.probability(features, action)
        loss_before = policy.logprob_and_grads(features, action, 1.0)[0]
        for _ in range(6):
            _loss, grads = policy.logprob_and_grads(features, action, 1.0)
            policy.apply_grads(grads, 0.05)
        after = policy.probability(features, action)
        loss_after = policy.logprob_and_grads(features, action, 1.0)[0]
        self.assertGreater(after, before)
        self.assertLess(loss_after, loss_before)
        self.assertGreater(after, 0.0)
        self.assertLessEqual(after, 1.0)

    def test_zero_advantage_yields_no_gradient(self) -> None:
        """advantage=0 时 loss=0 且梯度全为 0（不会莫名漂移）。"""
        policy = make_policy(seed=2)
        features = np.full(policy_feature_dim(policy), 0.1, dtype=np.float64)
        action = policy.sample(features)
        loss, grads = policy.logprob_and_grads(features, action, 0.0)
        self.assertEqual(loss, 0.0)
        for name, grad in grads.items():
            self.assertTrue(np.allclose(grad, 0.0), f"{name} 非零")

    def test_invalid_inputs_raise(self) -> None:
        """错误特征维度 / 非有限特征 / 越界动作 / 坏梯度必须明确报错。"""
        policy = make_policy(seed=0)
        good = np.zeros(policy_feature_dim(policy), dtype=np.float64)
        with self.assertRaises(ValueError):
            policy.sample(np.zeros(policy_feature_dim(policy) + 1))
        bad = good.copy()
        bad[0] = float("nan")
        with self.assertRaises(ValueError):
            policy.sample(bad)
        with self.assertRaises(ValueError):
            policy.logprob(good, ActionSample(mode=len(MODE_NAMES), move=0, jump=False, speak=False))
        with self.assertRaises(ValueError):
            policy.logprob(good, ActionSample(mode=0, move=MOVE_DIM, jump=False, speak=False))
        action = policy.sample(good)
        with self.assertRaises(ValueError):
            policy.logprob_and_grads(good, action, float("nan"))
        with self.assertRaises(ValueError):
            policy.apply_grads({"w1": np.zeros((1, 1))}, 0.01)


class LowLevelPolicySerializationTest(unittest.TestCase):
    """checkpoint 序列化 / 语言头隔离。"""

    def test_save_load_roundtrip(self) -> None:
        """save→load 权重与采样序列一致；无后缀路径自动补 .npz。"""
        policy = make_policy(seed=21)
        features = np.linspace(-0.2, 0.2, policy_feature_dim(policy))
        rng_a = np.random.default_rng(5)
        expected = [policy.sample(features, rng_a) for _ in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "lowlevel_policy"
            saved = policy.save(target)
            self.assertEqual(saved.suffix, ".npz")
            self.assertTrue(saved.exists())
            loaded = LowLevelPolicy.load(saved, seed=21)
            self.assertTrue(
                np.allclose(loaded.flat_weights(), policy.flat_weights())
            )
            rng_b = np.random.default_rng(5)
            actual = [loaded.sample(features, rng_b) for _ in range(10)]
            self.assertEqual(
                [(a.mode, a.move, a.jump, a.speak) for a in expected],
                [(a.mode, a.move, a.jump, a.speak) for a in actual],
            )
            # 不带后缀的路径也能加载
            loaded_plain = LowLevelPolicy.load(target, seed=21)
            self.assertTrue(
                np.allclose(loaded_plain.flat_weights(), policy.flat_weights())
            )

    def test_load_missing_checkpoint_raises(self) -> None:
        """加载不存在的 checkpoint 必须报错。"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                LowLevelPolicy.load(Path(tmp) / "nope.npz")

    def test_language_head_not_trained(self) -> None:
        """语言头梯度恒 0、更新后权重逐比特不变；ActionSample 不含语言字段。"""
        policy = make_policy(seed=1)
        neural = policy.neural_policy
        features = np.full(policy_feature_dim(policy), 0.15, dtype=np.float64)
        action = policy.sample(features)
        _loss, grads = policy.logprob_and_grads(features, action, 1.0)
        self.assertTrue(np.all(grads["w3"][:, LANG_START:] == 0.0))
        self.assertTrue(np.all(grads["b3"][LANG_START:] == 0.0))
        w3_before = neural.w3.copy()
        b3_before = neural.b3.copy()
        policy.apply_grads(grads, 0.1)
        self.assertTrue(np.array_equal(neural.w3[:, LANG_START:], w3_before[:, LANG_START:]))
        self.assertTrue(np.array_equal(neural.b3[LANG_START:], b3_before[LANG_START:]))
        # 四个 RL 头确实更新了（否则"只训练 RL 头"的说法没有意义）
        self.assertFalse(np.allclose(neural.w3[:, :LANG_START], w3_before[:, :LANG_START]))
        self.assertEqual(
            set(ActionSample.__dataclass_fields__),
            {"mode", "move", "jump", "speak", "logprob"},
        )

    def test_deterministic_matches_neural_act(self) -> None:
        """deterministic() 与 NeuralPolicy.act 的 argmax 语义一致。"""
        policy = make_policy(seed=8)
        features = np.linspace(0.0, 0.5, policy_feature_dim(policy))
        action = policy.deterministic(features)
        output = policy.neural_policy.act(features)
        self.assertEqual(action.mode, MODES.index(output.mode))
        self.assertEqual(action.move, MOVE_VALUES.index(output.move))
        self.assertEqual(action.jump, bool(output.jump))
        self.assertEqual(action.speak, bool(output.speak))
        self.assertTrue(math.isfinite(action.logprob))

    def test_deterministic_vs_sample_both_legal(self) -> None:
        """确定性解码与随机采样都能在 batch 特征上稳定工作。"""
        policy = make_policy(seed=6)
        for offset in (0.0, 0.25, -0.25):
            features = np.full(policy_feature_dim(policy), offset, dtype=np.float64)
            sampled = policy.sample(features, np.random.default_rng(3))
            greedy = policy.deterministic(features)
            for action in (sampled, greedy):
                self.assertIn(action.mode, range(len(MODE_NAMES)))
                self.assertIn(action.move, range(MOVE_DIM))
                self.assertTrue(math.isfinite(action.logprob))


class LowLevelTrainerTest(unittest.TestCase):
    """trainer 流程（fake env）：采样→真实奖励→按 agent 折扣→更新→评估。"""

    def test_run_episode_uses_env_and_drain(self) -> None:
        """run_episode 驱动 env、读取真实奖励、合并 env.result()、发生一次更新。"""
        policy = make_policy(seed=0)
        before = policy.flat_weights().copy()
        rewards = [0.5, 1.0, -0.2, 2.0]
        env = _FakeEnv(
            policy.feature_dim,
            budget=4,
            reward_fn=lambda step, agent: rewards[step],
        )
        trainer = LowLevelTrainer(policy, _constant_factory(env), seed=0, lr=0.05)
        metrics = trainer.run_episode(seed=0, steps=10, dt=0.05)
        self.assertAlmostEqual(metrics["return"], sum(rewards), places=9)
        self.assertEqual(metrics["steps"], 4.0)
        self.assertEqual(metrics["transitions"], 4.0)
        self.assertEqual(metrics["agents"], 1.0)
        self.assertGreaterEqual(env.population.drain_calls, 1)
        self.assertGreaterEqual(env.population.attach_calls, 1)
        self.assertIs(env.population.attached_policy, policy)
        self.assertIn("world_time_s", metrics)  # env.result() 数值指标被并入
        self.assertFalse(np.allclose(policy.flat_weights(), before))
        self.assertEqual(trainer._updates, 1)

    def test_train_and_evaluate_fake_env(self) -> None:
        """train 完成多回合、周期评估、独立种子最终评估。"""
        policy = make_policy(seed=0)
        created: list[_FakeEnv] = []

        def factory(seed, *, steps=None, dt=None):  # noqa: ARG001
            env = _FakeEnv(policy.feature_dim, budget=int(steps or 4))
            created.append(env)
            return env

        trainer = LowLevelTrainer(policy, factory, seed=0, lr=0.02)
        result = trainer.train(
            episodes=2, steps=3, dt=0.05, eval_seeds=[100, 101], eval_steps=2
        )
        self.assertEqual(len(result["train_returns"]), 2)
        self.assertEqual(len(result["eval"]), 2)
        self.assertEqual(result["final_eval"]["episodes"], 2.0)
        self.assertTrue(math.isfinite(result["final_eval"]["mean_return"]))
        self.assertGreater(result["updates"], 0.0)
        self.assertIn("0", result["baselines"])
        self.assertGreaterEqual(len(created), 1)
        self.assertTrue(all(env.population.attach_calls > 0 for env in created))

    def test_per_agent_rewards_grouped(self) -> None:
        """两个 agent 的转移按 agent 分别累计（基线字典含两个 agent）。"""
        reward_table = {0: [1.0, 2.0], 1: [0.5, -0.5]}

        def reward_fn(step, agent_id):
            return reward_table[agent_id][step]

        policy = make_policy(seed=0)
        env = _FakeEnv(
            policy.feature_dim, budget=2, agents=(0, 1), reward_fn=reward_fn
        )
        trainer = LowLevelTrainer(policy, _constant_factory(env), seed=0)
        metrics = trainer.run_episode(seed=0, steps=2)
        self.assertEqual(metrics["transitions"], 4.0)
        self.assertEqual(metrics["agents"], 2.0)
        self.assertEqual(set(trainer._baselines), {0, 1})
        self.assertEqual(trainer._updates, 1)
        self.assertTrue(math.isfinite(metrics["mean_advantage"]))

    def test_early_end_and_invalid_budget(self) -> None:
        """env 提前 ended 时步数如实；非法回合/步数必须报错。"""
        policy = make_policy(seed=0)
        env = _FakeEnv(policy.feature_dim, budget=2)
        trainer = LowLevelTrainer(policy, _constant_factory(env), seed=0)
        metrics = trainer.run_episode(seed=0, steps=5)
        self.assertEqual(metrics["steps"], 2.0)
        with self.assertRaises(ValueError):
            trainer.run_episode(seed=0, steps=0)
        with self.assertRaises(ValueError):
            trainer.train(episodes=0, steps=5)
        with self.assertRaises(ValueError):
            trainer.train(episodes=1, steps=0)
        with self.assertRaises(ValueError):
            trainer.evaluate(seeds=[], steps=5)

    def test_no_transitions_does_not_update(self) -> None:
        """没有转移时不得假装学习：参数不变、transitions=0、updates=0。"""
        policy = make_policy(seed=0)
        before = policy.flat_weights().copy()
        env = _FakeEnv(policy.feature_dim, budget=3, agents=())
        trainer = LowLevelTrainer(policy, _constant_factory(env), seed=0)
        metrics = trainer.run_episode(seed=0, steps=3)
        self.assertEqual(metrics["transitions"], 0.0)
        self.assertEqual(metrics["return"], 0.0)
        self.assertTrue(np.allclose(policy.flat_weights(), before))
        self.assertEqual(trainer._updates, 0)

    def test_discounted_returns_formula(self) -> None:
        """折扣回报公式正确：``G_t = r_t + γ r_{t+1} + …``。"""
        returns = LowLevelTrainer.discounted_returns([1.0, 2.0, 3.0], 0.5)
        expected = [1.0 + 0.5 * 2.0 + 0.25 * 3.0, 2.0 + 0.5 * 3.0, 3.0]
        self.assertTrue(np.allclose(returns, expected))

    def test_real_lowlevel_env_hook_optional(self) -> None:
        """可选联调：Lead 的 make_lowlevel_rl_env 钩子就绪后验证转移格式。

        钩子未落地 / 未产生转移（例如 low_level_mode 不是 rl）时 skip；
        落地后它检查 ``(agent_id, features, action, reward)`` 的契约。
        """
        try:
            from newworld.simulation import make_lowlevel_rl_env  # noqa: WPS433
        except (ImportError, AttributeError):
            self.skipTest("Lead 的 make_lowlevel_rl_env 尚未落地")
        try:
            env = make_lowlevel_rl_env(0, steps=60)
        except Exception as exc:  # pragma: no cover - 集成演进中
            self.skipTest(f"make_lowlevel_rl_env 暂不可用: {type(exc).__name__}: {exc}")
        population = getattr(env, "population", None)
        attach = getattr(population, "attach_lowlevel_policy", None)
        drain = getattr(population, "drain_lowlevel_transitions", None)
        if not callable(attach) or not callable(drain):
            self.skipTest("population 低层策略钩子尚未落地")
        policy = make_policy(seed=0)
        attach(policy)
        try:
            for _ in range(60):
                if env.ended():
                    break
                env.step()
            transitions = list(drain())
        finally:
            detach = getattr(population, "detach_lowlevel_policy", None)
            if callable(detach):
                detach()
        if not transitions:
            self.skipTest("钩子未产生转移（low_level_mode 可能不是 'rl'）")
        for item in transitions[:20]:
            self.assertEqual(len(item), 4)
            agent_id, features, action, reward = item
            self.assertIsInstance(agent_id, (int, np.integer))
            self.assertEqual(np.asarray(features).shape, (policy.feature_dim,))
            self.assertIsInstance(action, ActionSample)
            self.assertIn(int(action.mode), range(len(MODE_NAMES)))
            self.assertIn(int(action.move), range(MOVE_DIM))
            self.assertTrue(math.isfinite(float(reward)))


if __name__ == "__main__":
    unittest.main()
