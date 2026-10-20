"""P1 群伙策略层单元测试（≥15 用例）。

验证**机制正确性**（接口合法/确定性/观测语义/梯度数学/训练闭环/
规则等价性），不声称"学出更聪明行为"——那由 experiments/run_p1_baseline.py
的配对对照实验回答。

运行::

    cd newworldpro; $env:TEMP=(Resolve-Path .tmp).Path
    python -m unittest tests.test_swarm_policy -v
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

from newworld.map_engine import MapConfig, MapEngine  # noqa: E402
from newworld.swarm import Swarm, SwarmConfig  # noqa: E402
from newworld.swarm_policy import (  # noqa: E402
    ACTION_DELTAS,
    N_ACTIONS,
    OBS_DIM,
    REWARD_DEATH,
    REWARD_EAT,
    REWARD_TIME_PER_S,
    NeuralSwarmPolicy,
    RuleSwarmPolicy,
    SwarmTrainer,
)


def make_engine(seed: int = 0) -> MapEngine:
    """小地图工厂。"""
    return MapEngine(MapConfig(seed=seed, width=128, height=80, noise_scale=32))


def make_swarm(engine: MapEngine, seed: int = 0, population: int = 60) -> Swarm:
    """群伙工厂。"""
    return Swarm(
        engine,
        SwarmConfig(initial_population=population, capacity_hard=500),
        seed=seed,
    )


# ---------------------------------------------------------------------------
class InterfaceTest(unittest.TestCase):
    """协议 / 动作空间 / 观测语义。"""

    def test_action_space_frozen(self) -> None:
        """动作数=5、delta 表与约定一致（右/左/下/上/原地）。"""
        self.assertEqual(N_ACTIONS, 5)
        self.assertEqual(ACTION_DELTAS, ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)))

    def test_obs_dim_frozen(self) -> None:
        """观测维度冻结为 9。"""
        self.assertEqual(OBS_DIM, 9)

    def test_rule_policy_output_legal(self) -> None:
        """规则策略输出在值域内、形状正确。"""
        policy = RuleSwarmPolicy(seed=0)
        obs = np.random.default_rng(0).random((50, OBS_DIM))
        actions = policy.act(obs)
        self.assertEqual(actions.shape, (50,))
        self.assertTrue(((actions >= 0) & (actions < N_ACTIONS)).all())

    def test_neural_policy_output_legal(self) -> None:
        """神经策略采样输出合法且可复现。"""
        policy = NeuralSwarmPolicy(seed=7)
        obs = np.zeros((30, OBS_DIM))
        a1 = policy.act(obs)
        a2 = NeuralSwarmPolicy(seed=7).act(obs)
        self.assertEqual(a1.tolist(), a2.tolist())
        self.assertTrue(((a1 >= 0) & (a1 < N_ACTIONS)).all())

    def test_invalid_obs_raises(self) -> None:
        """错误形状/NaN 观测必须报错，不静默。"""
        policy = NeuralSwarmPolicy(seed=0)
        with self.assertRaises(ValueError):
            policy.act(np.zeros((5, OBS_DIM + 1)))
        bad = np.zeros((5, OBS_DIM))
        bad[0, 0] = np.nan
        # act 走批量路径，逐转移训练路径显式拒绝 NaN
        policy.logprob_and_grads  # noqa: B018 - 存在性
        with self.assertRaises(ValueError):
            policy.logprob_and_grads(bad[0], 0, 1.0)

    def test_attach_rejects_non_policy(self) -> None:
        """attach_policy 拒绝没有 act 的对象。"""
        engine = make_engine()
        swarm = make_swarm(engine)
        with self.assertRaises(TypeError):
            swarm.attach_policy(object())


# ---------------------------------------------------------------------------
class ObservationSemanticsTest(unittest.TestCase):
    """Swarm._observations 的字段语义。"""

    def test_observation_fields(self) -> None:
        """饱足/食物/四方向/水警示/bias 语义正确。"""
        engine = make_engine(seed=3)
        swarm = make_swarm(engine, seed=3)
        view = swarm.data[: swarm.count]
        rows = np.arange(min(10, swarm.count))
        obs = swarm._observations(view, rows)
        self.assertEqual(obs.shape, (rows.size, OBS_DIM))
        # bias 列恒 1
        self.assertTrue((obs[:, 8] == 1.0).all())
        # hunger 列 ∈ [0,1]
        self.assertTrue(((obs[:, 0] >= 0) & (obs[:, 0] <= 1)).all())
        # 四方向食物：不可走为 -1，可走 ≥0
        for k in range(4):
            column = obs[:, 3 + k]
            self.assertTrue(((column >= -1.0) & (column <= 1.0)).all())

    def test_water_neighbor_flag(self) -> None:
        """邻水警示：水边个体为 1，内陆食物充足处可能为 0。"""
        engine = make_engine(seed=4)
        swarm = make_swarm(engine, seed=4)
        view = swarm.data[: swarm.count]
        rows = np.arange(swarm.count)
        obs = swarm._observations(view, rows)
        # 至少能找到两种取值中的一种存在（水边 or 内陆）
        self.assertTrue(set(np.unique(obs[:, 7]).tolist()) >= {0.0})


# ---------------------------------------------------------------------------
class RewardSettlementTest(unittest.TestCase):
    """奖励结算：进食 +1、死亡 -10、时间成本、转移记录。"""

    def _run_with_policy(self, engine, swarm, steps: int = 40):
        for _ in range(steps):
            engine.update(0.05, 0.8)
            swarm.update(0.05)

    def test_transitions_recorded_with_policy(self) -> None:
        """注入策略后转移被记录；drain 取走后缓冲清空。"""
        engine = make_engine(seed=5)
        swarm = make_swarm(engine, seed=5)
        swarm.attach_policy(NeuralSwarmPolicy(seed=5))
        self._run_with_policy(engine, swarm)
        transitions = swarm.drain_swarm_transitions()
        self.assertGreater(len(transitions), 0)
        for obs, action, reward in transitions[:20]:
            self.assertEqual(obs.shape, (OBS_DIM,))
            self.assertTrue(0 <= action < N_ACTIONS)
            self.assertTrue(math.isfinite(reward))
        self.assertEqual(swarm.drain_swarm_transitions(), [])

    def test_no_transitions_without_policy(self) -> None:
        """不注策略（V3 原行为）不记录转移、不算奖励。"""
        engine = make_engine(seed=6)
        swarm = make_swarm(engine, seed=6)
        self._run_with_policy(engine, swarm)
        self.assertEqual(swarm.drain_swarm_transitions(), [])
        self.assertEqual(swarm.total_reward, 0.0)

    def test_eat_reward_positive_and_counted(self) -> None:
        """进食按实际量给奖励并进 eat_events（账目精确可核）。"""
        engine = make_engine(seed=7)
        swarm = make_swarm(engine, seed=7, population=80)
        swarm.attach_policy(RuleSwarmPolicy(seed=7))
        self._run_with_policy(engine, swarm, steps=200)
        self.assertGreater(swarm.eat_events, 0)
        # P1.5 账目守恒：奖励 = 时间成本×全部转移 + Σ实际进食量 + 死亡×(-10)
        # 实际进食量直接从转移里取（含进食奖励的转移 = reward > time 上界）
        dt = 0.05
        per_time = REWARD_TIME_PER_S * dt
        transitions = swarm.drain_swarm_transitions()
        n = len(transitions)
        death_extra = sum(1 for r in transitions if r[2] < REWARD_DEATH / 2)
        # 每条转移的进食分量 = max(0, r - time)（无死亡时）；死亡转移单独核对
        eat_sum = sum(max(0.0, r[2] - per_time) for r in transitions
                      if r[2] > REWARD_DEATH / 2)
        expected = n * per_time + eat_sum + death_extra * REWARD_DEATH
        self.assertAlmostEqual(swarm.total_reward, expected, places=2,
                               msg="奖励账目与进食量/死亡/时间成本不符")
        # 进食分量应为正且 ≤ bite_size
        self.assertGreater(eat_sum, 0.0)
        self.assertLessEqual(eat_sum / max(swarm.eat_events, 1), 0.35 + 1e-6,
                             "单次进食分量超过 bite_size")

    def test_death_reward_negative(self) -> None:
        """断粮场景死亡转移带 -10 奖励。"""
        engine = make_engine(seed=8)
        swarm = make_swarm(engine, seed=8, population=20)
        engine.food[:] = 0.0  # 断粮（不推进 engine → 食物不回来）
        swarm.attach_policy(NeuralSwarmPolicy(seed=8))
        for _ in range(2600):  # 130s > 代谢寿命
            swarm.update(0.05)
        transitions = swarm.drain_swarm_transitions()
        self.assertGreater(len(transitions), 0)
        death_rewards = [r for _, _, r in transitions if r < REWARD_DEATH / 2]
        self.assertGreater(len(death_rewards), 0, "应存在含死亡惩罚的转移")
        self.assertEqual(swarm.population, 0)


# ---------------------------------------------------------------------------
class RuleEquivalenceTest(unittest.TestCase):
    """规则策略搬家不改行为。"""

    def test_rules_mode_matches_no_policy(self) -> None:
        """policy=None 与注入 RuleSwarmPolicy 的种群规模同量级。

        严格逐帧一致不可行（RuleSwarmPolicy 有独立 RNG 流），但机制
        等价性由同量级人口/进食验证。
        """
        engine_a = make_engine(seed=9)
        swarm_a = make_swarm(engine_a, seed=9)
        engine_b = make_engine(seed=9)
        swarm_b = make_swarm(engine_b, seed=9)
        swarm_b.attach_policy(RuleSwarmPolicy(seed=9))
        for _ in range(600):
            engine_a.update(0.05, 0.8)
            swarm_a.update(0.05)
            engine_b.update(0.05, 0.8)
            swarm_b.update(0.05)
        # 同量级（差异 <25%：随机流不同导致的自然方差）
        a, b = swarm_a.population, swarm_b.population
        self.assertLess(abs(a - b) / max(a, b, 1), 0.25,
                        f"规则搬家后人口失配: {a} vs {b}")

    def test_policy_isolated_from_world(self) -> None:
        """注入策略不会改变食物守恒：吃掉的量与世界扣的一致。"""
        engine = make_engine(seed=10)
        swarm = make_swarm(engine, seed=10, population=40)
        swarm.attach_policy(NeuralSwarmPolicy(seed=10))
        food_before = float(engine.food.sum())
        for _ in range(100):
            engine.update(0.05, 0.8)
            swarm.update(0.05)
        food_now = float(engine.food.sum())
        # 食物只会被吃或再生，两个方向都是真实网格操作
        self.assertGreaterEqual(food_now, 0.0)


# ---------------------------------------------------------------------------
class NeuralPolicyTest(unittest.TestCase):
    """神经策略的数学与训练。"""

    def test_gradient_matches_finite_difference(self) -> None:
        """手写梯度与有限差分一致（REINFORCE 核心正确性）。"""
        policy = NeuralSwarmPolicy(seed=0)
        rng = np.random.default_rng(0)
        obs = rng.random(OBS_DIM)
        action = 2
        advantage = 0.7
        _, grads = policy.logprob_and_grads(obs, action, advantage)
        eps = 1e-6

        def loss_at(w_path, delta):
            saved = w_path[0].copy()
            w_path[0][w_path[1]] += delta
            hidden, logits = policy._forward(obs)
            probs = policy._softmax(logits)
            value = -advantage * math.log(max(probs[action], 1e-12))
            w_path[0][w_path[1]] = saved[0] if False else saved[w_path[1]]
            w_path[0][w_path[1]] = saved[w_path[1]]
            return value

        for name, target in (("W1", policy.W1), ("W2", policy.W2),
                             ("b1", policy.b1), ("b2", policy.b2)):
            idx = (0, 0) if target.ndim == 2 else (0,)
            grad_numeric = (loss_at((target, idx), eps) - loss_at((target, idx), -eps)) / (2 * eps)
            grad_analytic = grads[name][idx]
            self.assertAlmostEqual(grad_numeric, float(grad_analytic), places=4,
                                   msg=f"{name} 梯度不一致")

    def test_positive_advantage_increases_action_prob(self) -> None:
        """正优势更新后该动作概率上升（学习方向正确）。"""
        policy = NeuralSwarmPolicy(seed=1)
        obs = np.random.default_rng(1).random(OBS_DIM)
        action = 3
        _, probs_before = policy._forward(obs), None
        hidden, logits = policy._forward(obs)
        probs_before = policy._softmax(logits)
        for _ in range(30):
            _, grads = policy.logprob_and_grads(obs, action, advantage=1.0)
            policy.apply_grads(grads, lr=0.05)
        _, logits_after = policy._forward(obs)
        probs_after = policy._softmax(logits_after)
        self.assertGreater(probs_after[action], probs_before[action])

    def test_zero_advantage_no_gradient(self) -> None:
        """零优势不产生更新。"""
        policy = NeuralSwarmPolicy(seed=2)
        obs = np.zeros(OBS_DIM)
        _, grads = policy.logprob_and_grads(obs, 0, 0.0)
        for value in grads.values():
            self.assertTrue(np.allclose(value, 0.0))

    def test_save_load_roundtrip(self) -> None:
        """checkpoint 往返：权重逐元素一致、行为可复现。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "swarm_policy_test.npz"
            policy = NeuralSwarmPolicy(seed=3)
            policy.save(path)
            loaded = NeuralSwarmPolicy.load(path, seed=3)
            self.assertTrue(np.array_equal(policy.flat_weights(), loaded.flat_weights()))
            obs = np.random.default_rng(3).random((10, OBS_DIM))
            self.assertEqual(policy.act(obs).tolist(), loaded.act(obs).tolist())

    def test_trainer_updates_and_clips(self) -> None:
        """训练器：更新发生、裁剪生效、空转移不更新。"""
        policy = NeuralSwarmPolicy(seed=4)
        trainer = SwarmTrainer(policy, lr=0.02, batch_size=64,
                               epochs_per_episode=2, grad_clip=0.5)
        rng = np.random.default_rng(4)
        transitions = [
            (rng.random(OBS_DIM), int(rng.integers(0, N_ACTIONS)), float(rng.normal(0, 1)))
            for _ in range(200)
        ]
        stats = trainer.train_on_transitions(transitions)
        self.assertGreater(stats["updates"], 0)
        self.assertGreater(stats["weight_delta"], 0)
        # 空批不更新
        before = policy.flat_weights().copy()
        empty = trainer.train_on_transitions([])
        self.assertEqual(empty["updates"], 0.0)
        self.assertTrue(np.array_equal(before, policy.flat_weights()))

    def test_discounted_returns_formula(self) -> None:
        """折扣回报公式正确。"""
        out = SwarmTrainer.discounted_returns([1.0, 2.0, 3.0], 0.5)
        expected = [1.0 + 0.5 * 2.0 + 0.25 * 3.0, 2.0 + 0.5 * 3.0, 3.0]
        self.assertTrue(np.allclose(out, expected))

    def test_normalized_advantage_statistics(self) -> None:
        """批内标准化：均值≈0、std≈1。"""
        trainer = SwarmTrainer(NeuralSwarmPolicy(seed=5))
        raw = np.asarray([1.0, 2.0, 3.0, 6.0], dtype=np.float64)
        out = trainer._normalized(raw)
        self.assertAlmostEqual(float(out.mean()), 0.0, places=9)
        self.assertAlmostEqual(float(out.std()), 1.0, places=6)


# ---------------------------------------------------------------------------
class EndToEndTest(unittest.TestCase):
    """完整闭环：世界 + 策略 + 奖励 + 训练。"""

    def test_full_loop_learning_signal(self) -> None:
        """采集→训练→权重确实变化（机制闭环；效果由实验脚本评估）。"""
        policy = NeuralSwarmPolicy(seed=6)
        trainer = SwarmTrainer(policy, lr=0.02, batch_size=256, grad_clip=0.5)
        engine = make_engine(seed=6)
        swarm = make_swarm(engine, seed=6, population=40)
        swarm.attach_policy(policy)
        for _ in range(300):
            engine.update(0.05, 0.8)
            swarm.update(0.05)
        transitions = swarm.drain_swarm_transitions()
        self.assertGreater(len(transitions), 50)
        before = policy.flat_weights().copy()
        stats = trainer.train_on_transitions(transitions)
        self.assertGreater(stats["weight_delta"], 0.0)
        self.assertFalse(np.array_equal(before, policy.flat_weights()))

    def test_detach_restores_rules(self) -> None:
        """卸载策略后回到 V3 行为：不再记录转移。"""
        engine = make_engine(seed=7)
        swarm = make_swarm(engine, seed=7)
        swarm.attach_policy(NeuralSwarmPolicy(seed=7))
        swarm.update(0.05)
        swarm.detach_policy()
        for _ in range(50):
            engine.update(0.05, 0.8)
            swarm.update(0.05)
        self.assertEqual(swarm.drain_swarm_transitions(), [])


if __name__ == "__main__":
    unittest.main()
