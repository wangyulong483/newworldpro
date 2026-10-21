"""P6 通信层单元测试：通道机制、信息论统计、指称博弈、配置/注册。

测试范围内的仿真都很小（96x64 世界、40 个体、关闭社交），只验证机制，
不跑实验规模；端到端协议冒烟用极小 protocol_params 跑一遍。
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newworld.communication import (
    CommunicationLayer,
    ReferentialGame,
    build_communication_layer,
    entropy_bits,
    mapping_similarity,
    mi_expected_bias_bits,
    mutual_information_bits,
)
from newworld.experiment import build_simulation, run_experiment
from newworld.registry import PROTOCOLS
from newworld.settings import CommSettings, apply_overrides, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
(PROJECT_ROOT / ".tmp").mkdir(exist_ok=True)
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
P6_CONFIG = PROJECT_ROOT / "configs" / "p6_communication.yaml"


def tiny_settings(*, comm_train: bool = False, n_symbols: int = 4):
    """小世界配置：只改世界/种群/社交/通信，其余沿用 base.yaml。"""
    settings = load_config(BASE_CONFIG)
    return dataclasses.replace(
        settings,
        world=dataclasses.replace(settings.world, width=96, height=64),
        swarm=dataclasses.replace(settings.swarm, initial_population=40, capacity_hard=200),
        social=dataclasses.replace(settings.social, enabled=False),
        policy=dataclasses.replace(settings.policy, kind="neural"),
        comm=CommSettings(
            enabled=True,
            n_symbols=int(n_symbols),
            radius=6.0,
            train=bool(comm_train),
            lr=0.05,
            batch_size=16,
            epochs=1,
        ),
    )


class ChannelMechanicsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = tiny_settings(comm_train=False)
        self.sim = build_simulation(self.settings, seed=0)
        self.layer = self.sim.comm
        self.swarm = self.sim.swarm

    def test_obs_dim_and_provider_shape(self) -> None:
        # obs_dim == n_symbols；群伙总观测 = 基础 9 + K
        self.assertEqual(self.layer.obs_dim, 4)
        self.assertEqual(self.swarm.obs_dim, 9 + 4)
        view = self.swarm.data[: self.swarm.count]
        alive = np.nonzero(view["alive"])[0]
        self.assertGreater(alive.size, 0)
        rows = alive[:5]
        block = self.layer._heard_histogram(view, rows)
        self.assertEqual(block.shape, (rows.size, 4))
        self.assertTrue(np.all(np.isfinite(block)))
        self.assertTrue(np.all(block >= 0.0) and np.all(block <= 1.0))
        # 每一行要么全 0（没听到），要么和为 1（归一化直方图）
        sums = block.sum(axis=1)
        self.assertTrue(np.all((np.abs(sums) < 1e-9) | (np.abs(sums - 1.0) < 1e-9)))
        obs = self.swarm._observations(view, rows)
        self.assertEqual(obs.shape, (rows.size, 9 + 4))
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_provider_uses_previous_symbols_and_radius(self) -> None:
        from newworld.swarm import SWARM_DTYPE

        view = np.zeros(2, dtype=SWARM_DTYPE)
        view["alive"] = True
        view["x"] = [10.0, 10.0 + self.settings.comm.radius + 5.0]
        view["y"] = [10.0, 10.0]
        self.layer.symbols[0] = 0
        self.layer.symbols[1] = 2
        far = self.layer._heard_histogram(view, np.asarray([0]))
        self.assertAlmostEqual(float(far.sum()), 0.0, places=12)
        view["x"][1] = 11.0  # 进入 radius
        near = self.layer._heard_histogram(view, np.asarray([0]))
        self.assertAlmostEqual(float(near[0, 2]), 1.0, places=12)

    def test_symbols_in_range_and_stats_keys(self) -> None:
        self.sim.run(seconds=1.0)  # 20 步
        # attach 与每步重采样都计入消息数
        self.assertGreaterEqual(self.layer.stats["messages"], 1)
        view = self.swarm.data[: self.swarm.count]
        alive = np.nonzero(view["alive"])[0]
        symbols = self.layer.symbols[alive]
        self.assertTrue(np.all((symbols >= 0) & (symbols < 4)))
        for key in (
            "messages",
            "symbol_entropy",
            "mutual_information",
            "food_alignment",
            "train_updates",
            "group_reward_mean",
            "symbol_counts",
        ):
            self.assertIn(key, self.layer.stats)
        self.assertEqual(len(self.layer.stats["symbol_counts"]), 4)
        self.assertTrue(0.0 <= self.layer.stats["symbol_entropy"] <= 2.0 + 1e-9)
        self.assertTrue(0.0 <= self.layer.stats["mutual_information"] <= 2.0 + 1e-9)
        self.assertTrue(0.0 <= self.layer.stats["food_alignment"] <= 1.0)
        self.assertIn("MI", self.layer.summary_zh())
        self.assertIn(int(self.layer.symbol_of(int(alive[0]))), range(4))

    def test_reindex_across_compaction(self) -> None:
        self.sim.run(seconds=0.5)
        old_symbols = self.layer.symbols[: self.swarm.count].copy()
        old_count = self.swarm.count
        view = self.swarm.data[: self.swarm.count]
        for row in (1, 3, 5):
            view["alive"][row] = False
        self.swarm._compact()
        self.assertEqual(self.swarm.compact_generation, 1)
        keep = np.asarray(self.swarm.compact_keep)
        expected = old_symbols[keep]
        self.assertTrue(self.layer._maybe_reindex())
        np.testing.assert_array_equal(self.layer.symbols[: self.swarm.count], expected)
        # 压实后的 update 不应崩溃，符号仍在值域内
        self.layer.update(0.05)
        alive = np.nonzero(self.swarm.data[: self.swarm.count]["alive"])[0]
        self.assertTrue(np.all((self.layer.symbols[alive] >= 0) & (self.layer.symbols[alive] < 4)))
        self.assertLess(old_count, 200)

    def test_train_step_updates_sender_when_enabled(self) -> None:
        settings = tiny_settings(comm_train=True)
        sim = build_simulation(settings, seed=1)
        layer = sim.comm
        before = layer.policy.flat_weights().copy()
        sim.run(seconds=1.0)  # 20 步 * ~40 pending 个体 => 超过 batch_size 16
        self.assertGreater(layer.stats["train_updates"], 0)
        delta = float(np.max(np.abs(layer.policy.flat_weights() - before)))
        self.assertGreater(delta, 0.0)

    def test_message_cost_is_real_when_configured(self) -> None:
        settings = tiny_settings(comm_train=False)
        settings = dataclasses.replace(
            settings, comm=dataclasses.replace(settings.comm, message_cost=0.01)
        )
        sim = build_simulation(settings, seed=0)
        sim.run(seconds=0.5)
        self.assertGreater(sim.comm._message_cost_total, 0.0)


class InfoTheoryTest(unittest.TestCase):
    def test_entropy(self) -> None:
        self.assertAlmostEqual(entropy_bits([0, 0, 0, 0], 4), 0.0, places=12)
        self.assertAlmostEqual(entropy_bits([0, 1, 2, 3] * 50, 4), 2.0, places=9)
        self.assertAlmostEqual(entropy_bits([0, 1] * 100, 2), 1.0, places=9)
        self.assertAlmostEqual(entropy_bits([], 4), 0.0, places=12)

    def test_mutual_information_synthetic(self) -> None:
        # 完全相同的离散变量：MI == H(x)；常量输入：MI=H=0
        x = np.asarray([0, 1, 2, 3] * 100)
        self.assertAlmostEqual(mutual_information_bits(x, x, 4, 4), entropy_bits(x, 4), places=9)
        self.assertAlmostEqual(mutual_information_bits([0] * 100, [0] * 100, 4, 4), 0.0, places=12)
        self.assertAlmostEqual(entropy_bits([0] * 100, 4), 0.0, places=12)
        # 独立变量：n=4000 时 MI 应接近 0 且低于偏置量级
        rng = np.random.default_rng(0)
        a = rng.integers(0, 4, size=4000)
        b = rng.integers(0, 4, size=4000)
        mi = mutual_information_bits(a, b, 4, 4)
        self.assertGreaterEqual(mi, 0.0)
        self.assertLess(mi, 0.01)
        bias = mi_expected_bias_bits(4000, 4, 4)
        self.assertAlmostEqual(bias, 9.0 / (2 * 4000 * np.log(2.0)), places=9)
        self.assertAlmostEqual(mi_expected_bias_bits(8192, 4, 4), 0.000792, places=5)

    def test_mapping_similarity_allows_relabeling(self) -> None:
        left = {0: 0, 1: 1, 2: 2, 3: 3}
        renamed = {0: 3, 1: 2, 2: 1, 3: 0}  # 同一协议，只是符号换了名字
        self.assertAlmostEqual(mapping_similarity(left, renamed, 4), 1.0, places=12)
        different = {0: 0, 1: 0, 2: 2, 3: 2}  # 有 cued 共用符号，无法对齐成 1
        self.assertLess(mapping_similarity(left, different, 4), 1.0)


class ReferentialGameTest(unittest.TestCase):
    def test_reaches_above_chance(self) -> None:
        game = ReferentialGame(cue_dim=4, n_symbols=4, hidden=32, seed=0, lr=0.5)
        before = game.evaluate(n=500)
        self.assertLessEqual(before["accuracy"], 0.6)  # 未训练：接近机会水平
        game.train(episodes=200, batch=64)
        result = game.evaluate(n=1000)
        # 4 选 1 的机会水平 = 0.25；训练后应显著高于 0.30
        self.assertGreater(result["accuracy"], 0.30)
        self.assertGreater(result["mutual_information_bits"], 0.1)
        self.assertEqual(result["n"], 1000)
        cues = set(result["mapping"])
        self.assertEqual(cues, {0, 1, 2, 3})

    def test_deterministic_across_same_seed(self) -> None:
        def train_once() -> float:
            game = ReferentialGame(cue_dim=4, n_symbols=4, seed=3, lr=0.3)
            game.train(episodes=20, batch=32)
            return game.evaluate(n=500)["accuracy"]

        self.assertAlmostEqual(train_once(), train_once(), places=12)

    def test_invalid_shapes_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReferentialGame(cue_dim=1, n_symbols=4)
        with self.assertRaises(ValueError):
            ReferentialGame(cue_dim=4, n_symbols=1)
        game = ReferentialGame(cue_dim=4, n_symbols=4)
        with self.assertRaises(ValueError):
            game.train(episodes=0)


class ConfigAndRegistrationTest(unittest.TestCase):
    def test_config_loads(self) -> None:
        settings = load_config(P6_CONFIG)
        self.assertEqual(settings.protocol, "communication")
        self.assertTrue(settings.comm.enabled)
        self.assertEqual(settings.comm.n_symbols, 4)
        self.assertAlmostEqual(settings.comm.radius, 6.0)
        self.assertEqual(len(settings.run.seeds), 5)
        self.assertLessEqual(settings.world.food_regrow_per_s, 0.006)
        for key in ("game_seeds", "game_episodes", "train_episodes", "eval_seeds", "eval_seconds"):
            self.assertIn(key, settings.protocol_params)
        self.assertEqual(len(settings.protocol_params["game_seeds"]), 5)

    def test_protocol_registered(self) -> None:
        from newworld.protocols import p6_communication  # noqa: F401  触发注册

        self.assertIn("communication", PROTOCOLS)
        self.assertTrue(callable(PROTOCOLS.get("communication")))


class ProtocolSmokeTest(unittest.TestCase):
    def test_end_to_end_tiny(self) -> None:
        settings = apply_overrides(
            load_config(P6_CONFIG),
            [
                "protocol_params.game_episodes=5",
                "protocol_params.game_eval_n=100",
                "protocol_params.train_episodes=1",
                "protocol_params.train_episode_seconds=2.0",
                "protocol_params.eval_seconds=2.0",
                "protocol_params.train_seeds=[1000]",
                "protocol_params.eval_seeds=[2000,2001]",
                "run.save_models=false",
            ],
        )
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / ".tmp")) as directory:
            summary = run_experiment(settings, directory=directory)
            self.assertEqual(summary["protocol"], "communication")
            self.assertIn("part_a_referential", summary)
            self.assertIn("part_b_swarm_ablation", summary)
            self.assertIn("verdict", summary)
            part_a = summary["part_a_referential"]
            self.assertIn("accuracy", part_a)
            self.assertEqual(part_a["game_seeds"], [0, 1, 2, 3, 4])
            paired = summary["part_b_swarm_ablation"]["paired_treatment_minus_control"]
            self.assertIn("eat_events", paired)
            self.assertEqual(int(paired["eat_events"]["n"]), 2)
            channel = summary["part_b_swarm_ablation"]["channel_stats_eval"]
            self.assertIn("mi_bits", channel)
            self.assertTrue(channel["sender_frozen"])
            path = Path(directory) / "summary.json"
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["protocol"], "communication")


if __name__ == "__main__":
    unittest.main()
