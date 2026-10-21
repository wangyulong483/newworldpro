"""P6-B 去 LLM 生成式记忆/反思层单元测试。

覆盖（见任务验收）：
* 检索打分公式与排序：新近 / 重要 / 相关各自的排序、权重组合、归一化、
  零范数（query/feature 为 0）不产生 NaN；
* 容量淘汰规则：保留分数最低者出局、同分淘汰更旧；
* 反思聚合的确定性 + 方向/风险/成功率语义 + importance >= 0.7；
* 跨 compaction 的行号重排（记忆跟着个体走，不跟槽位走）；
* 集成：memory enabled -> swarm.obs_dim == 9+6，obs 有限，跑一段不崩，
  stats/count_of/summary_zh 可用；同 seed 完全确定性；
* 协议/配置接线：configs/p6_memory.yaml 可载入，memory_ablation 已注册。

运行::

    cd newworldpro; $env:TEMP=(Resolve-Path .tmp).Path
    python -m unittest tests.test_agent_memory -v
"""

from __future__ import annotations

import dataclasses
import math
import unittest
from pathlib import Path

import numpy as np

from newworld.agent_memory import (
    FEATURE_DIM,
    KIND_ATE,
    KIND_NEAR_DEATH,
    KIND_REFLECTION,
    OBS_DIM,
    MemoryLayer,
    MemoryRecord,
    MemoryStream,
    NullMemoryLayer,
    NullMemorySettings,
    aggregate_reflection,
    build_memory_layer,
    cosine_similarity,
    direction_bucket,
    parse_retrieval_weights,
    retrieval_scores,
)
from newworld.experiment import build_simulation
from newworld.settings import (
    ExperimentSettings,
    MemorySettings,
    PolicySettings,
    SocialSettings,
    SwarmSettings,
    WorldSettings,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
(PROJECT_ROOT / ".tmp").mkdir(exist_ok=True)

PROJECT_CONFIG = PROJECT_ROOT / "configs" / "p6_memory.yaml"


def feature(dx: float = 0.0, dy: float = 0.0, strength: float = 0.0,
            danger: float = 0.0, urgency: float = 0.0) -> np.ndarray:
    return np.asarray((dx, dy, strength, danger, urgency), dtype=np.float64)


def record(
    t: float = 0.0,
    kind: str = KIND_ATE,
    tag: str = "food:E",
    importance: float = 0.5,
    vec: np.ndarray | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        t, kind, tag, importance, feature() if vec is None else vec
    )


def tiny_settings(*, memory: MemorySettings | None = None,
                  policy_kind: str = "rule") -> ExperimentSettings:
    return ExperimentSettings(
        name="tiny_memory",
        world=WorldSettings(
            width=64,
            height=48,
            seed=3,
            ocean_margin=4,
            noise_scale=24,
            food_regrow_per_s=0.0005,
        ),
        swarm=SwarmSettings(initial_population=20, capacity_hard=200),
        policy=PolicySettings(kind=policy_kind),
        social=SocialSettings(enabled=False),
        memory=memory or MemorySettings(enabled=True, reflection_every_s=0.0),
    )


# ---------------------------------------------------------------------------
class ScoringTest(unittest.TestCase):
    """新近 × 重要 × 相关：每一项单独都能改变排序，且归一化正确。"""

    def test_recency_only_orders_newer_first(self) -> None:
        stream = MemoryStream(8, weights=(1.0, 0.0, 0.0), halflife=10.0)
        stream.add(record(t=0.0, tag="old", importance=0.1))
        stream.add(record(t=10.0, tag="new", importance=0.1))
        picked = stream.retrieve(None, now_t=10.0, top_k=2)
        self.assertEqual([r.tag for _s, r in picked], ["new", "old"])
        self.assertAlmostEqual(picked[0][0], 2.0 ** (-0.0))  # 1.0
        self.assertAlmostEqual(picked[1][0], 2.0 ** (-1.0))  # 0.5

    def test_importance_only_orders_high_importance_first(self) -> None:
        stream = MemoryStream(8, weights=(0.0, 1.0, 0.0))
        stream.add(record(t=5.0, tag="low", importance=0.2))
        stream.add(record(t=5.0, tag="high", importance=0.9))
        picked = stream.retrieve(None, now_t=5.0, top_k=2)
        self.assertEqual([r.tag for _s, r in picked], ["high", "low"])
        self.assertAlmostEqual(picked[0][0], 0.9)

    def test_relevance_only_orders_aligned_first(self) -> None:
        query = feature(dx=1.0, dy=0.0, strength=1.0)
        stream = MemoryStream(8, weights=(0.0, 0.0, 1.0))
        stream.add(record(t=1.0, tag="east", vec=feature(1.0, 0.0, 1.0)))
        stream.add(record(t=1.0, tag="south", vec=feature(0.0, 1.0, 1.0)))
        picked = stream.retrieve(query, now_t=1.0, top_k=2)
        self.assertEqual(picked[0][1].tag, "east")
        self.assertAlmostEqual(picked[0][0], 1.0, places=6)
        self.assertLess(picked[1][0], picked[0][0])

    def test_cosine_normalises_zero_norm(self) -> None:
        """零 feature / 零 query 的余弦项必须为 0，不能 0/0。"""
        zeros = np.zeros((2, FEATURE_DIM))
        self.assertTrue(np.allclose(cosine_similarity(zeros, zeros[0]), 0.0))
        ones = np.ones((2, FEATURE_DIM))
        self.assertTrue(np.allclose(cosine_similarity(ones, zeros[0]), 0.0))
        # 幅值只影响范数不影响方向：cos([2,0,..],[1,0,..]) == 1
        self.assertAlmostEqual(
            float(cosine_similarity(np.asarray([[2.0, 0, 0, 0, 0]]), np.asarray([1.0, 0, 0, 0, 0]))[0]),
            1.0,
        )

    def test_score_formula_exact(self) -> None:
        times = np.asarray([0.0, 10.0])
        importances = np.asarray([0.3, 0.7])
        feats = np.asarray(
            [feature(1.0, 0.0, 1.0), feature(0.0, 1.0, 1.0)], dtype=np.float64
        )
        query = feature(1.0, 0.0, 1.0)
        scores = retrieval_scores(
            times, importances, feats, 10.0, query, weights=(1.0, 1.0, 1.0), halflife=10.0
        )
        expected0 = 1.0 * (2.0 ** -1.0) + 1.0 * 0.3 + 1.0 * 1.0
        # 特征 (0,1,1) 与 query (1,0,1) 的余弦 = 1/(sqrt2*sqrt2) = 0.5
        expected1 = 1.0 * (2.0 ** 0.0) + 1.0 * 0.7 + 1.0 * 0.5
        self.assertAlmostEqual(float(scores[0]), expected0, places=9)
        self.assertAlmostEqual(float(scores[1]), expected1, places=9)

    def test_ties_are_deterministic_by_importance_age_index(self) -> None:
        stream = MemoryStream(8, weights=(0.0, 0.0, 0.0))
        stream.add(record(t=5.0, tag="a", importance=0.5))
        stream.add(record(t=1.0, tag="b", importance=0.5))
        stream.add(record(t=1.0, tag="c", importance=0.5))
        first = stream.retrieve(None, now_t=9.0, top_k=3)
        second = stream.retrieve(None, now_t=9.0, top_k=3)
        self.assertEqual([r.tag for _s, r in first], [r.tag for _s, r in second])
        # 分数全 0：先比 importance（同），再比 t（新者优先），最后插入序
        self.assertEqual([r.tag for _s, r in first], ["a", "b", "c"])

    def test_future_timestamp_recency_capped_at_one(self) -> None:
        stream = MemoryStream(4, weights=(1.0, 0.0, 0.0), halflife=10.0)
        stream.add(record(t=100.0))
        picked = stream.retrieve(None, now_t=10.0, top_k=1)
        self.assertAlmostEqual(picked[0][0], 1.0)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            parse_retrieval_weights((1.0, 2.0))
        with self.assertRaises(ValueError):
            retrieval_scores(np.zeros(1), np.zeros(1), np.zeros((1, FEATURE_DIM)),
                             0.0, None, halflife=0.0)
        with self.assertRaises(ValueError):
            record(importance=1.5)
        with self.assertRaises(ValueError):
            MemoryRecord(0.0, KIND_ATE, "x", 0.5, np.zeros(FEATURE_DIM - 1))
        with self.assertRaises(ValueError):
            record(kind="unknown_kind")
        with self.assertRaises(ValueError):
            MemoryStream(0)

    def test_weight_presets(self) -> None:
        self.assertEqual(parse_retrieval_weights(None), (1.0, 1.0, 1.0))
        self.assertEqual(
            parse_retrieval_weights({"recency": 2, "importance": 0}),
            (2.0, 0.0, 1.0),
        )


# ---------------------------------------------------------------------------
class CapacityTest(unittest.TestCase):
    """容量淘汰：保留分数最低者出局，同分淘汰更旧。"""

    def test_evicts_lowest_retention_score(self) -> None:
        stream = MemoryStream(3, weights=(0.0, 1.0, 0.0), halflife=30.0)
        stream.add(record(t=0.0, tag="low", importance=0.1))
        stream.add(record(t=1.0, tag="mid", importance=0.5))
        stream.add(record(t=2.0, tag="high", importance=0.9))
        evicted = stream.add(record(t=3.0, tag="new", importance=0.6))
        self.assertIsNotNone(evicted)
        self.assertEqual(evicted.tag, "low")
        self.assertEqual(len(stream), 3)
        self.assertEqual({r.tag for r in stream.records()}, {"mid", "high", "new"})

    def test_equal_scores_evict_oldest(self) -> None:
        stream = MemoryStream(2, weights=(0.0, 0.0, 0.0), halflife=30.0)
        stream.add(record(t=0.0, tag="oldest", importance=0.5))
        stream.add(record(t=5.0, tag="newer", importance=0.5))
        evicted = stream.add(record(t=9.0, tag="new", importance=0.5))
        self.assertEqual(evicted.tag, "oldest")

    def test_records_returns_copy(self) -> None:
        stream = MemoryStream(4)
        stream.add(record(tag="a"))
        copied = stream.records()
        copied.clear()
        self.assertEqual(len(stream), 1)
        self.assertEqual(stream.records()[0].tag, "a")


# ---------------------------------------------------------------------------
class ReflectionAggregationTest(unittest.TestCase):
    """反思聚合：确定性 + 方向/风险/成功率语义 + importance 下界。"""

    def collect(self) -> list[MemoryRecord]:
        return [
            record(t=0.0, tag="food:E", importance=0.5, vec=feature(1.0, 0.0, 0.5, 0.0, 0.2)),
            record(t=1.0, tag="food:E", importance=0.5, vec=feature(1.0, 0.0, 0.5, 0.0, 0.2)),
            record(t=2.0, tag="food:E", importance=0.5, vec=feature(1.0, 0.0, 0.5, 0.0, 0.2)),
            record(t=3.0, tag="food:W", importance=0.5, vec=feature(-1.0, 0.0, 0.5, 0.0, 0.5)),
            record(t=4.0, kind=KIND_NEAR_DEATH, tag="danger", importance=0.9,
                   vec=feature(0.0, 0.0, 0.0, 1.0, 1.0)),
        ]

    def test_dominant_direction_success_and_danger(self) -> None:
        tag, feat, importance = aggregate_reflection(self.collect())
        self.assertEqual(tag, "reflect:E")
        self.assertGreater(feat[0], 0.9)  # 归一化后近似 (1, 0)
        self.assertAlmostEqual(feat[1], 0.0, places=6)
        self.assertAlmostEqual(feat[2], 4.0 / 5.0)  # 成功率（3 个 E + 1 个 W）
        self.assertAlmostEqual(feat[3], 1.0 / 5.0)  # 风险率
        self.assertGreaterEqual(importance, 0.7)

    def test_deterministic(self) -> None:
        first = aggregate_reflection(self.collect())
        second = aggregate_reflection(self.collect())
        self.assertEqual(first[0], second[0])
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2], second[2])

    def test_empty_and_danger_only(self) -> None:
        tag, feat, importance = aggregate_reflection([])
        self.assertTrue(tag.startswith("reflect:"))
        self.assertGreaterEqual(importance, 0.7)
        danger = [record(kind=KIND_NEAR_DEATH, tag="danger", importance=0.9,
                         vec=feature(0.0, 0.0, 0.0, 1.0, 1.0))]
        tag2, feat2, importance2 = aggregate_reflection(danger)
        self.assertEqual(tag2, "reflect:danger")
        self.assertAlmostEqual(feat2[3], 1.0)
        self.assertGreaterEqual(importance2, 0.7)

    def test_reflection_records_ignored_in_aggregation(self) -> None:
        events = self.collect()
        reflection = record(kind=KIND_REFLECTION, tag="reflect:E", importance=0.9,
                            vec=feature(0.0, 1.0, 0.0, 1.0, 0.0))
        tag, feat, _importance = aggregate_reflection(events + [reflection])
        self.assertEqual(tag, "reflect:E")

    def test_direction_bucket(self) -> None:
        self.assertEqual(direction_bucket(5.0, 1.0), "E")
        self.assertEqual(direction_bucket(-5.0, 1.0), "W")
        self.assertEqual(direction_bucket(1.0, 5.0), "S")
        self.assertEqual(direction_bucket(1.0, -5.0), "N")
        self.assertEqual(direction_bucket(1e-12, 0.0), "here")


# ---------------------------------------------------------------------------
class ReindexTest(unittest.TestCase):
    """死亡压实后，记忆流/缓存必须跟着个体（旧行号 -> 新行号）重排。"""

    def test_memory_follows_individual_across_compaction(self) -> None:
        sim = build_simulation(tiny_settings(), seed=3)
        sim.step()  # 一帧后 memory.update 已为所有存活行建流
        layer: MemoryLayer = sim.memory
        self.assertIsNotNone(layer)
        swarm = sim.swarm
        self.assertEqual(len(layer._streams), swarm.count)
        marker = record(t=0.0, tag="marker", importance=0.99,
                        vec=feature(1.0, 0.0, 1.0, 0.0, 0.0))
        layer._streams[5].add(marker)
        layer._features[5] = np.arange(OBS_DIM, dtype=np.float64)
        # 杀死第 2 行并压实（等价于 swarm.update 末尾的 _compact）
        swarm.data["alive"][2] = False
        swarm.deaths += 1
        swarm._compact()
        self.assertGreaterEqual(swarm.compact_generation, 1)
        old_to_new = layer._reindex_if_needed()
        self.assertIsNotNone(old_to_new)
        self.assertEqual(old_to_new[5], 4)
        self.assertIn(marker, layer._streams[4].records())
        self.assertEqual(layer._streams[4].records()[-1].tag, "marker")
        self.assertTrue(
            np.array_equal(layer._features[4], np.arange(OBS_DIM, dtype=np.float64))
        )
        layer.update(0.05)
        self.assertEqual(layer.reindexed, 1)
        self.assertEqual(
            set(layer._streams), set(range(swarm.count))
        )
        self.assertEqual(layer.count_of(4), len(layer._streams[4]))


# ---------------------------------------------------------------------------
class IntegrationTest(unittest.TestCase):
    """build_simulation 接线：9+6 维观测、有限值、可跑、统计键齐全。"""

    def test_obs_dim_and_stats(self) -> None:
        sim = build_simulation(tiny_settings(), seed=3)
        self.assertEqual(sim.swarm.obs_dim, 9 + OBS_DIM)
        self.assertEqual(sim.memory.memory_obs_dim, OBS_DIM)
        metrics = sim.run(seconds=1.0)
        self.assertIn("memory", metrics)
        for key in (
            "events_recorded",
            "retrievals",
            "reflections",
            "insights",
            "mean_memories",
            "memory_obs_dim",
        ):
            self.assertIn(key, metrics["memory"])
        self.assertGreater(metrics["memory"]["events_recorded"], 0)
        self.assertIn("记忆", sim.memory.summary_zh())
        # 观测形状/有限性（直接走 swarm 的 provider 路径）
        view = sim.swarm.data[: sim.swarm.count]
        rows = np.nonzero(view["alive"])[0].astype(np.int64)
        obs = sim.swarm._observations(view, rows)
        self.assertEqual(obs.shape, (rows.size, 9 + OBS_DIM))
        self.assertTrue(np.all(np.isfinite(obs)))
        # 个体记忆条数
        for row in rows.tolist():
            self.assertGreaterEqual(sim.memory.count_of(int(row)), 0)
        self.assertEqual(sim.memory.count_of(10_000), 0)

    def test_memory_attached_before_policy(self) -> None:
        """策略 obs_dim 必须与注册后的 swarm.obs_dim 一致（可训练）。"""
        sim = build_simulation(
            tiny_settings(policy_kind="neural", memory=MemorySettings(
                enabled=True, reflection_every_s=0.0)), seed=3
        )
        self.assertEqual(sim.policy.obs_dim, 9 + OBS_DIM)
        sim.run(seconds=0.5)

    def test_rule_policy_path_records_events(self) -> None:
        """V3 规则策略路径（_eat_amount=None）也能靠饥饿变化检测进食。"""
        sim = build_simulation(tiny_settings(), seed=3)
        sim.run(seconds=1.5)
        self.assertGreater(sim.memory.events_recorded, 0)

    def test_reflection_runs_and_is_deterministic(self) -> None:
        memory = MemorySettings(
            enabled=True,
            reflection_every_s=0.2,
            reflection_min_events=2,
            capacity=12,
        )
        first = build_simulation(tiny_settings(memory=memory), seed=3)
        second = build_simulation(tiny_settings(memory=memory), seed=3)
        for _ in range(30):
            first.step()
            second.step()
        self.assertEqual(first.memory.stats, second.memory.stats)
        self.assertGreaterEqual(first.memory.reflections, 1)
        self.assertGreater(first.memory.insights, 0)
        # 反思记忆 importance >= 0.7（任务硬约束）
        for stream in first.memory._streams.values():
            for item in stream.records():
                if item.kind == KIND_REFLECTION:
                    self.assertGreaterEqual(item.importance, 0.7)

    def test_feature_only_from_stored_events(self) -> None:
        """没有记忆时特征全 0；有记忆后方向特征来自事件而非当前感知。"""
        sim = build_simulation(tiny_settings(), seed=3)
        layer: MemoryLayer = sim.memory
        view = sim.swarm.data[: sim.swarm.count]
        rows = np.nonzero(view["alive"])[0].astype(np.int64)
        fresh = layer._provide_observations(view, rows)
        self.assertTrue(np.allclose(fresh, 0.0))
        sim.step()
        sim.step()
        after = layer._provide_observations(view, np.nonzero(
            sim.swarm.data[: sim.swarm.count]["alive"])[0].astype(np.int64))
        self.assertTrue(np.all(np.isfinite(after)))
        self.assertEqual(after.shape[1], OBS_DIM)


# ---------------------------------------------------------------------------
class WiringTest(unittest.TestCase):
    """配置/协议接线：YAML 可载入、协议已注册、四臂配置正确。"""

    def test_project_config_loads(self) -> None:
        settings = load_config(PROJECT_CONFIG)
        self.assertEqual(settings.protocol, "memory_ablation")
        self.assertTrue(settings.memory.enabled)
        self.assertEqual(settings.memory.capacity, 24)
        self.assertEqual(settings.memory.top_k, 3)
        self.assertEqual(settings.swarm.initial_population, 120)
        self.assertIn("train_episodes", settings.protocol_params)
        self.assertIn("eval_seeds", settings.protocol_params)

    def test_protocol_registered(self) -> None:
        import newworld.protocols.p6_memory  # noqa: F401
        from newworld.registry import PROTOCOLS

        self.assertIn("memory_ablation", PROTOCOLS.names())

    def test_arm_configs(self) -> None:
        from newworld.protocols.p6_memory import ARM_NAMES, _arm_memory_configs

        self.assertEqual(
            ARM_NAMES,
            ("off", "off_padded", "recency", "full", "full_reflect"),
        )
        base_memory = MemorySettings(enabled=True, reflection_every_s=20.0)
        arms = _arm_memory_configs(base_memory, 20.0)
        self.assertEqual(tuple(arms), ARM_NAMES)
        self.assertFalse(arms["off"].enabled)
        # off_padded：走 enabled 构造路径，但 zero_observation 令观测恒 0
        self.assertIsInstance(arms["off_padded"], NullMemorySettings)
        self.assertTrue(arms["off_padded"].enabled)
        self.assertTrue(arms["off_padded"].zero_observation)
        self.assertEqual(arms["off_padded"].top_k, base_memory.top_k)
        self.assertEqual(
            (arms["recency"].weight_recency, arms["recency"].weight_importance,
             arms["recency"].weight_relevance),
            (1.0, 0.0, 0.0),
        )
        self.assertEqual(arms["recency"].reflection_every_s, 0.0)
        self.assertEqual(
            (arms["full"].weight_recency, arms["full"].weight_importance,
             arms["full"].weight_relevance),
            (1.0, 1.0, 1.0),
        )
        self.assertEqual(arms["full"].reflection_every_s, 0.0)
        self.assertEqual(arms["full_reflect"].reflection_every_s, 20.0)

    def test_reflection_fires_within_40s_run(self) -> None:
        """任务要求：40s 评估窗口内反思必须真的触发（否则 full_reflect==full）。"""
        memory = MemorySettings(
            enabled=True,
            reflection_every_s=20.0,
            reflection_min_events=6,
        )
        sim = build_simulation(tiny_settings(memory=memory), seed=3)
        sim.run(seconds=40.0)
        metrics = sim.metrics()["memory"]
        self.assertGreater(metrics["reflections"], 0.0)
        self.assertGreater(metrics["insights"], 0.0)
        self.assertGreater(metrics["obs_reflect_food_mean"], 0.0)

    def test_off_padded_null_memory(self) -> None:
        """空记忆对照：obs_dim=9+6，多出的 6 列恒 0，且 stats/count_of 为 0。"""
        padded = NullMemorySettings(enabled=True, reflection_every_s=0.0)
        sim = build_simulation(tiny_settings(memory=padded), seed=3)
        self.assertIsInstance(sim.memory, NullMemoryLayer)
        self.assertEqual(sim.swarm.obs_dim, 9 + OBS_DIM)
        sim.run(seconds=1.0)
        view = sim.swarm.data[: sim.swarm.count]
        rows = np.nonzero(view["alive"])[0].astype(np.int64)
        obs = sim.swarm._observations(view, rows)
        self.assertEqual(obs.shape, (rows.size, 9 + OBS_DIM))
        self.assertTrue(np.allclose(obs[:, 9:], 0.0))
        self.assertEqual(sim.memory.count_of(int(rows[0])), 0)
        stats = sim.metrics()["memory"]
        self.assertEqual(stats["events_recorded"], 0)
        self.assertEqual(stats["memory_obs_dim"], OBS_DIM)
        self.assertIn("空记忆", sim.memory.summary_zh())


if __name__ == "__main__":
    unittest.main()
