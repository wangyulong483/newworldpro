"""``newworld.memory`` 单元测试（P3 个体记忆 / 8 维特征 / 传承 / 死亡档案）。

只使用纯 Python 假数据，不构造 World / Character（避免把集成层拖进单元测试）。
运行：``python -m unittest tests.test_memory -v``。
"""

from __future__ import annotations

import json
import unittest

import numpy as np

from newworld.memory import (
    MEMORY_FEATURE_DIM,
    MemoryEntry,
    MemoryStore,
    Procedure,
)


class TestMemoryRecordAndQuery(unittest.TestCase):
    def test_record_and_entries_order_and_kind_filter(self) -> None:
        store = MemoryStore()
        store.record(0, 5.0, "episode", "gather:wood", "ok:wood+1", True, salience=0.7)
        store.record(0, 6.0, "procedural", "craft:stone_axe", "ok:axe", True, salience=0.9)
        store.record(0, 7.0, "social", "teach:0->1", "ok:skill", True, salience=0.4)
        entries = store.entries(0)
        self.assertEqual(len(entries), 3)
        self.assertEqual([entry.t for entry in entries], [5.0, 6.0, 7.0])
        self.assertEqual(len(store.entries(0, kind="procedural")), 1)
        self.assertEqual(len(store.entries(0, limit=2)), 2)
        self.assertEqual(store.entries(0, limit=2)[-1].kind, "social")
        self.assertEqual(store.entries(0, limit=0), [])
        self.assertEqual(store.entries(99), [])
        self.assertEqual(store.agent_count(), 1)

    def test_recent_failures_and_failure_counts(self) -> None:
        store = MemoryStore()
        for index in range(4):
            store.record(1, float(index), "episode", "gather:wood", "ok:wood+1", True)
        store.record(1, 4.0, "episode", "gather:wood", "fail:no_resource", False)
        store.record(1, 5.0, "episode", "craft:stone_axe", "fail:no_wood", False)
        store.record(1, 6.0, "episode", "gather:stone", "ok:stone+1", True)
        failures = store.recent_failures(1, k=1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].action, "craft:stone_axe")
        self.assertEqual(store.failure_counts(1), {"gather:wood": 1, "craft:stone_axe": 1})
        self.assertEqual(store.failure_counts(1, window=1), {})
        self.assertEqual(store.recent_failures(1, k=0), [])


class TestCapacityEviction(unittest.TestCase):
    def test_evicts_lowest_salience(self) -> None:
        store = MemoryStore(capacity_per_agent=3)
        first = store.record(0, 0.0, "episode", "a", "ok", True, salience=0.2)
        store.record(0, 1.0, "episode", "b", "ok", True, salience=0.9)
        store.record(0, 2.0, "episode", "c", "ok", True, salience=0.5)
        store.record(0, 3.0, "episode", "d", "ok", True, salience=0.8)
        kept = store.entries(0)
        self.assertEqual(len(kept), 3)
        self.assertNotIn(first.memory_id, [entry.memory_id for entry in kept])
        self.assertEqual([entry.action for entry in kept], ["b", "c", "d"])

    def test_evicts_oldest_when_salience_tied(self) -> None:
        store = MemoryStore(capacity_per_agent=2)
        oldest = store.record(0, 10.0, "episode", "old", "ok", True, salience=0.5)
        store.record(0, 20.0, "episode", "mid", "ok", True, salience=0.5)
        store.record(0, 30.0, "episode", "new", "ok", True, salience=0.5)
        kept = store.entries(0)
        self.assertEqual([entry.action for entry in kept], ["mid", "new"])
        self.assertNotIn(oldest.memory_id, [entry.memory_id for entry in kept])
        self.assertEqual(store.retention(0), 2.0 / 3.0)


class TestFeatureVector(unittest.TestCase):
    def test_empty_data_returns_zeros(self) -> None:
        store = MemoryStore()
        vector = store.feature_vector(42)
        self.assertEqual(vector.shape, (MEMORY_FEATURE_DIM,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertTrue(np.allclose(vector, 0.0))

    def test_shape_dtype_range_and_determinism(self) -> None:
        def build() -> MemoryStore:
            store = MemoryStore()
            store.record(0, 0.0, "procedural", "craft:stone_axe", "ok:axe", True, salience=0.9)
            store.record(0, 1.0, "procedural", "craft:stone_axe", "fail:no_wood", False)
            store.record(0, 2.0, "episode", "gather:wood", "ok:wood+1", True)
            store.record(0, 3.0, "episode", "gather:wood", "ok:wood+1", True)
            store.record(0, 4.0, "episode", "goal:build_hut", "ok:built", True)
            store.learn_procedure(0, "stone_axe", ("gather", "craft"))
            return store

        first = build().feature_vector(0)
        second = build().feature_vector(0)
        self.assertEqual(first.shape, (MEMORY_FEATURE_DIM,))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.array_equal(first, second), "同一输入必须得到同一特征向量")
        self.assertTrue(np.all(first >= -1.0) and np.all(first <= 1.0))
        # 5 条近期：4 成功 / 1 失败；掌握技能；目标成功率 1.0
        self.assertAlmostEqual(float(first[0]), 4.0 / 5.0, places=6)
        self.assertAlmostEqual(float(first[1]), 1.0 / 5.0, places=6)
        self.assertAlmostEqual(float(first[4]), 1.0, places=6)
        self.assertAlmostEqual(float(first[5]), 0.0, places=6)
        self.assertAlmostEqual(float(first[6]), 1.0, places=6)
        self.assertGreaterEqual(float(first[2]), 0.0)
        self.assertLessEqual(float(first[2]), 1.0)
        # 动作种类：craft:stone_axe / gather:wood / goal:build_hut = 3 种 → 3/8
        self.assertAlmostEqual(float(first[3]), 3.0 / 8.0, places=6)

    def test_success_rate_dim_uses_procedures(self) -> None:
        store = MemoryStore()
        store.record(0, 0.0, "episode", "gather:wood", "ok", True)
        procedure = store.learn_procedure(0, "gather", ("move_to", "gather"))
        procedure.successes = 3
        procedure.failures = 1
        vector = store.feature_vector(0)
        self.assertAlmostEqual(float(vector[5]), 0.75, places=6)


class TestTransfer(unittest.TestCase):
    def _source_store(self) -> MemoryStore:
        store = MemoryStore()
        store.record(1, 0.0, "episode", "gather:wood", "ok:wood+1", True, salience=0.4)
        store.record(1, 1.0, "procedural", "craft:stone_axe", "ok:axe", True, salience=0.9)
        store.record(1, 2.0, "social", "teach:1->2", "ok:skill", True, salience=0.6)
        store.learn_procedure(1, "stone_axe", ("gather:wood", "craft"))
        return store

    def test_copies_only_selected_kinds(self) -> None:
        store = self._source_store()
        copied = store.transfer(1, 2)
        self.assertEqual(copied, 1)
        target_entries = store.entries(2)
        self.assertEqual(len(target_entries), 1)
        self.assertTrue(all(entry.kind == "procedural" for entry in target_entries))
        self.assertEqual([procedure.skill for procedure in store.procedures(2)], ["stone_axe"])

        copied = store.transfer(1, 3, kinds=("episode", "social"))
        self.assertEqual(copied, 2)
        self.assertEqual(sorted(entry.kind for entry in store.entries(3)), ["episode", "social"])
        self.assertEqual(store.procedures(3), [])

    def test_respects_limit(self) -> None:
        store = MemoryStore()
        for index in range(5):
            store.record(1, float(index), "procedural", f"craft:{index}", "ok", True)
        copied = store.transfer(1, 2, limit=2)
        self.assertEqual(copied, 2)
        self.assertEqual([entry.action for entry in store.entries(2)], ["craft:3", "craft:4"])
        self.assertEqual(store.transfer(1, 1), 0)
        self.assertEqual(store.transfer(1, 4, limit=0), 0)


class TestArchiveAndRetention(unittest.TestCase):
    def test_archive_does_not_delete_anything(self) -> None:
        store = MemoryStore()
        store.record(0, 0.0, "episode", "gather:wood", "ok:wood+1", True)
        store.record(0, 1.0, "procedural", "craft:stone_axe", "ok:axe", True)
        store.learn_procedure(0, "stone_axe", ("gather:wood", "craft"))
        before = len(store.entries(0))
        archive = store.archive(0)
        after = len(store.entries(0))
        self.assertEqual(after, before)
        self.assertEqual(archive["entry_count"], before)
        self.assertEqual(len(archive["entries"]), before)
        self.assertEqual(archive["procedures"]["stone_axe"]["skill"], "stone_axe")
        self.assertEqual(len(archive["feature_vector"]), MEMORY_FEATURE_DIM)
        self.assertIsInstance(archive["retention"], float)
        json.dumps(archive)  # 必须可 JSON 序列化
        self.assertEqual(len(store.entries(0)), after)

    def test_retention_counts_evictions(self) -> None:
        store = MemoryStore(capacity_per_agent=2)
        self.assertEqual(store.retention(0), 0.0)
        for index in range(4):
            store.record(0, float(index), "episode", f"a{index}", "ok", True)
        self.assertEqual(len(store.entries(0)), 2)
        self.assertEqual(store.retention(0), 2.0 / 4.0)
        self.assertAlmostEqual(store.retention(1), 0.0)


class TestProceduresAndSerialization(unittest.TestCase):
    def test_learn_procedure_and_success_rate(self) -> None:
        store = MemoryStore()
        procedure = store.learn_procedure(0, "stone_axe", ("gather", "craft"))
        self.assertIsInstance(procedure, Procedure)
        self.assertEqual(procedure.steps, ("gather", "craft"))
        self.assertEqual(procedure.success_rate(), 0.0)
        procedure.successes = 2
        procedure.failures = 2
        self.assertAlmostEqual(procedure.success_rate(), 0.5)
        # 重复学习覆盖 steps 但保留计数
        store.learn_procedure(0, "stone_axe", ("move_to", "gather", "craft"))
        procedures = store.procedures(0)
        self.assertEqual(len(procedures), 1)
        self.assertEqual(procedures[0].steps, ("move_to", "gather", "craft"))
        self.assertEqual(procedures[0].successes, 2)

    def test_to_dict_from_dict_roundtrip(self) -> None:
        store = MemoryStore(capacity_per_agent=8)
        store.record(0, 0.0, "episode", "gather:wood", "ok:wood+1", True, salience=0.4)
        store.record(0, 1.0, "procedural", "craft:stone_axe", "fail:no_wood", False, salience=0.9)
        store.record(1, 2.0, "social", "teach:0->1", "ok", True, salience=0.6)
        store.learn_procedure(0, "stone_axe", ("gather:wood", "craft"))
        payload = json.loads(json.dumps(store.to_dict()))
        restored = MemoryStore.from_dict(payload)
        self.assertEqual(len(restored.entries(0)), 2)
        self.assertEqual(len(restored.entries(1)), 1)
        self.assertEqual(restored.agent_count(), 2)
        self.assertEqual([procedure.skill for procedure in restored.procedures(0)], ["stone_axe"])
        self.assertTrue(
            np.array_equal(restored.feature_vector(0), store.feature_vector(0)),
            "序列化往返后 8 维特征必须完全一致",
        )
        self.assertEqual(restored.capacity_per_agent, 8)
        self.assertEqual(restored.retention(0), store.retention(0))
        # 往返后继续写入不冲突
        entry = restored.record(0, 3.0, "episode", "gather:stone", "ok", True)
        self.assertGreater(entry.memory_id, 2)
        self.assertEqual(len(restored.entries(0)), 3)

    def test_entry_and_procedure_dict_roundtrip(self) -> None:
        entry = MemoryEntry(3, 1, 2.5, "episode", "gather", "ok", True, "d", 0.7)
        self.assertEqual(MemoryEntry.from_dict(entry.to_dict()), entry)
        procedure = Procedure("axe", ("a", "b"), 2, 1, 4.0)
        self.assertEqual(Procedure.from_dict(procedure.to_dict()), procedure)


if __name__ == "__main__":
    unittest.main()
