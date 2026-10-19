"""``newworld.knowledge`` 的单元测试（P2 验收：test_knowledge）。

运行::

    cd newworldpro
    $env:TEMP=(Resolve-Path .tmp).Path
    python -m unittest tests.test_knowledge -v

全部测试只使用 fake world / fake technology / fake agent，**不依赖** Lead 尚未
落地的集成代码。世界规则的唯一权威由 ``FakeTechnology.simulate_recipe`` 扮演：
它记录被调用的配方并返回 ``(ok, reason)``，因此可以验证"知识层从不自己扣材料、
所有结论都来自世界模拟器"。
"""

from __future__ import annotations

import json
import unittest

from newworld.knowledge import (
    KNOWLEDGE_STATUS,
    KNOWLEDGE_TYPES,
    MAX_FAILED_USES,
    NO_SIMULATOR_REASON,
    RECIPE_CATALOG,
    VERIFIED_CONFIDENCE_FLOOR,
    Knowledge,
    KnowledgeStore,
    Recipe,
    ValidationRecord,
)

# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------
class FakeAgent:
    """最小个体：只需要 agent_id 与可写的已知知识集合。"""

    def __init__(self, agent_id: int = 1, known: list[str] | None = None) -> None:
        self.agent_id = agent_id
        self.known_knowledge_ids: list[str] = list(known or [])


class FakeTechnology:
    """世界规则替身：记录调用，按预设结果返回 ``(ok, reason)``。"""

    def __init__(
        self,
        *,
        outcomes: list[object] | None = None,
        result: bool = True,
        reason: str = "ok",
        afford: bool = True,
        error: BaseException | None = None,
    ) -> None:
        self.outcomes = list(outcomes) if outcomes is not None else None
        self.result = bool(result)
        self.reason = reason
        self.afford = afford
        self.error = error
        self.calls: list[str] = []
        self.afford_calls: list[str] = []

    def simulate_recipe(self, agent, recipe) -> tuple[bool, str]:
        self.calls.append(recipe.recipe_id)
        if self.error is not None:
            raise self.error
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, tuple):
                return outcome  # type: ignore[return-value]
            return (bool(outcome), "ok" if outcome else "failed")
        return (self.result, self.reason)

    def can_afford(self, agent, recipe) -> bool:
        self.afford_calls.append(recipe.recipe_id)
        return self.afford


class FakeWorld:
    """带 technology 的世界替身（``technology=None`` 模拟未接线）。"""

    def __init__(self, technology=None) -> None:
        self.technology = technology


class BareWorld:
    """完全没有 ``technology`` 属性的世界（P1 未接线）。"""


def make_world(**technology_kwargs) -> FakeWorld:
    """构造一个默认返回成功的 fake world。"""
    return FakeWorld(FakeTechnology(**technology_kwargs))


def verify(store: KnowledgeStore, recipe_id: str, world, agent, t: float = 0.0) -> Knowledge:
    """通过世界模拟器把一条知识验证到 ``verified``。"""
    knowledge = store.get(recipe_id)
    assert knowledge is not None, recipe_id
    for _ in range(knowledge.required_work()):
        ok, reason = store.attempt(agent, knowledge, world, t)
        if not ok:
            raise AssertionError(f"{recipe_id} 验证失败: {reason}")
    return knowledge


# ---------------------------------------------------------------------------
# 世界规则目录
# ---------------------------------------------------------------------------
class TestRecipeCatalog(unittest.TestCase):
    """RECIPE_CATALOG 是"世界规则"：内容与任务书 §5 一致，并含集成阶段新增的粮仓。"""

    EXPECTED = {
        "stone_axe": ({"wood": 2, "stone": 1}, {"axe": 1}, 1, 0.2, ()),
        "fiber_cord": ({"fiber": 3}, {"cord": 1}, 2, 0.3, ()),
        "basket": ({"wood": 2, "cord": 1}, {"basket": 1}, 2, 0.4, ("fiber_cord",)),
        "campfire": ({"wood": 2, "stone": 1}, {"campfire": 1}, 1, 0.3, ("stone_axe",)),
        "hut": ({"wood": 4, "stone": 2, "cord": 1}, {"hut": 1}, 3, 0.5, ("basket",)),
        "well": ({"stone": 3, "cord": 1}, {"well": 1}, 2, 0.4, ("fiber_cord",)),
        # 集成阶段新增：食物储存技术（让群体能越过浆果丛的自然承载力）
        "granary": ({"wood": 2, "stone": 2}, {"granary": 1}, 1, 0.25, ("stone_axe",)),
    }

    def test_catalog_contains_expected_technologies(self) -> None:
        self.assertEqual(set(RECIPE_CATALOG), set(self.EXPECTED))

    def test_catalog_recipes_match_the_frozen_contract(self) -> None:
        for recipe_id, (inputs, outputs, work, difficulty, requires) in self.EXPECTED.items():
            recipe = RECIPE_CATALOG[recipe_id]
            self.assertIsInstance(recipe, Recipe)
            self.assertEqual(recipe.recipe_id, recipe_id)
            self.assertEqual(recipe.inputs, inputs)
            self.assertEqual(recipe.outputs, outputs)
            self.assertEqual(recipe.work, work)
            self.assertAlmostEqual(recipe.difficulty, difficulty)
            self.assertEqual(recipe.requires_knowledge, requires)
            self.assertIsNone(recipe.requires_tool)
            self.assertTrue(recipe.name_zh)

    def test_recipe_clamps_difficulty_and_work(self) -> None:
        low = Recipe("low", "低", {}, {}, difficulty=-1.0, work=0)
        high = Recipe("high", "高", {}, {}, difficulty=2.0)
        self.assertEqual(low.difficulty, 0.0)
        self.assertEqual(low.work, 1)
        self.assertEqual(high.difficulty, 1.0)

    def test_recipe_dict_roundtrip(self) -> None:
        recipe = RECIPE_CATALOG["hut"]
        self.assertEqual(Recipe.from_dict(recipe.to_dict()), recipe)


# ---------------------------------------------------------------------------
# bootstrap / 初始状态
# ---------------------------------------------------------------------------
class TestBootstrap(unittest.TestCase):
    def test_bootstrap_registers_every_technology_as_hypothesis(self) -> None:
        store = KnowledgeStore()
        items = store.all()
        self.assertEqual(len(items), len(RECIPE_CATALOG))
        self.assertTrue(all(item.status == "hypothesis" for item in items))
        self.assertTrue(all(not item.usable() for item in items))
        self.assertTrue(all(item.confidence == 0.0 for item in items))
        self.assertTrue(all(item.discovered_by == -1 for item in items))
        self.assertTrue(all(item.recipe is not None for item in items))
        self.assertIn(items[0].status, KNOWLEDGE_STATUS)
        self.assertIn(items[0].knowledge_type, KNOWLEDGE_TYPES)

    def test_bootstrap_fills_prerequisites_from_recipe(self) -> None:
        store = KnowledgeStore()
        hut = store.get("hut")
        self.assertEqual(hut.prerequisites, ("basket",))
        self.assertEqual(hut.dependency_ids(), ("basket",))
        self.assertEqual(store.get("stone_axe").dependency_ids(), ())

    def test_bootstrap_can_be_disabled(self) -> None:
        store = KnowledgeStore(bootstrap=False)
        self.assertEqual(store.all(), [])
        self.assertEqual(set(store.catalog), set(RECIPE_CATALOG))


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------
class TestKnowledgeStateMachine(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KnowledgeStore()
        self.agent = FakeAgent()

    def test_work_consecutive_successes_switch_to_verified(self) -> None:
        world = make_world()
        knowledge = self.store.get("fiber_cord")
        self.assertEqual(knowledge.required_work(), 2)

        ok, _ = self.store.attempt(self.agent, knowledge, world, 1.0)
        self.assertTrue(ok)
        self.assertEqual(knowledge.status, "hypothesis")
        self.assertFalse(knowledge.usable())
        self.assertEqual(knowledge.successful_uses, 1)

        ok, _ = self.store.attempt(self.agent, knowledge, world, 2.0)
        self.assertTrue(ok)
        self.assertEqual(knowledge.status, "verified")
        self.assertTrue(knowledge.usable())
        self.assertEqual(knowledge.successful_uses, 2)
        self.assertEqual(len(world.technology.calls), 2)
        self.assertGreaterEqual(knowledge.confidence, VERIFIED_CONFIDENCE_FLOOR)

    def test_failure_breaks_the_consecutive_streak(self) -> None:
        world = make_world(outcomes=[True, False, True, True])
        knowledge = self.store.get("fiber_cord")

        self.store.attempt(self.agent, knowledge, world, 1.0)  # 连续 1
        self.store.attempt(self.agent, knowledge, world, 2.0)  # 失败打断
        self.store.attempt(self.agent, knowledge, world, 3.0)  # 连续 1
        self.assertEqual(knowledge.status, "hypothesis")
        self.assertEqual(knowledge.successful_uses, 2)
        self.assertEqual(knowledge.failed_uses, 1)
        self.assertEqual(knowledge.consecutive_successes(), 1)

        self.store.attempt(self.agent, knowledge, world, 4.0)  # 连续 2 → 验证
        self.assertEqual(knowledge.status, "verified")
        self.assertEqual(knowledge.consecutive_successes(), 2)

    def test_three_failures_invalidate_the_knowledge(self) -> None:
        world = make_world(result=False, reason="no_stone")
        knowledge = self.store.get("stone_axe")
        for index in range(MAX_FAILED_USES):
            ok, reason = self.store.attempt(self.agent, knowledge, world, float(index))
            self.assertFalse(ok)
            self.assertEqual(reason, "no_stone")

        self.assertEqual(knowledge.failed_uses, MAX_FAILED_USES)
        self.assertEqual(knowledge.status, "invalidated")
        self.assertFalse(knowledge.usable())
        self.assertEqual(knowledge.confidence, 0.0)

        # 失效是终态：后续尝试不再联系世界模拟器
        calls_before = len(world.technology.calls)
        ok, reason = self.store.attempt(self.agent, knowledge, world, 99.0)
        self.assertFalse(ok)
        self.assertEqual(reason, "invalidated")
        self.assertEqual(len(world.technology.calls), calls_before)

    def test_invalidated_is_terminal_even_after_a_success(self) -> None:
        knowledge = self.store.get("stone_axe")
        world = make_world(result=False, reason="fail")
        for index in range(MAX_FAILED_USES):
            self.store.attempt(self.agent, knowledge, world, float(index))
        self.assertEqual(knowledge.status, "invalidated")

        knowledge.record(99.0, self.agent.agent_id, True, "late success")
        self.assertEqual(knowledge.status, "invalidated")
        self.assertEqual(knowledge.successful_uses, 1)

    def test_failure_penalty_scales_with_difficulty(self) -> None:
        easy = Knowledge("easy", "tool", "石斧", recipe=RECIPE_CATALOG["stone_axe"], confidence=0.5)
        hard = Knowledge("hard", "structure", "小屋", recipe=RECIPE_CATALOG["hut"], confidence=0.5)
        easy.record(0.0, 1, False, "失败")
        hard.record(0.0, 1, False, "失败")
        self.assertLess(easy.confidence, 0.5)
        self.assertLess(hard.confidence, easy.confidence)

    def test_success_raises_confidence_and_sets_floor_when_verified(self) -> None:
        knowledge = self.store.get("stone_axe")
        knowledge.record(0.0, 1, True, "成功")
        self.assertGreater(knowledge.confidence, 0.0)
        self.assertLessEqual(knowledge.confidence, 1.0)
        self.assertEqual(knowledge.status, "verified")
        self.assertAlmostEqual(knowledge.confidence, VERIFIED_CONFIDENCE_FLOOR)
        knowledge.record(1.0, 1, True, "再次成功")
        self.assertGreaterEqual(knowledge.confidence, VERIFIED_CONFIDENCE_FLOOR)
        self.assertLessEqual(knowledge.confidence, 1.0)

    def test_usable_only_for_verified(self) -> None:
        for status, expected in (
            ("hypothesis", False),
            ("verified", True),
            ("invalidated", False),
        ):
            knowledge = Knowledge("k", "tool", "测试", status=status)
            self.assertEqual(knowledge.usable(), expected, status)


# ---------------------------------------------------------------------------
# attempt：唯一验证路径
# ---------------------------------------------------------------------------
class TestAttempt(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KnowledgeStore()
        self.agent = FakeAgent()

    def test_accepts_string_id_and_records_history(self) -> None:
        world = make_world()
        ok, reason = self.store.attempt(self.agent, "stone_axe", world, 5.0)
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")
        knowledge = self.store.get("stone_axe")
        self.assertEqual(len(knowledge.validation_history), 1)
        record = knowledge.validation_history[-1]
        self.assertIsInstance(record, ValidationRecord)
        self.assertEqual(record.agent_id, self.agent.agent_id)
        self.assertAlmostEqual(record.t, 5.0)
        self.assertTrue(record.success)
        self.assertEqual(world.technology.calls, ["stone_axe"])

    def test_unknown_id_is_rejected(self) -> None:
        ok, reason = self.store.attempt(self.agent, "ghost_tech", make_world(), 0.0)
        self.assertFalse(ok)
        self.assertEqual(reason, "unknown_knowledge")

    def test_knowledge_without_recipe_is_rejected(self) -> None:
        self.store.add(Knowledge("weird", "technique", "怪知识"))
        ok, reason = self.store.attempt(self.agent, "weird", make_world(), 0.0)
        self.assertFalse(ok)
        self.assertEqual(reason, "no_recipe")

    def test_missing_simulator_degrades_without_polluting_state(self) -> None:
        knowledge = self.store.get("stone_axe")
        ok, reason = self.store.attempt(self.agent, knowledge, BareWorld(), 0.0)
        self.assertFalse(ok)
        self.assertEqual(reason, NO_SIMULATOR_REASON)
        self.assertEqual(self.store.degraded_attempts, 1)
        self.assertEqual(knowledge.status, "hypothesis")
        self.assertEqual(knowledge.failed_uses, 0)
        self.assertEqual(knowledge.validation_history, [])

        # FakeWorld(technology=None) 同样降级
        ok, reason = self.store.attempt(self.agent, knowledge, FakeWorld(None), 1.0)
        self.assertFalse(ok)
        self.assertEqual(reason, NO_SIMULATOR_REASON)
        self.assertEqual(self.store.degraded_attempts, 2)

    def test_missing_prerequisites_are_not_sent_to_the_simulator(self) -> None:
        world = make_world()
        basket = self.store.get("basket")
        ok, reason = self.store.attempt(self.agent, basket, world, 0.0)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_prerequisites")
        self.assertEqual(world.technology.calls, [])
        self.assertEqual(basket.failed_uses, 0)

        verify(self.store, "fiber_cord", world, self.agent)
        ok, _ = self.store.attempt(self.agent, basket, world, 10.0)
        self.assertTrue(ok)
        self.assertEqual(basket.successful_uses, 1)

    def test_simulator_exception_is_recorded_as_failure(self) -> None:
        world = make_world(error=RuntimeError("boom"))
        knowledge = self.store.get("stone_axe")
        ok, reason = self.store.attempt(self.agent, knowledge, world, 1.0)
        self.assertFalse(ok)
        self.assertEqual(reason, "simulator_error:RuntimeError")
        self.assertEqual(knowledge.failed_uses, 1)
        self.assertEqual(knowledge.validation_history[-1].evidence, "simulator_error:RuntimeError")

    def test_attempting_verified_knowledge_records_usage(self) -> None:
        world = make_world()
        knowledge = verify(self.store, "stone_axe", world, self.agent)
        self.assertEqual(knowledge.successful_uses, 1)
        ok, _ = self.store.attempt(self.agent, knowledge, world, 1.0)
        self.assertTrue(ok)
        self.assertEqual(knowledge.successful_uses, 2)
        self.assertEqual(knowledge.status, "verified")


# ---------------------------------------------------------------------------
# 候选实验筛选
# ---------------------------------------------------------------------------
class TestCandidates(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KnowledgeStore()
        self.agent = FakeAgent()

    def test_candidates_require_verified_prerequisites(self) -> None:
        candidates = {item.knowledge_id for item in self.store.candidates_for(self.agent)}
        self.assertEqual(candidates, {"stone_axe", "fiber_cord"})

        world = make_world()
        verify(self.store, "fiber_cord", world, self.agent)
        candidates = {item.knowledge_id for item in self.store.candidates_for(self.agent)}
        self.assertIn("basket", candidates)
        self.assertIn("well", candidates)
        self.assertNotIn("campfire", candidates)  # stone_axe 仍是 hypothesis
        self.assertNotIn("hut", candidates)  # basket 未验证

    def test_candidates_respect_can_afford(self) -> None:
        world = make_world()
        verify(self.store, "fiber_cord", world, self.agent)

        poor = FakeWorld(FakeTechnology(afford=False))
        self.assertEqual(self.store.candidates_for(self.agent, poor), [])

        rich = FakeWorld(FakeTechnology(afford=True))
        ids = {item.knowledge_id for item in self.store.candidates_for(self.agent, rich)}
        self.assertIn("basket", ids)
        self.assertIn("well", ids)

    def test_broken_can_afford_does_not_block_candidates(self) -> None:
        class ExplodingTechnology(FakeTechnology):
            def can_afford(self, agent, recipe) -> bool:
                raise RuntimeError("broken")

        world = FakeWorld(ExplodingTechnology())
        ids = {item.knowledge_id for item in self.store.candidates_for(self.agent, world)}
        self.assertEqual(ids, {"stone_axe", "fiber_cord"})


# ---------------------------------------------------------------------------
# 依赖图
# ---------------------------------------------------------------------------
class TestDependencyGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KnowledgeStore()
        self.agent = FakeAgent()

    def test_edges_combine_both_sources(self) -> None:
        expected = {
            ("fiber_cord", "basket"),
            ("stone_axe", "campfire"),
            ("basket", "hut"),
            ("fiber_cord", "well"),
            ("stone_axe", "granary"),
        }
        self.assertEqual(set(self.store.edges()), expected)

        # Knowledge.prerequisites 是第二种边来源：手工知识也能被查到
        self.store.add(
            Knowledge(
                "custom_tool",
                "tool",
                "自定义工具",
                prerequisites=("stone_axe",),
                recipe=Recipe("custom_tool", "自定义工具", {"wood": 1}, {"custom_tool": 1}),
            )
        )
        self.assertIn(("stone_axe", "custom_tool"), self.store.edges())

    def test_dependents_is_transitive(self) -> None:
        self.assertEqual(self.store.direct_dependents("fiber_cord"), {"basket", "well"})
        self.assertEqual(self.store.dependents("fiber_cord"), {"basket", "well", "hut"})
        self.assertEqual(self.store.dependents("hut"), set())
        self.assertEqual(self.store.dependents("ghost"), set())

    def test_depth_counts_only_the_verified_subgraph(self) -> None:
        world = make_world()
        self.assertEqual(self.store.depth(), 0)  # 全是 hypothesis

        verify(self.store, "fiber_cord", world, self.agent)
        self.assertEqual(self.store.depth(), 0)  # basket/well 仍未验证

        verify(self.store, "basket", world, self.agent)
        self.assertEqual(self.store.depth(), 1)  # fiber_cord → basket

        verify(self.store, "hut", world, self.agent)
        self.assertEqual(self.store.depth(), 2)  # fiber_cord → basket → hut
        self.assertEqual(len(self.store.verified_edges()), 2)

    def test_verified_edges_ignore_hypothesis_nodes(self) -> None:
        world = make_world()
        verify(self.store, "fiber_cord", world, self.agent)
        # well/basket 仍是 hypothesis → 已验证子图里没有边
        self.assertEqual(self.store.verified_edges(), [])

        verify(self.store, "well", world, self.agent)
        self.assertEqual(self.store.verified_edges(), [("fiber_cord", "well")])
        # 全图里有 basket 的边，但已验证子图里没有
        self.assertIn(("fiber_cord", "basket"), self.store.edges())
        self.assertNotIn(("fiber_cord", "basket"), self.store.verified_edges())


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KnowledgeStore()
        self.agent = FakeAgent()

    def test_retention(self) -> None:
        # 没有已验证知识 → 1.0（空集约定）
        self.assertEqual(self.store.retention(["stone_axe"]), 1.0)
        self.assertEqual(self.store.retention([]), 1.0)

        world = make_world()
        verify(self.store, "stone_axe", world, self.agent)
        verify(self.store, "fiber_cord", world, self.agent)
        self.assertAlmostEqual(self.store.retention(["stone_axe", "basket", "ghost"]), 0.5)
        self.assertAlmostEqual(self.store.retention(["stone_axe", "fiber_cord"]), 1.0)
        self.assertAlmostEqual(self.store.retention([]), 0.0)
        self.assertAlmostEqual(self.store.retention(None), 0.0)

        self.agent.known_knowledge_ids = ["stone_axe"]
        self.assertAlmostEqual(self.store.retention(self.agent), 0.5)

    def test_innovation_efficiency(self) -> None:
        self.assertEqual(self.store.innovation_efficiency(), 0.0)

        world = make_world()
        verify(self.store, "stone_axe", world, self.agent)  # 1 次尝试 → 1 项验证
        self.assertAlmostEqual(self.store.innovation_efficiency(), 1.0)

        self.store.attempt(self.agent, "fiber_cord", world, 1.0)  # 尝试 +1，未验证
        self.assertAlmostEqual(self.store.innovation_efficiency(), 1 / 2)

        failing = make_world(result=False, reason="no_stone")
        self.store.attempt(self.agent, "campfire", failing, 2.0)  # 失败尝试 +1
        self.assertAlmostEqual(self.store.innovation_efficiency(), 1 / 3)


# ---------------------------------------------------------------------------
# 发现 / 传播
# ---------------------------------------------------------------------------
class TestDiscovery(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KnowledgeStore()
        self.agent = FakeAgent(agent_id=7)

    def test_hypothesize_records_discovery_and_teaches(self) -> None:
        knowledge = self.store.hypothesize(self.agent, "stone_axe", 3.0)
        self.assertIsNotNone(knowledge)
        self.assertEqual(knowledge.discovered_by, 7)
        self.assertAlmostEqual(knowledge.discovered_at, 3.0)
        self.assertIn("stone_axe", self.agent.known_knowledge_ids)

        again = self.store.hypothesize(self.agent, "stone_axe", 9.0)
        self.assertIs(again, knowledge)
        self.assertAlmostEqual(again.discovered_at, 3.0)  # 不覆盖首次发现时间
        self.assertEqual(self.agent.known_knowledge_ids.count("stone_axe"), 1)

    def test_hypothesize_unknown_recipe_returns_none(self) -> None:
        self.assertIsNone(self.store.hypothesize(self.agent, "ghost_tech", 0.0))

    def test_hypothesize_creates_missing_knowledge_in_empty_store(self) -> None:
        custom = Recipe("custom_rope", "自定义绳", {"fiber": 2}, {"cord": 1}, work=1, difficulty=0.1)
        store = KnowledgeStore(catalog={"custom_rope": custom}, bootstrap=False)
        knowledge = store.hypothesize(self.agent, "custom_rope", 1.5)
        self.assertIsNotNone(knowledge)
        self.assertEqual(store.get("custom_rope"), knowledge)
        self.assertEqual(knowledge.status, "hypothesis")
        self.assertIn("custom_rope", self.agent.known_knowledge_ids)

    def test_known_by_reads_agent_ids(self) -> None:
        self.agent.known_knowledge_ids = ["stone_axe", "ghost_tech", "fiber_cord"]
        known = self.store.known_by(self.agent)
        self.assertEqual([item.knowledge_id for item in known], ["fiber_cord", "stone_axe"])


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------
class TestSerialization(unittest.TestCase):
    def _build_scenario(self) -> KnowledgeStore:
        store = KnowledgeStore(seed=11)
        agent = FakeAgent()
        world = make_world()
        verify(store, "stone_axe", world, agent)
        store.attempt(agent, "fiber_cord", world, 1.0)  # 仅 1 次成功（work=2）
        store.mark_invalid("campfire", "岩石枯竭", 7.5)
        store.hypothesize(agent, "well", 2.0)
        store.degraded_attempts = 4
        return store

    def test_to_dict_is_json_serialisable_and_roundtrips(self) -> None:
        store = self._build_scenario()
        payload = store.to_dict()
        json.dumps(payload)  # 必须可写 JSON 日志

        restored = KnowledgeStore.from_dict(payload)
        self.assertEqual(restored.to_dict(), payload)
        self.assertEqual(restored.seed, store.seed)
        self.assertEqual(restored.degraded_attempts, 4)
        self.assertEqual(set(restored.catalog), set(store.catalog))
        self.assertEqual(
            [item.knowledge_id for item in restored.all()],
            [item.knowledge_id for item in store.all()],
        )

    def test_roundtrip_preserves_state_machine_and_history(self) -> None:
        store = self._build_scenario()
        restored = KnowledgeStore.from_dict(store.to_dict())

        stone_axe = restored.get("stone_axe")
        self.assertEqual(stone_axe.status, "verified")
        self.assertTrue(stone_axe.usable())
        self.assertEqual(stone_axe.successful_uses, 1)
        self.assertAlmostEqual(stone_axe.confidence, store.get("stone_axe").confidence)
        self.assertEqual(len(stone_axe.validation_history), 1)
        self.assertTrue(stone_axe.validation_history[0].success)

        fiber_cord = restored.get("fiber_cord")
        self.assertEqual(fiber_cord.status, "hypothesis")
        self.assertEqual(fiber_cord.successful_uses, 1)
        self.assertEqual(fiber_cord.consecutive_successes(), 1)

        campfire = restored.get("campfire")
        self.assertEqual(campfire.status, "invalidated")
        self.assertEqual(campfire.confidence, 0.0)
        self.assertTrue(campfire.validation_history[-1].evidence.startswith("invalidated:"))
        self.assertEqual(campfire.validation_history[-1].agent_id, -1)

        self.assertIn("well", restored.to_dict()["learn_log"][-1]["knowledge_id"])

    def test_roundtrip_keeps_metrics(self) -> None:
        store = self._build_scenario()
        restored = KnowledgeStore.from_dict(store.to_dict())
        self.assertAlmostEqual(restored.retention(["stone_axe"]), store.retention(["stone_axe"]))
        self.assertAlmostEqual(restored.innovation_efficiency(), store.innovation_efficiency())
        self.assertEqual(restored.depth(), store.depth())
        self.assertEqual(restored.edges(), store.edges())


# ---------------------------------------------------------------------------
# 失效入口与摘要
# ---------------------------------------------------------------------------
class TestInvalidationAndSummary(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KnowledgeStore()
        self.agent = FakeAgent()

    def test_mark_invalid_is_recorded_without_touching_use_counters(self) -> None:
        knowledge = verify(self.store, "stone_axe", make_world(), self.agent)
        self.store.mark_invalid("stone_axe", "世界规则变化", 12.0)

        self.assertEqual(knowledge.status, "invalidated")
        self.assertFalse(knowledge.usable())
        self.assertEqual(knowledge.confidence, 0.0)
        self.assertEqual(knowledge.successful_uses, 1)
        self.assertEqual(knowledge.failed_uses, 0)
        audit = knowledge.validation_history[-1]
        self.assertEqual(audit.agent_id, -1)
        self.assertEqual(audit.evidence, "invalidated:世界规则变化")

    def test_mark_invalid_unknown_id_is_a_no_op(self) -> None:
        self.store.mark_invalid("ghost", "无", 0.0)  # 不抛异常
        self.assertIsNone(self.store.get("ghost"))

    def test_summary_zh_reports_counts(self) -> None:
        self.store.attempt(self.agent, "stone_axe", make_world(), 0.0)
        summary = self.store.summary_zh()
        self.assertIn("知识库", summary)
        self.assertIn("已验证1", summary)
        self.assertIn("依赖边", summary)


if __name__ == "__main__":
    unittest.main()
