"""``newworld.innovation`` 的单元测试（升级契约 §3 验收：test_innovation）。

运行::

    cd newworldpro
    $env:TEMP=(Resolve-Path .tmp).Path
    python -m unittest tests.test_innovation -v

测试策略
--------
* 只使用 **fake agent / fake technology / fake world**（duck typing）以及真实的
  ``Inventory`` / ``KnowledgeStore`` / ``Recipe``——不依赖 Lead 尚未落地的
  ``Technology.register_custom_effect`` 集成代码（用桩验证"缺接口时返回
  ``no_effect_backend`` 且不扣材料"）。
* 随机性用 ``FixedRng`` 注入（恒成功 / 恒失败），因此断言的是机制而不是运气。
* 最后一个用例在**真实 World** 上验证 P3 早期降级路径；若 Lead 已经实现
  effect 后端则 skip。
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from newworld.innovation import (
    ATTRIBUTE_CATALOG,
    EFFECT_KEYS,
    EFFECT_RULES,
    KNOWN_TAGS,
    MAX_INNOVATIONS,
    NO_EFFECT_BACKEND_REASON,
    Innovation,
    InnovationLab,
    Proposal,
    RecipeAttribute,
    attribute,
    combination_difficulty,
    compatible_pair,
    derive_effects,
    material_cost,
    register_attribute,
    rule_for,
)
from newworld.knowledge import RECIPE_CATALOG, KnowledgeStore
from newworld.technology import Inventory

# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------
class FakeAgent:
    """最小个体：agent_id / known_knowledge_ids / inventory / nearby_facilities。"""

    def __init__(
        self,
        agent_id: int = 7,
        known: tuple[str, ...] | list[str] = (),
        *,
        tool: str | None = None,
        wood: int = 0,
        stone: int = 0,
        fiber: int = 0,
        cord: int = 0,
        basket: int = 0,
        capacity_base: int = 8,
    ) -> None:
        self.agent_id = int(agent_id)
        self.known_knowledge_ids: list[str] = list(known)
        self.inventory = Inventory(
            wood=wood,
            stone=stone,
            fiber=fiber,
            cord=cord,
            tool=tool,
            basket=basket,
            capacity_base=capacity_base,
        )
        self.nearby_facilities: list[object] = []


class FixedRng:
    """固定结果的随机源：``random()`` 恒返回同一个值（可复现成功/失败）。

    失败概率 = 组合难度（默认 0.35）：``value=0.99`` 恒成功（0.99 >= 0.35），
    ``value=0.0`` 恒失败（0.0 < 0.35）。随机性只属于"实验是否暴露缺陷"，
    与"名字/效果随机"无关。
    """

    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return self.value


class FakeTechnology:
    """含 ``register_custom_effect`` 的世界规则桩（记录调用）。"""

    def __init__(self) -> None:
        self.custom_effects: dict[str, dict[str, float]] = {}
        self.register_calls: list[tuple[str, dict[str, float]]] = []
        self.afford_calls: list[object] = []

    def can_afford(self, agent, recipe) -> bool:
        self.afford_calls.append(recipe)
        return all(
            agent.inventory.has(kind, amount)
            for kind, amount in dict(recipe.inputs).items()
        )

    def register_custom_effect(self, tool_name: str, effects: dict[str, float]) -> None:
        self.register_calls.append((tool_name, dict(effects)))
        self.custom_effects[str(tool_name)] = dict(effects)

    # -- 模拟集成层：效果必须真的改变世界数值 -------------------------
    def harvest_yield(self, agent) -> int:
        """采集产量：1 + 手上工具已注册的 ``harvest_bonus``。"""
        effects = self.custom_effects.get(agent.inventory.tool or "", {})
        return 1 + int(effects.get("harvest_bonus", 0.0))

    def capacity_for(self, agent) -> int:
        """载重上限：Inventory.capacity() + 手上工具已注册的 capacity_bonus。"""
        effects = self.custom_effects.get(agent.inventory.tool or "", {})
        return agent.inventory.capacity() + int(effects.get("capacity_bonus", 0.0))


class FakeTechnologyNoEffects:
    """故意**没有** ``register_custom_effect`` 的世界规则桩（P3 早期）。"""

    def __init__(self) -> None:
        self.afford_calls: list[object] = []

    def can_afford(self, agent, recipe) -> bool:
        self.afford_calls.append(recipe)
        return all(
            agent.inventory.has(kind, amount)
            for kind, amount in dict(recipe.inputs).items()
        )


class FakeTechnologyBroken(FakeTechnology):
    """注册效果时抛异常，用于验证"材料回滚"。"""

    def register_custom_effect(self, tool_name: str, effects: dict[str, float]) -> None:
        raise RuntimeError("effect backend exploded")


class FakeWorld:
    def __init__(self, technology, knowledge=None, t: float = 12.5) -> None:
        self.technology = technology
        self.knowledge = knowledge
        self.t = float(t)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def make_store(*verified_ids: str) -> KnowledgeStore:
    """构造真实 KnowledgeStore，并把指定技术置为 ``verified``。"""
    store = KnowledgeStore(seed=0)
    for knowledge_id in verified_ids:
        record = store.get(knowledge_id)
        assert record is not None, knowledge_id
        record.status = "verified"
    return store


def make_agent(*verified_ids: str, **kwargs) -> FakeAgent:
    kwargs.setdefault("tool", "axe")
    return FakeAgent(known=list(verified_ids), **kwargs)


def first_proposal_for(lab: InnovationLab, agent: FakeAgent, store: KnowledgeStore) -> Proposal:
    proposals = lab.proposals(agent, store)
    assert proposals, "测试前置：至少应有一条提案"
    return proposals[0]


# ---------------------------------------------------------------------------
# 1) 属性 / 规则表：世界规则本身
# ---------------------------------------------------------------------------
class TestAttributesAndRules(unittest.TestCase):
    def test_attribute_catalog_covers_every_real_recipe(self) -> None:
        for recipe_id in RECIPE_CATALOG:
            attr = attribute(recipe_id)
            self.assertIsNotNone(attr, recipe_id)
            assert attr is not None
            self.assertEqual(attr.knowledge_id, recipe_id)
            self.assertGreaterEqual(len(attr.tags), 1)
            for tag in attr.tags:
                self.assertIn(tag, KNOWN_TAGS)
            for key, value in attr.base_effects.items():
                self.assertIn(key, EFFECT_KEYS)
                self.assertGreater(value, 0.0)
        self.assertIsNone(attribute("mystery_tech"))

    def test_rule_table_matches_frozen_contract(self) -> None:
        expected = {
            frozenset({"edge", "binder"}): {"harvest_bonus": 1.0},
            frozenset({"binder", "container"}): {"capacity_bonus": 4.0},
            frozenset({"heat", "container"}): {"energy_regen": 0.15},
            frozenset({"binder", "heat"}): {"energy_regen": 0.10, "water_regen": 0.05},
            frozenset({"container", "structure"}): {"capacity_bonus": 2.0},
        }
        self.assertEqual(len(EFFECT_RULES), len(expected))
        observed = {rule.tags: dict(rule.effects) for rule in EFFECT_RULES}
        self.assertEqual(observed, expected)
        for rule in EFFECT_RULES:
            self.assertEqual(len(rule.tags), 2)
            self.assertTrue(rule.output_tool)
            self.assertTrue(rule.name_zh)
            for key in rule.effects:
                self.assertIn(key, EFFECT_KEYS)
        # 规则产出工具名互不重复（否则注册会互相覆盖）
        tools = [rule.output_tool for rule in EFFECT_RULES]
        self.assertEqual(len(tools), len(set(tools)))

    def test_derive_effects_is_deterministic_from_tags(self) -> None:
        cases = {
            ("stone_axe", "fiber_cord"): {"harvest_bonus": 1.0},
            ("basket", "fiber_cord"): {"capacity_bonus": 4.0},
            ("campfire", "basket"): {"energy_regen": 0.15},
            ("campfire", "fiber_cord"): {"energy_regen": 0.10, "water_regen": 0.05},
            ("hut", "basket"): {"capacity_bonus": 2.0},
        }
        for (left, right), expected in cases.items():
            attr_left, attr_right = attribute(left), attribute(right)
            effects = derive_effects(attr_left, attr_right)
            self.assertEqual(effects, expected, (left, right))
            # 顺序无关、多次调用一致
            self.assertEqual(derive_effects(attr_right, attr_left), expected)
            self.assertEqual(derive_effects(attr_left, attr_right), expected)

    def test_derived_effects_are_copies_not_shared_state(self) -> None:
        effects = derive_effects(attribute("stone_axe"), attribute("fiber_cord"))
        assert effects is not None
        effects["harvest_bonus"] = 999.0
        again = derive_effects(attribute("stone_axe"), attribute("fiber_cord"))
        self.assertEqual(again, {"harvest_bonus": 1.0})

    def test_incompatible_pairs_are_rejected(self) -> None:
        # edge+structure / edge+heat / container+container 都不在冻结规则表里
        for left, right in (("stone_axe", "hut"), ("stone_axe", "campfire"), ("basket", "basket")):
            attr_left, attr_right = attribute(left), attribute(right)
            self.assertIsNone(rule_for(attr_left, attr_right), (left, right))
            self.assertIsNone(derive_effects(attr_left, attr_right), (left, right))
            self.assertIsNone(compatible_pair(attr_left, attr_right), (left, right))
        self.assertIsNone(rule_for(None, attribute("stone_axe")))
        self.assertIsNone(derive_effects(attribute("stone_axe"), None))
        # (basket, hut) 是合法相容对，作为对照
        self.assertIsNotNone(rule_for(attribute("basket"), attribute("hut")))

    def test_compatible_pair_reports_matched_tags(self) -> None:
        self.assertEqual(
            compatible_pair(attribute("stone_axe"), attribute("fiber_cord")),
            ("edge", "binder"),
        )
        self.assertEqual(
            compatible_pair(attribute("basket"), attribute("campfire")),
            ("container", "heat"),
        )

    def test_combination_difficulty_counts_distinct_tags(self) -> None:
        # 两条单 tag 技术 -> 0.15 + 0.1*2 = 0.35
        self.assertAlmostEqual(
            combination_difficulty((attribute("stone_axe"), attribute("fiber_cord"))),
            0.35,
        )
        self.assertAlmostEqual(combination_difficulty([]), 0.15)
        # 多 tag 属性：不同 tag 数增加 -> 难度增加
        multi = RecipeAttribute("multi", ("edge", "binder", "heat"), {})
        self.assertAlmostEqual(
            combination_difficulty((multi, attribute("stone_axe"))),
            0.15 + 0.1 * 3,
        )

    def test_material_cost_is_minimal_prototype(self) -> None:
        """成本 = 第一组件 inputs 中字母序第一个材料键，计 1（最简原型试制）。"""
        store = make_store(
            "stone_axe", "fiber_cord", "basket", "hut", "campfire", "granary"
        )
        # 第一组件 stone_axe keys {"wood","stone"} 排序 -> "stone"
        self.assertEqual(
            material_cost(store, ("stone_axe", "fiber_cord")),
            {"stone": 1},
        )
        # 契约例子：第一组件 campfire keys {"stone","wood"} -> "stone"
        self.assertEqual(
            material_cost(store, ("campfire", "fiber_cord")),
            {"stone": 1},
        )
        # 契约例子：granary / campfire 的 keys 都是 {"stone","wood"} -> "stone"
        self.assertEqual(material_cost(store, ("granary", "campfire")), {"stone": 1})
        self.assertEqual(material_cost(store, ("campfire", "granary")), {"stone": 1})
        # 第一组件 basket keys {"cord","wood"} 排序 -> "cord"
        self.assertEqual(material_cost(store, ("basket", "fiber_cord")), {"cord": 1})
        # 第一组件 hut keys {"cord","stone","wood"} 排序 -> "cord"
        self.assertEqual(material_cost(store, ("hut", "basket")), {"cord": 1})
        # 顺序确定：第一组件不同 -> 成本键不同（组件顺序由 Proposal 排序保证）
        self.assertEqual(material_cost(store, ("fiber_cord", "stone_axe")), {"fiber": 1})
        # 第一组件没有配方 -> 回退到第二组件同样取法
        no_inputs = {
            "mystery": SimpleNamespace(
                knowledge_id="mystery", name_zh="无配方", status="verified", recipe=None
            ),
            "basket": store.get("basket"),
        }
        self.assertEqual(material_cost(no_inputs, ("mystery", "basket")), {"cord": 1})
        # 取不到任何配方时使用 fallback（原样采用，不凭空造成本）
        self.assertEqual(
            material_cost(None, ("stone_axe",), fallback={"wood": 1, "stone": 1}),
            {"wood": 1, "stone": 1},
        )
        self.assertEqual(material_cost(None, ("stone_axe",)), {})

    def test_register_attribute_validates_and_extends_catalog(self) -> None:
        attr = RecipeAttribute("test_new_tech", ("edge", "binder"), {"harvest_bonus": 0.5})
        try:
            register_attribute(attr)
            self.assertIs(attribute("test_new_tech"), attr)
            with self.assertRaises(ValueError):
                register_attribute(RecipeAttribute("bad_tag", ("nope",), {}))
            with self.assertRaises(ValueError):
                register_attribute(RecipeAttribute("bad_key", ("edge",), {"sparkle": 1.0}))
            with self.assertRaises(ValueError):
                register_attribute(RecipeAttribute("", ("edge",), {}))
        finally:
            ATTRIBUTE_CATALOG.pop("test_new_tech", None)
        self.assertIsNone(attribute("test_new_tech"))


# ---------------------------------------------------------------------------
# 2) 提案：只组合"掌握且 verified"的相容技术
# ---------------------------------------------------------------------------
class TestProposals(unittest.TestCase):
    def test_requires_at_least_two_verified_techniques(self) -> None:
        store = make_store("stone_axe")
        agent = make_agent("stone_axe")
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        self.assertEqual(lab.proposals(agent, store), [])
        # 空知识库 / 空已知集合也是 no-op
        empty = make_store()
        self.assertEqual(lab.proposals(FakeAgent(known=[]), empty), [])

    def test_only_verified_components_are_proposed(self) -> None:
        store = make_store("stone_axe")  # fiber_cord 仍是 hypothesis
        agent = make_agent("stone_axe", "fiber_cord")
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        self.assertEqual(lab.proposals(agent, store), [])

    def test_site_condition_is_required(self) -> None:
        store = make_store("stone_axe", "fiber_cord")
        agent = make_agent("stone_axe", "fiber_cord", tool=None)
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        self.assertEqual(lab.proposals(agent, store), [])
        # 2) 附近有活跃工作台 -> 满足
        agent.nearby_facilities = [SimpleNamespace(kind="workshop", active=True)]
        self.assertTrue(lab.proposals(agent, store))
        # 3) 集成层显式钩子 can_innovate 优先
        other = make_agent("stone_axe", "fiber_cord", tool=None)
        other.can_innovate = lambda: True  # type: ignore[attr-defined]
        self.assertTrue(lab.proposals(other, store))

    def test_proposals_are_deterministic_and_attribute_driven(self) -> None:
        store = make_store("stone_axe", "fiber_cord")
        agent = make_agent("stone_axe", "fiber_cord", wood=4, stone=2, fiber=6)
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        first = lab.proposals(agent, store)
        second = lab.proposals(agent, store)
        self.assertEqual(len(first), 1)
        self.assertEqual([item.to_dict() for item in first], [item.to_dict() for item in second])

        proposal = first[0]
        self.assertEqual(proposal.components, ("fiber_cord", "stone_axe"))
        self.assertEqual(proposal.output_tool, "hafted_blade")
        self.assertEqual(proposal.effects, {"harvest_bonus": 1.0})
        self.assertEqual(proposal.inputs, {"fiber": 1})
        self.assertEqual(proposal.agent_id, 7)
        self.assertIn("装柄刃器", proposal.name_zh)
        self.assertIn("石斧", proposal.name_zh)
        self.assertIn("纤维绳", proposal.name_zh)
        self.assertIn("edge", proposal.rationale)
        self.assertIn("binder", proposal.rationale)
        self.assertIn("0.35", proposal.rationale)
        # 效果必须等于属性确定性推导的结果
        self.assertEqual(
            proposal.effects,
            derive_effects(attribute("stone_axe"), attribute("fiber_cord")),
        )
        self.assertAlmostEqual(lab.difficulty(proposal), 0.35)

    def test_incompatible_verified_pair_produces_no_proposal(self) -> None:
        store = make_store("stone_axe", "hut")  # edge + structure 无相容规则
        agent = make_agent("stone_axe", "hut")
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        self.assertEqual(lab.proposals(agent, store), [])

    def test_limit_and_rule_deduplication(self) -> None:
        verified = ("stone_axe", "fiber_cord", "basket", "campfire", "hut", "well")
        store = make_store(*verified)
        agent = make_agent(*verified, wood=9, stone=5, fiber=9, cord=9)
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))

        proposals = lab.proposals(agent, store, limit=99)
        tools = [item.output_tool for item in proposals]
        # 同一规则只提一次（binder+container 有 basket/well 两条，container+heat 有两条…）
        self.assertEqual(len(tools), len(set(tools)))
        self.assertTrue(set(tools) <= {rule.output_tool for rule in EFFECT_RULES})
        # 5 条规则全部可用：edge+binder, binder+container, container+heat,
        # heat+binder, container+structure（每规则只提一次）
        self.assertEqual(len(proposals), 5)

        limited = lab.proposals(agent, store, limit=2)
        self.assertEqual(len(limited), 2)
        self.assertEqual(
            [item.to_dict() for item in limited],
            [item.to_dict() for item in proposals[:2]],
        )

    def test_affordable_only_filters_by_inventory(self) -> None:
        store = make_store("stone_axe", "fiber_cord")
        # 最简原型成本 {fiber:1}（第一组件 fiber_cord 的字母序第一个材料键）
        poor = make_agent("stone_axe", "fiber_cord", wood=0, stone=0, fiber=0)
        exact = make_agent(
            "stone_axe", "fiber_cord", wood=0, stone=0, fiber=1, capacity_base=20
        )
        rich = make_agent("stone_axe", "fiber_cord", wood=4, stone=2, fiber=6)
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        self.assertEqual(lab.proposals(poor, store, affordable_only=True), [])
        # 只要 1 单位基础材料就负担得起（新的可达性验收点）
        self.assertEqual(len(lab.proposals(exact, store, affordable_only=True)), 1)
        self.assertEqual(len(lab.proposals(rich, store, affordable_only=True)), 1)
        # 默认不按库存过滤（材料检查留给 attempt，契约语义）
        self.assertEqual(len(lab.proposals(poor, store)), 1)


# ---------------------------------------------------------------------------
# 3) 实验：前置 / 材料 / 难度 / 注册 / 记录
# ---------------------------------------------------------------------------
class TestAttempt(unittest.TestCase):
    def setUp(self) -> None:
        self.store = make_store("stone_axe", "fiber_cord")
        self.agent = make_agent(
            "stone_axe", "fiber_cord", wood=2, stone=1, fiber=3, capacity_base=20
        )
        self.technology = FakeTechnology()
        self.world = FakeWorld(self.technology, self.store, t=42.0)
        self.lab = InnovationLab(seed=0, rng=FixedRng(0.99))  # 恒成功
        self.proposal = first_proposal_for(self.lab, self.agent, self.store)

    def test_success_deducts_materials_registers_effect_and_records(self) -> None:
        ok, reason, innovation = self.lab.attempt(self.agent, self.proposal, self.world)
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "ok")
        self.assertIsInstance(innovation, Innovation)

        # 1) 真实扣材料（最简原型成本 = 第一组件 fiber_cord 的 "fiber" 1 单位）
        self.assertEqual(self.agent.inventory.wood, 2)
        self.assertEqual(self.agent.inventory.stone, 1)
        self.assertEqual(self.agent.inventory.fiber, 2)

        # 2) 效果经 technology 注册可见
        self.assertEqual(
            self.technology.register_calls,
            [("hafted_blade", {"harvest_bonus": 1.0})],
        )
        self.assertEqual(
            self.technology.custom_effects["hafted_blade"], {"harvest_bonus": 1.0}
        )
        self.assertEqual(self.lab.effects_for("hafted_blade"), {"harvest_bonus": 1.0})
        self.assertTrue(self.lab.is_registered("hafted_blade"))

        # 3) Innovation 记录真实字段
        assert innovation is not None
        self.assertEqual(innovation.innovation_id, "innovation_001")
        self.assertEqual(innovation.invented_by, 7)
        self.assertEqual(innovation.invented_at, 42.0)
        self.assertEqual(innovation.components, ("fiber_cord", "stone_axe"))
        self.assertEqual(innovation.effects, {"harvest_bonus": 1.0})
        self.assertEqual(innovation.successful_uses, 0)
        self.assertIn("装柄刃器", innovation.name_zh)
        self.assertEqual(self.lab.innovations(), [innovation])

        # 4) 审计记录真实成本/原因
        entry = self.lab.history[-1]
        self.assertTrue(entry["ok"])
        self.assertEqual(entry["reason"], "ok")
        self.assertEqual(entry["inputs"], {"fiber": 1})
        self.assertEqual(entry["agent_id"], 7)
        self.assertEqual(entry["innovation_id"], "innovation_001")

        # 5) 世界时间/知识状态未被本模块改动（只改库存 + 委托注册）
        self.assertEqual(self.world.t, 42.0)
        self.assertEqual(self.store.get("stone_axe").status, "verified")

    def test_success_equips_new_tool_on_inventor(self) -> None:
        """成功后新工具必须真的挂到 inventer 的 inventory 上，效果才会生效。"""
        self.assertEqual(self.agent.inventory.tool, "axe")
        ok, _, innovation = self.lab.attempt(self.agent, self.proposal, self.world)
        self.assertTrue(ok)
        assert innovation is not None
        self.assertEqual(self.agent.inventory.tool, "hafted_blade")
        self.assertTrue(self.lab.history[-1]["equipped"])
        # 真实带工具采集：1 + harvest_bonus 1 = 2
        self.assertEqual(self.technology.harvest_yield(self.agent), 2)

    def test_non_harvest_tool_does_not_replace_axe(self) -> None:
        """纯恢复/载重类新工具不得把斧头换掉（否则创新让世界变差）。"""
        store = make_store("campfire", "fiber_cord")
        agent = make_agent(
            "campfire", "fiber_cord", wood=2, stone=1, fiber=3, capacity_base=20
        )
        technology = FakeTechnology()
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        proposal = lab.proposals(agent, store)[0]
        self.assertEqual(proposal.effects, {"energy_regen": 0.10, "water_regen": 0.05})
        ok, reason, _ = lab.attempt(agent, proposal, FakeWorld(technology, store))
        self.assertTrue(ok, reason)
        self.assertEqual(agent.inventory.tool, "axe")  # 保留斧头
        self.assertFalse(lab.history[-1]["equipped"])
        # 但效果本身仍然真实注册到世界
        self.assertEqual(
            technology.custom_effects[proposal.output_tool],
            {"energy_regen": 0.10, "water_regen": 0.05},
        )

    def test_equip_when_agent_has_no_tool(self) -> None:
        """没有工具时任何新工具都会装备（新鲜工具总是比空手强）。"""
        store = make_store("campfire", "fiber_cord")
        agent = make_agent(
            "campfire", "fiber_cord", tool=None, wood=2, stone=1, fiber=3, capacity_base=20
        )
        agent.can_innovate = lambda: True  # type: ignore[attr-defined]
        technology = FakeTechnology()
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        proposal = lab.proposals(agent, store)[0]
        ok, reason, _ = lab.attempt(agent, proposal, FakeWorld(technology, store))
        self.assertTrue(ok, reason)
        self.assertEqual(agent.inventory.tool, proposal.output_tool)
        self.assertTrue(lab.history[-1]["equipped"])

    def test_equip_can_be_disabled(self) -> None:
        ok, _, _ = self.lab.attempt(self.agent, self.proposal, self.world, equip=False)
        self.assertTrue(ok)
        self.assertEqual(self.agent.inventory.tool, "axe")
        self.assertFalse(self.lab.history[-1]["equipped"])

    def test_world_effect_is_queryable_through_technology(self) -> None:
        self.assertEqual(self.technology.harvest_yield(self.agent), 1)
        ok, _, _ = self.lab.attempt(self.agent, self.proposal, self.world)
        self.assertTrue(ok)
        self.agent.inventory.tool = "hafted_blade"
        self.assertEqual(self.technology.harvest_yield(self.agent), 2)  # 采集 +1

        # 另一条规则：binder+container -> capacity_bonus +4
        store = make_store("basket", "fiber_cord")
        agent = make_agent(
            "basket", "fiber_cord", wood=2, cord=1, fiber=3, capacity_base=20
        )
        technology = FakeTechnology()
        world = FakeWorld(technology, store)
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        proposal = lab.proposals(agent, store)[0]
        ok, reason, _ = lab.attempt(agent, proposal, world)
        self.assertTrue(ok, reason)
        self.assertEqual(proposal.output_tool, "woven_creel")
        agent.inventory.tool = proposal.output_tool
        self.assertEqual(technology.capacity_for(agent), agent.inventory.capacity() + 4)

    def test_failure_consumes_materials_and_records_no_innovation(self) -> None:
        lab = InnovationLab(seed=0, rng=FixedRng(0.0))  # 0.0 < 0.35 -> 失败
        ok, reason, innovation = lab.attempt(self.agent, self.proposal, self.world)
        self.assertFalse(ok)
        self.assertEqual(reason, "defect")
        self.assertIsNone(innovation)
        # 失败也真实消耗材料（最简原型成本：第一组件 fiber_cord 的 "fiber" 1 单位）
        self.assertEqual(self.agent.inventory.wood, 2)
        self.assertEqual(self.agent.inventory.stone, 1)
        self.assertEqual(self.agent.inventory.fiber, 2)
        # 没注册效果、没记录创新，但有审计记录
        self.assertEqual(self.technology.register_calls, [])
        self.assertEqual(self.lab.innovations(), [])
        self.assertEqual(self.agent.inventory.tool, "axe")  # 失败不装备新工具
        self.assertFalse(lab.history[-1]["ok"])
        self.assertEqual(lab.history[-1]["reason"], "defect")
        self.assertIsNone(lab.history[-1]["innovation_id"])

    def test_missing_material_returns_false_and_deducts_nothing(self) -> None:
        poor = make_agent(
            "stone_axe", "fiber_cord", wood=0, stone=0, fiber=0, capacity_base=20
        )
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        proposal = lab.proposals(poor, self.store)[0]
        before = (poor.inventory.wood, poor.inventory.stone, poor.inventory.fiber)
        ok, reason, innovation = lab.attempt(poor, proposal, self.world)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_material")
        self.assertIsNone(innovation)
        self.assertEqual(
            (poor.inventory.wood, poor.inventory.stone, poor.inventory.fiber), before
        )
        self.assertEqual(lab.history, [])  # 前置失败不写"实验"记录
        self.assertEqual(lab.rng.calls, 0)  # 也没做难度判定
        self.assertEqual(self.technology.register_calls, [])

    def test_unverified_component_rejected(self) -> None:
        store = make_store("stone_axe")  # fiber_cord 未验证
        agent = make_agent(
            "stone_axe", "fiber_cord", wood=2, stone=1, fiber=3, capacity_base=20
        )
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        proposal = Proposal(
            proposal_id="proposal_manual",
            agent_id=agent.agent_id,
            name_zh="手工提案",
            components=("fiber_cord", "stone_axe"),
            inputs={"fiber": 1},
            output_tool="hafted_blade",
            effects={"harvest_bonus": 1.0},
            rationale="测试用",
        )
        ok, reason, innovation = lab.attempt(agent, proposal, FakeWorld(self.technology, store))
        self.assertFalse(ok)
        self.assertEqual(reason, "component_not_verified")
        self.assertIsNone(innovation)
        self.assertEqual(agent.inventory.wood, 2)
        self.assertEqual(agent.inventory.fiber, 3)

    def test_unknown_component_not_known_by_agent_rejected(self) -> None:
        agent = make_agent("stone_axe", wood=2, stone=1, fiber=3, capacity_base=20)
        proposal = first_proposal_for(
            InnovationLab(seed=0, rng=FixedRng(0.99)),
            make_agent("stone_axe", "fiber_cord"),
            self.store,
        )
        ok, reason, _ = InnovationLab(seed=0, rng=FixedRng(0.99)).attempt(
            agent, proposal, self.world
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "component_not_known")

    def test_incompatible_components_rejected(self) -> None:
        store = make_store("stone_axe", "hut")
        agent = make_agent("stone_axe", "hut", wood=8, stone=4, cord=4, capacity_base=40)
        proposal = Proposal(
            proposal_id="proposal_manual",
            agent_id=agent.agent_id,
            name_zh="不相容提案",
            components=("hut", "stone_axe"),
            inputs={"wood": 1, "stone": 1, "cord": 1},
            output_tool="hafted_blade",
            effects={"harvest_bonus": 1.0},
            rationale="edge + structure 不相容",
        )
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        ok, reason, _ = lab.attempt(agent, proposal, FakeWorld(self.technology, store))
        self.assertFalse(ok)
        self.assertEqual(reason, "incompatible_tags")
        self.assertEqual(self.technology.register_calls, [])

    def test_unknown_attribute_rejected(self) -> None:
        mystery = SimpleNamespace(
            knowledge_id="mystery_tech",
            name_zh="神秘技术",
            status="verified",
            recipe=None,
        )
        knowledge = {"mystery_tech": mystery, "stone_axe": self.store.get("stone_axe")}
        agent = make_agent("mystery_tech", "stone_axe", wood=4, stone=2, capacity_base=20)
        proposal = Proposal(
            proposal_id="proposal_manual",
            agent_id=agent.agent_id,
            name_zh="未知属性提案",
            components=("mystery_tech", "stone_axe"),
            inputs={"wood": 1, "stone": 1},
            output_tool="hafted_blade",
            effects={"harvest_bonus": 1.0},
            rationale="没有属性声明",
        )
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        ok, reason, _ = lab.attempt(agent, proposal, FakeWorld(self.technology, knowledge))
        self.assertFalse(ok)
        self.assertEqual(reason, "unknown_attribute")

    def test_without_effect_backend_returns_reason_and_keeps_materials(self) -> None:
        technology = FakeTechnologyNoEffects()
        world = FakeWorld(technology, self.store)
        before = (
            self.agent.inventory.wood,
            self.agent.inventory.stone,
            self.agent.inventory.fiber,
        )
        ok, reason, innovation = self.lab.attempt(self.agent, self.proposal, world)
        self.assertFalse(ok)
        self.assertEqual(reason, NO_EFFECT_BACKEND_REASON)
        self.assertIsNone(innovation)
        self.assertEqual(
            (self.agent.inventory.wood, self.agent.inventory.stone, self.agent.inventory.fiber),
            before,
        )
        self.assertEqual(self.lab.history, [])
        # 世界没有 technology 时同样降级（不抛异常、不扣材料）
        ok, reason, _ = self.lab.attempt(
            self.agent, self.proposal, FakeWorld(None, self.store)
        )
        self.assertFalse(ok)
        self.assertEqual(reason, NO_EFFECT_BACKEND_REASON)
        ok, reason, _ = self.lab.attempt(self.agent, self.proposal, None)
        self.assertFalse(ok)
        self.assertEqual(reason, NO_EFFECT_BACKEND_REASON)

    def test_effect_backend_error_rolls_back_materials(self) -> None:
        technology = FakeTechnologyBroken()
        world = FakeWorld(technology, self.store)
        before = (
            self.agent.inventory.wood,
            self.agent.inventory.stone,
            self.agent.inventory.fiber,
        )
        ok, reason, innovation = self.lab.attempt(self.agent, self.proposal, world)
        self.assertFalse(ok)
        self.assertEqual(reason, "effect_backend_error:RuntimeError")
        self.assertIsNone(innovation)
        self.assertEqual(self.lab.innovations(), [])
        self.assertEqual(
            (self.agent.inventory.wood, self.agent.inventory.stone, self.agent.inventory.fiber),
            before,
        )
        self.assertEqual(self.lab.history[-1]["reason"], "effect_backend_error:RuntimeError")

    def test_second_attempt_at_same_pair_is_already_invented(self) -> None:
        ok, _, _ = self.lab.attempt(self.agent, self.proposal, self.world)
        self.assertTrue(ok)
        technology_calls = len(self.technology.register_calls)
        ok, reason, innovation = self.lab.attempt(self.agent, self.proposal, self.world)
        self.assertFalse(ok)
        self.assertEqual(reason, "already_invented")
        self.assertIsNone(innovation)
        self.assertEqual(len(self.technology.register_calls), technology_calls)

    def test_invalid_proposal_structures_rejected(self) -> None:
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        ok, reason, _ = lab.attempt(self.agent, "not-a-proposal", self.world)  # type: ignore[arg-type]
        self.assertFalse(ok)
        self.assertEqual(reason, "invalid_proposal")
        single = Proposal(
            proposal_id="p",
            agent_id=7,
            name_zh="单组件",
            components=("stone_axe",),
            inputs={},
            output_tool="hafted_blade",
            effects={},
            rationale="",
        )
        ok, reason, _ = lab.attempt(self.agent, single, self.world)
        self.assertFalse(ok)
        self.assertEqual(reason, "invalid_components")

    def test_two_labs_with_same_seed_share_outcome_sequence(self) -> None:
        verified = ("stone_axe", "fiber_cord", "basket", "hut")

        def run() -> tuple[list[str], int]:
            store = make_store(*verified)
            agent = make_agent(
                *verified, wood=9, stone=3, fiber=9, cord=3, capacity_base=60
            )
            technology = FakeTechnology()
            world = FakeWorld(technology, store, t=1.0)
            lab = InnovationLab(seed=42)
            reasons: list[str] = []
            for proposal in lab.proposals(agent, store, limit=9):
                ok, reason, _ = lab.attempt(agent, proposal, world)
                reasons.append(reason)
            return reasons, len(lab.innovations())

        self.assertEqual(run(), run())
        reasons, _ = run()
        self.assertTrue(reasons)  # 至少跑过实验


# ---------------------------------------------------------------------------
# 4) 上限 / 序列化 / 摘要
# ---------------------------------------------------------------------------
class TestLimitsAndPersistence(unittest.TestCase):
    def _invent_three(
        self,
    ) -> tuple[InnovationLab, FakeAgent, KnowledgeStore, FakeTechnology, FakeWorld]:
        verified = ("stone_axe", "fiber_cord", "basket", "hut")
        store = make_store(*verified)
        agent = make_agent(*verified, wood=9, stone=3, fiber=9, cord=3, capacity_base=60)
        technology = FakeTechnology()
        world = FakeWorld(technology, store, t=5.0)
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        proposals = {item.output_tool: item for item in lab.proposals(agent, store, limit=9)}
        self.assertEqual(set(proposals), {"hafted_blade", "woven_creel", "storage_frame"})
        for tool in ("hafted_blade", "woven_creel", "storage_frame"):
            ok, reason, _ = lab.attempt(agent, proposals[tool], world)
            self.assertTrue(ok, reason)
        return lab, agent, store, technology, world

    def test_max_three_innovations_per_world(self) -> None:
        lab, agent, store, technology, world = self._invent_three()
        self.assertEqual(len(lab.innovations()), MAX_INNOVATIONS)
        # 超过上限后 proposals 返回空（契约）
        self.assertEqual(lab.proposals(agent, store), [])
        # attempt 也不允许第 4 项，且不扣材料
        extra = Proposal(
            proposal_id="proposal_extra",
            agent_id=agent.agent_id,
            name_zh="超额提案",
            components=("campfire", "basket"),
            inputs={},
            output_tool="fire_vessel",
            effects={"energy_regen": 0.15},
            rationale="超过上限",
        )
        before = (agent.inventory.wood, agent.inventory.stone, agent.inventory.fiber)
        ok, reason, innovation = lab.attempt(agent, extra, world)
        self.assertFalse(ok)
        self.assertEqual(reason, "max_innovations")
        self.assertIsNone(innovation)
        self.assertEqual(
            (agent.inventory.wood, agent.inventory.stone, agent.inventory.fiber), before
        )
        self.assertEqual(len(technology.register_calls), MAX_INNOVATIONS)

    def test_record_use_counts_successful_uses(self) -> None:
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        store = make_store("stone_axe", "fiber_cord")
        agent = make_agent(
            "stone_axe", "fiber_cord", wood=2, stone=1, fiber=3, capacity_base=20
        )
        proposal = first_proposal_for(lab, agent, store)
        ok, _, innovation = lab.attempt(agent, proposal, FakeWorld(FakeTechnology(), store))
        self.assertTrue(ok)
        assert innovation is not None
        self.assertEqual(lab.record_use(innovation), 1)
        self.assertEqual(lab.record_use(innovation.innovation_id, uses=2), 3)
        self.assertEqual(innovation.successful_uses, 3)
        self.assertEqual(lab.record_use("innovation_999"), 0)

    def test_to_dict_from_dict_roundtrip(self) -> None:
        lab, agent, store, technology, world = self._invent_three()
        lab.record_use(lab.innovations()[0], uses=2)
        data = lab.to_dict()
        self.assertEqual(json.loads(json.dumps(data)), data)  # 可 JSON 序列化

        restored = InnovationLab.from_dict(data)
        self.assertEqual(len(restored.innovations()), MAX_INNOVATIONS)
        self.assertEqual(
            [item.to_dict() for item in restored.innovations()],
            [item.to_dict() for item in lab.innovations()],
        )
        self.assertEqual(restored.summary_zh(), lab.summary_zh())
        self.assertEqual(restored.history, lab.history)
        self.assertEqual(restored.stats(), lab.stats())
        self.assertEqual(restored.custom_effects(), lab.custom_effects())
        self.assertEqual(restored.effects_for("hafted_blade"), {"harvest_bonus": 1.0})
        self.assertEqual(lab.innovations()[0].successful_uses, 2)
        self.assertEqual(restored.innovations()[0].successful_uses, 2)

    def test_restored_lab_continues_with_unique_ids(self) -> None:
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        store = make_store("stone_axe", "fiber_cord")
        agent = make_agent(
            "stone_axe", "fiber_cord", wood=2, stone=1, fiber=3, capacity_base=20
        )
        ok, _, first = lab.attempt(
            agent, first_proposal_for(lab, agent, store), FakeWorld(FakeTechnology(), store)
        )
        self.assertTrue(ok)
        restored = InnovationLab.from_dict(lab.to_dict())
        restored.rng = FixedRng(0.99)
        # 第二条规则（binder+container）继续发明，id 不与旧记录冲突
        store2 = make_store("basket", "fiber_cord")
        agent2 = make_agent("basket", "fiber_cord", wood=2, cord=1, fiber=3, capacity_base=20)
        proposal = restored.proposals(agent2, store2)[0]
        ok, reason, second = restored.attempt(agent2, proposal, FakeWorld(FakeTechnology(), store2))
        self.assertTrue(ok, reason)
        assert first is not None and second is not None
        self.assertNotEqual(first.innovation_id, second.innovation_id)
        self.assertEqual(
            [item.innovation_id for item in restored.innovations()],
            ["innovation_001", "innovation_002"],
        )

    def test_summary_zh_is_chinese_and_non_empty(self) -> None:
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        empty = lab.summary_zh()
        self.assertTrue(empty)
        self.assertIn("创新实验室", empty)
        self.assertIn("尚无发明", empty)
        self.assertIn(str(MAX_INNOVATIONS), empty)

        store = make_store("stone_axe", "fiber_cord")
        agent = make_agent(
            "stone_axe", "fiber_cord", wood=2, stone=1, fiber=3, capacity_base=20
        )
        lab.attempt(
            agent, first_proposal_for(lab, agent, store), FakeWorld(FakeTechnology(), store)
        )
        text = lab.summary_zh()
        self.assertIn("已发明 1/3", text)
        self.assertIn("装柄刃器", text)
        self.assertIn("采集加成+1", text)
        self.assertIn("记录尝试1次", text)

    def test_stats_counts_real_costs(self) -> None:
        lab, agent, store, technology, world = self._invent_three()
        stats = lab.stats()
        self.assertEqual(stats["innovations"], 3)
        self.assertEqual(stats["max_innovations"], MAX_INNOVATIONS)
        self.assertEqual(stats["attempts"], 3)
        self.assertEqual(stats["successes"], 3)
        self.assertEqual(stats["failures"], 0)
        # 三次成本（最简原型，各 1 单位）: fiber_cord{stone_axe,fiber_cord}->fiber
        # + basket{basket,fiber_cord}->cord + basket{basket,hut}->cord = 1+1+1
        self.assertEqual(stats["material_spent"], 1 + 1 + 1)


# ---------------------------------------------------------------------------
# 5) 真实 World 集成：有 effect 后端就走真实注册/真实数值；没有就走降级路径
# ---------------------------------------------------------------------------
def _make_verified_agent(world, *knowledge_ids: str, wood: int = 2, stone: int = 1,
                         fiber: int = 3, tool: str = "axe"):
    """把真实 World 的个体设为"已掌握且已验证"指定技术并给足材料。"""
    agent = world.agents[0]
    agent.known_knowledge_ids = list(knowledge_ids)
    for knowledge_id in knowledge_ids:
        record = world.knowledge.get(knowledge_id)
        if record is None:
            record = world.knowledge.hypothesize(agent, knowledge_id, 0.0)
        assert record is not None, knowledge_id
        record.status = "verified"
    agent.inventory.tool = tool
    agent.inventory.wood = wood
    agent.inventory.stone = stone
    agent.inventory.fiber = fiber
    return agent


class TestRealWorldIntegration(unittest.TestCase):
    def _make_world(self):
        from newworld.config import make_config
        from newworld.world import World

        return World(make_config(320, 240), seed=3)

    def test_real_world_attempt_registers_real_effect(self) -> None:
        """真实 World + Lead 已实现的 register_custom_effect：真实扣材料 + 真实注册。"""
        try:
            world = self._make_world()
        except Exception as error:  # noqa: BLE001 - 并行开发期间导入失败就 skip
            self.skipTest(f"无法导入真实世界: {error!r}")
        if not hasattr(world.technology, "register_custom_effect"):
            self.skipTest("Lead 还没实现 register_custom_effect：由降级路径用例覆盖")
        agent = _make_verified_agent(world, "fiber_cord", "stone_axe")

        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        proposals = lab.proposals(agent, world.knowledge)
        self.assertTrue(proposals)
        proposal = proposals[0]
        ok, reason, innovation = lab.attempt(agent, proposal, world)
        self.assertTrue(ok, reason)
        assert innovation is not None
        self.assertEqual(innovation.effects, {"harvest_bonus": 1.0})
        # 真实材料被扣除（最简原型成本：第一组件 fiber_cord 的 "fiber" 1 单位）
        self.assertEqual(
            (agent.inventory.wood, agent.inventory.stone, agent.inventory.fiber),
            (2, 1, 2),
        )
        # 效果真的进了 world.technology，可被世界查询
        tool = proposal.output_tool
        self.assertIn(tool, world.technology.custom_effects)
        self.assertEqual(world.technology.custom_effect(tool, "harvest_bonus"), 1.0)
        self.assertEqual(lab.innovations(), [innovation])

    def test_real_world_energy_effect_changes_physiology(self) -> None:
        """heat+binder 工具注册后，真实 physiology 的体力/水分恢复被改变。"""
        try:
            world = self._make_world()
        except Exception as error:  # noqa: BLE001 - 并行开发期间导入失败就 skip
            self.skipTest(f"无法导入真实世界: {error!r}")
        if not hasattr(world.technology, "register_custom_effect"):
            self.skipTest("Lead 还没实现 register_custom_effect")
        agent = _make_verified_agent(world, "campfire", "fiber_cord")
        lab = InnovationLab(seed=0, rng=FixedRng(0.99))
        proposals = lab.proposals(agent, world.knowledge)
        self.assertTrue(proposals)
        proposal = proposals[0]
        self.assertEqual(proposal.effects, {"energy_regen": 0.10, "water_regen": 0.05})
        ok, reason, _ = lab.attempt(agent, proposal, world)
        self.assertTrue(ok, reason)

        agent.inventory.tool = proposal.output_tool
        agent.physiology.energy = 50.0
        agent.physiology.hydration = 50.0
        world.technology.apply_facility_effects(agent, [], 1.0)
        # 真实世界数值：每秒 +0.10 体力 / +0.05 水分（dt=1.0）
        self.assertAlmostEqual(agent.physiology.energy, 50.1, places=6)
        self.assertAlmostEqual(agent.physiology.hydration, 50.05, places=6)

    def test_real_world_without_effect_backend_degrades(self) -> None:
        """P3 早期（没有 effect 后端）时：返回 ``no_effect_backend`` 且不扣材料。"""
        try:
            world = self._make_world()
        except Exception as error:  # noqa: BLE001 - 并行开发期间导入失败就 skip
            self.skipTest(f"无法导入真实世界: {error!r}")
        if hasattr(world.technology, "register_custom_effect"):
            self.skipTest("Lead 已实现 register_custom_effect，P3 降级路径不再适用")
        agent = _make_verified_agent(world, "fiber_cord", "stone_axe")

        lab = InnovationLab(seed=0)
        proposals = lab.proposals(agent, world.knowledge)
        self.assertTrue(proposals)
        ok, reason, innovation = lab.attempt(agent, proposals[0], world)
        self.assertFalse(ok)
        self.assertEqual(reason, NO_EFFECT_BACKEND_REASON)
        self.assertIsNone(innovation)
        # 前置失败不扣真实材料，也不写实验记录
        self.assertEqual(
            (agent.inventory.wood, agent.inventory.stone, agent.inventory.fiber),
            (2, 1, 3),
        )
        self.assertEqual(lab.history, [])


if __name__ == "__main__":
    unittest.main()
