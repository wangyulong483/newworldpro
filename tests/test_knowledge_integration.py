"""验收测试：知识验证唯一路径 / 技术前置条件 / 失效 / 落地效果。

对应任务书 §8.3：
- 技术前置条件测试；
- 知识验证和失效测试；
- 只有经过模拟器验证的成果才能标为已验证技术；
- 已验证技术必须真实改变世界数值（采集产量、负重、恢复、取水）。

所有测试都通过真实 ``World`` / ``Technology`` 执行，不使用假模拟器，
以便证明"世界规则唯一权威"没有被打穿。
"""

from __future__ import annotations

import unittest

from newworld.config import CivilizationConfig, make_config
from newworld.knowledge import RECIPE_CATALOG, KnowledgeStore
from newworld.world import World


def make_world() -> World:
    civ = CivilizationConfig(seed=0, enable_perception=False)
    world = World(make_config(480, 300), seed=0, civ_config=civ)
    world.reset(seed=0)
    return world


class TestRecipePrerequisites(unittest.TestCase):
    def test_catalog_has_dependency_chain(self) -> None:
        self.assertIn("stone_axe", RECIPE_CATALOG)
        self.assertIn("fiber_cord", RECIPE_CATALOG)
        self.assertIn("basket", RECIPE_CATALOG)
        self.assertIn("granary", RECIPE_CATALOG)
        self.assertIn("fiber_cord", RECIPE_CATALOG["basket"].requires_knowledge)
        self.assertIn("basket", RECIPE_CATALOG["hut"].requires_knowledge)

    def test_unverified_prerequisite_blocks_attempt(self) -> None:
        world = make_world()
        agent = world.agent
        # give the materials for a hut, but the prerequisites are not verified
        agent.inventory.wood = 10
        agent.inventory.stone = 6
        agent.inventory.cord = 2
        store = world.knowledge
        hut = store.get("hut")
        before = agent.inventory.load()
        ok, reason = store.attempt(agent, hut, world, world.t)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_prerequisites")
        self.assertEqual(agent.inventory.load(), before)  # nothing consumed
        self.assertEqual(hut.status, "hypothesis")
        self.assertEqual(hut.failed_uses, 0)

    def test_missing_material_is_not_a_technical_defect(self) -> None:
        world = make_world()
        agent = world.agent
        store = world.knowledge
        axe = store.get("stone_axe")
        for _ in range(5):
            ok, reason = store.attempt(agent, axe, world, world.t)
            self.assertFalse(ok)
            self.assertIn(reason, ("missing_material", "missing_tool"))
        self.assertEqual(axe.failed_uses, 0, "材料不足不能累计为技术缺陷")
        self.assertNotEqual(axe.status, "invalidated")
        self.assertGreater(store.blocked_attempts, 0)


class TestValidationAndEffects(unittest.TestCase):
    def test_validated_axe_changes_harvest_yield(self) -> None:
        world = make_world()
        agent = world.agent
        store = world.knowledge
        axe = store.get("stone_axe")
        agent.inventory.wood = 2
        agent.inventory.stone = 1
        ok, reason = store.attempt(agent, axe, world, world.t)
        self.assertTrue(ok, reason)
        self.assertEqual(axe.status, "verified")
        self.assertIn("stone_axe", agent.known_knowledge_ids)
        self.assertEqual(agent.inventory.tool, "axe")

        # real world effect: the axe doubles harvested material per atom
        target = world.technology.nearest_resource(agent, "stone")
        self.assertIsNotNone(target)
        agent.x = target[0]
        before = agent.inventory.stone
        self.assertTrue(world.technology.harvest(agent, "stone"))
        self.assertEqual(agent.inventory.stone, before + 2)

    def test_granary_effect_feeds_agents(self) -> None:
        world = make_world()
        agent = world.agent
        store = world.knowledge
        # unlock the chain through the simulator
        for recipe_id, inputs in (
            ("stone_axe", {"wood": 2, "stone": 1}),
            ("granary", {"wood": 2, "stone": 2}),
        ):
            agent.inventory.wood += inputs.get("wood", 0) + 2
            agent.inventory.stone += inputs.get("stone", 0) + 2
            ok, reason = store.attempt(agent, store.get(recipe_id), world, world.t)
            self.assertTrue(ok, f"{recipe_id}: {reason}")
        self.assertEqual(store.get("granary").status, "verified")
        # The simulator queues structure construction; Population drains the
        # queue every frame (that is where the civilization layer owns public
        # facilities).  Here we drain it explicitly because this test drives the
        # world directly instead of through a Population.
        for kind, x_screen, agent_id in list(world.technology.pending_facilities):
            world.civilization.build_facility(kind, x_screen, agent_id, world.t)
        world.technology.pending_facilities.clear()
        self.assertTrue(world.civilization.facilities, "验证成功应真的建造设施")
        granary = world.civilization.facilities[-1]
        self.assertEqual(granary.kind, "granary")
        agent.x = granary.x_screen
        agent.physiology.energy = 40.0
        world.technology.apply_facility_effects(agent, world.civilization.facilities, 1.0)
        self.assertGreater(agent.physiology.energy, 40.0)

    def test_repeated_defects_invalidate_knowledge(self) -> None:
        world = make_world()
        agent = world.agent
        store = world.knowledge
        axe = store.get("stone_axe")
        # force failures by giving no materials repeatedly: those are blocked
        # (not defects), so instead validate the state machine directly through
        # the store's public record API.
        for _ in range(3):
            axe.record(world.t, agent.agent_id, False, "defect")
        self.assertEqual(axe.status, "invalidated")
        self.assertFalse(axe.usable())
        ok, reason = store.attempt(agent, axe, world, world.t)
        self.assertFalse(ok)
        self.assertEqual(reason, "invalidated")

    def test_invalidated_knowledge_cannot_be_crafted(self) -> None:
        world = make_world()
        agent = world.agent
        store = world.knowledge
        axe = store.get("stone_axe")
        store.mark_invalid("stone_axe", "environment_changed", world.t)
        self.assertFalse(axe.usable())
        # the world simulator itself does not consult knowledge status (the
        # planner does); here we assert the knowledge is not usable, so a
        # planner must choose a different plan.
        self.assertEqual(store.verified_ids(), [])


class TestKnowledgeSerialization(unittest.TestCase):
    def test_store_survives_json_roundtrip(self) -> None:
        world = make_world()
        agent = world.agent
        store = world.knowledge
        agent.inventory.wood = 2
        agent.inventory.stone = 1
        store.attempt(agent, store.get("stone_axe"), world, world.t)
        data = store.to_dict()
        restored = KnowledgeStore.from_dict(data)
        self.assertEqual(restored.get("stone_axe").status, "verified")
        self.assertEqual(restored.verified_ids(), store.verified_ids())
        self.assertEqual(restored.to_dict(), data)


if __name__ == "__main__":
    unittest.main()
