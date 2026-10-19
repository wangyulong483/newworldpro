"""验收测试：8 个历史创建名额的硬约束 / 死亡不可行动 / 死亡档案 / 灭绝。

对应任务书 §2.1（Agent 数量限制）与 §8.3（测试要求）：
- total_agents_created <= 8，且即使存活为 0 也不能突破；
- 死亡后不能再次行动，历史记录不删除；
- 死亡个体保留家族关系与继承内容（inventory / knowledge / skills）；
- 8 个名额用完后世界不能无限生成新个体。
"""

from __future__ import annotations

import unittest

from newworld.config import CivilizationConfig, make_config
from newworld.population import Population
from newworld.world import NoAgentsError, World


def small_world(**civ) -> World:
    config = make_config(480, 300)
    civ_config = CivilizationConfig(seed=0, enable_perception=False, **civ)
    world = World(config, seed=0, civ_config=civ_config)
    world.reset(seed=0)
    return world


class TestCreationBudget(unittest.TestCase):
    def test_initial_population_is_registered(self) -> None:
        world = small_world()
        population = Population(world, seed=0)
        self.assertEqual(world.registry.total_created, 2)
        self.assertEqual(world.registry.alive_count(), 2)
        self.assertEqual(len(population.minds), 2)

    def test_total_created_never_exceeds_eight(self) -> None:
        world = small_world()
        population = Population(world, seed=0)
        created = 0
        for index in range(20):
            child = population.create_agent(
                sex="female" if index % 2 else "male",
                age=10.0,
                name=f"spawn-{index}",
                cause="test",
            )
            if child is not None:
                created += 1
        self.assertEqual(world.registry.total_created, 8)
        self.assertEqual(created, 6)  # 8 budget - 2 initial
        self.assertFalse(world.registry.can_create())
        self.assertGreaterEqual(len(world.creation_denied), 1)
        self.assertEqual(world.creation_denied[-1]["total_created"], 8)

    def test_budget_does_not_reset_when_agents_die(self) -> None:
        """死亡释放的是"存活名额"，不是"历史创建名额"。"""
        world = small_world()
        population = Population(world, seed=0)
        while world.registry.can_create():
            population.create_agent(sex="male", age=10.0, cause="test")
        self.assertEqual(world.registry.total_created, 8)

        # kill everyone (old age) -> alive 0 but budget still exhausted
        for agent in list(world.agents):
            agent.age = 10_000.0
        population.society.update(world, 0.1)
        self.assertEqual(world.registry.alive_count(), 0)
        self.assertFalse(world.registry.can_create())
        denied_before = len(world.creation_denied)
        self.assertIsNone(population.create_agent(sex="female", age=10.0, cause="test"))
        self.assertEqual(world.registry.total_created, 8)
        self.assertEqual(len(world.creation_denied), denied_before + 1)

    def test_extinct_flag_and_time(self) -> None:
        world = small_world()
        population = Population(world, seed=0)
        for agent in list(world.agents):
            agent.age = 10_000.0
        population.step(0.05)
        self.assertTrue(world.civilization.extinct)
        self.assertIsNotNone(world.civilization.extinct_at)
        self.assertEqual(world.civilization.collapse_state, "collapsed")
        # nobody can be created because the budget is only 2 used... but the
        # civilization is not allowed to resurrect members spontaneously:
        with self.assertRaises(NoAgentsError):
            _ = world.agent


class TestDeathArchive(unittest.TestCase):
    def test_dead_agent_is_archived_not_deleted(self) -> None:
        world = small_world()
        population = Population(world, seed=0)
        victim = world.agents[0]
        victim.known_knowledge_ids = ["stone_axe"]
        victim.inventory.wood = 3
        victim.skills = {"gather": 0.5}
        victim.age = 10_000.0
        population.step(0.05)

        record = world.registry.record(victim.agent_id)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertFalse(record.alive)
        self.assertIsNotNone(record.died_at)
        self.assertEqual(record.death_cause, "old_age")
        self.assertEqual(record.known_knowledge_ids, ("stone_axe",))
        self.assertEqual(record.inventory.get("wood"), 3)
        self.assertEqual(record.skills.get("gather"), 0.5)
        self.assertIn(victim.agent_id, [item.agent_id for item in world.registry.records()])
        self.assertNotIn(victim.agent_id, world.registry.alive_ids())
        # the live world no longer contains the dead body
        self.assertNotIn(victim.agent_id, [agent.agent_id for agent in world.agents])

    def test_dead_agent_cannot_act_again(self) -> None:
        world = small_world()
        population = Population(world, seed=0)
        victim = world.agents[0]
        population.step(0.05)
        victim.age = 10_000.0
        population.step(0.05)
        self.assertNotIn(victim.agent_id, population.minds)
        steps_at_death = victim.steps
        x_at_death = victim.x
        for _ in range(30):
            population.step(0.05)
        self.assertEqual(victim.steps, steps_at_death)
        self.assertEqual(victim.x, x_at_death)
        self.assertFalse(victim.alive)

    def test_lineage_is_recorded(self) -> None:
        world = small_world()
        population = Population(world, seed=0)
        mother = next(agent for agent in world.agents if agent.sex == "female")
        father = next(agent for agent in world.agents if agent.sex == "male")
        child = population.create_agent(
            sex="female",
            age=0.0,
            parents=(mother.agent_id, father.agent_id),
            parent_id=mother.agent_id,
            generation=1,
            cause="birth",
        )
        self.assertIsNotNone(child)
        assert child is not None
        record = world.registry.record(child.agent_id)
        assert record is not None
        self.assertEqual(record.parent_id, mother.agent_id)
        self.assertIn((mother.agent_id, child.agent_id), world.registry.lineage())


class TestNoImplicitResurrection(unittest.TestCase):
    def test_empty_world_does_not_spawn_on_attribute_access(self) -> None:
        world = small_world()
        world.agents.clear()
        with self.assertRaises(NoAgentsError):
            _ = world.agent
        self.assertEqual(len(world.agents), 0)


if __name__ == "__main__":
    unittest.main()
