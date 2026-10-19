"""验收测试：知识传承 / 记忆传承 / 社会学习 / 死亡后知识留存。

对应任务书 §8.3 的知识传承测试与 P3 验收：
- 知识能按规则跨个体保存（教学）与跨代保存（出生继承）；
- 个体死亡后其掌握的知识仍可从历史档案中查到（不随个体删除）；
- 程序性记忆可以传给下一代。
"""

from __future__ import annotations

import unittest

from newworld.config import CivilizationConfig, make_config
from newworld.population import Population
from newworld.world import World


def make_population(**civ) -> tuple[World, Population]:
    config = make_config(480, 300)
    civ_config = CivilizationConfig(seed=0, enable_perception=False, **civ)
    world = World(config, seed=0, civ_config=civ_config)
    world.reset(seed=0)
    return world, Population(world, seed=0)


class TestKnowledgeInheritance(unittest.TestCase):
    def test_newborn_inherits_verified_knowledge(self) -> None:
        world, population = make_population()
        mother = next(agent for agent in world.agents if agent.sex == "female")
        father = next(agent for agent in world.agents if agent.sex == "male")
        # validate a technique so it counts as knowledge
        mother.inventory.wood = 2
        mother.inventory.stone = 1
        ok, reason = world.knowledge.attempt(mother, world.knowledge.get("stone_axe"), world, world.t)
        self.assertTrue(ok, reason)
        self.assertIn("stone_axe", mother.known_knowledge_ids)

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
        population.note_birth(child, mother=mother, father_id=father.agent_id)
        self.assertIn("stone_axe", child.known_knowledge_ids)

    def test_inheritance_can_be_switched_off(self) -> None:
        world, population = make_population(enable_inheritance=False)
        mother = next(agent for agent in world.agents if agent.sex == "female")
        father = next(agent for agent in world.agents if agent.sex == "male")
        mother.inventory.wood = 2
        mother.inventory.stone = 1
        world.knowledge.attempt(mother, world.knowledge.get("stone_axe"), world, world.t)
        child = population.create_agent(
            sex="male",
            age=0.0,
            parents=(mother.agent_id, father.agent_id),
            parent_id=mother.agent_id,
            generation=1,
            cause="birth",
        )
        assert child is not None
        population.note_birth(child, mother=mother, father_id=father.agent_id)
        self.assertEqual(child.known_knowledge_ids, [])

    def test_dead_agent_keeps_its_knowledge_record(self) -> None:
        world, population = make_population()
        agent = world.agents[0]
        agent.known_knowledge_ids = ["stone_axe", "fiber_cord"]
        agent.skills = {"gather": 0.7}
        agent.age = 10_000.0
        population.step(0.05)
        record = world.registry.record(agent.agent_id)
        assert record is not None
        self.assertFalse(record.alive)
        self.assertEqual(set(record.known_knowledge_ids), {"stone_axe", "fiber_cord"})
        self.assertEqual(record.skills.get("gather"), 0.7)
        # retention is still computable from the archive after death
        known = set()
        for item in world.registry.records():
            known.update(item.known_knowledge_ids)
        self.assertEqual(known, {"stone_axe", "fiber_cord"})


class TestProceduralMemory(unittest.TestCase):
    def test_procedure_transferred_to_child(self) -> None:
        world, population = make_population()
        mother = next(agent for agent in world.agents if agent.sex == "female")
        father = next(agent for agent in world.agents if agent.sex == "male")
        world.memory.record(mother.agent_id, 1.0, "procedural", "gather:wood", "ok:wood+1", True)
        world.memory.learn_procedure(mother.agent_id, "gather_wood", ("move_to", "gather"))
        child = population.create_agent(
            sex="female",
            age=0.0,
            parents=(mother.agent_id, father.agent_id),
            parent_id=mother.agent_id,
            generation=1,
            cause="birth",
        )
        assert child is not None
        population.note_birth(child, mother=mother, father_id=father.agent_id)
        skills = [procedure.skill for procedure in world.memory.procedures(child.agent_id)]
        self.assertIn("gather_wood", skills)

    def test_death_archives_memory_without_deleting_entries(self) -> None:
        world, population = make_population()
        agent = world.agents[0]
        world.memory.record(agent.agent_id, 1.0, "episode", "state:gather", "ok", True)
        before_ids = {entry.memory_id for entry in world.memory.entries(agent.agent_id)}
        self.assertGreaterEqual(len(before_ids), 1)
        agent.age = 10_000.0
        population.step(0.05)
        after_ids = {entry.memory_id for entry in world.memory.entries(agent.agent_id)}
        self.assertTrue(before_ids <= after_ids, "死亡不能删除已有的情景记忆")


class TestSocialLearning(unittest.TestCase):
    def test_neighbour_learns_verified_knowledge(self) -> None:
        world, population = make_population()
        teacher, student = world.agents[0], world.agents[1]
        teacher.known_knowledge_ids = ["stone_axe"]
        student.known_knowledge_ids = []
        teacher.x = student.x = 200.0
        # the population only teaches verified knowledge
        world.knowledge.get("stone_axe").status = "verified"
        for _ in range(200):
            population._social_learning(0.05)
        self.assertIn("stone_axe", student.known_knowledge_ids)

    def test_no_teaching_beyond_range(self) -> None:
        world, population = make_population()
        teacher, student = world.agents[0], world.agents[1]
        teacher.known_knowledge_ids = ["stone_axe"]
        student.known_knowledge_ids = []
        teacher.x = 100.0
        student.x = 700.0
        world.knowledge.get("stone_axe").status = "verified"
        for _ in range(100):
            population._social_learning(0.05)
        self.assertEqual(student.known_knowledge_ids, [])


if __name__ == "__main__":
    unittest.main()
