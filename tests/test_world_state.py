"""验收测试：世界状态保存与加载（含历史注册表/知识/文明状态）。

对应任务书 §8.3 的"世界状态保存与加载测试"。
要求：保存后重新加载，
- 历史累计创建数、死亡档案、知识状态、公共设施、灭绝标记都保留；
- 死亡个体不会被复活成活人；
- 活着的个体被恢复。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from newworld.config import CivilizationConfig, make_config
from newworld.population import Population
from newworld.world import World

try:  # "python -m unittest tests.test_world_state"
    from tests._env_compat import patch_tempdir
except ImportError:  # "python -m unittest discover -s tests"
    from _env_compat import patch_tempdir
patch_tempdir()


def make_population() -> tuple[World, Population]:
    civ = CivilizationConfig(seed=0, enable_perception=False)
    world = World(make_config(480, 300), seed=0, civ_config=civ)
    world.reset(seed=0)
    return world, Population(world, seed=0)


class TestWorldStateRoundtrip(unittest.TestCase):
    def test_save_load_preserves_history_and_knowledge(self) -> None:
        world, population = make_population()
        # build up some history: validated knowledge + a death + a facility
        worker = world.agents[0]
        worker.inventory.wood = 2
        worker.inventory.stone = 1
        ok, reason = world.knowledge.attempt(
            worker, world.knowledge.get("stone_axe"), world, world.t
        )
        self.assertTrue(ok, reason)
        for kind, x_screen, agent_id in list(world.technology.pending_facilities):
            world.civilization.build_facility(kind, x_screen, agent_id, world.t)
        world.technology.pending_facilities.clear()

        victim = world.agents[1]
        victim.known_knowledge_ids = ["stone_axe"]
        victim.age = 10_000.0
        population.step(0.05)
        self.assertFalse(world.registry.record(victim.agent_id).alive)
        total_created = world.registry.total_created

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "world.json"
            world.save(path)
            self.assertTrue(path.exists())
            loaded = World.load(path)

        self.assertEqual(loaded.registry.total_created, total_created)
        self.assertEqual(loaded.registry.total_deaths, world.registry.total_deaths)
        self.assertEqual(loaded.knowledge.verified_ids(), world.knowledge.verified_ids())
        self.assertEqual(len(loaded.civilization.facilities), len(world.civilization.facilities))
        dead_record = loaded.registry.record(victim.agent_id)
        self.assertIsNotNone(dead_record)
        assert dead_record is not None
        self.assertFalse(dead_record.alive)
        self.assertEqual(dead_record.known_knowledge_ids, ("stone_axe",))
        # the dead are not resurrected as living agents
        self.assertNotIn(victim.agent_id, [agent.agent_id for agent in loaded.agents])
        self.assertIn(worker.agent_id, [agent.agent_id for agent in loaded.agents])
        loaded_worker = loaded.agent_by_id(worker.agent_id)
        assert loaded_worker is not None
        self.assertIn("stone_axe", loaded_worker.known_knowledge_ids)

    def test_world_dict_is_json_serialisable(self) -> None:
        import json

        world, population = make_population()
        population.step(0.05)
        payload = json.dumps(world.to_dict(), ensure_ascii=False, default=str)
        self.assertIn("registry", payload)
        self.assertIn("knowledge", payload)
        self.assertIn("civilization", payload)


if __name__ == "__main__":
    unittest.main()
