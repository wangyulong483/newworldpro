"""社会网络与人口学测试：亲密度、出生/死亡、8 人预算。

Tests for society (network, reproduction, death) and technology/community."""

from __future__ import annotations

import unittest

from newworld import make_config, World
from newworld.population import Population
from newworld.society import MAX_POPULATION, Society, SocietyNetwork
from newworld.technology import CommunitySite, Inventory, Technology


def small_world() -> World:
    world = World(make_config(800, 500), seed=0)
    world.reset(seed=0)
    return world


class TestSocietyNetwork(unittest.TestCase):
    def test_graph_network_output(self) -> None:
        import numpy as np

        network = SocietyNetwork(seed=0)
        features = np.random.default_rng(0).normal(size=(3, 11))
        affinity = np.full((3, 3), 0.4)
        context = network.forward(features, affinity)
        self.assertEqual(context.shape, (3, 6))
        self.assertTrue(np.all(np.isfinite(context)))

    def test_context_is_written_to_agents(self) -> None:
        world = small_world()
        population = Population(world, seed=0)
        population.society.update(world, 1 / 60)
        self.assertTrue(all(len(agent.social_context) == 6 for agent in world.agents))


class TestDemography(unittest.TestCase):
    def test_pregnancy_birth_and_death(self) -> None:
        world = small_world()
        world.agents.clear()
        male = world.spawn(sex="male", age=150.0, x=400.0)
        female = world.spawn(sex="female", age=130.0, x=430.0)
        society = Society(seed=0)
        society.relate(male.agent_id, female.agent_id, 0.9)
        society.relate(female.agent_id, male.agent_id, 0.9)
        for _ in range(5):
            society.update(world, 0.1)
        self.assertTrue(female.pregnant)
        female.pregnancy_timer = 0.2
        society.update(world, 0.5)
        self.assertGreater(len(world.agents), 2)
        self.assertGreater(society.births, 0)
        self.assertEqual(world.agents[-1].generation, 1)
        self.assertLess(world.agents[-1].body_scale, 1.0)
        # death by old age
        world.agents[0].age = 10_000.0
        society.update(world, 0.1)
        self.assertFalse(male.alive)
        self.assertGreater(society.deaths, 0)

    def test_population_caps_reproduction(self) -> None:
        world = small_world()
        world.agents.clear()
        for index in range(MAX_POPULATION):
            world.spawn(sex="female" if index % 2 else "male", age=200.0, x=300.0 + index * 30)
        society = Society(seed=0)
        society.update(world, 1 / 60)
        self.assertLessEqual(len(world.agents), MAX_POPULATION)


class TestTechnology(unittest.TestCase):
    def test_inventory_and_craft(self) -> None:
        world = small_world()
        agent = world.agent
        agent.inventory = Inventory(wood=2, stone=1)
        tech = Technology(world.config, world.pixel_scene)
        self.assertTrue(tech.craft(agent))
        self.assertEqual(agent.inventory.tool, "axe")
        self.assertEqual(agent.inventory.wood, 0)

    def test_contribute_builds_community(self) -> None:
        world = small_world()
        agent = world.agent
        tech = Technology(world.config, world.pixel_scene)
        agent.x = tech.community.x_screen
        agent.inventory = Inventory(wood=4, stone=2)
        self.assertTrue(tech.contribute(agent))
        self.assertTrue(tech.contribute(agent))
        self.assertTrue(tech.community.built)
        self.assertTrue(tech.community.campfire)

    def test_summary_is_chinese(self) -> None:
        world = small_world()
        tech = world.technology
        self.assertIn("社区", tech.summary_zh())


class TestPopulationStep(unittest.TestCase):
    def test_population_runs(self) -> None:
        world = small_world()
        population = Population(world, seed=0)
        self.assertEqual(len(world.agents), 2)
        for _ in range(30):
            population.step(1 / 60)
        self.assertTrue(all(agent.alive for agent in world.agents))
        self.assertEqual(len(population.minds), len(world.agents))


if __name__ == "__main__":
    unittest.main()
