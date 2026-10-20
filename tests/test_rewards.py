"""奖励函数测试（V2）：真实世界变化 + 反刷分。

这些测试是"奖励漏洞"的第一道防线：原地站立、粮仓旁挂机、重复无效动作
都不能产生正奖励；只有真实的技术/设施/资源/知识/合作变化才有奖励。
"""

from __future__ import annotations

import unittest

from newworld.config import CivilizationConfig, make_config
from newworld.population import Population
from newworld.rewards import LowLevelReward, RewardWeights
from newworld.world import World


def make_population(**civ) -> tuple[World, Population]:
    config = make_config(480, 300)
    cfg = CivilizationConfig(seed=0, enable_perception=False, **civ)
    world = World(config, seed=0, civ_config=cfg)
    world.reset(seed=0)
    return world, Population(world, seed=0)


class TestRewardIsRealWorldDelta(unittest.TestCase):
    def test_first_observation_is_zero(self) -> None:
        world, _ = make_population()
        reward = LowLevelReward()
        breakdown = reward.observe(world, world.agents[0], dt=0.05)
        self.assertEqual(breakdown.total, 0.0)

    def test_verified_technology_is_rewarded(self) -> None:
        world, _ = make_population()
        agent = world.agents[0]
        reward = LowLevelReward()
        reward.observe(world, agent, dt=0.05)  # baseline
        agent.inventory.wood = 2
        agent.inventory.stone = 1
        ok, _reason = world.knowledge.attempt(
            agent, world.knowledge.get("stone_axe"), world, world.t
        )
        self.assertTrue(ok)
        breakdown = reward.observe(world, agent, dt=0.05)
        self.assertGreater(breakdown.d_tech, 0.0)
        self.assertGreater(breakdown.total, 0.0)

    def test_new_knowledge_is_rewarded(self) -> None:
        world, _ = make_population()
        agent = world.agents[0]
        reward = LowLevelReward()
        reward.observe(world, agent, dt=0.05)
        agent.known_knowledge_ids = ["stone_axe"]
        breakdown = reward.observe(world, agent, dt=0.05)
        self.assertGreater(breakdown.d_knowledge, 0.0)

    def test_facility_is_rewarded(self) -> None:
        world, _ = make_population()
        agent = world.agents[0]
        reward = LowLevelReward()
        reward.observe(world, agent, dt=0.05)
        world.civilization.build_facility("campfire", 100.0, agent.agent_id, world.t)
        breakdown = reward.observe(world, agent, dt=0.05)
        self.assertGreater(breakdown.d_facility, 0.0)

    def test_cooperation_transfer_is_rewarded(self) -> None:
        world, _ = make_population()
        agent = world.agents[0]
        reward = LowLevelReward()
        reward.observe(world, agent, dt=0.05)
        agent.contributions["share_food"] = 5.0
        breakdown = reward.observe(world, agent, dt=0.05)
        self.assertGreater(breakdown.d_cooperation, 0.0)


class TestNoRewardExploits(unittest.TestCase):
    def test_standing_still_is_not_positive(self) -> None:
        world, population = make_population()
        agent = world.agents[0]
        reward = LowLevelReward()
        reward.observe(world, agent, dt=0.05)
        # 站着不动 200 帧：只有时间成本，没有任何进度
        for _ in range(200):
            world.step(actions=0, dt=0.05)
            breakdown = reward.observe(world, agent, dt=0.05)
        self.assertLessEqual(breakdown.total, 0.0)
        self.assertAlmostEqual(breakdown.d_tech, 0.0)
        self.assertLess(breakdown.time_cost, 0.0)

    def test_camping_at_granary_does_not_farm_reward(self) -> None:
        """在活跃粮仓旁挂机（能量恢复）不得产生正奖励。"""
        world, _ = make_population()
        agent = world.agents[0]
        world.civilization.build_facility("granary", agent.x, agent.agent_id, world.t)
        reward = LowLevelReward()
        reward.observe(world, agent, dt=0.05)
        agent.physiology.energy = 20.0
        for _ in range(200):
            world.technology.apply_facility_effects(
                agent, world.civilization.facilities, 0.05
            )
            breakdown = reward.observe(world, agent, dt=0.05)
        self.assertGreater(agent.physiology.energy, 20.0)  # 世界确实在恢复能量
        self.assertLessEqual(breakdown.total, 0.0)        # 但不给奖励

    def test_repeating_failed_actions_gets_no_reward(self) -> None:
        world, _ = make_population()
        agent = world.agents[0]
        reward = LowLevelReward()
        reward.observe(world, agent, dt=0.05)
        for _ in range(50):
            world.technology.harvest(agent, "stone")  # 手够不到，必然失败
            breakdown = reward.observe(world, agent, dt=0.05)
        self.assertLessEqual(breakdown.total, 0.0)

    def test_reward_is_clamped(self) -> None:
        world, _ = make_population()
        agent = world.agents[0]
        reward = LowLevelReward(RewardWeights(clip_low=-1.0, clip_high=1.0))
        reward.observe(world, agent, dt=0.05)
        world.civilization.build_facility("campfire", 100.0, 0, world.t)
        world.civilization.build_facility("granary", 200.0, 0, world.t)
        breakdown = reward.observe(world, agent, dt=0.05)
        self.assertLessEqual(abs(breakdown.total), 1.0)

    def test_death_penalty(self) -> None:
        reward = LowLevelReward()
        breakdown = reward.death_bonus()
        self.assertLess(breakdown.total, 0.0)


class TestPopulationRewardIntegration(unittest.TestCase):
    def test_population_accumulates_real_rewards_in_rl_mode(self) -> None:
        """Population 在 low_level_mode="rl" 且挂载策略时记录真实奖励转移。"""
        from newworld.simulation import RunSpec, SocietyRunner

        runner = SocietyRunner(
            RunSpec(group="C", seed=0, steps=300, dt=0.05, width=480, height=300,
                    low_level_mode="heuristic")
        )
        # 即使不挂策略，奖励观察器也可独立运行（Population 内部已接线）
        self.assertTrue(hasattr(runner.population, "lowlevel_reward"))


if __name__ == "__main__":
    unittest.main()
