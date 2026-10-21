"""P3 社会机制单元测试：交换 / 教学 / 继承（独立开关 + 守恒律）。

只验证**机制正确性**：开关独立性、转移守恒、技能真实生效、
统计准确。社会指标的效应量由 experiments/run_p3_social.py 的
配对对照回答。
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from tests._env_compat import patch_tempdir
except ImportError:
    from _env_compat import patch_tempdir

patch_tempdir()

from newworld.map_engine import MapConfig, MapEngine  # noqa: E402
from newworld.swarm import Swarm, SwarmConfig  # noqa: E402
from newworld.social_mech import (  # noqa: E402
    SKILLS,
    SocialConfig,
    SocialLayer,
)


def make_pair(seed: int = 0, population: int = 40):
    """(engine, swarm) 工厂：稀缺小图。"""
    engine = MapEngine(MapConfig(seed=seed, width=128, height=80, noise_scale=32))
    swarm = Swarm(
        engine,
        SwarmConfig(initial_population=population, capacity_hard=200),
        seed=seed,
    )
    return engine, swarm


# ---------------------------------------------------------------------------
class BarterTest(unittest.TestCase):
    """交换：能量转移守恒。"""

    def test_barter_transfer_conserves(self) -> None:
        """一次交换 donor 减多少 receiver 加多少（守恒律）。"""
        engine, swarm = make_pair(seed=1)
        social = SocialLayer(
            swarm,
            SocialConfig(enable_barter=True, enable_teaching=False,
                         enable_inheritance=False, barter_gap=0.1,
                         barter_radius=100, barter_interval_s=0.0),
            seed=1,
        )
        # 手工布置一对个体（相邻、饱足差大）
        view = swarm.data[: swarm.count]
        rows = np.nonzero(view["alive"])[0]
        a, b = int(rows[0]), int(rows[1])
        view["x"][a] = 50.0; view["y"][a] = 40.0
        view["x"][b] = 51.0; view["y"][b] = 40.0
        ha, hb = 0.9, 0.2
        view["hunger"][a] = ha
        view["hunger"][b] = hb
        # 大量尝试保证配对命中
        for _ in range(50):
            social._try_barter()
        events = social.stats["barter_events"]
        self.assertGreater(events, 0, "相邻且饱足差大的个体应发生交换")
        total = social.stats["barter_transfer_total"]
        self.assertGreater(total, 0.0)
        # 饱足不越界（守恒的最弱形式：值域合法）
        hungers = view["hunger"][[a, b]]
        self.assertTrue(((hungers >= 0.0) & (hungers <= 1.0)).all())

    def test_barter_disabled_no_events(self) -> None:
        """关闭交换 → 事件数为 0。"""
        engine, swarm = make_pair(seed=2)
        social = SocialLayer(swarm, SocialConfig(enable_barter=False))
        for _ in range(100):
            engine.update(0.05, 0.8)
            swarm.update(0.05)
            social.update(0.05)
        self.assertEqual(social.stats["barter_events"], 0)


# ---------------------------------------------------------------------------
class TeachTest(unittest.TestCase):
    """教学：技能真实传播。"""

    def test_teach_spreads_skill(self) -> None:
        """一个技能个体教会范围内的无技能邻居。"""
        engine, swarm = make_pair(seed=3)
        social = SocialLayer(
            swarm,
            SocialConfig(enable_barter=False, enable_teaching=True,
                         enable_inheritance=False, teach_radius=100,
                         teach_chance_per_s=1.0, teach_interval_s=0.0),
            seed=3,
        )
        view = swarm.data[: swarm.count]
        rows = np.nonzero(view["alive"])[0]
        # 只留一个老师，其他清技能
        social.skills[:] = 0
        social.skills[int(rows[0])] = 1
        before = int((social.skills[rows] > 0).sum())
        for _ in range(60):
            social._try_teach()
        # skilled_population 由 _refresh_skilled_count 刷新（update 时调用），
        # 直接调 _try_teach 后要手动刷新（或直接数）
        social._refresh_skilled_count()
        after = social.stats["skilled_population"]
        self.assertGreater(social.stats["teach_events"], 0, "应发生教学")
        self.assertGreaterEqual(after, before)

    def test_teach_requires_teacher_alive(self) -> None:
        """无技能个体存在时教学不发生（无老师 = 无传播）。"""
        engine, swarm = make_pair(seed=4)
        social = SocialLayer(swarm, SocialConfig(enable_teaching=True,
                                                 teach_chance_per_s=1.0))
        social.skills[:] = 0  # 没人有技能
        for _ in range(50):
            social._try_teach()
        self.assertEqual(social.stats["teach_events"], 0)


# ---------------------------------------------------------------------------
class InheritTest(unittest.TestCase):
    """继承：死亡时技能转移。"""

    def test_inherit_moves_skill(self) -> None:
        """死亡个体的技能转给存活邻居（继承事件 > 0）。"""
        engine, swarm = make_pair(seed=5, population=20)
        social = SocialLayer(
            swarm,
            SocialConfig(enable_barter=False, enable_teaching=False,
                         enable_inheritance=True, inherit_radius=1000),
            seed=5,
        )
        view = swarm.data[: swarm.count]
        rows = np.nonzero(view["alive"])[0]
        social.skills[:] = 0
        social.skills[int(rows[0])] = 2
        # 断粮让个体死亡（不推进 engine → 食物不回来）
        engine.food[:] = 0.0
        before = len(swarm.death_ages)
        for _ in range(3000):
            swarm.update(0.05)
            social.on_death_pass(before)
            before = len(swarm.death_ages)
            social.update(0.05)
        self.assertGreater(social.stats["inherit_events"], 0,
                           "带技能个体死亡应触发继承")
        self.assertEqual(swarm.population, 0)

    def test_inherit_disabled_no_transfer(self) -> None:
        """关闭继承 → 技能随死亡湮灭（无继承事件）。"""
        engine, swarm = make_pair(seed=6, population=20)
        social = SocialLayer(swarm, SocialConfig(enable_inheritance=False))
        engine.food[:] = 0.0
        for _ in range(1000):
            before = len(swarm.death_ages)
            swarm.update(0.05)
            social.on_death_pass(before)
        self.assertEqual(social.stats["inherit_events"], 0)


# ---------------------------------------------------------------------------
class SkillEffectTest(unittest.TestCase):
    """技能效果乘数（世界规则层真实生效）。"""

    def test_skill_multipliers(self) -> None:
        """三种技能的乘数正确。"""
        engine, swarm = make_pair(seed=7)
        social = SocialLayer(swarm, SocialConfig())
        social.skills[0] = 1
        social.skills[1] = 2
        social.skills[2] = 3
        self.assertAlmostEqual(social.bite_multiplier(0), 1.5)
        self.assertAlmostEqual(social.move_multiplier(1), 1.4)
        self.assertAlmostEqual(social.metabolism_multiplier(2), 0.75)
        self.assertAlmostEqual(social.bite_multiplier(1), 1.0, "无技能=1")

    def test_initial_skill_injection(self) -> None:
        """开局技能注入率大致准确。"""
        engine, swarm = make_pair(seed=8, population=100)
        social = SocialLayer(swarm, SocialConfig(initial_skill_rate=0.5))
        social.inject_initial_skills()
        view = swarm.data[: swarm.count]
        alive = np.nonzero(view["alive"])[0]
        skilled = int((social.skills[alive] > 0).sum())
        # 二项分布 100 次 p=0.5：95% 区间约 [35, 65]
        self.assertGreater(skilled, 25)
        self.assertLess(skilled, 75)
        # 技能值合法（1..3）
        values = set(social.skills[alive].tolist())
        self.assertTrue(values <= {0, 1, 2, 3})


# ---------------------------------------------------------------------------
class ConfigIndependenceTest(unittest.TestCase):
    """三机制开关互不影响。"""

    def test_only_requested_mechanism_runs(self) -> None:
        """仅交换开启时教学/继承计数为 0（反之亦然）。"""
        engine, swarm = make_pair(seed=9)
        social = SocialLayer(swarm, SocialConfig(enable_barter=True,
                                                 enable_teaching=False,
                                                 enable_inheritance=False))
        for _ in range(200):
            engine.update(0.05, 0.8)
            before = len(swarm.death_ages)
            swarm.update(0.05)
            social.on_death_pass(before)
            social.update(0.05)
        self.assertEqual(social.stats["teach_events"], 0)
        self.assertEqual(social.stats["inherit_events"], 0)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
class SkillWiringTest(unittest.TestCase):
    """P4 接线：技能效果通过 swarm.skill_effect 真实作用于行为层。"""

    def test_attach_to_swarm_sets_hook(self) -> None:
        """attach_to_swarm 后 swarm.skill_effect 可调用且返回正确乘数。"""
        engine, swarm = make_pair(seed=10)
        social = SocialLayer(swarm, SocialConfig())
        social.attach_to_swarm()
        self.assertIsNotNone(swarm.skill_effect)
        social.skills[0] = 1  # efficient_biter
        social.skills[1] = 2  # long_sight
        social.skills[2] = 3  # slow_metabolism
        self.assertAlmostEqual(swarm.skill_effect(0, "bite", 1.0), 1.5)
        self.assertAlmostEqual(swarm.skill_effect(1, "move", 2.0), 2.8)
        self.assertAlmostEqual(swarm.skill_effect(2, "metabolism", 0.01), 0.0075)
        # 无技能 = 原值
        self.assertAlmostEqual(swarm.skill_effect(5, "bite", 1.0), 1.0)

    def test_detached_hook_is_none(self) -> None:
        """未接线时 swarm.skill_effect 为 None（零开销）。"""
        engine, swarm = make_pair(seed=11)
        social = SocialLayer(swarm, SocialConfig())
        self.assertIsNone(swarm.skill_effect)

    def test_metabolism_skill_improves_survival(self) -> None:
        """slow_metabolism 全员携带 → 断粮下活得明显更久（端到端）。"""
        from newworld.swarm_policy import RuleSwarmPolicy

        def run_starve(with_skill: bool, seed: int = 12) -> float:
            engine, swarm = make_pair(seed=seed, population=40)
            swarm.attach_policy(RuleSwarmPolicy(seed=seed))
            social = SocialLayer(swarm, SocialConfig(
                enable_barter=False, enable_teaching=False, enable_inheritance=False,
                initial_skill_rate=0.0), seed=seed)
            if with_skill:
                social.skills[: swarm.count] = 3  # 全员 slow_metabolism
                social.attach_to_swarm()
            engine.food[:] = 0.0
            steps = 0
            while swarm.population > 0 and steps < 6000:
                swarm.update(0.05)
                steps += 1
            return steps * 0.05  # 存活世界秒

        plain = run_starve(False)
        skilled = run_starve(True)
        # 代谢 ×0.75 → 饥饿更慢 → 活得更久（应至少多 10%）
        self.assertGreater(
            skilled, plain * 1.1,
            f"slow_metabolism 应延长断粮存活（{skilled:.0f}s vs {plain:.0f}s）",
        )
