"""telemetry.py 的单元/集成测试（运行方式见文件末）。

覆盖：
* 契约 §8 指标键与类型（真实小世界 ``World(make_config(320, 240))`` 跑 60 步）；
* 两个"停滞"口径（``no_progress_repeat_ratio`` / ``state_identical_ratio``）；
* ``repeat_run_max`` / ``dominant_state_ratio`` / ``tech_actions_per_1000_steps``；* 生存率（个体-秒）、灭绝检测、扰动恢复；
* 可选人口属性（goals/knowledge/civilization/memory）的映射与**缺失降级**；
* 事件 JSONL、``to_json``、``close``；
* telemetry **绝不修改世界**（前后 ``world.state()`` 完全一致）。

测试用两种驱动方式：
1. 真实小世界 + ``behavior.InnerLife``（不依赖 Lead 尚未落地的集成层）；
2. 最小 fake world/agent/population 对象（确定性断言，覆盖边界路径）。

运行::

    cd newworldpro
    $env:TEMP = (Resolve-Path .tmp).Path
    python -m unittest tests.test_telemetry -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from newworld.behavior import InnerLife
from newworld.config import make_config
from newworld.telemetry import (
    ALL_METRIC_NAMES,
    EXTRA_METRIC_NAMES,
    METRIC_NAMES,
    RECOVERY_WINDOW_S,
    Telemetry,
)
from newworld.world import World

# DSH/Windows 沙箱下 tempfile.mkdtemp 创建的目录随后不可写（环境问题，
# 非被测代码问题）；Lead 的垫片把 mkdtemp 换成普通 mkdir，语义不变。
try:  # "python -m unittest tests.test_telemetry"
    from tests._env_compat import patch_tempdir
except ImportError:  # "python -m unittest discover -s tests"
    from _env_compat import patch_tempdir

patch_tempdir()

DT = 1.0 / 60.0
SIZE = (320, 240)
COUNT_METRICS = {
    "repeat_run_max",
    "tech_verified",
    "tech_graph_depth",
    "tech_graph_edges",
    "total_agents_created",
}


# ---------------------------------------------------------------------------
# 最小 fake 对象（确定性测试用；不依赖 Lead 的集成代码）
# ---------------------------------------------------------------------------
class FakePhysiology:
    def __init__(self, energy: float = 78.0, hydration: float = 80.0) -> None:
        self.energy = energy
        self.hydration = hydration


class FakeInventory:
    def __init__(self, wood: int = 0, stone: int = 0, tool: str | None = None) -> None:
        self.wood = wood
        self.stone = stone
        self.tool = tool


class FakeAgent:
    def __init__(
        self,
        agent_id: int = 0,
        *,
        x: float = 100.0,
        y: float = 200.0,
        state: str = "explore",
        known: tuple[str, ...] = (),
    ) -> None:
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.vx = 0.0
        self.facing = 1
        self.on_ground = True
        self.alive = True
        self.behavior_state = state
        self.physiology = FakePhysiology()
        self.inventory = FakeInventory()
        self.known_knowledge_ids = tuple(known)


class FakeCommunity:
    def __init__(self) -> None:
        self.wood = 0
        self.stone = 0
        self.structures = 0
        self.campfire = False


class FakeTechnology:
    def __init__(self) -> None:
        self.community = FakeCommunity()


class FakeWorld:
    """鸭子类型 world：agents/t/technology/config 都可缺省。"""

    def __init__(self, agents=None, *, t: float = 0.0, config=None) -> None:
        self.agents = list(agents or [])
        self.t = float(t)
        self.technology = FakeTechnology()
        self.config = config


class FakeFacility:
    def __init__(self, condition: float = 0.9) -> None:
        self.condition = condition
        self.active = condition > 0.2


class FakeContract:
    def __init__(self) -> None:
        self.active = True


class FakeRecord:
    def __init__(self, known=()) -> None:
        self.known_knowledge_ids = tuple(known)


class FakeRegistry:
    def __init__(self, total_created: int = 8, records=()) -> None:
        self.total_created = total_created
        self._records = list(records)

    def records(self):
        return self._records

    def alive_count(self):
        return 0


class FakeCivilization:
    def __init__(
        self,
        *,
        total_created: int = 8,
        facilities=None,
        contracts=None,
        extinct: bool = False,
        extinction_cause: str = "",
        innovation_count: int = 0,
    ) -> None:
        self.registry = FakeRegistry(total_created, [FakeRecord(("a", "c"))])
        self.facilities = list(facilities if facilities is not None else [FakeFacility()])
        self.contracts = list(contracts if contracts is not None else [FakeContract()])
        self.extinct = extinct
        self.extinction_cause = extinction_cause
        self.innovation_count = innovation_count
        self.cooperation_index = 0.4


class FakePlan:
    def __init__(self, plan_id: int = 1, current: int = 2, version: int = 1) -> None:
        self.plan_id = plan_id
        self.current = current
        self.version = version


class SeriesGoals:
    """``stats()`` 每次调用返回序列中的下一项（模拟目标计数增长）。"""

    def __init__(self, created, completed) -> None:
        self.created = list(created)
        self.completed = list(completed)
        self.calls = 0
        self.plan = FakePlan()

    def stats(self):
        index = min(self.calls, len(self.created) - 1)
        self.calls += 1
        return {
            "goals_created": self.created[index],
            "goals_completed": self.completed[index],
            "goals_failed": 0.0,
            "goals_abandoned": 0.0,
            "plan_success_rate": 0.5,
            "cost_estimate_error": 0.23,
        }

    def active_plan(self, agent_id: int):
        return self.plan


class FakeKnowledge:
    def __init__(self) -> None:
        self.last_ids: set[str] | None = None

    def all(self, status=None):
        return [object(), object()]  # 2 条 verified

    def depth(self):
        return 3

    def edges(self):
        return [("stone_axe", "basket"), ("fiber_cord", "basket")]

    def retention(self, ids):
        self.last_ids = set(ids)
        return len(ids) / 10.0

    def innovation_efficiency(self):
        return 0.25


def advance(fake_world: FakeWorld, telemetry: Telemetry, steps: int, dt: float) -> None:
    """推进 fake world 的虚拟时钟并采样。"""
    for _ in range(steps):
        fake_world.t = round(fake_world.t + dt, 9)
        telemetry.step(fake_world, dt=dt)


# ---------------------------------------------------------------------------
# 1) 真实小世界集成（320x240，60 步）
# ---------------------------------------------------------------------------
class TestRealSmallWorld(unittest.TestCase):
    """真实 ``World(make_config(320, 240))`` + ``InnerLife`` 驱动 60 步。"""

    @classmethod
    def setUpClass(cls) -> None:
        config = make_config(*SIZE)
        cls.world = World(config, seed=7)
        cls.world.reset(seed=7)
        cls.telemetry = Telemetry(window_s=60.0, seed=7, group="A")
        life = InnerLife(seed=3)
        for _ in range(60):
            decision = life.decide(cls.world, DT)
            cls.world.step(actions=decision, dt=DT)
            cls.telemetry.step(cls.world, dt=DT)
        cls.summary = cls.telemetry.summary()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.telemetry.close()

    def test_contract_metric_names_present(self) -> None:
        for name in METRIC_NAMES:
            self.assertIn(name, self.summary, f"缺少契约指标 {name}")
        for name in EXTRA_METRIC_NAMES:
            self.assertIn(name, self.summary, f"缺少附加诊断指标 {name}")

    def test_metric_types_are_json_friendly(self) -> None:
        for name in ALL_METRIC_NAMES:
            value = self.summary[name]
            if name == "extinction_cause":
                self.assertIsInstance(value, str)
            elif name in COUNT_METRICS:
                self.assertIsInstance(value, int)
                self.assertNotIsInstance(value, bool)
            else:
                self.assertIsInstance(value, float)
            self.assertNotIsInstance(value, (list, dict, set, tuple))
        self.assertIsInstance(self.summary["steps"], int)
        self.assertIsInstance(self.summary["seed"], int)
        self.assertIsInstance(self.summary["group"], str)
        self.assertEqual(self.summary["steps"], 60)
        self.assertAlmostEqual(self.summary["world_time"], 60 * DT, places=6)

    def test_summary_is_json_serialisable_and_round_trips(self) -> None:
        text = json.dumps(self.summary, ensure_ascii=False)
        loaded = json.loads(text)
        self.assertEqual(set(loaded), set(self.summary))
        for name in ALL_METRIC_NAMES:
            self.assertEqual(loaded[name], self.summary[name])

    def test_action_distribution_sums_to_one(self) -> None:
        distribution = self.telemetry.action_distribution()
        self.assertTrue(distribution, "有 agent 步时动作分布不能为空")
        self.assertAlmostEqual(sum(distribution.values()), 1.0, places=9)
        for value in distribution.values():
            self.assertGreater(value, 0.0)
        window = self.telemetry.action_distribution(whole_run=False)
        if window:
            self.assertAlmostEqual(sum(window.values()), 1.0, places=9)

    def test_entropy_and_repeat_run_bounds(self) -> None:
        entropy = self.summary["action_entropy"]
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)
        repeat_max = self.summary["repeat_run_max"]
        self.assertGreaterEqual(repeat_max, 1)
        self.assertLessEqual(repeat_max, 60)
        # 60 步里状态种类越少，归一化熵必然越低；至少不能超出上界
        states = self.telemetry.state_distribution()
        self.assertAlmostEqual(sum(states.values()), 1.0, places=9)
        self.assertAlmostEqual(self.summary["dominant_state_ratio"], max(states.values()), places=9)

    def test_stagnation_metrics_are_consistent(self) -> None:
        # 两个口径都是比例，都在 [0,1]
        for name in ("no_progress_repeat_ratio", "state_identical_ratio", "dominant_state_ratio"):
            self.assertGreaterEqual(self.summary[name], 0.0, name)
            self.assertLessEqual(self.summary[name], 1.0, name)
        # 生理量连续变化 ⇒ 严格快照口径通常极低（Lead 基线探针同样观察）
        self.assertLess(self.summary["state_identical_ratio"], 0.5)

    def test_window_summary_has_all_keys_and_types(self) -> None:
        window = self.telemetry.window_summary()
        self.assertEqual(set(window), set(ALL_METRIC_NAMES))
        for name in ALL_METRIC_NAMES:
            if name == "extinction_cause":
                self.assertIsInstance(window[name], str)
            else:
                self.assertIsInstance(window[name], float)
        self.assertIsInstance(self.summary["windows"], int)

    def test_telemetry_does_not_mutate_world(self) -> None:
        world = World(make_config(*SIZE), seed=11)
        world.reset(seed=11)
        before = json.dumps(world.state(), sort_keys=True, ensure_ascii=False)
        telemetry = Telemetry()
        for _ in range(5):
            telemetry.step(world, dt=DT)  # 只采样、不推进世界
        after = json.dumps(world.state(), sort_keys=True, ensure_ascii=False)
        self.assertEqual(before, after, "telemetry.step 修改了世界状态")
        self.assertEqual(len(telemetry.events), 0)


# ---------------------------------------------------------------------------
# 2) 无 population / 属性缺失时的降级
# ---------------------------------------------------------------------------
class TestGracefulDegradation(unittest.TestCase):
    def test_empty_telemetry_defaults(self) -> None:
        telemetry = Telemetry()
        summary = telemetry.summary()
        self.assertTrue(set(ALL_METRIC_NAMES) <= set(summary))
        self.assertEqual(summary["extinction_time"], -1.0)
        self.assertEqual(summary["extinction_cause"], "")
        self.assertEqual(summary["recovery_after_disturbance"], 1.0)
        self.assertEqual(summary["steps"], 0)
        self.assertEqual(telemetry.action_distribution(), {})
        self.assertEqual(telemetry.state_distribution(), {})
        window = telemetry.window_summary()
        self.assertEqual(set(window), set(ALL_METRIC_NAMES))

    def test_world_without_population(self) -> None:
        world = FakeWorld([FakeAgent(0)])
        telemetry = Telemetry()
        advance(world, telemetry, 5, dt=1.0)
        summary = telemetry.summary()
        self.assertEqual(summary["goal_completion_rate"], 0.0)
        self.assertEqual(summary["plan_cost_error"], 0.0)
        self.assertEqual(summary["tech_verified"], 0)
        self.assertEqual(summary["knowledge_retention"], 0.0)
        self.assertEqual(summary["innovation_efficiency"], 0.0)
        # 退化口径：观测期内见过的 agent_id 数
        self.assertEqual(summary["total_agents_created"], 1)
        self.assertEqual(summary["alive_count"], 1)

    def test_population_with_none_attributes(self) -> None:
        world = FakeWorld([FakeAgent(0)])
        population = SimpleNamespace(goals=None, knowledge=None, civilization=None, memory=None)
        telemetry = Telemetry()
        for _ in range(3):
            world.t += 1.0
            telemetry.step(world, population, dt=1.0)
        summary = telemetry.summary()
        self.assertEqual(summary["tech_verified"], 0)
        self.assertEqual(summary["memory_agents"], 0)
        json.dumps(summary, ensure_ascii=False)

    def test_plain_object_population(self) -> None:
        world = FakeWorld([FakeAgent(0)])
        telemetry = Telemetry()
        telemetry.step(world, object(), dt=1.0)
        summary = telemetry.summary()  # 不抛异常
        self.assertEqual(summary["steps"], 1)

    def test_broken_attributes_never_raise(self) -> None:
        class BrokenAgent:
            agent_id = 0

            @property
            def alive(self):
                raise RuntimeError("boom alive")

            @property
            def x(self):
                raise RuntimeError("boom x")

            @property
            def behavior_state(self):
                raise RuntimeError("boom state")

            @property
            def inventory(self):
                raise RuntimeError("boom inventory")

            @property
            def physiology(self):
                raise RuntimeError("boom physiology")

        class BrokenKnowledge:
            def all(self, status=None):
                raise RuntimeError("boom all")

            def depth(self):
                raise RuntimeError("boom depth")

            def edges(self):
                raise RuntimeError("boom edges")

            def retention(self, ids):
                raise RuntimeError("boom retention")

            def innovation_efficiency(self):
                raise RuntimeError("boom efficiency")

        class BrokenCiv:
            @property
            def registry(self):
                raise RuntimeError("boom registry")

            @property
            def facilities(self):
                raise RuntimeError("boom facilities")

            @property
            def contracts(self):
                raise RuntimeError("boom contracts")

        world = FakeWorld([BrokenAgent()])
        population = SimpleNamespace(
            goals=object(), knowledge=BrokenKnowledge(), civilization=BrokenCiv(), memory=object()
        )
        telemetry = Telemetry()
        for _ in range(3):
            world.t += 1.0
            telemetry.step(world, population, dt=1.0)
        summary = telemetry.summary()
        self.assertEqual(summary["tech_verified"], 0)
        self.assertIn("no_progress_repeat_ratio", summary)
        telemetry.close()

    def test_world_without_agents_attribute(self) -> None:
        class EmptyWorld:
            t = 0.0

        telemetry = Telemetry()
        telemetry.step(EmptyWorld(), dt=1.0)
        summary = telemetry.summary()
        self.assertEqual(summary["alive_count"], 0)
        self.assertGreaterEqual(summary["extinction_time"], 0.0)


# ---------------------------------------------------------------------------
# 3) 可选 population 属性的指标映射
# ---------------------------------------------------------------------------
class TestOptionalPopulationMetrics(unittest.TestCase):
    def _population(self):
        goals = SeriesGoals(created=[0, 0, 1, 1, 2], completed=[0, 0, 0, 1, 1])
        knowledge = FakeKnowledge()
        civ = FakeCivilization(total_created=8, innovation_count=2)
        memory = SimpleNamespace(agent_count=lambda: 3)
        return SimpleNamespace(
            goals=goals, knowledge=knowledge, civilization=civ, memory=memory
        ), knowledge

    def test_goal_knowledge_civilization_mapping(self) -> None:
        agent = FakeAgent(0, known=("a", "b"))
        world = FakeWorld([agent])
        population, knowledge = self._population()
        telemetry = Telemetry()
        for _ in range(5):
            world.t += 0.1
            telemetry.step(world, population, dt=0.1)
        summary = telemetry.summary()

        self.assertAlmostEqual(summary["plan_cost_error"], 0.23, places=9)
        # 生成 2 个目标、完成 1 个 → 0.5
        self.assertAlmostEqual(summary["goal_completion_rate"], 0.5, places=9)
        self.assertEqual(summary["tech_verified"], 2)
        self.assertEqual(summary["tech_graph_depth"], 3)
        self.assertEqual(summary["tech_graph_edges"], 2)
        self.assertAlmostEqual(summary["knowledge_retention"], 0.3, places=9)
        self.assertAlmostEqual(summary["innovation_efficiency"], 0.25, places=9)
        self.assertEqual(summary["total_agents_created"], 8)
        self.assertEqual(summary["memory_agents"], 3)
        # retention 收到的应是"存活 ∪ 死亡档案"的知识 id 并集
        self.assertEqual(knowledge.last_ids, {"a", "b", "c"})

    def test_total_agents_created_capped_by_registry(self) -> None:
        world = FakeWorld([FakeAgent(0)])
        population = SimpleNamespace(civilization=FakeCivilization(total_created=6))
        telemetry = Telemetry()
        telemetry.step(world, population, dt=1.0)
        summary = telemetry.summary()
        self.assertEqual(summary["total_agents_created"], 6)
        self.assertEqual(summary["tech_verified"], 0)  # innovation_count=0 时
        json.dumps(summary, ensure_ascii=False)

    def test_known_union_affects_retention(self) -> None:
        agents = [FakeAgent(0, known=("a",)), FakeAgent(1, known=("b",))]
        world = FakeWorld(agents)
        population, knowledge = self._population()
        telemetry = Telemetry()
        telemetry.step(world, population, dt=1.0)
        self.assertEqual(knowledge.last_ids, {"a", "b", "c"})


# ---------------------------------------------------------------------------
# 4) 两个"停滞"口径 + 附加诊断指标
# ---------------------------------------------------------------------------
class TestStagnationMetrics(unittest.TestCase):
    def test_repeat_run_and_no_progress_on_frozen_agent(self) -> None:
        agent = FakeAgent(0, x=100.0, state="rest")
        world = FakeWorld([agent])
        telemetry = Telemetry()
        advance(world, telemetry, 20, dt=1.0)
        summary = telemetry.summary()
        self.assertEqual(summary["repeat_run_max"], 20)
        self.assertAlmostEqual(summary["no_progress_repeat_ratio"], 19 / 20, places=9)
        self.assertAlmostEqual(summary["state_identical_ratio"], 19 / 20, places=9)
        self.assertAlmostEqual(summary["dominant_state_ratio"], 1.0, places=9)

    def test_moving_agent_is_not_no_progress(self) -> None:
        agent = FakeAgent(0, x=100.0, state="explore")

        def move(_world, _index):
            agent.x += 2.0

        world = FakeWorld([agent])
        telemetry = Telemetry()
        for index in range(20):
            world.t += DT
            move(world, index)
            telemetry.step(world, dt=DT)
        summary = telemetry.summary()
        # 每步都真的移动了 → 没有"无进展重复动作"
        self.assertEqual(summary["no_progress_repeat_ratio"], 0.0)
        self.assertLess(summary["state_identical_ratio"], 0.5)

    def test_inventory_change_counts_as_progress(self) -> None:
        agent = FakeAgent(0, x=100.0, state="gather")
        world = FakeWorld([agent])
        telemetry = Telemetry()
        for index in range(20):
            world.t += 1.0
            if index % 2 == 0:
                agent.inventory.wood += 1
            telemetry.step(world, dt=1.0)
        summary = telemetry.summary()
        # 库存每两步变化一次 → 无进展比例明显低于"完全冻结"
        self.assertLess(summary["no_progress_repeat_ratio"], 0.6)
        self.assertGreater(summary["no_progress_repeat_ratio"], 0.0)

    def test_tech_actions_per_1000_steps_material_proxy(self) -> None:
        # 技术动作 = 材料单位正增量代理：每步 +2 wood，10 步中后 9 步计入 2 单位
        agent = FakeAgent(0, state="gather")
        world = FakeWorld([agent])
        telemetry = Telemetry()
        for index in range(10):
            agent.inventory.wood = index * 2
            world.t += 1.0
            telemetry.step(world, dt=1.0)
        # 18 材料单位 / 10 步 × 1000 = 1800
        self.assertAlmostEqual(
            telemetry.summary()["tech_actions_per_1000_steps"], 1800.0, places=9
        )

    def test_tech_actions_count_community_contributions(self) -> None:
        world = FakeWorld([FakeAgent(0, state="build")])
        telemetry = Telemetry()
        telemetry.step(world, dt=1.0)
        world.technology.community.wood += 2
        world.technology.community.stone += 1
        world.t += 1.0
        telemetry.step(world, dt=1.0)
        # 3 材料单位 / 2 步 × 1000 = 1500
        self.assertAlmostEqual(
            telemetry.summary()["tech_actions_per_1000_steps"], 1500.0, places=9
        )

    def test_tech_actions_zero_without_material_gain(self) -> None:
        agent = FakeAgent(0, x=100.0, state="social")
        world = FakeWorld([agent])
        telemetry = Telemetry()
        advance(world, telemetry, 20, dt=1.0)
        self.assertEqual(telemetry.summary()["tech_actions_per_1000_steps"], 0.0)

    def test_state_identical_ratio_is_strict_contrast(self) -> None:
        # 生理量每步连续变化：严格快照几乎不相等，但动作/位置可以完全不变
        agent = FakeAgent(0, x=50.0, state="rest")
        world = FakeWorld([agent])
        telemetry = Telemetry()
        for _ in range(30):
            world.t += 1.0
            agent.physiology.energy -= 0.1
            telemetry.step(world, dt=1.0)
        summary = telemetry.summary()
        self.assertEqual(summary["state_identical_ratio"], 0.0)
        # 但"有意义的进展"口径仍然能识别停滞（没有库存/位置/状态变化）
        self.assertGreater(summary["no_progress_repeat_ratio"], 0.9)


# ---------------------------------------------------------------------------
# 5) 生存率 / 灭绝 / 扰动恢复 / 合作收益
# ---------------------------------------------------------------------------
class TestLifecycleMetrics(unittest.TestCase):
    def test_survival_rate_person_seconds(self) -> None:
        agents = [FakeAgent(0), FakeAgent(1)]
        world = FakeWorld(agents)
        telemetry = Telemetry()
        for index in range(40):
            if index == 20:
                world.agents.pop()  # 第 21 步起只剩 1 个存活个体
            world.t += 1.0
            telemetry.step(world, dt=1.0)
        # 20 步 × 2 个 + 20 步 × 1 个 = 60 个体-秒；分母 2 × 40 = 80
        self.assertAlmostEqual(telemetry.summary()["survival_rate"], 0.75, places=9)

    def test_extinction_time_and_cause(self) -> None:
        agents = [FakeAgent(0), FakeAgent(1)]
        world = FakeWorld(agents)
        civ = FakeCivilization(extinct=True, extinction_cause="starvation")
        population = SimpleNamespace(civilization=civ)
        telemetry = Telemetry()
        for _ in range(3):
            world.t += 1.0
            telemetry.step(world, population, dt=1.0)
        world.agents.clear()
        world.t += 1.0
        telemetry.step(world, population, dt=1.0)
        summary = telemetry.summary()
        self.assertGreaterEqual(summary["extinction_time"], 0.0)
        self.assertEqual(summary["extinction_cause"], "starvation")
        # 灭绝后继续采样不得抛错
        world.t += 1.0
        telemetry.step(world, population, dt=1.0)
        self.assertEqual(telemetry.summary()["alive_count"], 0)

    def test_extinction_cause_from_death_event(self) -> None:
        world = FakeWorld([FakeAgent(0)])
        telemetry = Telemetry()
        telemetry.step(world, dt=1.0)
        telemetry.event("death", agent_id=0, cause="old_age")
        world.agents.clear()
        world.t += 1.0
        telemetry.step(world, dt=1.0)
        self.assertEqual(telemetry.summary()["extinction_cause"], "old_age")

    def test_cooperation_benefit_symmetric_log_ratio(self) -> None:
        balanced = Telemetry()
        balanced.event("coop_output", amount=4.0)
        balanced.event("solo_output", amount=4.0)
        self.assertAlmostEqual(balanced.summary()["cooperation_benefit"], 0.0, places=9)

        positive = Telemetry()
        positive.event("coop_output", amount=9.0)
        self.assertGreater(positive.summary()["cooperation_benefit"], 0.0)

        negative = Telemetry()
        negative.event("solo_output", amount=9.0)
        self.assertLess(negative.summary()["cooperation_benefit"], 0.0)

        # 对称化：互换分子分母恰好取负
        left = Telemetry()
        left.event("coop_output", amount=1.0)
        left.event("solo_output", amount=3.0)
        right = Telemetry()
        right.event("coop_output", amount=3.0)
        right.event("solo_output", amount=1.0)
        self.assertAlmostEqual(
            left.summary()["cooperation_benefit"],
            -right.summary()["cooperation_benefit"],
            places=9,
        )
        # 有界 [-1, 1]
        extreme = Telemetry()
        extreme.event("coop_output", amount=1000.0)
        self.assertLessEqual(extreme.summary()["cooperation_benefit"], 1.0)

    def test_private_inventory_gain_counts_as_solo_output(self) -> None:
        agent = FakeAgent(0, x=10.0, state="gather")
        world = FakeWorld([agent])
        telemetry = Telemetry()
        telemetry.step(world, dt=1.0)
        agent.inventory.wood += 5
        world.t += 1.0
        telemetry.step(world, dt=1.0)
        # 只有独立产出 → 合作收益应为负
        self.assertLess(telemetry.summary()["cooperation_benefit"], 0.0)

    def test_recovery_after_disturbance(self) -> None:
        agents = [FakeAgent(0), FakeAgent(1)]
        world = FakeWorld(agents)
        population = SimpleNamespace(
            knowledge=FakeKnowledge(), civilization=FakeCivilization()
        )
        telemetry = Telemetry()
        telemetry.step(world, population, dt=1.0)
        telemetry.event("disturbance", kind_detail="storm")
        self.assertLessEqual(RECOVERY_WINDOW_S, 600.0)  # 常量存在且有意义
        world.agents.pop()  # 人口减半，知识/设施不变
        world.t += 1.0
        telemetry.step(world, population, dt=1.0)
        expected = (1 / 2 + 1.0 + 1.0) / 3.0
        self.assertAlmostEqual(
            telemetry.summary()["recovery_after_disturbance"], expected, places=6
        )


# ---------------------------------------------------------------------------
# 6) 窗口结算 / JSONL / to_json / close
# ---------------------------------------------------------------------------
class TestOutputAndLifecycle(unittest.TestCase):
    def test_window_rollover(self) -> None:
        world = FakeWorld([FakeAgent(0), FakeAgent(1)])
        telemetry = Telemetry(window_s=0.5)
        advance(world, telemetry, 20, dt=0.1)  # 2.0 s → 4 个窗口
        self.assertEqual(len(telemetry.windows), 4)
        for window in telemetry.windows:
            self.assertEqual(set(window), set(ALL_METRIC_NAMES))
            for name, value in window.items():
                if name != "extinction_cause":
                    self.assertIsInstance(value, float)
        # 刚结算完：window_summary 返回最近一次结算值
        self.assertEqual(set(telemetry.window_summary()), set(ALL_METRIC_NAMES))

    def test_event_jsonl_and_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            telemetry = Telemetry(log_path=path, seed=3, group="C")
            telemetry.event("birth", agent_id=2)
            telemetry.event("death", agent_id=0, cause="starvation")
            telemetry.event("discover", knowledge_id="stone_axe", agent_id=2)
            self.assertEqual(len(telemetry.events), 3)
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(lines), 3)
            first = json.loads(lines[0])
            self.assertEqual(first["kind"], "birth")
            self.assertEqual(first["seed"], 3)
            self.assertEqual(first["group"], "C")
            self.assertIn("t", first)
            self.assertEqual(json.loads(lines[1])["cause"], "starvation")
            telemetry.close()
            telemetry.close()  # 幂等
            telemetry.event("after_close", agent_id=1)  # 静默忽略
            self.assertEqual(len(telemetry.events), 3)
            lines_after = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines_after), 3)
            with self.assertRaises(RuntimeError):
                telemetry.step(FakeWorld([FakeAgent(0)]), dt=1.0)

    def test_to_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.jsonl"
            telemetry = Telemetry(window_s=0.2, log_path=log_path)
            world = FakeWorld([FakeAgent(0)], t=0.0)
            telemetry.event("birth", agent_id=0)
            advance(world, telemetry, 10, dt=0.1)
            summary = telemetry.summary()
            path = telemetry.to_json()  # 默认 <log_path>.summary.json
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "run.jsonl.summary.json")
            data = json.loads(path.read_text(encoding="utf-8"))
            for name in ALL_METRIC_NAMES:
                self.assertEqual(data[name], summary[name])
            self.assertIsInstance(data["windows"], list)
            self.assertTrue(data["windows"])
            self.assertEqual(data["events_count"], 1)
            self.assertAlmostEqual(sum(data["action_distribution"].values()), 1.0, places=9)
            telemetry.close()  # Windows: 句柄不关则临时目录删不掉

    def test_to_json_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            telemetry = Telemetry()
            telemetry.step(FakeWorld([FakeAgent(0)]), dt=1.0)
            target = Path(tmp) / "nested" / "summary.json"
            written = telemetry.to_json(target)
            self.assertEqual(written, target)
            data = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(data["steps"], 1)
            self.assertIn("action_distribution", data)
            self.assertAlmostEqual(sum(data["action_distribution"].values()), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
