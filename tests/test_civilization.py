"""``newworld.civilization`` 单元测试（P3 注册表 / 公共设施 / 文明状态机）。

只使用最小 fake world / fake agent（``agent_id/name/sex/generation/parents/
inventory/known_knowledge_ids/physiology``），不构造真实 ``World`` / ``Character``，
不依赖 Lead 尚未落地的集成代码。运行：``python -m unittest tests.test_civilization -v``。
"""

from __future__ import annotations

import json
import unittest

from newworld.civilization import (
    COLLAPSE_STATE_CODES,
    FACILITY_ACTIVE_THRESHOLD,
    FACILITY_DECAY_PER_SECOND,
    HISTORY_LIMIT,
    MAX_AGENTS_CREATED,
    STAGNATION_S,
    AgentRecord,
    AgentRegistry,
    CivilizationState,
    Facility,
    PopulationLimitError,
    SocialContract,
)
from newworld.memory import MemoryStore


# ---------------------------------------------------------------------------
# 最小 fake world / agent（只提供契约用到的属性）
# ---------------------------------------------------------------------------
class FakeInventory:
    def __init__(self, wood: int = 0, stone: int = 0, tool: str | None = None) -> None:
        self.wood = int(wood)
        self.stone = int(stone)
        self.tool = tool


class FakePhysiology:
    def __init__(self, energy: float = 80.0, hydration: float = 80.0) -> None:
        self.energy = float(energy)
        self.hydration = float(hydration)
        self.sleep_pressure = 0.0


class FakeAgent:
    def __init__(
        self,
        agent_id: int,
        *,
        name: str = "",
        sex: str = "male",
        generation: int = 0,
        parents: tuple[int, ...] = (),
        age: float = 120.0,
        energy: float = 80.0,
        wood: int = 0,
        stone: int = 0,
    ) -> None:
        self.agent_id = int(agent_id)
        self.name = name or f"ender-{agent_id}"
        self.sex = sex
        self.generation = int(generation)
        self.parents = tuple(parents)
        self.age = float(age)
        self.alive = True
        self.x = 300.0
        self.behavior_state = "explore"
        self.inventory = FakeInventory(wood=wood, stone=stone)
        self.known_knowledge_ids: tuple[str, ...] = ()
        self.skills: dict[str, float] = {}
        self.contributions: dict[str, float] = {}
        self.physiology = FakePhysiology(energy=energy)


class FakeWorld:
    def __init__(self, t: float = 0.0, agents=()) -> None:
        self.t = float(t)
        self.agents = list(agents)


def make_registry(max_agents: int = MAX_AGENTS_CREATED) -> AgentRegistry:
    return AgentRegistry(max_agents=max_agents)


# ---------------------------------------------------------------------------
# AgentRegistry：8 个创建名额硬约束
# ---------------------------------------------------------------------------
class TestAgentLimit(unittest.TestCase):
    def test_ninth_register_raises_and_counter_frozen(self) -> None:
        registry = make_registry()
        for index in range(MAX_AGENTS_CREATED):
            record = registry.register(FakeAgent(index), t=float(index))
            self.assertTrue(record.alive)
            self.assertEqual(record.born_at, float(index))
        self.assertEqual(registry.total_created, MAX_AGENTS_CREATED)
        self.assertEqual(registry.remaining(), 0)
        self.assertFalse(registry.can_create())

        ninth = FakeAgent(MAX_AGENTS_CREATED)
        with self.assertRaises(PopulationLimitError):
            registry.register(ninth, t=99.0)
        self.assertEqual(registry.total_created, MAX_AGENTS_CREATED)
        self.assertEqual(registry.remaining(), 0)
        self.assertIsNone(registry.record(MAX_AGENTS_CREATED))
        self.assertEqual(len(registry.records()), MAX_AGENTS_CREATED)

        # 死亡不退还名额：计数只增不减
        registry.mark_dead(0, t=100.0, cause="old_age")
        self.assertEqual(registry.total_created, MAX_AGENTS_CREATED)
        with self.assertRaises(PopulationLimitError):
            registry.register(FakeAgent(100), t=101.0)

    def test_duplicate_registration_rejected(self) -> None:
        registry = make_registry(max_agents=4)
        registry.register(FakeAgent(0), t=0.0)
        with self.assertRaises(ValueError):
            registry.register(FakeAgent(0), t=1.0)
        self.assertEqual(registry.total_created, 1)

    def test_records_include_dead_and_alive_ids_exclude_dead(self) -> None:
        registry = make_registry(max_agents=4)
        for index in range(3):
            registry.register(FakeAgent(index), t=float(index))
        registry.mark_dead(1, t=10.0, cause="starvation")
        self.assertEqual([record.agent_id for record in registry.records()], [0, 1, 2])
        self.assertEqual(registry.alive_ids(), [0, 2])
        self.assertEqual(registry.alive_count(), 2)
        self.assertEqual(registry.total_deaths, 1)
        self.assertEqual(registry.total_created, 3)


# ---------------------------------------------------------------------------
# 死亡档案（§3：死亡不可行动，历史不能删）
# ---------------------------------------------------------------------------
class TestDeathRecords(unittest.TestCase):
    def test_mark_dead_keeps_full_archive(self) -> None:
        registry = make_registry(max_agents=4)
        agent = FakeAgent(0, name="end-0", sex="female", generation=2)
        agent.inventory = FakeInventory(wood=3, stone=1)
        agent.known_knowledge_ids = ("stone_axe", "fiber_cord")
        agent.skills = {"stone_axe": 0.8}
        agent.contributions = {"contribute": 2.0}
        record = registry.register(agent, t=0.0)
        self.assertEqual(record.contributions, {"contribute": 2.0})

        dead = registry.mark_dead(
            0,
            t=100.0,
            cause="starvation",
            final_state={
                "inventory": {"wood": 3, "stone": 1},
                "known_knowledge_ids": ["stone_axe", "fiber_cord"],
                "skills": {"stone_axe": 0.8},
                "energy": 0.0,
            },
        )
        self.assertIs(dead, record)
        self.assertFalse(record.alive)
        self.assertEqual(record.died_at, 100.0)
        self.assertEqual(record.death_cause, "starvation")
        self.assertEqual(record.cause, "starvation")
        self.assertEqual(record.inventory, {"wood": 3, "stone": 1})
        self.assertEqual(record.known_knowledge_ids, ("stone_axe", "fiber_cord"))
        self.assertEqual(record.skills, {"stone_axe": 0.8})
        self.assertEqual(record.final_state, {"energy": 0.0})
        self.assertEqual(registry.alive_ids(), [])
        self.assertEqual(registry.records()[0].agent_id, 0)
        self.assertEqual(registry.total_deaths, 1)

        # 重复标记幂等：不重复计数、不覆盖死亡时间
        again = registry.mark_dead(0, t=500.0, cause="old_age")
        self.assertIs(again, record)
        self.assertEqual(record.death_cause, "starvation")
        self.assertEqual(record.died_at, 100.0)
        self.assertEqual(registry.total_deaths, 1)
        self.assertIsNone(registry.mark_dead(99, t=1.0, cause="x"))

    def test_register_death_snapshots_agent_and_memory_archive(self) -> None:
        memory = MemoryStore()
        memory.record(0, 0.0, "procedural", "craft:stone_axe", "ok:axe", True)
        memory.learn_procedure(0, "stone_axe", ("gather:wood", "craft"))
        registry = make_registry(max_agents=4)
        agent = FakeAgent(0, name="end-0", age=300.0)
        registry.register(agent, t=0.0)
        state = CivilizationState(registry, memory=memory)

        record = state.register_death(agent, t=320.0, cause="old_age")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.died_at, 320.0)
        self.assertEqual(record.death_cause, "old_age")
        self.assertIn("age", record.final_state)
        self.assertEqual(record.final_state["age"], 300.0)
        self.assertEqual(record.inventory, {"wood": 0, "stone": 0})
        self.assertEqual(record.memory_archive["entry_count"], 1)
        self.assertEqual(
            record.memory_archive["procedures"]["stone_axe"]["skill"], "stone_axe"
        )
        # archive 不删除任何记忆
        self.assertEqual(len(memory.entries(0)), 1)


# ---------------------------------------------------------------------------
# 家谱 / 序列化
# ---------------------------------------------------------------------------
class TestLineageAndSerialization(unittest.TestCase):
    def test_lineage_bidirectional(self) -> None:
        registry = make_registry()
        registry.register(FakeAgent(0), t=0.0)
        registry.register(FakeAgent(1, parents=(0,)), t=10.0)
        registry.register(FakeAgent(2), t=20.0)
        # parent_id 显式给出（agent.parents 为空）也应建立关系
        registry.register(FakeAgent(3, parents=()), t=30.0, parent_id=2)
        registry.register(FakeAgent(4, parents=(1,)), t=40.0)
        self.assertEqual(registry.record(0).children, (1,))
        self.assertEqual(registry.record(2).children, (3,))
        self.assertIn((0, 1), registry.lineage())
        self.assertIn((2, 3), registry.lineage())
        self.assertIn((1, 4), registry.lineage())

    def test_registry_to_dict_from_dict_roundtrip(self) -> None:
        registry = make_registry(max_agents=3)
        registry.register(FakeAgent(0), t=0.0)
        registry.register(FakeAgent(1, parents=(0,)), t=5.0)
        registry.register(FakeAgent(2, parents=(1,)), t=10.0)
        registry.mark_dead(1, t=20.0, cause="old_age")
        payload = json.loads(json.dumps(registry.to_dict()))

        restored = AgentRegistry.from_dict(payload)
        self.assertEqual(restored.max_agents, 3)
        self.assertEqual(restored.total_created, 3)
        self.assertEqual(restored.total_deaths, 1)
        self.assertFalse(restored.can_create())
        self.assertEqual(restored.alive_ids(), [0, 2])
        self.assertEqual(len(restored.records()), 3)
        self.assertEqual(restored.record(1).death_cause, "old_age")
        self.assertEqual(restored.record(1).died_at, 20.0)
        self.assertEqual(restored.lineage(), [(0, 1), (1, 2)])

    def test_agent_record_roundtrip(self) -> None:
        record = AgentRecord(
            agent_id=7,
            parent_id=3,
            name="end-7",
            sex="female",
            generation=2,
            born_at=12.5,
            died_at=900.0,
            death_cause="starvation",
            alive=False,
            final_state={"energy": 0.0},
            inventory={"wood": 2},
            known_knowledge_ids=("stone_axe",),
            skills={"stone_axe": 0.5},
            children=(9,),
            contributions={"share_food": 1.0},
            memory_archive={"entry_count": 3},
        )
        self.assertEqual(AgentRecord.from_dict(record.to_dict()), record)


# ---------------------------------------------------------------------------
# Facility：损耗 / 维护 / active 阈值
# ---------------------------------------------------------------------------
class TestFacility(unittest.TestCase):
    def test_decay_and_active_threshold(self) -> None:
        facility = Facility(
            facility_id="campfire-01",
            kind="campfire",
            name_zh="篝火",
            x_screen=400.0,
            built_at=0.0,
            last_maintained_at=0.0,
        )
        self.assertTrue(facility.active)
        self.assertAlmostEqual(facility.condition, 1.0)
        # 损耗率由模块常量决定（1/1200 每秒，可在文明层调整）
        self.assertAlmostEqual(
            facility.decay(300.0), 1.0 - 300.0 * FACILITY_DECAY_PER_SECOND, delta=1e-6
        )
        self.assertTrue(facility.active)
        condition = facility.decay(1000.0)
        self.assertLessEqual(condition, FACILITY_ACTIVE_THRESHOLD + 1e-9)
        self.assertFalse(facility.active)
        self.assertGreaterEqual(facility.condition, 0.0)
        # 同一时刻重复 decay 不重复扣减
        anchored = facility.condition
        self.assertEqual(facility.decay(1000.0), anchored)

    def test_maintenance_restores_condition(self) -> None:
        facility = Facility(
            facility_id="campfire-01",
            kind="campfire",
            name_zh="篝火",
            x_screen=400.0,
            built_at=0.0,
            last_maintained_at=0.0,
        )
        facility.decay(1000.0)
        self.assertFalse(facility.active)
        restored = min(1.0, facility.condition + 0.35)
        facility.maintained(3, 1000.0)
        self.assertAlmostEqual(facility.condition, restored, delta=1e-6)
        self.assertTrue(facility.active)
        self.assertEqual(facility.last_maintained_at, 1000.0)
        # 维护重置损耗锚点：同一时刻再 decay 不扣减
        self.assertAlmostEqual(facility.decay(1000.0), restored, delta=1e-6)
        # 继续损耗会再次失效
        facility.decay(2000.0)
        self.assertFalse(facility.active)

    def test_facility_dict_roundtrip_preserves_decay_anchor(self) -> None:
        facility = Facility(
            facility_id="hut-01",
            kind="hut",
            name_zh="小屋",
            x_screen=250.0,
            built_at=10.0,
            built_by=2,
            maintenance_wood=1,
            maintenance_stone=1,
            last_maintained_at=10.0,
        )
        facility.decay(310.0)
        payload = json.loads(json.dumps(facility.to_dict()))
        restored = Facility.from_dict(payload)
        self.assertAlmostEqual(restored.condition, facility.condition, delta=1e-9)
        self.assertAlmostEqual(restored.decay(610.0), facility.decay(610.0), delta=1e-9)


# ---------------------------------------------------------------------------
# 建造 / 维护（文明状态）
# ---------------------------------------------------------------------------
class TestBuildAndMaintain(unittest.TestCase):
    def test_build_costs_and_unknown_kind(self) -> None:
        registry = make_registry()
        agent = FakeAgent(0, wood=5, stone=2)
        registry.register(agent, t=0.0)
        state = CivilizationState(registry)

        facility = state.build_facility("campfire", 400.0, 0, 0.0, inventory=agent.inventory)
        self.assertIsNotNone(facility)
        assert facility is not None
        self.assertEqual(facility.kind, "campfire")
        self.assertEqual(facility.name_zh, "篝火")
        self.assertEqual(facility.built_by, 0)
        self.assertEqual(facility.built_at, 0.0)
        self.assertEqual((agent.inventory.wood, agent.inventory.stone), (3, 1))
        self.assertIsNotNone(state.build_facility("campfire", 400.0, 0, 0.0, inventory=agent.inventory))
        self.assertEqual(len(state.facilities), 2)
        # 材料不足
        self.assertIsNone(state.build_facility("campfire", 400.0, 0, 0.0, inventory=agent.inventory))
        self.assertEqual(len(state.facilities), 2)
        # 未知类型
        self.assertIsNone(state.build_facility("temple", 400.0, 0, 0.0))
        self.assertEqual(len(state.facilities), 2)

    def test_maintain_requires_resources_and_counts_contribution(self) -> None:
        registry = make_registry()
        agent = FakeAgent(0)
        registry.register(agent, t=0.0)
        state = CivilizationState(registry)
        facility = state.build_facility("campfire", 400.0, 0, 0.0)
        assert facility is not None
        self.assertFalse(state.maintain(facility.facility_id, 0, 60.0))
        self.assertTrue(state.maintain(facility.facility_id, 0, 60.0, wood=1, stone=0))
        self.assertEqual(facility.last_maintained_at, 60.0)
        record = registry.record(0)
        assert record is not None
        self.assertGreater(record.contributions.get("maintain", 0.0), 0.0)
        self.assertFalse(state.maintain("missing-99", 0, 60.0, wood=1))


# ---------------------------------------------------------------------------
# 灭绝判定（§3）
# ---------------------------------------------------------------------------
class TestExtinction(unittest.TestCase):
    def test_extinct_when_all_dead_and_cannot_create(self) -> None:
        registry = make_registry(max_agents=2)
        first, second = FakeAgent(0), FakeAgent(1)
        registry.register(first, t=0.0)
        registry.register(second, t=0.0)
        state = CivilizationState(registry)
        world = FakeWorld(t=0.0, agents=[first, second])
        state.update(world, 0.0)
        self.assertFalse(state.extinct)

        state.register_death(first, t=5.0, cause="starvation")
        self.assertFalse(state.extinct)
        state.register_death(second, t=6.0, cause="old_age")
        self.assertTrue(state.extinct)
        self.assertEqual(state.extinct_at, 6.0)
        self.assertEqual(state.extinction_cause, "old_age")
        self.assertEqual(state.collapse_state, "collapsed")

        world.t = 7.0
        world.agents = []
        state.update(world, 1.0)
        self.assertTrue(state.extinct)
        self.assertEqual(state.extinct_at, 6.0)
        self.assertEqual(state.collapse_state, "collapsed")
        metrics = state.metrics(world)
        self.assertEqual(metrics["extinct"], 1.0)
        self.assertEqual(metrics["extinction_time"], 6.0)
        self.assertEqual(metrics["collapse_state_code"], COLLAPSE_STATE_CODES["collapsed"])

    def test_not_extinct_while_creation_slots_remain(self) -> None:
        registry = make_registry(max_agents=4)
        first, second = FakeAgent(0), FakeAgent(1)
        registry.register(first, t=0.0)
        registry.register(second, t=0.0)
        state = CivilizationState(registry)
        state.register_death(first, t=5.0, cause="starvation")
        state.register_death(second, t=6.0, cause="old_age")
        # 名额还有 2 个 → 按 §3 不判灭绝，但没有存活个体的文明状态为 collapsed
        self.assertFalse(state.extinct)
        world = FakeWorld(t=10.0, agents=[])
        state.update(world, 1.0)
        self.assertFalse(state.extinct)
        self.assertEqual(state.collapse_state, "collapsed")

    def test_extinct_when_inheritance_disabled(self) -> None:
        registry = make_registry(max_agents=4)
        first, second = FakeAgent(0), FakeAgent(1)
        registry.register(first, t=0.0)
        registry.register(second, t=0.0)
        state = CivilizationState(registry, config={"enable_inheritance": False})
        state.register_death(first, t=5.0, cause="starvation")
        self.assertFalse(state.extinct)
        state.register_death(second, t=6.0, cause="old_age")
        self.assertTrue(state.extinct)


# ---------------------------------------------------------------------------
# collapse_state 五态机
# ---------------------------------------------------------------------------
class TestCollapseStateMachine(unittest.TestCase):
    def test_stable_then_stagnating_without_innovation(self) -> None:
        registry = make_registry()
        agents = [FakeAgent(0, energy=80.0), FakeAgent(1, energy=80.0)]
        for index, agent in enumerate(agents):
            registry.register(agent, t=float(index))
        state = CivilizationState(registry)
        world = FakeWorld(t=0.0, agents=agents)
        state.update(world, 0.0)
        self.assertEqual(state.collapse_state, "stable")
        self.assertGreaterEqual(STAGNATION_S, 1.0)

        world.t = 190.0
        state.update(world, 190.0)
        self.assertEqual(state.collapse_state, "stagnating")

    def test_collapsing_then_recovering(self) -> None:
        registry = make_registry()
        agents = [FakeAgent(index, energy=80.0) for index in range(3)]
        for agent in agents:
            registry.register(agent, t=0.0)
        state = CivilizationState(registry)
        world = FakeWorld(t=0.0, agents=list(agents))
        state.update(world, 0.0)
        self.assertEqual(state.collapse_state, "stable")

        state.register_death(agents[0], t=60.0, cause="starvation")
        world.agents = [agents[1], agents[2]]
        world.t = 130.0
        state.update(world, 70.0)
        self.assertLess(state._pop_change_ratio, 0.0)
        self.assertEqual(state.collapse_state, "collapsing")

        newborn = FakeAgent(3, parents=(1,), age=0.0, energy=80.0)
        registry.register(newborn, t=140.0)
        world.agents.append(newborn)
        world.t = 260.0
        state.update(world, 130.0)
        self.assertGreater(state._pop_change_ratio, 0.0)
        self.assertEqual(state.collapse_state, "recovering")

    def test_inactive_facilities_block_recovery(self) -> None:
        # 设施存在但全部失效时，即使人口正增长也不能判定为 recovering
        registry = make_registry()
        agents = [FakeAgent(index, energy=80.0) for index in range(3)]
        for agent in agents:
            registry.register(agent, t=0.0)
        state = CivilizationState(registry)
        state.build_facility("campfire", 400.0, 0, 0.0)
        world = FakeWorld(t=0.0, agents=list(agents))
        state.update(world, 0.0)
        self.assertEqual(state.collapse_state, "stable")

        state.register_death(agents[0], t=60.0, cause="starvation")
        world.agents = [agents[1], agents[2]]
        world.t = 1100.0
        state.update(world, 1100.0)
        self.assertFalse(state.facilities[0].active)
        self.assertEqual(state.collapse_state, "collapsing")

        newborn = FakeAgent(3, parents=(1,), age=0.0, energy=80.0)
        registry.register(newborn, t=1120.0)
        world.agents.append(newborn)
        world.t = 1220.0
        state.update(world, 120.0)
        self.assertGreater(state._pop_change_ratio, 0.0)
        self.assertNotEqual(state.collapse_state, "recovering")
        self.assertEqual(state.collapse_state, "stable")

    def test_facility_decay_does_not_affect_stagnation(self) -> None:
        # 设施失效只影响"恢复"判定，不影响停滞判定（避免隐式耦合）
        registry = make_registry()
        agent = FakeAgent(0, energy=80.0)
        registry.register(agent, t=0.0)
        state = CivilizationState(registry)
        state.build_facility("campfire", 400.0, 0, 0.0)
        world = FakeWorld(t=0.0, agents=[agent])
        state.update(world, 0.0)
        self.assertEqual(state.collapse_state, "stable")
        world.t = 1000.0
        state.update(world, 1000.0)
        self.assertFalse(state.facilities[0].active)
        self.assertEqual(state.collapse_state, "stagnating")


# ---------------------------------------------------------------------------
# 合作指数（§8 定义）
# ---------------------------------------------------------------------------
class TestCooperationIndex(unittest.TestCase):
    def _state_with_agent(self, agent: FakeAgent, **kwargs) -> CivilizationState:
        registry = make_registry()
        registry.register(agent, t=0.0)
        return CivilizationState(registry, **kwargs)

    def test_zero_without_data(self) -> None:
        agent = FakeAgent(0)
        state = self._state_with_agent(agent)
        world = FakeWorld(t=0.0, agents=[agent])
        state.update(world, 0.0)
        self.assertEqual(state.cooperation_index, 0.0)

    def test_positive_from_contributions(self) -> None:
        agent = FakeAgent(0)
        state = self._state_with_agent(agent)
        world = FakeWorld(t=0.0, agents=[agent])
        agent.contributions = {"contribute": 3.0}
        state.update(world, 0.0)
        # gain=3, independent=0 → 3 / (3 + 0 + 1) = 0.75
        self.assertAlmostEqual(state.cooperation_index, 3.0 / 4.0, places=6)
        self.assertGreater(state.cooperation_index, 0.0)
        self.assertLessEqual(state.cooperation_index, 1.0)

    def test_zero_when_social_disabled(self) -> None:
        agent = FakeAgent(0)
        state = self._state_with_agent(agent, config={"enable_social": False})
        world = FakeWorld(t=0.0, agents=[agent])
        agent.contributions = {"contribute": 3.0}
        state.update(world, 0.0)
        self.assertEqual(state.cooperation_index, 0.0)

    def test_independent_events_and_contracts_enter_formula(self) -> None:
        memory = MemoryStore()
        memory.record(0, 0.0, "episode", "gather:wood", "ok:wood+1", True)
        agent = FakeAgent(0)
        state = self._state_with_agent(agent, memory=memory)
        world = FakeWorld(t=0.0, agents=[agent])
        agent.contributions = {"contribute": 3.0}
        state.update(world, 0.0)
        # gain=3, independent=1 → 3 / (3 + 1 + 1) = 0.6
        self.assertAlmostEqual(state.cooperation_index, 3.0 / 5.0, places=6)

        state.contracts.append(
            SocialContract(
                contract_id="c-1",
                kind="share_food",
                name_zh="共享食物",
                proposed_by=0,
                created_at=0.0,
                participants=(0, 1),
                active=True,
            )
        )
        state.update(world, 0.0)
        # gain=3+1=4, independent=1 → 4 / (4 + 1 + 1)
        self.assertAlmostEqual(state.cooperation_index, 4.0 / 6.0, places=6)


# ---------------------------------------------------------------------------
# history / metrics / 序列化 / 摘要
# ---------------------------------------------------------------------------
class TestHistoryMetricsAndSerialization(unittest.TestCase):
    def test_history_ring_is_bounded(self) -> None:
        registry = make_registry()
        agent = FakeAgent(0)
        registry.register(agent, t=0.0)
        state = CivilizationState(registry)
        world = FakeWorld(t=0.0, agents=[agent])
        for step in range(HISTORY_LIMIT + 50):
            world.t = step * 60.0
            state.update(world, 60.0)
        self.assertEqual(len(state.history), HISTORY_LIMIT)
        self.assertLessEqual(len(state.history), 200)
        self.assertEqual(state.history[-1]["t"], float((HISTORY_LIMIT + 49) * 60))
        self.assertEqual(state.history[0]["t"], float(50 * 60))
        required = {
            "t",
            "alive",
            "total_created",
            "deaths",
            "innovation_count",
            "facilities_total",
            "facilities_active",
            "avg_energy",
            "population_change_ratio",
            "cooperation_index",
            "collapse_state",
            "extinct",
        }
        self.assertTrue(required.issubset(state.history[-1]))

    def test_metrics_are_json_serializable_floats(self) -> None:
        registry = make_registry()
        agent = FakeAgent(0, energy=70.0)
        registry.register(agent, t=0.0)
        state = CivilizationState(registry)
        world = FakeWorld(t=5.0, agents=[agent])
        state.update(world, 1.0)
        metrics = state.metrics(world)
        json.dumps(metrics)
        for name, value in metrics.items():
            self.assertIsInstance(value, float, f"{name} 必须是原生 float")
        self.assertEqual(metrics["alive_count"], 1.0)
        self.assertEqual(metrics["total_agents_created"], 1.0)
        self.assertEqual(metrics["extinction_time"], -1.0)
        self.assertEqual(metrics["collapse_state_code"], COLLAPSE_STATE_CODES["stable"])

    def test_state_roundtrip(self) -> None:
        registry = make_registry(max_agents=4)
        first, second = FakeAgent(0), FakeAgent(1, parents=(0,))
        registry.register(first, t=0.0)
        registry.register(second, t=10.0)
        state = CivilizationState(registry)
        state.build_facility("campfire", 400.0, 0, 0.0)
        state.contracts.append(
            SocialContract(
                contract_id="c-1",
                kind="apprentice",
                name_zh="师徒",
                proposed_by=0,
                created_at=0.0,
                participants=(0, 1),
            )
        )
        world = FakeWorld(t=0.0, agents=[first, second])
        state.update(world, 0.0)
        state.register_death(second, t=30.0, cause="starvation")
        world.t = 50.0
        world.agents = [first]
        state.update(world, 50.0)

        payload = json.loads(json.dumps(state.to_dict()))
        restored = CivilizationState.from_dict(payload)
        self.assertEqual(restored.registry.total_created, 2)
        self.assertEqual(restored.registry.total_deaths, 1)
        self.assertEqual(restored.registry.alive_ids(), [0])
        self.assertEqual(restored.registry.record(1).death_cause, "starvation")
        self.assertEqual(len(restored.facilities), 1)
        self.assertEqual(restored.facilities[0].name_zh, "篝火")
        self.assertEqual(len(restored.contracts), 1)
        self.assertEqual(restored.contracts[0].participants, (0, 1))
        self.assertEqual(len(restored.history), len(state.history))
        self.assertEqual(restored.collapse_state, state.collapse_state)
        self.assertEqual(restored.cooperation_index, state.cooperation_index)
        self.assertFalse(restored.extinct)
        self.assertEqual(restored.metrics(world), state.metrics(world))

    def test_summary_is_chinese(self) -> None:
        registry = make_registry()
        registry.register(FakeAgent(0), t=0.0)
        state = CivilizationState(registry)
        state.build_facility("campfire", 400.0, 0, 0.0)
        summary = state.summary_zh()
        for token in ("文明", "存活", "设施", "合作", "状态"):
            self.assertIn(token, summary)


if __name__ == "__main__":
    unittest.main()
