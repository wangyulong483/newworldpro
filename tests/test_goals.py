"""`newworld.goals`（P1 目标/计划/规划器）的单元测试。

覆盖：
* Condition 的各类条件求值，且“任何异常都返回 False”；
* Goal 的完成/失败/超时语义、progress_ge 语义；
* Plan / PlanStep 的推进与序列化；
* Planner 在**无知识库**时退化为内置基础链，在**有知识库**时走 recipe
  依赖图（识别中间产物 craft 步骤）；
* GoalManager 的需求驱动目标生成（能量/水分/库存/资源可用性）；
* 多步计划真实跑通（fake world 里驱动移动+采集+制作+建造）；
* 无进展 45 秒 → 重规划（plan.version 增长）；
* 连续重规划 ≥3 次 → 目标放弃 + 目标类型冷却 60 秒；
* 步骤 attempts 超限 → 计划失败；
* 未知 verb → 计划失败并记录；
* stats() 契约键、cost_estimate_error、to_dict/from_dict（JSON 往返）；
* 真实 `World(make_config(...))` 上的集成冒烟测试（不依赖 Lead 的集成改动）。

运行（项目根 `newworldpro`）::

    $env:TEMP=(Resolve-Path .tmp).Path
    python -m unittest tests.test_goals -v
"""

from __future__ import annotations

import json
import unittest

from newworld import World, make_config
from newworld.behavior import Decision
from newworld.goals import (
    CONDITION_KINDS,
    GOAL_STATUS,
    GOAL_TYPES,
    PLAN_STATUS,
    STEP_STATUS,
    VERBS,
    Condition,
    Goal,
    GoalManager,
    Plan,
    Planner,
    PlanStep,
)

STATS_KEYS = {
    "goals_created",
    "goals_completed",
    "goals_failed",
    "goals_abandoned",
    "plans_created",
    "replans",
    "steps_executed",
    "plan_success_rate",
    "cost_estimate_error",
}


# ---------------------------------------------------------------------------
# 最小 fake 世界（不依赖 Lead 尚未完成的集成改动）
# ---------------------------------------------------------------------------
class FakePhysiology:
    def __init__(self, energy: float = 80.0, hydration: float = 80.0,
                 sleep_pressure: float = 10.0) -> None:
        self.energy = energy
        self.hydration = hydration
        self.sleep_pressure = sleep_pressure
        self.sleeping = False

    def eat(self) -> None:
        self.energy = min(100.0, self.energy + 26.0)

    def drink(self) -> None:
        self.hydration = min(100.0, self.hydration + 30.0)

    def sleep(self) -> None:
        self.sleeping = True

    def wake(self) -> None:
        self.sleeping = False


class FakeInventory:
    def __init__(self, wood: int = 0, stone: int = 0, tool: str | None = None) -> None:
        self.wood = wood
        self.stone = stone
        self.tool = tool

    def add(self, kind: str, amount: int = 1) -> None:
        if kind == "wood":
            self.wood += amount
        elif kind == "stone":
            self.stone += amount


class FakeCommunity:
    def __init__(self, x_screen: float = 160.0) -> None:
        self.x_screen = x_screen
        self.wood = 0
        self.stone = 0
        self.structures = 0
        self.campfire = False

    @property
    def built(self) -> bool:
        return self.structures >= 1


class FakeTechnology:
    """与真实 Technology 同名的公开方法；resources 为材料 -> 屏幕 x。"""

    def __init__(self, resources: dict | None = None,
                 harvest_fails: bool = False) -> None:
        self.community = FakeCommunity()
        self.resources = (
            {"wood": 100.0, "stone": 300.0} if resources is None else dict(resources)
        )
        self.harvest_fails = harvest_fails
        self.harvest_calls = 0

    def nearest_resource(self, character, kind: str):
        x = self.resources.get(kind)
        if x is None:
            return None
        return (float(x), object())

    def harvest(self, character, kind: str) -> bool:
        if kind not in ("wood", "stone") or kind not in self.resources:
            return False
        if self.harvest_fails:
            return False
        character.inventory.add(kind, 1)
        self.harvest_calls += 1
        return True

    def near_community(self, character) -> bool:
        return abs(character.x - self.community.x_screen) < 90.0

    def craft(self, character) -> bool:
        inventory = character.inventory
        if inventory.tool or inventory.wood < 2 or inventory.stone < 1:
            return False
        inventory.wood -= 2
        inventory.stone -= 1
        inventory.tool = "axe"
        return True

    def contribute(self, character) -> bool:
        if not self.near_community(character):
            return False
        inventory = character.inventory
        if inventory.wood < 2 or inventory.stone < 1:
            return False
        inventory.wood -= 2
        inventory.stone -= 1
        self.community.wood += 2
        self.community.stone += 1
        if (
            self.community.wood >= 4
            and self.community.stone >= 2
            and self.community.structures < 1
        ):
            self.community.structures = 1
            self.community.campfire = True
        return True


class FakeAgent:
    def __init__(
        self,
        agent_id: int = 0,
        x: float = 200.0,
        energy: float = 80.0,
        hydration: float = 80.0,
        sleep_pressure: float = 10.0,
    ) -> None:
        self.agent_id = agent_id
        self.x = float(x)
        self.facing = 1
        self.y = 235.0
        self.inventory = FakeInventory()
        self.physiology = FakePhysiology(energy, hydration, sleep_pressure)
        self.behavior_state = "explore"
        self.alive = True
        self.age = 200.0
        self.sex = "male"
        self.on_ground = True
        self.parents: tuple[int, ...] = ()
        self.known_knowledge_ids: tuple[str, ...] = ()


class FakeWorld:
    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)
        self.config = make_config(320, 240)
        self.agents: list[FakeAgent] = []
        self.technology = FakeTechnology()
        self.pixel_scene = None
        self.knowledge = None
        self.civilization = None


class FakeMemory:
    def __init__(self) -> None:
        self.entries: list[tuple] = []

    def record(self, agent_id, t, kind, action, outcome, success,
               detail="", salience=0.5) -> None:
        self.entries.append((agent_id, t, kind, action, outcome, success, detail))


class FakeFacility:
    def __init__(self, facility_id: str = "campfire#1", kind: str = "campfire",
                 condition: float = 0.1) -> None:
        self.facility_id = facility_id
        self.kind = kind
        self.condition = float(condition)

    @property
    def active(self) -> bool:
        return self.condition > 0.2


class FakeCivilization:
    def __init__(self, facilities: list[FakeFacility] | None = None) -> None:
        self.facilities = list(facilities or [])
        self.maintained: list[tuple] = []

    def maintain(self, facility_id, agent_id, t, *, wood=0, stone=0) -> bool:
        self.maintained.append((facility_id, agent_id, t, wood, stone))
        for facility in self.facilities:
            if facility.facility_id == facility_id:
                facility.condition = 1.0
        return True


class FakeBerryScene:
    """只提供 harvest/eat/drink 需要的公开查询方法的假场景。"""

    def __init__(self, berries: list[tuple[int, int]], pond=(20, 30)) -> None:
        self.berries = list(berries)
        self.pond = tuple(pond)
        self.picked: list[tuple[int, int]] = []

    def pick_berry(self, berry) -> bool:
        if berry in self.berries:
            self.berries.remove(berry)
            self.picked.append(berry)
            return True
        return False

    def water_near(self, atom_x: int, atom_y: int, radius_atoms: float = 3.0) -> bool:
        left, right = self.pond
        return left - radius_atoms <= atom_x <= right + radius_atoms


class FakeRecipe:
    def __init__(self, inputs: dict, outputs: dict, requires_knowledge: tuple = (),
                 requires_tool: str | None = None) -> None:
        self.inputs = dict(inputs)
        self.outputs = dict(outputs)
        self.requires_knowledge = tuple(requires_knowledge)
        self.requires_tool = requires_tool
        self.work = 1
        self.difficulty = 0.2


class FakeKnowledge:
    def __init__(self, knowledge_id: str, recipe: FakeRecipe,
                 status: str = "verified") -> None:
        self.knowledge_id = knowledge_id
        self.name_zh = knowledge_id
        self.recipe = recipe
        self.status = status


class FakeKnowledgeStore:
    def __init__(self, items: list[FakeKnowledge]) -> None:
        self._items = {item.knowledge_id: item for item in items}
        self.attempts: list[str] = []
        self.candidate_worlds: list[object] = []

    def get(self, knowledge_id: str):
        return self._items.get(knowledge_id)

    def all(self, *, status: str | None = None) -> list[FakeKnowledge]:
        return [
            item for item in self._items.values()
            if status is None or item.status == status
        ]

    def candidates_for(self, agent, world=None) -> list[FakeKnowledge]:
        self.candidate_worlds.append(world)
        return [item for item in self._items.values() if item.status != "verified"]

    def attempt(self, agent, knowledge, world, t):
        self.attempts.append(knowledge.knowledge_id)
        knowledge.status = "verified"
        return True, "ok"


def drive(manager: GoalManager, world, agent, steps: int, dt: float = 0.1,
          speed: float = 30.0) -> int:
    """推进若干步，并把 Decision.move 应用为 agent 的位移（模拟 world.step）。"""
    moved = 0
    for _ in range(steps):
        world.t += dt
        decision = manager.update(world, agent, dt)
        if decision is None:
            continue
        moved += 1
        if decision.move:
            agent.x += decision.move * speed * dt
            agent.x = max(world.config.min_x, min(world.config.max_x, agent.x))
    return moved


# ---------------------------------------------------------------------------
# Condition
# ---------------------------------------------------------------------------
class TestCondition(unittest.TestCase):
    def test_inventory_and_tool(self) -> None:
        agent = FakeAgent()
        agent.inventory = FakeInventory(wood=3, stone=1, tool="axe")
        self.assertTrue(Condition("inventory_ge", "wood", value=3).evaluate(None, agent))
        self.assertFalse(Condition("inventory_ge", "wood", value=4).evaluate(None, agent))
        self.assertTrue(Condition("has_tool", "axe").evaluate(None, agent))
        self.assertTrue(Condition("has_tool").evaluate(None, agent))
        self.assertFalse(Condition("has_tool", "basket").evaluate(None, agent))
        agent.inventory.tool = None
        self.assertFalse(Condition("has_tool").evaluate(None, agent))

    def test_energy_hydration_and_always(self) -> None:
        agent = FakeAgent(energy=52.0, hydration=30.0)
        self.assertTrue(Condition("energy_ge", value=50).evaluate(None, agent))
        self.assertFalse(Condition("energy_ge", value=60).evaluate(None, agent))
        self.assertFalse(Condition("hydration_ge", value=40).evaluate(None, agent))
        self.assertTrue(Condition("always").evaluate(None, agent))

    def test_condition_never_raises(self) -> None:
        # 世界/agent 完全为 None 或损坏时，条件一律 False（不抛错）
        for kind in CONDITION_KINDS:
            condition = Condition(kind, "wood", value=1.0)
            self.assertFalse(condition.evaluate(None, None) and kind != "always")

        class BoomWorld:
            @property
            def technology(self):
                raise RuntimeError("boom")

            @property
            def civilization(self):
                raise RuntimeError("boom")

            @property
            def agents(self):
                raise RuntimeError("boom")

            @property
            def knowledge(self):
                raise RuntimeError("boom")

        agent = FakeAgent()
        for kind in CONDITION_KINDS:
            result = Condition(kind, "campfire", value=1.0).evaluate(BoomWorld(), agent)
            self.assertIsInstance(result, bool)
        self.assertFalse(Condition("energy_ge", value=1.0).evaluate(BoomWorld(), None))

    def test_community_and_structure_conditions(self) -> None:
        world = FakeWorld()
        agent = FakeAgent()
        self.assertTrue(Condition("structure_count_ge", value=0).evaluate(world, agent))
        self.assertFalse(Condition("structure_count_ge", value=1).evaluate(world, agent))
        world.technology.community.structures = 1
        self.assertTrue(Condition("structure_count_ge", value=1).evaluate(world, agent))

    def test_knowledge_verified_condition(self) -> None:
        world = FakeWorld()
        agent = FakeAgent()
        recipe = FakeRecipe({"wood": 1}, {"tool": 1})
        world.knowledge = FakeKnowledgeStore(
            [FakeKnowledge("stone_axe", recipe, status="verified"),
             FakeKnowledge("hut", recipe, status="hypothesis")]
        )
        self.assertTrue(
            Condition("knowledge_verified", "stone_axe").evaluate(world, agent)
        )
        self.assertFalse(Condition("knowledge_verified", "hut").evaluate(world, agent))
        # 没有 world.knowledge 时回退到 agent.known_knowledge_ids
        world.knowledge = None
        agent.known_knowledge_ids = ("basket",)
        self.assertTrue(Condition("knowledge_verified", "basket").evaluate(world, agent))
        self.assertFalse(Condition("knowledge_verified", "well").evaluate(world, agent))

    def test_near_agent_and_at_site(self) -> None:
        world = FakeWorld()
        agent = FakeAgent(agent_id=0, x=100.0)
        other = FakeAgent(agent_id=1, x=130.0)
        world.agents = [agent, other]
        self.assertTrue(Condition("near_agent", value=60.0).evaluate(world, agent))
        self.assertFalse(Condition("near_agent", value=10.0).evaluate(world, agent))
        self.assertTrue(Condition("at_site", "community").evaluate(world, agent))
        agent.x = 10.0
        self.assertFalse(Condition("at_site", "community").evaluate(world, agent))


# ---------------------------------------------------------------------------
# Goal / Plan
# ---------------------------------------------------------------------------
class TestGoal(unittest.TestCase):
    def test_is_done_and_failed(self) -> None:
        agent = FakeAgent(energy=70.0)
        goal = Goal(
            goal_id=1, agent_id=0, goal_type="survive",
            completion=(Condition("energy_ge", value=60.0),),
        )
        self.assertTrue(goal.is_done(None, agent))
        agent.physiology.energy = 50.0
        self.assertFalse(goal.is_done(None, agent))

        goal.failure = (Condition("inventory_ge", "wood", value=5.0),)
        self.assertFalse(goal.is_failed(None, agent))
        agent.inventory.wood = 5
        self.assertTrue(goal.is_failed(None, agent))

        # 超时也算失败
        timeout_goal = Goal(goal_id=2, agent_id=0, goal_type="explore", max_steps=10)
        timeout_goal.steps_spent = 10
        self.assertTrue(timeout_goal.is_failed(None, agent))

    def test_progress_condition_uses_goal_progress(self) -> None:
        agent = FakeAgent()
        goal = Goal(
            goal_id=1, agent_id=0, goal_type="explore",
            completion=(Condition("progress_ge", value=1.0),),
        )
        goal.progress = 0.5
        self.assertFalse(goal.is_done(None, agent))
        goal.progress = 1.0
        self.assertTrue(goal.is_done(None, agent))
        # 空 completion 不会被真空满足（只能靠 max_steps 超时终结）
        empty = Goal(goal_id=2, agent_id=0, goal_type="gather")
        self.assertFalse(empty.is_done(None, agent))

    def test_plan_advance_and_serialization(self) -> None:
        plan = Plan(
            plan_id=7, agent_id=0, goal_id=3,
            steps=[
                PlanStep(index=0, verb="gather", args={"kind": "wood", "target": 2.0}),
                PlanStep(index=1, verb="craft", args={"recipe": "stone_axe"}),
            ],
            expected_cost=3.0,
        )
        self.assertEqual(plan.current_step().verb, "gather")
        plan.advance()
        self.assertEqual(plan.current_step().verb, "craft")
        self.assertEqual(plan.actual_steps, 1)
        plan.advance()
        self.assertIsNone(plan.current_step())
        restored = Plan.from_dict(plan.to_dict())
        self.assertEqual(restored.current, 2)
        self.assertEqual(restored.steps[1].args["recipe"], "stone_axe")
        goal = Goal(goal_id=3, agent_id=0, goal_type="gather",
                    params={"kind": "wood", "target": 2.0})
        self.assertEqual(Goal.from_dict(goal.to_dict()).params["kind"], "wood")


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class TestPlanner(unittest.TestCase):
    def test_basic_chain_without_knowledge(self) -> None:
        world = FakeWorld()
        agent = FakeAgent()
        planner = Planner()
        goal = Goal(goal_id=1, agent_id=0, goal_type="build")
        plan = planner.plan(world, agent, goal)
        self.assertIsNotNone(plan)
        verbs = [step.verb for step in plan.steps]
        self.assertIn("gather", verbs)
        self.assertIn("move_to", verbs)
        self.assertIn("build", verbs)
        self.assertTrue(all(verb in VERBS for verb in verbs))
        self.assertGreater(plan.expected_cost, 0.0)
        self.assertEqual([step.index for step in plan.steps], list(range(len(plan.steps))))

    def test_craft_goal_without_knowledge_uses_basic_recipe(self) -> None:
        world = FakeWorld()
        agent = FakeAgent()
        goal = Goal(goal_id=1, agent_id=0, goal_type="craft", recipe_id="stone_axe")
        plan = Planner().plan(world, agent, goal)
        self.assertIsNotNone(plan)
        verbs = [step.verb for step in plan.steps]
        self.assertEqual(verbs.count("craft"), 1)
        self.assertEqual(verbs.count("gather"), 2)  # wood + stone
        self.assertEqual(verbs[-1], "craft")

    def test_recipe_graph_uses_knowledge_dependencies(self) -> None:
        cord = FakeKnowledge("fiber_cord", FakeRecipe({"wood": 2}, {"cord": 1}))
        hut = FakeKnowledge(
            "hut",
            FakeRecipe({"wood": 2, "stone": 2, "cord": 1}, {"hut": 1},
                       requires_knowledge=("fiber_cord",)),
        )
        store = FakeKnowledgeStore([cord, hut])
        world = FakeWorld()
        agent = FakeAgent()
        goal = Goal(goal_id=1, agent_id=0, goal_type="craft", recipe_id="hut")

        planned = Planner(knowledge=store).plan(world, agent, goal)
        self.assertIsNotNone(planned)
        recipes = [step.args.get("recipe") for step in planned.steps]
        self.assertIn("fiber_cord", recipes)
        self.assertIn("hut", recipes)
        # 中间产物（cord）必须先于最终产物（hut）被制作
        self.assertLess(recipes.index("fiber_cord"), recipes.index("hut"))
        # 知识计划来自已验证 recipe（带 knowledge_id），最后一步是 hut
        self.assertEqual(planned.steps[-1].args.get("knowledge_id"), "hut")

        fallback = Planner().plan(world, agent, goal)
        self.assertIsNotNone(fallback)
        # 无知识库时用内置配方表，同样能解析出中间产物 cord（P1 降级链）
        fallback_recipes = [s.args.get("recipe") for s in fallback.steps]
        self.assertIn("fiber_cord", fallback_recipes)
        self.assertIn("hut", fallback_recipes)
        self.assertIsNone(fallback.steps[-1].args.get("knowledge_id"))
        # 两者按各自的配方输入采集：知识库 hut 需要 wood 2，内置表是 wood 4
        planned_gather = {
            step.args["kind"]: step.args["target"]
            for step in planned.steps if step.verb == "gather"
        }
        fallback_gather = {
            step.args["kind"]: step.args["target"]
            for step in fallback.steps if step.verb == "gather"
        }
        self.assertEqual(planned_gather.get("wood"), 2.0)
        self.assertEqual(fallback_gather.get("wood"), 4.0)

    def test_plan_returns_none_for_unsupported_goal(self) -> None:
        world = FakeWorld()
        agent = FakeAgent()
        goal = Goal(goal_id=1, agent_id=0, goal_type="experiment",
                    recipe_id="stone_axe")
        self.assertIsNone(Planner().plan(world, agent, goal))
        unknown = Goal(goal_id=2, agent_id=0, goal_type="telepathy")
        self.assertIsNone(Planner().plan(world, agent, unknown))

    def test_replan_increments_version(self) -> None:
        world = FakeWorld()
        agent = FakeAgent()
        planner = Planner()
        goal = Goal(goal_id=1, agent_id=0, goal_type="gather",
                    params={"kind": "wood", "target": 2.0})
        first = planner.plan(world, agent, goal)
        second = planner.replan(world, agent, goal, first, "no_progress")
        self.assertIsNotNone(second)
        self.assertEqual(second.version, first.version + 1)


# ---------------------------------------------------------------------------
# GoalManager：生成 / 完成 / 反卡死 / 序列化
# ---------------------------------------------------------------------------
class TestGoalManagerGeneration(unittest.TestCase):
    def test_returns_none_without_technology(self) -> None:
        world = FakeWorld()
        world.technology = None
        agent = FakeAgent(energy=10.0)
        manager = GoalManager(seed=0)
        self.assertIsNone(manager.update(world, agent, 0.1))
        self.assertEqual(manager.goals_of(0), [])

    def test_disabled_planning_returns_none(self) -> None:
        class DisabledConfig:
            enable_planning = False

        world = FakeWorld()
        agent = FakeAgent(energy=10.0)
        manager = GoalManager(seed=0, config=DisabledConfig())
        self.assertIsNone(manager.update(world, agent, 0.1))
        self.assertEqual(manager.goals_of(0), [])

    def test_low_energy_generates_survive_goal(self) -> None:
        world = FakeWorld()
        agent = FakeAgent(energy=15.0)
        world.agents = [agent]
        manager = GoalManager(seed=0)
        manager.update(world, agent, 0.1)
        goals = manager.goals_of(0)
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0].goal_type, "survive")
        self.assertEqual(goals[0].params.get("need"), "eat")
        self.assertEqual(goals[0].source, "need")
        self.assertGreater(goals[0].priority, 1.0)

    def test_low_hydration_generates_drink_goal(self) -> None:
        world = FakeWorld()
        agent = FakeAgent(energy=80.0, hydration=20.0)
        world.agents = [agent]
        manager = GoalManager(seed=0)
        manager.update(world, agent, 0.1)
        goals = manager.goals_of(0)
        self.assertEqual(goals[0].goal_type, "survive")
        self.assertEqual(goals[0].params.get("need"), "drink")

    def test_inventory_state_generates_gather_then_build(self) -> None:
        world = FakeWorld()
        agent = FakeAgent()
        world.agents = [agent]
        manager = GoalManager(seed=0)
        manager.update(world, agent, 0.1)
        self.assertEqual(manager.goals_of(0)[0].goal_type, "gather")

        # 库存足够后，下一次生成的目标应是 build（不是写死的回合脚本）
        world2 = FakeWorld()
        agent2 = FakeAgent()
        agent2.inventory = FakeInventory(wood=2, stone=1, tool="axe")
        world2.agents = [agent2]
        manager2 = GoalManager(seed=0)
        manager2.update(world2, agent2, 0.1)
        self.assertEqual(manager2.goals_of(0)[0].goal_type, "build")

    def test_no_available_resource_falls_back_to_explore(self) -> None:
        world = FakeWorld()
        world.technology.resources = {}  # 世界没有可采材料
        agent = FakeAgent()
        world.agents = [agent]
        manager = GoalManager(seed=0)
        manager.update(world, agent, 0.1)
        self.assertEqual(manager.goals_of(0)[0].goal_type, "explore")

    def test_memory_receives_goal_and_plan_entries(self) -> None:
        world = FakeWorld()
        agent = FakeAgent()
        world.agents = [agent]
        memory = FakeMemory()
        manager = GoalManager(seed=0, memory=memory)
        manager.update(world, agent, 0.1)
        actions = [entry[3] for entry in memory.entries]
        self.assertTrue(any(action.startswith("goal:") for action in actions))
        self.assertTrue(any(action.startswith("plan:") for action in actions))


class TestGoalManagerExecution(unittest.TestCase):
    def test_multi_step_goal_completes(self) -> None:
        """真实驱动 fake 世界：移动→采集→制作→贡献→建造 全部跑通。"""
        world = FakeWorld()
        agent = FakeAgent(x=200.0)
        world.agents = [agent]
        manager = GoalManager(seed=0)
        drive(manager, world, agent, 900, dt=0.1)
        stats = manager.stats()
        self.assertGreaterEqual(stats["goals_completed"], 1.0)
        self.assertGreaterEqual(stats["steps_executed"], 1.0)
        # 至少完成过一次采集（库存/背包变化是世界规则改的，不是 manager 直接改的）
        self.assertGreaterEqual(world.technology.harvest_calls, 1)

    def test_unknown_verb_fails_plan_and_replans(self) -> None:
        world = FakeWorld()
        agent = FakeAgent()
        world.agents = [agent]
        manager = GoalManager(seed=0)
        manager.update(world, agent, 0.1)
        plan = manager.active_plan(0)
        self.assertIsNotNone(plan)
        plan.steps = [PlanStep(index=0, verb="teleport")]
        plan.current = 0
        for _ in range(10):
            world.t += 0.1
            manager.update(world, agent, 0.1)
        self.assertGreaterEqual(manager.stats()["replans"], 1.0)
        self.assertTrue("unknown_verb" in plan.steps[0].last_error)
        self.assertEqual(plan.status, "failed")

    def test_no_progress_triggers_replan(self) -> None:
        """agent 不动、状态不变 → 45 世界秒后 replan（plan.version 增长）。"""
        world = FakeWorld()
        agent = FakeAgent(x=10.0)
        agent.inventory = FakeInventory(wood=2, stone=1, tool="axe")
        world.agents = [agent]
        manager = GoalManager(seed=0)
        manager.update(world, agent, 1.0)
        first = manager.active_plan(0)
        self.assertIsNotNone(first)
        for _ in range(50):
            world.t += 1.0
            manager.update(world, agent, 1.0)
        self.assertGreaterEqual(manager.stats()["replans"], 1.0)
        current = manager.active_plan(0)
        self.assertIsNotNone(current)
        self.assertGreater(current.version, first.version)

    def test_consecutive_replans_abandon_and_cool_down(self) -> None:
        """采集永远失败 → 连续 replan ≥3 → abandoned + 目标类型冷却 60 秒。"""
        world = FakeWorld()
        # 世界里有木头资源（所以会生成采集目标），但每次采集都失败
        world.technology = FakeTechnology(resources={"wood": 100.0}, harvest_fails=True)
        agent = FakeAgent(x=100.0)
        world.agents = [agent]
        manager = GoalManager(seed=0)
        for _ in range(40):
            world.t += 0.1
            manager.update(world, agent, 0.1)
        gather_before = [g for g in manager.goals_of(0) if g.goal_type == "gather"]
        self.assertTrue(any(goal.status == "abandoned" for goal in gather_before))
        self.assertGreaterEqual(manager.stats()["goals_abandoned"], 1.0)
        self.assertGreaterEqual(manager.stats()["replans"], 3.0)

        count_after_abandon = len(gather_before)
        # 冷却期内不得再生成 gather 目标
        for _ in range(20):
            world.t += 0.1
            manager.update(world, agent, 0.1)
        self.assertEqual(
            len([g for g in manager.goals_of(0) if g.goal_type == "gather"]),
            count_after_abandon,
        )
        # 冷却结束后（资源仍不可用）允许重新尝试 gather
        world.t += 70.0
        for _ in range(30):
            world.t += 1.0
            manager.update(world, agent, 1.0)
        self.assertGreater(
            len([g for g in manager.goals_of(0) if g.goal_type == "gather"]),
            count_after_abandon,
        )

    def test_stats_keys_and_summary(self) -> None:
        world = FakeWorld()
        agent = FakeAgent(energy=15.0)
        world.agents = [agent]
        manager = GoalManager(seed=0)
        manager.update(world, agent, 0.1)
        stats = manager.stats()
        self.assertEqual(set(stats), STATS_KEYS)
        self.assertTrue(all(isinstance(value, float) for value in stats.values()))
        self.assertGreaterEqual(stats["goals_created"], 1.0)
        self.assertGreaterEqual(stats["plans_created"], 1.0)
        summary = manager.summary_zh()
        self.assertIn("目标", summary)
        self.assertIn("计划", summary)

    def test_to_dict_from_dict_roundtrip(self) -> None:
        world = FakeWorld()
        agent = FakeAgent()
        world.agents = [agent]
        manager = GoalManager(seed=3)
        drive(manager, world, agent, 40, dt=0.1)
        data = manager.to_dict()
        text = json.dumps(data, ensure_ascii=False)  # 必须可 JSON 序列化
        restored = GoalManager.from_dict(json.loads(text))
        self.assertEqual(set(restored.stats()), set(manager.stats()))
        self.assertEqual(
            [goal.goal_type for goal in restored.goals_of(0)],
            [goal.goal_type for goal in manager.goals_of(0)],
        )
        self.assertEqual(restored.active_plan(0) is None, manager.active_plan(0) is None)
        # 恢复后还能继续推进
        for _ in range(10):
            world.t += 0.1
            restored.update(world, agent, 0.1)

    def test_cost_estimate_error_is_measured(self) -> None:
        world = FakeWorld()
        agent = FakeAgent()
        world.agents = [agent]
        manager = GoalManager(seed=0)
        drive(manager, world, agent, 600, dt=0.1)
        stats = manager.stats()
        self.assertGreaterEqual(stats["cost_estimate_error"], 0.0)
        self.assertGreaterEqual(stats["plan_success_rate"], 0.0)
        self.assertLessEqual(stats["plan_success_rate"], 1.0)
        self.assertGreaterEqual(stats["steps_executed"], 1.0)


class TestSurviveAndSocialVerbs(unittest.TestCase):
    """执行器 verb 级覆盖：eat / drink / rest / maintain / socialize / reproduce / experiment。"""

    def test_eat_goal_completes_on_berry_scene(self) -> None:
        world = FakeWorld()
        agent = FakeAgent(energy=30.0, x=100.0)
        world.agents = [agent]
        world.pixel_scene = FakeBerryScene([(20, 40), (20, 40), (20, 40)])
        manager = GoalManager(seed=0)
        for _ in range(20):
            world.t += 0.1
            manager.update(world, agent, 0.1)
        goals = manager.goals_of(0)
        self.assertEqual(goals[0].goal_type, "survive")
        self.assertEqual(goals[0].params.get("need"), "eat")
        self.assertEqual(goals[0].status, "completed")
        self.assertGreaterEqual(agent.physiology.energy, 62.0)
        self.assertGreaterEqual(len(world.pixel_scene.picked), 1)

    def test_drink_goal_completes_on_pond_scene(self) -> None:
        world = FakeWorld()
        agent = FakeAgent(hydration=20.0, x=125.0)
        world.agents = [agent]
        world.pixel_scene = FakeBerryScene([], pond=(20, 30))
        manager = GoalManager(seed=0)
        for _ in range(20):
            world.t += 0.1
            manager.update(world, agent, 0.1)
        goals = manager.goals_of(0)
        self.assertEqual(goals[0].params.get("need"), "drink")
        self.assertEqual(goals[0].status, "completed")
        self.assertGreaterEqual(agent.physiology.hydration, 66.0)

    def test_rest_goal_completes_via_progress(self) -> None:
        world = FakeWorld()
        agent = FakeAgent(sleep_pressure=70.0)
        world.agents = [agent]
        manager = GoalManager(seed=0)
        for _ in range(20):
            world.t += 0.5
            manager.update(world, agent, 0.5)
        goals = manager.goals_of(0)
        self.assertEqual(goals[0].params.get("need"), "rest")
        self.assertEqual(goals[0].status, "completed")

    def test_maintain_goal_revives_inactive_facility(self) -> None:
        world = FakeWorld()
        world.technology.community.structures = 1  # 营地已建，避免生成采集目标
        agent = FakeAgent(x=160.0)
        world.agents = [agent]
        facility = FakeFacility(condition=0.1)
        world.civilization = FakeCivilization([facility])
        manager = GoalManager(seed=0)
        for _ in range(10):
            world.t += 0.1
            manager.update(world, agent, 0.1)
        goals = manager.goals_of(0)
        self.assertEqual(goals[0].goal_type, "maintain")
        self.assertEqual(goals[0].status, "completed")
        self.assertTrue(world.civilization.maintained)
        self.assertEqual(facility.condition, 1.0)

    def test_socialize_goal_completes_when_near_peer(self) -> None:
        world = FakeWorld()
        world.technology.community.structures = 1
        agent = FakeAgent(agent_id=0, x=100.0)
        # 距离 120px：在“可社交”范围内（160），但还没满足“靠近”（55）
        other = FakeAgent(agent_id=1, x=220.0)
        world.agents = [agent, other]
        manager = GoalManager(seed=0)
        drive(manager, world, agent, 80, dt=0.1)
        goals = manager.goals_of(0)
        self.assertEqual(goals[0].goal_type, "socialize")
        self.assertEqual(goals[0].status, "completed")

    def test_reproduce_goal_uses_world_hook(self) -> None:
        world = FakeWorld()
        world.technology.community.structures = 1
        agent = FakeAgent(agent_id=0, x=100.0)
        agent.sex = "male"
        agent.age = 200.0
        other = FakeAgent(agent_id=1, x=200.0)
        other.sex = "female"
        other.age = 200.0
        world.agents = [agent, other]
        world.try_reproduce = lambda character: True
        manager = GoalManager(seed=0)
        for _ in range(10):
            world.t += 0.1
            manager.update(world, agent, 0.1)
        goals = manager.goals_of(0)
        self.assertEqual(goals[0].goal_type, "reproduce")
        self.assertEqual(goals[0].status, "completed")

    def test_experiment_goal_uses_knowledge_store(self) -> None:
        world = FakeWorld()
        world.technology.community.structures = 1
        agent = FakeAgent()
        world.agents = [agent]
        hypothesis = FakeKnowledge(
            "fiber_cord", FakeRecipe({"wood": 3}, {"cord": 1}), status="hypothesis"
        )
        store = FakeKnowledgeStore([hypothesis])
        world.knowledge = store
        manager = GoalManager(seed=0, knowledge=store)
        # 规划器会先补采实验所需材料（wood 3），所以需要驱动位移+采集
        drive(manager, world, agent, 120, dt=0.1)
        goals = manager.goals_of(0)
        self.assertEqual(goals[0].goal_type, "experiment")
        self.assertEqual(goals[0].recipe_id, "fiber_cord")
        self.assertEqual(goals[0].status, "completed")
        self.assertEqual(store.attempts, ["fiber_cord"])
        # 知识候选查询必须拿到 world（P2 的 can_afford 依赖它）
        self.assertTrue(store.candidate_worlds)
        self.assertEqual(hypothesis.status, "verified")


# ---------------------------------------------------------------------------
# 真实世界冒烟（不依赖 Lead 的集成改动）
# ---------------------------------------------------------------------------
class TestRealWorldSmoke(unittest.TestCase):
    def test_updates_against_real_world(self) -> None:
        world = World(make_config(800, 500), seed=0)
        agent = world.agent
        manager = GoalManager(seed=0)
        decisions = 0
        for _ in range(300):
            decision = manager.update(world, agent, 1 / 60)
            self.assertTrue(decision is None or isinstance(decision, Decision))
            if decision is not None:
                decisions += 1
            world.step(1 / 60, decision if decision is not None else 0)
        self.assertGreater(world.t, 0.0)
        self.assertGreater(decisions, 0)
        self.assertGreaterEqual(manager.stats()["goals_created"], 1.0)
        self.assertGreater(agent.distance_walked, 0.0)
        for value in manager.stats().values():
            self.assertTrue(value == value)  # 无 NaN
        self.assertIsInstance(manager.summary_zh(), str)

    def test_contract_constants_are_consistent(self) -> None:
        self.assertEqual(len(GOAL_TYPES), 9)
        self.assertIn("survive", GOAL_TYPES)
        self.assertIn("reproduce", GOAL_TYPES)
        self.assertEqual(set(GOAL_STATUS), {"active", "completed", "failed", "abandoned"})
        self.assertEqual(
            set(PLAN_STATUS),
            {"active", "completed", "failed", "replanned", "abandoned"},
        )
        self.assertEqual(set(STEP_STATUS), {"pending", "running", "done", "failed"})
        self.assertIn("progress_ge", CONDITION_KINDS)
        self.assertIn("deliver", VERBS)
        self.assertEqual(len(VERBS), 15)


if __name__ == "__main__":
    unittest.main()
