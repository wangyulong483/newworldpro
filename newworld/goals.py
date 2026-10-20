"""目标 / 计划 / 规划器（P1）——行为层之上的“任务层”。

设计意图
========
`behavior.py` 提供的是*反应式*自由行为（饿→觅食，渴→喝水），`social.py`
提供的是手写的社会/技术教师。两者都只能一次决定一个动作，无法表达
“先攒 4 根木头 + 2 块石头，再搬到公共营地，最后建起结构”这种多步任务。
本模块补上缺失的一层：**把当前状态翻译成结构化目标，再把目标翻译成
多步计划，最后把计划步骤执行成 `behavior.Decision`**。

三层结构
--------
1. :class:`GoalManager`：目标生命周期（生成 → 选择 → 推进 → 完成/失败/放弃），
   以及“反卡死”机制：无进展 → 重规划；同一目标连续重规划 3 次 → 放弃 +
   该个体该目标类型冷却 60 世界秒（按个体冷却，避免一人放弃冻结所有人）。
2. :class:`Planner`：把目标翻译成 :class:`Plan`。**有知识库时**用 recipe
   依赖图做代价搜索（缺什么材料就递归找已验证的产出配方）；**没有知识库时**
   退化为内置基础动作链（gather → craft → build），因此 P1 可以在
   `knowledge.py` 尚未落地时独立验证。
3. 执行器（:meth:`GoalManager._run_verb`）：把结构化 verb 映射为真实动作。

公开接口（冻结契约见 ``docs/IMPLEMENTATION_REPORT.md`` §4）
-------------------------------------------------------
``Condition / Goal / PlanStep / Plan / Planner / GoalManager`` 的字段与签名
必须保持兼容；本模块额外提供 ``Goal.to_dict/from_dict`` 等序列化小工具、
``BASIC_RECIPES`` 内置配方表（无知识库时的降级来源），以及
``Goal.params``（可选扩展字段，用于携带 kind/target/need/facility_id 等
生成期参数；缺省为空字典，不破坏契约构造方式）。

世界规则唯一权威
----------------
本模块**不直接写** scene 网格、库存数值或知识状态：所有状态变化都通过
``world.technology`` 的公开方法（``nearest_resource / harvest / craft /
contribute / near_community / can_afford`` 等）请求。唯一的补充是
“吃东西”的物理入口：当前 `Technology` 没有 ``eat`` 方法，而
`behavior.py` 的现有权限做法是 ``scene.pick_berry`` + ``physiology.eat``，
这里保持同一条路径（``scene.pick_berry`` 是场景的公开方法，不是直接改
网格），并在下方“集成钩子”里给出建议的正式方法签名。

与集成层（Lead）的钩子清单
--------------------------
已落地并直接使用（P1 必需）：
* ``world.technology.nearest_resource(character, kind) -> (x, obj) | None``
* ``world.technology.harvest(character, kind) -> bool``（含 fiber，库存满了返回 False）
* ``world.technology.craft(character) -> bool``（旧 stone_axe 链）
* ``world.technology.contribute(character) -> bool``
* ``world.technology.near_community(character) -> bool``
* ``world.technology.community.x_screen / wood / stone / structures / built``
* ``world.technology.execute_recipe(character, recipe) -> (ok, reason)``
  （已验证配方的唯一权威执行入口；本模块只做请求）
* ``world.technology.repair(character, facility) -> bool``（维护公共设施）
* ``world.technology.recipe_blocker(character, recipe)``（工具/材料前置检查；
  公共设施可在个体所在位置建造，因此不再有 too_far_from_site 前置条件）
* ``world.knowledge``（``KnowledgeStore``）：``get / all / candidates_for / attempt``
* ``world.civilization``：``facilities``（含 ``facility_id / kind / condition /
  active / maintenance_wood / maintenance_stone``）
* ``agent.current_goal_id``（Character 字段）与 ``agent.goal_progress``（本模块写入的镜像）

仍缺、本模块已在缺失时优雅降级（返回 None 或该步骤失败）的钩子：
* ``world.technology.eat(agent) -> bool``：缺失时走 ``scene.pick_berry`` +
  ``physiology.eat``（与 behavior.py 相同的既有权限路径）
* ``world.technology.drink(agent) -> bool``：缺失时走 ``physiology.drink``
* ``world.try_reproduce(agent) -> bool``：缺失时不生成 reproduce 目标（P3）
* ``world.technology.craft_recipe(agent, recipe_id) -> bool``：仅在
  ``world.knowledge`` 不可用而仍需制造非 stone_axe 配方时使用
* ``world.technology.maintain(agent) -> bool``：缺失时用 ``repair``；
  ``CivilizationState.maintain`` 的契约是“调用方已扣材料”，本模块不直接写库存，
  因此那条兜底只用于无材料需求的设施。

集成注意
--------
``update()`` 返回 ``None`` 表示“本帧没把握”，集成层必须回退到神经/社会教师，
**不能把 None 直接传给 ``world.step``**（world.step 只接受 int 或带 move/jump
的动作对象）。

反固定剧情
----------
目标生成完全由「生理需求 + 库存 + 环境资源 + 知识候选 + 设施损耗 + 种群
名额/伴侣 + 无目标时探索」驱动；没有“第 N 回合出现 X”的脚本。允许目标
失败、重规划、放弃，也允许一直探索；没把握时 ``update()`` 返回 ``None``，
由集成层回退到神经/社会教师。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .behavior import Decision
from .materials import ATOM_SIZE

# ---------------------------------------------------------------------------
# 枚举常量（冻结契约 §4）
# ---------------------------------------------------------------------------
GOAL_TYPES = (
    "survive", "gather", "craft", "build", "experiment",
    "explore", "socialize", "reproduce", "maintain",
)
GOAL_STATUS = ("active", "completed", "failed", "abandoned")
PLAN_STATUS = ("active", "completed", "failed", "replanned", "abandoned")
STEP_STATUS = ("pending", "running", "done", "failed")

CONDITION_KINDS = (
    "inventory_ge", "has_tool", "knowledge_verified", "structure_count_ge",
    "facility_active", "energy_ge", "hydration_ge", "at_site", "near_agent",
    "progress_ge", "always",
)

#: 规划器可生成的结构化步骤（执行器必须实现，未知 verb → 计划失败并记录）
VERBS = (
    "move_to", "gather", "craft", "experiment", "contribute", "build",
    "maintain", "drink", "eat", "rest", "explore", "observe",
    "deliver", "socialize", "reproduce",
)

# ---------------------------------------------------------------------------
# 反卡死 / 需求阈值常量
# ---------------------------------------------------------------------------
#: 无进展容忍时间（世界秒）与步数上限（§4 反卡死要求）
NO_PROGRESS_S = 45.0
NO_PROGRESS_STEPS = 240
#: 同一目标连续重规划上限，达到后放弃
MAX_REPLANS = 3
#: 放弃/失败后的目标类型冷却（世界秒，按个体计）
GOAL_COOLDOWN_S = 60.0

SURVIVE_ENERGY_LOW = 38.0
SURVIVE_ENERGY_TARGET = 62.0
SURVIVE_HYDRATION_LOW = 40.0
SURVIVE_HYDRATION_TARGET = 66.0
SLEEP_PRESSURE_HIGH = 62.0

#: 公共营地最小材料（与 technology.CommunitySite.need 一致）
COMMUNITY_WOOD_NEED = 4
COMMUNITY_STONE_NEED = 2

#: 规划器允许的 recipe 递归深度
MAX_RECIPE_DEPTH = 6

#: 产出公共设施的配方产物（与 technology.FACILITY_OUTPUTS 一致）。
#: 保留为常量以便规划器/调试判断“这条配方是否会造出设施”；
#: 当前 recipe_blocker 不再要求靠近营地（设施可在个体位置建造）。
FACILITY_RECIPE_OUTPUTS = frozenset({"campfire", "hut", "well", "workshop", "granary"})

_EPS = 1e-6

#: 无知识库时的内置配方表（世界规则的最小集合，作为 P1 降级来源）。
#: 与 ``knowledge.RECIPE_CATALOG`` 的最小集合保持一致：inputs = 材料 -> 数量，
#: outputs = 产物 -> 数量（用于递归解析中间产物，例如 cord → basket）。
#: 注意：它只用于**生成计划文本**；真正执行仍必须经过 world.technology。
BASIC_RECIPES: dict[str, dict[str, object]] = {
    "stone_axe": {
        "name_zh": "石斧",
        "inputs": {"wood": 2, "stone": 1},
        "outputs": {"axe": 1},
    },
    "fiber_cord": {
        "name_zh": "纤维绳",
        "inputs": {"fiber": 3},
        "outputs": {"cord": 1},
    },
    "basket": {
        "name_zh": "背篓",
        "inputs": {"wood": 2, "cord": 1},
        "outputs": {"basket": 1},
    },
    "campfire": {
        "name_zh": "篝火",
        "inputs": {"wood": 2, "stone": 1},
        "outputs": {"campfire": 1},
    },
    "hut": {
        "name_zh": "小屋",
        "inputs": {"wood": 4, "stone": 2, "cord": 1},
        "outputs": {"hut": 1},
    },
    "well": {
        "name_zh": "水井",
        "inputs": {"stone": 3, "cord": 1},
        "outputs": {"well": 1},
    },
}


# ---------------------------------------------------------------------------
# 小工具：全部是“读世界/读 agent”的防御式访问，缺失字段不抛错
# ---------------------------------------------------------------------------
def _get(obj, name: str, default=None):
    """``getattr`` 的容错版本：任何异常都返回 default。"""
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _arg(args, key: str, default=None):
    """读步骤参数：``args`` 是字典（PlanStep.args），必须按键读而不是 getattr。"""
    if isinstance(args, dict):
        try:
            return args.get(key, default)
        except Exception:
            return default
    return _get(args, key, default)


def _num(value, default: float = 0.0) -> float:
    """把任意值安全转成有限浮点数。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _compare(actual: float, op: str, expected: float) -> bool:
    """按 ``op`` 比较两个数；未知 op 一律 False（不抛错）。"""
    if op == ">=":
        return actual >= expected
    if op == "<=":
        return actual <= expected
    if op == ">":
        return actual > expected
    if op == "<":
        return actual < expected
    if op == "==":
        return abs(actual - expected) <= _EPS
    if op == "!=":
        return abs(actual - expected) > _EPS
    return False


def _inventory(agent):
    """读 agent 的库存（duck-typed）。"""
    return _get(agent, "inventory", None)


def _inv_get(agent, kind: str) -> float:
    """读库存数量：同时兼容 ``Inventory`` 对象与字典。"""
    inventory = _inventory(agent)
    if inventory is None:
        return 0.0
    if isinstance(inventory, dict):
        return _num(inventory.get(kind, 0.0))
    return _num(_get(inventory, kind, 0.0))


def _tool_of(agent) -> str:
    """读 agent 手持工具名（无则 None）。"""
    tool = _get(_inventory(agent), "tool", None)
    return str(tool) if tool else ""


def _phys(agent, name: str, default: float = 0.0) -> float:
    """读 agent 的生理状态。"""
    return _num(_get(_get(agent, "physiology", None), name, default), default)


def _agent_x(agent) -> float:
    """读 agent 的 x 坐标。"""
    return _num(_get(agent, "x", 0.0))


def _feet_y(world, agent) -> float:
    """读 agent 脚部 y（缺省用地面线）。"""
    y = _get(agent, "y", None)
    if y is not None:
        return _num(y, 0.0)
    return _num(_get(_get(world, "config", None), "ground_y", 0.0))


def _set_facing(agent, direction: int) -> None:
    """让 agent 朝向坐标 x。"""
    if direction == 0:
        return
    try:
        agent.facing = 1 if direction > 0 else -1
    except Exception:
        pass


def _set_state(agent, state: str) -> None:
    """写 behavior_state。"""
    try:
        agent.behavior_state = state
    except Exception:
        pass


def _world_time(world) -> float | None:
    """世界时钟；世界没有 ``t`` 时返回 None（由 GoalManager 内部计时）。"""
    t = _get(world, "t", None)
    if t is None:
        return None
    value = _num(t, float("nan"))
    if math.isfinite(value):
        return value
    return None


def _tech(world):
    """读世界的技术层。"""
    return _get(world, "technology", None)


def _community(world):
    """读旧版社区地点。"""
    return _get(_tech(world), "community", None)


def _scene(world):
    """读像素场景。"""
    return _get(world, "pixel_scene", None)


def _nearest_resource(world, agent, kind: str):
    """``technology.nearest_resource`` 的容错封装，返回 ``(x, obj) | None``。"""
    tech = _tech(world)
    fn = _get(tech, "nearest_resource", None)
    if not callable(fn):
        return None
    try:
        return fn(agent, kind)
    except Exception:
        return None


def _water_x(world) -> float | None:
    """池塘中心 x（屏幕像素）；场景没有 pond 信息时返回 None。"""
    pond = _get(_scene(world), "pond", None)
    if not pond:
        return None
    try:
        left, right = pond
        return float(left + right) * 0.5 * float(ATOM_SIZE)
    except Exception:
        return None


def _near_water(world, agent) -> bool:
    """手部是否够得到水（与 behavior._near_water 同一判定）。"""
    scene = _scene(world)
    fn = _get(scene, "water_near", None)
    if not callable(fn):
        return False
    hand_x = _agent_x(agent) + float(_get(agent, "facing", 1) or 1) * 40.0
    hand_y = _feet_y(world, agent) - 40.0
    try:
        return bool(
            fn(int(hand_x / ATOM_SIZE), int(hand_y / ATOM_SIZE), radius_atoms=4.0)
        )
    except Exception:
        return False


def _nearest_berry(scene, x: float, feet_y: float):
    """最近的浆果（屏幕坐标换算后返回 ``(screen_x, atom)``）。"""
    berries = _get(scene, "berries", None) or ()
    best = None
    best_distance = math.inf
    for berry in berries:
        try:
            bx = float(berry[0]) * ATOM_SIZE
            by = float(berry[1]) * ATOM_SIZE
        except (TypeError, ValueError, IndexError):
            continue
        if by < feet_y - 190.0:
            continue
        distance = abs(bx - x)
        if distance < best_distance:
            best_distance = distance
            best = (bx, berry)
    return best


def _safe_float(value, default: float = 0.0) -> float:
    """安全转 float，失败给 default。"""
    return _num(value, default)


# ---------------------------------------------------------------------------
# Condition：可序列化、可求值、永不抛错的条件
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    """一个目标条件（完成/失败判定、步骤前置条件）。

    ``kind`` 取自 :data:`CONDITION_KINDS`。``target`` 与 ``value`` 的含义随
    kind 变化，例如 ``Condition("inventory_ge", "wood", value=4)``。
    ``progress_ge`` 在 :meth:`Goal.is_done` 内部改用该目标的 ``progress``
    字段求值；直接调用 :meth:`evaluate` 时读 ``agent.goal_progress`` 镜像。
    """

    kind: str
    target: str = ""
    op: str = ">="
    value: float = 0.0

    # -- 求值 ----------------------------------------------------------
    def evaluate(self, world, agent) -> bool:
        """在 ``world``/``agent`` 上求值；任何异常都返回 False（不抛错）。"""
        try:
            return bool(self._evaluate(world, agent))
        except Exception:
            return False

    def _evaluate(self, world, agent) -> bool:
        """评估条件是否满足。"""
        kind = self.kind
        if kind == "always":
            return True
        if kind == "inventory_ge":
            return _compare(_inv_get(agent, self.target), self.op, self.value)
        if kind == "has_tool":
            tool = _tool_of(agent)
            if self.target:
                return tool == self.target
            return bool(tool)
        if kind == "knowledge_verified":
            return self._knowledge_verified(world, agent)
        if kind == "structure_count_ge":
            structures = _num(_get(_community(world), "structures", 0.0))
            return _compare(structures, self.op, self.value)
        if kind == "facility_active":
            facilities = _get(_get(world, "civilization", None), "facilities", None) or ()
            for facility in facilities:
                if str(_get(facility, "kind", "")) != self.target:
                    continue
                if bool(_get(facility, "active", False)):
                    return True
            return False
        if kind == "energy_ge":
            return _compare(_phys(agent, "energy", 0.0), self.op, self.value)
        if kind == "hydration_ge":
            return _compare(_phys(agent, "hydration", 0.0), self.op, self.value)
        if kind == "at_site":
            return self._at_site(world, agent)
        if kind == "near_agent":
            threshold = self.value if self.value > 0.0 else 60.0
            return self._near_agent(world, agent, threshold)
        if kind == "progress_ge":
            return _compare(_num(_get(agent, "goal_progress", 0.0)), self.op, self.value)
        return False

    def _knowledge_verified(self, world, agent) -> bool:
        """该知识是否已被验证。"""
        store = _get(world, "knowledge", None)
        if store is not None:
            getter = _get(store, "get", None)
            if callable(getter):
                knowledge = getter(self.target)
                if knowledge is not None and str(_get(knowledge, "status", "")) == "verified":
                    return True
        known = _get(agent, "known_knowledge_ids", None) or ()
        try:
            return self.target in known
        except TypeError:
            return False

    def _at_site(self, world, agent) -> bool:
        """是否已到达目标 x 附近。"""
        tolerance = self.value if self.value > 0.0 else 14.0
        if self.target == "community":
            fn = _get(_tech(world), "near_community", None)
            if not callable(fn):
                return False
            try:
                return bool(fn(agent))
            except Exception:
                return False
        if self.target in ("water", "pond"):
            return _near_water(world, agent)
        if self.target.startswith("resource:"):
            kind = self.target.split(":", 1)[1]
            found = _nearest_resource(world, agent, kind)
            if found is None:
                return False
            try:
                return abs(float(found[0]) - _agent_x(agent)) <= tolerance
            except (TypeError, ValueError, IndexError):
                return False
        return False

    def _near_agent(self, world, agent, threshold: float) -> bool:
        """是否与某个体靠近。"""
        others = _get(world, "agents", None) or ()
        x = _agent_x(agent)
        best = math.inf
        for other in others:
            if other is agent:
                continue
            if not bool(_get(other, "alive", True)):
                continue
            best = min(best, abs(_agent_x(other) - x))
        return best <= threshold

    # -- 序列化 --------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """序列化为普通字典。"""
        return {
            "kind": self.kind,
            "target": self.target,
            "op": self.op,
            "value": float(self.value),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Condition":
        """从字典重建实例。"""
        return cls(
            kind=str(data.get("kind", "always")),
            target=str(data.get("target", "")),
            op=str(data.get("op", ">=")),
            value=_safe_float(data.get("value", 0.0)),
        )


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------
@dataclass
class Goal:
    """一个目标：完成条件 + 失败条件 + 预算 + 来源/理由。"""

    goal_id: int
    agent_id: int
    goal_type: str
    status: str = "active"
    priority: float = 0.5
    completion: tuple[Condition, ...] = ()
    failure: tuple[Condition, ...] = ()
    estimated_cost: float = 1.0
    max_steps: int = 1800            # 每个 update 计 1 步（dt≈1/60 时约 30 秒）
    created_at: float = 0.0
    steps_spent: int = 0
    progress: float = 0.0
    last_progress_at: float = 0.0
    attempts: int = 0
    source: str = "need"             # need | opportunity | inherited | social
    rationale: str = ""
    recipe_id: str | None = None     # experiment/craft 目标关联的知识
    #: 可选扩展：生成期参数（kind / target / need / facility_id / duration …）。
    #: 缺省为空字典，不改变契约中的构造方式。
    params: dict[str, object] = field(default_factory=dict)

    # -- 求值 ----------------------------------------------------------
    def condition_holds(self, condition: Condition, world, agent) -> bool:
        """求值单个条件；``progress_ge`` 使用本目标的 ``progress``。"""
        if condition.kind == "progress_ge":
            return _compare(float(self.progress), condition.op, condition.value)
        return condition.evaluate(world, agent)

    def is_done(self, world, agent) -> bool:
        """completion 全满足。空 completion 视为“未完成”（靠 max_steps 超时）。"""
        if not self.completion:
            return False
        try:
            return all(self.condition_holds(c, world, agent) for c in self.completion)
        except Exception:
            return False

    def is_failed(self, world, agent) -> bool:
        """failure 任一满足，或已超时。"""
        try:
            if any(c.evaluate(world, agent) for c in self.failure):
                return True
        except Exception:
            pass
        return self.steps_spent >= self.max_steps

    # -- 序列化 --------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """序列化为普通字典。"""
        return {
            "goal_id": int(self.goal_id),
            "agent_id": int(self.agent_id),
            "goal_type": self.goal_type,
            "status": self.status,
            "priority": float(self.priority),
            "completion": [c.to_dict() for c in self.completion],
            "failure": [c.to_dict() for c in self.failure],
            "estimated_cost": float(self.estimated_cost),
            "max_steps": int(self.max_steps),
            "created_at": float(self.created_at),
            "steps_spent": int(self.steps_spent),
            "progress": float(self.progress),
            "last_progress_at": float(self.last_progress_at),
            "attempts": int(self.attempts),
            "source": self.source,
            "rationale": self.rationale,
            "recipe_id": self.recipe_id,
            "params": _jsonable_params(self.params),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Goal":
        """从字典重建实例。"""
        return cls(
            goal_id=int(data.get("goal_id", 0)),
            agent_id=int(data.get("agent_id", 0)),
            goal_type=str(data.get("goal_type", "explore")),
            status=str(data.get("status", "active")),
            priority=_safe_float(data.get("priority", 0.5)),
            completion=tuple(Condition.from_dict(c) for c in data.get("completion", ())),
            failure=tuple(Condition.from_dict(c) for c in data.get("failure", ())),
            estimated_cost=_safe_float(data.get("estimated_cost", 1.0)),
            max_steps=int(data.get("max_steps", 1800)),
            created_at=_safe_float(data.get("created_at", 0.0)),
            steps_spent=int(data.get("steps_spent", 0)),
            progress=_safe_float(data.get("progress", 0.0)),
            last_progress_at=_safe_float(data.get("last_progress_at", 0.0)),
            attempts=int(data.get("attempts", 0)),
            source=str(data.get("source", "need")),
            rationale=str(data.get("rationale", "")),
            recipe_id=data.get("recipe_id"),
            params=dict(data.get("params", {}) or {}),
        )


def _jsonable_params(params: dict | None) -> dict[str, object]:
    """params 里只保留 JSON 可序列化的标量/元组，避免污染存档。"""
    result: dict[str, object] = {}
    for key, value in (params or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[str(key)] = value
        elif isinstance(value, (list, tuple)):
            result[str(key)] = [
                item for item in value
                if isinstance(item, (str, int, float, bool)) or item is None
            ]
        else:
            result[str(key)] = str(value)
    return result


# ---------------------------------------------------------------------------
# PlanStep / Plan
# ---------------------------------------------------------------------------
@dataclass
class PlanStep:
    """计划中的一步：结构化 verb + 参数 + 前置条件 + 代价。"""

    index: int
    verb: str
    args: dict[str, object] = field(default_factory=dict)
    preconditions: tuple[Condition, ...] = ()
    expected_effect: dict[str, float] = field(default_factory=dict)
    cost: float = 1.0
    status: str = "pending"
    attempts: int = 0
    max_attempts: int = 4
    last_error: str = ""

    def to_dict(self) -> dict[str, object]:
        """序列化为普通字典。"""
        return {
            "index": int(self.index),
            "verb": self.verb,
            "args": _jsonable_params(self.args),
            "preconditions": [c.to_dict() for c in self.preconditions],
            "expected_effect": {str(k): float(v) for k, v in self.expected_effect.items()},
            "cost": float(self.cost),
            "status": self.status,
            "attempts": int(self.attempts),
            "max_attempts": int(self.max_attempts),
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlanStep":
        """从字典重建实例。"""
        return cls(
            index=int(data.get("index", 0)),
            verb=str(data.get("verb", "observe")),
            args=dict(data.get("args", {}) or {}),
            preconditions=tuple(
                Condition.from_dict(c) for c in data.get("preconditions", ())
            ),
            expected_effect={
                str(k): _safe_float(v) for k, v in (data.get("expected_effect") or {}).items()
            },
            cost=_safe_float(data.get("cost", 1.0)),
            status=str(data.get("status", "pending")),
            attempts=int(data.get("attempts", 0)),
            max_attempts=int(data.get("max_attempts", 4)),
            last_error=str(data.get("last_error", "")),
        )


@dataclass
class Plan:
    """一条多步计划；``current`` 指向正在执行/待执行的步骤。"""

    plan_id: int
    agent_id: int
    goal_id: int
    steps: list[PlanStep] = field(default_factory=list)
    current: int = 0
    status: str = "active"
    created_at: float = 0.0
    last_progress_at: float = 0.0
    failure_reason: str = ""
    version: int = 1
    expected_cost: float = 0.0
    actual_steps: int = 0

    def current_step(self) -> PlanStep | None:
        """当前步骤；计划执行完返回 None。"""
        if 0 <= self.current < len(self.steps):
            return self.steps[self.current]
        return None

    def advance(self) -> None:
        """推进到下一步（并累计实际步数）。"""
        self.current += 1
        self.actual_steps += 1
        if self.current >= len(self.steps):
            self.current = len(self.steps)

    def to_dict(self) -> dict[str, object]:
        """序列化为普通字典。"""
        return {
            "plan_id": int(self.plan_id),
            "agent_id": int(self.agent_id),
            "goal_id": int(self.goal_id),
            "steps": [step.to_dict() for step in self.steps],
            "current": int(self.current),
            "status": self.status,
            "created_at": float(self.created_at),
            "last_progress_at": float(self.last_progress_at),
            "failure_reason": self.failure_reason,
            "version": int(self.version),
            "expected_cost": float(self.expected_cost),
            "actual_steps": int(self.actual_steps),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        """从字典重建实例。"""
        return cls(
            plan_id=int(data.get("plan_id", 0)),
            agent_id=int(data.get("agent_id", 0)),
            goal_id=int(data.get("goal_id", 0)),
            steps=[PlanStep.from_dict(s) for s in data.get("steps", ())],
            current=int(data.get("current", 0)),
            status=str(data.get("status", "active")),
            created_at=_safe_float(data.get("created_at", 0.0)),
            last_progress_at=_safe_float(data.get("last_progress_at", 0.0)),
            failure_reason=str(data.get("failure_reason", "")),
            version=int(data.get("version", 1)),
            expected_cost=_safe_float(data.get("expected_cost", 0.0)),
            actual_steps=int(data.get("actual_steps", 0)),
        )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class Planner:
    """把目标翻译成计划：有知识库走 recipe 依赖图，否则用内置基础链。"""

    def __init__(self, knowledge=None, *, memory=None) -> None:
        """初始化。"""
        self.knowledge = knowledge
        self.memory = memory
        self._next_plan_id = 0

    # -- 公开接口 ------------------------------------------------------
    def plan(self, world, agent, goal: Goal) -> Plan | None:
        """为目标生成计划；无法表达该目标时返回 None（不抛错）。"""
        try:
            steps = self._build_steps(world, agent, goal)
        except Exception:
            steps = None
        if not steps:
            return None
        for index, step in enumerate(steps):
            step.index = index
        self._next_plan_id += 1
        now = _world_time(world) or 0.0
        return Plan(
            plan_id=self._next_plan_id,
            agent_id=int(_get(goal, "agent_id", 0) or 0),
            goal_id=int(_get(goal, "goal_id", 0) or 0),
            steps=steps,
            created_at=now,
            last_progress_at=now,
            expected_cost=float(sum(_safe_float(step.cost, 1.0) for step in steps)),
        )

    def replan(self, world, agent, goal: Goal, failed_plan: Plan, reason: str) -> Plan | None:
        """依据失败原因重建计划（P1 采用“同一目标重新展开”的保守策略）。"""
        plan = self.plan(world, agent, goal)
        if plan is not None:
            plan.version = int(failed_plan.version) + 1
            plan.failure_reason = str(reason)
        return plan

    # -- 序列化（供 GoalManager 存档） ---------------------------------
    def to_dict(self) -> dict[str, object]:
        """序列化为普通字典。"""
        return {"next_plan_id": int(self._next_plan_id)}

    def from_dict(self, data: dict) -> None:
        """从字典重建实例。"""
        self._next_plan_id = int(data.get("next_plan_id", 0) or 0)

    # -- 计划展开 ------------------------------------------------------
    def _build_steps(self, world, agent, goal: Goal) -> list[PlanStep] | None:
        """为目标构建计划步骤序列。"""
        goal_type = str(_get(goal, "goal_type", ""))
        params = dict(_get(goal, "params", {}) or {})
        if goal_type == "survive":
            need = str(params.get("need", "eat"))
            if need == "drink":
                return [
                    self._move_step(site="water"),
                    self._step("drink", cost=1.0),
                ]
            if need == "rest":
                return [
                    self._step(
                        "rest",
                        cost=2.0,
                        duration=_safe_float(params.get("duration", 8.0), 8.0),
                    )
                ]
            return [
                self._move_step(site="berry"),
                self._step("eat", cost=1.0),
            ]
        if goal_type == "gather":
            kind, target = self._gather_target(goal)
            return [self._gather_step(kind, target)]
        if goal_type == "craft":
            return self._craft_chain(world, agent, goal, params)
        if goal_type == "build":
            return self._build_chain(world, agent, goal, params)
        if goal_type == "experiment":
            store = self.knowledge or _get(world, "knowledge", None)
            if store is None:
                return None
            knowledge_id = _get(goal, "recipe_id", None) or params.get("knowledge_id")
            if not knowledge_id:
                return None
            knowledge_id = str(knowledge_id)
            steps: list[PlanStep] = []
            getter = _get(store, "get", None)
            knowledge = getter(knowledge_id) if callable(getter) else None
            recipe = _get(knowledge, "recipe", None)
            # collect the materials the experiment needs *before* spending them:
            # without this the goal would just retry an unaffordable attempt.
            inputs = dict(_get(recipe, "inputs", {}) or {})
            if inputs:
                steps.extend(self._material_steps(world, agent, inputs))
            steps.append(
                self._step(
                    "experiment",
                    cost=3.0,
                    knowledge_id=knowledge_id,
                )
            )
            return steps
        if goal_type == "explore":
            return [
                self._step(
                    "explore",
                    cost=2.0,
                    duration=_safe_float(params.get("duration", 8.0), 8.0),
                )
            ]
        if goal_type == "socialize":
            return [
                self._step(
                    "socialize",
                    cost=1.5,
                    duration=_safe_float(params.get("duration", 6.0), 6.0),
                )
            ]
        if goal_type == "reproduce":
            return [self._step("reproduce", cost=2.0)]
        if goal_type == "maintain":
            facility_id = params.get("facility_id")
            steps: list[PlanStep] = []
            facility = None
            civilization = _get(world, "civilization", None)
            for item in _get(civilization, "facilities", None) or ():
                if str(_get(item, "facility_id", "")) == str(facility_id):
                    facility = item
                    break
            if facility is not None:
                costs = {
                    "wood": int(_num(_get(facility, "maintenance_wood", 0))),
                    "stone": int(_num(_get(facility, "maintenance_stone", 0))),
                }
                steps.extend(
                    self._material_steps(
                        world, agent, {k: v for k, v in costs.items() if v > 0}
                    )
                )
            steps.append(self._move_step(site="community"))
            steps.append(
                PlanStep(
                    index=0,
                    verb="maintain",
                    args={"facility_id": facility_id},
                    cost=2.0,
                )
            )
            return steps
        return None

    # -- 配方 / 依赖图 --------------------------------------------------
    def _craft_chain(self, world, agent, goal: Goal, params: dict) -> list[PlanStep] | None:
        """为配方解析“先做什么”的制造链。"""
        recipe_id = _get(goal, "recipe_id", None) or params.get("recipe")
        if not recipe_id:
            recipe_id = "stone_axe"
        recipe_id = str(recipe_id)
        # 1) 有知识库：按 recipe 依赖图做代价搜索
        if self.knowledge is not None:
            chain = self._knowledge_chain(world, agent, recipe_id)
            if chain:
                return chain
        # 2) 降级：内置配方表（gather → craft 的最小链）
        recipe = BASIC_RECIPES.get(recipe_id, BASIC_RECIPES["stone_axe"])
        inputs = dict(recipe.get("inputs", {}) or {})
        steps = self._material_steps(world, agent, inputs)
        steps.append(self._step("craft", cost=2.0, recipe=recipe_id))
        return steps

    def _build_chain(self, world, agent, goal: Goal, params: dict) -> list[PlanStep] | None:
        """递归解析配方的依赖链（有环即停）。"""
        community = _community(world)
        if community is None:
            return None
        need_wood = max(0, COMMUNITY_WOOD_NEED - int(_num(_get(community, "wood", 0))))
        need_stone = max(0, COMMUNITY_STONE_NEED - int(_num(_get(community, "stone", 0))))
        # 一次 contribute 交出 2 木 1 石；两次贡献即满足 4 木 2 石
        contributions = max((need_wood + 1) // 2, need_stone)
        total_wood = contributions * 2
        total_stone = contributions * 1
        steps: list[PlanStep] = []
        steps.extend(
            self._material_steps(world, agent, {"wood": total_wood, "stone": total_stone})
        )
        steps.append(self._move_step(site="community"))
        for _ in range(max(0, contributions - 1)):
            steps.append(self._step("contribute", cost=2.0))
        if contributions >= 1:
            steps.append(self._step("build", cost=2.0))
        elif not steps:
            steps.append(self._step("build", cost=2.0))
        return steps

    def _material_steps(
        self,
        world,
        agent,
        inputs: dict[str, int],
        *,
        visited: set[str] | None = None,
        depth: int = 0,
    ) -> list[PlanStep]:
        """为缺失材料生成 gather 或“先做中间产物”的步骤（代价搜索核心）。"""
        steps: list[PlanStep] = []
        for material, amount in inputs.items():
            needed = int(_num(amount))
            if needed <= 0:
                continue
            if _inv_get(agent, material) >= needed:
                continue
            producer = self._producer_for(str(material)) if self.knowledge is not None else None
            if producer and depth < MAX_RECIPE_DEPTH:
                sub = self._knowledge_chain(
                    world, agent, producer, visited=visited, depth=depth + 1
                )
                if sub:
                    steps.extend(sub)
                    continue
            # 无知识库时用内置配方表解析中间产物（例如 cord），保证 P1 的
            # 基础链仍是 gather → craft → build 的最小多步链。
            basic = self._basic_producer_for(str(material))
            if basic and depth < MAX_RECIPE_DEPTH:
                sub = self._basic_recipe_chain(world, agent, basic, depth=depth + 1)
                if sub:
                    steps.extend(sub)
                    continue
            steps.append(self._gather_step(str(material), needed))
        return steps

    @staticmethod
    def _basic_producer_for(material: str) -> str | None:
        """在内置配方表里找“产出 material”的 recipe_id。"""
        for recipe_id, recipe in BASIC_RECIPES.items():
            outputs = recipe.get("outputs", {}) or {}
            if material in outputs:
                return recipe_id
        return None

    def _basic_recipe_chain(
        self, world, agent, recipe_id: str, *, depth: int = 0
    ) -> list[PlanStep] | None:
        """用内置配方表递归展开一个 recipe（无知识库时的降级计划）。"""
        recipe = BASIC_RECIPES.get(str(recipe_id))
        if recipe is None or depth > MAX_RECIPE_DEPTH:
            return None
        steps = self._material_steps(
            world, agent, dict(recipe.get("inputs", {}) or {}), depth=depth + 1
        )
        steps.append(self._step("craft", cost=2.0, recipe=str(recipe_id)))
        return steps

    def _knowledge_chain(
        self,
        world,
        agent,
        recipe_id: str,
        *,
        visited: set[str] | None = None,
        depth: int = 0,
    ) -> list[PlanStep] | None:
        """在 knowledge 的 recipe 依赖图里递归展开 ``recipe_id``。"""
        store = self.knowledge
        if store is None or depth > MAX_RECIPE_DEPTH:
            return None
        visited = set() if visited is None else visited
        if recipe_id in visited:
            return None
        getter = _get(store, "get", None)
        knowledge = getter(recipe_id) if callable(getter) else None
        recipe = _get(knowledge, "recipe", None)
        if recipe is None:
            return None
        visited.add(recipe_id)
        try:
            inputs = dict(_get(recipe, "inputs", {}) or {})
        except Exception:
            inputs = {}
        steps = self._material_steps(
            world, agent, inputs, visited=visited, depth=depth + 1
        )
        status = str(_get(knowledge, "status", "hypothesis"))
        verb = "craft" if status == "verified" else "experiment"
        steps.append(
            self._step(verb, cost=2.0, recipe=recipe_id, knowledge_id=recipe_id)
        )
        visited.discard(recipe_id)
        return steps

    def _producer_for(self, material: str) -> str | None:
        """在已验证知识里找“产出 material”的 recipe_id。"""
        store = self.knowledge
        if store is None:
            return None
        lister = _get(store, "all", None)
        if not callable(lister):
            return None
        try:
            items = lister(status="verified")
        except Exception:
            try:
                items = lister()
            except Exception:
                items = []
        for knowledge in items or ():
            if str(_get(knowledge, "status", "")) != "verified":
                continue
            recipe = _get(knowledge, "recipe", None)
            outputs = _get(recipe, "outputs", None) or {}
            try:
                if _num(outputs.get(material, 0)) > 0:
                    return str(_get(knowledge, "knowledge_id", "")) or None
            except Exception:
                continue
        return None

    # -- 步骤构造小工具 -------------------------------------------------
    @staticmethod
    def _step(verb: str, *, cost: float = 1.0, **args) -> PlanStep:
        """执行计划的当前步骤。"""
        return PlanStep(
            index=0,
            verb=verb,
            args={key: value for key, value in args.items() if value is not None},
            cost=float(cost),
        )

    @staticmethod
    def _move_step(*, site: str, cost: float = 1.5) -> PlanStep:
        """move_to 步骤：朝目标走。"""
        return PlanStep(index=0, verb="move_to", args={"site": site}, cost=float(cost))

    @staticmethod
    def _gather_step(kind: str, target: float) -> PlanStep:
        """gather 步骤：采集资源。"""
        return PlanStep(
            index=0,
            verb="gather",
            args={"kind": kind, "target": float(max(1.0, target))},
            cost=float(max(1.0, target)),
        )

    @staticmethod
    def _gather_target(goal: Goal) -> tuple[str, float]:
        """找当前采集目标坐标。"""
        params = dict(_get(goal, "params", {}) or {})
        kind = str(params.get("kind", "") or "")
        target = _num(params.get("target", 0.0), 0.0)
        if kind and target > 0.0:
            return kind, target
        for condition in _get(goal, "completion", ()) or ():
            if condition.kind == "inventory_ge" and condition.target:
                return condition.target, max(1.0, condition.value)
        return (kind or "wood"), max(1.0, target or 1.0)


# ---------------------------------------------------------------------------
# 执行结果
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    """一步执行的结果：动作（可 None）/ 是否完成 / 错误原因。"""

    decision: Decision | None = None
    done: bool = False
    error: str = ""

    @classmethod
    def ok(cls) -> "StepResult":
        """步骤成功。"""
        return cls(None, True, "")

    @classmethod
    def moving(cls, decision: Decision | None) -> "StepResult":
        """步骤还在移动中。"""
        return cls(decision, False, "")

    @classmethod
    def fail(cls, error: str) -> "StepResult":
        """步骤失败。"""
        return cls(None, False, str(error))

    @classmethod
    def delegated(cls) -> "StepResult":
        """本层只决定"要探索"，具体移动交给底层策略（神经/教师）执行。"""
        return cls(None, False, "")


# ---------------------------------------------------------------------------
# GoalManager
# ---------------------------------------------------------------------------
class GoalManager:
    """目标生成 / 选择 / 计划推进 / 反卡死（P1 任务层入口）。"""

    def __init__(
        self,
        *,
        seed: int = 0,
        knowledge=None,
        memory=None,
        civilization=None,
        config=None,
    ) -> None:
        """初始化。"""
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.knowledge = knowledge
        self.memory = memory
        self.civilization = civilization
        self.config = config
        #: 集成层可能只把 CivilizationConfig 挂在 world.civ_config 上（Population
        #: 构造 GoalManager 时未必传 config），此时用运行时探测到的配置做开关判断
        self._runtime_config = config
        self._planner = Planner(knowledge, memory=memory)

        self._clock = 0.0
        self._next_goal_id = 0
        #: agent_id -> list[Goal]（含历史，不删除）
        self._goals: dict[int, list[Goal]] = {}
        #: agent_id -> 当前 Plan
        self._plans: dict[int, Plan] = {}
        #: agent_id -> 上次有效进展的世界时间
        self._progress_since: dict[int, float] = {}
        #: agent_id -> 连续无进展步数
        self._progress_steps: dict[int, int] = {}
        #: goal_id -> 连续重规划次数
        self._replan_chain: dict[int, int] = {}
        #: agent_id(str) -> {goal_type: 冷却截止时间}（按个体冷却，避免一人放弃
        #: 导致所有个体的同类目标都被冻结）
        self._cooldowns: dict[str, dict[str, float]] = {}
        #: agent_ids whose per-frame action is delegated to the low-level policy
        #: this frame (semantic step chosen, primitive action left to BC network)
        self._delegated: set[int] = set()

        self._stats: dict[str, float] = {
            "goals_created": 0.0,
            "goals_completed": 0.0,
            "goals_failed": 0.0,
            "goals_abandoned": 0.0,
            "plans_created": 0.0,
            "replans": 0.0,
            "steps_executed": 0.0,
            "plan_success_rate": 0.0,
            "cost_estimate_error": 0.0,
        }
        #: 内部计数器（不进 stats() 的 9 个契约键）
        self._plans_completed = 0
        self._cost_error_sum = 0.0
        self._cost_error_count = 0

    # ==================================================================
    # 公开接口
    # ==================================================================
    def update(self, world, agent, dt: float) -> Decision | None:
        """推进一帧：返回物理动作 Decision；没把握时返回 None。

        集成层在返回 None 时应回退到神经/社会教师，因此本方法在
        “世界接口不支持 / 计划无法建立 / 目标刚被放弃”时都返回 None，
        而不是硬塞一个无意义动作。
        """
        if not self._planning_enabled():
            return None
        if world is None or agent is None:
            return None
        if self.config is None:
            self._runtime_config = _get(world, "civ_config", None)
        dt = max(0.0, _safe_float(dt, 0.0))
        now = _world_time(world)
        if now is None:
            self._clock += dt
            now = self._clock
        else:
            self._clock = now

        agent_id = int(_safe_float(_get(agent, "agent_id", 0), 0.0))

        # 1) 结算已完成/失败目标
        self._settle(world, agent, agent_id, now)

        # 2) 选择/生成目标（需求与机会驱动）
        goal = self._select_goal(agent_id)
        if goal is None:
            goal = self._generate_goal(world, agent, agent_id, now)
            if goal is None:
                self._clear_goal_handle(agent)
                return None

        # 知识库可能只挂在 world 上（集成层已提供 world.knowledge），同步给规划器
        knowledge = self._store_knowledge(world)
        if self._planner.knowledge is not knowledge:
            self._planner.knowledge = knowledge
        # 当前目标句柄写回 agent（Character.current_goal_id 是集成层约定的字段）
        try:
            agent.current_goal_id = int(goal.goal_id)
        except Exception:
            pass

        # 3) 目标老化：每个 update 记 1 步（用于 max_steps 超时）
        for candidate in self._goals.get(agent_id, ()):
            if candidate.status == "active":
                candidate.steps_spent += 1
        self._mirror_progress(agent)

        # 4) 确保有可执行计划
        plan = self._plans.get(agent_id)
        if plan is None or plan.goal_id != goal.goal_id or plan.status != "active":
            plan = self._planner.plan(world, agent, goal)
            if plan is None:
                self._abandon(world, agent, goal, now, "no_plan")
                return None
            self._plans[agent_id] = plan
            self._stats["plans_created"] += 1.0
            self._reset_progress(agent_id, now)
            self._memory(
                agent_id, now, "procedural", f"plan:{goal.goal_type}",
                f"plan#{plan.plan_id} steps={len(plan.steps)}", True,
                detail=goal.rationale,
            )

        # 5) 推进计划一步
        return self._step_plan(world, agent, agent_id, goal, plan, dt, now)

    def goals_of(self, agent_id: int) -> list[Goal]:
        """该 agent 的历史目标（含已完成/失败/放弃）。"""
        return list(self._goals.get(int(agent_id), ()))

    def active_plan(self, agent_id: int) -> Plan | None:
        """该 agent 当前活跃计划。"""
        plan = self._plans.get(int(agent_id))
        if plan is not None and plan.status == "active":
            return plan
        return None

    # ------------------------------------------------------------------
    # 外部（学习型高层策略）接口
    # ------------------------------------------------------------------
    def request_goal(self, world, agent, goal_type: str) -> Goal | None:
        """由外部策略请求一个 **指定类型** 的新目标。

        返回 ``None`` 表示该类型当前不可行（前置条件不满足 / 处于冷却 /
        生成器没有该候选）。集成层据此回退到启发式并对策略给一个小负奖励，
        因此学习型策略不会被"硬塞"一个无法执行的目标。
        """
        if not self._planning_enabled():
            return None
        if world is None or agent is None:
            return None
        if self.config is None:
            self._runtime_config = _get(world, "civ_config", None)
        now = _world_time(world)
        if now is None:
            now = self._clock
        agent_id = int(_safe_float(_get(agent, "agent_id", 0), 0.0))
        self._settle(world, agent, agent_id, now)
        if not self._cooled(agent_id, str(goal_type), now):
            return None
        return self._generate_goal(world, agent, agent_id, now, preferred_type=str(goal_type))

    def has_active_goal(self, agent_id: int) -> bool:
        """该 agent 是否有活跃目标（用于判断是否需要请求新目标）。"""
        return self._select_goal(int(agent_id)) is not None

    def consume_delegation(self, agent_id: int) -> bool:
        """本帧是否把逐帧动作委托给底层网络（消费一次标记）。"""
        key = int(agent_id)
        delegated = key in self._delegated
        self._delegated.discard(key)
        return delegated

    def current_goal(self, agent_id: int) -> Goal | None:
        """该 agent 当前的活跃目标（无则 None）。"""
        return self._select_goal(int(agent_id))

    def stats(self) -> dict[str, float]:
        """规划层统计（契约 §4 的 9 个键）。"""
        created = max(1.0, self._stats["plans_created"])
        plan_success_rate = self._plans_completed / created
        cost_error = (
            self._cost_error_sum / self._cost_error_count
            if self._cost_error_count
            else 0.0
        )
        return {
            "goals_created": float(self._stats["goals_created"]),
            "goals_completed": float(self._stats["goals_completed"]),
            "goals_failed": float(self._stats["goals_failed"]),
            "goals_abandoned": float(self._stats["goals_abandoned"]),
            "plans_created": float(self._stats["plans_created"]),
            "replans": float(self._stats["replans"]),
            "steps_executed": float(self._stats["steps_executed"]),
            "plan_success_rate": float(plan_success_rate),
            "cost_estimate_error": float(cost_error),
        }

    def summary_zh(self) -> str:
        """中文摘要。"""
        stats = self.stats()
        active = sum(
            1
            for goals in self._goals.values()
            for goal in goals
            if goal.status == "active"
        )
        return (
            f"目标 生成{int(stats['goals_created'])} 完成{int(stats['goals_completed'])} "
            f"失败{int(stats['goals_failed'])} 放弃{int(stats['goals_abandoned'])} "
            f"活跃{active} · 计划 生成{int(stats['plans_created'])} "
            f"重规划{int(stats['replans'])} 步骤{int(stats['steps_executed'])} "
            f"成功率{stats['plan_success_rate']:.0%} "
            f"代价误差{stats['cost_estimate_error']:.2f}"
        )

    # -- 序列化 --------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """序列化为普通字典。"""
        state = self.rng.getstate()
        return {
            "version": 1,
            "seed": self.seed,
            "clock": float(self._clock),
            "next_goal_id": int(self._next_goal_id),
            "planner": self._planner.to_dict(),
            "goals": {
                str(agent_id): [goal.to_dict() for goal in goals]
                for agent_id, goals in self._goals.items()
            },
            "plans": {
                str(agent_id): plan.to_dict() for agent_id, plan in self._plans.items()
            },
            "stats": {key: float(value) for key, value in self._stats.items()},
            "internal": {
                "plans_completed": int(self._plans_completed),
                "cost_error_sum": float(self._cost_error_sum),
                "cost_error_count": int(self._cost_error_count),
            },
            "rng_state": [state[0], list(state[1]), state[2]],
            "cooldowns": {
                str(agent_id): {
                    kind: float(until) for kind, until in entries.items()
                }
                for agent_id, entries in self._cooldowns.items()
            },
            "replan_chain": {
                str(goal_id): int(count) for goal_id, count in self._replan_chain.items()
            },
            "progress": {
                str(agent_id): {
                    "since": float(self._progress_since.get(agent_id, 0.0)),
                    "steps": int(self._progress_steps.get(agent_id, 0)),
                }
                for agent_id in set(self._progress_since) | set(self._progress_steps)
            },
        }

    @classmethod
    def from_dict(cls, data: dict, **deps) -> "GoalManager":
        """恢复存档；``deps`` 可传 knowledge/memory/civilization/config。"""
        manager = cls(
            seed=int(data.get("seed", 0) or 0),
            knowledge=deps.get("knowledge"),
            memory=deps.get("memory"),
            civilization=deps.get("civilization"),
            config=deps.get("config"),
        )
        manager._clock = _safe_float(data.get("clock", 0.0))
        manager._next_goal_id = int(data.get("next_goal_id", 0) or 0)
        manager._planner.from_dict(data.get("planner", {}) or {})
        manager._goals = {
            int(agent_id): [Goal.from_dict(item) for item in items]
            for agent_id, items in (data.get("goals", {}) or {}).items()
        }
        manager._plans = {
            int(agent_id): Plan.from_dict(item)
            for agent_id, item in (data.get("plans", {}) or {}).items()
        }
        for key, value in (data.get("stats", {}) or {}).items():
            if key in manager._stats:
                manager._stats[key] = _safe_float(value)
        internal = data.get("internal", {}) or {}
        manager._plans_completed = int(internal.get("plans_completed", 0) or 0)
        manager._cost_error_sum = _safe_float(internal.get("cost_error_sum", 0.0))
        manager._cost_error_count = int(internal.get("cost_error_count", 0) or 0)
        manager._cooldowns = {
            str(agent_id): {
                str(kind): _safe_float(until) for kind, until in entries.items()
            }
            for agent_id, entries in (data.get("cooldowns", {}) or {}).items()
        }
        manager._replan_chain = {
            int(goal_id): int(count)
            for goal_id, count in (data.get("replan_chain", {}) or {}).items()
        }
        for agent_id, item in (data.get("progress", {}) or {}).items():
            manager._progress_since[int(agent_id)] = _safe_float(item.get("since", 0.0))
            manager._progress_steps[int(agent_id)] = int(item.get("steps", 0) or 0)
        rng_state = data.get("rng_state")
        if rng_state:
            try:
                manager.rng.setstate(
                    (
                        int(rng_state[0]),
                        tuple(int(value) for value in rng_state[1]),
                        rng_state[2],
                    )
                )
            except Exception:
                pass
        return manager

    # ==================================================================
    # 目标结算
    # ==================================================================
    def _settle(self, world, agent, agent_id: int, now: float) -> None:
        """结算一个目标（完成/失败/放弃三选一）。"""
        for goal in self._goals.get(agent_id, ()):
            if goal.status != "active":
                continue
            if goal.is_done(world, agent):
                self._complete(world, agent, goal, now)
            elif goal.is_failed(world, agent):
                self._fail(world, agent, goal, now)

    def _complete(self, world, agent, goal: Goal, now: float) -> None:
        """按完成结算目标。"""
        goal.status = "completed"
        goal.progress = max(goal.progress, 1.0)
        goal.last_progress_at = now
        self._stats["goals_completed"] += 1.0
        self._plans_completed += 1
        plan = self._plans.pop(int(goal.agent_id), None)
        if plan is not None and plan.goal_id == goal.goal_id:
            plan.status = "completed"
            expected = max(1.0, float(plan.expected_cost))
            error = abs(float(plan.actual_steps) - float(plan.expected_cost)) / expected
            self._cost_error_sum += error
            self._cost_error_count += 1
        self._replan_chain.pop(goal.goal_id, None)
        self._clear_goal_handle(agent)
        self._memory(
            int(goal.agent_id), now, "episode", f"goal:{goal.goal_type}",
            f"ok:goal#{goal.goal_id}", True, detail=goal.rationale, salience=0.7,
        )

    def _fail(self, world, agent, goal: Goal, now: float) -> None:
        """按失败结算目标。"""
        goal.status = "failed"
        self._stats["goals_failed"] += 1.0
        self._plans.pop(int(goal.agent_id), None)
        self._replan_chain.pop(goal.goal_id, None)
        self._clear_goal_handle(agent)
        self._cool_down(int(goal.agent_id), goal.goal_type, now)
        self._memory(
            int(goal.agent_id), now, "episode", f"goal:{goal.goal_type}",
            f"fail:goal#{goal.goal_id}", False, detail=goal.rationale, salience=0.6,
        )

    def _abandon(self, world, agent, goal: Goal, now: float, reason: str) -> None:
        """按放弃结算目标并进入冷却。"""
        if goal.status != "active":
            return
        goal.status = "abandoned"
        goal.attempts += 1
        self._stats["goals_abandoned"] += 1.0
        self._plans.pop(int(goal.agent_id), None)
        self._replan_chain.pop(goal.goal_id, None)
        self._clear_goal_handle(agent)
        self._cool_down(int(goal.agent_id), goal.goal_type, now)
        self._memory(
            int(goal.agent_id), now, "episode", f"goal:{goal.goal_type}",
            f"abandon:{reason}", False, detail=goal.rationale, salience=0.7,
        )

    # ==================================================================
    # 目标选择 / 生成（状态驱动，无固定剧情）
    # ==================================================================
    def _select_goal(self, agent_id: int) -> Goal | None:
        """从候选里选出下一个目标。"""
        best: Goal | None = None
        for goal in self._goals.get(agent_id, ()):
            if goal.status != "active":
                continue
            if best is None or goal.priority > best.priority:
                best = goal
        return best

    def _cool_down(self, agent_id: int, goal_type: str, now: float) -> None:
        """给某个个体的某种目标类型设置冷却（不影响其他个体）。"""
        entries = self._cooldowns.setdefault(str(int(agent_id)), {})
        entries[str(goal_type)] = float(now) + GOAL_COOLDOWN_S

    def _cooled(self, agent_id: int, goal_type: str, now: float) -> bool:
        """该 agent 的该类目标是否还在冷却。"""
        entries = self._cooldowns.get(str(int(agent_id)), None)
        until = float((entries or {}).get(str(goal_type), 0.0))
        return now + _EPS >= until

    def _generate_goal(
        self, world, agent, agent_id: int, now: float, *, preferred_type: str | None = None
    ) -> Goal | None:
        """按当前状态生成新目标；没有可行动世界接口时返回 None。

        ``preferred_type`` 是给"学习到的高层策略"用的：只接受该类型候选
        （不可行时返回 None，由集成层回退/惩罚），不改变其余生成规则。
        """
        tech = _tech(world)
        if tech is None:
            return None
        self._mirror_progress(agent)

        candidates: list[Goal] = []

        # -- 生理需求 ---------------------------------------------------
        energy = _phys(agent, "energy", 100.0)
        hydration = _phys(agent, "hydration", 100.0)
        sleep_pressure = _phys(agent, "sleep_pressure", 0.0)
        if energy < SURVIVE_ENERGY_LOW and self._cooled(agent_id, "survive", now):
            urgency = (SURVIVE_ENERGY_LOW - energy) / max(1.0, SURVIVE_ENERGY_LOW)
            candidates.append(
                self._make_goal(
                    agent_id, "survive", now,
                    priority=2.0 + urgency,
                    completion=(Condition("energy_ge", value=SURVIVE_ENERGY_TARGET),),
                    max_steps=1800,
                    rationale=f"能量{energy:.0f}低于{SURVIVE_ENERGY_LOW:.0f}，需要进食",
                    source="need",
                    params={"need": "eat"},
                )
            )
        if hydration < SURVIVE_HYDRATION_LOW and self._cooled(agent_id, "survive", now):
            urgency = (SURVIVE_HYDRATION_LOW - hydration) / max(1.0, SURVIVE_HYDRATION_LOW)
            candidates.append(
                self._make_goal(
                    agent_id, "survive", now,
                    priority=1.95 + urgency,
                    completion=(Condition("hydration_ge", value=SURVIVE_HYDRATION_TARGET),),
                    max_steps=1800,
                    rationale=f"水分{hydration:.0f}低于{SURVIVE_HYDRATION_LOW:.0f}，需要饮水",
                    source="need",
                    params={"need": "drink"},
                )
            )
        if sleep_pressure > SLEEP_PRESSURE_HIGH and self._cooled(agent_id, "survive", now):
            candidates.append(
                self._make_goal(
                    agent_id, "survive", now,
                    priority=1.8,
                    completion=(Condition("progress_ge", value=1.0),),
                    max_steps=1200,
                    rationale=f"困意{sleep_pressure:.0f}过高，需要休息",
                    source="need",
                    params={"need": "rest", "duration": 8.0},
                )
            )

        # -- 知识机会（P2）----------------------------------------------
        candidates.extend(self._knowledge_goals(world, agent, agent_id, now))

        # -- 库存 / 技术 / 公共设施 -------------------------------------
        wood = _inv_get(agent, "wood")
        stone = _inv_get(agent, "stone")
        tool = _tool_of(agent)
        community = _community(world)
        if community is not None and not bool(_get(community, "built", False)):
            if tool == "" and wood >= 2.0 and stone >= 1.0 and self._cooled(agent_id, "craft", now):
                candidates.append(
                    self._make_goal(
                        agent_id, "craft", now,
                        priority=1.35,
                        completion=(Condition("has_tool", "axe"),),
                        max_steps=900,
                        rationale="库存足够制作石斧（采集产量翻倍）",
                        source="opportunity",
                        recipe_id="stone_axe",
                        params={"recipe": "stone_axe"},
                    )
                )
            if wood >= 2.0 and stone >= 1.0:
                if self._cooled(agent_id, "build", now):
                    candidates.append(
                        self._make_goal(
                            agent_id, "build", now,
                            priority=1.25,
                            completion=(Condition("structure_count_ge", value=1.0),),
                            max_steps=1800,
                            rationale="公共营地缺乏结构，库存材料足够贡献",
                            source="need",
                            params={"need": "build"},
                        )
                    )
            else:
                # 只对“世界里真的能采到”的材料生成采集目标（状态驱动，
                # 不是写死 wood），否则退回探索/其他目标，允许失败与停滞。
                for kind, target in (("wood", 2.0), ("stone", 1.0)):
                    if _inv_get(agent, kind) >= target:
                        continue
                    if _nearest_resource(world, agent, kind) is None:
                        continue
                    if not self._cooled(agent_id, "gather", now):
                        break
                    candidates.append(
                        self._make_goal(
                            agent_id, "gather", now,
                            priority=1.2,
                            completion=(Condition("inventory_ge", kind, value=target),),
                            max_steps=1200,
                            rationale=f"营地需要{kind}，库存不足且附近有资源",
                            source="need",
                            params={"kind": kind, "target": target},
                        )
                    )
                    break

        # -- 公共设施维护（P4 前置）------------------------------------
        candidates.extend(self._maintain_goals(world, agent_id, now))

        # -- 繁殖（P3 钩子存在时才生成）--------------------------------
        reproduce_hook = _get(world, "try_reproduce", None) or _get(world, "reproduce", None)
        if callable(reproduce_hook) and self._cooled(agent_id, "reproduce", now):
            if bool(_get(community, "built", False)) and self._mate_exists(world, agent):
                candidates.append(
                    self._make_goal(
                        agent_id, "reproduce", now,
                        priority=0.9,
                        completion=(Condition("progress_ge", value=1.0),),
                        max_steps=600,
                        rationale="营地已建，且存在可配对个体",
                        source="social",
                    )
                )

        # -- 社交 -------------------------------------------------------
        if self._nearest_other(world, agent) is not None and self._cooled(agent_id, "socialize", now):
            candidates.append(
                self._make_goal(
                    agent_id, "socialize", now,
                    priority=0.55,
                    completion=(Condition("near_agent", value=55.0),),
                    max_steps=600,
                    rationale="附近有同伴",
                    source="social",
                    params={"duration": 6.0},
                )
            )

        # -- 无目标时探索 -----------------------------------------------
        if self._cooled(agent_id, "explore", now):
            candidates.append(
                self._make_goal(
                    agent_id, "explore", now,
                    priority=0.3,
                    completion=(Condition("progress_ge", value=1.0),),
                    max_steps=600,
                    rationale="没有更紧迫的目标，探索环境",
                    source="opportunity",
                    params={"duration": 8.0},
                )
            )

        if not candidates:
            return None
        # 丢弃“当前状态已经满足完成条件”的候选：否则会出现
        # “生成 → 下一帧立刻完成 → 再生成”的空转（例如两个同伴本就靠得很近
        # 时的 socialize 目标）。只有真正需要行动的目标才值得占用计划层。
        usable: list[Goal] = []
        for candidate in candidates:
            if preferred_type is not None and candidate.goal_type != preferred_type:
                continue
            try:
                if candidate.is_done(world, agent):
                    continue
            except Exception:
                pass
            usable.append(candidate)
        if not usable:
            return None
        best = max(usable, key=lambda goal: goal.priority)
        self._goals.setdefault(agent_id, []).append(best)
        self._stats["goals_created"] += 1.0
        self._reset_progress(agent_id, now)
        self._memory(
            agent_id, now, "episode", f"goal:{best.goal_type}",
            f"new:goal#{best.goal_id}", True, detail=best.rationale, salience=0.5,
        )
        return best

    def _store_knowledge(self, world):
        """知识库：优先构造参数，其次从 ``world.knowledge`` 探测（集成层已提供）。"""
        return self.knowledge or _get(world, "knowledge", None)

    def _knowledge_goals(self, world, agent, agent_id: int, now: float) -> list[Goal]:
        """知识候选驱动的 experiment 目标（P2；无知识库时为空）。"""
        store = self._store_knowledge(world)
        if store is None:
            return []
        if not self._flag("enable_knowledge", True):
            return []
        if not self._cooled(agent_id, "experiment", now):
            return []
        finder = _get(store, "candidates_for", None)
        if not callable(finder):
            return []
        # Ask for candidates **without** the affordability filter (world=None):
        # the planner inserts the missing material-gathering steps itself, so a
        # hypothesis that currently lacks materials is still worth a plan.  If
        # the store only supports the two-argument form, fall back to it.
        try:
            candidates = finder(agent) or []
        except TypeError:
            candidates = []
        except Exception:
            candidates = []
        if not candidates:
            try:
                candidates = finder(agent, world) or []
            except Exception:
                candidates = []
        if not candidates:
            return []
        # Prefer the hypothesis with the fewest attempts so the population does
        # not get stuck re-testing the same hard recipe forever (e.g. basket
        # difficulty 0.4) while other affordable techniques are never tried.
        def _attempts(item) -> tuple[int, str]:
            """（嵌套）解析实验记录为 (次数, 原因)。"""
            uses = int(_num(_get(item, "successful_uses", 0), 0)) + int(
                _num(_get(item, "failed_uses", 0), 0)
            )
            return (uses, str(_get(item, "knowledge_id", "")))

        knowledge = min(candidates, key=_attempts)
        knowledge_id = str(_get(knowledge, "knowledge_id", "") or "")
        if not knowledge_id:
            return []
        name = str(_get(knowledge, "name_zh", "") or knowledge_id)
        return [
            self._make_goal(
                agent_id, "experiment", now,
                priority=1.6,
                completion=(Condition("knowledge_verified", target=knowledge_id),),
                max_steps=900,
                rationale=f"尝试未验证的技术：{name}",
                source="opportunity",
                recipe_id=knowledge_id,
                params={"knowledge_id": knowledge_id},
            )
        ]

    def _maintain_goals(self, world, agent_id: int, now: float) -> list[Goal]:
        """公共设施损耗驱动的 maintain 目标（设施接口缺失时为空）。"""
        if not self._cooled(agent_id, "maintain", now):
            return []
        civilization = _get(world, "civilization", None) or self.civilization
        facilities = _get(civilization, "facilities", None) or ()
        for facility in facilities:
            # 只对“已经失效”的设施生成维护目标：completion 条件
            # facility_active 只有在设施从 inactive 变回 active 时才有意义
            if bool(_get(facility, "active", False)):
                continue
            condition = _num(_get(facility, "condition", 0.0), 0.0)
            facility_id = str(_get(facility, "facility_id", "") or "")
            if not facility_id:
                continue
            return [
                self._make_goal(
                    agent_id, "maintain", now,
                    priority=1.1,
                    completion=(
                        Condition("facility_active", target=str(_get(facility, "kind", ""))),
                    ),
                    max_steps=900,
                    rationale=f"设施{facility_id}损耗到{condition:.2f}，需要维护",
                    source="need",
                    params={"facility_id": facility_id},
                )
            ]
        return []

    def _mate_exists(self, world, agent) -> bool:
        """是否存在可选伴侣。"""
        for other in _get(world, "agents", None) or ():
            if other is agent:
                continue
            if not bool(_get(other, "alive", True)):
                continue
            if int(_safe_float(_get(other, "age", 0.0))) < 70:
                continue
            if str(_get(other, "sex", "")) != str(_get(agent, "sex", "")):
                return True
        return False

    def _nearest_other(self, world, agent):
        """找最近的另一个体。"""
        others = _get(world, "agents", None) or ()
        best = None
        best_distance = 160.0
        for other in others:
            if other is agent or not bool(_get(other, "alive", True)):
                continue
            distance = abs(_agent_x(other) - _agent_x(agent))
            if distance <= best_distance:
                best_distance = distance
                best = other
        return best

    def _make_goal(
        self,
        agent_id: int,
        goal_type: str,
        now: float,
        *,
        priority: float,
        completion: tuple[Condition, ...] = (),
        failure: tuple[Condition, ...] = (),
        estimated_cost: float = 1.0,
        max_steps: int = 1800,
        source: str = "need",
        rationale: str = "",
        recipe_id: str | None = None,
        params: dict | None = None,
    ) -> Goal:
        """构造一个目标实例。"""
        self._next_goal_id += 1
        # 小幅随机抖动：同级目标不总是按固定顺序出现（但仍可复现）
        priority = float(priority) + self.rng.uniform(0.0, 0.05)
        return Goal(
            goal_id=self._next_goal_id,
            agent_id=int(agent_id),
            goal_type=goal_type,
            priority=priority,
            completion=tuple(completion),
            failure=tuple(failure),
            estimated_cost=float(estimated_cost),
            max_steps=int(max_steps),
            created_at=now,
            last_progress_at=now,
            source=source,
            rationale=rationale,
            recipe_id=recipe_id,
            params=dict(params or {}),
        )

    # ==================================================================
    # 计划推进 / 反卡死
    # ==================================================================
    def _step_plan(
        self,
        world,
        agent,
        agent_id: int,
        goal: Goal,
        plan: Plan,
        dt: float,
        now: float,
    ) -> Decision | None:
        """推进当前计划一步，产出 Decision。"""
        step = plan.current_step()
        if step is None:
            plan.status = "completed"
            self._handle_plan_failure(world, agent, goal, plan, "plan_exhausted", now)
            return None

        before = self._snapshot(world, agent, goal)
        step.status = "running"
        if step.verb not in VERBS:
            result = StepResult.fail(f"unknown_verb:{step.verb}")
        else:
            handler = _get(self, f"_verb_{step.verb}", None)
            if not callable(handler):
                result = StepResult.fail(f"unknown_verb:{step.verb}")
            else:
                try:
                    result = handler(world, agent, goal, plan, step, dt, now)
                except Exception as exc:  # 执行器绝不能把异常抛回主循环
                    result = StepResult.fail(f"exception:{type(exc).__name__}:{exc}")
                if not isinstance(result, StepResult):
                    result = StepResult.fail("bad_step_result")

        if result.error:
            step.attempts += 1
            step.last_error = result.error
            self._memory(
                agent_id, now, "episode", f"{step.verb}:{result.error}",
                f"fail:{result.error}", False, salience=0.5,
            )
            if step.attempts > step.max_attempts:
                step.status = "failed"
                plan.status = "failed"
                plan.failure_reason = result.error
                self._handle_plan_failure(world, agent, goal, plan, result.error, now)
            return result.decision

        if result.decision is None and not result.done:
            # “语义已定、逐帧动作委托底层”：标记给集成层，让它直接交给
            # 行为克隆网络（而不是被求偶/建造教师接管后原地站立）。
            self._delegated.add(int(agent_id))
        else:
            self._delegated.discard(int(agent_id))

        if result.done:
            step.status = "done"
            plan.advance()
            plan.last_progress_at = now
            goal.last_progress_at = now
            self._stats["steps_executed"] += 1.0
            self._reset_progress(agent_id, now)
            self._memory(
                agent_id, now, "procedural", f"{step.verb}",
                f"ok:{step.verb}", True, salience=0.5,
            )
            if plan.current_step() is None:
                # 计划跑完：目标满足则完成，否则视为“计划耗尽”重规划
                if goal.is_done(world, agent):
                    self._complete(world, agent, goal, now)
                else:
                    plan.status = "failed"
                    plan.failure_reason = "plan_exhausted"
                    self._handle_plan_failure(
                        world, agent, goal, plan, "plan_exhausted", now
                    )
            return result.decision

        # 未完成：对比执行前后的状态，判断是否“有进展”
        after = self._snapshot(world, agent, goal)
        if after != before:
            self._reset_progress(agent_id, now)
        else:
            self._progress_steps[agent_id] = self._progress_steps.get(agent_id, 0) + 1
            since = self._progress_since.get(agent_id, now)
            if (
                now - since >= NO_PROGRESS_S - _EPS
                or self._progress_steps[agent_id] >= NO_PROGRESS_STEPS
            ):
                self._handle_plan_failure(world, agent, goal, plan, "no_progress", now)
                self._reset_progress(agent_id, now)
        return result.decision

    def _handle_plan_failure(
        self,
        world,
        agent,
        goal: Goal,
        plan: Plan,
        reason: str,
        now: float,
    ) -> None:
        """计划失败/无进展：重规划；连续达到上限则放弃并冷却。"""
        plan.status = "failed"
        plan.failure_reason = str(reason)
        count = int(self._replan_chain.get(goal.goal_id, 0)) + 1
        self._replan_chain[goal.goal_id] = count
        self._stats["replans"] += 1.0
        self._memory(
            int(goal.agent_id), now, "episode", f"replan:{goal.goal_type}",
            f"replan:{reason}", False, detail=plan.failure_reason, salience=0.6,
        )
        if count >= MAX_REPLANS:
            self._abandon(world, agent, goal, now, f"replan_limit:{reason}")
            return
        new_plan = self._planner.replan(world, agent, goal, plan, reason)
        if new_plan is None:
            self._abandon(world, agent, goal, now, f"no_replan:{reason}")
            return
        new_plan.version = int(plan.version) + 1
        self._plans[int(goal.agent_id)] = new_plan
        self._stats["plans_created"] += 1.0
        self._reset_progress(int(goal.agent_id), now)

    # -- 进展镜像 / 快照 -----------------------------------------------
    def _reset_progress(self, agent_id: int, now: float) -> None:
        """重置该 agent 的反卡死进度计数。"""
        self._progress_since[agent_id] = float(now)
        self._progress_steps[agent_id] = 0

    def _snapshot(self, world, agent, goal: Goal) -> list:
        """“是否真的推进了任务”的快照：位置/库存/营地/目标进度/条件满足数。"""
        community = _community(world)
        completion_flags = [
            1 if goal.condition_holds(condition, world, agent) else 0
            for condition in goal.completion
        ]
        try:
            x = round(_agent_x(agent), 1)
        except Exception:
            x = 0.0
        return [
            x,
            int(_inv_get(agent, "wood")),
            int(_inv_get(agent, "stone")),
            _tool_of(agent),
            int(_num(_get(community, "wood", 0.0))),
            int(_num(_get(community, "stone", 0.0))),
            int(_num(_get(community, "structures", 0.0))),
            round(float(goal.progress), 3),
            completion_flags,
        ]

    @staticmethod
    def _mirror_progress(agent) -> None:
        """确保 ``agent.goal_progress`` 存在（progress_ge 条件的读取镜像）。"""
        try:
            if not hasattr(agent, "goal_progress"):
                agent.goal_progress = 0.0
        except Exception:
            pass

    def _set_goal_progress(self, agent, goal: Goal, progress: float) -> None:
        """写入该 agent 的目标进度镜像。"""
        goal.progress = max(0.0, min(1.0, float(progress)))
        try:
            agent.goal_progress = goal.progress
        except Exception:
            pass

    @staticmethod
    def _clear_goal_handle(agent) -> None:
        """清空 agent.current_goal_id（集成层约定的“当前目标句柄”）。"""
        try:
            agent.current_goal_id = None
        except Exception:
            pass

    def _memory(
        self,
        agent_id: int,
        now: float,
        kind: str,
        action: str,
        outcome: str,
        success: bool,
        detail: str = "",
        salience: float = 0.5,
    ) -> None:
        """读记忆库（可缺失）。"""
        store = self.memory
        if store is None:
            return
        recorder = _get(store, "record", None)
        if not callable(recorder):
            return
        try:
            recorder(
                int(agent_id), float(now), kind, action, outcome, bool(success),
                detail=str(detail), salience=float(salience),
            )
        except Exception:
            pass

    def _flag(self, name: str, default: bool) -> bool:
        """读消融开关：构造参数优先，其次运行时探测到的 world.civ_config。"""
        config = self.config if self.config is not None else self._runtime_config
        value = _get(config, name, None)
        if value is None:
            return bool(default)
        return bool(value)

    def _planning_enabled(self) -> bool:
        """规划机制是否开启。"""
        return self._flag("enable_planning", True)

    # ==================================================================
    # 执行器：verb -> 世界公开方法 -> Decision
    # ==================================================================
    def _resolve_site_x(self, world, agent, site: str) -> float | None:
        """解析某类设施/地点的目标 x 坐标。"""
        site = str(site or "")
        if site == "community":
            community = _community(world)
            if community is None:
                return None
            return _num(_get(community, "x_screen", None), 0.0) or None
        if site in ("water", "pond"):
            return _water_x(world)
        if site.startswith("resource:"):
            kind = site.split(":", 1)[1]
            found = _nearest_resource(world, agent, kind)
            if found is None:
                return None
            return _num(found[0], 0.0)
        if site == "berry":
            scene = _scene(world)
            if scene is None:
                return None
            berry = _nearest_berry(scene, _agent_x(agent), _feet_y(world, agent))
            return None if berry is None else _num(berry[0], 0.0)
        if site == "agent":
            other = self._nearest_other(world, agent)
            return None if other is None else _agent_x(other)
        return None

    def _verb_move_to(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：朝步骤目标移动。"""
        target = _arg(step.args, "x", None)
        if target is None:
            target = self._resolve_site_x(world, agent, str(_arg(step.args, "site", "")))
        if target is None:
            return StepResult.fail("no_target")
        tolerance = _num(_arg(step.args, "tolerance", None), 14.0)
        delta = _num(target) - _agent_x(agent)
        if abs(delta) <= tolerance:
            _set_state(agent, "arrive")
            return StepResult.ok()
        direction = 1 if delta > 0 else -1
        _set_facing(agent, direction)
        return StepResult.moving(Decision(direction, False, False))

    def _verb_gather(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：采集材料。"""
        kind = str(_arg(step.args, "kind", "wood") or "wood")
        target = _arg(step.args, "target", None)
        if target is None:
            target = _inv_get(agent, kind) + _num(_arg(step.args, "count", 1.0), 1.0)
        target = _num(target, 1.0)
        if _inv_get(agent, kind) >= target:
            return StepResult.ok()
        found = _nearest_resource(world, agent, kind)
        if found is None:
            return StepResult.fail(f"no_{kind}_resource")
        target_x = _num(found[0], _agent_x(agent))
        delta = target_x - _agent_x(agent)
        if abs(delta) > 14.0:
            direction = 1 if delta > 0 else -1
            _set_facing(agent, direction)
            _set_state(agent, "gather")
            return StepResult.moving(Decision(direction, False, False))
        _set_facing(agent, 1 if delta >= 0 else -1)
        # 库存容量上限（背篓会提高它）：满了就是真实失败，交给反卡死处理
        inventory = _inventory(agent)
        free_fn = _get(inventory, "free", None)
        if callable(free_fn):
            try:
                if float(free_fn()) <= 0.0:
                    return StepResult.fail("inventory_full")
            except Exception:
                pass
        before = _inv_get(agent, kind)
        harvester = _get(_tech(world), "harvest", None)
        if not callable(harvester):
            return StepResult.fail("no_harvest_api")
        try:
            harvested = bool(harvester(agent, kind))
        except Exception:
            harvested = False
        gained = _inv_get(agent, kind) > before
        if not harvested or not gained:
            return StepResult.fail(f"harvest_failed:{kind}")
        _set_state(agent, "gather")
        if _inv_get(agent, kind) >= target:
            return StepResult.ok()
        return StepResult.moving(Decision(0, False, False))

    def _verb_craft(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：执行配方制作。"""
        if _tool_of(agent):
            return StepResult.ok()
        recipe_id = _arg(step.args, "recipe", None)
        tech = _tech(world)
        store = self.knowledge or _get(world, "knowledge", None)
        # 1) 有知识库且该配方已验证：交给唯一权威 technology.execute_recipe
        if recipe_id and store is not None:
            getter = _get(store, "get", None)
            knowledge = getter(str(recipe_id)) if callable(getter) else None
            recipe = _get(knowledge, "recipe", None)
            executor = _get(tech, "execute_recipe", None)
            if recipe is not None and callable(executor):
                if str(_get(knowledge, "status", "")) != "verified":
                    # hypothesis 只能走 experiment 步骤，不能直接“制作”
                    return StepResult.fail("recipe_not_verified")
                try:
                    crafted, _reason = executor(agent, recipe)
                except Exception:
                    crafted = False
                if not crafted:
                    return StepResult.fail("craft_infeasible")
                _set_state(agent, "craft")
                return StepResult.ok()
        # 2) 旧接口钩子 / 原始 craft（P1 在无知识库时仍可验证 stone_axe 链）
        handler = _get(tech, "craft_recipe", None)
        if callable(handler) and recipe_id:
            try:
                crafted = bool(handler(agent, str(recipe_id)))
            except Exception:
                crafted = False
        else:
            if recipe_id and str(recipe_id) not in BASIC_RECIPES:
                return StepResult.fail("no_craft_recipe_hook")
            crafter = _get(tech, "craft", None)
            if not callable(crafter):
                return StepResult.fail("no_craft_api")
            try:
                crafted = bool(crafter(agent))
            except Exception:
                crafted = False
        if not crafted:
            return StepResult.fail("craft_infeasible")
        _set_state(agent, "craft")
        return StepResult.ok()

    def _verb_experiment(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：做技术实验（可能失败烧材料）。"""
        store = self.knowledge or _get(world, "knowledge", None)
        knowledge_id = (
            _arg(step.args, "knowledge_id", None)
            or _arg(step.args, "recipe", None)
            or _get(goal, "recipe_id", None)
        )
        if store is None or not knowledge_id:
            return StepResult.fail("no_knowledge_store")
        getter = _get(store, "get", None)
        knowledge = getter(str(knowledge_id)) if callable(getter) else None
        if knowledge is None:
            return StepResult.fail(f"unknown_knowledge:{knowledge_id}")
        attempt = _get(store, "attempt", None)
        if not callable(attempt):
            return StepResult.fail("no_attempt_api")
        try:
            success, reason = attempt(agent, knowledge, world, now)
        except Exception as exc:
            return StepResult.fail(f"experiment_exception:{type(exc).__name__}")
        _set_state(agent, "experiment")
        if success:
            return StepResult.ok()
        return StepResult.fail(f"experiment_failed:{reason}")

    def _verb_contribute(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：向社区捐献材料。"""
        tech = _tech(world)
        near = _get(tech, "near_community", None)
        if callable(near):
            try:
                close = bool(near(agent))
            except Exception:
                close = False
            if not close:
                target_x = self._resolve_site_x(world, agent, "community")
                if target_x is None:
                    return StepResult.fail("no_community")
                delta = target_x - _agent_x(agent)
                direction = 1 if delta > 0 else -1
                _set_facing(agent, direction)
                _set_state(agent, "forage")
                return StepResult.moving(Decision(direction, False, False))
        contributor = _get(tech, "contribute", None)
        if not callable(contributor):
            return StepResult.fail("no_contribute_api")
        try:
            contributed = bool(contributor(agent))
        except Exception:
            contributed = False
        if not contributed:
            return StepResult.fail("contribute_infeasible")
        _set_state(agent, "build")
        return StepResult.ok()

    def _verb_build(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：向社区捐料建结构。"""
        community = _community(world)
        if community is None:
            return StepResult.fail("no_community")
        if bool(_get(community, "built", False)):
            return StepResult.ok()
        if _inv_get(agent, "wood") < 2 or _inv_get(agent, "stone") < 1:
            return StepResult.fail("no_build_materials")
        return self._verb_contribute(world, agent, goal, plan, step, dt, now)

    def _verb_deliver(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：搬运材料（contribute 的别名）。"""
        return self._verb_contribute(world, agent, goal, plan, step, dt, now)

    def _verb_maintain(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：维护公共设施（三级降级：repair → maintain → CivilizationState.maintain）。"""
        tech = _tech(world)
        civilization = _get(world, "civilization", None) or self.civilization
        facility_id = _arg(step.args, "facility_id", None)
        if not facility_id:
            return StepResult.fail("no_facility_id")
        facility = None
        for item in _get(civilization, "facilities", None) or ():
            if str(_get(item, "facility_id", "")) == str(facility_id):
                facility = item
                break
        # 1) 首选 technology.repair(agent, facility)：唯一权威，负责扣材料并恢复 condition
        repair = _get(tech, "repair", None)
        if callable(repair):
            if facility is None:
                return StepResult.fail(f"unknown_facility:{facility_id}")
            try:
                repaired = bool(repair(agent, facility))
            except Exception:
                repaired = False
            if not repaired:
                return StepResult.fail("repair_infeasible")
            _set_state(agent, "maintain")
            return StepResult.ok()
        # 2) 旧钩子 world.technology.maintain(agent)
        tech_maintain = _get(tech, "maintain", None)
        if callable(tech_maintain):
            try:
                if bool(tech_maintain(agent)):
                    _set_state(agent, "maintain")
                    return StepResult.ok()
            except Exception:
                pass
        # 3) CivilizationState.maintain：契约要求调用方已从库存扣除材料，
        #    本模块不直接写库存，因此只在“无材料需求”时走这条兜底路径。
        if civilization is None:
            return StepResult.fail("no_civilization")
        maintain = _get(civilization, "maintain", None)
        if not callable(maintain):
            return StepResult.fail("no_maintain_api")
        cost_wood = int(_num(_get(facility, "maintenance_wood", 0)))
        cost_stone = int(_num(_get(facility, "maintenance_stone", 0)))
        try:
            maintained = bool(
                maintain(
                    str(facility_id),
                    int(_safe_float(_get(agent, "agent_id", 0))),
                    now,
                    wood=cost_wood,
                    stone=cost_stone,
                )
            )
        except Exception:
            maintained = False
        if not maintained:
            return StepResult.fail("maintain_failed")
        _set_state(agent, "maintain")
        return StepResult.ok()

    def _verb_drink(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：找水喝水。"""
        physiology = _get(agent, "physiology", None)
        if physiology is None:
            return StepResult.fail("no_physiology")
        if not _near_water(world, agent):
            water_x = _water_x(world)
            if water_x is None:
                return StepResult.fail("no_water")
            delta = water_x - _agent_x(agent)
            if abs(delta) > 10.0:
                direction = 1 if delta > 0 else -1
                _set_facing(agent, direction)
                _set_state(agent, "forage")
                return StepResult.moving(Decision(direction, False, False))
            return StepResult.fail("not_near_water")
        drinker = _get(_tech(world), "drink", None)
        if callable(drinker):
            try:
                if bool(drinker(agent)):
                    _set_state(agent, "drink")
                    return StepResult.ok()
            except Exception:
                pass
        fn = _get(physiology, "drink", None)
        if not callable(fn):
            return StepResult.fail("no_drink_api")
        try:
            fn()
        except Exception:
            return StepResult.fail("drink_failed")
        _set_state(agent, "drink")
        return StepResult.ok()

    def _verb_eat(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：摘浆果吃。"""
        scene = _scene(world)
        if scene is None:
            return StepResult.fail("no_scene")
        feet_y = _feet_y(world, agent)
        berry = _nearest_berry(scene, _agent_x(agent), feet_y)
        if berry is None:
            return StepResult.fail("no_food")
        berry_x, atom = berry
        horizontal = abs(berry_x - _agent_x(agent))
        berry_y = _num(atom[1]) * ATOM_SIZE
        low = feet_y - 115.0 <= berry_y <= feet_y + 5.0
        if horizontal <= 42.0 and low:
            # 优先正式钩子 world.technology.eat(agent)；否则走 behavior 同款路径
            eater = _get(_tech(world), "eat", None)
            if callable(eater):
                try:
                    if bool(eater(agent)):
                        _set_state(agent, "eat")
                        return StepResult.ok()
                except Exception:
                    pass
            picker = _get(scene, "pick_berry", None)
            physiology = _get(agent, "physiology", None)
            eat_fn = _get(physiology, "eat", None)
            if not callable(picker) or not callable(eat_fn):
                return StepResult.fail("no_eat_api")
            try:
                picker(atom)
                eat_fn()
            except Exception:
                return StepResult.fail("eat_failed")
            _set_state(agent, "eat")
            return StepResult.ok()
        if horizontal > 18.0:
            direction = 1 if berry_x > _agent_x(agent) else -1
            _set_facing(agent, direction)
            _set_state(agent, "forage")
            return StepResult.moving(Decision(direction, False, False))
        if berry_y < feet_y - 115.0 and bool(_get(agent, "on_ground", True)):
            _set_state(agent, "forage")
            return StepResult.moving(Decision(0, True, False))
        return StepResult.fail("berry_unreachable")

    def _verb_rest(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：原地休息恢复。"""
        duration = _num(_arg(step.args, "duration", 8.0), 8.0)
        started = step.args.get("_started_at")
        if started is None:
            step.args["_started_at"] = float(now)
            started = now
        elapsed = max(0.0, float(now) - _num(started, now))
        self._set_goal_progress(agent, goal, elapsed / max(_EPS, duration))
        physiology = _get(agent, "physiology", None)
        sleep_fn = _get(physiology, "sleep", None)
        wake_fn = _get(physiology, "wake", None)
        if elapsed >= duration:
            if callable(wake_fn):
                try:
                    wake_fn()
                except Exception:
                    pass
            _set_state(agent, "rest")
            return StepResult.ok()
        if callable(sleep_fn):
            try:
                if not bool(_get(physiology, "sleeping", False)):
                    sleep_fn()
            except Exception:
                pass
        _set_state(agent, "rest")
        return StepResult.moving(Decision(0, False, False))

    def _verb_explore(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：随机探索（产出移动决策）。"""
        duration = _num(_arg(step.args, "duration", 8.0), 8.0)
        started = step.args.get("_started_at")
        if started is None:
            step.args["_started_at"] = float(now)
            started = now
        elapsed = max(0.0, float(now) - _num(started, now))
        self._set_goal_progress(agent, goal, elapsed / max(_EPS, duration))
        if elapsed >= duration:
            _set_state(agent, "explore")
            return StepResult.ok()
        # 状态驱动的方向：优先走向更开阔的一侧，偶尔反向
        config = _get(world, "config", None)
        min_x = _num(_get(config, "min_x", 0.0))
        max_x = _num(_get(config, "max_x", 0.0))
        if max_x > min_x:
            direction = 1 if (max_x - _agent_x(agent)) >= (_agent_x(agent) - min_x) else -1
            if self.rng.random() < 0.2:
                direction = -direction
        else:
            direction = self.rng.choice((-1, 1))
        _set_facing(agent, direction)
        _set_state(agent, "explore")
        # 语义层只声明"探索"，逐帧移动/跳跃/说话交给底层策略执行：
        # 这样行为克隆网络才真正参与决策，而不是被规划器完全覆盖。
        return StepResult.delegated()

    def _verb_observe(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：原地观察（观察其它个体）。"""
        duration = _num(_arg(step.args, "duration", 1.5), 1.5)
        started = step.args.get("_started_at")
        if started is None:
            step.args["_started_at"] = float(now)
            started = now
        elapsed = max(0.0, float(now) - _num(started, now))
        _set_state(agent, "observe")
        if elapsed >= duration:
            return StepResult.ok()
        return StepResult.moving(Decision(0, False, False))

    def _verb_socialize(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：走向最近个体社交。"""
        other = self._nearest_other(world, agent)
        if other is None:
            return StepResult.fail("no_social_target")
        delta = _agent_x(other) - _agent_x(agent)
        if abs(delta) > 55.0:
            direction = 1 if delta > 0 else -1
            _set_facing(agent, direction)
            _set_state(agent, "social")
            return StepResult.moving(Decision(direction, False, False))
        _set_facing(agent, 1 if delta >= 0 else -1)
        _set_state(agent, "social")
        word = self.rng.choice(("你好", "嗯", "一起", "看", "走"))
        return StepResult.moving(Decision(0, False, True, word))

    def _verb_reproduce(self, world, agent, goal, plan, step, dt, now) -> StepResult:
        """执行器：求偶/生育尝试。"""
        hook = _get(world, "try_reproduce", None) or _get(world, "reproduce", None)
        if not callable(hook):
            # TODO(Lead/P3)：提供 world.try_reproduce(agent) -> bool
            return StepResult.fail("no_reproduce_hook")
        try:
            if bool(hook(agent)):
                self._set_goal_progress(agent, goal, 1.0)
                _set_state(agent, "social")
                return StepResult.ok()
        except Exception:
            pass
        return StepResult.fail("reproduce_rejected")
