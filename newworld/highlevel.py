"""高层目标选择策略（REINFORCE）——真正的策略梯度，不是行为克隆。

设计意图
========
项目原有的神经网络（``neural.py``）是**监督式行为克隆**：用手写教师的标签训练
24→64→32→58 的 MLP，没有奖励、没有回报、没有策略梯度。审计实测它只参与约
0.3% 的逐帧决策，主策略仍是规则。本轮补上缺失的高层决策层：

* **状态**：不是像素/位置，而是个体与文明的宏观状态（能量、水分、库存、已
  验证技术、公共设施、人口、目标成败、记忆、昼夜……）。
* **动作**：8 个**目标类型**（``GOAL_ACTIONS``），每个目标持续几十到上千帧，
  决策频率远低于逐帧动作——这才是真正的"高层策略"。
* **奖励**：由集成层按**世界真实变化**计算（生存时间、技术、设施、资源、知识、
  合作、目标失败、死亡……），本模块不塑造奖励、不使用任何行为克隆标签。
* **学习**：采样目标 → 目标结束时取回 ``(features, action, reward)`` → 用折扣
  回报减基线得到 advantage → numpy 手写反向传播做 REINFORCE 更新。

集成边界（冻结契约见 ``docs/upgrade_contracts_v1.md`` §2）
---------------------------------------------------------
本模块**只**依赖 numpy + 标准库，不 import ``world`` / ``population`` /
``simulation``（避免循环依赖）。与集成层（Lead）的约定：

* ``population.attach_goal_policy(policy)`` 之后，``GoalManager`` 每次需要新目
  标时调用 ``policy.sample(extract_features(world, agent, population))``。本模块的
  :meth:`GoalPolicy.sample` 的 ``rng`` 参数是**可选**的，所以
  ``policy.sample(features)`` 与 ``policy.sample(features, rng)`` 两种写法都成立。
* 目标结束时 ``GoalManager`` 把 ``(features, action_index, reward)`` 追加到
  transitions；``population.drain_policy_transitions()`` 返回并按时间顺序给出它们。
* ``simulation.make_goal_policy_env(seed, ...)`` 返回带
  ``.population / .step() / .ended() / .result()`` 的训练环境；
  :class:`GoalPolicyTrainer` 只消费这四个接口，不碰世界内部。
* ``GOAL_ACTIONS`` 的顺序就是 action_index 的唯一定义，集成层的
  action_index → ``goal_type`` 映射必须与之一致。

特征（``FEATURE_NAMES``，30 维）
--------------------------------
契约要求至少 16 维且必须含：能量、水分、库存、已验证技术数、活跃设施数、存活
人口、最近目标成败、记忆摘要、昼夜。本模块 30 维全部归一化到有限值（多数
``[0, 1]``），缺失字段一律降级为 0 而不是抛错，因此对 fake world 也可用：

===  ===================== ==================================================
#    名称                   含义 / 归一化
===  ===================== ==================================================
0    energy                能量 / 100
1    hydration             水分 / 100
2    sleep_pressure        困意 / 100
3    alertness             警觉 / 100
4    wood                  木头 / 库存容量
5    stone                 石头 / 库存容量
6    fiber                 纤维 / 库存容量
7    cord                  绳索 / 库存容量
8    inventory_fill        库存占用 / 容量
9    tool                  是否有工具（1/0）
10   basket                背篓数 / 3
11   tech_verified         文明已验证技术数 / 技术总数
12   tech_known            个体掌握且已验证的技术数 / 技术总数
13   tech_depth            已验证技术依赖图深度 / 6
14   facility_active       活跃公共设施数 / 5
15   facility_condition    公共设施平均完整度
16   population_alive      存活人口 / 创建上限
17   population_created    历史累计创建 / 创建上限
18   population_deaths     历史死亡 / 创建上限
19   goal_success          最近 8 个已结算目标的成功率
20   goal_failure          最近 8 个已结算目标的失败/放弃率
21   memory_recent_success 记忆摘要：近期成功率
22   memory_goal_success   记忆摘要：目标类记忆成功率
23   memory_diversity      记忆摘要：动作多样性
24   memory_retention      记忆保留率
25   daylight              日照强度 0..1（昼夜）
26   hour_sin              时刻 sin
27   hour_cos              时刻 cos
28   x_position            个体在可走范围内的位置
29   age                   年龄 / 寿命
===  ===================== ==================================================

注意：**不**从这些特征里读出"当前目标类型"之类的策略自身输出，避免自反馈。

训练流程（真实策略梯度）
------------------------
1. :meth:`GoalPolicyTrainer.run_episode` 驱动 env 推进世界，每帧后用
   ``drain_policy_transitions()`` 取走目标级转移（避免缓冲区无限增长）。
2. 回合结束后按时间顺序计算折扣回报 ``G_t = Σ γ^k r_{t+k}``，advantage 为
   ``G_t - baseline``（基线是回报的指数滑动平均，即最简单的 REINFORCE 基线）。
3. 对每个转移用 :meth:`GoalPolicy.loss_and_grads` 求 ``-advantage·log p(a|s)``
   的梯度（numpy 手写反向传播），累加后 ``apply_grads`` 一次，完成一次策略
   梯度更新。
4. ``train`` 周期性用**独立种子**评估（评估不更新参数、不参与选择训练种子），
   ``evaluate`` 返回逐 seed 汇总。

**不得**声称"已训练出智能策略"：本模块只保证学习机制正确（单测验证机制），
最终效果由 Lead 的实验报告给出。训练预算有限（几分钟、数千步）时，策略很可能
只学到很弱的相关性，甚至没有可测提升——这是真实的、必须如实报告的结果。
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 冻结的动作空间 / 特征表
# ---------------------------------------------------------------------------
#: 高层目标动作（顺序即 action_index；不含 goals.GOAL_TYPES 里的 reproduce）
GOAL_ACTIONS: tuple[str, ...] = (
    "survive",
    "gather",
    "craft",
    "experiment",
    "build",
    "maintain",
    "explore",
    "socialize",
)

#: 30 维特征名（见模块 docstring 表）
FEATURE_NAMES: tuple[str, ...] = (
    "energy",
    "hydration",
    "sleep_pressure",
    "alertness",
    "wood",
    "stone",
    "fiber",
    "cord",
    "inventory_fill",
    "tool",
    "basket",
    "tech_verified",
    "tech_known",
    "tech_depth",
    "facility_active",
    "facility_condition",
    "population_alive",
    "population_created",
    "population_deaths",
    "goal_success",
    "goal_failure",
    "memory_recent_success",
    "memory_goal_success",
    "memory_diversity",
    "memory_retention",
    "daylight",
    "hour_sin",
    "hour_cos",
    "x_position",
    "age",
)

#: 特征维度（由名称表推导，保证 ``FEATURE_DIM == len(FEATURE_NAMES)``）
FEATURE_DIM: int = len(FEATURE_NAMES)

#: 归一化用的世界规则常量（缺失世界字段时的保守默认值）
MAX_FACILITIES = 5.0
MAX_TECH_DEPTH = 6.0
TECH_TOTAL_FALLBACK = 6.0
MAX_BASKETS = 3.0
DEFAULT_INVENTORY_CAPACITY = 8.0
DEFAULT_MAX_AGENTS = 8.0
DEFAULT_LIFESPAN_S = 900.0
DEFAULT_DAY_LENGTH_S = 600.0
DEFAULT_START_HOUR = 8.0

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 防御式读取工具：任何异常/缺失都返回默认值，绝不把异常抛给集成层
# ---------------------------------------------------------------------------
def _get(obj, name: str, default=None):
    """``getattr`` 的容错版本：任何异常都返回 default。"""
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _opt_num(value) -> float | None:
    """把任意值转成有限 float；无法转换或非有限返回 None。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _num(value, default: float = 0.0) -> float:
    """把任意值安全转成有限 float（失败/非有限 → default）。"""
    result = _opt_num(value)
    return float(default) if result is None else result


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _mean(values) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def _inventory(agent):
    return _get(agent, "inventory", None)


def _inventory_number(agent, kind: str) -> float:
    """读库存数量：兼容 ``Inventory`` 对象与 dict，负数按 0 处理。"""
    inventory = _inventory(agent)
    if inventory is None:
        return 0.0
    try:
        if isinstance(inventory, dict):
            value = inventory.get(kind, 0.0)
        else:
            value = _get(inventory, kind, 0.0)
    except Exception:
        return 0.0
    return max(0.0, _num(value))


def _inventory_capacity(agent) -> float:
    inventory = _inventory(agent)
    capacity = _get(inventory, "capacity", None)
    if callable(capacity):
        try:
            value = _opt_num(capacity())
        except Exception:
            value = None
        if value is not None and value > 0.0:
            return value
    value = _opt_num(_get(inventory, "capacity_base", None))
    if value is not None and value > 0.0:
        return value
    return DEFAULT_INVENTORY_CAPACITY


def _inventory_load(agent) -> float:
    return sum(
        _inventory_number(agent, kind) for kind in ("wood", "stone", "fiber", "cord")
    )


def _knowledge_store(world, population):
    store = _get(world, "knowledge", None)
    if store is None:
        store = _get(population, "knowledge", None)
    return store


def _tech_total(store) -> float:
    """技术总数：优先 ``store.all()``，缺失时用内置目录规模 6。"""
    lister = _get(store, "all", None)
    if callable(lister):
        try:
            items = lister()
            if items is not None:
                count = len(list(items))
                if count > 0:
                    return float(count)
        except Exception:
            pass
    catalog = _get(store, "catalog", None)
    try:
        if catalog is not None:
            return float(max(1, len(catalog)))
    except Exception:
        pass
    return TECH_TOTAL_FALLBACK


def _tech_verified_count(store) -> float:
    getter = _get(store, "verified_ids", None)
    if not callable(getter):
        return 0.0
    try:
        return float(len(list(getter() or ())))
    except Exception:
        return 0.0


def _tech_depth(store) -> float:
    depth = _get(store, "depth", None)
    if not callable(depth):
        return 0.0
    try:
        return float(_num(depth()))
    except Exception:
        return 0.0


def _agent_known_verified(store, agent) -> float:
    """个体掌握且**已验证**的技术数（只有可用的技术才算数）。"""
    known = _get(agent, "known_knowledge_ids", None) or ()
    getter = _get(store, "get", None)
    count = 0
    try:
        identifiers = list(known)
    except TypeError:
        return 0.0
    for identifier in identifiers:
        item = None
        if callable(getter):
            try:
                item = getter(str(identifier))
            except Exception:
                item = None
        if item is not None and str(_get(item, "status", "")) == "verified":
            count += 1
    return float(count)


def _civilization(world, population):
    civilization = _get(world, "civilization", None)
    if civilization is None:
        civilization = _get(population, "civilization", None)
    return civilization


def _facility_stats(civilization) -> tuple[float, float]:
    """返回 ``(活跃设施数, 平均完整度)``；无设施时均为 0。"""
    facilities = _get(civilization, "facilities", None) or ()
    try:
        items = list(facilities)
    except TypeError:
        return 0.0, 0.0
    active = 0
    conditions: list[float] = []
    for facility in items:
        if bool(_get(facility, "active", False)):
            active += 1
        conditions.append(_clip01(_num(_get(facility, "condition", 0.0))))
    return float(active), _mean(conditions)


def _world_agents(world, population) -> list:
    agents = _get(world, "agents", None)
    if agents is None:
        agents = _get(_get(population, "world", None), "agents", None)
    try:
        return list(agents or ())
    except TypeError:
        return []


def _population_stats(world, population) -> tuple[float, float, float, float]:
    """返回 ``(存活人口, 累计创建, 历史死亡, 创建上限)``。"""
    agents = _world_agents(world, population)
    alive = float(sum(1 for agent in agents if bool(_get(agent, "alive", True))))
    registry = _get(world, "registry", None)
    if registry is None:
        registry = _get(population, "registry", None)
    created = max(0.0, _num(_get(registry, "total_created", 0.0)))
    deaths = max(0.0, _num(_get(registry, "total_deaths", 0.0)))
    civ_config = _get(world, "civ_config", None)
    maximum = _opt_num(_get(civ_config, "max_agents_created", None))
    if maximum is None or maximum <= 0.0:
        maximum = DEFAULT_MAX_AGENTS
    return alive, created, deaths, maximum


def _goal_stats(world, population, agent) -> tuple[float, float, float]:
    """返回 ``(最近目标成功率, 最近目标失败率, 当前目标进度)``。"""
    manager = _get(population, "goals", None)
    if manager is None:
        manager = _get(world, "goal_manager", None)
    goals = []
    lister = _get(manager, "goals_of", None)
    if callable(lister):
        try:
            goals = list(lister(int(_num(_get(agent, "agent_id", 0.0)))) or ())
        except Exception:
            goals = []
    settled = [
        goal
        for goal in goals
        if str(_get(goal, "status", "")) in ("completed", "failed", "abandoned")
    ][-8:]
    success = _mean(
        1.0 if str(_get(goal, "status", "")) == "completed" else 0.0
        for goal in settled
    )
    failure = _mean(
        1.0 if str(_get(goal, "status", "")) in ("failed", "abandoned") else 0.0
        for goal in settled
    )
    progress = max(
        (
            _clip01(_num(_get(goal, "progress", 0.0)))
            for goal in goals
            if str(_get(goal, "status", "")) == "active"
        ),
        default=0.0,
    )
    return success, failure, progress


def _memory_stats(world, population, agent) -> tuple[float, float, float, float]:
    """返回记忆摘要 ``(近期成功, 目标类成功, 动作多样性, 保留率)``。"""
    store = _get(world, "memory", None)
    if store is None:
        store = _get(population, "memory", None)
    if store is None:
        return 0.0, 0.0, 0.0, 0.0
    agent_id = int(_num(_get(agent, "agent_id", 0.0)))
    recent_success = goal_success = diversity = 0.0
    vector_fn = _get(store, "feature_vector", None)
    if callable(vector_fn):
        try:
            vector = np.asarray(vector_fn(agent_id), dtype=np.float64).ravel()
        except Exception:
            vector = np.zeros(0, dtype=np.float64)
        if vector.size >= 8 and np.all(np.isfinite(vector)):
            recent_success = _clip01(float(vector[0]))
            diversity = _clip01(float(vector[3]))
            goal_success = _clip01(float(vector[6]))
    retention_fn = _get(store, "retention", None)
    retention = 0.0
    if callable(retention_fn):
        try:
            retention = _clip01(_num(retention_fn(agent_id)))
        except Exception:
            retention = 0.0
    return recent_success, goal_success, diversity, retention


def _day_stats(world) -> tuple[float, float]:
    """返回 ``(daylight, hour)``，兼容缺失的环境字段。"""
    hour = _opt_num(_get(world, "hour", None))
    if hour is None:
        day = _get(world, "day", None)
        hour_fn = _get(day, "hour", None)
        time_s = _num(_get(world, "t", 0.0))
        if callable(hour_fn):
            try:
                hour = _opt_num(hour_fn(time_s))
            except Exception:
                hour = None
        if hour is None:
            day_length = _opt_num(_get(day, "day_length_s", None)) or DEFAULT_DAY_LENGTH_S
            hour = DEFAULT_START_HOUR + time_s / day_length * 24.0
    hour = float(hour) % 24.0
    daylight = _opt_num(_get(world, "daylight", None))
    if daylight is None:
        day = _get(world, "day", None)
        daylight_fn = _get(day, "daylight", None)
        if callable(daylight_fn):
            try:
                daylight = _opt_num(daylight_fn(hour))
            except Exception:
                daylight = None
        if daylight is None:
            daylight = 0.0
    return _clip01(float(daylight)), hour


def _x_position(world, agent) -> float:
    x = _num(_get(agent, "x", 0.0))
    config = _get(world, "config", None)
    min_x = _num(_get(config, "min_x", 0.0))
    max_x = _num(_get(config, "max_x", 0.0))
    span = max_x - min_x
    if span <= 0.0:
        return 0.5
    return _clip01((x - min_x) / span)


def _age_norm(world, agent) -> float:
    age = max(0.0, _num(_get(agent, "age", 0.0)))
    lifespan = _opt_num(_get(_get(world, "civ_config", None), "lifespan_s", None))
    if lifespan is None or lifespan <= 0.0:
        lifespan = DEFAULT_LIFESPAN_S
    return _clip01(age / lifespan)


# ---------------------------------------------------------------------------
# 特征提取：真实 World/agent/population -> 归一化特征
# ---------------------------------------------------------------------------
def extract_features(world, agent, population=None) -> np.ndarray:
    """把真实世界状态编码成 ``(FEATURE_DIM,)`` 的 float64 特征向量。

    参数
    ----
    world:
        真实 :class:`~newworld.world.World`（也接受 duck-typed fake world）。
    agent:
        目标个体（``physiology / inventory / known_knowledge_ids / skills``…）。
    population:
        可选的 :class:`~newworld.population.Population`；缺失时尽量从 world 上
        探测 ``goals / memory / civilization / registry``。

    返回
    ----
    np.ndarray
        ``(FEATURE_DIM,)``、``float64``、**全部有限**；缺失字段降级为 0 而
        不抛错（fake world 也因此可用）。同一世界状态必然得到同一向量。
    """
    # -- 生理 ----------------------------------------------------------
    physiology = _get(agent, "physiology", None)
    energy = _clip01(_num(_get(physiology, "energy", 0.0)) / 100.0)
    hydration = _clip01(_num(_get(physiology, "hydration", 0.0)) / 100.0)
    sleep_pressure = _clip01(_num(_get(physiology, "sleep_pressure", 0.0)) / 100.0)
    alertness = _clip01(_num(_get(physiology, "alertness", 0.0)) / 100.0)

    # -- 库存 ----------------------------------------------------------
    capacity = max(1.0, _inventory_capacity(agent))
    wood = _clip01(_inventory_number(agent, "wood") / capacity)
    stone = _clip01(_inventory_number(agent, "stone") / capacity)
    fiber = _clip01(_inventory_number(agent, "fiber") / capacity)
    cord = _clip01(_inventory_number(agent, "cord") / capacity)
    load = _clip01(_inventory_load(agent) / capacity)
    tool = 1.0 if str(_get(_inventory(agent), "tool", "") or "") else 0.0
    basket = _clip01(_inventory_number(agent, "basket") / MAX_BASKETS)

    # -- 知识 / 技术 ---------------------------------------------------
    store = _knowledge_store(world, population)
    total_tech = max(1.0, _tech_total(store))
    verified = _tech_verified_count(store)
    known_verified = _agent_known_verified(store, agent)
    depth = _tech_depth(store)

    # -- 公共设施 ------------------------------------------------------
    civilization = _civilization(world, population)
    active_facilities, facility_condition = _facility_stats(civilization)

    # -- 人口 / 文明 ---------------------------------------------------
    alive, created, deaths, max_created = _population_stats(world, population)

    # -- 目标 / 记忆 ---------------------------------------------------
    goal_success, goal_failure, _progress = _goal_stats(world, population, agent)
    memory_success, memory_goal_success, memory_diversity, memory_retention = (
        _memory_stats(world, population, agent)
    )

    # -- 昼夜 / 位置 / 年龄 --------------------------------------------
    daylight, hour = _day_stats(world)
    hour_rad = 2.0 * math.pi * hour / 24.0
    x_position = _x_position(world, agent)
    age = _age_norm(world, agent)

    vector = np.asarray(
        [
            energy,
            hydration,
            sleep_pressure,
            alertness,
            wood,
            stone,
            fiber,
            cord,
            load,
            tool,
            basket,
            _clip01(verified / total_tech),
            _clip01(known_verified / total_tech),
            _clip01(depth / MAX_TECH_DEPTH),
            _clip01(active_facilities / MAX_FACILITIES),
            facility_condition,
            _clip01(alive / max_created),
            _clip01(created / max_created),
            _clip01(deaths / max_created),
            goal_success,
            goal_failure,
            memory_success,
            memory_goal_success,
            memory_diversity,
            memory_retention,
            daylight,
            math.sin(hour_rad),
            math.cos(hour_rad),
            x_position,
            age,
        ],
        dtype=np.float64,
    )
    if vector.shape != (FEATURE_DIM,):  # pragma: no cover - 结构性保护
        raise RuntimeError(
            f"feature vector shape {vector.shape} != ({FEATURE_DIM},)"
        )
    np.nan_to_num(vector, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(vector, -1e6, 1e6, out=vector)
    return vector


# ---------------------------------------------------------------------------
# 策略网络：MLP(FEATURE_DIM -> HIDDEN -> len(GOAL_ACTIONS))，numpy 手写
# ---------------------------------------------------------------------------
class GoalPolicy:
    """目标选择策略网络：两层 MLP，softmax 输出 8 个目标类型的概率。

    前向 / 反向全部用 numpy 手写（无自动微分、无 torch）：

    * 前向：``h = tanh(x W1 + b1)``，``logits = h W2 + b2``，
      ``p = softmax(logits)``；
    * 损失（REINFORCE）：``loss = -advantage * log p[a]``；
    * 反向：``dlogits = advantage * (p - onehot(a))``，再按链式法则回传。

    参数
    ----
    seed:
        参数初始化与内部采样用的随机种子（可复现）。
    """

    HIDDEN: int = 32

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / math.sqrt(float(FEATURE_DIM))
        #: 隐层权重 / 偏置
        self.W1 = rng.normal(0.0, scale, (FEATURE_DIM, self.HIDDEN)).astype(np.float64)
        self.b1 = np.zeros(self.HIDDEN, dtype=np.float64)
        #: 输出层权重 / 偏置
        out_scale = 1.0 / math.sqrt(float(self.HIDDEN))
        self.W2 = rng.normal(0.0, out_scale, (self.HIDDEN, len(GOAL_ACTIONS))).astype(
            np.float64
        )
        self.b2 = np.zeros(len(GOAL_ACTIONS), dtype=np.float64)

    # -- 参数视图 ------------------------------------------------------
    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("W1", "b1", "W2", "b2")

    def parameters(self) -> dict[str, np.ndarray]:
        """返回参数名 → 数组的浅拷贝（用于测试/诊断，不应原地修改）。"""
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }

    def flat_weights(self) -> np.ndarray:
        """把全部参数拼成一维数组（便于比较"参数是否变化"）。"""
        return np.concatenate(
            [self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()]
        )

    # -- 前向 ----------------------------------------------------------
    @staticmethod
    def _as_features(features) -> np.ndarray:
        vector = np.asarray(features, dtype=np.float64).ravel()
        if vector.shape != (FEATURE_DIM,):
            raise ValueError(
                f"features 维度必须为 ({FEATURE_DIM},)，得到 {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("features 含非有限值（NaN/inf）")
        return vector

    def _forward(self, features) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 ``(hidden, logits, probabilities)``。"""
        x = self._as_features(features)
        hidden = np.tanh(x @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        total = float(np.sum(exp))
        if not math.isfinite(total) or total <= 0.0:  # pragma: no cover
            probabilities = np.full_like(logits, 1.0 / float(logits.size))
        else:
            probabilities = exp / total
        return hidden, logits, probabilities

    def probabilities(self, features) -> np.ndarray:
        """``p(a|s)``：长度 ``len(GOAL_ACTIONS)``、和为 1、全有限。"""
        _, _, probs = self._forward(features)
        return probs

    # -- 采样 ----------------------------------------------------------
    def sample(self, features, rng=None) -> int:
        """按策略分布采样一个 action_index。

        ``rng`` 可为 ``np.random.Generator`` / ``random.Random`` / ``None``；
        ``None`` 时用策略内部的 RNG（因此 ``policy.sample(features)`` 也成立）。
        """
        probabilities = self.probabilities(features)
        generator = self._rng if rng is None else rng
        draw_fn = _get(generator, "random", None)
        if not callable(draw_fn):
            raise TypeError("rng 必须提供 random() 方法")
        try:
            draw = float(draw_fn())
        except Exception as exc:  # pragma: no cover - 非法 rng
            raise TypeError(f"rng.random() 调用失败: {exc}") from exc
        if not math.isfinite(draw):
            draw = 0.0
        draw = min(max(draw, 0.0), 1.0 - _EPS)
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += float(probability)
            if draw < cumulative:
                return int(index)
        return int(len(probabilities) - 1)

    def distribution(self) -> dict[str, float]:
        """当前参数下"零特征"的目标分布（诊断用，不是学习结果）。"""
        probabilities = self.probabilities(np.zeros(FEATURE_DIM, dtype=np.float64))
        return {
            action: float(probability)
            for action, probability in zip(GOAL_ACTIONS, probabilities)
        }

    @staticmethod
    def action_name(index: int) -> str:
        """action_index → 目标类型名；越界抛 ``ValueError``。"""
        try:
            return GOAL_ACTIONS[int(index)]
        except (IndexError, TypeError, ValueError):
            raise ValueError(
                f"action_index 必须在 0..{len(GOAL_ACTIONS) - 1}，得到 {index!r}"
            ) from None

    # -- 反向：手写策略梯度 --------------------------------------------
    def loss_and_grads(
        self,
        features,
        action_index: int,
        advantage: float,
    ) -> tuple[float, dict[str, np.ndarray]]:
        """计算 ``-advantage * log p(a|s)`` 及其对全部参数的梯度。

        这是 REINFORCE 的损失（advantage 由 trainer 用折扣回报减基线得到）。
        返回 ``(loss, grads)``，``grads`` 的键为 ``W1/b1/W2/b2``。
        """
        index = int(action_index)
        if index < 0 or index >= len(GOAL_ACTIONS):
            raise ValueError(
                f"action_index 必须在 0..{len(GOAL_ACTIONS) - 1}，得到 {index}"
            )
        advantage = float(advantage)
        if not math.isfinite(advantage):
            raise ValueError("advantage 必须是有限数值")

        x = self._as_features(features)
        hidden, _logits, probabilities = self._forward(x)
        probability = max(float(probabilities[index]), _EPS)
        loss = -advantage * math.log(probability)

        # d loss / d logits = advantage * (p - onehot(a))
        dlogits = advantage * probabilities
        dlogits[index] -= advantage

        # 输出层
        dW2 = np.outer(hidden, dlogits)
        db2 = dlogits
        # 隐层（tanh 导数）
        dhidden = dlogits @ self.W2.T
        dpre = dhidden * (1.0 - hidden * hidden)
        dW1 = np.outer(x, dpre)
        db1 = dpre

        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }
        return float(loss), grads

    def apply_grads(self, grads, lr: float) -> None:
        """梯度下降一步：``param -= lr * grad``（grads 缺失的键跳过）。"""
        step = float(lr)
        if not math.isfinite(step) or step < 0.0:
            raise ValueError("lr 必须是有限非负数")
        if step == 0.0:
            return
        for name, parameter in (
            ("W1", self.W1),
            ("b1", self.b1),
            ("W2", self.W2),
            ("b2", self.b2),
        ):
            if isinstance(grads, dict):
                grad = grads.get(name, None)
            else:
                grad = _get(grads, name, None)
            if grad is None:
                continue
            array = np.asarray(grad, dtype=np.float64)
            if array.shape != parameter.shape:
                raise ValueError(
                    f"梯度 {name} 形状 {array.shape} != 参数形状 {parameter.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"梯度 {name} 含非有限值")
            parameter -= step * array

    # -- 序列化：npz ---------------------------------------------------
    @staticmethod
    def _normalize_path(path) -> Path:
        resolved = Path(path)
        if resolved.suffix.lower() != ".npz":
            if resolved.suffix:
                resolved = resolved.with_name(resolved.name + ".npz")
            else:
                resolved = resolved.with_name(resolved.name + ".npz")
        return resolved

    def save(self, path) -> Path:
        """保存为 ``.npz`` checkpoint（缺失 ``.npz`` 后缀时自动补上）。"""
        target = self._normalize_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            feature_dim=np.int64(FEATURE_DIM),
            hidden=np.int64(self.HIDDEN),
            seed=np.int64(self.seed),
            feature_names=np.asarray(FEATURE_NAMES, dtype=str),
            actions=np.asarray(GOAL_ACTIONS, dtype=str),
        )
        return target

    @classmethod
    def load(cls, path) -> "GoalPolicy":
        """从 ``.npz`` checkpoint 还原；维度不匹配抛 ``ValueError``。"""
        source = Path(path)
        if not source.exists() and cls._normalize_path(source).exists():
            source = cls._normalize_path(source)
        if not source.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {source}")
        with np.load(source, allow_pickle=False) as data:
            required = {"W1", "b1", "W2", "b2"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"checkpoint 缺少参数: {sorted(missing)}")
            feature_dim = int(np.asarray(data["feature_dim"]).reshape(-1)[0])
            hidden = int(np.asarray(data["hidden"]).reshape(-1)[0])
            if feature_dim != FEATURE_DIM:
                raise ValueError(
                    f"checkpoint 特征维度 {feature_dim} != 当前 {FEATURE_DIM}"
                )
            if hidden != cls.HIDDEN:
                raise ValueError(f"checkpoint 隐层维度 {hidden} != 当前 {cls.HIDDEN}")
            seed = int(np.asarray(data["seed"]).reshape(-1)[0])
            policy = cls(seed=seed)
            for name, expected_shape in (
                ("W1", (FEATURE_DIM, cls.HIDDEN)),
                ("b1", (cls.HIDDEN,)),
                ("W2", (cls.HIDDEN, len(GOAL_ACTIONS))),
                ("b2", (len(GOAL_ACTIONS),)),
            ):
                array = np.asarray(data[name], dtype=np.float64)
                if array.shape != expected_shape:
                    raise ValueError(
                        f"checkpoint 参数 {name} 形状 {array.shape} != {expected_shape}"
                    )
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"checkpoint 参数 {name} 含非有限值")
                setattr(policy, name, array)
        return policy

    # -- 序列化：dict --------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """导出为可 JSON 序列化的 dict（权重转 list）。"""
        return {
            "format": "newworld.goal_policy/v1",
            "feature_dim": int(FEATURE_DIM),
            "hidden": int(self.HIDDEN),
            "seed": int(self.seed),
            "actions": list(GOAL_ACTIONS),
            "feature_names": list(FEATURE_NAMES),
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoalPolicy":
        """从 :meth:`to_dict` 的结果还原；维度不匹配抛 ``ValueError``。"""
        if not isinstance(data, dict):
            raise TypeError("GoalPolicy.from_dict 需要 dict")
        feature_dim = int(data.get("feature_dim", FEATURE_DIM))
        hidden = int(data.get("hidden", cls.HIDDEN))
        if feature_dim != FEATURE_DIM:
            raise ValueError(f"dict 特征维度 {feature_dim} != 当前 {FEATURE_DIM}")
        if hidden != cls.HIDDEN:
            raise ValueError(f"dict 隐层维度 {hidden} != 当前 {cls.HIDDEN}")
        actions = tuple(data.get("actions", GOAL_ACTIONS))
        if actions != GOAL_ACTIONS:
            raise ValueError(f"dict 动作顺序不匹配: {actions!r}")
        policy = cls(seed=int(data.get("seed", 0)))
        for name, expected_shape in (
            ("W1", (FEATURE_DIM, cls.HIDDEN)),
            ("b1", (cls.HIDDEN,)),
            ("W2", (cls.HIDDEN, len(GOAL_ACTIONS))),
            ("b2", (len(GOAL_ACTIONS),)),
        ):
            array = np.asarray(data.get(name), dtype=np.float64)
            if array.shape != expected_shape:
                raise ValueError(
                    f"dict 参数 {name} 形状 {array.shape} != {expected_shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"dict 参数 {name} 含非有限值")
            setattr(policy, name, array)
        return policy


# ---------------------------------------------------------------------------
# 训练器：REINFORCE + 基线 + 折扣
# ---------------------------------------------------------------------------
class GoalPolicyTrainer:
    """用真实的 ``(features, action, reward)`` 转移训练 :class:`GoalPolicy`。

    奖励**不是**本类计算的：集成层按世界真实变化算出每个目标的奖励，本类只做
    折扣回报、基线、策略梯度更新与评估。

    参数
    ----
    policy:
        :class:`GoalPolicy` 实例。
    env_factory:
        可调用对象，``env_factory(seed, steps=..., dt=...)`` 返回训练环境
        （契约 §2.2 的 ``simulation.make_goal_policy_env``）。若它的签名不接
        受关键字参数，会自动回退成 ``env_factory(seed)``。
    seed, lr, gamma, baseline_decay:
        采样种子、学习率、折扣因子、基线指数滑动平均衰减系数。
    """

    def __init__(
        self,
        policy: GoalPolicy,
        env_factory,
        *,
        seed: int = 0,
        lr: float = 0.05,
        gamma: float = 0.95,
        baseline_decay: float = 0.9,
    ) -> None:
        if not callable(env_factory):
            raise TypeError("env_factory 必须可调用")
        for method in ("probabilities", "sample", "loss_and_grads", "apply_grads"):
            if not callable(_get(policy, method, None)):
                raise TypeError(f"policy 缺少方法 {method}")
        lr = float(lr)
        gamma = float(gamma)
        baseline_decay = float(baseline_decay)
        if not math.isfinite(lr) or lr < 0.0:
            raise ValueError("lr 必须是有限非负数")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma 必须在 [0, 1]")
        if not 0.0 <= baseline_decay < 1.0:
            raise ValueError("baseline_decay 必须在 [0, 1)")
        self.policy = policy
        self.env_factory = env_factory
        self.seed = int(seed)
        self.lr = lr
        self.gamma = gamma
        self.baseline_decay = baseline_decay
        self.rng = np.random.default_rng(self.seed)
        #: 回报的指数滑动平均基线（最简单的 REINFORCE 基线）
        self._baseline = 0.0
        self._baseline_samples = 0
        self._updates = 0
        #: 每回合的训练记录 / 周期评估记录
        self.history: list[dict[str, float]] = []
        self.eval_history: list[dict[str, object]] = []
        #: 最近一次 evaluate 的逐 seed 明细（数值聚合之外的诊断信息）
        self.last_eval_details: list[dict[str, float]] = []

    # -- 环境 / 转移读取 ------------------------------------------------
    def _attach_policy(self, env) -> bool:
        """把当前策略挂到新环境上（契约 §2.2 的 ``attach_goal_policy``）。

        训练/评估环境由 ``make_goal_policy_env`` 构造，但"用哪一份策略"必须由
        trainer 决定：否则评估会用到环境自带的初始策略，训练出的参数根本不会
        被使用。因此这里显式把 ``self.policy`` 挂到 ``env.population``（也兼容
        挂在 env 本身的实现）。挂载失败不抛错（fake env 没有该钩子时跳过）。
        """
        population = _get(env, "population", None)
        candidates = (
            (population, "attach_goal_policy"),
            (env, "attach_goal_policy"),
            (env, "attach_policy"),
        )
        for owner, name in candidates:
            attach = _get(owner, name, None)
            if not callable(attach):
                continue
            try:
                attach(self.policy)
                return True
            except Exception:
                continue
        return False

    def _make_env(self, seed: int, steps: int, dt: float):
        """构造训练环境：优先带关键字参数，签名不支持时回退只传 seed。"""
        try:
            env = self.env_factory(int(seed), steps=int(steps), dt=float(dt))
        except TypeError:
            env = self.env_factory(int(seed))
        self._attach_policy(env)
        return env

    @staticmethod
    def _drain(env) -> list[tuple[np.ndarray, int, float]]:
        """从 ``env.population.drain_policy_transitions()`` 取走并规范化转移。"""
        population = _get(env, "population", None)
        drain = _get(population, "drain_policy_transitions", None)
        if not callable(drain):
            return []
        try:
            raw = drain() or []
        except Exception:
            return []
        transitions: list[tuple[np.ndarray, int, float]] = []
        for item in raw:
            try:
                features = np.asarray(item[0], dtype=np.float64).ravel()
                action = int(item[1])
                reward = float(item[2])
            except (TypeError, ValueError, IndexError):
                continue
            if features.shape != (FEATURE_DIM,):
                continue
            if not np.all(np.isfinite(features)) or not math.isfinite(reward):
                continue
            if action < 0 or action >= len(GOAL_ACTIONS):
                continue
            transitions.append((features, action, reward))
        return transitions

    @staticmethod
    def _ended(env) -> bool:
        ended = _get(env, "ended", None)
        if not callable(ended):
            return False
        try:
            return bool(ended())
        except Exception:
            return False

    @staticmethod
    def _env_metrics(env, metrics: dict[str, float]) -> None:
        """把 ``env.result()`` 里的数值型标量并进回合指标（不覆盖已有键）。"""
        result_fn = _get(env, "result", None)
        if not callable(result_fn):
            return
        try:
            result = result_fn()
        except Exception:
            return
        if not isinstance(result, dict):
            return
        for key, value in result.items():
            name = str(key)
            if name in metrics:
                continue
            number = _opt_num(value)
            if number is not None:
                metrics[name] = float(number)

    # -- 折扣回报 ------------------------------------------------------
    @staticmethod
    def discounted_returns(rewards, gamma: float) -> np.ndarray:
        """按时间顺序计算折扣回报 ``G_t = r_t + γ r_{t+1} + …``。"""
        values = [float(reward) for reward in rewards]
        returns = np.zeros(len(values), dtype=np.float64)
        running = 0.0
        for index in range(len(values) - 1, -1, -1):
            running = values[index] + float(gamma) * running
            returns[index] = running
        return returns

    # -- 单回合：采样 + 更新 -------------------------------------------
    def run_episode(self, *, seed: int, steps: int, dt: float = 0.05) -> dict[str, float]:
        """跑一个训练回合（采样并用该回合转移做一次策略梯度更新）。

        返回真实指标字典：``return``（奖励之和）、``steps``（实际世界步数）、
        ``transitions``（目标级转移数）、``mean_advantage``、``baseline``，以及
        ``env.result()`` 提供的数值型快照（``extinction_time`` /
        ``tech_verified`` 等，名字以集成层为准）。
        """
        budget = int(steps)
        if budget <= 0:
            raise ValueError("steps 必须是正整数")
        env = self._make_env(seed, budget, dt)
        transitions: list[tuple[np.ndarray, int, float]] = []
        step_count = 0
        for _ in range(budget):
            step_fn = _get(env, "step", None)
            if not callable(step_fn):
                raise TypeError("env 缺少 step() 方法")
            step_fn()
            step_count += 1
            transitions.extend(self._drain(env))
            if self._ended(env):
                break
        # 有些集成实现会在最后一帧之后才结算目标：再取一次，避免丢掉最后转移
        transitions.extend(self._drain(env))

        rewards = [transition[2] for transition in transitions]
        episode_return = float(sum(rewards))
        mean_advantage = 0.0
        if transitions:
            returns = self.discounted_returns(rewards, self.gamma)
            advantages = returns - self._baseline
            # 只做数值裁剪（防止极端奖励让一次更新炸掉），不做奖励塑形
            advantages = np.clip(advantages, -50.0, 50.0)
            mean_advantage = float(np.mean(advantages))
            grad_sum = {
                "W1": np.zeros_like(self.policy.W1),
                "b1": np.zeros_like(self.policy.b1),
                "W2": np.zeros_like(self.policy.W2),
                "b2": np.zeros_like(self.policy.b2),
            }
            for features, action, advantage in zip(
                [transition[0] for transition in transitions],
                [transition[1] for transition in transitions],
                advantages,
            ):
                _, grads = self.policy.loss_and_grads(
                    features, action, float(advantage)
                )
                for name in grad_sum:
                    grad_sum[name] += grads[name]
            scale = 1.0 / float(len(transitions))
            averaged = {name: value * scale for name, value in grad_sum.items()}
            self.policy.apply_grads(averaged, self.lr)
            self._updates += 1
            # 基线：按"平均回报"做指数滑动平均（首个样本直接初始化）
            baseline_target = float(np.mean(returns))
            if self._baseline_samples == 0:
                self._baseline = baseline_target
            else:
                self._baseline = (
                    self.baseline_decay * self._baseline
                    + (1.0 - self.baseline_decay) * baseline_target
                )
            self._baseline_samples += 1

        metrics: dict[str, float] = {
            "return": episode_return,
            "steps": float(step_count),
            "transitions": float(len(transitions)),
            "mean_advantage": mean_advantage,
            "baseline": float(self._baseline),
        }
        self._env_metrics(env, metrics)
        return metrics

    # -- 评估（不更新参数，用独立种子）--------------------------------
    def evaluate(
        self, *, seeds, steps: int, dt: float = 0.05
    ) -> dict[str, float]:
        """在给定种子上评估当前策略（不更新参数）。

        返回逐 seed 的**数值聚合**；逐 seed 明细见 ``last_eval_details``。
        """
        seed_list = [int(seed) for seed in seeds]
        if not seed_list:
            raise ValueError("seeds 不能为空")
        budget = int(steps)
        if budget <= 0:
            raise ValueError("steps 必须是正整数")
        runs: list[dict[str, float]] = []
        for seed in seed_list:
            env = self._make_env(seed, budget, dt)
            step_count = 0
            for _ in range(budget):
                step_fn = _get(env, "step", None)
                if not callable(step_fn):
                    raise TypeError("env 缺少 step() 方法")
                step_fn()
                step_count += 1
                if self._ended(env):
                    break
            # 评估回合同样读取真实奖励，但**不**更新参数
            transitions = self._drain(env)
            metrics: dict[str, float] = {
                "seed": float(seed),
                "return": float(sum(transition[2] for transition in transitions)),
                "steps": float(step_count),
                "transitions": float(len(transitions)),
            }
            self._env_metrics(env, metrics)
            runs.append(metrics)
        self.last_eval_details = runs

        returns = np.asarray([run["return"] for run in runs], dtype=np.float64)
        aggregate: dict[str, float] = {
            "episodes": float(len(runs)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "mean_steps": _mean(run["steps"] for run in runs),
            "total_transitions": float(sum(run["transitions"] for run in runs)),
        }
        # 其它 env.result() 指标按均值一并报告（只在全部 run 都提供时）
        common_keys = set(runs[0]) - {"seed", "return", "steps", "transitions"}
        for run in runs[1:]:
            common_keys &= set(run)
        for key in sorted(common_keys):
            aggregate[f"mean_{key}"] = _mean(run[key] for run in runs)
        return aggregate

    # -- 完整训练循环 --------------------------------------------------
    def train(
        self,
        *,
        episodes: int,
        steps: int,
        dt: float = 0.05,
        eval_episodes: int = 2,
        eval_steps: int | None = None,
        eval_every: int | None = None,
        eval_seeds: list[int] | None = None,
        out_path=None,
    ) -> dict[str, object]:
        """真实训练循环：采样 → 奖励 → 策略梯度更新 → 周期评估。

        参数
        ----
        episodes, steps:
            训练回合数与每回合世界步数预算。
        dt:
            世界步长（秒）。
        eval_episodes:
            周期性评估使用的**独立种子**数量（0 表示跳过评估）。
        eval_steps:
            评估回合步数（默认与训练相同）。
        eval_every:
            每隔多少回合做一次周期评估（默认 ``max(1, episodes // 4)``）。
        eval_seeds:
            可选的评估种子列表（调用方必须保证与训练种子不重叠）。给出时
            周期评估取前 ``eval_episodes`` 个；不给时自动生成
            ``seed+100000+7919*k``（仍与训练种子不重叠）。
        out_path:
            可选的 checkpoint 保存路径；训练结束保存最终策略。

        返回
        ----
        dict
            ``train``（逐回合曲线）、``train_returns``、``eval``（周期评估）、
            ``final_eval``（独立种子最终评估）、``config``、``policy``（最终
            零特征分布诊断）、``out``（保存路径或 None）。
        """
        episode_count = int(episodes)
        if episode_count <= 0:
            raise ValueError("episodes 必须是正整数")
        step_budget = int(steps)
        if step_budget <= 0:
            raise ValueError("steps 必须是正整数")
        eval_count = int(eval_episodes)
        if eval_count < 0:
            raise ValueError("eval_episodes 必须 >= 0")
        eval_budget = step_budget if eval_steps is None else int(eval_steps)
        if eval_count > 0 and eval_budget <= 0:
            raise ValueError("eval_steps 必须是正整数")
        every = max(1, episode_count // 4) if eval_every is None else int(eval_every)
        if every <= 0:
            raise ValueError("eval_every 必须是正整数")
        if eval_seeds is not None:
            provided = [int(seed) for seed in eval_seeds]
            if eval_count > 0 and len(provided) < eval_count:
                raise ValueError(
                    f"eval_seeds 提供 {len(provided)} 个，少于 eval_episodes={eval_count}"
                )
            eval_seed_list = provided[:eval_count]
        else:
            # 评估种子与训练种子**不重叠**：训练用 seed+101*i，评估用 seed+100000+7919*k
            eval_seed_list = [
                self.seed + 100_000 + 7919 * index for index in range(eval_count)
            ]
        self.history = []
        self.eval_history = []

        for episode in range(episode_count):
            train_seed = self.seed + 101 * episode
            metrics = self.run_episode(
                seed=train_seed, steps=step_budget, dt=float(dt)
            )
            metrics["episode"] = float(episode + 1)
            self.history.append(metrics)
            if eval_count > 0 and (
                (episode + 1) % every == 0 or (episode + 1) == episode_count
            ):
                evaluation = self.evaluate(
                    seeds=eval_seed_list, steps=eval_budget, dt=float(dt)
                )
                self.eval_history.append(
                    {"episode": float(episode + 1), "eval": evaluation}
                )

        final_eval: dict[str, float] = {}
        if eval_count > 0:
            final_eval = self.evaluate(
                seeds=eval_seed_list, steps=eval_budget, dt=float(dt)
            )
        saved = None
        if out_path is not None:
            saved = str(self.policy.save(out_path))
        return {
            "config": {
                "episodes": float(episode_count),
                "steps": float(step_budget),
                "dt": float(dt),
                "eval_episodes": float(eval_count),
                "eval_steps": float(eval_budget),
                "eval_every": float(every),
                "eval_seeds": [float(seed) for seed in eval_seed_list],
                "lr": float(self.lr),
                "gamma": float(self.gamma),
                "baseline_decay": float(self.baseline_decay),
                "seed": float(self.seed),
            },
            "train": [dict(record) for record in self.history],
            "train_returns": [float(record["return"]) for record in self.history],
            "train_transitions": [float(record["transitions"]) for record in self.history],
            "eval": [dict(record) for record in self.eval_history],
            "final_eval": dict(final_eval),
            "updates": float(self._updates),
            "baseline": float(self._baseline),
            "policy": self.policy.distribution(),
            "out": saved,
        }


# ---------------------------------------------------------------------------
# 便利函数：从 checkpoint 继续训练 / 加载
# ---------------------------------------------------------------------------
def load_policy(path) -> GoalPolicy:
    """从 checkpoint 加载策略（:meth:`GoalPolicy.load` 的短别名）。"""
    return GoalPolicy.load(path)


__all__ = [
    "GOAL_ACTIONS",
    "FEATURE_NAMES",
    "FEATURE_DIM",
    "extract_features",
    "GoalPolicy",
    "GoalPolicyTrainer",
    "load_policy",
]
