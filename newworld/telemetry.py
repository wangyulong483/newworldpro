"""Telemetry —— headless 实验的**只读**指标采集（冻结契约见设计文档 §8）。

本模块的唯一职责是**观测**：每个仿真步读一次世界/人口状态，累计计数，
最后给出可 JSON 序列化的指标汇总。它绝不写入 ``world`` / ``agent`` /
``population`` 的任何字段，只保留自己的快照与计数器（有
``tests/test_telemetry.py`` 的"不改世界"用例把守）。

依赖方向（§1）
-------------
``telemetry.py`` 只 import 标准库；**不 import** ``population`` / ``world`` /
``goals`` / ``knowledge`` / ``civilization``。所有可选能力都通过
``getattr(population, "goals", None)`` 这类鸭子类型读取，缺失或抛异常时
**优雅降级**到安全默认值（见下表）。这样 P0 可以在 Lead 集成之前独立使用，
集成之后自动开始统计更丰富的指标。

缺数据时的安全默认值
--------------------
=====================  ==========================================================
场景                   行为
=====================  ==========================================================
无 ``population``      ``goal_*`` / ``plan_cost_error`` / ``tech_*`` /
                       ``knowledge_retention`` / ``innovation_efficiency`` /
                       ``cooperation_benefit`` 用 0.0；``total_agents_created``
                       退化为"观测期内见过的 agent_id 数"
``goals`` 属性缺失      ``goals_*`` = 0，``plan_cost_error`` = 0.0
``knowledge`` 缺失      ``tech_*`` = 0；若有 ``civilization.innovation_count``
                       则用它作为 ``tech_verified``（该字段本来就同步自知识库）
``civilization`` 缺失   ``total_agents_created`` 用观测计数；
                       ``extinction_time`` 仍由存活数判断；
                       ``extinction_cause`` 退化为最后记录的死亡原因
``memory`` 缺失         ``memory_agents`` = 0（附加指标，不属于 §8 冻结表）
未发生扰动              ``recovery_after_disturbance`` = 1.0（视为无需恢复）
尚未开始                ``extinction_time`` = -1.0，其余比率 0.0
=====================  ==========================================================

指标定义与统计窗口
------------------
比率类指标（``survival_rate`` / ``action_entropy`` /
``no_progress_repeat_ratio`` / ``state_identical_ratio`` /
``goal_completion_rate`` / ``cooperation_benefit`` 等）是**窗口内/全程累积**
算出来的；水平类指标（``tech_verified`` 等）取最近一次采样值。
``window_summary()`` 给出**当前未结算窗口**（``window_s`` 世界秒）的指标；
``summary()`` 给出**全程**聚合。``step()`` 发现窗口累积时间已到
``window_s`` 时会自动结算并把结果追加到 ``self.windows``（``list[dict]``，
供 ``CivilizationState.history`` 使用）。

两个"停滞"口径（**必须区分，不可混用**）
----------------------------------------
Lead 的 P0 基线探针实测发现：若把"无进展"定义为
``(x, energy, hydration, 库存, …)`` 快照完全相等，则因为能量/水分每步都在
连续下降，该指标**恒为 0**（探针得到 ``state_change_step_ratio = 1.0``），
毫无诊断价值。因此本模块同时记录两个口径：

``state_identical_ratio``（严格对照口径）
    相邻两步 agent 快照**逐项完全相等**的 agent-步比例。快照含
    ``(x, y, energy, hydration, 库存, 工具, behavior_state, 目标计划进度)``，
    其中能量/水分保留 2 位小数。生理量连续衰减 ⇒ 该值通常接近 0，**只能**
    作为"状态是否逐比特不变"的对照，不能当作"有进展"的证据。

``no_progress_repeat_ratio``（有意义的进展口径，主指标）
    分子 = 本步**动作类别与上一步相同**、且以下"有意义的进展信号"**全部**
    未变化的 agent-步；分母 = 总 agent-步。进展信号：

    1. 库存数量变化（wood/stone/tool 以及未来扩展材料的任意字段）；
    2. 知识状态变化（``known_knowledge_ids`` 并集大小或已验证知识数变化）；
    3. 公共设施/社区进度变化（``community.wood/stone/structures``、篝火、
       活跃设施、活跃契约）；
    4. 目标进度变化（该 agent 的活跃计划 ``(plan_id, current, version)``，
       或全局目标 生成/完成/失败/放弃 计数变化）；
    5. 位置位移超过阈值（``0.5 × world.config.speed × dt``，即"真的移动了"
       而不是物理抖动；配置缺失时用 100 px/s 兜底）。**与 Lead 建议的
       ``2 × speed × dt`` 有出入**：最大步进位移本身就等于 ``speed × dt``，
       取 2 倍会让"全速步行"也判为没动，从而把正常探索误报成停滞，故改用
       0.5 倍（真实步行必超过该阈值、摩擦抖动不超过）；
    6. ``behavior_state`` 变更。

    该口径能暴露"资源耗尽后只剩重复 social/rest"的停滞（见
    ``repeat_run_max`` / ``dominant_state_ratio`` /
    ``tech_actions_per_1000_steps`` 一起看）。

事件 JSONL
----------
``event(kind, **fields)`` 向 ``log_path``（若给出）追加一行 JSON：
``{"t": 世界时间, "step": 采集步数, "kind": kind, "seed": ..., "group": ..., **fields}``。
``close()`` 关闭文件句柄；``to_json(path)`` 写最终汇总（顶层直接是
``summary()`` 的键，外加 ``windows`` / ``action_distribution`` / ``events_count``）。
``log_path`` 为 ``None`` 时事件只保留在内存 ``self.events`` 里。

§8 之外额外记录的诊断指标
--------------------------
``dominant_state_ratio``
    出现最多的 ``behavior_state`` 占全部 agent-步的比例（0..1）。接近 1.0
    表示群体锁死在同一状态（Lead 基线：单个个体 2227/2400 步连续同状态）。
``state_identical_ratio``
    见上，严格快照口径。
``tech_actions_per_1000_steps``
    技术动作的**材料单位增量代理** / 采样步数 × 1000：存活个体私有材料
    （wood/stone/草绳等任何数值字段）的正增量、工具获得、以及社区公共材料
    （wood/stone/structures）的正增量。这是**代理口径**，不是真实尝试次数：
    一次 harvest 成功通常记 1 个单位、一次 contribute 记 3 个单位、失败的
    experiment 不产生材料而不计数；真实动作次数需结合
    ``event("harvest"/"craft"/…)`` 或 ``Technology`` 计数器交叉核对。之所以
    不用 ``behavior_state`` 上升沿：个体可以**一直停留在** gather/craft 等技术
    状态内反复动作（实测中所有状态转移都发生在技术状态之间，上升沿恒为 0）。

统计口径（实现层面的约定，实验文档引用同一套）
----------------------------------------------
* **动作类别**：``f"{behavior_state}:{move:+d}"``。``behavior_state`` 取
  ``agent.behavior_state``（缺失时 ``"unknown"``）；``move`` 由本步 x 位移
  符号决定（``|dx| <= 1e-3`` 记为 0；首步没有前一帧时用 ``vx`` 符号）。
* **重复动作**：同一 agent 连续两步动作类别相同即延续当前 run，跨 agent
  取最长值 ``repeat_run_max``（单位：步）。
* **生存率**：按"个体-秒"计。``存活个体-秒 / (窗口开始存活数 × 窗口
  时长)``，截断到 ``[0, 1]``；这样出生会自然抬高分子（§8 表中
  "含出生分母按个体-秒计"的解释）。
* **动作熵**：``H = -Σ p ln p``，用**观测到的类别数** ``k`` 归一化：
  ``H / ln k``（``k <= 1`` 时为 0.0）。0..1，**不是**智能指标。
* **合作收益**：``0.5 * ln((合作产出 + 1) / (独立产出 + 1))``，截断到
  ``[-1, 1]``；这是对称化 log 比（互换分子分母恰好取负、相等为 0）。
  合作产出 = 公共库存（社区木材/石头/结构/篝火、活跃设施、活跃契约）
  的正增量 + ``event("coop_output", amount=...)``；独立产出 = 存活个体
  私有库存（木材/石头/工具）的正增量 + ``event("solo_output", amount=...)``。
* **扰动恢复**：收到 ``event("disturbance", ...)`` 时记录基线
  （存活数 / 已验证知识数 / 活跃设施数）与时间；之后逐个采样点计算
  ``mean_i clamp(current_i / baseline_i, 0, 1)``，到
  ``RECOVERY_WINDOW_S`` 世界秒后冻结该值。从未扰动则为 1.0。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

__all__ = [
    "ALL_METRIC_NAMES",
    "EXTRA_METRIC_NAMES",
    "METRIC_NAMES",
    "RECOVERY_WINDOW_S",
    "Telemetry",
]


#: §8 指标定义表的全部键（顺序与文档表格一致）。
METRIC_NAMES: tuple[str, ...] = (
    "survival_rate",
    "action_entropy",
    "repeat_run_max",
    "no_progress_repeat_ratio",
    "goal_completion_rate",
    "plan_cost_error",
    "tech_verified",
    "tech_graph_depth",
    "tech_graph_edges",
    "knowledge_retention",
    "innovation_efficiency",
    "cooperation_benefit",
    "recovery_after_disturbance",
    "total_agents_created",
    "extinction_time",
    "extinction_cause",
)

#: §8 冻结表之外的诊断指标（Lead 要求暴露原始系统的"到顶即停滞"）。
EXTRA_METRIC_NAMES: tuple[str, ...] = (
    "state_identical_ratio",
    "dominant_state_ratio",
    "tech_actions_per_1000_steps",
)

#: ``summary()`` / ``window_summary()`` 的完整键集合。
ALL_METRIC_NAMES: tuple[str, ...] = METRIC_NAMES + EXTRA_METRIC_NAMES

#: 扰动后计算恢复率的时间窗（世界秒）。§8 写的是"N 秒"，这里固定为 60 s。
RECOVERY_WINDOW_S = 60.0

#: 计数型指标：``summary()`` 里用 ``int`` 输出，``window_summary()`` 里转 float。
_COUNT_METRICS = frozenset(
    {
        "repeat_run_max",
        "tech_verified",
        "tech_graph_depth",
        "tech_graph_edges",
        "total_agents_created",
    }
)

#: 位移小于该像素数视为"没动"（抑制浮点噪声）。
_MOVE_EPS = 1e-3

#: 无法读取 ``world.config.speed`` 时用于"真的移动了"阈值的兜底速度（px/s）。
_FALLBACK_SPEED = 100.0

#: "真的移动了"的阈值 = 该系数 × 世界速度 × dt。
#: 用 0.5 而非 2.0：全速步行单步位移 = speed×dt，取 2 倍会把正常走动判成停滞。
_MOVE_PROGRESS_FACTOR = 0.5

#: 动作类别、快照缺省值。
_UNKNOWN_ACTION = "unknown"


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    """``getattr`` 的永不抛错版本（属性描述符也可能抛异常，故包 try）。"""
    if obj is None:
        return default
    try:
        return getattr(obj, name, default)
    except Exception:  # pragma: no cover - 防御性分支
        return default


def _safe_call(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """调用只读方法；方法不存在或抛异常时返回 ``None``。"""
    func = _safe_get(obj, name, None)
    if not callable(func):
        return None
    try:
        return func(*args, **kwargs)
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    """把任意值转 float，失败返回 ``default``（``None`` 也返回默认值）。"""
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _json_default(obj: Any) -> Any:
    """JSON 兜底：numpy 标量/数组转 Python 原生类型，其余转 ``str``。"""
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass
    return str(obj)


class _Accumulator:
    """一个统计窗口（或全程）的累积器。

    只存计数与求和，不持有世界对象引用，避免无意中延长对象生命周期。
    """

    __slots__ = (
        "steps",
        "agent_steps",
        "alive_person_s",
        "elapsed_s",
        "start_alive",
        "start_t",
        "first",
        "actions",
        "states",
        "no_progress_agent_steps",
        "identical_agent_steps",
        "tech_action_count",
        "run_max",
        "coop_output",
        "solo_output",
        "goals_created_start",
        "goals_completed_start",
    )

    def __init__(self) -> None:
        """清零一个统计窗口的全部累计器。"""
        self.steps = 0
        self.agent_steps = 0
        self.alive_person_s = 0.0
        self.elapsed_s = 0.0
        self.start_alive = 0
        self.start_t = 0.0
        self.first = True
        self.actions: Counter[str] = Counter()
        self.states: Counter[str] = Counter()
        self.no_progress_agent_steps = 0
        self.identical_agent_steps = 0
        self.tech_action_count = 0
        self.run_max = 0
        self.coop_output = 0.0
        self.solo_output = 0.0
        self.goals_created_start = 0.0
        self.goals_completed_start = 0.0


class Telemetry:
    """只读指标采集器（§8 冻结接口）。

    Parameters
    ----------
    window_s:
        统计窗口长度（**世界秒**，按 ``step(dt=...)`` 的 dt 累积；不是墙钟）。
    log_path:
        事件 JSONL 输出路径；``None`` 表示只存内存（``self.events``）。
        首次写事件时自动创建父目录。
    seed:
        记录的随机种子（写进每条事件和最终汇总，便于复现）。
    group:
        消融分组 ``"A"`` / ``"B"`` / ``"C"`` / ``"D"``（写进汇总）。

    典型用法::

        telemetry = Telemetry(window_s=60.0, log_path="run.jsonl", seed=0, group="D")
        for _ in range(steps):
            population.step(dt)
            telemetry.step(world, population, dt)
            if world.t >= last_window + 60.0:
                history.append(telemetry.window_summary())
        telemetry.to_json("run.summary.json")
        telemetry.close()

    注意：``step()`` 应**在**世界推进之后调用，这样读到的是本步动作的
    结果（位置变化、behavior_state、库存增量）。采集器自己不会推进世界。
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        window_s: float = 60.0,
        log_path: str | Path | None = None,
        seed: int = 0,
        group: str = "",
    ) -> None:
        """初始化采集器（窗口长度、日志路径、事件/累积器状态）。"""
        self.window_s = max(1e-6, float(window_s))
        self.log_path = Path(log_path) if log_path is not None else None
        self.seed = int(seed)
        self.group = str(group)

        # -- 事件 ---------------------------------------------------------
        #: 事件字典列表（即使没有 log_path 也保留，便于断言/调试）。
        self.events: list[dict[str, Any]] = []
        #: 已结算的窗口指标（每个窗口一条，契约 §3 的 history 可用它）。
        self.windows: list[dict[str, Any]] = []
        self._event_fh: Any = None
        self._closed = False

        # -- 累积器 -------------------------------------------------------
        self._run = _Accumulator()  # 全程
        self._win = _Accumulator()  # 当前窗口

        # -- 逐 agent 的短期状态 -----------------------------------------
        #: agent_id -> {"action", "snapshot", "run", "x", "y", "state", "goal"}
        self._agent_state: dict[int, dict[str, Any]] = {}
        #: 观测期内见过的 agent_id（registry 缺失时用于 total_agents_created）
        self._seen_ids: set[int] = set()

        # -- 水平指标（最近一次采样） ------------------------------------
        self._level: dict[str, float] = {}
        self._t = 0.0
        self._steps_seen = 0
        self._extinction_time: float = -1.0
        self._extinction_cause: str = ""
        self._last_death_cause: str = ""

        # -- 上一帧的"有意义进展"对照量 ----------------------------------
        self._public_stock_prev: float | None = None
        self._private_stock_prev: float | None = None
        self._progress_prev: dict[str, Any] | None = None
        #: 社区公共材料 (wood, stone, structures) 的上一帧值（技术动作代理用）
        self._community_prev: tuple[float, float, float] | None = None

        # -- 扰动恢复 -----------------------------------------------------
        self._disturbed_at: float | None = None
        self._disturbance_baseline: dict[str, float] = {}
        self._recovery: float = 1.0

    # ------------------------------------------------------------------
    # 事件输出
    # ------------------------------------------------------------------
    def _ensure_event_file(self) -> None:
        """惰性打开事件 JSONL（追加模式），自动创建父目录。"""
        if self._event_fh is not None or self.log_path is None or self._closed:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._event_fh = self.log_path.open("a", encoding="utf-8")

    def event(self, kind: str, **fields: Any) -> None:
        """追加一条事件（出生/死亡/发现/验证/建造/灭绝/扰动/合作产出 …）。

        事件同时保存在内存 ``self.events`` 里；给出 ``log_path`` 时追加一行
        JSON 到文件。字段值必须是 JSON 可序列化的（numpy 标量会被自动
        转成 Python 原生类型，其他对象 ``str`` 化）。

        识别并被指标逻辑消费的事件种类：

        * ``"disturbance"`` —— 开始一个扰动窗口（记录恢复基线）；
        * ``"death"`` / ``"extinction"`` —— ``cause=...`` 用于灭绝原因；
        * ``"coop_output"`` / ``"solo_output"`` —— ``amount=...`` 计入合作收益。

        其余种类只存储、不参与指标计算（供 ``SocietyRunner`` 记录文明事件）。
        """
        if self._closed:
            return
        record: dict[str, Any] = {
            "t": round(float(self._t), 6),
            "step": self._steps_seen,
            "kind": str(kind),
            "seed": self.seed,
            "group": self.group,
        }
        record.update(fields)
        self.events.append(record)

        lowered = str(kind).lower()
        try:
            if lowered == "disturbance":
                self._begin_disturbance()
            elif lowered in ("death", "died"):
                cause = fields.get("cause", fields.get("death_cause", ""))
                if cause:
                    self._last_death_cause = str(cause)
            elif lowered in ("extinction", "extinct"):
                cause = fields.get("cause", fields.get("extinction_cause", ""))
                if cause:
                    self._extinction_cause = str(cause)
                if self._extinction_time < 0.0:
                    self._extinction_time = float(self._t)
            elif lowered == "coop_output":
                self._add_coop_output(_safe_float(fields.get("amount"), 0.0))
            elif lowered == "solo_output":
                self._add_solo_output(_safe_float(fields.get("amount"), 0.0))
        except Exception:  # pragma: no cover - 指标更新绝不影响事件写入
            pass

        if self.log_path is not None:
            self._ensure_event_file()
            if self._event_fh is not None:
                line = json.dumps(record, ensure_ascii=False, default=_json_default)
                self._event_fh.write(line + "\n")
                self._event_fh.flush()

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------
    def step(self, world: Any, population: Any = None, dt: float = 1 / 60) -> None:
        """采样一个仿真步（**不推进世界**，只读状态）。

        Parameters
        ----------
        world:
            鸭子类型的 ``World``（需要 ``agents`` / ``t``；``technology``、
            ``config`` 可选）。也允许缺少这些属性，缺失项按缺省值处理。
        population:
            可选集成层对象。为 ``None`` 时退化为 ``getattr(world, "population", None)``；
            两者都没有时只统计世界级指标。
        dt:
            本步的**世界秒**步长（与 ``world.step(dt=...)`` 保持一致）。
        """
        if self._closed:
            raise RuntimeError("Telemetry 已 close()，不能再采样")
        dt = max(0.0, _safe_float(dt, 0.0))

        if population is None:
            population = _safe_get(world, "population", None)

        # -- 世界时间（优先 world.t，否则自己累加） ----------------------
        world_t = _safe_get(world, "t", None)
        if world_t is None:
            self._t += dt
        else:
            self._t = max(self._t, _safe_float(world_t, self._t))
        t = self._t
        self._steps_seen += 1

        # -- 存活个体 -----------------------------------------------------
        agents_raw = _safe_get(world, "agents", ()) or ()
        try:
            agents = [agent for agent in agents_raw if _safe_get(agent, "alive", True)]
        except Exception:
            agents = []
        alive_count = len(agents)
        for agent in agents:
            try:
                self._seen_ids.add(int(_safe_get(agent, "agent_id", -1)))
            except (TypeError, ValueError):
                pass

        # -- 水平指标（最近一次采样值） ---------------------------------
        level = self._collect_level(world, population, alive_count, t)
        self._level = level

        # -- 公共/私有库存增量（合作收益 + "有进展"信号） ----------------
        public_delta, private_delta = self._account_resource_deltas(world, population, agents)
        world_progress = self._world_progress_changed(level, public_delta)

        # -- 累积器：窗口与全程共用同一份逐 agent 观测 -------------------
        self._accumulate(self._run, level, alive_count, dt)
        self._accumulate(self._win, level, alive_count, dt)

        # -- 逐 agent 动作/快照/进展 -------------------------------------
        goal_progress = self._goal_progress_map(population, agents)
        speed = _safe_float(_safe_get(_safe_get(world, "config", None), "speed", None), _FALLBACK_SPEED)
        move_threshold = max(_MOVE_EPS, _MOVE_PROGRESS_FACTOR * speed * dt)
        tech_units = self._community_tech_units(world)
        next_state: dict[int, dict[str, Any]] = {}
        for agent in agents:
            try:
                agent_id = int(_safe_get(agent, "agent_id", -1))
            except (TypeError, ValueError):
                continue
            prev = self._agent_state.get(agent_id)
            move = self._move_of(agent, prev)
            state_name = str(_safe_get(agent, "behavior_state", _UNKNOWN_ACTION) or _UNKNOWN_ACTION)
            action = f"{state_name}:{move:+d}"
            snapshot = self._snapshot_of(agent, goal_progress.get(agent_id))
            x = _safe_float(_safe_get(agent, "x", 0.0), 0.0)
            y = _safe_float(_safe_get(agent, "y", 0.0), 0.0)
            inventory_q = _inventory_quantities(_safe_get(agent, "inventory", None))
            goal_sig = goal_progress.get(agent_id)

            if prev is not None:
                identical = snapshot == prev["snapshot"]
                if identical:
                    self._run.identical_agent_steps += 1
                    self._win.identical_agent_steps += 1
                moved = (
                    abs(x - _safe_float(prev.get("x"), x)) > move_threshold
                    or abs(y - _safe_float(prev.get("y"), y)) > move_threshold
                )
                meaningful = (
                    moved
                    or inventory_q != prev["inventory"]
                    or state_name != prev["state"]
                    or goal_sig != prev["goal"]
                    or world_progress
                )
                if prev["action"] == action and not meaningful:
                    self._run.no_progress_agent_steps += 1
                    self._win.no_progress_agent_steps += 1
                run = int(prev["run"]) + 1 if prev["action"] == action else 1
                # 技术动作代理：私有材料/工具的正增量（单位：材料单位）
                tech_units += _positive_delta(prev["inventory"], inventory_q)
            else:
                run = 1

            self._run.actions[action] += 1
            self._win.actions[action] += 1
            self._run.states[state_name] += 1
            self._win.states[state_name] += 1
            self._run.run_max = max(self._run.run_max, run)
            self._win.run_max = max(self._win.run_max, run)
            next_state[agent_id] = {
                "action": action,
                "snapshot": snapshot,
                "inventory": inventory_q,
                "goal": goal_sig,
                "run": run,
                "x": x,
                "y": y,
                "state": state_name,
            }
        # 清掉已离开世界的 agent，避免字典无界增长
        self._agent_state = next_state
        tech_units_rounded = int(round(tech_units))
        if tech_units_rounded > 0:
            self._run.tech_action_count += tech_units_rounded
            self._win.tech_action_count += tech_units_rounded

        # -- 灭绝检测 -----------------------------------------------------
        if alive_count == 0 and self._extinction_time < 0.0:
            self._extinction_time = float(t)

        # -- 扰动恢复 -----------------------------------------------------
        self._update_recovery(world, population, alive_count, t)

        # -- 窗口结算 -----------------------------------------------------
        if self._win.steps > 0 and self._win.elapsed_s + 1e-9 >= self.window_s:
            self.windows.append(self._window_metrics())
            self._win = _Accumulator()

    # ------------------------------------------------------------------
    # 累积器更新
    # ------------------------------------------------------------------
    def _accumulate(
        self,
        acc: _Accumulator,
        level: dict[str, float],
        alive_count: int,
        dt: float,
    ) -> None:
        """把一步的存活/时间/目标计数累进指定的窗口累积器。"""
        if acc.first:
            acc.first = False
            acc.start_alive = int(alive_count)
            acc.start_t = float(self._t)
            acc.goals_created_start = float(level.get("goals_created", 0.0))
            acc.goals_completed_start = float(level.get("goals_completed", 0.0))
        acc.steps += 1
        acc.agent_steps += int(alive_count)
        acc.alive_person_s += float(alive_count) * dt
        acc.elapsed_s += dt

    def _collect_level(
        self,
        world: Any,
        population: Any,
        alive_count: int,
        t: float,
    ) -> dict[str, float]:
        """读一次"水平"指标（目标/知识/文明/记忆），全部 try 包裹。"""
        level: dict[str, float] = {"alive_count": float(alive_count)}

        # -- 目标/计划 ----------------------------------------------------
        goals = _resolve(population, world, "goals")
        stats = _safe_call(goals, "stats") if goals is not None else None
        stats = stats if isinstance(stats, dict) else {}
        level["goals_created"] = _safe_float(stats.get("goals_created"), 0.0)
        level["goals_completed"] = _safe_float(stats.get("goals_completed"), 0.0)
        level["goals_failed"] = _safe_float(stats.get("goals_failed"), 0.0)
        level["goals_abandoned"] = _safe_float(stats.get("goals_abandoned"), 0.0)
        level["plan_cost_error"] = _safe_float(stats.get("cost_estimate_error"), 0.0)
        level["plan_success_rate"] = _safe_float(stats.get("plan_success_rate"), 0.0)

        # -- 文明状态 -----------------------------------------------------
        civ = _resolve(population, world, "civilization")
        registry = _safe_get(civ, "registry", None)
        total_created = _safe_call(registry, "total_created")
        if total_created is None:
            total_created = _safe_get(registry, "total_created", None)
        if total_created is None:
            # 退化：观测期内见过的 id 数
            total_created = len(self._seen_ids)
        level["total_agents_created"] = float(_safe_float(total_created, 0.0))

        facilities_active = 0.0
        facilities_condition = 0.0
        facilities = _safe_get(civ, "facilities", None)
        if facilities:
            for facility in facilities:
                if _safe_get(facility, "active", False):
                    facilities_active += 1.0
                facilities_condition += _safe_float(_safe_get(facility, "condition", 0.0), 0.0)
        level["facilities_active"] = facilities_active
        level["facilities_condition"] = facilities_condition

        contracts_active = 0.0
        contracts = _safe_get(civ, "contracts", None)
        if contracts:
            for contract in contracts:
                if _safe_get(contract, "active", True):
                    contracts_active += 1.0
        level["contracts_active"] = contracts_active
        level["cooperation_index"] = _safe_float(_safe_get(civ, "cooperation_index", 0.0), 0.0)

        # -- 知识/技术 ----------------------------------------------------
        knowledge = _resolve(population, world, "knowledge")
        known_ids = self._known_ids(world, civ, population)
        level["known_count"] = float(len(known_ids))
        verified = 0.0
        depth = 0.0
        edges = 0.0
        retention = 0.0
        innovation = 0.0
        if knowledge is not None:
            verified_items = _safe_call(knowledge, "all", status="verified")
            if verified_items is None:
                verified_items = _safe_call(knowledge, "all")
            try:
                verified = float(len(verified_items)) if verified_items is not None else 0.0
            except TypeError:
                verified = 0.0
            depth = _safe_float(_safe_call(knowledge, "depth"), 0.0)
            edge_list = _safe_call(knowledge, "edges")
            try:
                edges = float(len(edge_list)) if edge_list is not None else 0.0
            except TypeError:
                edges = 0.0
            retention = _safe_float(
                _safe_call(knowledge, "retention", known_ids),
                self._retention_fallback(known_ids, verified),
            )
            innovation = _safe_float(_safe_call(knowledge, "innovation_efficiency"), 0.0)
        else:
            # 没有知识库时，CivilizationState.innovation_count 就是已验证数
            verified = _safe_float(_safe_get(civ, "innovation_count", 0.0), 0.0)
            retention = self._retention_fallback(known_ids, verified)
        level["tech_verified"] = verified
        level["tech_graph_depth"] = depth
        level["tech_graph_edges"] = edges
        level["knowledge_retention"] = retention
        level["innovation_efficiency"] = innovation

        # -- 记忆（附加指标，不属于 §8 冻结表） --------------------------
        memory = _resolve(population, world, "memory")
        memory_agents = _safe_call(memory, "agent_count")
        level["memory_agents"] = _safe_float(memory_agents, 0.0)

        # -- 灭绝 ---------------------------------------------------------
        extinct = bool(_safe_get(civ, "extinct", False))
        extinct_at = _safe_get(civ, "extinct_at", None) if civ is not None else None
        if extinct and self._extinction_time < 0.0:
            self._extinction_time = _safe_float(extinct_at if extinct_at is not None else t, t)
        cause = _safe_get(civ, "extinction_cause", "") if civ is not None else ""
        if cause:
            self._extinction_cause = str(cause)

        return level

    @staticmethod
    def _known_ids(world: Any, civ: Any, population: Any) -> set[str]:
        """当前存活个体 + 死亡档案掌握的知识 id 并集（用于知识保留率）。"""
        known: set[str] = set()
        agents = _safe_get(world, "agents", ()) or ()
        for agent in agents:
            ids = _safe_get(agent, "known_knowledge_ids", ()) or ()
            try:
                known.update(str(item) for item in ids)
            except TypeError:
                pass
        registry = _safe_get(civ, "registry", None)
        records = _safe_call(registry, "records")
        if records:
            for record in records:
                ids = _safe_get(record, "known_knowledge_ids", ()) or ()
                try:
                    known.update(str(item) for item in ids)
                except TypeError:
                    pass
        # 集成的 knowledge 模块有时把掌握情况挂在 population 上
        extra = _safe_get(population, "known_knowledge_ids", None) if population is not None else None
        if extra:
            try:
                known.update(str(item) for item in extra)
            except TypeError:
                pass
        return known

    @staticmethod
    def _retention_fallback(known_ids: set[str], verified: float) -> float:
        """无知识库方法时的保留率近似：``|已知 ∪ 档案| / 已验证总数``。"""
        if verified <= 0.0:
            return 0.0
        return min(1.0, len(known_ids) / verified)

    # ------------------------------------------------------------------
    # 逐 agent 读数
    # ------------------------------------------------------------------
    @staticmethod
    def _move_of(agent: Any, prev: dict[str, Any] | None) -> int:
        """动作的移动分量：本步 x 位移符号，首步退回 vx 符号。"""
        x = _safe_float(_safe_get(agent, "x", 0.0), 0.0)
        if prev is not None:
            dx = x - _safe_float(prev.get("x"), x)
            if dx > _MOVE_EPS:
                return 1
            if dx < -_MOVE_EPS:
                return -1
            return 0
        vx = _safe_float(_safe_get(agent, "vx", 0.0), 0.0)
        if vx > 1.0:
            return 1
        if vx < -1.0:
            return -1
        return 0

    @staticmethod
    def _snapshot_of(agent: Any, goal_progress: Any) -> tuple[Any, ...]:
        """**严格快照**（``state_identical_ratio`` 用，含连续变化的生理量）。"""
        physiology = _safe_get(agent, "physiology", None)
        inventory = _safe_get(agent, "inventory", None)
        tool = _safe_get(inventory, "tool", None)
        return (
            round(_safe_float(_safe_get(agent, "x", 0.0), 0.0), 2),
            round(_safe_float(_safe_get(agent, "y", 0.0), 0.0), 2),
            round(_safe_float(_safe_get(physiology, "energy", 0.0), 0.0), 2),
            round(_safe_float(_safe_get(physiology, "hydration", 0.0), 0.0), 2),
            int(_safe_float(_safe_get(inventory, "wood", 0), 0.0)),
            int(_safe_float(_safe_get(inventory, "stone", 0), 0.0)),
            str(tool) if tool is not None else "",
            str(_safe_get(agent, "behavior_state", _UNKNOWN_ACTION) or _UNKNOWN_ACTION),
            goal_progress,
        )

    def _goal_progress_map(self, population: Any, agents: list[Any]) -> dict[int, Any]:
        """每个存活 agent 的"目标进度"元组；无目标管理器时为 ``None``。"""
        goals = _resolve(population, None, "goals")
        result: dict[int, Any] = {}
        if goals is None:
            return result
        for agent in agents:
            try:
                agent_id = int(_safe_get(agent, "agent_id", -1))
            except (TypeError, ValueError):
                continue
            plan = _safe_call(goals, "active_plan", agent_id)
            if plan is None:
                result[agent_id] = None
                continue
            result[agent_id] = (
                _safe_float(_safe_get(plan, "plan_id", 0.0), 0.0),
                _safe_float(_safe_get(plan, "current", 0.0), 0.0),
                _safe_float(_safe_get(plan, "version", 0.0), 0.0),
            )
        return result

    # ------------------------------------------------------------------
    # 有意义的进展 / 合作收益
    # ------------------------------------------------------------------
    @staticmethod
    def _public_stock(world: Any, population: Any) -> float:
        """公共产出代理量：社区库存 + 结构 + 篝火 + 活跃设施 + 活跃契约。"""
        total = 0.0
        technology = _safe_get(world, "technology", None)
        community = _safe_get(technology, "community", None)
        if community is not None:
            total += _safe_float(_safe_get(community, "wood", 0), 0.0)
            total += _safe_float(_safe_get(community, "stone", 0), 0.0)
            total += 4.0 * _safe_float(_safe_get(community, "structures", 0), 0.0)
            if _safe_get(community, "campfire", False):
                total += 1.0
        civ = _resolve(population, world, "civilization")
        facilities = _safe_get(civ, "facilities", None)
        if facilities:
            for facility in facilities:
                if _safe_get(facility, "active", False):
                    total += 1.0 + _safe_float(_safe_get(facility, "condition", 0.0), 0.0)
        contracts = _safe_get(civ, "contracts", None)
        if contracts:
            for contract in contracts:
                if _safe_get(contract, "active", True):
                    total += 2.0
        return total

    @staticmethod
    def _private_stock(agents: list[Any]) -> float:
        """独立产出代理量：存活个体私有库存（木材 + 石头 + 工具）。"""
        total = 0.0
        for agent in agents:
            inventory = _safe_get(agent, "inventory", None)
            total += _safe_float(_safe_get(inventory, "wood", 0), 0.0)
            total += _safe_float(_safe_get(inventory, "stone", 0), 0.0)
            if _safe_get(inventory, "tool", None):
                total += 1.0
        return total

    def _community_tech_units(self, world: Any) -> float:
        """社区公共材料的正增量（技术动作代理的一部分，每步调用一次）。"""
        community = _safe_get(_safe_get(world, "technology", None), "community", None)
        if community is None:
            self._community_prev = None
            return 0.0
        current = (
            _safe_float(_safe_get(community, "wood", 0), 0.0),
            _safe_float(_safe_get(community, "stone", 0), 0.0),
            _safe_float(_safe_get(community, "structures", 0), 0.0),
        )
        previous = self._community_prev
        self._community_prev = current
        if previous is None:
            return 0.0
        return sum(max(0.0, cur - prev) for cur, prev in zip(current, previous))

    def _account_resource_deltas(
        self, world: Any, population: Any, agents: list[Any]
    ) -> tuple[float, float]:
        """把公共/私有库存的正增量计入合作/独立产出，并返回本步增量。"""
        try:
            public = self._public_stock(world, population)
            private = self._private_stock(agents)
        except Exception:  # pragma: no cover - 防御性分支
            return 0.0, 0.0
        public_delta = 0.0
        private_delta = 0.0
        if self._public_stock_prev is not None:
            public_delta = public - self._public_stock_prev
            if public_delta > 0.0:
                self._add_coop_output(public_delta)
        if self._private_stock_prev is not None:
            private_delta = private - self._private_stock_prev
            if private_delta > 0.0:
                self._add_solo_output(private_delta)
        self._public_stock_prev = public
        self._private_stock_prev = private
        return public_delta, private_delta

    def _world_progress_changed(self, level: dict[str, float], public_delta: float) -> bool:
        """世界级"有进展"信号：公共库存 / 知识 / 目标计数是否变化。"""
        current = {
            "verified": _safe_float(level.get("tech_verified"), 0.0),
            "known": _safe_float(level.get("known_count"), 0.0),
            "goals": (
                _safe_float(level.get("goals_created"), 0.0),
                _safe_float(level.get("goals_completed"), 0.0),
                _safe_float(level.get("goals_failed"), 0.0),
                _safe_float(level.get("goals_abandoned"), 0.0),
            ),
        }
        previous = self._progress_prev
        self._progress_prev = current
        if previous is None:
            return False
        progress = public_delta > 0.0
        if current["verified"] != previous["verified"]:
            progress = True
        if current["known"] != previous["known"]:
            progress = True
        if current["goals"] != previous["goals"]:
            progress = True
        return progress

    def _add_coop_output(self, amount: float) -> None:
        """把一笔合作产出（资源转移量）累进全程与当前窗口。"""
        if amount <= 0.0:
            return
        self._run.coop_output += amount
        self._win.coop_output += amount

    def _add_solo_output(self, amount: float) -> None:
        """把一笔独力产出（个人资源产出）累进全程与当前窗口。"""
        if amount <= 0.0:
            return
        self._run.solo_output += amount
        self._win.solo_output += amount

    # ------------------------------------------------------------------
    # 扰动恢复
    # ------------------------------------------------------------------
    def _begin_disturbance(self) -> None:
        """记录扰动基线（存活数 / 已验证知识 / 活跃设施）。"""
        self._disturbed_at = float(self._t)
        self._disturbance_baseline = {
            "alive": max(1.0, _safe_float(self._level.get("alive_count"), 0.0)),
            "tech": max(1.0, _safe_float(self._level.get("tech_verified"), 0.0)),
            "facilities": max(1.0, _safe_float(self._level.get("facilities_active"), 0.0)),
        }
        self._recovery = 0.0

    def _update_recovery(self, world: Any, population: Any, alive_count: int, t: float) -> None:
        """扰动后逐点计算恢复率，到 ``RECOVERY_WINDOW_S`` 后冻结。"""
        if self._disturbed_at is None:
            self._recovery = 1.0
            return
        age = t - self._disturbed_at
        baseline = self._disturbance_baseline
        if not baseline:
            self._recovery = 1.0
            return
        current = {
            "alive": float(alive_count),
            "tech": _safe_float(self._level.get("tech_verified"), 0.0),
            "facilities": _safe_float(self._level.get("facilities_active"), 0.0),
        }
        ratios = []
        for key, base in baseline.items():
            if base <= 0.0:
                ratios.append(1.0)
            else:
                ratios.append(min(1.0, max(0.0, current.get(key, 0.0) / base)))
        self._recovery = sum(ratios) / max(1, len(ratios))
        if age >= RECOVERY_WINDOW_S:
            # 冻结：N 秒后的恢复率不再变化
            self._disturbed_at = None
            self._disturbance_baseline = {}

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------
    def _metrics_from(self, acc: _Accumulator) -> dict[str, Any]:
        """从一个累积器算全部指标（未做 int 转换，便于窗口统一转 float）。"""
        actions_total = sum(acc.actions.values())
        action_entropy = _entropy(acc.actions) if actions_total else 0.0

        if acc.start_alive > 0 and acc.elapsed_s > 1e-9:
            survival = acc.alive_person_s / (acc.start_alive * acc.elapsed_s)
        elif acc.steps > 0 and acc.start_alive > 0:
            # 世界时间没走（dt=0）：退化为步数比
            survival = (acc.alive_person_s / max(1e-9, acc.steps)) / acc.start_alive
        else:
            survival = 0.0
        survival = min(1.0, max(0.0, survival))

        created_delta = _safe_float(self._level.get("goals_created"), 0.0) - acc.goals_created_start
        completed_delta = (
            _safe_float(self._level.get("goals_completed"), 0.0) - acc.goals_completed_start
        )
        if created_delta > 0.0:
            completion = min(1.0, max(0.0, completed_delta / created_delta))
        else:
            completion = 0.0

        coop = _safe_float(acc.coop_output, 0.0)
        solo = _safe_float(acc.solo_output, 0.0)
        benefit = 0.5 * math.log((coop + 1.0) / (solo + 1.0))
        benefit = min(1.0, max(-1.0, benefit))

        no_progress = (
            acc.no_progress_agent_steps / acc.agent_steps if acc.agent_steps else 0.0
        )
        identical = acc.identical_agent_steps / acc.agent_steps if acc.agent_steps else 0.0
        states_total = sum(acc.states.values())
        dominant = max(acc.states.values()) / states_total if states_total else 0.0
        tech_rate = acc.tech_action_count * 1000.0 / acc.steps if acc.steps else 0.0

        return {
            # ---- §8 冻结表 -------------------------------------------
            "survival_rate": survival,
            "action_entropy": action_entropy,
            "repeat_run_max": acc.run_max,
            "no_progress_repeat_ratio": no_progress,
            "goal_completion_rate": completion,
            "plan_cost_error": _safe_float(self._level.get("plan_cost_error"), 0.0),
            "tech_verified": _safe_float(self._level.get("tech_verified"), 0.0),
            "tech_graph_depth": _safe_float(self._level.get("tech_graph_depth"), 0.0),
            "tech_graph_edges": _safe_float(self._level.get("tech_graph_edges"), 0.0),
            "knowledge_retention": _safe_float(self._level.get("knowledge_retention"), 0.0),
            "innovation_efficiency": _safe_float(self._level.get("innovation_efficiency"), 0.0),
            "cooperation_benefit": benefit,
            "recovery_after_disturbance": float(self._recovery),
            "total_agents_created": _safe_float(self._level.get("total_agents_created"), 0.0),
            "extinction_time": float(self._extinction_time),
            "extinction_cause": self._extinction_cause or self._last_death_cause or "",
            # ---- 附加诊断指标 ---------------------------------------
            "state_identical_ratio": identical,
            "dominant_state_ratio": dominant,
            "tech_actions_per_1000_steps": tech_rate,
        }

    def _window_metrics(self) -> dict[str, Any]:
        """结算当前窗口：键与 ``window_summary`` 相同，值统一转 float（原因除外）。"""
        return _cast_metrics(self._metrics_from(self._win), as_float=True)

    def window_summary(self) -> dict[str, Any]:
        """当前（未结算）统计窗口的指标；窗口刚结算完则返回最近一次结算值。

        键 = ``METRIC_NAMES + EXTRA_METRIC_NAMES``；除 ``extinction_cause``
        为 ``str``、计数型为 ``int`` 外，其余为 ``float``。尚无任何采样时
        返回缺省值（比率 0.0、计数 0、``extinction_time`` -1.0、原因空串、
        ``recovery_after_disturbance`` 1.0）。
        """
        acc = self._win if self._win.steps > 0 else self._run
        if acc.steps == 0 and self.windows:
            return dict(self.windows[-1])
        metrics = self._metrics_from(acc)
        return _cast_metrics(metrics, as_float=True)

    def summary(self) -> dict[str, Any]:
        """全程聚合指标（契约 §8 全表 + 附加诊断指标），可直接 ``json.dumps``。

        返回键包含 ``ALL_METRIC_NAMES`` 全部 19 项，另附少量便于日志的元数据
        （``seed`` / ``group`` / ``steps`` / ``world_time`` / ``window_s`` /
        ``windows`` / ``alive_count`` / ``memory_agents``）。缺少数据时按模块
        docstring 的表格给出安全默认值。
        """
        metrics = self._metrics_from(self._run)
        result: dict[str, Any] = {}
        for name in ALL_METRIC_NAMES:
            value = metrics.get(name, _default_metric(name))
            if name in _COUNT_METRICS:
                result[name] = int(round(_safe_float(value, 0.0)))
            elif name == "extinction_cause":
                result[name] = str(value or "")
            elif name == "extinction_time":
                result[name] = float(value)
            else:
                result[name] = float(value)
        result.update(
            {
                "seed": self.seed,
                "group": self.group,
                "steps": int(self._run.steps),
                "world_time": round(float(self._t), 6),
                "window_s": float(self.window_s),
                "windows": int(len(self.windows)),
                "alive_count": int(_safe_float(self._level.get("alive_count"), 0.0)),
                "memory_agents": int(_safe_float(self._level.get("memory_agents"), 0.0)),
            }
        )
        return result

    def action_distribution(self, *, whole_run: bool = True) -> dict[str, float]:
        """动作类别分布（附加查询，不属于 §8 冻结表）：值之和为 1.0。

        ``whole_run=False`` 时只统计当前窗口。没有任何 agent 步时返回空字典。
        """
        acc = self._run if whole_run else self._win
        total = sum(acc.actions.values())
        if total <= 0:
            return {}
        return {key: count / total for key, count in acc.actions.most_common()}

    def state_distribution(self, *, whole_run: bool = True) -> dict[str, float]:
        """behavior_state 分布（附加查询）：值之和为 1.0。"""
        acc = self._run if whole_run else self._win
        total = sum(acc.states.values())
        if total <= 0:
            return {}
        return {key: count / total for key, count in acc.states.most_common()}

    # ------------------------------------------------------------------
    # 输出 / 生命周期
    # ------------------------------------------------------------------
    def to_json(self, path: str | Path | None = None) -> Path:
        """写最终汇总 JSON，返回实际写入路径。

        ``path`` 为 ``None`` 时：默认 ``<log_path>.summary.json``（``log_path``
        也缺失时为当前目录 ``telemetry_summary.json``）。文件内容顶层就是
        ``summary()`` 的键，另加 ``windows`` / ``action_distribution`` /
        ``state_distribution`` / ``events_count``，便于直接 ``json.load`` 取指标。
        """
        if path is None:
            if self.log_path is not None:
                path = self.log_path.with_name(self.log_path.name + ".summary.json")
            else:
                path = Path("telemetry_summary.json")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.summary())
        payload["windows"] = [dict(window) for window in self.windows]
        payload["action_distribution"] = self.action_distribution()
        payload["state_distribution"] = self.state_distribution()
        payload["events_count"] = len(self.events)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return path

    def close(self) -> None:
        """关闭事件文件句柄（幂等）；之后 ``event()`` 静默忽略、``step()`` 报错。"""
        self._closed = True
        handle = self._event_fh
        self._event_fh = None
        if handle is not None:
            try:
                handle.flush()
                handle.close()
            except Exception:  # pragma: no cover - 防御性分支
                pass

    # -- 上下文管理（便利，不属于冻结接口） -----------------------------
    def __enter__(self) -> "Telemetry":
        """``with`` 进入：返回自身。"""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """``with`` 退出：确保日志文件被关闭。"""
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        """调试用的简要表示。"""
        return (
            f"<Telemetry steps={self._run.steps} t={self._t:.2f} "
            f"window_s={self.window_s:g} seed={self.seed} group={self.group!r}>"
        )


# ---------------------------------------------------------------------------
# 模块级辅助
# ---------------------------------------------------------------------------
def _resolve(population: Any, world: Any, name: str) -> Any:
    """先读 ``population.<name>``，再退回 ``world.<name>``（两者都可缺失）。"""
    if population is not None:
        value = _safe_get(population, name, None)
        if value is not None:
            return value
    if world is not None:
        return _safe_get(world, name, None)
    return None


def _inventory_quantities(inventory: Any) -> dict[str, float]:
    """库存的**可比较量化签名**：数值字段 + 工具拥有标记。

    收集 ``Inventory.__dict__`` 里所有数值字段（wood/stone/cord/basket/…，
    Lead 给 ``Inventory`` 增加材料时无需改 telemetry），并额外用
    ``"tool_owned"`` 表示是否持有工具。布尔字段与非数值对象不参与比较。
    """
    if inventory is None:
        return {}
    data = _safe_get(inventory, "__dict__", None)
    quantities: dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                quantities[str(key)] = round(float(value), 6)
    else:
        # 退化：只读已知的三个字段
        quantities["wood"] = round(_safe_float(_safe_get(inventory, "wood", 0), 0.0), 6)
        quantities["stone"] = round(_safe_float(_safe_get(inventory, "stone", 0), 0.0), 6)
    quantities["tool_owned"] = 1.0 if _safe_get(inventory, "tool", None) else 0.0
    return quantities


def _positive_delta(previous: dict[str, float], current: dict[str, float]) -> float:
    """两个库存签名之间**正增量**之和（材料单位；减少不计）。"""
    total = 0.0
    for key, value in current.items():
        delta = value - float(previous.get(key, 0.0))
        if delta > 0.0:
            total += delta
    return total


def _entropy(counter: Counter[str]) -> float:
    """归一化 Shannon 熵：``H / ln(k)``，观测到单一类别时为 0.0。"""
    total = sum(counter.values())
    kinds = len(counter)
    if total <= 0 or kinds <= 1:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return min(1.0, max(0.0, entropy / math.log(kinds)))


def _default_metric(name: str) -> Any:
    """尚未采样时各指标的安全缺省值。"""
    if name == "extinction_cause":
        return ""
    if name == "extinction_time":
        return -1.0
    if name == "recovery_after_disturbance":
        return 1.0
    return 0.0


def _cast_metrics(metrics: dict[str, Any], *, as_float: bool) -> dict[str, Any]:
    """``window_summary`` 用：除原因字符串外统一转 float。"""
    result: dict[str, Any] = {}
    for name in ALL_METRIC_NAMES:
        value = metrics.get(name, _default_metric(name))
        if name == "extinction_cause":
            result[name] = str(value or "")
        else:
            result[name] = float(_safe_float(value, 0.0)) if as_float else value
    return result
