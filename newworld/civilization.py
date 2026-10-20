"""历史注册表 / 文明状态 / 公共设施（P3/P4）。

契约来源：``docs/IMPLEMENTATION_REPORT.md`` §3（冻结接口）。本模块只依赖标准库，
**不** import ``knowledge`` / ``memory`` / ``world`` / ``population``：
``knowledge`` 与 ``memory`` 以 duck typing 形式从构造参数注入（只调用契约里的
公开方法），避免循环依赖；对世界的读取也全部经 ``getattr`` 完成。

8 个创建名额是硬约束（§0.3）
-----------------------------
``AgentRegistry.total_created`` 只增不减，且 ``total_created <= max_agents``
（默认 ``MAX_AGENTS_CREATED = 8``）。``register()`` 在满额时抛
:class:`PopulationLimitError` 且**不改变任何计数**。死亡个体不会从 ``records()``
删除，只标记 ``alive=False`` 并保存 ``died_at / death_cause / final_state /
inventory / known_knowledge_ids / skills / memory_archive``；``alive_ids()``
只返回存活 id。``lineage()`` 同时依据 ``parent_id`` 与 ``children`` 双向可查。

灭绝规则（§3）
--------------
``alive_count() == 0`` 且 ``can_create() == False`` → ``extinct=True``；
``enable_inheritance=False``（不允许新生）时 ``alive_count() == 0`` 同样灭绝。
``extinct_at`` 记录首次判定灭绝的世界时间，``extinction_cause`` 取最近一次死亡
档案的 cause（无死亡记录时为 ``"unknown"``）。灭绝后 ``collapse_state`` 固定为
``"collapsed"``，只保留历史。**注意**：只要 ``alive_count() == 0``，
``collapse_state`` 也为 ``"collapsed"``（没有存活个体的文明无法执行任何动作），
但 ``extinct`` 仍严格按上面的规则判定——尚有创建名额时 ``alive_count()==0``
不等于 ``extinct``。

``collapse_state`` 五态机（确定公式，无随机数）
----------------------------------------------
输入（全部来自 ``update()`` 能拿到的数据；输入不足时给保守值）：

* ``ratio``：人口变化率，``(alive_now - alive_ref) / max(1, alive_ref)``。
  ``alive_ref`` 是 ``POP_WINDOW_S = 120`` 世界秒前最近一次采样到的存活数；
  世界运行不足 120s（没有参照样本）→ ``ratio = 0.0``（保守，不触发崩塌）。
* ``avg_energy``：``world.agents`` 中存活个体 ``physiology.energy`` 的均值；
  没有存活个体 → 0.0；有存活个体但读不到 ``physiology`` → 50.0（中性保守值）。
* ``stagnation_s``：``t - last_tech_change_t``；``innovation_count`` 每次变化都
  刷新 ``last_tech_change_t``，从未变化则从世界 ``t = 0`` 起算。
* ``facilities_active``：``Facility.active``（``condition > 0.2``）的数量；
  **只影响恢复判定**：已存在公共设施但全部失效时，不能判定为 ``"recovering"``
  （公共设施全灭的群体不算恢复），其余状态不受它影响。

判定顺序（先命中者生效，常量见模块级定义）：

1. ``extinct`` 或 ``alive_now == 0`` → ``"collapsed"``。
2. 上一次状态属于 ``{"collapsing", "collapsed", "recovering"}`` 且 ``ratio > 0``
   且 ``avg_energy >= COLLAPSE_ENERGY_RECOVER (25.0)`` 且
   （没有公共设施 或 ``facilities_active >= 1``）→ ``"recovering"``。
3. 上一次状态属于 ``{"collapsing", "recovering"}`` 且 ``ratio <= 0``
   → ``"collapsing"``（崩塌 / 恢复一旦开始，必须靠**正的**人口增长才能中止）。
4. ``ratio <= COLLAPSE_POP_DROP_RATIO (-0.10)`` 或
   ``avg_energy < COLLAPSE_ENERGY_CRITICAL (15.0)`` → ``"collapsing"``。
5. ``stagnation_s >= STAGNATION_S (180.0)`` 且 ``ratio <= 0`` → ``"stagnating"``。
6. 其余 → ``"stable"``。

``cooperation_index``（§8 指标定义）
------------------------------------
::

    coop_gain          = Σ 存活 AgentRecord.contributions 中的所有正值
    coop_events        = 贡献条目数 + Σ 活跃 SocialContract 的 max(0, 参与人数-1)
    independent_events = 存活个体在 MemoryStore 中成功的 "episode" 条目数
                         （memory=None 或读不到 → 0）
    cooperation_index  = coop_gain / (coop_gain + independent_events
                                      + COOPERATION_SMOOTHING)

其中 ``COOPERATION_SMOOTHING = 1.0`` 是平滑项；``coop_gain`` 等于
"合作事件数 × 平均收益"（平均收益 = ``coop_gain / coop_events``），因此公式与
§8 定义一致。``coop_gain == 0`` 或 ``enable_social = False`` 或数据缺失 →
``0.0``；结果始终在 ``[0, 1]``。``coop_events`` 只用于记录与调参，不直接进公式。
``AgentRecord.contributions`` 由 ``update()`` 从 ``agent.contributions`` 同步。

公共设施（§3）
--------------
``Facility.decay(t)`` 从 ``built_at``（或上一次 decay / 维护时刻）起按
``FACILITY_DECAY_PER_SECOND = 1/600``（每秒线性损耗）下降，下限 0；
``maintained()`` 把 ``condition`` 提升 ``FACILITY_MAINTENANCE_BOOST = 0.35``
并重置损耗锚点（同时更新 ``last_maintained_at``）；``active`` 定义为
``condition > FACILITY_ACTIVE_THRESHOLD (0.2)``。
``CivilizationState.maintain()`` 只校验传入的消耗额是否达到设施要求（实际库存
扣减由集成层完成），并把这次维护计入维护者的合作贡献。``maintenance_interval_s``
与 ``private_uses`` 只是元数据：维护周期由集成层调度（本模块不因"未到周期"
拒绝维护），``private_uses`` 由集成层在设施被使用时累加。

``history``（§3）
-----------------
``update()`` 每累计 ``HISTORY_WINDOW_S = 60`` 世界秒追加一条窗口记录（首次调用
立即追加一条），最多保留 ``HISTORY_LIMIT = 200`` 条（超出丢弃最旧），每条都是
JSON 可序列化 dict，字段见 ``_append_history()``。
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: 创建名额硬约束（§0.3）
MAX_AGENTS_CREATED = 8
#: 设施"活跃"的 condition 阈值（严格大于才算活跃）
FACILITY_ACTIVE_THRESHOLD = 0.2
#: 设施自然损耗速率（condition / 世界秒）
FACILITY_DECAY_PER_SECOND = 1.0 / 1200.0
#: 一次维护把 condition 提升的量
FACILITY_MAINTENANCE_BOOST = 0.35
#: history 环形窗口上限
HISTORY_LIMIT = 200
#: history 统计窗口长度（世界秒）
HISTORY_WINDOW_S = 60.0
#: 计算人口变化率的参照窗口（世界秒）
POP_WINDOW_S = 120.0
#: 人口变化率低于该值视为崩塌（120s 窗口内损失 10%）
COLLAPSE_POP_DROP_RATIO = -0.10
#: 平均能量低于该值视为崩塌
COLLAPSE_ENERGY_CRITICAL = 15.0
#: 平均能量高于该值且人口正增长时允许判定为恢复中
COLLAPSE_ENERGY_RECOVER = 25.0
#: 技术停滞多少世界秒视为停滞（stagnating）
STAGNATION_S = 180.0
#: cooperation_index 的平滑项
COOPERATION_SMOOTHING = 1.0
#: collapse_state 的合法取值（顺序即 §3 列出的顺序）
COLLAPSE_STATES = ("stable", "stagnating", "collapsing", "collapsed", "recovering")
#: collapse_state -> 数值编码（供 metrics 使用）
COLLAPSE_STATE_CODES = {
    "stable": 0.0,
    "stagnating": 1.0,
    "collapsing": 2.0,
    "collapsed": 3.0,
    "recovering": 4.0,
}
#: collapse_state 的中文名（summary_zh）
COLLAPSE_STATE_ZH = {
    "stable": "稳定",
    "stagnating": "停滞",
    "collapsing": "崩塌中",
    "collapsed": "已崩溃",
    "recovering": "恢复中",
}
#: 可建造公共设施的世界规则（建造消耗 + 维护需求），不是剧情
FACILITY_KINDS: dict[str, dict[str, object]] = {
    "campfire": {
        "name_zh": "篝火",
        "wood": 2,
        "stone": 1,
        "maintenance_wood": 1,
        "maintenance_stone": 0,
        "maintenance_interval_s": 120.0,
    },
    "hut": {
        "name_zh": "小屋",
        "wood": 4,
        "stone": 2,
        "maintenance_wood": 1,
        "maintenance_stone": 1,
        "maintenance_interval_s": 180.0,
    },
    "well": {
        "name_zh": "水井",
        "wood": 1,
        "stone": 3,
        "maintenance_wood": 0,
        "maintenance_stone": 1,
        "maintenance_interval_s": 150.0,
    },
    "workshop": {
        "name_zh": "工坊",
        "wood": 3,
        "stone": 2,
        "maintenance_wood": 1,
        "maintenance_stone": 1,
        "maintenance_interval_s": 180.0,
    },
    "granary": {
        "name_zh": "粮仓",
        "wood": 2,
        "stone": 2,
        "maintenance_wood": 1,
        "maintenance_stone": 0,
        "maintenance_interval_s": 150.0,
    },
}


class PopulationLimitError(RuntimeError):
    """创建第 9 个 agent（或超过 ``max_agents``）时抛出。"""


# ---------------------------------------------------------------------------
# 历史档案
# ---------------------------------------------------------------------------
@dataclass
class AgentRecord:
    """一个个体（含已死亡个体）的完整历史档案，不随死亡删除。"""

    agent_id: int
    parent_id: int | None
    name: str
    sex: str
    generation: int
    born_at: float
    died_at: float | None = None
    death_cause: str = ""
    alive: bool = True
    final_state: dict[str, object] = field(default_factory=dict)
    inventory: dict[str, int] = field(default_factory=dict)
    known_knowledge_ids: tuple[str, ...] = ()
    skills: dict[str, float] = field(default_factory=dict)
    children: tuple[int, ...] = ()
    contributions: dict[str, float] = field(default_factory=dict)
    memory_archive: dict[str, object] = field(default_factory=dict)

    @property
    def cause(self) -> str:
        """``death_cause`` 的别名（便于集成层按"cause"读死亡原因）。"""
        return self.death_cause

    def to_dict(self) -> dict[str, object]:
        """转为可 JSON 序列化的 dict。"""
        return {
            "agent_id": int(self.agent_id),
            "parent_id": None if self.parent_id is None else int(self.parent_id),
            "name": str(self.name),
            "sex": str(self.sex),
            "generation": int(self.generation),
            "born_at": float(self.born_at),
            "died_at": None if self.died_at is None else float(self.died_at),
            "death_cause": str(self.death_cause),
            "alive": bool(self.alive),
            "final_state": dict(self.final_state),
            "inventory": {str(key): int(value) for key, value in self.inventory.items()},
            "known_knowledge_ids": [str(item) for item in self.known_knowledge_ids],
            "skills": {str(key): float(value) for key, value in self.skills.items()},
            "children": [int(child) for child in self.children],
            "contributions": {str(key): float(value) for key, value in self.contributions.items()},
            "memory_archive": dict(self.memory_archive),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentRecord":
        """从 :meth:`to_dict` 的结果还原（缺失字段用默认值，容忍旧存档）。"""
        if not isinstance(data, dict):
            raise TypeError("AgentRecord.from_dict 需要 dict")
        died_at = data.get("died_at")
        parent_id = data.get("parent_id")
        return cls(
            agent_id=int(data.get("agent_id", -1)),
            parent_id=None if parent_id is None else int(parent_id),
            name=str(data.get("name", "")),
            sex=str(data.get("sex", "male")),
            generation=int(data.get("generation", 0)),
            born_at=float(data.get("born_at", 0.0)),
            died_at=None if died_at is None else float(died_at),
            death_cause=str(data.get("death_cause", "")),
            alive=bool(data.get("alive", True)),
            final_state=dict(data.get("final_state", {}) or {}),
            inventory={
                str(key): int(value) for key, value in (data.get("inventory", {}) or {}).items()
            },
            known_knowledge_ids=tuple(
                str(item) for item in (data.get("known_knowledge_ids", ()) or ())
            ),
            skills={
                str(key): float(value) for key, value in (data.get("skills", {}) or {}).items()
            },
            children=tuple(int(child) for child in (data.get("children", ()) or ())),
            contributions={
                str(key): float(value)
                for key, value in (data.get("contributions", {}) or {}).items()
            },
            memory_archive=dict(data.get("memory_archive", {}) or {}),
        )


# ---------------------------------------------------------------------------
# 注册表：8 个创建名额的硬约束 + 死亡档案
# ---------------------------------------------------------------------------
class AgentRegistry:
    """受约束的创建路径：``total_created`` 只增不减，死亡只标记不删除。"""

    def __init__(self, max_agents: int = MAX_AGENTS_CREATED) -> None:
        """初始化空注册表；``max_agents`` 为历史累计创建上限。"""
        self.max_agents = max(0, int(max_agents))
        self._records: dict[int, AgentRecord] = {}
        self._total_created = 0
        self._total_deaths = 0

    # -- 只读属性 / 容量 ------------------------------------------------
    @property
    def total_created(self) -> int:
        """历史累计创建数（只增不减，含已死亡个体）。"""
        return self._total_created

    @property
    def total_deaths(self) -> int:
        """历史累计死亡数（只增不减）。"""
        return self._total_deaths

    def remaining(self) -> int:
        """剩余创建名额。"""
        return max(0, self.max_agents - self._total_created)

    def can_create(self) -> bool:
        """是否还有创建名额。"""
        return self._total_created < self.max_agents

    # -- 登记 / 查询 ----------------------------------------------------
    def register(self, agent, *, t: float, parent_id: int | None = None) -> AgentRecord:
        """登记一个新个体（唯一受约束的创建入口）。

        ``agent`` 需要具备 ``agent_id / name / sex / generation / parents``
        属性；``parent_id`` 省略时从 ``agent.parents`` 取第一个父/母 id。
        满额抛 :class:`PopulationLimitError` 且计数不变；同一 ``agent_id``
        重复登记抛 :class:`ValueError`。
        """
        agent_id = int(_attr(agent, "agent_id", -1))
        if agent_id < 0:
            raise ValueError("agent 缺少合法的 agent_id，无法登记")
        if agent_id in self._records:
            raise ValueError(f"agent_id {agent_id} 已登记，不能重复登记")
        if not self.can_create():
            raise PopulationLimitError(
                f"创建名额已用尽：total_created={self._total_created} / max_agents={self.max_agents}"
            )
        parents = _safe_tuple(_attr(agent, "parents", ()))
        if parent_id is None and parents:
            parent_id = int(parents[0])
        parent_id = None if parent_id is None else int(parent_id)
        contributions = _attr(agent, "contributions", None)
        record = AgentRecord(
            agent_id=agent_id,
            parent_id=parent_id,
            name=str(_attr(agent, "name", "") or ""),
            sex=str(_attr(agent, "sex", "male") or "male"),
            generation=int(_attr(agent, "generation", 0) or 0),
            born_at=float(t),
            alive=True,
            contributions=_float_dict(contributions),
        )
        self._records[agent_id] = record
        self._total_created += 1
        parent_record = self._records.get(parent_id) if parent_id is not None else None
        if parent_record is not None and agent_id not in parent_record.children:
            parent_record.children = parent_record.children + (agent_id,)
        return record

    def record(self, agent_id: int) -> AgentRecord | None:
        """按 id 取档案（含已死亡），不存在返回 ``None``。"""
        return self._records.get(int(agent_id))

    def records(self) -> list[AgentRecord]:
        """全部档案（按 ``agent_id`` 升序，含已死亡个体）。"""
        return [self._records[key] for key in sorted(self._records)]

    def alive_ids(self) -> list[int]:
        """存活个体 id（升序，不含已死亡个体）。"""
        return sorted(key for key, record in self._records.items() if record.alive)

    def alive_count(self) -> int:
        """当前存活个体数。"""
        return sum(1 for record in self._records.values() if record.alive)

    def mark_dead(
        self,
        agent_id: int,
        *,
        t: float,
        cause: str,
        final_state: dict | None = None,
    ) -> AgentRecord | None:
        """标记死亡并保留档案；已死亡个体再次标记为幂等（不重复计数）。

        ``final_state`` 中若含 ``inventory / known_knowledge_ids / skills /
        memory_archive`` 这几个键，会被分别吸收进同名档案字段，其余键原样保留在
        ``AgentRecord.final_state``。
        """
        record = self._records.get(int(agent_id))
        if record is None:
            return None
        if not record.alive:
            return record
        record.alive = False
        record.died_at = float(t)
        record.death_cause = str(cause)
        if final_state is not None:
            snapshot = dict(final_state)
            inventory = snapshot.pop("inventory", None)
            if isinstance(inventory, dict):
                record.inventory = {str(key): int(value) for key, value in inventory.items()}
            known = snapshot.pop("known_knowledge_ids", None)
            if known is not None:
                record.known_knowledge_ids = tuple(str(item) for item in known)
            skills = snapshot.pop("skills", None)
            if isinstance(skills, dict):
                record.skills = {str(key): float(value) for key, value in skills.items()}
            memory_archive = snapshot.pop("memory_archive", None)
            if isinstance(memory_archive, dict):
                record.memory_archive = memory_archive
            record.final_state = snapshot
        self._total_deaths += 1
        return record

    def lineage(self) -> list[tuple[int, int]]:
        """亲子关系列表 ``(parent_id, child_id)``（由 ``parent_id`` 与 ``children`` 双向合并）。"""
        pairs: set[tuple[int, int]] = set()
        for record in self._records.values():
            if record.parent_id is not None:
                pairs.add((int(record.parent_id), int(record.agent_id)))
            for child in record.children:
                pairs.add((int(record.agent_id), int(child)))
        return sorted(pairs)

    # -- 序列化 ---------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """导出全部档案与计数（JSON 可序列化）。"""
        return {
            "max_agents": int(self.max_agents),
            "total_created": int(self._total_created),
            "total_deaths": int(self._total_deaths),
            "records": [record.to_dict() for record in self.records()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentRegistry":
        """从 :meth:`to_dict` 的结果还原。"""
        if not isinstance(data, dict):
            raise TypeError("AgentRegistry.from_dict 需要 dict")
        registry = cls(max_agents=int(data.get("max_agents", MAX_AGENTS_CREATED)))
        for item in data.get("records", ()) or ():
            record = AgentRecord.from_dict(item)
            registry._records[int(record.agent_id)] = record
        registry._total_created = max(
            int(data.get("total_created", len(registry._records))), len(registry._records)
        )
        registry._total_deaths = int(
            data.get(
                "total_deaths",
                sum(1 for record in registry._records.values() if not record.alive),
            )
        )
        return registry


# ---------------------------------------------------------------------------
# 公共设施
# ---------------------------------------------------------------------------
@dataclass
class Facility:
    """一个公共设施：``condition`` 随世界时间线性损耗，维护可恢复。"""

    facility_id: str
    kind: str
    name_zh: str
    x_screen: float
    condition: float = 1.0
    built_at: float = 0.0
    built_by: int = -1
    maintenance_wood: int = 0
    maintenance_stone: int = 0
    maintenance_interval_s: float = 120.0
    last_maintained_at: float = 0.0
    private_uses: int = 0

    def __post_init__(self) -> None:
        """初始化损耗锚点与维护者追踪（非冻结字段）。"""
        #: 损耗计算的锚点（上次 decay / 维护时刻），不属于冻结字段
        self._decay_anchor: float = float(self.built_at)
        #: 最近一次维护者 id（用于追踪），不属于冻结字段
        self._last_maintained_by: int = -1

    def decay(self, t: float) -> float:
        """推进自然损耗并返回新的 ``condition``（不维护则下降，下限 0）。"""
        try:
            moment = float(t)
        except (TypeError, ValueError):
            return float(self.condition)
        elapsed = max(0.0, moment - float(self._decay_anchor))
        if elapsed > 0.0:
            self.condition = max(
                0.0,
                min(1.0, float(self.condition) - FACILITY_DECAY_PER_SECOND * elapsed),
            )
            self._decay_anchor = max(float(self._decay_anchor), moment)
        return float(self.condition)

    def maintained(self, agent_id: int, t: float) -> None:
        """维护：提升 ``condition`` 并重置损耗锚点 / ``last_maintained_at``。"""
        self.condition = min(1.0, max(0.0, float(self.condition) + FACILITY_MAINTENANCE_BOOST))
        try:
            moment = float(t)
        except (TypeError, ValueError):
            moment = self._decay_anchor
        self.last_maintained_at = moment
        self._decay_anchor = moment
        self._last_maintained_by = int(agent_id)

    @property
    def active(self) -> bool:
        """是否活跃：``condition > FACILITY_ACTIVE_THRESHOLD (0.2)``。"""
        return float(self.condition) > FACILITY_ACTIVE_THRESHOLD

    def to_dict(self) -> dict[str, object]:
        """转为可 JSON 序列化的 dict。"""
        return {
            "facility_id": str(self.facility_id),
            "kind": str(self.kind),
            "name_zh": str(self.name_zh),
            "x_screen": float(self.x_screen),
            "condition": float(self.condition),
            "built_at": float(self.built_at),
            "built_by": int(self.built_by),
            "maintenance_wood": int(self.maintenance_wood),
            "maintenance_stone": int(self.maintenance_stone),
            "maintenance_interval_s": float(self.maintenance_interval_s),
            "last_maintained_at": float(self.last_maintained_at),
            "private_uses": int(self.private_uses),
            "decay_anchor": float(self._decay_anchor),
            "last_maintained_by": int(self._last_maintained_by),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Facility":
        """从 :meth:`to_dict` 的结果还原。"""
        if not isinstance(data, dict):
            raise TypeError("Facility.from_dict 需要 dict")
        facility = cls(
            facility_id=str(data.get("facility_id", "")),
            kind=str(data.get("kind", "")),
            name_zh=str(data.get("name_zh", "")),
            x_screen=float(data.get("x_screen", 0.0)),
            condition=float(data.get("condition", 1.0)),
            built_at=float(data.get("built_at", 0.0)),
            built_by=int(data.get("built_by", -1)),
            maintenance_wood=int(data.get("maintenance_wood", 0)),
            maintenance_stone=int(data.get("maintenance_stone", 0)),
            maintenance_interval_s=float(data.get("maintenance_interval_s", 120.0)),
            last_maintained_at=float(data.get("last_maintained_at", 0.0)),
            private_uses=int(data.get("private_uses", 0)),
        )
        anchor = data.get("decay_anchor")
        facility._decay_anchor = float(anchor) if anchor is not None else float(facility.built_at)
        facility._last_maintained_by = int(data.get("last_maintained_by", -1))
        return facility


# ---------------------------------------------------------------------------
# 社会契约（规则实例）
# ---------------------------------------------------------------------------
@dataclass
class SocialContract:
    """一条由个体提出并被参与者执行的规则（共享 / 公共储藏 / 师徒）。"""

    contract_id: str
    kind: str
    name_zh: str
    proposed_by: int
    created_at: float
    participants: tuple[int, ...] = ()
    active: bool = True

    def to_dict(self) -> dict[str, object]:
        """转为可 JSON 序列化的 dict。"""
        return {
            "contract_id": str(self.contract_id),
            "kind": str(self.kind),
            "name_zh": str(self.name_zh),
            "proposed_by": int(self.proposed_by),
            "created_at": float(self.created_at),
            "participants": [int(item) for item in self.participants],
            "active": bool(self.active),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SocialContract":
        """从 :meth:`to_dict` 的结果还原。"""
        if not isinstance(data, dict):
            raise TypeError("SocialContract.from_dict 需要 dict")
        return cls(
            contract_id=str(data.get("contract_id", "")),
            kind=str(data.get("kind", "")),
            name_zh=str(data.get("name_zh", "")),
            proposed_by=int(data.get("proposed_by", -1)),
            created_at=float(data.get("created_at", 0.0)),
            participants=tuple(int(item) for item in (data.get("participants", ()) or ())),
            active=bool(data.get("active", True)),
        )


# ---------------------------------------------------------------------------
# 文明状态
# ---------------------------------------------------------------------------
class CivilizationState:
    """文明级状态：人口历史、设施、规则、合作指数与 collapse 状态机。

    ``knowledge`` / ``memory`` 是 duck-typed 依赖（可为 ``None``）：
    ``knowledge`` 只需能被读出"已验证知识数量"；``memory`` 只需提供
    ``entries(agent_id)`` 与 ``archive(agent_id)``。``config``（可选）只读
    ``enable_social`` / ``enable_inheritance`` 两个开关（属性或 dict 均可），
    缺省视为开启，保证关闭机制时行为保守（合作指数 0）。
    """

    def __init__(self, registry: AgentRegistry, knowledge=None, memory=None, config=None) -> None:
        """初始化文明状态（知识/记忆为鸭子类型依赖，可为 None）。"""
        self.registry = registry
        self.knowledge = knowledge
        self.memory = memory
        self.config = config
        self.facilities: list[Facility] = []
        self.contracts: list[SocialContract] = []
        #: 已验证知识数量（从 ``knowledge`` 同步）
        self.innovation_count = int(self._knowledge_verified_count())
        #: 合作指数（公式见模块 docstring / §8）
        self.cooperation_index = 0.0
        #: stable | stagnating | collapsing | collapsed | recovering
        self.collapse_state = "stable"
        self.extinct = False
        self.extinct_at: float | None = None
        self.extinction_cause = ""
        #: 每个统计窗口一条记录（≤ HISTORY_LIMIT）
        self.history: list[dict[str, object]] = []

        self._t = 0.0
        self._last_tech_change_t = 0.0
        self._last_innovation_count = self.innovation_count
        self._last_death_cause = ""
        self._avg_energy = 0.0
        self._pop_change_ratio = 0.0
        self._history_accum = 0.0
        #: (世界时间, 存活数) 采样，用于人口变化率
        self._pop_samples: list[tuple[float, int]] = [(0.0, int(self.registry.alive_count()))]

    # ------------------------------------------------------------------
    # 每步更新
    # ------------------------------------------------------------------
    def update(self, world, dt: float) -> None:
        """推进一个统计步：设施损耗 → 合作/技术同步 → 灭绝检测 → 状态机 → history。"""
        step = float(dt) if isinstance(dt, (int, float)) and not isinstance(dt, bool) else 0.0
        step = max(0.0, step)
        t = self._world_time(world, step)
        self._t = max(self._t, t)
        # 1) 设施损耗 / 失效
        for facility in self.facilities:
            facility.decay(self._t)
        # 2) 同步 agent 贡献并重算合作指数
        self._sync_contributions(world)
        self.cooperation_index = self._compute_cooperation_index()
        # 3) 同步 innovation_count（刷新技术停滞计时）
        self._sync_innovation(self._t)
        # 4) 存活 / 平均能量 / 人口变化率
        alive = self.registry.alive_count()
        avg_energy = self._average_energy(world, alive)
        self._avg_energy = avg_energy
        self._pop_change_ratio = self._population_change_ratio(self._t, alive)
        # 5) 灭绝检测（§3）
        if self._should_be_extinct(alive):
            self._set_extinct(self._t)
        # 6) collapse_state 状态机
        self.collapse_state = self._classify_collapse(alive, avg_energy)
        # 7) history 窗口记录
        self._history_accum += step
        if not self.history or self._history_accum >= HISTORY_WINDOW_S:
            self._append_history(alive, avg_energy)
            self._history_accum = 0.0
        # 8) 人口采样（保留最近 4 个窗口，足够算参照）
        self._pop_samples.append((self._t, int(alive)))
        if len(self._pop_samples) > 2048:
            self._pop_samples = self._pop_samples[-2048:]

    def register_death(self, agent, *, t, cause) -> AgentRecord | None:
        """登记死亡：保留档案 + 记忆档案快照，并立即刷新灭绝判定。"""
        agent_id = _attr(agent, "agent_id", None)
        if agent_id is None:
            return None
        record = self.registry.mark_dead(
            int(agent_id),
            t=float(t),
            cause=str(cause),
            final_state=self._snapshot_agent(agent),
        )
        if record is None:
            return None
        self._last_death_cause = str(cause)
        if self.memory is not None:
            try:
                archive = self.memory.archive(int(agent_id))
            except Exception:
                archive = None
            if isinstance(archive, dict):
                record.memory_archive = archive
        if self._should_be_extinct(self.registry.alive_count()):
            self._set_extinct(float(t))
            self.collapse_state = "collapsed"
        return record

    # ------------------------------------------------------------------
    # 公共设施
    # ------------------------------------------------------------------
    def build_facility(self, kind, x_screen, agent_id, t, *, inventory=None) -> Facility | None:
        """按世界规则建造一个公共设施；材料不足 / 未知类型返回 ``None``。

        ``inventory`` 传入建造者库存对象（有可写的 ``wood`` / ``stone`` 属性），
        提供时本方法负责扣除建造消耗；``None`` 时不做材料校验（测试 / 预置用）。
        """
        spec = FACILITY_KINDS.get(str(kind))
        if spec is None:
            return None
        cost_wood = int(spec["wood"])
        cost_stone = int(spec["stone"])
        if inventory is not None:
            wood = int(_attr(inventory, "wood", 0) or 0)
            stone = int(_attr(inventory, "stone", 0) or 0)
            if wood < cost_wood or stone < cost_stone:
                return None
            try:
                inventory.wood = wood - cost_wood
                inventory.stone = stone - cost_stone
            except Exception:
                return None
        facility = Facility(
            facility_id=f"{kind}-{len(self.facilities) + 1:02d}",
            kind=str(kind),
            name_zh=str(spec["name_zh"]),
            x_screen=float(x_screen),
            condition=1.0,
            built_at=float(t),
            built_by=int(agent_id),
            maintenance_wood=int(spec["maintenance_wood"]),
            maintenance_stone=int(spec["maintenance_stone"]),
            maintenance_interval_s=float(spec["maintenance_interval_s"]),
            last_maintained_at=float(t),
            private_uses=0,
        )
        self.facilities.append(facility)
        return facility

    def maintain(self, facility_id, agent_id, t, *, wood=0, stone=0) -> bool:
        """维护设施：消耗额不足返回 ``False``；成功则恢复 condition 并计贡献。

        ``wood`` / ``stone`` 是调用方**已从库存扣除**的资源量，本方法只校验是否
        达到 ``maintenance_wood`` / ``maintenance_stone`` 的要求。
        """
        facility = next(
            (item for item in self.facilities if item.facility_id == str(facility_id)), None
        )
        if facility is None:
            return False
        try:
            paid_wood = int(wood)
            paid_stone = int(stone)
        except (TypeError, ValueError):
            return False
        if paid_wood < facility.maintenance_wood or paid_stone < facility.maintenance_stone:
            return False
        facility.maintained(int(agent_id), float(t))
        record = self.registry.record(int(agent_id))
        if record is not None and record.alive:
            record.contributions["maintain"] = float(record.contributions.get("maintain", 0.0)) + 1.0
        return True

    # ------------------------------------------------------------------
    # 指标 / 摘要
    # ------------------------------------------------------------------
    def metrics(self, world) -> dict[str, float]:
        """§8 文明级指标；所有值都是**原生 float**（可直接 ``json.dumps``）。"""
        alive = int(self.registry.alive_count())
        conditions = [float(facility.condition) for facility in self.facilities]
        active = sum(1 for facility in self.facilities if facility.active)
        generation_max = max((record.generation for record in self.registry.records()), default=0)
        return {
            "alive_count": float(alive),
            "alive_generation_max": float(generation_max),
            "total_agents_created": float(self.registry.total_created),
            "total_deaths": float(self.registry.total_deaths),
            "facilities_total": float(len(self.facilities)),
            "facilities_active": float(active),
            "facility_condition_mean": (
                float(sum(conditions) / len(conditions)) if conditions else 0.0
            ),
            "contracts_active": float(sum(1 for contract in self.contracts if contract.active)),
            "innovation_count": float(self.innovation_count),
            "tech_verified": float(self.innovation_count),
            "cooperation_index": float(self.cooperation_index),
            "collapse_state_code": float(COLLAPSE_STATE_CODES.get(self.collapse_state, 0.0)),
            "extinct": 1.0 if self.extinct else 0.0,
            "extinction_time": float(self.extinct_at) if self.extinct_at is not None else -1.0,
            "extinction_cause_known": 1.0 if self.extinction_cause else 0.0,
            "avg_energy": float(self._avg_energy),
            "population_change_ratio": float(self._pop_change_ratio),
            "history_len": float(len(self.history)),
        }

    def summary_zh(self) -> str:
        """中文单行摘要（供 GUI / 日志）。"""
        active = sum(1 for facility in self.facilities if facility.active)
        parts = [
            f"文明 存活{self.registry.alive_count()}/{self.registry.total_created}",
            f"死亡{self.registry.total_deaths}",
            f"设施{active}/{len(self.facilities)}",
            f"已验证技术{self.innovation_count}",
            f"合作{self.cooperation_index:.2f}",
            f"状态{COLLAPSE_STATE_ZH.get(self.collapse_state, self.collapse_state)}",
        ]
        if self.extinct:
            moment = "?" if self.extinct_at is None else f"{self.extinct_at:.1f}s"
            parts.append(f"已灭绝@{moment}（{self.extinction_cause or 'unknown'}）")
        return " · ".join(parts)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """导出全部状态（JSON 可序列化）。"""
        return {
            "registry": self.registry.to_dict(),
            "facilities": [facility.to_dict() for facility in self.facilities],
            "contracts": [contract.to_dict() for contract in self.contracts],
            "innovation_count": int(self.innovation_count),
            "last_innovation_count": int(self._last_innovation_count),
            "cooperation_index": float(self.cooperation_index),
            "collapse_state": str(self.collapse_state),
            "extinct": bool(self.extinct),
            "extinct_at": None if self.extinct_at is None else float(self.extinct_at),
            "extinction_cause": str(self.extinction_cause),
            "history": [dict(item) for item in self.history],
            "t": float(self._t),
            "last_tech_change_t": float(self._last_tech_change_t),
            "history_accum": float(self._history_accum),
            "avg_energy": float(self._avg_energy),
            "pop_change_ratio": float(self._pop_change_ratio),
            "last_death_cause": str(self._last_death_cause),
        }

    @classmethod
    def from_dict(cls, data: dict, knowledge=None, memory=None) -> "CivilizationState":
        """从 :meth:`to_dict` 的结果还原；``knowledge`` / ``memory`` 由调用方注入。"""
        if not isinstance(data, dict):
            raise TypeError("CivilizationState.from_dict 需要 dict")
        registry = AgentRegistry.from_dict(data.get("registry", {}) or {})
        state = cls(registry, knowledge=knowledge, memory=memory)
        state.facilities = [Facility.from_dict(item) for item in (data.get("facilities", ()) or ())]
        state.contracts = [
            SocialContract.from_dict(item) for item in (data.get("contracts", ()) or ())
        ]
        state.innovation_count = int(data.get("innovation_count", state.innovation_count))
        state._last_innovation_count = int(
            data.get("last_innovation_count", state.innovation_count)
        )
        state.cooperation_index = float(data.get("cooperation_index", 0.0))
        collapse_state = str(data.get("collapse_state", "stable"))
        state.collapse_state = collapse_state if collapse_state in COLLAPSE_STATES else "stable"
        state.extinct = bool(data.get("extinct", False))
        extinct_at = data.get("extinct_at")
        state.extinct_at = None if extinct_at is None else float(extinct_at)
        state.extinction_cause = str(data.get("extinction_cause", ""))
        state.history = [dict(item) for item in (data.get("history", ()) or ())][-HISTORY_LIMIT:]
        state._t = float(data.get("t", 0.0))
        state._last_tech_change_t = float(data.get("last_tech_change_t", state._t))
        state._history_accum = float(data.get("history_accum", 0.0))
        state._avg_energy = float(data.get("avg_energy", 0.0))
        state._pop_change_ratio = float(data.get("pop_change_ratio", 0.0))
        state._last_death_cause = str(data.get("last_death_cause", ""))
        state._pop_samples = [(state._t, int(registry.alive_count()))]
        return state

    # ------------------------------------------------------------------
    # 内部：政策开关
    # ------------------------------------------------------------------
    def _flag(self, name: str, default: bool) -> bool:
        """读取配置开关（``config`` 可为属性对象、dict 或 None）。"""
        config = self.config
        if config is None:
            return default
        if isinstance(config, dict):
            return bool(config.get(name, default))
        return bool(getattr(config, name, default))

    def _inheritance_enabled(self) -> bool:
        """继承机制（记忆/知识转移）是否开启。"""
        return self._flag("enable_inheritance", True)

    def _social_enabled(self) -> bool:
        """社会机制（合作/契约）是否开启。"""
        return self._flag("enable_social", True)

    # ------------------------------------------------------------------
    # 内部：世界时间 / 能量 / 人口变化率
    # ------------------------------------------------------------------
    def _world_time(self, world, dt: float) -> float:
        """读取世界时间（``t`` / ``time`` / ``now`` / ``sim_time``）；读不到则本地累加。"""
        for name in ("t", "time", "now", "sim_time"):
            value = _attr(world, name, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return self._t + max(0.0, float(dt))

    def _average_energy(self, world, alive: int) -> float:
        """存活个体 ``physiology.energy`` 均值；输入不足时给保守值。"""
        agents = [
            agent
            for agent in (_attr(world, "agents", ()) or ())
            if bool(_attr(agent, "alive", True))
        ]
        values: list[float] = []
        for agent in agents:
            physiology = _attr(agent, "physiology", None)
            energy = _attr(physiology, "energy", None)
            if isinstance(energy, (int, float)) and not isinstance(energy, bool):
                values.append(float(energy))
        if values:
            return float(sum(values) / len(values))
        if agents or alive > 0:
            # 有存活个体但字段缺失 → 中性保守值，避免误判崩塌
            return 50.0
        return 0.0

    def _population_change_ratio(self, t: float, alive: int) -> float:
        """``(alive_now - alive_ref) / max(1, alive_ref)``；样本不足 → 0.0。"""
        reference: int | None = None
        for sample_t, sample_alive in reversed(self._pop_samples):
            if sample_t <= t - POP_WINDOW_S:
                reference = int(sample_alive)
                break
        if reference is None:
            return 0.0
        if reference <= 0:
            return 0.0 if alive <= 0 else 1.0
        return float(alive - reference) / float(max(1, reference))

    # ------------------------------------------------------------------
    # 内部：灭绝 / collapse 状态机
    # ------------------------------------------------------------------
    def _should_be_extinct(self, alive: int) -> bool:
        """§3 灭绝规则：alive==0 且（名额用尽 或 不允许新生）。"""
        return int(alive) <= 0 and (
            not self.registry.can_create() or not self._inheritance_enabled()
        )

    def _set_extinct(self, t: float) -> None:
        """记录首次灭绝时间与原因（幂等，只写一次）。"""
        if self.extinct:
            return
        self.extinct = True
        self.extinct_at = float(t)
        self.extinction_cause = self._last_death_cause or "unknown"

    def _classify_collapse(self, alive: int, avg_energy: float) -> str:
        """按模块 docstring 的确定公式给出 collapse_state。"""
        previous = self.collapse_state
        if self.extinct or int(alive) <= 0:
            return "collapsed"
        ratio = float(self._pop_change_ratio)
        energy_ok = float(avg_energy) >= COLLAPSE_ENERGY_RECOVER
        facilities_ok = (not self.facilities) or any(
            facility.active for facility in self.facilities
        )
        if previous in ("collapsing", "collapsed", "recovering"):
            if ratio > 0.0 and energy_ok and facilities_ok:
                return "recovering"
            if previous in ("collapsing", "recovering") and ratio <= 0.0:
                return "collapsing"
        if ratio <= COLLAPSE_POP_DROP_RATIO or float(avg_energy) < COLLAPSE_ENERGY_CRITICAL:
            return "collapsing"
        stagnation_s = self._t - float(self._last_tech_change_t)
        if stagnation_s >= STAGNATION_S and ratio <= 0.0:
            return "stagnating"
        return "stable"

    # ------------------------------------------------------------------
    # 内部：知识 / 合作 / 贡献同步
    # ------------------------------------------------------------------
    def _knowledge_verified_count(self) -> int:
        """从 ``knowledge`` 读取已验证知识数量（duck typing，失败按 0）。"""
        knowledge = self.knowledge
        if knowledge is None:
            return 0
        try:
            all_method = getattr(knowledge, "all", None)
            if callable(all_method):
                try:
                    items = all_method(status="verified")
                except TypeError:
                    items = all_method()
                return int(len(items))
            if isinstance(knowledge, (list, tuple, set, frozenset, dict)):
                return int(len(knowledge))
            verified = getattr(knowledge, "verified", None)
            if verified is not None:
                return int(len(verified))
        except Exception:
            return 0
        return 0

    def _sync_innovation(self, t: float) -> None:
        """同步 ``innovation_count``；变化时刷新技术停滞计时。"""
        count = int(self._knowledge_verified_count())
        if count != self._last_innovation_count:
            self._last_tech_change_t = float(t)
        self.innovation_count = count
        self._last_innovation_count = count

    def _sync_contributions(self, world) -> None:
        """把 ``agent.contributions`` 同步进存活个体的 ``AgentRecord``。"""
        for agent in _attr(world, "agents", ()) or ():
            agent_id = _attr(agent, "agent_id", None)
            if agent_id is None:
                continue
            record = self.registry.record(int(agent_id))
            if record is None or not record.alive:
                continue
            contributions = _attr(agent, "contributions", None)
            if isinstance(contributions, dict):
                record.contributions = _float_dict(contributions)

    def _compute_cooperation_index(self) -> float:
        """按模块 docstring 的公式计算合作指数；关闭/无数据 → 0.0。

        与 docstring 的一处有意差异（集成后修订）：``coop_gain`` 统计**所有**
        ``AgentRecord``（含已死亡并归档的个体），而不是只统计存活个体。
        理由：合作是**文明级**的累计指标，任务书 §4.4 明确要求文明状态独立于
        某个 Agent 的短期状态；若只统计存活者，最后一个人死亡时会把它一生的
        合作贡献清零，指标会突然从 0.73 掉到 0，失去解释意义。
        """
        if not self._social_enabled():
            return 0.0
        gain = 0.0
        for record in self.registry.records():
            for value in record.contributions.values():
                try:
                    amount = float(value)
                except (TypeError, ValueError):
                    continue
                if amount > 0.0:
                    gain += amount
        for contract in self.contracts:
            if not contract.active:
                continue
            pairs = max(0, len(contract.participants) - 1)
            gain += float(pairs)
        independent = float(self._independent_event_count())
        if gain <= 0.0 and independent <= 0.0:
            return 0.0
        coverage = gain / (gain + independent + COOPERATION_SMOOTHING)
        return max(0.0, min(1.0, float(coverage)))

    def _independent_event_count(self) -> int:
        """存活个体在 MemoryStore 中成功的 ``episode`` 条目数（数据缺失 → 0）。"""
        memory = self.memory
        if memory is None:
            return 0
        total = 0
        for record in self.registry.records():
            if not record.alive:
                continue
            try:
                entries = memory.entries(record.agent_id)
            except Exception:
                return total
            for entry in entries:
                if _attr(entry, "kind", "") != "episode":
                    continue
                if bool(_attr(entry, "success", False)):
                    total += 1
        return total

    # ------------------------------------------------------------------
    # 内部：死亡快照 / history
    # ------------------------------------------------------------------
    def _snapshot_agent(self, agent) -> dict[str, object]:
        """死亡瞬间的状态快照（inventory / 知识 / 技能 / 最终生理与位置）。"""
        snapshot: dict[str, object] = {}
        inventory = _attr(agent, "inventory", None)
        if inventory is not None:
            snapshot["inventory"] = _inventory_snapshot(inventory)
        known = _attr(agent, "known_knowledge_ids", None)
        if known is not None:
            snapshot["known_knowledge_ids"] = tuple(str(item) for item in known)
        skills = _attr(agent, "skills", None)
        if isinstance(skills, dict):
            snapshot["skills"] = _float_dict(skills)
        physiology = _attr(agent, "physiology", None)
        for name in ("energy", "hydration", "sleep_pressure"):
            value = _attr(physiology, name, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                snapshot[name] = float(value)
        snapshot["age"] = float(_attr(agent, "age", 0.0) or 0.0)
        snapshot["x"] = float(_attr(agent, "x", 0.0) or 0.0)
        snapshot["generation"] = int(_attr(agent, "generation", 0) or 0)
        snapshot["name"] = str(_attr(agent, "name", "") or "")
        snapshot["sex"] = str(_attr(agent, "sex", "male") or "male")
        behavior_state = _attr(agent, "behavior_state", None)
        if behavior_state is not None:
            snapshot["behavior_state"] = str(behavior_state)
        return snapshot

    def _append_history(self, alive: int, avg_energy: float) -> None:
        """追加一条窗口记录（环形，最多 ``HISTORY_LIMIT`` 条）。"""
        active = sum(1 for facility in self.facilities if facility.active)
        record = {
            "t": float(self._t),
            "alive": int(alive),
            "total_created": int(self.registry.total_created),
            "deaths": int(self.registry.total_deaths),
            "innovation_count": int(self.innovation_count),
            "facilities_total": int(len(self.facilities)),
            "facilities_active": int(active),
            "avg_energy": float(avg_energy),
            "population_change_ratio": float(self._pop_change_ratio),
            "cooperation_index": float(self.cooperation_index),
            "collapse_state": str(self.collapse_state),
            "extinct": bool(self.extinct),
        }
        self.history.append(record)
        if len(self.history) > HISTORY_LIMIT:
            del self.history[: len(self.history) - HISTORY_LIMIT]


# ---------------------------------------------------------------------------
# 小工具（duck typing 安全读取）
# ---------------------------------------------------------------------------
def _attr(obj, name: str, default=None):
    """安全 ``getattr``：对象为 ``None`` 或属性访问抛错时返回默认值。"""
    if obj is None:
        return default
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _safe_tuple(value) -> tuple:
    """把可能为 ``None`` / list 的亲子字段规范成 tuple。"""
    if value is None:
        return ()
    try:
        return tuple(value)
    except TypeError:
        return ()


def _float_dict(value) -> dict[str, float]:
    """把来源不确定的 dict 规范成 ``dict[str, float]``（忽略非数值/异常）。"""
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            result[str(key)] = float(item)
    return result


def _inventory_snapshot(inventory) -> dict[str, int]:
    """把库存对象快照成 ``dict[str, int]``（材料数量；``tool`` 等非数值忽略）。"""
    if isinstance(inventory, dict):
        return {
            str(key): int(value)
            for key, value in inventory.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    snapshot: dict[str, int] = {}
    for name in ("wood", "stone", "fiber", "cord", "basket", "axe", "campfire", "hut", "well"):
        value = _attr(inventory, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            snapshot[name] = int(value)
    return snapshot
