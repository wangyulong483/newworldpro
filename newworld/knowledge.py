"""持久化知识库与技术依赖图（P2/P3）。

本模块实现 ``docs/civilization_design.md`` §5 的冻结契约：一个**只描述知识状态**的
数据层，负责「假设 → 验证 → 失效」的状态机、技术依赖图、知识留存率与创新效率，
以及完整的序列化。它不拥有世界规则。

设计原则（与 §0 一致，禁止违反）
--------------------------------
1. **世界规则唯一权威**：本模块**绝不**扣材料、改库存、建设施。任何"这次尝试是否
   成功"的判断只能由 ``world.technology.simulate_recipe(agent, recipe) -> (ok, reason)``
   给出（唯一合法验证入口，见 :meth:`KnowledgeStore.attempt`）。因此本模块只依赖
   duck-typed 的 ``world`` / ``agent``，不 import ``world`` / ``technology`` /
   ``population`` / ``memory`` / ``civilization``，避免循环依赖。
2. **反固定剧情**：``RECIPE_CATALOG`` 是"这个世界里存在哪些技术"的**世界规则**，
   不是剧情脚本。开局所有技术都是 ``hypothesis``（未被任何个体发现），是否被
   发现、验证、传播、遗失完全由 agent 行为与随机实验决定；允许技术永远不被发明。
3. **只有被验证的知识可规划**：``usable()`` 仅在 ``status == "verified"`` 时为真；
   ``hypothesis`` 只能用于 ``experiment`` 步骤（见 §4 ``VERBS``）。
4. **降级可用**：P1 阶段 ``world.technology`` 可能还没有 ``simulate_recipe``。
   :meth:`KnowledgeStore.attempt` 用 ``getattr`` 探测，缺失时返回
   ``(False, "no_simulator")`` 并累加 ``degraded_attempts``，**不**污染知识状态机
   （缺模拟器不是"技术有缺陷"的证据），从而让 P1 可以先跑通。

状态机与置信度
--------------
::

    hypothesis --连续 work 次成功--> verified        （终态之一，可再被使用）
    hypothesis --failed_uses >= 3--> invalidated     （终态，不可复活）
    verified   --failed_uses >= 3--> invalidated     （世界规则变化导致失效）
    any        --mark_invalid()----> invalidated     （环境失效，审计记录 agent_id=-1）

* ``work`` 来自 ``recipe.work``，指**连续**成功次数（失败会打断连击）。
* ``confidence`` 每次尝试都更新：

  - 成功：``c += (1 - c) * (0.25 + 0.4 * difficulty)``
    （越难的技术成功一次越有说服力）；
  - 失败：``c -= 0.15 + 0.35 * difficulty``，下限 0
    （越难的技术失败一次更伤信心）；
  - 首次转为 ``verified`` 时置信度不低于 ``VERIFIED_CONFIDENCE_FLOOR``。

依赖图语义
----------
一条边 ``(parent, child)`` 表示"``parent`` 是 ``child`` 的前置技术"。
边的来源有两种，**都必须**能被查到：

* ``Knowledge.prerequisites``（知识自身声明的前置）；
* ``recipe.requires_knowledge``（配方声明的"没有它就做不出来"的能力）。

:meth:`KnowledgeStore.edges` 返回全部已登记知识的边（用于全图观测），
:meth:`KnowledgeStore.verified_edges` / :meth:`KnowledgeStore.depth` 只在
``verified`` 子图上工作（``depth`` = 已验证子图最长依赖链的**边数**）。
:meth:`KnowledgeStore.dependents` 返回传递闭包（直接 + 间接依赖者），
另有 :meth:`KnowledgeStore.direct_dependents` 取直接依赖者。

序列化
------
:meth:`KnowledgeStore.to_dict` / :meth:`KnowledgeStore.from_dict` 往返一致，
包含技术目录、每项知识的完整 ``validation_history``、降级计数与学习日志；
``from_dict(to_dict()) == 原状态`` 是测试硬要求。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "KNOWLEDGE_TYPES",
    "KNOWLEDGE_STATUS",
    "MAX_FAILED_USES",
    "NO_SIMULATOR_REASON",
    "VERIFIED_CONFIDENCE_FLOOR",
    "Recipe",
    "ValidationRecord",
    "Knowledge",
    "RECIPE_CATALOG",
    "RECIPE_KNOWLEDGE_TYPES",
    "KnowledgeStore",
]

#: 知识类别（与契约一致）
KNOWLEDGE_TYPES: tuple[str, ...] = ("tool", "technique", "structure", "resource")
#: 知识状态（与契约一致；``invalidated`` 是终态）
KNOWLEDGE_STATUS: tuple[str, ...] = ("hypothesis", "verified", "invalidated")

#: 失败尝试累计到该次数即判定技术失效（契约：failed_uses >= 3 → invalidated）
MAX_FAILED_USES: int = 3
#: 没有世界模拟器时 ``attempt`` 返回的原因码（P1 降级路径）
NO_SIMULATOR_REASON: str = "no_simulator"
#: 世界模拟器返回的**前置条件**失败原因码（集成层 Technology 使用）：
#: 这些不是"技术有缺陷"的证据，不能累计 ``failed_uses``，否则材料暂缺会把
#: 一条本来有效的技术误判为 failed_uses>=3 → invalidated。
PRECONDITION_REASONS: frozenset[str] = frozenset(
    {
        "missing_material",
        "missing_tool",
        "too_far_from_site",
        "insufficient_input",
        "no_site",
        "inventory_full",
    }
)
#: 刚转为 verified 时的置信度下限
VERIFIED_CONFIDENCE_FLOOR: float = 0.6

#: 成功一次对置信度的增益（与 difficulty 线性相关）
SUCCESS_CONFIDENCE_BASE: float = 0.25
SUCCESS_CONFIDENCE_PER_DIFFICULTY: float = 0.4
#: 失败一次对置信度的惩罚（与 difficulty 线性相关）
FAILURE_CONFIDENCE_BASE: float = 0.15
FAILURE_CONFIDENCE_PER_DIFFICULTY: float = 0.35

#: 学习/失效日志上限（防止长跑时无界增长）
LEARN_LOG_LIMIT: int = 1024
#: 序列化结构版本号
STORE_SCHEMA_VERSION: int = 1


def _clamp01(value: float) -> float:
    """把数值裁剪到 ``[0, 1]``。"""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


# ---------------------------------------------------------------------------
# 配方：世界规则的静态描述
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Recipe:
    """一条"这个世界允许怎样把材料变成产物"的规则。

    它是**世界规则**而不是剧情：``inputs`` / ``outputs`` 只声明材料键与数量，
    真正的扣减与产出由 ``world.technology.simulate_recipe`` 执行；本类从不修改
    任何库存。``difficulty`` 是 0..1 的"单次尝试暴露缺陷的概率"标度，由世界
    RNG 使用，本模块只把它用于置信度更新。

    参数
    ----
    recipe_id:
        稳定标识，同时用作 :class:`Knowledge` 的 ``knowledge_id``。
    name_zh:
        中文名（展示用）。
    inputs:
        材料键 -> 数量，例如 ``{"wood": 2, "stone": 1}``。
    outputs:
        材料/工具键 -> 数量，例如 ``{"axe": 1}``。
    requires_tool:
        需要的工具键（``None`` 表示徒手可做），由世界模拟器检查。
    requires_knowledge:
        必须先 ``verified`` 的前置技术 id（依赖图的第二种边来源）。
    work:
        连续成功多少次才 ``verified``。
    difficulty:
        0..1，难度；越大世界 RNG 越容易暴露缺陷。
    """

    recipe_id: str
    name_zh: str
    inputs: dict[str, int]
    outputs: dict[str, int]
    requires_tool: str | None = None
    requires_knowledge: tuple[str, ...] = ()
    work: int = 1
    difficulty: float = 0.5

    def __post_init__(self) -> None:
        """冻结 dataclass 的字段归一化：难度裁剪到 0..1，work 至少为 1。"""
        if self.difficulty < 0.0:
            object.__setattr__(self, "difficulty", 0.0)
        elif self.difficulty > 1.0:
            object.__setattr__(self, "difficulty", 1.0)
        if self.work < 1:
            object.__setattr__(self, "work", 1)

    def to_dict(self) -> dict[str, object]:
        """序列化为 JSON 可写的普通字典。"""
        return {
            "recipe_id": self.recipe_id,
            "name_zh": self.name_zh,
            "inputs": {str(key): int(value) for key, value in self.inputs.items()},
            "outputs": {str(key): int(value) for key, value in self.outputs.items()},
            "requires_tool": self.requires_tool,
            "requires_knowledge": list(self.requires_knowledge),
            "work": int(self.work),
            "difficulty": float(self.difficulty),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Recipe":
        """从 :meth:`to_dict` 的结果恢复。"""
        requires_knowledge = data.get("requires_knowledge") or ()
        requires_tool = data.get("requires_tool")
        return cls(
            recipe_id=str(data["recipe_id"]),
            name_zh=str(data.get("name_zh", data["recipe_id"])),
            inputs={str(key): int(value) for key, value in (data.get("inputs") or {}).items()},
            outputs={str(key): int(value) for key, value in (data.get("outputs") or {}).items()},
            requires_tool=None if requires_tool is None else str(requires_tool),
            requires_knowledge=tuple(str(item) for item in requires_knowledge),
            work=int(data.get("work", 1)),
            difficulty=float(data.get("difficulty", 0.5)),
        )


# ---------------------------------------------------------------------------
# 单次验证记录
# ---------------------------------------------------------------------------
@dataclass
class ValidationRecord:
    """一次尝试的审计记录（成功或失败）。"""

    t: float
    agent_id: int
    success: bool
    evidence: str

    def to_dict(self) -> dict[str, object]:
        """序列化为普通字典。"""
        return {
            "t": float(self.t),
            "agent_id": int(self.agent_id),
            "success": bool(self.success),
            "evidence": str(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ValidationRecord":
        """从 :meth:`to_dict` 的结果恢复。"""
        return cls(
            t=float(data.get("t", 0.0)),
            agent_id=int(data.get("agent_id", -1)),
            success=bool(data.get("success", False)),
            evidence=str(data.get("evidence", "")),
        )


# ---------------------------------------------------------------------------
# 知识条目：状态机持有者
# ---------------------------------------------------------------------------
@dataclass
class Knowledge:
    """一项技术的"社会状态"：谁发现的、当前是假设还是已验证、置信度多少。

    本类**不**执行配方，只根据 ``world.technology.simulate_recipe`` 的结论更新
    状态；``record`` 是唯一的状态迁移入口。
    """

    knowledge_id: str
    knowledge_type: str
    name_zh: str
    prerequisites: tuple[str, ...] = ()
    recipe: Recipe | None = None
    status: str = "hypothesis"
    confidence: float = 0.0
    discovered_by: int = -1
    discovered_at: float = 0.0
    successful_uses: int = 0
    failed_uses: int = 0
    validation_history: list[ValidationRecord] = field(default_factory=list)

    # -- 查询 ---------------------------------------------------------
    def usable(self) -> bool:
        """是否可被规划器直接用于 ``craft`` / ``build``（只有已验证才行）。"""
        return self.status == "verified"

    def required_work(self) -> int:
        """连续成功多少次才验证（来自 ``recipe.work``，无配方时按 1 次）。"""
        if self.recipe is None:
            return 1
        return max(1, int(self.recipe.work))

    def consecutive_successes(self) -> int:
        """当前连续成功次数（从验证历史尾部往前数）。"""
        streak = 0
        for entry in reversed(self.validation_history):
            if not entry.success:
                break
            streak += 1
        return streak

    def dependency_ids(self) -> tuple[str, ...]:
        """合并边来源：``prerequisites`` + ``recipe.requires_knowledge``（去重保序）。"""
        merged: list[str] = []
        for identifier in self.prerequisites:
            if identifier not in merged:
                merged.append(identifier)
        if self.recipe is not None:
            for identifier in self.recipe.requires_knowledge:
                if identifier not in merged:
                    merged.append(identifier)
        return tuple(merged)

    # -- 状态迁移 -----------------------------------------------------
    def record(self, t: float, agent_id: int, success: bool, evidence: str = "") -> None:
        """记录一次世界给出的验证结论并推进状态机。

        参数
        ----
        t:
            世界时间（秒）。
        agent_id:
            尝试者；``-1`` 预留给环境级失效审计（见 ``mark_invalid``）。
        success:
            世界模拟器返回的 ``ok``。
        evidence:
            世界模拟器返回的 ``reason`` 或人类可读证据。

        说明
        ----
        * 只有 ``hypothesis`` 才会因连续 ``work`` 次成功转为 ``verified``；
          ``invalidated`` 是终态，不会被后续成功"复活"。
        * 任意状态累计 ``failed_uses >= MAX_FAILED_USES`` 都会失效。
        """
        self.validation_history.append(
            ValidationRecord(
                t=float(t),
                agent_id=int(agent_id),
                success=bool(success),
                evidence=str(evidence),
            )
        )
        difficulty = self.recipe.difficulty if self.recipe is not None else 0.5
        if success:
            self.successful_uses += 1
            gain = SUCCESS_CONFIDENCE_BASE + SUCCESS_CONFIDENCE_PER_DIFFICULTY * difficulty
            self.confidence = _clamp01(self.confidence + (1.0 - self.confidence) * gain)
            if self.status == "hypothesis" and self.consecutive_successes() >= self.required_work():
                self.status = "verified"
                self.confidence = max(self.confidence, VERIFIED_CONFIDENCE_FLOOR)
        else:
            self.failed_uses += 1
            penalty = FAILURE_CONFIDENCE_BASE + FAILURE_CONFIDENCE_PER_DIFFICULTY * difficulty
            self.confidence = _clamp01(self.confidence - penalty)
            if self.failed_uses >= MAX_FAILED_USES and self.status != "invalidated":
                self.status = "invalidated"

    # -- 序列化 -------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """序列化为 JSON 可写的普通字典（含完整 ``validation_history``）。"""
        return {
            "knowledge_id": self.knowledge_id,
            "knowledge_type": self.knowledge_type,
            "name_zh": self.name_zh,
            "prerequisites": list(self.prerequisites),
            "recipe": None if self.recipe is None else self.recipe.to_dict(),
            "status": self.status,
            "confidence": float(self.confidence),
            "discovered_by": int(self.discovered_by),
            "discovered_at": float(self.discovered_at),
            "successful_uses": int(self.successful_uses),
            "failed_uses": int(self.failed_uses),
            "validation_history": [entry.to_dict() for entry in self.validation_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Knowledge":
        """从 :meth:`to_dict` 的结果恢复（容忍旧版本缺少可选键）。"""
        recipe_data = data.get("recipe")
        recipe = None if recipe_data is None else Recipe.from_dict(dict(recipe_data))
        prerequisites = data.get("prerequisites") or ()
        history = data.get("validation_history") or ()
        return cls(
            knowledge_id=str(data["knowledge_id"]),
            knowledge_type=str(data.get("knowledge_type", "technique")),
            name_zh=str(data.get("name_zh", data["knowledge_id"])),
            prerequisites=tuple(str(item) for item in prerequisites),
            recipe=recipe,
            status=str(data.get("status", "hypothesis")),
            confidence=_clamp01(float(data.get("confidence", 0.0))),
            discovered_by=int(data.get("discovered_by", -1)),
            discovered_at=float(data.get("discovered_at", 0.0)),
            successful_uses=int(data.get("successful_uses", 0)),
            failed_uses=int(data.get("failed_uses", 0)),
            validation_history=[ValidationRecord.from_dict(dict(entry)) for entry in history],
        )


# ---------------------------------------------------------------------------
# 世界规则：技术目录（不是剧情表）
# ---------------------------------------------------------------------------
#: 世界里"存在但未被发现"的全部技术。每条都真实改变世界数值（采集产量、负重上限、
#: 夜间恢复、休息/存储、取水半径），具体效果由 ``technology.py`` 的世界模拟器实现；
#: 本模块只声明规则与依赖，不执行效果。
RECIPE_CATALOG: dict[str, Recipe] = {
    "stone_axe": Recipe(
        recipe_id="stone_axe",
        name_zh="石斧",
        inputs={"wood": 2, "stone": 1},
        outputs={"axe": 1},
        requires_knowledge=(),
        work=1,
        difficulty=0.2,
    ),
    "fiber_cord": Recipe(
        recipe_id="fiber_cord",
        name_zh="纤维绳",
        inputs={"fiber": 3},
        outputs={"cord": 1},
        requires_knowledge=(),
        work=2,
        difficulty=0.3,
    ),
    "basket": Recipe(
        recipe_id="basket",
        name_zh="背篓",
        inputs={"wood": 2, "cord": 1},
        outputs={"basket": 1},
        requires_knowledge=("fiber_cord",),
        work=2,
        difficulty=0.4,
    ),
    "campfire": Recipe(
        recipe_id="campfire",
        name_zh="篝火",
        inputs={"wood": 2, "stone": 1},
        outputs={"campfire": 1},
        requires_knowledge=("stone_axe",),
        work=1,
        difficulty=0.3,
    ),
    "hut": Recipe(
        recipe_id="hut",
        name_zh="小屋",
        inputs={"wood": 4, "stone": 2, "cord": 1},
        outputs={"hut": 1},
        requires_knowledge=("basket",),
        work=3,
        difficulty=0.5,
    ),
    "well": Recipe(
        recipe_id="well",
        name_zh="水井",
        inputs={"stone": 3, "cord": 1},
        outputs={"well": 1},
        requires_knowledge=("fiber_cord",),
        work=2,
        difficulty=0.4,
    ),
    "granary": Recipe(
        recipe_id="granary",
        name_zh="粮仓",
        inputs={"wood": 2, "stone": 2},
        outputs={"granary": 1},
        requires_knowledge=("stone_axe",),
        work=1,
        difficulty=0.25,
    ),
}

#: recipe_id -> ``KNOWLEDGE_TYPES`` 之一（``Recipe`` 不携带类别，故单独声明）。
RECIPE_KNOWLEDGE_TYPES: dict[str, str] = {
    "stone_axe": "tool",
    "fiber_cord": "technique",
    "basket": "tool",
    "campfire": "structure",
    "hut": "structure",
    "well": "structure",
    "granary": "structure",
}


def _normalize_simulation_result(result: object) -> tuple[bool, str]:
    """把世界模拟器的返回值归一为 ``(ok, reason)``。

    契约要求返回 ``(ok: bool, reason: str)``；这里同时容忍裸 ``bool`` 与其它
    可 truthy 对象，避免集成层的轻微签名偏差直接把知识层打崩。
    """
    if isinstance(result, tuple) and len(result) == 2:
        ok, reason = result
        return bool(ok), str(reason)
    if isinstance(result, bool):
        return result, ("ok" if result else "failed")
    return bool(result), str(result)


# ---------------------------------------------------------------------------
# 知识库：状态机 + 依赖图 + 序列化
# ---------------------------------------------------------------------------
class KnowledgeStore:
    """一个文明的知识集合：技术状态、依赖图、留存率与创新效率。

    ``bootstrap=True`` 时把 ``RECIPE_CATALOG`` 全部登记为 ``hypothesis``（"世界上
    存在这些技术，但还没人掌握"），**不会**让任何个体"已知"——个体的已知集合由
    ``agent.known_knowledge_ids`` 保存，通过 :meth:`hypothesize` / :meth:`attempt`
    逐步写入。

    所有修改世界状态的请求都委托给 ``world.technology.simulate_recipe``；本类从不
    扣减材料。
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        catalog: dict[str, Recipe] | None = None,
        bootstrap: bool = True,
    ) -> None:
        """创建知识库。

        参数
        ----
        seed:
            保留的随机种子（世界的随机性由 ``world.rng`` 负责，这里只做可复现标记）。
        catalog:
            技术目录；``None`` 表示使用 :data:`RECIPE_CATALOG`。
        bootstrap:
            是否把目录中的全部技术登记为 ``hypothesis``。
        """
        self.seed: int = int(seed)
        self.catalog: dict[str, Recipe] = (
            dict(catalog) if catalog is not None else dict(RECIPE_CATALOG)
        )
        #: 没有模拟器时被跳过的尝试次数（P1 阶段可观测的降级指标）
        self.degraded_attempts: int = 0
        #: 因材料/工具/位置不满足而被世界拒绝的尝试次数（不是技术缺陷）
        self.blocked_attempts: int = 0
        self._knowledge: dict[str, Knowledge] = {}
        #: 发现/学会/失效的审计日志（有上限），便于追踪知识传播与遗失
        self._learn_log: list[dict[str, Any]] = []
        if bootstrap:
            for recipe_id in sorted(self.catalog):
                self.add(self._knowledge_from_recipe(self.catalog[recipe_id]))

    # ------------------------------------------------------------------
    # 基础访问
    # ------------------------------------------------------------------
    def add(self, knowledge: Knowledge) -> Knowledge:
        """登记（或替换）一项知识；若知识缺配方则从目录补全。"""
        if knowledge.recipe is None:
            recipe = self.catalog.get(knowledge.knowledge_id)
            if recipe is not None:
                knowledge.recipe = recipe
        if knowledge.recipe is not None:
            self.catalog.setdefault(knowledge.recipe.recipe_id, knowledge.recipe)
        self._knowledge[knowledge.knowledge_id] = knowledge
        return knowledge

    def get(self, knowledge_id: str) -> Knowledge | None:
        """按 id 取知识，不存在返回 ``None``。"""
        return self._knowledge.get(knowledge_id)

    def all(self, *, status: str | None = None) -> list[Knowledge]:
        """按 id 升序返回知识；``status`` 非空时只返回该状态。"""
        items = [self._knowledge[key] for key in sorted(self._knowledge)]
        if status is None:
            return items
        return [item for item in items if item.status == status]

    def verified_ids(self) -> list[str]:
        """全部已验证知识的 id（升序），供 telemetry 统计。"""
        return [item.knowledge_id for item in self.all(status="verified")]

    def known_by(self, agent) -> list[Knowledge]:
        """读取 ``agent.known_knowledge_ids``，返回该个体掌握的知识（忽略未知 id）。"""
        identifiers = getattr(agent, "known_knowledge_ids", ()) or ()
        result: list[Knowledge] = []
        for identifier in identifiers:
            knowledge = self._knowledge.get(str(identifier))
            if knowledge is not None:
                result.append(knowledge)
        result.sort(key=lambda item: item.knowledge_id)
        return result

    # ------------------------------------------------------------------
    # 候选实验
    # ------------------------------------------------------------------
    def prerequisites_met(self, knowledge: Knowledge) -> bool:
        """所有前置技术（两种边来源）是否都已在本库中 ``verified``。"""
        for identifier in knowledge.dependency_ids():
            dependency = self._knowledge.get(identifier)
            if dependency is None or dependency.status != "verified":
                return False
        return True

    def candidates_for(self, agent, world=None) -> list[Knowledge]:
        """可进入 "experiment" 步骤的假设：前置已验证 + （若可查询）材料可负担。

        * ``agent`` 目前只用于把 ``recipe`` 交给 ``world.technology.can_afford``
          判断，不改变候选集合的其它语义。
        * ``world`` 为 ``None`` 或没有 ``technology.can_afford`` 时**跳过**可负担性
          检查（P1 降级路径），只按前置知识过滤。
        * ``can_afford`` 抛异常时不阻断候选（世界模拟器仍是最终权威，会在
          ``attempt`` 时拒绝并记录失败）。
        """
        technology = getattr(world, "technology", None) if world is not None else None
        can_afford = getattr(technology, "can_afford", None)
        result: list[Knowledge] = []
        for knowledge in self.all(status="hypothesis"):
            if knowledge.recipe is None:
                continue
            if not self.prerequisites_met(knowledge):
                continue
            if callable(can_afford):
                try:
                    affordable = bool(can_afford(agent, knowledge.recipe))
                except Exception:  # noqa: BLE001 - 降级：不让集成层小 bug 卡死探索
                    affordable = True
                if not affordable:
                    continue
            result.append(knowledge)
        return result

    # ------------------------------------------------------------------
    # 发现 / 验证 / 失效
    # ------------------------------------------------------------------
    def hypothesize(self, agent, recipe_id: str, t: float) -> Knowledge | None:
        """让某个 agent 提出（或第一次学会）一条技术假设。

        行为：
        * ``recipe_id`` 不在目录中 → 返回 ``None``；
        * 知识尚未登记（非 bootstrap 库）→ 按目录创建并登记；
        * 第一次被发现时写入 ``discovered_by`` / ``discovered_at`` 与审计日志；
        * 把 id 写入 ``agent.known_knowledge_ids``（传播/学会的可追踪入口）。
        """
        recipe = self.catalog.get(str(recipe_id))
        knowledge = self._knowledge.get(str(recipe_id))
        if knowledge is None:
            if recipe is None:
                return None
            knowledge = self._knowledge_from_recipe(recipe)
            self.add(knowledge)
        agent_id = int(getattr(agent, "agent_id", -1))
        if knowledge.discovered_by < 0:
            knowledge.discovered_by = agent_id
            knowledge.discovered_at = float(t)
            self._log(t, agent_id, knowledge.knowledge_id, "discovered")
        self._teach(agent, knowledge.knowledge_id, t)
        return knowledge

    def attempt(self, agent, knowledge, world, t: float) -> tuple[bool, str]:
        """执行一次"实验"：把配方交给世界模拟器验证（唯一合法验证入口）。

        参数
        ----
        agent:
            duck-typed 个体，需要 ``agent_id``；成功时会把知识写入
            ``known_knowledge_ids``（使用即学习）。
        knowledge:
            :class:`Knowledge` 实例或知识 id 字符串。
        world:
            duck-typed 世界，唯一被调用的方法是
            ``world.technology.simulate_recipe(agent, recipe) -> (ok, reason)``。
        t:
            世界时间（秒），写入验证记录。

        返回
        ----
        ``(ok, reason)``。除世界模拟器的结论外，还有本层的拦截原因码：

        * ``"unknown_knowledge"``：id 不在库中；
        * ``"invalidated"``：知识已失效（终态，不再尝试）；
        * ``"no_recipe"``：知识没有配方（不该发生）；
        * ``"no_simulator"``：世界还没有 ``simulate_recipe``（P1 降级，计入
          ``degraded_attempts``，**不**改知识状态）；
        * ``"missing_prerequisites"``：前置技术未验证（世界还没能力，不让模拟器白跑）；
        * ``"simulator_error:<异常类型>"``：模拟器抛异常（按失败记录，知识层不崩）。

        注意：一旦调用模拟器，其 ``ok=False`` 就会作为一次**失败尝试**累计
        （``failed_uses`` / ``confidence``），因此调用方应先用
        :meth:`candidates_for` 过滤材料不足的假设，避免把"材料不够"误记成
        "技术有缺陷"。
        """
        target = knowledge if isinstance(knowledge, Knowledge) else self.get(str(knowledge))
        if target is None:
            return (False, "unknown_knowledge")
        if target.status == "invalidated":
            return (False, "invalidated")
        if target.recipe is None:
            return (False, "no_recipe")

        technology = getattr(world, "technology", None) if world is not None else None
        simulate = getattr(technology, "simulate_recipe", None)
        if not callable(simulate):
            # P1 降级：世界规则还没接线，不当成技术缺陷，只留下可观测计数。
            self.degraded_attempts += 1
            return (False, NO_SIMULATOR_REASON)

        if not self.prerequisites_met(target):
            return (False, "missing_prerequisites")
        if self._knowledge.get(target.knowledge_id) is None:
            self.add(target)

        agent_id = int(getattr(agent, "agent_id", -1))
        # 第一次被某个个体尝试 = 这条技术被"发现"（可追踪发现者与时间）
        if target.discovered_by < 0:
            target.discovered_by = agent_id
            target.discovered_at = float(t)
            self._log(t, agent_id, target.knowledge_id, "discovered")
        try:
            raw_result = simulate(agent, target.recipe)
        except Exception as error:  # noqa: BLE001 - 世界异常不得击穿知识层
            reason = f"simulator_error:{type(error).__name__}"
            target.record(t, agent_id, False, reason)
            return (False, reason)

        ok, reason = _normalize_simulation_result(raw_result)
        if not ok and reason in PRECONDITION_REASONS:
            # 材料/工具/位置不满足：世界没有执行也**没有**扣材料，
            # 这不是技术缺陷，不写验证记录、不改 confidence / failed_uses。
            self.blocked_attempts += 1
            return (False, reason)
        target.record(t, agent_id, ok, reason)
        if ok:
            self._teach(agent, target.knowledge_id, t)
        return (ok, reason)

    def mark_invalid(self, knowledge_id: str, reason: str, t: float = 0.0) -> None:
        """环境失效：把知识标记为 ``invalidated``（终态）。

        ``reason`` 写入 ``validation_history`` 的一条审计记录（``agent_id=-1``
        表示"环境/世界判定的失效"，而非某个体实验失败），因此**不会**改动
        ``failed_uses`` / ``successful_uses``。
        """
        knowledge = self._knowledge.get(str(knowledge_id))
        if knowledge is None:
            return
        knowledge.status = "invalidated"
        knowledge.confidence = 0.0
        knowledge.validation_history.append(
            ValidationRecord(
                t=float(t),
                agent_id=-1,
                success=False,
                evidence=f"invalidated:{reason}",
            )
        )
        self._log(t, -1, knowledge.knowledge_id, "invalidated")

    # ------------------------------------------------------------------
    # 依赖图
    # ------------------------------------------------------------------
    def edges(self) -> list[tuple[str, str]]:
        """全部已登记知识的依赖边 ``(parent, child)``（去重、升序）。

        ``parent`` 是前置技术，``child`` 是依赖它的技术。两种边来源
        （``prerequisites`` 与 ``recipe.requires_knowledge``）都会出现，且只保留
        两端都在本库中的边。
        """
        edge_set: set[tuple[str, str]] = set()
        for knowledge in self._knowledge.values():
            child = knowledge.knowledge_id
            for parent in knowledge.dependency_ids():
                if parent == child:
                    continue
                if parent in self._knowledge:
                    edge_set.add((parent, child))
        return sorted(edge_set)

    def verified_edges(self) -> list[tuple[str, str]]:
        """已验证子图的依赖边（telemetry 的 ``tech_graph_edges`` 用这个）。"""
        verified = set(self.verified_ids())
        return [
            (parent, child)
            for parent, child in self.edges()
            if parent in verified and child in verified
        ]

    def dependents(self, knowledge_id: str) -> set[str]:
        """所有（直接或间接）依赖该知识的知识 id —— 传递闭包。"""
        adjacency: dict[str, set[str]] = {}
        for parent, child in self.edges():
            adjacency.setdefault(parent, set()).add(child)
        found: set[str] = set()
        stack = list(adjacency.get(str(knowledge_id), ()))
        while stack:
            node = stack.pop()
            if node in found:
                continue
            found.add(node)
            stack.extend(adjacency.get(node, ()))
        found.discard(str(knowledge_id))
        return found

    def direct_dependents(self, knowledge_id: str) -> set[str]:
        """直接依赖该知识的知识 id（只看一跳）。"""
        return {child for parent, child in self.edges() if parent == str(knowledge_id)}

    def depth(self) -> int:
        """已验证技术依赖图的最长依赖链长度（按**边数**计；空图为 0）。"""
        verified = set(self.verified_ids())
        if not verified:
            return 0
        parents: dict[str, set[str]] = {node: set() for node in verified}
        for parent, child in self.verified_edges():
            parents[child].add(parent)
        memo: dict[str, int] = {}

        def longest(node: str, visiting: frozenset[str]) -> int:
            if node in memo:
                return memo[node]
            if node in visiting:  # 理论上 DAG；防御性处理意外环
                return 0
            best = 0
            for parent in parents.get(node, ()):
                best = max(best, 1 + longest(parent, visiting | {node}))
            memo[node] = best
            return best

        return max(longest(node, frozenset()) for node in sorted(verified))

    # ------------------------------------------------------------------
    # 指标
    # ------------------------------------------------------------------
    @staticmethod
    def _as_id_set(known_ids) -> set[str]:
        """把 ``known_ids``（id 序列 / 单个 id / 带 ``known_knowledge_ids`` 的对象）转成集合。"""
        if known_ids is None:
            return set()
        if isinstance(known_ids, str):
            return {known_ids}
        maybe_agent_ids = getattr(known_ids, "known_knowledge_ids", None)
        if maybe_agent_ids is not None:
            known_ids = maybe_agent_ids
        try:
            return {str(item) for item in known_ids}
        except TypeError:
            return set()

    def retention(self, known_ids) -> float:
        """知识留存率：``|known ∩ 已验证| / |已验证|``；没有已验证知识时返回 ``1.0``。"""
        verified = set(self.verified_ids())
        if not verified:
            return 1.0
        known = self._as_id_set(known_ids)
        return len(known & verified) / len(verified)

    def innovation_efficiency(self) -> float:
        """创新效率：``已验证知识数 / 总尝试次数``；没有尝试时返回 ``0.0``。

        总尝试次数 = 全部知识的 ``successful_uses + failed_uses``（含验证后的继续
        使用），因此它衡量"多少次实验换来一项可复用技术"。
        """
        verified = len(self.all(status="verified"))
        attempts = sum(
            knowledge.successful_uses + knowledge.failed_uses
            for knowledge in self._knowledge.values()
        )
        if attempts <= 0:
            return 0.0
        return verified / attempts

    def summary_zh(self) -> str:
        """一行中文摘要（状态计数 / 依赖边 / 深度 / 创新效率）。"""
        counts = {status: 0 for status in KNOWLEDGE_STATUS}
        for knowledge in self._knowledge.values():
            counts[knowledge.status] = counts.get(knowledge.status, 0) + 1
        return (
            f"知识库 {len(self._knowledge)} 项（已验证{counts['verified']} "
            f"假设{counts['hypothesis']} 失效{counts['invalidated']}）"
            f" 依赖边{len(self.edges())} 深度{self.depth()}"
            f" 创新效率{self.innovation_efficiency():.2f}"
        )

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """序列化整库（技术目录 + 全部知识 + 审计日志 + 降级计数）。"""
        catalog_out: dict[str, object] = {
            recipe_id: recipe.to_dict() for recipe_id, recipe in self.catalog.items()
        }
        # 手工 ``add`` 进来但不在目录中的配方也要能往返
        for knowledge in self.all():
            recipe = knowledge.recipe
            if recipe is not None and recipe.recipe_id not in catalog_out:
                catalog_out[recipe.recipe_id] = recipe.to_dict()
        return {
            "version": STORE_SCHEMA_VERSION,
            "seed": self.seed,
            "catalog": catalog_out,
            "knowledge": [knowledge.to_dict() for knowledge in self.all()],
            "degraded_attempts": self.degraded_attempts,
            "learn_log": [dict(entry) for entry in self._learn_log],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "KnowledgeStore":
        """从 :meth:`to_dict` 的结果恢复（不重新 bootstrap，逐条还原状态）。"""
        raw_catalog = data.get("catalog") or {}
        catalog = {
            str(recipe_id): Recipe.from_dict(dict(recipe_data))
            for recipe_id, recipe_data in raw_catalog.items()
        }
        store = cls(
            seed=int(data.get("seed", 0)),
            catalog=catalog if catalog else None,
            bootstrap=False,
        )
        for entry in data.get("knowledge") or ():
            store.add(Knowledge.from_dict(dict(entry)))
        store.degraded_attempts = int(data.get("degraded_attempts", 0))
        store._learn_log = [dict(entry) for entry in (data.get("learn_log") or ())]
        return store

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _knowledge_from_recipe(self, recipe: Recipe) -> Knowledge:
        """按目录配方构造一条 ``hypothesis`` 知识（尚未被任何个体发现）。"""
        return Knowledge(
            knowledge_id=recipe.recipe_id,
            knowledge_type=RECIPE_KNOWLEDGE_TYPES.get(recipe.recipe_id, "technique"),
            name_zh=recipe.name_zh,
            prerequisites=tuple(recipe.requires_knowledge),
            recipe=recipe,
            status="hypothesis",
        )

    def _teach(self, agent, knowledge_id: str, t: float) -> None:
        """把知识 id 写入个体的 ``known_knowledge_ids``（传播/学会）。

        兼容 list / set / tuple 三种容器；个体没有该属性时静默跳过，避免知识层
        反向要求集成层的数据布局。
        """
        identifiers = getattr(agent, "known_knowledge_ids", None)
        if identifiers is None:
            return
        target_id = str(knowledge_id)
        try:
            if target_id in identifiers:
                self._log(t, int(getattr(agent, "agent_id", -1)), target_id, "known")
                return
            if isinstance(identifiers, set):
                identifiers.add(target_id)
            elif isinstance(identifiers, list):
                identifiers.append(target_id)
            else:
                setattr(agent, "known_knowledge_ids", tuple(identifiers) + (target_id,))
        except (AttributeError, TypeError):  # pragma: no cover - 防御性
            return
        self._log(t, int(getattr(agent, "agent_id", -1)), target_id, "learned")

    def _log(self, t: float, agent_id: int, knowledge_id: str, event: str) -> None:
        """追加一条审计日志（超过上限时丢弃最旧记录）。"""
        self._learn_log.append(
            {
                "t": float(t),
                "agent_id": int(agent_id),
                "knowledge_id": str(knowledge_id),
                "event": str(event),
            }
        )
        if len(self._learn_log) > LEARN_LOG_LIMIT:
            self._learn_log = self._learn_log[-LEARN_LOG_LIMIT:]
