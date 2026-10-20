"""受限的组合式创新原型（V1 契约 §3：规则驱动、真实成本、上限 3）。

定位
====
本项目已有**固定目录的有限科技树**：``knowledge.RECIPE_CATALOG`` 声明这个世界
"存在哪些技术"，``technology.Technology`` 是唯一的世界权威（扣材料、注册效果、
决定一次制作/实验的成败）。本模块在两者**之上**增加一层"组合式创新"原型：

* 个体只能把**自己已掌握且 verified** 的两条技术的**属性（tags）**按冻结的相容
  规则组合，得到一件新工具；
* 新工具的效果由属性**确定性推导**（不是随机数值）；
* 材料成本是**真实扣减**（真实材料经济，失败也消耗）；
* 成功与否由组合难度决定（真实实验成本，与 ``Technology.simulate_recipe``
  的失败语义一致）；
* 成功后通过 ``technology.register_custom_effect`` 把效果注册进世界，之后
  采集产量 / 载重上限 / 体力恢复 / 取水 / 制作成功率等**真实数值**随工具变化。

设计原则（与契约 §0/§3 一致，禁止违反）
--------------------------------------
1. **禁止随机拼接名字或数值**：``ATTRIBUTE_CATALOG`` 与 ``EFFECT_RULES`` 是模块级
   世界规则常量；``Proposal`` 的名称、效果、成本、难度全部由
   ``(components, tags)`` 确定性推导。本模块唯一的随机性是"实验是否成功"
   （``rng.random() < 组合难度``）。
2. **不 import population / world / knowledge**：只依赖 duck typing
   （``agent.agent_id`` / ``agent.known_knowledge_ids`` / ``agent.inventory`` /
   ``world.t`` / ``world.knowledge`` / ``world.technology``），避免循环依赖。
3. **只改库存 + 委托 technology**：本模块对世界的写入仅两处——按成本调用
   ``Inventory.take`` 扣材料、调用 ``technology.register_custom_effect`` 注册效果；
   不写 memory / telemetry / 知识状态（那是 Lead 的 ``population.py`` 集成层职责）。
4. **前置失败不扣材料**：没有 effect 后端（P3 早期）、材料不足、组件未验证/不相容
   都属于前置条件，返回原因码且材料原样保留；只有真正"做了实验但失败"
   （原因码 ``defect``）才消耗材料——与 ``knowledge.PRECONDITION_REASONS`` 同一
   哲学（材料暂缺不是技术缺陷，也不是一次真实的失败实验）。

确定性推导（冻结规则表，见 :data:`EFFECT_RULES`）
-------------------------------------------------
::

    edge + binder       -> harvest_bonus +1
    container + binder  -> capacity_bonus +4
    heat + container    -> energy_regen +0.15
    heat + binder       -> energy_regen +0.10, water_regen +0.05
    structure + container -> capacity_bonus +2

一条技术可以带多个 tag；多条规则同时匹配时按 :data:`EFFECT_RULES` 的**顺序**取
第一条（确定性、可解释，不做隐式叠加）。

:attr:`RecipeAttribute.base_effects` 是该技术**自身**的基础效果（世界规则声明，
例如背篓本身 +4 载重、篝火本身每秒恢复 0.35 体力）。组合产物只产出 tag 相容对的
**协同**效果，不重复叠加组分自身效果（否则同一收益会被计算两次）。

材料成本与难度
--------------
* 材料成本 = **最简原型**：第一组件（``Proposal.components[0]``，已按 id 排序）
  配方 ``inputs`` 中**字母序第一个** material key，计 1 单位（第一组件没有 inputs
  时回退到第二组件，再不行成本为空；见 :func:`material_cost`）；
* 组合难度 = ``0.15 + 0.1 × 不同 tag 数``（两条单 tag 技术 → 0.35，成功率 0.65）。

对 ``technology.py`` 的接口要求（由 Lead 实现）
---------------------------------------------
``register_custom_effect(self, tool_name: str, effects: dict[str, float]) -> None``
把 ``effects``（键必须属于 :data:`EFFECT_KEYS`，值有限且 > 0）登记到
``self.custom_effects[tool_name]``，供 ``harvest``（按 ``agent.inventory.tool``
查 ``harvest_bonus``）、``Inventory.capacity``（``capacity_bonus``）、
``apply_facility_effects``（``energy_regen`` / ``water_regen``）、制作成功率
（``craft_success_bonus``）使用。本模块用 ``getattr`` 探测该接口：缺失时
:meth:`InnovationLab.attempt` 返回 ``no_effect_backend`` 且**不扣材料**，让 P3
早期集成可以先跑通。

真实效果如何生效
----------------
实验成功后，本模块把新工具交给发明者（``agent.inventory.tool = output_tool``，
可用 ``attempt(..., equip=False)`` 关闭）。工具名一旦被 ``technology`` 侧的
``harvest`` / ``simulate_recipe`` / ``apply_facility_effects`` 查询到，效果就
直接作用于世界数值，无需 UI。``capacity_bonus`` 需要集成层把
``technology.custom_effect(tool, "capacity_bonus")`` 同步到
``Inventory.capacity_bonus`` 字段（该字段由 ``technology.py`` 提供，本模块不
越权改载重缓存）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "EFFECT_KEYS",
    "KNOWN_TAGS",
    "MAX_INNOVATIONS",
    "NO_EFFECT_BACKEND_REASON",
    "EffectRule",
    "EFFECT_RULES",
    "RecipeAttribute",
    "ATTRIBUTE_CATALOG",
    "Proposal",
    "Innovation",
    "InnovationLab",
    "attribute",
    "register_attribute",
    "rule_for",
    "compatible_pair",
    "derive_effects",
    "combination_difficulty",
    "material_cost",
]

#: 组合产物允许改变的世界数值键（与契约 §3 完全一致）
EFFECT_KEYS: tuple[str, ...] = (
    "harvest_bonus",
    "capacity_bonus",
    "energy_regen",
    "water_regen",
    "craft_success_bonus",
)

#: 世界规则里存在的技术属性标签（与契约 §3 一致）
KNOWN_TAGS: tuple[str, ...] = ("edge", "binder", "heat", "container", "structure")

#: 每个世界最多保留几项组合创新（契约：3 项/世界）
MAX_INNOVATIONS: int = 3

#: 没有 ``technology.register_custom_effect`` 时的原因码（P3 早期降级路径）
NO_EFFECT_BACKEND_REASON: str = "no_effect_backend"

#: 审计日志上限（防止长跑时无界增长）
HISTORY_LIMIT: int = 256
#: 序列化结构版本号
LAB_SCHEMA_VERSION: int = 1

#: 效果键的中文说明（用于 ``summary_zh``）
EFFECT_LABELS_ZH: dict[str, str] = {
    "harvest_bonus": "采集加成",
    "capacity_bonus": "负重上限",
    "energy_regen": "体力恢复",
    "water_regen": "水分恢复",
    "craft_success_bonus": "制作成功率",
}


# ---------------------------------------------------------------------------
# 属性：每条已验证技术的基础属性（世界规则）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecipeAttribute:
    """一条技术"能被怎样组合"的属性声明（世界规则，不是剧情）。

    参数
    ----
    knowledge_id:
        技术 id，与 ``knowledge.RECIPE_CATALOG`` / :class:`~newworld.knowledge.Knowledge`
        的 ``knowledge_id`` 一一对应。
    tags:
        属性标签，取值属于 :data:`KNOWN_TAGS`。组合只看 tag，不看名字。
    base_effects:
        该技术**自身**已经成立的基础效果（世界规则声明）；组合产物只取 tag 协同
        效果，不重复叠加这里的效果（见模块 docstring）。
    """

    knowledge_id: str
    tags: tuple[str, ...]
    base_effects: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """序列化为 JSON 可写的普通字典。"""
        return {
            "knowledge_id": self.knowledge_id,
            "tags": list(self.tags),
            "base_effects": {str(key): float(value) for key, value in self.base_effects.items()},
        }


#: 默认属性目录：对应 ``knowledge.RECIPE_CATALOG`` 的 7 条技术。
#:
#: ``base_effects`` 与 ``technology.py`` 的真实世界数值保持一致（石斧采集产量翻倍、
#: 背篓 +4 载重、篝火 0.35/s 体力、水井 1.2/s 水分……），因此"属性"不是装饰性
#: 标签，而是可被核对的既有世界规则。
ATTRIBUTE_CATALOG: dict[str, RecipeAttribute] = {
    "stone_axe": RecipeAttribute("stone_axe", ("edge",), {"harvest_bonus": 1.0}),
    "fiber_cord": RecipeAttribute("fiber_cord", ("binder",), {"craft_success_bonus": 0.05}),
    "basket": RecipeAttribute("basket", ("container",), {"capacity_bonus": 4.0}),
    "campfire": RecipeAttribute("campfire", ("heat",), {"energy_regen": 0.35}),
    "hut": RecipeAttribute("hut", ("structure",), {"energy_regen": 0.30}),
    "well": RecipeAttribute("well", ("container",), {"water_regen": 1.2}),
    "granary": RecipeAttribute("granary", ("container",), {"energy_regen": 0.22}),
}


def attribute(knowledge_id: str) -> RecipeAttribute | None:
    """按技术 id 取属性；未知技术返回 ``None``（未定义属性 ⇒ 不可组合）。"""
    return ATTRIBUTE_CATALOG.get(str(knowledge_id))


def register_attribute(attr: RecipeAttribute) -> RecipeAttribute:
    """把一条新的技术属性登记进世界规则目录（用于扩展目录，不做随机生成）。"""
    if not attr.knowledge_id:
        raise ValueError("RecipeAttribute.knowledge_id 不能为空")
    unknown = [tag for tag in attr.tags if tag not in KNOWN_TAGS]
    if unknown:
        raise ValueError(f"未知 tag: {unknown!r}（允许值 {KNOWN_TAGS!r}）")
    bad_keys = [key for key in attr.base_effects if key not in EFFECT_KEYS]
    if bad_keys:
        raise ValueError(f"未知效果键: {bad_keys!r}（允许值 {EFFECT_KEYS!r}）")
    ATTRIBUTE_CATALOG[attr.knowledge_id] = attr
    return attr


# ---------------------------------------------------------------------------
# 组合规则：tag 相容对 -> 确定性效果（冻结表，顺序即优先级）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EffectRule:
    """一条"tag 相容对 → 新工具/确定性效果"的世界规则。"""

    tags: frozenset[str]
    effects: dict[str, float]
    output_tool: str
    name_zh: str

    def to_dict(self) -> dict[str, object]:
        """序列化为 JSON 可写的普通字典。"""
        return {
            "tags": sorted(self.tags),
            "effects": {str(key): float(value) for key, value in self.effects.items()},
            "output_tool": self.output_tool,
            "name_zh": self.name_zh,
        }


#: 冻结的相容规则表（契约 §3 的最小可解释原型）。顺序 = 多规则匹配时的优先级。
EFFECT_RULES: tuple[EffectRule, ...] = (
    EffectRule(
        tags=frozenset({"edge", "binder"}),
        effects={"harvest_bonus": 1.0},
        output_tool="hafted_blade",
        name_zh="装柄刃器",
    ),
    EffectRule(
        tags=frozenset({"binder", "container"}),
        effects={"capacity_bonus": 4.0},
        output_tool="woven_creel",
        name_zh="编织背篓具",
    ),
    EffectRule(
        tags=frozenset({"container", "heat"}),
        effects={"energy_regen": 0.15},
        output_tool="fire_vessel",
        name_zh="火陶容器",
    ),
    EffectRule(
        tags=frozenset({"binder", "heat"}),
        effects={"energy_regen": 0.10, "water_regen": 0.05},
        output_tool="corded_lamp",
        name_zh="绳火具",
    ),
    EffectRule(
        tags=frozenset({"container", "structure"}),
        effects={"capacity_bonus": 2.0},
        output_tool="storage_frame",
        name_zh="储物架",
    ),
)


def _match(
    attr_a: RecipeAttribute | None,
    attr_b: RecipeAttribute | None,
) -> tuple[EffectRule, str, str] | None:
    """返回 ``(规则, tag_a, tag_b)``；两条技术没有任何相容对时返回 ``None``。"""
    if attr_a is None or attr_b is None:
        return None
    tags_a = tuple(str(tag) for tag in (attr_a.tags or ()))
    tags_b = tuple(str(tag) for tag in (attr_b.tags or ()))
    for rule in EFFECT_RULES:
        for tag_a in tags_a:
            for tag_b in tags_b:
                if frozenset((tag_a, tag_b)) == rule.tags:
                    return (rule, tag_a, tag_b)
    return None


def rule_for(
    attr_a: RecipeAttribute | None,
    attr_b: RecipeAttribute | None,
) -> EffectRule | None:
    """两条技术之间命中的第一条相容规则（无相容对返回 ``None``）。"""
    match = _match(attr_a, attr_b)
    return None if match is None else match[0]


def compatible_pair(
    attr_a: RecipeAttribute | None,
    attr_b: RecipeAttribute | None,
) -> tuple[str, str] | None:
    """两条技术命中的相容 tag 对 ``(tag_a, tag_b)``（无相容对返回 ``None``）。"""
    match = _match(attr_a, attr_b)
    return None if match is None else (match[1], match[2])


def derive_effects(
    attr_a: RecipeAttribute | None,
    attr_b: RecipeAttribute | None,
) -> dict[str, float] | None:
    """由两条技术的属性**确定性**推导组合效果；不相容返回 ``None``。

    返回值是规则表中效果字典的副本（调用方可以自由修改，不污染世界规则）。
    """
    rule = rule_for(attr_a, attr_b)
    return None if rule is None else dict(rule.effects)


def combination_difficulty(attrs: Iterable[RecipeAttribute]) -> float:
    """组合难度：``0.15 + 0.1 × 不同 tag 数``（契约 §3）。"""
    tags: set[str] = set()
    for attr in attrs:
        if attr is None:
            continue
        tags.update(str(tag) for tag in (attr.tags or ()))
    return 0.15 + 0.1 * len(tags)


def material_cost(
    knowledge: Any,
    components: Iterable[str],
    *,
    fallback: dict[str, int] | None = None,
) -> dict[str, int]:
    """组合**最简原型**试制的真实材料成本：**1 单位基础材料**。

    规则（确定性世界规则，不是随机数）
    --------------------------------
    按组件顺序取第一个组件（:class:`Proposal` 的 ``components`` 已按 id 排序，
    因此顺序确定），在它的 ``recipe.inputs`` 材料键中取**字母序第一个**键，成本
    即 ``{key: 1}``：

    * ``stone_axe(wood,stone) + fiber_cord(fiber)`` → 第一组件 stone_axe 的
      material keys ``Sorted(["stone", "wood"])`` 第一个 ``"stone"`` → ``{stone: 1}``；
    * ``campfire(wood,stone) + fiber_cord(fiber)`` → ``{stone: 1}``；
    * ``basket(wood,cord) + fiber_cord(fiber)`` → ``{cord: 1}``。

    若第一组件没有 ``inputs``（理论上不会），回退到第二个组件用同样方法取；全部
    组件都取不到配方时返回 ``fallback``（缺省空字典）。

    **为什么只需要 1 单位"基础材料"**：组合创新是**原型试制**（把两件已有工具的
    思路拼一件新工具），不是把两件工具重新造一遍。8 人上限、小库存的世界里，规划器
    会持续消耗材料，个体库存几乎从不"恰好"同时持有 3 类；实测 700 世界秒后唯一有
    工具的个体库存是 ``{stone:1, cord:1, wood:0, fiber:0}``，之前"并集每键 1"的
    成本仍然不可负担、``attempts=0``。收敛到"1 单位字母序第一个基础材料"后，只要
    个体手上有任意一种相关材料即可试制，创新机制才可能被真实观测到。这**仍是真实
    材料消耗**：:meth:`InnovationLab.attempt` 成功/失败都按此扣除（失败也消耗）。
    """
    for component in components:
        record = _get_record(knowledge, str(component))
        recipe = getattr(record, "recipe", None) if record is not None else None
        inputs = dict(getattr(recipe, "inputs", {}) or {}) if recipe is not None else {}
        if not inputs:
            continue
        key = sorted(str(kind) for kind in inputs)[0]
        return {key: 1}
    return {
        str(key): int(value)
        for key, value in dict(fallback or {}).items()
        if int(value) > 0
    }


# ---------------------------------------------------------------------------
# 提案 / 创新记录
# ---------------------------------------------------------------------------
@dataclass
class Proposal:
    """一条待验证的组合提案：完全由属性/规则/真实配方**确定性**推导。"""

    proposal_id: str
    agent_id: int
    name_zh: str
    components: tuple[str, ...]
    inputs: dict[str, int]
    output_tool: str
    effects: dict[str, float]
    rationale: str

    def to_dict(self) -> dict[str, object]:
        """序列化为 JSON 可写的普通字典。"""
        return {
            "proposal_id": self.proposal_id,
            "agent_id": int(self.agent_id),
            "name_zh": self.name_zh,
            "components": list(self.components),
            "inputs": {str(key): int(value) for key, value in self.inputs.items()},
            "output_tool": self.output_tool,
            "effects": {str(key): float(value) for key, value in self.effects.items()},
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Proposal":
        """从 :meth:`to_dict` 的结果恢复。"""
        return cls(
            proposal_id=str(data.get("proposal_id", "")),
            agent_id=int(data.get("agent_id", -1)),
            name_zh=str(data.get("name_zh", "")),
            components=tuple(str(item) for item in (data.get("components") or ())),
            inputs={str(key): int(value) for key, value in (data.get("inputs") or {}).items()},
            output_tool=str(data.get("output_tool", "")),
            effects={str(key): float(value) for key, value in (data.get("effects") or {}).items()},
            rationale=str(data.get("rationale", "")),
        )


@dataclass
class Innovation:
    """一项**已经验证成功**的组合创新（真实世界效果已注册）。"""

    innovation_id: str
    name_zh: str
    components: tuple[str, ...]
    effects: dict[str, float]
    invented_by: int
    invented_at: float
    successful_uses: int = 0

    def to_dict(self) -> dict[str, object]:
        """序列化为 JSON 可写的普通字典。"""
        return {
            "innovation_id": self.innovation_id,
            "name_zh": self.name_zh,
            "components": list(self.components),
            "effects": {str(key): float(value) for key, value in self.effects.items()},
            "invented_by": int(self.invented_by),
            "invented_at": float(self.invented_at),
            "successful_uses": int(self.successful_uses),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Innovation":
        """从 :meth:`to_dict` 的结果恢复。"""
        return cls(
            innovation_id=str(data.get("innovation_id", "")),
            name_zh=str(data.get("name_zh", "")),
            components=tuple(str(item) for item in (data.get("components") or ())),
            effects={str(key): float(value) for key, value in (data.get("effects") or {}).items()},
            invented_by=int(data.get("invented_by", -1)),
            invented_at=float(data.get("invented_at", 0.0)),
            successful_uses=int(data.get("successful_uses", 0)),
        )


@dataclass(frozen=True)
class _CostRecipe:
    """材料成本包装：只暴露 ``inputs``，让 ``technology.can_afford`` 直接可用。"""

    inputs: dict[str, int]


# ---------------------------------------------------------------------------
# 创新实验室：提案 → 实验 → 真实效果
# ---------------------------------------------------------------------------
class InnovationLab:
    """一个世界的组合式创新记账中心（每个世界一个实例）。

    本类**不**拥有世界规则：效果来自 :data:`EFFECT_RULES`，扣材料走
    ``Inventory.take``，注册效果走 ``world.technology.register_custom_effect``。
    """

    #: 上限 3 项/世界（契约）
    MAX_INNOVATIONS: int = MAX_INNOVATIONS

    def __init__(self, seed: int = 0, *, rng: Any = None) -> None:
        """创建实验室。

        参数
        ----
        seed:
            可复现标记；成功/失败的随机判定来自 ``random.Random(seed)``。
        rng:
            可注入的随机源（需要有 ``random() -> float``）；测试用它固定实验
            结果（例如恒成功 / 恒失败）。缺省用 ``random.Random(seed)``。
        """
        self.seed: int = int(seed)
        self.rng: Any = rng if rng is not None else random.Random(self.seed)
        self._innovations: list[Innovation] = []
        #: tool_name -> 已注册效果（与 technology 侧一致，便于外部查询/序列化）
        self._custom_effects: dict[str, dict[str, float]] = {}
        #: 每次实验尝试的审计记录（有上限）：t/agent_id/components/inputs/ok/reason...
        self.history: list[dict[str, Any]] = []
        self._serial: int = 0

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def innovations(self) -> list[Innovation]:
        """已发明的创新（列表副本；元素是真实对象，可配合 :meth:`record_use`）。"""
        return list(self._innovations)

    def effects_for(self, tool_name: str) -> dict[str, float]:
        """查询某件新工具已注册的世界效果；未知工具返回空字典。"""
        return dict(self._custom_effects.get(str(tool_name)) or {})

    def is_registered(self, tool_name: str) -> bool:
        """该工具名是否已经注册过效果。"""
        return str(tool_name) in self._custom_effects

    def custom_effects(self) -> dict[str, dict[str, float]]:
        """全部已注册效果（深拷贝，避免外部改坏内部状态）。"""
        return {tool: dict(effects) for tool, effects in self._custom_effects.items()}

    def difficulty(self, proposal: Proposal) -> float:
        """该提案的组合难度（由组分 tag 数决定，与 RNG 无关）。"""
        attrs = [attribute(component) for component in proposal.components]
        return combination_difficulty(attr for attr in attrs if attr is not None)

    def record_use(self, innovation: Innovation | str, uses: int = 1) -> int:
        """记录一项创新被成功使用（``successful_uses``）；返回最新计数。"""
        target = self._find_innovation(innovation)
        if target is None:
            return 0
        increment = max(0, int(uses))
        target.successful_uses = int(target.successful_uses) + increment
        return target.successful_uses

    def stats(self) -> dict[str, object]:
        """供 telemetry 使用的计数（纯只读统计）。"""
        successes = sum(1 for entry in self.history if entry.get("ok"))
        material = sum(
            int(amount)
            for entry in self.history
            for amount in dict(entry.get("inputs") or {}).values()
        )
        return {
            "innovations": len(self._innovations),
            "max_innovations": MAX_INNOVATIONS,
            "attempts": len(self.history),
            "successes": successes,
            "failures": len(self.history) - successes,
            "material_spent": material,
        }

    # ------------------------------------------------------------------
    # 提案
    # ------------------------------------------------------------------
    def proposals(
        self,
        agent: Any,
        knowledge: Any,
        *,
        limit: int = 3,
        affordable_only: bool = False,
    ) -> list[Proposal]:
        """为 ``agent`` 提出组合提案（仅 verified 组件、tag 相容、确定性顺序）。

        前置条件（任一不满足即返回空列表）：

        * 世界级上限未满（``MAX_INNOVATIONS``）；已满后契约要求返回空；
        * 站点/工作台条件满足（见 :meth:`_site_ready`）；
        * 该个体掌握 **≥2 条 verified** 技术，且它们的属性 tag 存在冻结的相容对；
        * 未与该个体或世界已发明的组合重复（按"组分配对"与"产出工具"双重去重）。

        参数
        ----
        limit:
            最多返回几条（contract 默认 3）。
        affordable_only:
            可选：只返回当前库存能负担材料成本的提案（``False`` 时由
            :meth:`attempt` 在真实扣材料前拒绝），便于集成层减少无效尝试。
        """
        if int(limit) <= 0 or len(self._innovations) >= MAX_INNOVATIONS:
            return []
        if not self._site_ready(agent):
            return []

        records = self._knowledge_records(agent, knowledge)
        verified = [record for record in records if self._is_verified(record)]
        verified.sort(key=lambda item: str(getattr(item, "knowledge_id", "")))
        attrs: list[RecipeAttribute] = []
        names: dict[str, str] = {}
        for record in verified:
            knowledge_id = str(getattr(record, "knowledge_id", ""))
            attr = attribute(knowledge_id)
            if attr is None or not attr.tags:
                continue
            attrs.append(attr)
            names[knowledge_id] = str(getattr(record, "name_zh", "") or knowledge_id)

        invented_pairs = {self._pair_key(item.components) for item in self._innovations}
        invented_tools = set(self._custom_effects)

        candidates: list[tuple[tuple[str, str], RecipeAttribute, RecipeAttribute, EffectRule]] = []
        for index, attr_a in enumerate(attrs):
            for attr_b in attrs[index + 1:]:
                rule = rule_for(attr_a, attr_b)
                if rule is None:
                    continue
                pair = self._pair_key((attr_a.knowledge_id, attr_b.knowledge_id))
                if pair in invented_pairs or rule.output_tool in invented_tools:
                    continue
                candidates.append((pair, attr_a, attr_b, rule))
        candidates.sort(key=lambda item: item[0])

        result: list[Proposal] = []
        used_tools: set[str] = set()
        for pair, attr_a, attr_b, rule in candidates:
            if rule.output_tool in used_tools:
                # 同一规则只提一次（第二次尝试必然 already_invented）
                continue
            cost = material_cost(knowledge, pair)
            if affordable_only and not self._inventory_has(agent, cost):
                continue
            used_tools.add(rule.output_tool)
            result.append(
                Proposal(
                    proposal_id=f"proposal_{pair[0]}__{pair[1]}",
                    agent_id=int(getattr(agent, "agent_id", -1)),
                    name_zh=self._compose_name(names, pair, rule),
                    components=pair,
                    inputs=dict(sorted(cost.items())),
                    output_tool=rule.output_tool,
                    effects=dict(rule.effects),
                    rationale=self._rationale(names, pair, attr_a, attr_b, rule, cost),
                )
            )
            if len(result) >= int(limit):
                break
        return result

    # ------------------------------------------------------------------
    # 实验
    # ------------------------------------------------------------------
    def attempt(
        self,
        agent: Any,
        proposal: Proposal,
        world: Any,
        *,
        equip: bool = True,
    ) -> tuple[bool, str, Innovation | None]:
        """真正做一次组合实验：前置检查 → 扣材料 → 难度判定 → 注册效果。

        参数
        ----
        agent:
            duck-typed 个体：``agent_id`` / ``known_knowledge_ids`` / ``inventory``。
        proposal:
            :class:`Proposal`；效果/工具名仍以世界规则**重新推导**为准（防止伪造
            提案绕过规则），``proposal.effects`` 只用于展示。
        world:
            duck-typed 世界：``technology``（必需）、``knowledge``（可选）、
            ``t``（可选）。
        equip:
            成功时是否把新工具交给发明者（``inventory.tool = output_tool``）。
            默认 True，但采用"不降级"策略：若个体手上已有工具，只有新工具的
            采集能力不低于旧工具时才替换（避免把 ``axe`` 的产量翻倍换成一个
            没有 ``harvest_bonus`` 的新工具，导致"创新反而变差"）。没有工具时
            总是装备。该写入仍属于"库存"范围（契约允许），不触碰其它世界数据。
            ``equip=False`` 完全不装备（由集成层自行决定工具分配）。

        返回
        ----
        ``(ok, reason, innovation)``。原因码：

        * ``"ok"``：实验成功，材料已扣，效果已注册，``innovation`` 非空；
        * ``"invalid_proposal"`` / ``"invalid_components"``：提案结构非法；
        * ``"max_innovations"``：已达 3 项/世界上限（不扣材料）；
        * ``"component_not_known"`` / ``"unknown_knowledge"`` /
          ``"component_not_verified"``：组件前置不满足（不扣材料）；
        * ``"unknown_attribute"`` / ``"incompatible_tags"`` / ``"already_invented"``：
          组合规则不允许（不扣材料）；
        * ``"no_effect_backend"``：``world.technology`` 还没有
          ``register_custom_effect``（P3 早期降级，不扣材料）；
        * ``"missing_material"``：库存不足或扣减失败（回滚，不扣材料）；
        * ``"defect"``：**真实失败实验**，材料已消耗，无创新；
        * ``"effect_backend_error:<异常类型>"``：注册抛异常（材料已回滚，无创新）。
        """
        if not isinstance(proposal, Proposal):
            return (False, "invalid_proposal", None)
        components = tuple(str(item) for item in proposal.components)
        if len(components) != 2 or components[0] == components[1]:
            return (False, "invalid_components", None)
        if len(self._innovations) >= MAX_INNOVATIONS:
            return (False, "max_innovations", None)

        world = world if world is not None else _EmptyWorld()
        technology = getattr(world, "technology", None)
        knowledge = getattr(world, "knowledge", None)

        # 1) 前置：组件必须被该个体掌握且 verified
        ok, reason = self._check_components(agent, knowledge, components)
        if not ok:
            return (False, reason, None)

        # 组合规则（用世界规则重算，不信任 Proposal 里的数值）
        attrs = [attribute(component) for component in components]
        if any(attr is None for attr in attrs):
            return (False, "unknown_attribute", None)
        attr_a, attr_b = attrs[0], attrs[1]
        assert attr_a is not None and attr_b is not None  # for type checkers
        rule = rule_for(attr_a, attr_b)
        if rule is None:
            return (False, "incompatible_tags", None)
        pair = self._pair_key(components)
        if pair in {self._pair_key(item.components) for item in self._innovations}:
            return (False, "already_invented", None)
        if self.is_registered(rule.output_tool):
            return (False, "already_invented", None)

        # 2) 效果后端（P3 早期没有 ⇒ 前置失败，不扣材料）
        register = getattr(technology, "register_custom_effect", None)
        if not callable(register):
            return (False, NO_EFFECT_BACKEND_REASON, None)

        # 3) 材料：检查 + 真实扣除（失败回滚）
        cost = material_cost(knowledge, components, fallback=proposal.inputs)
        if not self._can_afford(agent, cost, technology):
            return (False, "missing_material", None)
        if not self._take_materials(agent, cost):
            return (False, "missing_material", None)

        # 4) 难度判定：失败也消耗材料（真实实验成本）
        difficulty = combination_difficulty((attr_a, attr_b))
        now = float(getattr(world, "t", 0.0) or 0.0)
        agent_id = int(getattr(agent, "agent_id", -1))
        if float(self.rng.random()) < difficulty:
            self._log(
                now=now,
                agent_id=agent_id,
                proposal=proposal,
                pair=pair,
                rule=rule,
                cost=cost,
                difficulty=difficulty,
                ok=False,
                reason="defect",
                innovation=None,
            )
            return (False, "defect", None)

        # 5) 注册真实世界效果；后端抛异常则回滚材料（不吞成本）
        effects = dict(rule.effects)
        try:
            register(rule.output_tool, dict(effects))
        except Exception as error:  # noqa: BLE001 - 后端异常不得击穿创新层
            self._refund_materials(agent, cost)
            reason_code = f"effect_backend_error:{type(error).__name__}"
            self._log(
                now=now,
                agent_id=agent_id,
                proposal=proposal,
                pair=pair,
                rule=rule,
                cost=cost,
                difficulty=difficulty,
                ok=False,
                reason=reason_code,
                innovation=None,
            )
            return (False, reason_code, None)

        # 6) 记录创新（名称由组分真实名称确定性拼接，不随机生成）
        self._serial += 1
        innovation = Innovation(
            innovation_id=f"innovation_{self._serial:03d}",
            name_zh=self._canonical_name(knowledge, components, rule),
            components=pair,
            effects=effects,
            invented_by=agent_id,
            invented_at=now,
            successful_uses=0,
        )
        self._innovations.append(innovation)
        self._custom_effects[rule.output_tool] = dict(effects)
        # 新工具交给发明者：效果只有挂在真实 ``inventory.tool`` 上才会改变
        # harvest / 体力恢复 / 水分恢复 / 制作成功率等世界数值（"不降级"策略）。
        equipped = self._equip(agent, rule.output_tool, effects, technology) if equip else False
        self._log(
            now=now,
            agent_id=agent_id,
            proposal=proposal,
            pair=pair,
            rule=rule,
            cost=cost,
            difficulty=difficulty,
            ok=True,
            reason="ok",
            innovation=innovation,
            equipped=equipped,
        )
        return (True, "ok", innovation)

    # ------------------------------------------------------------------
    # 摘要 / 序列化
    # ------------------------------------------------------------------
    def summary_zh(self) -> str:
        """一行中文摘要（已发明/上限、效果、尝试次数与材料成本）。"""
        if not self._innovations:
            return (
                f"创新实验室：尚无发明（上限 {MAX_INNOVATIONS} 项）；"
                f"组合需要至少 2 条已验证且 tag 相容的技术，已记录尝试{len(self.history)}次"
            )
        parts: list[str] = []
        for item in self._innovations:
            effect_text = "、".join(
                f"{EFFECT_LABELS_ZH.get(key, key)}+{value:g}"
                for key, value in sorted(item.effects.items())
            )
            parts.append(f"{item.name_zh}（{effect_text}）")
        return (
            f"创新实验室：已发明 {len(self._innovations)}/{MAX_INNOVATIONS} 项——"
            + "；".join(parts)
            + f"；记录尝试{len(self.history)}次"
        )

    def to_dict(self) -> dict[str, object]:
        """序列化实验室状态（创新列表 + 已注册效果 + 审计日志）。"""
        return {
            "version": LAB_SCHEMA_VERSION,
            "seed": self.seed,
            "serial": self._serial,
            "innovations": [item.to_dict() for item in self._innovations],
            "custom_effects": {
                str(tool): {str(key): float(value) for key, value in effects.items()}
                for tool, effects in self._custom_effects.items()
            },
            "history": [dict(entry) for entry in self.history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "InnovationLab":
        """从 :meth:`to_dict` 的结果恢复（RNG 以 ``seed`` 重建，不持久化状态）。"""
        lab = cls(seed=int(data.get("seed", 0)))
        lab._innovations = [
            Innovation.from_dict(dict(item)) for item in (data.get("innovations") or ())
        ][:MAX_INNOVATIONS]
        lab._serial = max(int(data.get("serial", 0)), len(lab._innovations))
        lab._custom_effects = {
            str(tool): {str(key): float(value) for key, value in dict(effects).items()}
            for tool, effects in (data.get("custom_effects") or {}).items()
        }
        lab.history = [dict(entry) for entry in (data.get("history") or ())]
        return lab

    # ------------------------------------------------------------------
    # 内部：知识 / 站点
    # ------------------------------------------------------------------
    @staticmethod
    def _is_verified(record: Any) -> bool:
        """``Knowledge`` 的 verified 判定（状态优先，退化为 ``usable()``）。"""
        status = getattr(record, "status", None)
        if status is not None:
            return str(status) == "verified"
        usable = getattr(record, "usable", None)
        if callable(usable):
            try:
                return bool(usable())
            except Exception:  # noqa: BLE001 - duck-typed 对象异常不击穿创新层
                return False
        return False

    def _knowledge_records(self, agent: Any, knowledge: Any) -> list[Any]:
        """取该个体掌握的知识记录（优先 ``knowledge.known_by``，再退化为 id 列表）。"""
        known_by = getattr(knowledge, "known_by", None)
        if callable(known_by):
            try:
                records = [record for record in known_by(agent) if record is not None]
                if records:
                    return records
            except Exception:  # noqa: BLE001 - 集成层微小偏差不阻断提案
                pass
        records = []
        for knowledge_id in getattr(agent, "known_knowledge_ids", None) or ():
            record = _get_record(knowledge, str(knowledge_id))
            if record is not None:
                records.append(record)
        return records

    def _check_components(
        self,
        agent: Any,
        knowledge: Any,
        components: tuple[str, ...],
    ) -> tuple[bool, str]:
        """检查组件是否被该个体掌握且 verified（无知识库时降级为只查掌握）。"""
        known = {str(item) for item in (getattr(agent, "known_knowledge_ids", None) or ())}
        for component in components:
            if component not in known:
                return (False, "component_not_known")
        store_get = getattr(knowledge, "get", None)
        if knowledge is None or not callable(store_get):
            # 降级：没有知识库后端时无法区分 verified，退化为"个体已知即可"
            return (True, "ok")
        for component in components:
            record = _get_record(knowledge, component)
            if record is None:
                return (False, "unknown_knowledge")
            if not self._is_verified(record):
                return (False, "component_not_verified")
        return (True, "ok")

    @staticmethod
    def _site_ready(agent: Any) -> bool:
        """工作台/站点条件（duck typing，三级判定）。

        1. 集成层显式钩子 ``agent.can_innovate() -> bool``（最高优先级）；
        2. ``agent.nearby_facilities`` 里存在 **active** 的 ``workshop``；
        3. 兜底：手上已有工具（``agent.inventory.tool`` 非 None），代表最低限度的
           "能动手做实验"的条件。

        没有这三者之一时不提提案，但 :meth:`attempt` 只检查知识前置（集成层可以
        直接驱动实验）。
        """
        hook = getattr(agent, "can_innovate", None)
        if callable(hook):
            try:
                return bool(hook())
            except Exception:  # noqa: BLE001 - 钩子异常按"条件不满足"处理
                return False
        for facility in getattr(agent, "nearby_facilities", None) or ():
            if str(getattr(facility, "kind", "")) == "workshop" and bool(
                getattr(facility, "active", True)
            ):
                return True
        inventory = getattr(agent, "inventory", None)
        return getattr(inventory, "tool", None) is not None

    def _equip(
        self,
        agent: Any,
        tool_name: str,
        effects: dict[str, float],
        technology: Any,
    ) -> bool:
        """把验证成功的新工具交给发明者（不降低既有采集能力的"不降级"策略）。

        规则（确定性、可解释）：

        * 手上没有工具 → 直接装备；
        * 手上已有工具 → 只有新工具的采集能力（``harvest_power``）不低于旧工具
          时才替换。这样 ``edge+binder``（+1 采集）可以替换斧头（1+1=2 与斧头
          的 2 持平），而纯载重/纯恢复工具不会把斧头的产量翻倍换掉（避免"创新
          反而让世界变差"）。

        ``technology.custom_effect`` 可用时用它查询旧工具效果（其他实验室/集成层
        注册的工具也能量到），否则用本实验室的注册表。
        """
        inventory = getattr(agent, "inventory", None)
        if inventory is None:
            return False
        current = getattr(inventory, "tool", None)
        if current is not None:
            old_bonus = self._harvest_bonus_of(current, technology)
            new_bonus = float(effects.get("harvest_bonus", 0.0))
            if self._harvest_power(tool_name, new_bonus) < self._harvest_power(
                current, old_bonus
            ):
                return False
        try:
            inventory.tool = str(tool_name)
        except Exception:  # noqa: BLE001 - 只读/冻结库存实现按"未能装备"处理
            return False
        return True

    def _harvest_bonus_of(self, tool_name: Any, technology: Any) -> float:
        """查询某工具已注册的 ``harvest_bonus``（跨实验室/集成层也能查到）。"""
        query = getattr(technology, "custom_effect", None) if technology is not None else None
        if callable(query):
            try:
                return float(query(tool_name, "harvest_bonus"))
            except Exception:  # noqa: BLE001 - 查询失败退回本实验室注册表
                pass
        return float((self._custom_effects.get(str(tool_name or "")) or {}).get("harvest_bonus", 0.0))

    @staticmethod
    def _harvest_power(tool_name: Any, harvest_bonus: float) -> float:
        """采集能力估算：与 ``technology.harvest`` 的基础规则一致（斧/镐 2，其余 1）。"""
        base = 2.0 if str(tool_name or "") in ("axe", "pickaxe") else 1.0
        return base + float(harvest_bonus)

    # ------------------------------------------------------------------
    # 内部：材料（真实库存接口）
    # ------------------------------------------------------------------
    @staticmethod
    def _inventory_has(agent: Any, cost: dict[str, int]) -> bool:
        """库存是否满足成本（优先 ``Inventory.has``）。"""
        inventory = getattr(agent, "inventory", None)
        if inventory is None:
            return not cost
        for kind, amount in dict(cost).items():
            has = getattr(inventory, "has", None)
            if callable(has):
                try:
                    if not bool(has(kind, int(amount))):
                        return False
                    continue
                except Exception:  # noqa: BLE001 - 退化为读字段
                    pass
            if int(getattr(inventory, kind, 0) or 0) < int(amount):
                return False
        return True

    def _can_afford(self, agent: Any, cost: dict[str, int], technology: Any) -> bool:
        """先用 ``technology.can_afford`` 风格的公开接口，再退化为库存字段读取。"""
        if not cost:
            return True
        can_afford = getattr(technology, "can_afford", None) if technology is not None else None
        if callable(can_afford):
            try:
                return bool(can_afford(agent, _CostRecipe(dict(cost))))
            except Exception:  # noqa: BLE001 - 集成层条件不满足时退回库存检查
                pass
        return self._inventory_has(agent, cost)

    @staticmethod
    def _take_materials(agent: Any, cost: dict[str, int]) -> bool:
        """原子扣材料：任一材料不足则回滚已扣部分并返回 ``False``。"""
        inventory = getattr(agent, "inventory", None)
        if inventory is None:
            return not cost
        taken: dict[str, int] = {}
        take = getattr(inventory, "take", None)
        for kind, amount in dict(cost).items():
            amount = int(amount)
            if amount <= 0:
                continue
            if callable(take):
                try:
                    ok = bool(take(kind, amount))
                except Exception:  # noqa: BLE001 - 扣减异常按失败处理并回滚
                    ok = False
                if not ok:
                    InnovationLab._refund_materials(agent, taken)
                    return False
            else:
                current = int(getattr(inventory, kind, 0) or 0)
                if current < amount:
                    InnovationLab._refund_materials(agent, taken)
                    return False
                setattr(inventory, kind, current - amount)
            taken[kind] = taken.get(kind, 0) + amount
        return True

    @staticmethod
    def _refund_materials(agent: Any, cost: dict[str, int]) -> None:
        """把已扣的材料放回库存（数量刚刚腾出，必然放得下）。"""
        inventory = getattr(agent, "inventory", None)
        if inventory is None:
            return
        add = getattr(inventory, "add", None)
        for kind, amount in dict(cost).items():
            amount = int(amount)
            if amount <= 0:
                continue
            if callable(add):
                try:
                    if bool(add(kind, amount)):
                        continue
                except Exception:  # noqa: BLE001 - 回滚尽力而为
                    pass
            setattr(inventory, kind, int(getattr(inventory, kind, 0) or 0) + amount)

    # ------------------------------------------------------------------
    # 内部：命名 / 审计 / 序列化辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _pair_key(components: Iterable[str]) -> tuple[str, ...]:
        """组分配对的规范键（排序去重，用于"已发明"判定）。"""
        return tuple(sorted({str(item) for item in components}))

    @staticmethod
    def _compose_name(names: dict[str, str], pair: tuple[str, ...], rule: EffectRule) -> str:
        """由真实技术名与规则名确定性拼接（不随机生成名字）。"""
        parts = [names.get(component) or component for component in pair]
        return f"{rule.name_zh}（{'＋'.join(parts)}）"

    def _canonical_name(self, knowledge: Any, components: tuple[str, ...], rule: EffectRule) -> str:
        """按知识库里的真实技术名重算规范名称（attempt 用）。"""
        names: dict[str, str] = {}
        for component in components:
            record = _get_record(knowledge, component)
            names[component] = str(getattr(record, "name_zh", "") or component)
        return self._compose_name(names, self._pair_key(components), rule)

    @staticmethod
    def _rationale(
        names: dict[str, str],
        pair: tuple[str, ...],
        attr_a: RecipeAttribute,
        attr_b: RecipeAttribute,
        rule: EffectRule,
        cost: dict[str, int],
    ) -> str:
        """中文可解释理由：相容 tag 对、确定性效果、真实材料成本与难度。"""
        pair_tags = [tag for tag in rule.tags]
        pair_tags.sort()
        effects_text = "、".join(
            f"{EFFECT_LABELS_ZH.get(key, key)}+{value:g}"
            for key, value in sorted(rule.effects.items())
        )
        cost_text = "、".join(f"{kind}×{amount}" for kind, amount in sorted(cost.items())) or "无"
        difficulty = combination_difficulty((attr_a, attr_b))
        name_a = names.get(pair[0]) or pair[0]
        name_b = names.get(pair[1]) or pair[1]
        return (
            f"{name_a}（tags:{'/'.join(attr_a.tags)}）与 {name_b}"
            f"（tags:{'/'.join(attr_b.tags)}）通过 {'+'.join(pair_tags)} 相容规则组合，"
            f"确定性产出「{rule.name_zh}」：{effects_text}；"
            f"材料成本 {cost_text}，组合难度 {difficulty:.2f}"
        )

    def _log(
        self,
        *,
        now: float,
        agent_id: int,
        proposal: Proposal,
        pair: tuple[str, ...],
        rule: EffectRule,
        cost: dict[str, int],
        difficulty: float,
        ok: bool,
        reason: str,
        innovation: Innovation | None,
        equipped: bool = False,
    ) -> None:
        """追加一条审计记录（成功/失败、真实成本、原因码、是否装备）。"""
        self.history.append(
            {
                "t": float(now),
                "agent_id": int(agent_id),
                "proposal_id": proposal.proposal_id,
                "components": list(pair),
                "inputs": {str(key): int(value) for key, value in cost.items()},
                "output_tool": rule.output_tool,
                "difficulty": float(difficulty),
                "ok": bool(ok),
                "reason": str(reason),
                "innovation_id": None if innovation is None else innovation.innovation_id,
                "equipped": bool(equipped),
            }
        )
        if len(self.history) > HISTORY_LIMIT:
            self.history = self.history[-HISTORY_LIMIT:]

    def _find_innovation(self, innovation: Innovation | str) -> Innovation | None:
        """按对象身份或 id 找到内部记录的创新。"""
        if isinstance(innovation, Innovation):
            for item in self._innovations:
                if item is innovation:
                    return item
            return None
        target_id = str(innovation)
        for item in self._innovations:
            if item.innovation_id == target_id:
                return item
        return None


class _EmptyWorld:
    """``attempt(..., world=None)`` 的最小占位：没有 technology/knowledge。"""

    t: float = 0.0
    technology: Any = None
    knowledge: Any = None


# ---------------------------------------------------------------------------
# duck typing 辅助
# ---------------------------------------------------------------------------
def _get_record(knowledge: Any, knowledge_id: str) -> Any:
    """从 duck-typed 知识库取记录：支持 ``get`` 方法、普通 dict、``catalog`` 字典。"""
    if knowledge is None:
        return None
    store_get = getattr(knowledge, "get", None)
    if callable(store_get):
        try:
            return store_get(str(knowledge_id))
        except Exception:  # noqa: BLE001 - 非标准后端按"取不到"处理
            return None
    if isinstance(knowledge, dict):
        return knowledge.get(str(knowledge_id))
    catalog = getattr(knowledge, "catalog", None)
    if isinstance(catalog, dict):
        return catalog.get(str(knowledge_id))
    return None
