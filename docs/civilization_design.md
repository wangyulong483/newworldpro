# 微型文明演化机制设计（接口契约 v1）

> 本文档是 **P1–P4 阶段的冻结接口契约**。新增模块必须按此签名实现；
> 集成层（`population.py` / `world.py` / `society.py` / `technology.py`）
> 由 Lead 负责，新增模块不得反向 import 这些模块（避免循环依赖）。
>
> 基线事实（来自 P0 审计，见 `docs/architecture_audit.md`）：
> 现有"强化学习"实际是**监督式行为克隆**（MLP 24→64→32→58，模仿手写稳态教师），
> 没有奖励函数、没有 RL 训练循环；`env.py` 的 8 维 / 3 动作 RL 包装器**未被主程序使用**。
> 人口上限 `MAX_POPULATION = 8` 约束的是**当前存活数**，死亡后从 `world.agents`
> 与 `population.minds` 中删除，**历史与知识全部丢失**——这是本设计要修的根本缺陷。

---

## 0. 设计原则（不可违反）

1. **世界规则唯一权威**：任何材料/技术/设施的状态变化都必须经 `Technology`
   模拟器执行与验证；规划器/知识库/神经网络只能*请求*动作，不能直接改世界数据。
2. **反固定剧情**：目标选择、计划生成、技术发现全部由「需求 + 库存 + 环境 +
   已有知识 + 随机实验」驱动；允许停滞、失败、灭绝。禁止写死"第 N 回合出现 X"。
3. **8 个创建名额是硬约束**：`AgentRegistry.total_created <= max_agents_created`。
   死亡个体保留档案（`AgentRecord`），不能行动、不能从历史中删除。
4. **渐进可回滚**：所有新机制由 `CivilizationConfig` 开关控制（A/B/C/D 消融组），
   关闭时行为与原始系统一致。
5. **零初始化迁移**：扩展神经网络输入维度时，新增输入列的权重初始化为 0，
   保证旧 checkpoint 行为完全一致，可回滚、可比对。
6. **不新增第三方依赖**：只用 numpy / 标准库（+ 现有 pillow / tkinter）。

---

## 1. 模块与依赖方向

```
                    ┌──────────────────────────────────────────┐
                    │            population.py (集成层)          │
                    │  AgentMind.decide: 规划 → 社会教师 → 神经   │
                    └───┬────────┬────────┬────────┬───────┬────┘
                        │        │        │        │       │
              goals.py  │  memory.py  knowledge.py  civilization.py  telemetry.py
                        │        │        │        │       │
                        └────────┴────────┴────────┴───────┘
                                     ↓ 只依赖 duck-typed world/agent
        world.py / technology.py / society.py / character.py（Lead 修改）
```

依赖规则：
- `memory.py` / `knowledge.py` / `civilization.py` 互不 import（`civilization.py`
  可以用 `TYPE_CHECKING` 或 duck typing 持有 knowledge/memory 引用）。
- `goals.py` 可以 import `memory`、`knowledge`、`behavior.Decision`。
- `telemetry.py` 只 import 标准库 + numpy + `civilization`（可用 duck typing）。
- 任何模块都**不得** import `population` / `world`。

---

## 2. 配置：`config.CivilizationConfig`（Lead 实现）

```python
@dataclass
class CivilizationConfig:
    # -- 人口硬约束 -------------------------------------------------
    max_agents_created: int = 8
    lifespan_s: float = 900.0
    maturity_s: float = 70.0
    pregnancy_s: float = 24.0

    # -- 机制开关（消融实验） ---------------------------------------
    enable_planning: bool = True       # P1 目标规划
    enable_knowledge: bool = True      # P2 知识/技术
    enable_inheritance: bool = True    # P3 记忆/传承/人口上限
    enable_social: bool = True         # P4 合作/公共设施/规则
    enable_disturbance: bool = False   # P4 环境扰动
    expanded_observation: bool = True  # 神经输入 24→40（P3）
    enable_perception: bool = True     # headless 实验可关（视觉占 95% 时间）

    seed: int = 0

ABLATIONS: dict[str, dict[str, bool]] = {
    # A=原始系统，B=+规划，C=+持久化知识与传承，D=完整系统
    "A": dict(enable_planning=False, enable_knowledge=False, enable_inheritance=False,
              enable_social=False, enable_disturbance=False, expanded_observation=False),
    "B": dict(enable_planning=True,  enable_knowledge=False, enable_inheritance=False,
              enable_social=False, enable_disturbance=False, expanded_observation=False),
    "C": dict(enable_planning=True,  enable_knowledge=True,  enable_inheritance=True,
              enable_social=False, enable_disturbance=False, expanded_observation=True),
    "D": dict(enable_planning=True,  enable_knowledge=True,  enable_inheritance=True,
              enable_social=True,  enable_disturbance=True,  expanded_observation=True),
}
```

---

## 3. `civilization.py` —— 历史注册表 / 文明状态 / 公共设施（P3/P4）

```python
MAX_AGENTS_CREATED = 8

class PopulationLimitError(RuntimeError):
    """创建第 9 个 agent 时抛出。"""

@dataclass
class AgentRecord:
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
    inventory: dict[str, int] = field(default_factory=dict)      # 死亡快照
    known_knowledge_ids: tuple[str, ...] = ()                     # 死亡时掌握
    skills: dict[str, float] = field(default_factory=dict)
    children: tuple[int, ...] = ()
    contributions: dict[str, float] = field(default_factory=dict) # 对公共设施/合作
    memory_archive: dict[str, object] = field(default_factory=dict)
    def to_dict(self) -> dict[str, object]
    @classmethod
    def from_dict(cls, data: dict) -> "AgentRecord"

class AgentRegistry:
    def __init__(self, max_agents: int = MAX_AGENTS_CREATED) -> None
    @property
    def total_created(self) -> int            # 历史累计创建数（只增不减）
    @property
    def total_deaths(self) -> int
    def remaining(self) -> int                # max_agents - total_created
    def can_create(self) -> bool
    def register(self, agent, *, t: float, parent_id: int | None = None) -> AgentRecord
        # 满额抛 PopulationLimitError；agent 需有 agent_id/name/sex/generation/parents
    def record(self, agent_id: int) -> AgentRecord | None
    def records(self) -> list[AgentRecord]    # 按 agent_id 升序，含已死亡
    def alive_ids(self) -> list[int]
    def mark_dead(self, agent_id: int, *, t: float, cause: str,
                  final_state: dict | None = None) -> AgentRecord | None
    def alive_count(self) -> int
    def lineage(self) -> list[tuple[int, int]]      # (parent_id, child_id)
    def to_dict(self) -> dict[str, object]
    @classmethod
    def from_dict(cls, data: dict) -> "AgentRegistry"

@dataclass
class Facility:
    facility_id: str
    kind: str                 # "campfire" | "hut" | "well" | "workshop"
    name_zh: str
    x_screen: float
    condition: float = 1.0    # 0..1 损耗度
    built_at: float = 0.0
    built_by: int = -1
    maintenance_wood: int = 0
    maintenance_stone: int = 0
    maintenance_interval_s: float = 120.0
    last_maintained_at: float = 0.0
    private_uses: int = 0
    def decay(self, t: float) -> float        # 返回新 condition（不维护则下降）
    def maintained(self, agent_id: int, t: float) -> None
    @property
    def active(self) -> bool                  # condition > 0.2
    def to_dict(self) / from_dict(cls, data)

@dataclass
class SocialContract:
    contract_id: str
    kind: str                 # "share_food" | "communal_store" | "apprentice"
    name_zh: str
    proposed_by: int
    created_at: float
    participants: tuple[int, ...] = ()
    active: bool = True
    def to_dict(self) / from_dict(cls, data)

class CivilizationState:
    def __init__(self, registry: AgentRegistry, knowledge=None, memory=None) -> None
    registry: AgentRegistry
    facilities: list[Facility]
    contracts: list[SocialContract]
    innovation_count: int          # 已验证知识总数（从 knowledge 同步）
    cooperation_index: float       # 见 §8 指标定义
    collapse_state: str            # stable | stagnating | collapsing | collapsed | recovering
    extinct: bool
    extinct_at: float | None
    extinction_cause: str
    history: list[dict[str, object]]   # 每个统计窗口一条
    def update(self, world, dt: float) -> None
        # 1) 设施损耗/失效 2) 灭绝检测 3) collapse_state 4) 同步 innovation_count
    def register_death(self, agent, *, t, cause) -> AgentRecord | None
    def build_facility(self, kind, x_screen, agent_id, t, *, inventory=None) -> Facility | None
    def maintain(self, facility_id, agent_id, t, *, wood=0, stone=0) -> bool
    def metrics(self, world) -> dict[str, float]     # §8 的文明级指标
    def summary_zh(self) -> str
    def to_dict(self) / @classmethod from_dict(cls, data, knowledge=None, memory=None)
```

**灭绝规则**：`alive_count() == 0` 且 `can_create() == False` → `extinct=True`；
或 `alive_count()==0` 且 `enable_inheritance=False`（不允许新生）→ 同样灭绝。
灭绝后世界不再产生新个体，`collapse_state="collapsed"`，只保留历史。

---

## 4. `goals.py` —— 目标 / 计划 / 规划器（P1，P2 扩展）

```python
GOAL_TYPES = ("survive", "gather", "craft", "build", "experiment",
              "explore", "socialize", "reproduce", "maintain")
GOAL_STATUS = ("active", "completed", "failed", "abandoned")
PLAN_STATUS = ("active", "completed", "failed", "replanned", "abandoned")
STEP_STATUS = ("pending", "running", "done", "failed")

CONDITION_KINDS = (
    "inventory_ge", "has_tool", "knowledge_verified", "structure_count_ge",
    "facility_active", "energy_ge", "hydration_ge", "at_site", "near_agent",
    "progress_ge", "always",
)

@dataclass(frozen=True)
class Condition:
    kind: str
    target: str = ""
    op: str = ">="
    value: float = 0.0
    def evaluate(self, world, agent) -> bool      # 任一条件异常返回 False，不得抛错

@dataclass
class Goal:
    goal_id: int
    agent_id: int
    goal_type: str
    status: str = "active"
    priority: float = 0.5
    completion: tuple[Condition, ...] = ()
    failure: tuple[Condition, ...] = ()
    estimated_cost: float = 1.0
    max_steps: int = 1800            # 每步 dt 累加
    created_at: float = 0.0
    steps_spent: int = 0
    progress: float = 0.0
    last_progress_at: float = 0.0
    attempts: int = 0
    source: str = "need"             # need | opportunity | inherited | social
    rationale: str = ""
    recipe_id: str | None = None     # experiment/craft 目标关联的知识
    def is_done(self, world, agent) -> bool       # completion 全满足
    def is_failed(self, world, agent) -> bool     # failure 任一满足 或 超时

@dataclass
class PlanStep:
    index: int
    verb: str          # 见下方 VERBS
    args: dict[str, object] = field(default_factory=dict)
    preconditions: tuple[Condition, ...] = ()
    expected_effect: dict[str, float] = field(default_factory=dict)
    cost: float = 1.0
    status: str = "pending"
    attempts: int = 0
    max_attempts: int = 4
    last_error: str = ""

#: 规划器可生成的结构化步骤（执行器必须实现，未知 verb → 计划失败并记录）
VERBS = ("move_to", "gather", "craft", "experiment", "contribute", "build",
         "maintain", "drink", "eat", "rest", "explore", "observe",
         "deliver", "socialize", "reproduce")

@dataclass
class Plan:
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
    def current_step(self) -> PlanStep | None
    def advance(self) -> None

class Planner:
    def __init__(self, knowledge=None, *, memory=None) -> None
    def plan(self, world, agent, goal) -> Plan | None
    def replan(self, world, agent, goal, failed_plan: Plan, reason: str) -> Plan | None
    # 内部：从知识库 recipe 图做代价搜索（无知识时退化为内置基础动作链）

class GoalManager:
    def __init__(self, *, seed: int = 0, knowledge=None, memory=None,
                 civilization=None, config=None) -> None
    def update(self, world, agent, dt: float) -> Decision | None
        # 1) 结算已完成/失败目标 2) 必要时生成新目标（需求/机会驱动）
        # 3) 选择优先级最高的活跃目标 4) 推进计划一步 5) 返回原始动作或 None
        # 6) 无进展检测 → replan/abandon；所有结果写入 memory
    def goals_of(self, agent_id: int) -> list[Goal]
    def active_plan(self, agent_id: int) -> Plan | None
    def stats(self) -> dict[str, float]
        # {"goals_created","goals_completed","goals_failed","goals_abandoned",
        #  "plans_created","replans","steps_executed","plan_success_rate",
        #  "cost_estimate_error"}
    def summary_zh(self) -> str
    def to_dict(self) / from_dict(cls, data, **deps)
```

**反卡死要求（验收 P1）**：
- `last_progress_at` 超过 `NO_PROGRESS_S`（默认 45 世界秒，或 240 步）→ `replan`。
- 同一 `plan_id` 连续 `replan >= 3` 次 → `abandoned`，进入冷却（该目标类型 60s 内不再生成）。
- 步骤 `attempts > max_attempts` → 步骤失败 → 计划失败 → 依据失败原因重规划或放弃。

---

## 5. `knowledge.py` —— 持久化知识 / 技术依赖图（P2/P3）

```python
KNOWLEDGE_TYPES = ("tool", "technique", "structure", "resource")
KNOWLEDGE_STATUS = ("hypothesis", "verified", "invalidated")

@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    name_zh: str
    inputs: dict[str, int]                    # 材料 -> 数量
    outputs: dict[str, int]                   # 材料/工具 -> 数量
    requires_tool: str | None = None
    requires_knowledge: tuple[str, ...] = ()
    work: int = 1                             # 连续成功次数达到才 verified
    difficulty: float = 0.5                   # 0..1，单次尝试失败概率（世界规则随机）

@dataclass
class ValidationRecord:
    t: float
    agent_id: int
    success: bool
    evidence: str

@dataclass
class Knowledge:
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
    def usable(self) -> bool
    def record(self, t: float, agent_id: int, success: bool, evidence: str = "") -> None

#: 世界里"存在但未被发现"的全部技术（由 Recipe 定义，不是剧情）
RECIPE_CATALOG: dict[str, Recipe]

class KnowledgeStore:
    def __init__(self, *, seed: int = 0, catalog: dict[str, Recipe] | None = None,
                 bootstrap: bool = True) -> None
        # bootstrap=True 时把 RECIPE_CATALOG 全部登记为 status="hypothesis"
        # 但只有"可负担 + 前置已验证"的假设才会进入候选（见 candidates_for）
    def add(self, knowledge: Knowledge) -> Knowledge
    def get(self, knowledge_id: str) -> Knowledge | None
    def all(self, *, status: str | None = None) -> list[Knowledge]
    def known_by(self, agent) -> list[Knowledge]          # 读 agent.known_knowledge_ids
    def candidates_for(self, agent, world=None) -> list[Knowledge]
        # 前置知识已 verified + 材料可负担（world.technology.can_afford）
    def hypothesize(self, agent, recipe_id: str, t: float) -> Knowledge | None
    def attempt(self, agent, knowledge, world, t: float) -> tuple[bool, str]
        # 交给 world.technology.simulate_recipe(agent, recipe) —— 唯一权威验证
    def mark_invalid(self, knowledge_id: str, reason: str, t: float = 0.0) -> None
    def dependents(self, knowledge_id: str) -> set[str]
    def edges(self) -> list[tuple[str, str]]
    def depth(self) -> int                    # 已验证技术图的最长依赖链
    def retention(self, known_ids) -> float
    def innovation_efficiency(self) -> float  # verified / 总尝试次数
    def summary_zh(self) -> str
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeStore"
```

**验证语义（唯一合法路径）**：
- `hypothesis`：某个 agent 在库存里凑齐了材料并**主动尝试**后生成（不是开局全部已知）。
- 每次尝试调用 `world.technology.simulate_recipe()`：真实消耗/产出，按
  `difficulty` 用世界 RNG 判定是否暴露缺陷。
- 连续 `work` 次成功 → `verified`，写入 `successful_uses`；失败累计 →
  `confidence` 下降，`failed_uses >= 3` 或环境失效 → `invalidated`。
- 只有 `verified` 的 recipe 允许被规划器用于 `craft`/`build`；
  `hypothesis` 只能用于 `experiment` 步骤。
- 发现、传播（学会）、使用、遗失（掌握者全部死亡且无公共档案）都必须可追踪。

**技术内容（世界规则，非剧情）**：见 `docs/architecture_audit.md` §技术依赖图草案。
最小集合：`stone_axe`（木+石→斧，采集×2）、`fiber_cord`（草/叶纤维→绳）、
`basket`（木+绳→背篓，负重上限+）、`campfire`（木+石→篝火，夜间保暖/照明）、
`hut`（木+石+绳→小屋，休息恢复+知识存储）、`well`（石+绳→水井，取水距离+）。
每条都**真实改变世界数值**（采集成品、负重、恢复速率、移动成本、取水半径）。

---

## 6. `memory.py` —— 个体记忆（P3）

```python
MEMORY_FEATURE_DIM = 8
CAPACITY_PER_AGENT = 256

@dataclass
class MemoryEntry:
    memory_id: int
    agent_id: int
    t: float
    kind: str          # "episode" | "procedural" | "social"
    action: str        # 例如 "gather:wood" / "craft:stone_axe" / "move:+1"
    outcome: str       # 机器可读结果，例如 "ok:wood+1" / "fail:no_resource"
    success: bool
    detail: str = ""
    salience: float = 0.5

@dataclass
class Procedure:
    skill: str
    steps: tuple[str, ...]
    successes: int = 0
    failures: int = 0
    last_used_at: float = 0.0
    def success_rate(self) -> float

class MemoryStore:
    def __init__(self, capacity_per_agent: int = CAPACITY_PER_AGENT) -> None
    def record(self, agent_id, t, kind, action, outcome, success,
               detail="", salience=0.5) -> MemoryEntry
    def entries(self, agent_id, *, kind: str | None = None,
                limit: int | None = None) -> list[MemoryEntry]
    def recent_failures(self, agent_id, k: int = 5) -> list[MemoryEntry]
    def failure_counts(self, agent_id, window: int = 32) -> dict[str, int]
    def feature_vector(self, agent_id) -> np.ndarray   # (8,) float32, 每维 [-1,1]
    def learn_procedure(self, agent_id, skill: str, steps: tuple[str, ...]) -> Procedure
    def procedures(self, agent_id) -> list[Procedure]
    def transfer(self, source_id: int, target_id: int, *,
                 kinds: tuple[str, ...] = ("procedural",), limit: int = 64) -> int
        # 代际/师徒传承：复制程序性记忆（技能），返回复制条数
    def archive(self, agent_id: int) -> dict[str, object]
        # 死亡：返回可序列化档案；**不删除**条目
    def retention(self, agent_id: int) -> float
    def agent_count(self) -> int
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryStore"
```

`feature_vector` 8 维（供 P3 扩展神经输入，见 §7）：
`[近期成功率, 近期失败率, 重复动作比例, 动作种类数/8, 是否掌握技能,
最佳技能成功率, 目标成功率, 距上次有效进展归一化]`。

---

## 7. 神经网络观测扩展（Lead，P3）

- `neural.FEATURE_DIM` 24 → **40**：新增 16 维 =
  8 维记忆摘要 + 4 维目标/计划上下文（目标类型 one-hot 压缩、计划进度、
  无进展时长）+ 4 维文明/社会上下文（已掌握知识数、公共设施、群体规模、
  合作指数）。
- `NeuralPolicy.load()` 兼容旧 24 维 checkpoint：前 24 维权重按旧布局读取，
  新增输入列权重**初始化为 0**，因此旧模型行为逐比特一致（有专项测试）。
- P3 训练时教师数据也计算新特征（BC 仍只模仿稳态行为；记忆/目标特征的
  习得依赖规划层，相关局限写入最终报告，不夸大）。

---

## 8. `telemetry.py` —— 指标采集与定义（P0/P4）

```python
METRIC_NAMES = (...)

class Telemetry:
    def __init__(self, *, window_s: float = 60.0, log_path: str | Path | None = None,
                 seed: int = 0, group: str = "") -> None
    def step(self, world, population=None, dt: float = 1/60) -> None
    def event(self, kind: str, **fields) -> None           # 出生/死亡/发现/验证/建造/灭绝
    def window_summary(self) -> dict[str, float]
    def summary(self) -> dict[str, float]                   # 全程聚合（§9 指标）
    def to_json(self, path: str | Path | None = None) -> Path
    def close(self) -> None
```

### 指标定义（必须按此实现，实验文档引用同一套）

| 指标 | 定义 |
| --- | --- |
| `survival_rate` | 窗口内仍存活的历史个体数 / 窗口开始时存活数（含出生分母按个体-秒计） |
| `action_entropy` | 行为类别（verb/behavior_state）分布的归一化熵，0..1；**不得**当作智能指标 |
| `repeat_run_max` | 同一动作连续执行的最长步数 |
| `no_progress_repeat_ratio` | 状态无变化（能量/位置/库存/目标进度均无变化）的连续重复动作步数 / 总步数 |
| `goal_completion_rate` | 完成目标数 / 生成目标数 |
| `plan_cost_error` | mean(abs(actual_steps - expected_cost) / max(1, expected_cost)) |
| `tech_verified` | 已验证知识数量 |
| `tech_graph_depth` | 已验证技术依赖图最长链 |
| `tech_graph_edges` | 已验证技术依赖图边数 |
| `knowledge_retention` | 当前存活个体掌握的知识 ∪ 公共档案 / 全部已验证知识 |
| `innovation_efficiency` | 已验证数 / 总实验尝试数 |
| `cooperation_benefit` | 合作（共享/贡献/教学）产出的资源或进度 / 独立行动产出，区间对称化 log 比 |
| `recovery_after_disturbance` | 扰动后 N 秒内指标恢复率（人口/知识/设施） |
| `total_agents_created` | registry.total_created（必须 ≤ 8） |
| `extinction_time` | 首次 `alive_count==0` 的世界时间（无灭绝记 -1） |
| `extinction_cause` | "starvation" / "old_age" / "collapse" / ""（事件记录） |

`Telemetry` 只读世界状态；**训练奖励与文明指标分开记录**（本系统无奖励函数，
用 `utility` 事件值记录目标收益，不参与训练）。

---

## 9. `simulation.py` —— headless 文明运行器（Lead）

```python
@dataclass
class RunSpec:
    group: str = "D"                 # A/B/C/D
    seed: int = 0
    steps: int = 20000
    dt: float = 0.1
    width: int = 800
    height: int = 500
    log_dir: str | Path | None = None
    telemetry_path: str | Path | None = None
    save_path: str | Path | None = None

class SocietyRunner:
    def __init__(self, spec: RunSpec) -> None
    def run(self) -> dict[str, object]     # 指标 + 事件 + 配置 + seed
    def save(self, path) -> Path
    @classmethod
    def load(cls, path) -> "SocietyRunner"
```

---

## 10. 验收对照表（与任务书 P0–P4 对应）

| 阶段 | 验收项 | 实现位置 | 测试 |
| --- | --- | --- | --- |
| P0 | 基线可复现、动作/奖励/状态变化可记录 | telemetry, experiments/baseline | test_telemetry |
| P1 | 多步任务自动完成；失败/无进展可识别；可重规划 | goals | test_goals |
| P2 | 合法动作取得材料→制作→验证；知识可复用；技术改变世界能力 | knowledge, technology | test_knowledge |
| P3 | 创建数 ≤8；死亡不可行动；知识跨个体/跨代保存；技术可追踪 | civilization, memory, population | test_civilization, test_inheritance |
| P4 | 规则驱动持续积累或可解释协作；扰动影响路线；允许灭绝；消融可说明差异 | simulation, experiments/ablation | test_experiments |

---

## 11. 已知取舍（写入最终报告，不隐藏）

1. 本系统没有可用的奖励学习信号（原项目是 BC），因此**不引入** PPO/DQN；
   规划与知识层用规则/代价搜索实现，神经网络保持 BC + 零初始化扩维。
2. `World.spawn()` 是几何/测试用的原语（不做上限检查）；
   **受约束的创建路径**是 `Population.create_agent()` → `AgentRegistry.register()`。
   所有文明指标只统计受约束路径。
3. 为了可观测性压缩了时间尺度（寿命 900 世界秒）；实验预算与步长在
   `docs/experiment_protocol.md` 中固定并公开。
