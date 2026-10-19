# 升级契约 v1（Lead 冻结；实现必须以本文件为准）

> 本轮升级（鼠标交互 + 智能决策）的模块边界与接口。四个并行工作流：
> **UI**（ui-dev）、**高层策略**（rl-dev）、**组合式创新**（innov-dev）、
> **集成/实验/文档**（Lead）。除本文件列出的文件外，任何人不得修改其他文件。

## 0. 通用约束

- Python 3.14，仅标准库 + numpy（UI 用 tkinter/Pillow，已有依赖）；**不得新增依赖**。
- 测试：`cd newworldpro; $env:TEMP=(Resolve-Path .tmp).Path; python -m unittest tests.test_xxx -v`
- 测试文件需要临时目录时，在 import 之后加：
  ```python
  try:
      from tests._env_compat import patch_tempdir
  except ImportError:
      from _env_compat import patch_tempdir
  patch_tempdir()
  ```
- 中文 docstring/注释，与项目现有风格一致；不得修改 `git` 已提交的他人文件（除本契约指定）。
- **不得**把规则包装成"神经网络学习"；**不得**让 UI 直接改世界状态。

## 1. UI 工作流（ui-dev）

### 1.1 `newworld/ui_state.py`（纯逻辑，禁止 import tkinter）

必须实现（签名可加可选参数，不可改名/改返回语义）：

```python
@dataclass
class Viewport:
    world_width: int; world_height: int
    view_width: int; view_height: int
    scale: float = 1.0
    offset_x: float = 0.0   # 世界坐标在视图中的左上角偏移（世界像素）
    offset_y: float = 0.0
    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]
    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]
    def zoom_at(self, sx: float, sy: float, factor: float) -> "Viewport"  # 光标下点不动
    def pan(self, dx_screen: float, dy_screen: float) -> "Viewport"
    def clamp(self) -> "Viewport"          # 世界不被拖出视图；scale 限制 [0.25, 4.0]
    def resize(self, view_width: int, view_height: int) -> "Viewport"
    def visible_world_rect(self) -> tuple[float, float, float, float]

@dataclass(frozen=True)
class AgentBox:
    agent_id: int; x: float; y: float; sprite_w: float; sprite_h: float; alive: bool

@dataclass(frozen=True)
class AgentHit:
    agent_id: int; distance: float; method: str   # "bbox" | "nearest"

class HitTester:
    def __init__(self, viewport: Viewport, *, max_distance_px: float = 60.0)
    def hit(self, agents: list[AgentBox], sx: float, sy: float) -> AgentHit | None
    # 规则：先找屏幕坐标落在精灵包围盒内的存活个体（多者取最近）；
    # 没有则取中心距 <= max_distance_px 的最近存活个体；再没有返回 None。

class SelectionModel:
    selected_id: int | None
    def select(self, agent_id: int) -> bool           # 只允许存活 id；返回是否变化
    def clear(self)
    def sync(self, alive_ids: set[int]) -> bool       # 选中者死亡/不存在 → 清空
    @property
    def has_selection(self) -> bool

class ControlState:
    # 纯状态机：playing / paused；speed ∈ SPEEDS；单步只在 paused 时生效
    SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
    def __init__(self, *, playing: bool = True, speed: float = 1.0)
    def pause(self); def resume(self); def toggle(self)
    def set_speed(self, speed: float) -> bool
    def nudge_speed(self, direction: int) -> float   # -1 慢一档 / +1 快一档
    def request_step(self) -> bool                    # 只有 paused 才接受
    def consume_step(self) -> bool                    # 取走一次待执行单步
    def scaled_dt(self, fixed_dt: float) -> float     # 实际推进的世界秒（speed * dt）
    def available_actions(self) -> dict[str, bool]    # start/pause/resume/step/...

class Layout:
    # 布局区域（像素）：top(状态栏) / world(画布) / panel(右侧详情) / bottom(控制+日志)
    def __init__(self, width: int, height: int, *, panel_visible: bool = True)
    def region_at(self, x: float, y: float) -> str    # "top"|"world"|"panel"|"bottom"
    def rects(self) -> dict[str, tuple[int, int, int, int]]
    def with_panel(self, visible: bool) -> "Layout"
    def resize(self, width: int, height: int) -> "Layout"

@dataclass
class DetailRow: label: str; value: str
@dataclass
class AgentDetail:
    agent_id: int | None
    title: str
    rows: list[DetailRow]     # 真实数据；没有的数据用 "暂无数据"
    selected: bool = False

def build_agent_detail(world, population, agent_id: int | None) -> AgentDetail
def build_stats(world, population) -> list[DetailRow]
def build_event_log(world, population, *, limit: int = 12) -> list[str]
```

`build_agent_detail` 数据来源（真实状态，不允许编造）：
- ID/存活/年龄/代数/父母：`world.registry.record(id)`、`Character` 字段；
- 能量/水分/睡眠/健康：`agent.physiology`；
- 库存/工具：`agent.inventory`（wood/stone/fiber/cord/tool/basket/capacity）；
- 当前目标/计划/进度：`population.goals.goals_of(id)`、`active_plan(id)`、`agent.current_goal_id`；
- 已掌握知识/技术：`agent.known_knowledge_ids` + `world.knowledge.get()` 的状态；
- 技能：`agent.skills`；记忆摘要：`world.memory.entries(id, limit=5)`；
- 社会关系：`population.society.affinity_between(id, other)`（取前 3）；
- 最近一次目标失败/切换原因：`GoalManager` 的 goal.status/reason（尽力读取，缺失显示"暂无数据"）。
- 人口/文明统计：`world.registry.total_created/alive_count/total_deaths`、
  `world.knowledge.verified_ids()`、`world.civilization.facilities`、`world.civilization` 指标。

### 1.2 `newworld/app_tk.py`（重写为鼠标主导）

要求：
- 布局：顶部状态栏（时间/运行状态/累计创建/存活/死亡/剩余可创建）；中央世界画布；
  右侧详情面板（可折叠）；底部控制栏（开始/暂停/继续/单步/重置/速度/统计开关/事件日志）。
- 鼠标：点击世界选中/切换/取消选中；点击左侧（或面板内）智能体列表选中；
  滚轮缩放、右键或中键拖动平移；选中个体用高亮环/描边标出（不得暂停仿真、不得改行为）。
- 重置有确认对话框（`messagebox.askyesno`）；重置后清空选择与列表；
- 单步：暂停时按 `fixed_dt`（当前世界配置）推进**恰好一步**；
- 面板数据来自 `ui_state.build_agent_detail`，每 0.25 s 刷新（不要每帧重建所有 UI）；
- 选中个体死亡后自动清理选择（`SelectionModel.sync`）；
- 世界点击与 UI 点击隔离（用 `Layout.region_at`；UI 控件使用 tkinter widget，不转发画布事件）；
- 保留键盘作为可选辅助（空格/ F1 / F2 / Q 仍可用），但**所有核心操作必须有鼠标入口**；
- 高 DPI：沿用 `enable_dpi_awareness`；窗口 `<Configure>` 重建 Layout；
- 新增运行模式 `--ui-test`：无需人工，程序化注入点击/按钮事件并断言选择、切换、取消、
  单步、暂停查看、重置清理、视口命中，输出 JSON 到指定路径并以 exit code 表示通过与否。
  CLI 由 Lead 接入（`cli.py` 里加 `--ui-test PATH` 转发给 `run_app`）。
- `--screenshot` 与 `--autoclose-ms` 保持可用（已有测试依赖）。

### 1.3 测试（ui-dev 负责）

`tests/test_ui_state.py`（>=14 用例，纯逻辑，无 tkinter）：
视口往返、缩放锚点、平移 clamp、resize；命中（bbox/最近/重叠/无命中/死人不选）；
选择切换/取消/死亡清理；控制状态机（暂停时单步、运行时不单步、速度档位）；
布局区域判定（UI 点击不判为 world）；detail 构建（真实字段 + 缺失显示"暂无数据"）；
事件日志/统计行非空；异常输入不抛错。

`tests/test_ui_app.py`（>=6 用例，**无显示时自动 skip**）：
若 `tkinter.Tk()` 可用（或环境变量 `NEWWORLD_UI_TEST=1`）则构造 app 并调用
`run_app(..., ui_test_path=tmp)` 断言 exit code 0 与 JSON 结果；否则 `skipTest`。
（Lead 会在有显示的环境实际跑 `python -m newworld --ui-test` 并记录结果。）

## 2. 高层策略工作流（rl-dev）

### 2.1 `newworld/highlevel.py`

真正的策略梯度（REINFORCE + 基线），numpy 手写前向/反向，**不是**行为克隆：

```python
GOAL_ACTIONS: tuple[str, ...] = ("survive", "gather", "craft", "experiment",
                                 "build", "maintain", "explore", "socialize")
FEATURE_NAMES: tuple[str, ...]        # 至少 16 个，必须含：能量/水分/库存/已验证技术数/
                                      # 活跃设施数/存活人口/最近目标成功失败/记忆摘要/昼夜
FEATURE_DIM: int

def extract_features(world, agent, population) -> np.ndarray  # (FEATURE_DIM,) float64, 全有限

class GoalPolicy:
    HIDDEN = 32
    def __init__(self, seed: int = 0)              # MLP FEATURE_DIM->HIDDEN->len(GOAL_ACTIONS)
    def probabilities(self, features) -> np.ndarray        # softmax
    def sample(self, features, rng) -> int
    def action_name(self, index: int) -> str
    def loss_and_grads(self, features, action_index, advantage: float) -> tuple[float, dict[str, np.ndarray]]
    def apply_grads(self, grads, lr: float) -> None
    def save(self, path) -> Path
    @classmethod
    def load(cls, path) -> "GoalPolicy"
    def to_dict(self) / @classmethod from_dict(cls, data)

class GoalPolicyTrainer:
    def __init__(self, policy, env_factory, *, seed: int = 0, lr: float = 0.05,
                 gamma: float = 0.95, baseline_decay: float = 0.9)
    def run_episode(self, *, seed: int, steps: int, dt: float = 0.05) -> dict[str, float]
        # 驱动 env（见 2.2），从 env.population.drain_policy_transitions() 取
        # (features, action_index, reward)；奖励已在 env 内按世界真实变化计算。
        # 返回 {"return","steps","extinction_time","tech_verified",...}
    def train(self, *, episodes: int, steps: int, dt: float = 0.05,
              eval_episodes: int = 2, eval_steps: int | None = None) -> dict[str, object]
        # 真实采样→奖励→策略梯度更新→周期评估；返回训练/评估曲线与最终指标
    def evaluate(self, *, seeds: list[int], steps: int, dt: float = 0.05) -> dict[str, float]
```

**奖励必须来自真实世界状态变化**（在集成层计算，由 Lead 提供，rl-dev 只消费）：
每个目标 episode 的奖励 ≈ Δ生存时间 + 5·Δ已验证技术 + 3·Δ公共设施 + 0.5·Δ资源
+ 2·Δ知识掌握 + 1·Δ合作事件 − 2·目标失败 − 5·个体死亡（具体公式见集成层实现）。
**不得**用行为克隆标签、固定进度或 UI 数值作为奖励。

### 2.2 集成钩子（Lead 实现，rl-dev 按此编写与测试）

```python
# simulation.py
def make_goal_policy_env(seed: int, *, group: str = "D", steps: int = 4000,
                         dt: float = 0.05, width: int = 640, height: int = 400
                         ) -> "GoalPolicyEnv"
class GoalPolicyEnv:                       # 训练/评估用；与实验运行器同一套世界规则
    population: Population
    world: World
    def step(self) -> None                 # 推进一帧
    def ended(self) -> bool
    def result(self) -> dict[str, float]   # 真实指标快照
# population.py
def attach_goal_policy(policy) -> None     # 之后目标选择改由 policy 决定（每目标一次）
def detach_goal_policy() -> None
def drain_policy_transitions(self) -> list[tuple[np.ndarray, int, float]]
```
`attach_goal_policy` 后，`GoalManager` 每次需要新目标时调用
`policy.sample(extract_features(...))`；若该目标类型当前不可行（前置条件不满足），
回退到启发式并给一个小负奖励。目标结束时（完成/失败/放弃）把 (features, action,
reward) 追加到 transitions。

### 2.3 `tools/train_goal_policy.py`

真实训练入口：`python tools/train_goal_policy.py --episodes 30 --steps 4000 --out assets/goal_policy.npz`，
打印训练曲线与独立评估结果（seed 不重叠），保存 checkpoint。

### 2.4 测试 `tests/test_highlevel.py`（>=10 用例）

softmax 归一/有限；采样确定性（固定 seed）；梯度形状与有限性；
一次策略梯度更新后选中动作概率上升（构造固定 advantage 验证）；
save/load 往返一致；extract_features 对真实 World 输出有限且维度正确、
不同世界状态产生不同特征；trainer 用 fake env 完成一次 train 且调用 drain；
`test_features_are_sensitive`：扰动特征（能量/技术数）后策略概率向量明显变化
（训练后；可在测试内做几次更新）。
**不得**声称"已训练出智能策略"——只验证机制正确；最终效果由 Lead 的实验报告给出。

## 3. 组合式创新工作流（innov-dev）

### 3.1 `newworld/innovation.py`

受规则约束的组合式创新（**不是**随机拼接名字）：

```python
EFFECT_KEYS = ("harvest_bonus", "capacity_bonus", "energy_regen",
               "water_regen", "craft_success_bonus")
@dataclass(frozen=True)
class RecipeAttribute:            # 每个已验证技术的基础属性（世界规则）
    knowledge_id: str
    tags: tuple[str, ...]         # "edge","binder","heat","container","structure"
    base_effects: dict[str, float]
@dataclass
class Proposal:
    proposal_id: str
    agent_id: int
    name_zh: str
    components: tuple[str, ...]   # 已验证技术 id（2 个）
    inputs: dict[str, int]
    output_tool: str
    effects: dict[str, float]
    rationale: str
@dataclass
class Innovation:
    innovation_id: str; name_zh: str; components: tuple[str, ...]
    effects: dict[str, float]; invented_by: int; invented_at: float
    successful_uses: int = 0

class InnovationLab:
    MAX_INNOVATIONS = 3
    def __init__(self, seed: int = 0)
    def proposals(self, agent, knowledge, *, limit: int = 3) -> list[Proposal]
        # 仅当 agent 掌握 >=2 条 verified 技术、且工作台/站点条件满足时提出；
        # 组合必须是已定义的可组合对（tags 相容），效果由属性确定性推导
    def attempt(self, agent, proposal, world) -> tuple[bool, str, Innovation | None]
        # 1) 检查前置（agent 掌握 components 且 verified）
        # 2) 检查材料（technology.can_afford 风格）并真实扣除
        # 3) 用 technology.register_custom_effect 注册真实效果
        # 4) 记录成功/失败与成本；失败率由难度（组合复杂度）决定
    def innovations(self) -> list[Innovation]
    def summary_zh(self) -> str
    def to_dict(self) / @classmethod from_dict(cls, data)
```

规则（最小可解释原型）：
- tag 相容对 → 效果：
  `edge+binder → harvest_bonus +1`；`container+binder → capacity_bonus +4`；
  `heat+container → energy_regen +0.15`；`heat+binder → energy_regen +0.10, water_regen +0.05`；
  `structure+container → capacity_bonus +2`。
- 材料成本 = components 的 inputs 之和的一半（向上取整）。
- 组合难度 = 0.15 + 0.1×(不同 tag 数)；失败消耗材料（真实成本）。
- 上限 3 项/世界；超过后 `proposals` 返回空。

### 3.2 集成钩子（Lead 实现）

`technology.py` 新增：
```python
def register_custom_effect(self, tool_name: str, effects: dict[str, float]) -> None
# harvest 时按 agent.inventory.tool 查表：harvest_bonus 加到产量
# capacity 由 Inventory.capacity 读取 custom capacity_bonus
# facility/self effects 在 apply_facility_effects 中应用 energy_regen/water_regen
```
`population.py` 每 ~5 世界秒对每个存活且 enable_knowledge 的个体尝试一次创新
（有候选才尝试），把成功/失败写 memory + telemetry 事件 `innovation`。

### 3.3 测试 `tests/test_innovation.py`（>=10 用例）

仅 verified 可组合；tag 不相容拒绝；材料不足返回 False 且不扣材料；
成功时真实扣材料、注册效果、写 Innovation；效果经 technology 查询可见；
失败率随组合难度、失败消耗材料；上限 3；to_dict/from_dict 往返；
空知识库 no-op；summary 中文非空。

## 4. Lead 的集成与实验（不在队友写权限内）

- `config.py`：`enable_highlevel_policy`、`enable_innovation` 开关。
- `population.py`：决定来源计数 `decision_counts`、goal policy 挂载与 transitions、
  创新尝试、`attach_goal_policy`、`drain_policy_transitions`。
- `simulation.py`：`RunSpec.goal_policy` / `policy_mode`（"heuristic"|"neural"|"random"）、
  `make_goal_policy_env`、奖励计算。
- `cli.py`：`--ui-test PATH`、`--society` 友好入口。
- 实验：至少 10 个 seed × 4 配置（baseline / +planning / +neural goal policy / +innovation），
  真实运行并写 `docs/EXPERIMENT_REPORT.md`。
- 文档：`docs/PROJECT_AUDIT.md`、`docs/IMPLEMENTATION_REPORT.md`、`docs/USER_GUIDE.md`（与 ui-dev 协作）。
