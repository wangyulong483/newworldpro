# V2 升级契约（Lead 冻结；并行实现以此为准）

> 本轮方向（来自用户评审）：
> 1. **打通感知**：视网膜特征真正进入决策观测（不再"装饰性真实"）；
> 2. **低层从 BC 升级为 RL**：主路径加真实奖励，打破教师天花板；
> 3. **放开 8 人上限做对照实验**：观察人口动态、崩溃后恢复；
> 4. **无限世界**：不再受屏幕尺寸限制。
>
> 工作流：**perception-dev**（视觉特征）、**rl-lowlevel-dev**（低层 RL）、
> **infinite-dev**（无限场景）、**Lead**（集成/配置/奖励/实验/文档）。

## 0. 通用约束

- Python 3.14，仅标准库 + numpy + pillow（GUI tkinter）；**不得新增运行时依赖**。
  `pymupdf` 仅 asset 工具使用，不在本轮范围。
- 工作区当前有用户的中文注释/文档翻译改动（`git status` 大量 M）。**不要回退、不要
  格式化整个文件、不要重写文件头**；只做目标函数的增量修改。
- 测试：`cd newworldpro; $env:TEMP=(Resolve-Path .tmp).Path; python -m unittest tests.test_xxx -v`
  需要临时目录时加 `tests/_env_compat.py` 垫片（见任何现有测试文件头）。
- 一切新增能力必须有：真实实现、单元测试、可复现命令、如实结果（无编造数字）。

---

## 1. perception-dev：视网膜 → 决策特征

### 1.1 文件权限

只允许新增/修改：
- `newworld/visual_features.py`（新）
- `tests/test_visual_features.py`（新）
- `tools/bench_vision_features.py`（新，成本/方差基准；可删）

### 1.2 接口（冻结）

```python
VISUAL_FEATURE_DIM = 16

@dataclass
class VisualFeatureState:
    vector: np.ndarray          # (16,) float64, 全有限, 归一化到约 [-1,1]
    source: str                 # "retina" | "cache" | "none"
    cost_ms: float = 0.0

class VisualFeatureExtractor:
    """从真实眼睛/视网膜/注视系统提取紧凑视觉特征（不是几何启发式）。"""
    def __init__(self, seed: int = 0, *,
                 retina_parameters=None,   # RetinaParameters；默认低分辨率档
                 eye_parameters=None) -> None
    def observe(self, world, character, dt: float) -> VisualFeatureState
        # 调用 VisualSystem.perceive(...)（或等价真实光学/视网膜路径），
        # 从 PerceptionState/RetinaObservation 计算 16 维；异常/无场景时返回零向量+source="none"
    def last_perception(self): ...      # 最近一次 PerceptionState 或 None（供调试）
```

16 维组成（必须来自真实感知数据，写进 docstring）：
- 8 维：视网膜神经节细胞（ganglion）响应分箱降采样（把 `retina.feature_vector()` 的
  n_bins 箱按均值聚成 8 组）；
- 2 维：平均亮度、中央凹亮度（log 压缩）；
- 1 维：光适应状态；1 维：注视偏移角/最大偏移；
- 1 维：离焦模糊 σ；1 维：扫视抑制强度（若 PerceptionState 无该字段，用可得字段替代并注明）；
- 1 维：声压级 SPL；1 维：是否听到（0/1）。
`VisualFeatureExtractor` 必须**可注入低分辨率 RetinaParameters**（默认
`n_cones=96, n_rods=0, n_bins=16`，docstring 说明这是"学习用低分辨率档"，
不是运行 GUI 的高保真档）。

### 1.3 性能要求（必须实测并报告）

- 用 `tools/bench_vision_features.py` 测：640×400 世界、100 次 observe 的平均耗时；
- 目标：**≤ 2.5 ms/次**（低分辨率档）。若达不到，继续降低 retina 分辨率/箱数并如实报告；
- 必须证明特征**有真实方差**：同一世界不同位置/朝向/时间下 16 维特征不全相同
  （测试断言 std > 0），且改变亮/物体布局会改变特征。

### 1.4 测试（≥8 用例）

维度/有限性/归一化范围；真实世界下非零方差；不同朝向/位置特征变化；
异常输入降级为零向量不抛错；`source` 字段正确；低分辨率参数生效；
benchmark 可运行（不强制断言耗时上限，避免 CI 抖动，但报告实测）。

### 1.5 集成钩子（Lead 负责，不在你写权限）

- Lead 会把 `FEATURE_DIM` 从 40 扩到 **56**（+16 视觉），并让
  `neural.features()` 读取 `character.visual_features`（Population 每 `perception_stride`
  帧写入一次）；旧 24 维、当前 40 维 checkpoint 都零初始化迁移。
- 训练数据采集会带视觉特征（`training.collect_dataset`），并断言非零方差。
- 对照实验：`visual_features on/off`（同一世界、同 seed）。

---

## 2. rl-lowlevel-dev：低层动作策略的真实 RL

### 2.1 文件权限

只允许新增/修改：
- `newworld/lowlevel_rl.py`（新）
- `tools/train_lowlevel_rl.py`（新）
- `tests/test_lowlevel_rl.py`（新）

### 2.2 接口（冻结）

```python
@dataclass
class ActionSample:
    mode: int          # 0..len(MODES)-1
    move: int          # 0..2  (对应 -1/0/+1)
    jump: bool
    speak: bool
    logprob: float

class LowLevelPolicy:
    """把现有 NeuralPolicy（MLP 40/56→64→32→58）变成可采样的随机策略。"""
    def __init__(self, neural_policy, seed: int = 0) -> None
    @property
    def neural_policy(self)
    def sample(self, features: np.ndarray, rng=None) -> ActionSample
    def deterministic(self, features: np.ndarray) -> ActionSample   # argmax（评估用）
    def logprob_and_grads(self, features, action: ActionSample,
                          advantage: float) -> tuple[float, dict[str, np.ndarray]]
        # 手写反向：mode/move 为 softmax 交叉熵，jump/speak 为 BCE，
        # 语言头不参与 RL（argmax 保留）；返回对各权重的梯度
    def apply_grads(self, grads: dict[str, np.ndarray], lr: float) -> None
    def save(self, path) -> Path
    @classmethod
    def load(cls, path, seed: int = 0) -> "LowLevelPolicy"

class LowLevelTrainer:
    def __init__(self, policy: LowLevelPolicy, env_factory, *,
                 seed: int = 0, lr: float = 0.01, gamma: float = 0.95,
                 baseline_decay: float = 0.9) -> None
    def run_episode(self, *, seed: int, steps: int, dt: float = 0.05) -> dict[str, float]
    def train(self, *, episodes: int, steps: int, dt: float = 0.05,
              eval_seeds: list[int] | None = None, eval_steps: int | None = None
              ) -> dict[str, object]
    def evaluate(self, *, seeds: list[int], steps: int, dt: float = 0.05) -> dict[str, float]
```

- **真实策略梯度**：采样动作 → 累积 env 提供的**真实奖励** → 折扣回报（带指数滑动基线）
  → 手写反向 → 参数更新 → checkpoint；不得使用 BC 标签、固定进度或 UI 数字。
- 训练/评估 env 由 Lead 提供（`simulation.make_lowlevel_rl_env(seed, ...)`，
  与文明实验共享世界规则；训练 seed 与评估 seed 不重叠）。
- trainer 构造 env 后调用 `env.population.attach_lowlevel_policy(self.policy)`；
  每步从 `env.population.drain_lowlevel_transitions()` 取
  `(agent_id: int, features: np.ndarray, action: ActionSample, reward: float)`；
  按 agent 分别累计折扣回报后更新。
- 语言头不参与 RL（保持 argmax），在 docstring 写清楚"输出被 RL 训练的是
  mode/move/jump/speak 四个决策头"。

### 2.3 `tools/train_lowlevel_rl.py`

真实训练 CLI：`--episodes/--steps/--seeds/--eval-seeds/--out/--from-checkpoint/--json`。
钩子缺失时 exit 2 并明确报错，不打印假曲线。默认从
`assets/neural_policy.npz`（BC）初始化，输出 `assets/lowlevel_rl.npz`。

### 2.4 测试（≥10 用例）

softmax/BCE 对数概率正确；采样固定 seed 确定；`logprob_and_grads` 与有限差分一致
（至少 W1/W3 若干元素）；一次正 advantage 更新后该动作概率上升；
save/load 往返；确定性模式=argmax；fake env 的 trainer 全流程（drain/更新/评估）；
真实 `NeuralPolicy` 上采样动作合法；语言头 argmax 不变。

### 2.5 集成钩子（Lead 负责）

`CivilizationConfig.low_level_mode` 增加 `"rl"`；`Population.attach_lowlevel_policy` /
`drain_lowlevel_transitions` / 真实奖励计算；`simulation.make_lowlevel_rl_env`；
实验对照 `bc vs rl vs random`（10 seed，独立评估）。

---

## 3. infinite-dev：无限世界（确定性分块 + 局部窗口）

### 3.1 文件权限

只允许新增/修改：
- `newworld/infinite_scene.py`（新）
- `tests/test_infinite_scene.py`（新）

### 3.2 设计（冻结语义）

无限世界 = **横向无界、纵向固定**（侧视世界），屏幕尺寸只是"局部窗口"。
`InfiniteScene` 必须提供与 `PixelScene` **兼容的窗口化 API**，这样 `World` /
`Ecology` / `render` / `Technology` / `optics` 可以基本不改地工作：

```python
class InfiniteScene:
    def __init__(self, config, seed: int = 0, *, window_atoms: int | None = None)
    # 窗口属性（与 PixelScene 同名同义，坐标均为"窗口局部坐标"）
    aw: int; ah: int; surface: int; ground_row: int
    pond: tuple[int, int]; pond_surface: int
    objects: list; berries: list[tuple[int, int]]
    grid: np.ndarray  # (ah, aw) uint8
    ecology: Ecology
    # 全局<->局部
    window_start_atom: int          # 窗口左边缘的全局原子 x
    window_start_px: float          # = window_start_atom * ATOM_SIZE
    def global_atom_x(self, local_atom_x: int) -> int
    def local_atom_x(self, global_atom_x: int) -> int
    # 平移窗口（delta_atoms>0 表示窗口向右移动 delta 个原子；内容相对屏幕左移）
    def recenter(self, delta_atoms: int) -> None
    # 与 PixelScene 兼容的其余方法：
    set_atom/clear_atom/atom_material/refresh_caches/add_berry/pick_berry/water_near/
    berry_at/solid_at/material_at/is_blocked/surface_screen_y/refresh_lighting/
    update_ecology/trample/render_atoms/optics_scene/material_composition_text/
    material_name_zh
```

实现要求：
1. **确定性**：全局原子 (gx, y) 的材料由 `seed + gx + y` 的函数决定；两个相同 seed 的
   scene 在重叠区域材料必须完全一致；不同 seed 不同。
2. **对象/资源**：岩石/树/浆果丛按确定性格点放置（每 chunk 用 hash 决定是否有、
   位置与大小）；池塘周期性出现（保证任意窗口内经常有水源）。
3. **修改持久化**：`clear_atom`/`set_atom`/采果等修改记录在稀疏字典里（按全局坐标），
   `recenter` 后仍保留；切走再切回内容一致。
4. **窗口平移**：`recenter(delta)` 重建 grid/objects/berries/pond，并把
   `window_start_atom += delta`；ecology 的 moisture 数组随窗口平移（可近似重采样，
   docstring 说明）；性能目标：单次 recenter ≤ 30 ms（实测报告）。
5. `render_atoms(day, hour)` 输出窗口大小的 RGB（与 PixelScene 同形）；
   `optics_scene()` 返回窗口内物体的 Scene（坐标局部）。
6. 不修改 `World` / `pixelworld` / `render`；只通过本模块准备集成。

### 3.3 集成（Lead 负责）

- `WorldConfig.infinite: bool = False` + `window_atoms`；`World` 在 infinite 时用
  `InfiniteScene`；
- `World.step` 在代理接近窗口边缘时调用 `recenter`，并把**所有动态 x 坐标**
  （agents/facilities/community/pending/sound）平移 `-delta_px`，保持局部坐标体系；
- GUI：默认仍是有限世界；`--infinite` 开启无限跟随窗口；实验侧
  `RunSpec.infinite=True`。

### 3.4 测试（≥10 用例）

确定性（同 seed 重叠区域一致）；不同 seed 不同；窗口平移后全局点材料一致；
修改（清原子/采果）跨 recenter 持久；对象/berry 在重叠区一致；
池塘周期性存在；`aw/ah/surface` 等属性正确；`render_atoms` 尺寸正确；
`update_ecology` 可运行；非法 delta/极小窗口降级不崩。

---

## 4. Lead 集成契约（并行参考，非队友写权限）

### 4.1 感知进入观测

- `neural.FEATURE_DIM 40 → 56`（+16 视觉），版本 4；
  迁移链：24→(零初始化 16 上下文)→40→(零初始化 16 视觉)→56。
- `Population`：`perception_stride`（默认 4）控制 `Perceiver.perceive`，
  每帧从 `mind.perception` 计算 16 维写入 `agent.visual_features`；
  未启用感知的实验使用 `VisualFeatureExtractor`（每 stride 帧）；
- `training.collect_dataset`：带视觉特征采集，断言非零方差；
- 对照实验：`visual on/off`（同 seed、同预算）。

### 4.2 低层 RL 与奖励

- 新增 `newworld/rewards.py`：`LowLevelReward`，**完全基于真实世界状态变化**：
  `+1.0×Δ已验证技术 +0.5×Δ公共设施 +0.2×Δ资源单位 +0.2×Δ掌握知识
   +0.1×Δ合作转移 −0.001×dt −2.0×死亡`（clamp），惩罚 idle/time；
  必须提供反刷分测试（原地站立 ≤0；粮仓旁挂机 ≤0；重复无效动作不累积正奖励）。
- `Population`：`low_level_mode="rl"` 时用 `attach_lowlevel_policy` 采样；
  以 agent 为单位记录 `(agent_id, features, action, reward)` 到
  `self.lowlevel_transitions`；`drain_lowlevel_transitions()`。
- 实验对照：`bc vs rl vs random`，独立 seed，报告真实差异。

### 4.3 人口上限对照

- `max_agents_created` 已可配置；新增实验配置 `cap8 / cap32 / cap64`（配合
  infinite 世界与更大食物供给），报告人口时间序列、崩溃后恢复、灭绝率；
- 不得通过隐藏重生绕过上限：上限只由配置决定，指标同时记录累计创建/存活/死亡。

### 4.4 无限世界集成

- `WorldConfig.infinite/window_atoms`；`RunSpec.infinite`；`--infinite`（GUI 与
  simulation CLI）；现有 418 个测试必须继续通过（finite 为默认）。
- GUI：infinite 模式下窗口跟随人口重心（屏幕=窗口），缩放/平移限制在窗口内，
  选中/详情/控制按钮行为不变；USER_GUIDE 注明。

---

## 5. 报告与验证要求（所有队友适用）

- 报告必须区分：**已完成且通过测试 / 已实现未充分验证 / 只有设计 / 受限制未完成**；
- 不得声称"解决了感知/RL/无限世界"却没有真实运行证据；
- 每个数字都要能从命令复现；测试必须真实运行并贴出 `Ran N tests / OK / exit`。
