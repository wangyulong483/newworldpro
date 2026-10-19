# 项目审计（本轮升级 P0）

- 审计对象：`newworldpro`（newworld v0.9.0，Python 3.14，numpy + pillow + tkinter）
- 审计方法：只读源码 + 实际运行与插桩测量（不是根据 README 推测）
- 基线：`git 7ff0b4f`（上一轮文明改造完成态），`python -m unittest discover -s tests`
  → **Ran 297 tests ... OK**
- 本轮任务：① 提高文明演化真实性（决策链路 + 知识传播 + 组合式创新）
  ② 键盘交互改为鼠标点击主导

---

## 1. 真实架构

### 1.1 入口与主循环

| 入口 | 文件 | 说明 |
| --- | --- | --- |
| `python -m newworld` | `__main__.py` → `cli.main` → `app_tk.run_app` | 交互式（默认全屏） |
| `python -m newworld --headless` | `cli.run_headless` | 单个体仿真（`Mind`），不是多智能体社会 |
| `python -m newworld.simulation` | `simulation.main` | 多智能体文明实验（headless） |
| `experiments/run_ablation.py` | 消融实验 | A/B/C/D × 多 seed |
| `tools/train_neural_policy.py` | 行为克隆训练 CLI | 与此前的 BC 流程一致 |

交互主循环（`app_tk.run_app`）：

```
tk.Tk() → Canvas(fullscreen) → Population(world)
  root.after(16, tick)                      # ≈60 Hz 渲染循环
    tick:
      dt = min(now - last, 0.05)            # 墙钟 dt，上限 50 ms
      if not paused: population.step(dt)    # ← 仿真时钟 = GUI 帧时钟
      sync_visuals / refresh_agent
      每 0.8s refresh_background（render_world）
      每 0.3s refresh_panel（PIL 调试面板 → PhotoImage）
```

仿真时钟就是 GUI 帧间隔（`dt` 来自墙钟，不是固定步长）；`world.config.fixed_dt=1/60` 只用于
headless/CLI 路径。**暂停 = 不调用 population.step**，渲染与面板仍刷新。

### 1.2 智能体生命周期与资源

- 创建：唯一受约束路径 `Population.create_agent` → `World.create_agent` →
  `AgentRegistry.register`；`civ_config.max_agents_created = 8` 约束**累计创建**，
  第 9 次返回 `None` 并写入 `world.creation_denied`。
- 死亡：`Society.update` → `Population._on_death`（先归档 AgentRecord）→
  `agent.alive=False` → `world.remove_agent` → `Population.sync` 销毁脑对象。
- 资源：`Technology.harvest/execute_recipe/simulate_recipe` 是唯一权威；
  `Inventory(wood, stone, fiber, cord, tool, basket)`；公共设施 `CivilizationState.facilities`
  按半径给能量/水/休息效果。
- 当前实测（D 组 seed0，24000 步上限）：累计创建 8、存活峰值 8、全部老年死亡、
  验证技术 4–5 项、设施 4 个、灭绝时间 1030–1033s。

### 1.3 神经网络

- 模型：MLP `40 → 64 → 32 → 58`（tanh，numpy，手写反向传播）；5594+16×64 参数。
- 输入 40 维 = 原 24 维感知/生理 + 16 维文明上下文（8 记忆摘要 + 4 目标/计划 + 4 文明）。
- 输出 58 = 5 行为模式 + 3 移动 + 跳/说 + 3×16 语言头。
- 训练：`training.py` 的**行为克隆**——用 `behavior.InnerLife` 教师采集
  (状态 → 模式/移动/跳/说话/词) 数据集，监督学习；`pretrain`/`fine_tune` 保存/继续
  checkpoint；GUI 后台每 30s 训练一轮、F2 从零训练。
- **不是强化学习**：没有奖励、没有策略梯度、没有价值函数、没有环境采样循环。

### 1.4 三层职责（现状）

| 层 | 模块 | 实际职责 |
| --- | --- | --- |
| 高层「想做什么」 | `goals.GoalManager` | 需求/机会驱动的规则优先级；157 个目标/200s |
| 中层「怎么做」 | `goals.Planner` | 规则 + 代价搜索 → 结构化步骤（gather/craft/experiment/build…） |
| 底层「动作」 | 执行器 / `social.SocialLife` / `neural.NeuralInnerLife` | 规则执行、技能教师、BC 网络 |

此外 `population._infrastructure_directive` 在低能量时优先接管（去设施休息），
`population._cooperation` 做食物分享。

### 1.5 目标/记忆/社会/知识

- 目标：`Goal`/`Plan`/`PlanStep`，完成/失败/放弃与重规划均有记录；无进展 45s → replan。
- 记忆：`MemoryStore` 情景/程序记忆 + 8 维摘要；死亡归档。
- 社会：affinity 图、教学（170px 内按概率传播已验证知识）、出生继承。
- 知识：`KnowledgeStore`，固定目录 7 条技术（有限科技树），
  `hypothesis→verified→invalidated`，只有 verified 可 craft/build。
- 组合式创新：**不存在**（只有固定目录）。本轮新增原型。

### 1.6 奖励 / 训练数据 / 训练-评估

- 主路径**无奖励函数**；`env.py` 的 RL 包装器（8 维/3 动作/4 奖励）只被测试与示例使用。
- 训练数据来源：教师 `InnerLife` 在**裸 World**（`make_config(800, 500)`）中运行
  `collect_dataset`（7 个起始时刻 × steps_per_hour）。
- 评估：`training.evaluate_behaviour` 用另一个裸 World（seed 参数）；
  没有训练/评估 seed 分离协议，也没有"未见过的初始资源布局"测试。

### 1.7 交互与渲染（本轮要改的部分）

- 键盘绑定共 7 个（`app_tk.py:244-251`）：`<KeyPress>` 隐藏提示、`Escape`/`q`/`Q` 退出、
  `space` 暂停、`F1` 调试层、`F2` 从零训练、`<Button-1>` 仅用于 `focus_force()`。
- **没有任何鼠标交互**（无选中、无面板、无按钮、无列表、无缩放平移）。
- 渲染：背景 `render_world`（1280×800 实测 4.1 ms），个体 `character_layer` 每帧，
  调试面板每 0.3s 重建 PIL 图像；画布与世界 1:1，无视口变换。
- 性能：仿真 5095 步/s（关感知）/ 74 步/s（开感知，13.5 ms/步）。
  **注意：感知占 95% 时间且不进入决策输入**（`neural.features` 忽略 perception）。

### 1.8 测试与日志

- 297 个测试（111 原始 + 186 上轮新增），约 104–183s 跑完。
- 日志：`Telemetry`（19 指标 + JSONL 事件）；`experiments/results`、`experiments/records`。

---

## 2. 已实现 / 仅接口 / 未生效（实测判定）

| 能力 | 状态 | 证据 |
| --- | --- | --- |
| 累计创建 ≤8 硬约束 | ✅ 真实生效 | `registry.total_created`；跨组跨 seed = 8；第 9 次返回 None |
| 死亡档案 / 传承 / 教学 | ✅ 真实生效 | 测试 + D 组档案（知识随死亡归档、子代继承） |
| 目标/计划/重规划 | ✅ 真实生效 | 200s 内 157 目标、148 完成、replan 4 次 |
| 知识状态机 + 技术效果 | ✅ 真实生效 | 采集×2、粮仓供能等，测试可测 |
| 公共设施 + 合作 | ✅ 真实生效 | 合作指数 D=0.867、C=0 |
| **神经网络参与决策** | ⚠️ **几乎不生效** | 见 §3.1：逐帧决策仅 0.3% |
| **BC 的 16 维文明上下文特征** | ⚠️ **训练中未生效** | 见 §3.2：训练数据 16 维全零方差 |
| 组合式创新 | ❌ 未实现 | 固定 7 条目录 |
| 真实强化学习 | ❌ 未实现 | 无奖励/无策略梯度 |
| 鼠标交互 | ❌ 未实现 | 仅键盘 + 单击聚焦 |
| 训练/评估环境独立 | ⚠️ 部分 | 各自建 World，但无 seed 划分/未见布局评估 |
| `env.py` RL 包装器 | ⚠️ 仅测试/示例 | 主程序未使用 |

---

## 3. 主要问题（含代码位置与实测数据）

### 3.1 神经网络在逐帧决策中只占 0.3%（最高优先级）

插桩（D 组、640×400、4000 步、200 世界秒，8 个个体）：

```
decision sources: infrastructure=427, goal=20831, teacher_decide=186,
                  teacher_override=120, neural_used=66
总 agent-步 ≈ 21424 → goal 97.2%，infra 2.0%，teacher 0.6%，neural 0.3%
```

原因链：
1. `population._decide` 顺序为 infra → `goals.update` → social → neural；
2. `GoalManager.update` 几乎每帧都返回动作（默认 `explore` 目标，每 8 世界秒一个），
   从不返回 `None`；
3. 于是 `social`/`neural` 只剩饥饿/无目标等极少帧。

这直接违反本轮任务"神经网络必须真正参与决策"的要求。另外规划器的成果
（gather/craft/build）会被误读成"神经网络学会了规划"，必须在报告里区分。

### 3.2 BC 训练数据里 16 维新增特征全部是常数 0（死特征）

实测：
- 真实运行观测（13450 帧）：16 维上下文里 **14 维有非零方差**（2 维恒 0）；
- `training.collect_dataset` 的 `x`（60 帧样本）：**16/16 维标准差 = 0.0**。

原因：`collect_dataset` 只用裸 `World` + `InnerLife` 教师，没有 `Population`，
因此 `world.memory/goal_manager/civilization` 都不存在，`_context_features` 全部返回 0。
后果：即使 24→40 迁移正确，这 16 维在训练中梯度恒为 0，**永远不会被学习**。
必须修复数据采集（否则"新增特征"只是摆设）。

### 3.3 目标层过度活跃 / 目标抖动

- 200s 内 157 个目标（约 1.3 个/秒，8 个个体），其中 `goals_abandoned=1`、
  `replans=4`；`explore` 目标占比最高。
- 目标切换频率高 → 个体频繁改变方向，`repeat_run_max` 161 步；
- 规划器是规则+代价搜索，**没有学习**；把它当成"神经网络能力"会是虚假声明。

### 3.4 没有奖励信号，也没有真实 RL 循环

- 主路径每步标量奖励 = 0；`Telemetry` 只记录指标。
- 训练用的标签是教师动作（BC），不是环境奖励。
- 评估与训练共用同一世界生成器/尺寸，没有 held-out seed 与"未见初始资源布局"评估。

### 3.5 交互：完全键盘驱动

- 7 个键绑定；`Button-1` 只做 `focus_force`；
- 无选中、无详情、无控制按钮、无列表、无视口变换、无事件日志面板；
- 用户必须记住空格/F1/F2/Q；不满足本轮"鼠标点击主导"要求。

### 3.6 点击 UI 绕过世界规则的风险（设计约束）

现有代码没有任何 UI→世界写路径，风险低；但新增详情/控制/干预功能时必须保证：
- 选择/查看只读；
- 开始/暂停/单步/重置只影响仿真时钟与运行实例，不直接改世界状态；
- 任何"人工干预"必须单独标识且默认关闭（本轮不做干预功能，避免新增绕过路径）。

### 3.7 训练/评估效率与性能

- 仿真（关感知）5095 步/s；开感知 74 步/s；渲染 1280×800 约 4 ms/帧。
- 因此：实验默认关闭感知；GUI 保留感知但面板/背景按 0.3–0.8s 节流（现有实现）。

---

## 4. 本轮修改方案（P1–P4）

### P1 鼠标主导交互（新增/重写）

| 文件 | 内容 |
| --- | --- |
| `newworld/ui_state.py`（新） | 纯逻辑：`Viewport`（缩放/平移/坐标变换）、`HitTester`（包围盒+最近）、`SelectionModel`、`ControlState`（开始/暂停/单步/速度）、`Layout`（区域隔离）、`build_agent_detail/stats/event_log`（真实数据） |
| `newworld/app_tk.py`（重写） | 顶部状态栏 + 中央画布 + 右侧详情 + 底部控制/日志；点选/切换/取消、列表同步、选中高亮、滚轮缩放/拖动平移、重置确认、单步、`--ui-test` |
| `tests/test_ui_state.py`、`tests/test_ui_app.py` | 纯逻辑 ≥14 用例 + 可跳过 GUI 冒烟 ≥6 用例 |
| `docs/USER_GUIDE.md` | 鼠标操作说明 |
| `cli.py` | `--ui-test PATH` 转发 |

### P2 决策链路修复

1. **高层目标选择改为真实策略梯度**（新增 `newworld/highlevel.py` +
   `tools/train_goal_policy.py`）：REINFORCE + 基线，输入含记忆/目标/文明上下文，
   奖励来自真实世界变化（生存/技术/设施/资源/失败/死亡），独立 seed 评估。
   集成：`population.attach_goal_policy` 后在"需要新目标"时由策略采样。
2. **修复 BC 数据死特征**：`training.collect_dataset` 改为在 `Population` 环境内采集，
   使 8 记忆 + 4 目标 + 4 文明特征有真实方差；新增测试断言非零方差。
3. **决策来源计量**：`Population.decision_counts` 记录
   infrastructure/goal/teacher/neural 各自帧数，写入 telemetry 与实验报告，
   防止把规划器能力误报为神经网络能力。
4. **消融验证**：`policy_mode ∈ {heuristic, neural, random}` 与
   `--no-neural/--random-neural` 对比；关闭规划器对比；未见初始布局评估。

### P3 文明机制升级

1. **知识传播补全**：已有教学/继承；本轮增加"公共知识设施"效果（小屋/粮仓附近提高
   教学概率）与传播事件指标。
2. **组合式创新原型**（新增 `newworld/innovation.py`）：属性（tags）相容规则驱动的
   两技术组合 → 规则验证 → 真实材料成本 → 注册真实效果（采集/负重/恢复）。
   上限 3 项/世界；失败消耗材料；可序列化追踪。

### P4 实验与回归

- 回归：297 原有测试 + 新增测试；鼠标 10 项交互由 `--ui-test` 实际运行断言。
- 实验：≥10 个独立 seed × 4 配置（baseline / +planning / +neural goal policy /
  +innovation），报告均值/标准差/失败率；
- 硬约束：跨配置与 seed 断言 `total_agents_created ≤ 8`；
- 文档：`IMPLEMENTATION_REPORT.md`、`EXPERIMENT_REPORT.md`、`USER_GUIDE.md`。

---

## 5. 风险与回归测试方案

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 高层策略训练成本 | 不能跑大预算 | 小世界 640×400、关感知、episodes 有限；报告实际预算，不夸大 |
| 策略接入后目标抖动/退化 | 文明指标下降 | 不可行目标回退启发式 + 小负奖励；保留 heuristic 配置作对照 |
| 修复 BC 数据后旧 checkpoint 失配 | 加载失败 | 只改数据采集，不改网络维度；旧 checkpoint 仍可加载（零初始化） |
| UI 重写破坏 `--screenshot`/测试 | 回归 | 保留 `run_app` 原参数；`test_world` 的 CLI/预览测试必须继续通过 |
| UI 每帧重建面板导致卡顿 | 性能 | 面板 0.25s 节流、背景 0.8s、详情 diff 更新 |
| UI 点击误改世界 | 违反约束 | 只读选择；控制面板仅改时钟/运行实例；`Layout.region_at` 隔离 |
| 鼠标在无显示环境不可验证 | 无法证明 | 逻辑层 100% 可测 + `--ui-test` 程序化事件；实际环境跑截图/自检 |
| 实验过长 | 交付延误 | 降低单次步数/世界尺寸，报告实际预算与种子数 |

**回归基线**：改动前后都必须满足
`python -m unittest discover -s tests` 全绿（当前 297 个），
且 `python -m newworld --windowed --size 1280x800 --screenshot ... --autoclose-ms ...` 可运行。
