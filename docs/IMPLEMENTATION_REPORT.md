# 本轮升级实施报告（决策链路 + 鼠标交互 + 组合式创新）

- 项目：`newworldpro`（newworld v0.9.0，Python 3.14，numpy + pillow + tkinter）
- 升级前基线：`git 7ff0b4f`，`python -m unittest discover -s tests` → 297 tests OK
- 审计：`docs/PROJECT_AUDIT.md`（含实测插桩数据）
- 实验：`docs/EXPERIMENT_REPORT.md`（数据来自 `experiments/results/upgrade_final/`）
- 用户操作：`docs/USER_GUIDE.md`

---

## 1. 一句话结论

1. **鼠标交互已实际交付**：点击选中/切换/取消、详情面板、控制面板、智能体列表、
   事件日志、缩放平移、单步/暂停/重置；`--ui-test` 程序化注入真实事件，**13/13 通过**，
   并在本机窗口化运行截图验证。
2. **决策链路已修复**：神经网络从审计时的 **0.3% 逐帧决策** 提升到 **约 66%**
   （另一部分为基础设施反射 28%、规划语义动作 5%、教师 0.6%），
   并新增**真实 REINFORCE 高层目标策略**（有采样/真实奖励/手写反向传播/参数更新/
   checkpoint/独立 seed 评估）；修复了 BC 训练数据里 16 维上下文全零的"死特征"。
3. **文明机制升级**：新增有限规模组合式创新（属性规则 + 真实材料成本 + 真实世界效果，
   自然运行可触发）；知识传播/继承/档案保留沿用上一轮并纳入新实验。
4. **诚实结论**：`learned` 高层策略在 10 个未见 seed 上**改变了行为**
   （高层目标决策更多、不可行请求更少，10/10 显著），但**文明级指标
   （技术/设施/存活）没有观察到显著差异**（相对 heuristic 点估计略低，CI 跨 0）。

---

## 2. P1：鼠标主导交互（实际交付）

### 2.1 新增/重写文件

| 文件 | 内容 |
| --- | --- |
| `newworld/ui_state.py`（新） | 纯逻辑：`Viewport`（屏幕/世界坐标、缩放锚点、平移、clamp、resize）、`HitTester`（包围盒优先→最近邻，死人不选）、`SelectionModel`（选择/切换/取消/死亡清理）、`ControlState`（开始/暂停/继续/单步/速度档位）、`Layout`（top/world/panel/bottom 区域隔离）、`build_agent_detail/build_stats/build_event_log`（全部读真实世界状态） |
| `newworld/app_tk.py`（重写） | 顶部状态栏（时间/状态/累计创建/存活/死亡/剩余可创建）；中央可缩放平移画布；右侧可折叠面板（智能体列表 + 详情 + 统计）；底部控制栏（开始/暂停/继续/单步/重置/速度/面板开关/训练）+ 事件日志；选中黄色高亮环；`messagebox` 重置确认；`<Configure>` 重建布局；高 DPI；保留键盘辅助；新增 `ui_test_path` |
| `tests/test_ui_state.py` | 34 个纯逻辑用例 |
| `tests/test_ui_app.py` | 7 个 GUI 集成用例（无显示自动 skip） |
| `docs/USER_GUIDE.md` | 中文操作指南（四区域、鼠标/键盘表、数据来源、`--ui-test`、限制） |

### 2.2 鼠标交互的真实验证（Lead 独立复跑）

```
python -m newworld --ui-test .tmp\lead_ui_test.json --windowed --size 1024x720
→ EXIT=0，13/13 PASS：
  layout_regions_and_click_isolation / viewport_transform_and_zoom_anchor /
  world_click_selects_agent / world_click_switches_selection / world_click_toggles_off /
  world_click_empty_clears_selection / wheel_zoom_and_drag_pan_hit_consistency /
  pause_freezes_time_but_panel_visible / single_step_advances_exactly_one_frame /
  list_click_syncs_selection / dead_agent_clears_selection /
  detail_and_stats_use_real_data / reset_clears_selection_and_list

python -m newworld --windowed --size 1280x800 --autoclose-ms 2500 --screenshot .tmp\ui_shot3.png
→ 实际启动成功，打印四区域布局，截图已保存（1280x800），实测 19–21 fps
```

关键行为：
- 点击个体 = **只选中/查看**，不暂停仿真、不改变其目标或动作（契约 §6.2E）；
- UI 点击由 `Layout.region_at` 隔离，世界点击与按钮/列表事件互不误触发；
- 坐标经 `Viewport` 变换，缩放/平移后仍能正确命中；
- 单步仅在暂停时生效，且严格推进 `world.config.fixed_dt` 一步
  （`--ui-test` 断言两次单步各 0.016667s）；
- 暂停时详情/统计仍可查看；选中个体死亡后自动清理；
- 重置（确认对话框）构造全新世界与人口，清空选择/列表；**不隐藏重生机制**：
  累计创建计数属于"一次仿真运行"，重置等于开始新的运行（USER_GUIDE 已说明）。

### 2.3 键盘处理

保留为可选辅助：空格=暂停/继续、F1=统计面板、F2=手动从零训练一次、Q/Esc=退出。
所有核心操作都有鼠标入口；旧版"启动后自动后台训练线程"被移除（避免与仿真/资产竞争），
训练改为按钮/F2 手动触发。

---

## 3. P2：决策链路修复

### 3.1 审计发现（实测，见 PROJECT_AUDIT）

- 逐帧决策来源：goal 97.2%、infra 2.0%、teacher 0.6%、**neural 0.3%**；
- BC 训练数据 16 维文明上下文 **16/16 标准差为 0**（死特征）；
- 无奖励、无 RL；规划器规则化，不能算作"神经网络能力"。

### 3.2 修复 1：让底层网络真正参与（分层职责明确）

`goals._verb_explore` 改为**委托**（`StepResult.delegated()`）：
规划层只决定"要探索"，逐帧移动/跳跃/说话交给底层策略；
`Population._decide` 在收到委托标记时**直接调用 BC 网络**
（不再先经过求偶教师——第一版这样试过，实测导致原地站立循环
`repeat_run_max=9476`，已修复回 161）。

修复后实测（C 组、640×400、700 世界秒，8 个个体）：

| 决策来源 | 修复前 | 修复后 |
| --- | --- | --- |
| neural（BC 底层） | 0.3% | **66.1%** |
| infrastructure（设施反射） | 2.0% | 28.4% |
| goal（规划器语义动作） | 97.2% | 5.0% |
| teacher（手写技能教师） | 0.6% | 0.6% |

> 说明：goal 份额下降是因为"探索"原本每帧都由规划器给方向；现在语义仍归规划层，
> 逐帧动作归网络。基础设施反射占比上升是因为 C/D 组会建粮仓/篝火并围绕其定居
> （能量管理），这是世界规则驱动的行为，不是硬编码剧情。

### 3.3 修复 2：新增真实高层目标策略（REINFORCE）

- `newworld/highlevel.py`：`GoalPolicy` MLP（30 维特征 → 32 → 8 个目标类型 softmax），
  numpy 手写前向/反向；`extract_features` 含能量/水分/库存/已验证技术/活跃设施/
  存活人口/最近目标成败/记忆摘要/昼夜；`GoalPolicyTrainer` 采样真实转移 →
  折扣回报 + 指数滑动基线 → 策略梯度更新 → 周期/独立评估。
- 奖励**来自真实世界变化**（集成层计算）：
  `+0.02×存活秒 +5×Δ已验证技术 +3×Δ活跃设施 +0.5×Δ资源 +2×Δ掌握知识
   +1×Δ合作分享 −2.5×目标失败 −8×个体死亡`（clamp [-25,35]），
  不可行目标 −0.5；**没有**用 BC 标签或固定进度。
- 集成：`Population.attach_goal_policy` → 每次需要新目标时策略采样 →
  `GoalManager.request_goal(world, agent, goal_type)`（不可行返回 None 并给小负奖励）；
  目标结束/个体死亡时结算转移，`drain_policy_transitions()` 供训练器消费。
- 训练入口：`tools/train_goal_policy.py`；正式训练 30 episodes × 4000 steps
  （120,000 世界步，墙钟 159.7s，4,174 条真实转移，30 次参数更新，
  `max|Δw|=0.3575`），checkpoint `assets/goal_policy.npz`（15,120 B）。
  独立评估 10 个未见 seed：mean return +407.9，alive 8.0，tech 3.6，facilities 2.6。

### 3.4 修复 3：BC 训练数据"死特征"

`training.collect_dataset(use_population=True)`（默认）改为在真实 `Population` 内采集，
16 维上下文里有 **12 维出现非零方差**（原来 16/16 恒 0）；`use_population=False`
保留旧的纯教师路径用于对照。新增测试 `tests/test_decision_chain.py` 断言
"训练数据上下文不是死特征""真实观测上下文有变化"。

### 3.5 决策验证（任务书 §4.5，全部有测试）

| 验证项 | 结果 | 证据 |
| --- | --- | --- |
| 改环境后决策变化 | ✅ | `test_low_level_mode_changes_behaviour`；同 seed 下 full vs off 的决策分布/技术数不同 |
| 关闭神经网络后有可测差异 | ✅ | `lowlevel_mode=off/random` 使决策分布、技术数改变；`test_planning_off_changes_behaviour` |
| 关闭规划器后有预期变化 | ✅ | baseline(A) goal_completion_rate=0 vs planning(C) >0.5；决策分布不同 |
| 未见初始布局仍有效 | ✅ | `test_unseen_seeds_and_sizes_run`（seed 101/202，480×300/640×400） |
| 新特征影响模型输出 | ✅ | `test_context_affects_policy_output_after_training`；训练后扰动特征概率变化（highlevel 测试） |

---

## 4. P3：文明机制升级

### 4.1 知识传播（沿用并纳入指标）

- 邻近教学（170px，概率规则）、出生继承（已验证知识 + 程序记忆）、
  死亡档案永久保留；本轮实验把它们纳入 10-seed 指标（`knowledge_retention`、
  `innovation_efficiency`）。

### 4.2 组合式创新原型（新增 `newworld/innovation.py`）

- **受限、可解释、确定性**：7 条已有技术各自带 tag（edge/binder/heat/container/structure），
  规则表定义 tag 相容对与效果：
  `edge+binder → harvest_bonus+1`、`container+binder → capacity_bonus+4`、
  `heat+container → energy_regen+0.15`、`heat+binder → energy_regen+0.10/water_regen+0.05`、
  `structure+container → capacity_bonus+2`。
- 流程：agent 掌握 ≥2 条 **verified** 技术且满足试制条件（有工具或 ≥2 件材料）→
  生成提案 → 检查材料（真实扣除）→ 按组合难度判定成败（失败也消耗材料）→
  成功调用 `Technology.register_custom_effect` 注册**真实效果**
  （采集加成/负重/能量/水/制作成功率）→ 上限 3 项/世界，可序列化追踪。
- 真实效果由 Technology 消费：`harvest()` 加 `harvest_bonus`、
  `Inventory.capacity()` 加 `capacity_bonus`、`apply_facility_effects` 施加
  `energy_regen/water_regen`、`simulate_recipe` 降低失败率。
- 自然运行实测：C 组 640×400、800 世界秒内产生 **2 项创新**
  （energy_regen 0.15 与 energy+water regen 组合），无需人工预置。

### 4.3 衰退与灭绝

保留：8 个累计创建上限、设施损耗、饥饿/老年死亡、collapse 五态机；
实验报告如实统计灭绝时间与原因（starvation vs old_age），不做"保证繁荣"的脚本。

---

## 5. 交付物与命令

| 交付物 | 路径 | 状态 |
| --- | --- | --- |
| 项目审计 | `docs/PROJECT_AUDIT.md` | ✅ 实测数据 |
| 实施报告 | `docs/IMPLEMENTATION_REPORT.md` | ✅ 本文件 |
| 实验报告 | `docs/EXPERIMENT_REPORT.md` | ✅ 10 seed × 6 配置 |
| 用户指南 | `docs/USER_GUIDE.md` | ✅ 鼠标操作 + 限制 |
| 接口契约 | `docs/upgrade_contracts_v1.md` | ✅ 冻结记录 |
| 新模块 | `ui_state.py` / `highlevel.py` / `innovation.py` | ✅ |
| UI 重写 | `app_tk.py` | ✅ |
| 训练入口 | `tools/train_goal_policy.py` | ✅ |
| 实验入口 | `experiments/run_upgrade_experiments.py` | ✅ |
| checkpoint | `assets/goal_policy.npz` | ✅ 真实训练产物 |

常用命令：

```powershell
cd newworldpro
$env:TEMP = (Resolve-Path .tmp).Path

# 交互（鼠标主导）
python -m newworld --windowed --size 1280x800

# 无人工鼠标自检
python -m newworld --ui-test .tmp\ui_test.json --windowed --size 1280x800

# 训练高层目标策略
python tools/train_goal_policy.py --episodes 30 --steps 4000 --eval-seeds 100 101 102 103 104 105 106 107 108 109 --out assets/goal_policy.npz

# 单次文明实验（可切策略/创新）
python -m newworld.simulation --group C --steps 16000 --dt 0.05 --width 640 --height 400 --policy-mode neural --policy-path assets/goal_policy.npz --innovation

# 6 配置 × 10 seed 正式实验
python experiments/run_upgrade_experiments.py --seed-start 100 --num-seeds 10

# 全量测试
python -m unittest discover -s tests
```

---

## 6. 已完成 / 已实现未充分验证 / 未实现（严格区分）

### 6.1 已完成并通过测试

- 鼠标主导交互（13 项程序化交互断言 + 41 个 UI 测试 + 实际窗口截图）；
- 决策来源计量、探索委托、神经网络参与率 0.3%→66%；
- BC 训练数据上下文死特征修复 + 测试；
- 高层目标策略 REINFORCE（真实采样/奖励/更新/checkpoint/独立评估，23 个测试）；
- 组合式创新（42 个测试 + 自然运行触发 2 项）；
- 硬约束：累计创建 ≤8 在新路径（策略/创新/UI 选择/重置）下均有测试覆盖；
- 全量测试（见 EXPERIMENT_REPORT 最后一节的具体数字）。

### 6.2 已实现但尚未充分验证

- **learned 高层策略的效果**：机制真实、行为可测改变，但 10 seed × 4000 步下
  文明级指标无显著差异（详见 EXPERIMENT_REPORT §5）；预算/方差限制已写明；
- 创新机制的长期影响：只在 640×400、~800 世界秒内观测到 2 项，未做 10-seed 统计前
  不声称"创新提升文明"；
- UI 帧率 19–21 fps（仿真成本为主）；未做长时间（>10 分钟）GUI 稳定性测试。

### 6.3 只有设计、未实现

- 人工"下达指令/干预"模式（本轮刻意不做，避免新增绕过世界规则的路径，
  USER_GUIDE 已说明默认自主仿真）；
- 多步组合（>2 个组件）创新、跨代知识教学动画等视觉表现；
- 新的 RL 底层动作策略（底层仍是 BC，未改成 PPO）。

### 6.4 因限制未完成

- 10 seed 已达成（目标值），但每 seed 单次 4000 步（约 40 世界秒）预算偏小，
  不能支撑"文明长期演化"的强结论；增加预算需要更长墙钟（估算每 seed 10 倍算力）；
- 无显示环境下 GUI 仅逻辑层可测（`test_ui_app` 自动 skip），本机已实际运行验证。

---

## 7. 已知问题 / 风险

1. **高层策略方差大**：同一 checkpoint 仅策略内部采样 RNG 流不同，单 seed return 可差 13 分
   （rl-dev 诊断）；实验用固定 policy seed + 10 seed 平均，报告注明来源。
2. **创新触发依赖库存巧合**：已放宽试制条件与成本规则，但触发仍稀疏；
   不做人工预置，报告只陈述观测到的次数。
3. **底层 BC 与教师占比**：teacher 在新分层下占比很低（0.6%），因为探索委托直连网络；
   若希望教师重新参与，可调整委托策略（需要另行实验，不在本轮范围）。
4. **UI 重置语义**：重置 = 新仿真运行（累计创建重新计数），与"死亡不复活"不冲突，
   但必须向用户明示（USER_GUIDE + 重置对话框已写）。
5. **文件锁**：编辑期间偶发 `ReplaceFileW EIO`，已用 UTF-8 安全的补丁脚本处理，
   不影响最终代码；最终以 `git diff` 与全量测试为准。

---

## 8. 回归测试方案与结果

- 全量：`python -m unittest discover -s tests` → **Ran 414 tests / OK (skipped=1)**（约 148s）；
- 新增测试：`test_ui_state.py`(34)、`test_ui_app.py`(7)、`test_highlevel.py`(23)、
  `test_innovation.py`(42)、`test_decision_chain.py`(11)；
- UI 程序化验证：`--ui-test` 13/13；
- 硬约束：`test_population_limits.py`（8）+ `test_cap_never_exceeded_under_policy_and_innovation`；
- 实验侧：每个 run 都在结果 JSON 中记录 `total_agents_created`，aggregate 检查不超过 8。
