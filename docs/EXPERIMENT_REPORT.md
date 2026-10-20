# 实验报告：基线与改进组（本轮升级）

- 目的：用**多随机种子**回答三个问题：
  1. 鼠标交互改造是否真实可用（不是静态 UI）；
  2. 决策链路修复（神经网络参与 + 学习型高层策略）是否改变行为/结果；
  3. 组合式创新是否真实触发并改变世界数值。
- 所有数字来自真实运行；未做人工挑选。运行产物：
  - `experiments/results/upgrade_final/runs.csv`（逐 run）
  - `experiments/results/upgrade_final/aggregate.json`（均值/标准差）
  - `experiments/results/upgrade_final/report.md`（自动生成的原始表）
  - `experiments/results/goal_policy_training.json`、`goal_policy_ablation.json`（高层策略训练/对照，来自 rl-dev 的独立脚本）
- 随机种子：**10 个**（100–109，均未参与高层策略训练；训练见 §5）。
- 审计与实施细节见 `docs/PROJECT_AUDIT.md`、`docs/IMPLEMENTATION_REPORT.md`。

---

## 1. 固定预算（6 配置共用）

| 项目 | 值 |
| --- | --- |
| 世界尺寸 | 640×400 |
| 时间步长 dt | 0.05 s（20 Hz） |
| 每 run 步数上限 | 12,000 步（600 世界秒） |
| 累计创建上限 | 8（硬约束；每 run 断言的指标） |
| 感知 | 关闭（不进入决策输入，仅影响墙钟） |
| 环境扰动 | 跟随配置（本轮矩阵均为关闭，避免与决策对照混淆） |
| 高层策略 checkpoint | `assets/goal_policy.npz`（真实训练产物） |
| 种子 | 100–109（10 个） |

配置定义（`experiments/run_upgrade_experiments.py`）：

| 配置 | 高层目标选择 | 底层动作 | 知识/传承 | 创新 |
| --- | --- | --- | --- | --- |
| `baseline` | 无规划（A 组） | BC 网络 | 关闭 | 关闭 |
| `planning` | 规则启发式 | BC 网络 | 开启 | 关闭 |
| `learned` | 训练的高层 REINFORCE 策略 | BC 网络 | 开启 | 关闭 |
| `random_policy` | 均匀随机目标（对照） | BC 网络 | 开启 | 关闭 |
| `lowlevel_random` | 规则启发式 | 随机动作（对照） | 开启 | 关闭 |
| `innovation` | 训练的高层策略 | BC 网络 | 开启 | **开启** |

---

## 2. 结果（10 seed 汇总，真实数字）

自动统计源文件：`experiments/records/upgrade_final/runs.csv` /
`aggregate.json` / `report.md`（本报告数字由它们誊写，未人工调整）。

| 配置 | 技术（均值±标准差） | 设施 | 创新 | 目标完成率 | 最长重复动作 | 灭绝（饥饿） | 灭绝时间 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | **0.00±0.00** | 0.00±0.00 | 0 | 0.000 | 5007 | **8/10（全为饥饿）** | 483s（均值，仅灭绝 run） |
| planning | 4.10±0.70 | 3.20±1.08 | 0 | 0.972 | 1028 | 0/10 | - |
| learned | 4.20±0.75 | 3.00±0.89 | 0 | 0.969 | 1155 | 0/10 | - |
| random_policy | 4.70±1.00 | 3.50±0.92 | 0 | 0.958 | 762 | 0/10 | - |
| lowlevel_random | 4.80±0.87 | 3.80±1.08 | 0 | 0.963 | 190 | 0/10 | - |
| innovation | 4.20±0.75 | 3.30±1.00 | **1.90±1.37 [0,3]** | 0.969 | 1110 | 0/10 | - |

配对差（同 seed，df=9，95% CI）：

| 对比 | 指标 | 差值 [CI] | 胜/负/平 |
| --- | --- | --- | --- |
| planning − baseline | 技术 | **+4.10 [+3.60, +4.60]** | 10/0/0 |
| planning − baseline | 设施 | **+3.20 [+2.43, +3.97]** | 10/0/0 |
| planning − baseline | 灭绝时间 | **−386.9s [−528.4, −245.4]**（baseline 更早灭绝） | 0/8/2 |
| learned − planning | 技术 | +0.10 [−0.65, +0.85] | 3/2/5 |
| learned − planning | 设施 | −0.20 [−1.15, +0.75] | 3/4/3 |
| random_policy − planning | 技术 | +0.60 [−0.13, +1.33] | 5/1/4 |
| innovation − learned | 技术 | +0.00 [−0.32, +0.32] | 1/1/8 |
| innovation − learned | 设施 | +0.30 [−0.03, +0.63] | 3/0/7 |
| lowlevel_random − planning | 技术 | **+0.70 [+0.24, +1.16]** | 6/0/4 |

### 2.1 主要观察（如实）

1. **规划层是唯一有显著文明级收益的机制**：`planning` 相对 `baseline`
   技术 +4.1、设施 +3.2（均 10/10 显著），并把 8/10 的饥饿灭绝降为 0/10；
   baseline 的最长连续同动作 5007 步，planning 降到约 1000 步。
2. **学习型高层策略（learned）没有可测的文明级收益**：与 `planning` 的
   技术差 +0.10、设施差 −0.20，CI 均跨 0；这与独立 A/B（§5.2）一致。
   它的可测差异在行为层：高层目标决策更多、不可行目标请求更少（见 §5.2 与决策来源表）。
3. **随机目标策略（random_policy）也不比规划差**（技术 +0.60，CI 跨 0）——
   在本预算/本世界下，"选什么目标类型"不是瓶颈。
4. **底层网络不是瓶颈（甚至随机底层不差）**：`lowlevel_random` 的技术
   比 `planning` 高 0.70（CI 不跨 0），因为语义层（规划器）仍负责采集/制作/建造的
   具体动作，而随机移动在设施附近反而减少了走失。这直接证明了上一轮审计的判断：
   **文明进展主要由规则/规划驱动，不能归功于神经网络**。本轮已把这一事实
   用决策来源占比单独计量，而不是用文明指标冒充模型能力。
5. **组合式创新真实触发**：`innovation` 组 10 seed 中 7 个触发 1–3 项
   （均值 1.90±1.37，上限 3），并注册真实世界效果；对设施的影响 +0.30 处于
   边缘（CI 下界 −0.03），不能宣称显著。
6. **所有配置的累计创建都 ≤8**（CSV 每行可核验），没有通过任何新路径绕过上限。

## 3. 决策来源占比（逐帧，10 seed 均值）

| 配置 | neural | teacher | goal（规划语义动作） | infrastructure | lowlevel_random | policy_goal |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.492 | 0.508 | 0.000 | 0.000 | 0.000 | 0.000 |
| planning | 0.772 | 0.007 | 0.034 | 0.187 | 0.000 | 0.000 |
| learned | 0.773 | 0.007 | 0.028 | 0.186 | 0.000 | 0.004 |
| random_policy | 0.761 | 0.008 | 0.031 | 0.194 | 0.000 | 0.001 |
| lowlevel_random | 0.000 | 0.000 | 0.029 | 0.672 | 0.299 | 0.000 |
| innovation | 0.786 | 0.007 | 0.028 | 0.173 | 0.000 | 0.004 |

审计时神经网络只占 **0.3%** 的逐帧决策；修复后（探索语义委托 + 直连 BC 网络）
有规划的各组约 **77–79%**。`baseline` 无规划，网络/教师约各半。
注意：高层目标策略只占约 **0.4%** 的逐帧决策，因为它每次只选一个目标
（约 13–15 个/4000 帧），它的效果要看目标分布与行为指标，而不是帧占比。

## 4. 鼠标交互验证（任务书 §7.4 的 10 项）

命令（Lead 独立复跑，真实事件注入）：

```
python -m newworld --ui-test .tmp\lead_ui_test.json --windowed --size 1024x720
→ EXIT=0，13/13 PASS
```

| 任务书要求 | 对应断言 | 结果 |
| --- | --- | --- |
| 1 点击智能体可选中 | `world_click_selects_agent` | PASS |
| 2 详情面板展示对应个体数据 | `detail_and_stats_use_real_data`（22 行详情、15 行统计、真实事件日志） | PASS |
| 3 点击其他智能体可切换 | `world_click_switches_selection` | PASS |
| 4 点击空白取消选择 | `world_click_empty_clears_selection`（再次点击同一体也可取消） | PASS |
| 5 点击 UI 不误触发世界操作 | `layout_regions_and_click_isolation` | PASS |
| 6 开始/暂停/继续/单步 | `single_step_advances_exactly_one_frame`（每次恰好 0.016667s；运行时拒绝单步） | PASS |
| 7 重置后无失效选择/监听 | `reset_clears_selection_and_list`（选择清空、列表重建、运行恢复） | PASS |
| 8 缩放/平移后命中正确 | `wheel_zoom_and_drag_pan_hit_consistency`、`viewport_transform_and_zoom_anchor` | PASS |
| 9 暂停时仍可查看详情 | `pause_freezes_time_but_panel_visible`（世界时间冻结、详情 22 行） | PASS |
| 10 鼠标不绕过累计创建上限 | 选择/查看不创建个体；上限测试见 `test_population_limits.py` 与每个 run 的 `total_agents_created<=8` 断言 | PASS |

补充：`python -m newworld --windowed --size 1280x800 --autoclose-ms 2500 --screenshot .tmp\ui_shot3.png`
实际启动成功（19–21 fps），截图见 `.tmp/ui_shot3.png`（未入库的临时产物）。

---

## 5. 高层策略：训练与独立对照（rl-dev 真实运行）

### 5.1 训练

```
python tools/train_goal_policy.py --episodes 30 --steps 4000 \
  --eval-seeds 100 101 102 103 104 105 106 107 108 109 \
  --out assets/goal_policy.npz
```

- 预算：30 episodes × 4000 steps = 120,000 世界步；墙钟 **159.7s**
- 真实策略转移 **4,174**；参数更新 **30 次**；`max|Δw|=0.3575`（真实梯度更新）
- checkpoint：`assets/goal_policy.npz`（15,120 B）
- 训练中周期评估 + 最终独立评估（10 未见 seed）：mean return +407.9，alive 8.0，
  tech 3.6，facilities 2.6，未灭绝

### 5.2 三类对照（同 checkpoint、固定策略 RNG 流、10 未见 seed、4000 步）

| 指标 | trained | untrained | heuristic |
| --- | --- | --- | --- |
| tech_verified | 3.50±1.63 | 3.60±1.50 | **4.20±1.47** |
| facilities_active | 2.50±1.20 | 2.60±1.28 | **3.30±1.68** |
| alive_count | 8.0 | 8.0 | 8.0 |
| goal_decisions | **76.8±9.17** | 36.5±6.67 | 0 |
| policy_goal share | 0.0036 | 0.0017 | 0 |
| policy_infeasible share | 0.0032 | 0.0052 | 0 |

配对差（同 seed，95% CI）：`tech` trained−untrained −0.10 [−0.73, +0.53]、
trained−heuristic −0.70 [−1.60, +0.20]；`facilities` −0.10 / −0.80，**全部跨 0**；
`alive` 三者完全相同。显著差异只在行为层：`goal_decisions` +40.3 [+32.2, +48.4]（10/10），
可行目标请求比例上升、不可行请求下降。

**结论（必须如实写入）**：学习型高层策略**真实运行、真实改变了目标选择行为**，
但在本预算（10 seed × 4000 步）下**未观察到文明级收益**；相对 heuristic 的点估计略低
（差异在噪声内）。不声称"训练出了更好的文明策略"。

---

## 6. 组合式创新（真实触发）

自然运行（无人工预置 verified，C 组、seed 0、640×400、14000 步）：

```
python -m newworld.simulation --group C --steps 14000 --dt 0.05 \
  --width 640 --height 400 --innovation --json .tmp/innov.json
```

- `innovations = 2`：
  - `campfire + fiber_cord → corded_lamp`，效果 `energy_regen 0.1, water_regen 0.05`
  - `campfire + granary → fire_vessel`，效果 `energy_regen 0.15`
- 探针确认世界侧 `Technology.register_custom_effect` 被真实调用（registered=2），
  且 agent 的能量/水分恢复数值随工具变化（真实 physiology 变化，不是 UI 数字）。
- 10 seed 矩阵（`innovation` 配置）：多数 seed 触发 1–3 项；未触发 seed 原因如实记录
  （材料不足/随机失败/上限已满），脚本不补数。

---

## 7. 回归测试（真实数字）

| 命令 | 结果 |
| --- | --- |
| `python -m unittest tests.test_ui_state tests.test_ui_app` | 41 tests OK（GUI 无显示自动 skip 1） |
| `python -m unittest tests.test_highlevel` | 23 tests OK |
| `python -m unittest tests.test_innovation` | 42 tests OK（skipped=1） |
| `python -m unittest tests.test_decision_chain` | 11 tests OK |
| `python -m unittest discover -s tests` | **Ran 516 tests / OK (skipped=1)**（约 66s，含 V2 新增 95 个测试） |
| `python -m newworld --ui-test ...` | 13/13 PASS，exit 0 |
| 每个实验 run | `total_agents_created ≤ 8`（CSV 列可直接核验） |

---

## 8. 失败与限制（必须一起读）

1. **baseline 组灭绝率**：10 seed 中 8 个在 600 秒内饥饿灭绝（技术 0、设施 0）——
   这不是 bug，是"无规划/无知识"的可解释结果。
2. **learned 无显著收益**：见 §5.2；原因包括预算小、策略只影响目标类型、
   底层执行由规则/BC 主导、世界资源/食物供给是主要瓶颈。
3. **lowlevel_random 仍能发展技术**：说明文明指标不能用来"证明网络能力"；
   本报告把"行为层改变"与"文明层结果"分开陈述。
4. **创新触发稀疏**：依赖材料窗口与随机难度（0.35），并非每次都会发生；
   上限 3 项/世界。
5. **单次运行 600 世界秒**：受 8 人上限与寿命约束，单 run 必在老年期结束；
   指标聚焦积累（技术/设施/存活时间），不是"人口增长"。
6. 高层策略评估方差大（同一权重不同采样 RNG 流单 seed return 可差 13 分），
   因此所有结论均基于 10 seed 与配对差，而非单次成功案例。
