# V2 升级报告：感知进入决策、低层 RL、放开上限、无限世界

> **V3 注**：V2 无限世界命令在 V3 起需加 `--classic`（`python -m newworld --classic --infinite ...`）。V3 见 `UPGRADE_V3_REPORT.md`。

> 起因：用户通读源码后的评审指出三个结构性问题（感知与决策断开、底层 BC 有教师
> 天花板、8 人上限使系统只是"衰亡微缩模型"），并明确"不局限于屏幕，创造一个
> 无限的世界"。
>
> 本报告只写**真实实现 + 真实运行结果**，并按"已完成且测试通过 / 已实现未充分
> 验证 / 只有设计 / 未完成"分开陈述。契约见 V1/V2 契约（文档已删除）。
> 上一轮（鼠标交互 + 高层策略 + 创新）见 `docs/IMPLEMENTATION_REPORT.md` 与
> `docs/EXPERIMENT_REPORT.md`。

## 0. 一句话结论

| 工作流 | 状态 | 真实证据 |
| --- | --- | --- |
| 感知进入观测 | ✅ 已实现并测试 | 观测 40→**56 维**（+16 真实视网膜特征）；实验组 `visual_active_dims=16/16`（关闭组 0/16）；旧 checkpoint 零初始化迁移逐比特一致 |
| 低层 BC → RL | ✅ 已实现并测试（效果见实验） | 19 项单测 + 有限差分 max_abs_err **3.6e-11**；30×4000 真实训练（待填） |
| 放开人口上限 | ✅ 已实现（配置 + 实验） | `max_agents_created` 可配；8/32 人对照实验（待填） |
| 无限世界 | ✅ 已实现并测试 | 确定性分块 `InfiniteScene` + 窗口自动平移；GUI `--infinite` 真机截图；测试通过 |

---

## 1. 感知：视网膜特征真正进入决策

### 1.1 改了什么

- 新增 `newworld/visual_features.py`：`VisualFeatureExtractor` 从**真实光学/眼睛/
  视网膜/注视系统**（`VisualSystem.perceive`）提取 16 维紧凑特征（写进 docstring）：
  8 维 ganglion 拓扑分箱 + 平均/中央凹亮度 + 光适应 + 注视偏移 + 模糊 σ +
  扫视抑制代理 + 声压级 + 是否听到。
- `neural.py`：`FEATURE_DIM 40 → 56`（24 生理 + 16 文明上下文 + 16 视觉），
  新增 `from_v2_weights` 迁移；24→56、40→56 两条旧 checkpoint 链都零初始化，
  视觉段为 0 时输出**逐比特一致**。
- `population.py`：`perception_stride` 控制真实感知刷新（默认 4）；
  每 stride 帧把 16 维写入 `character.visual_features`；
  `visual_summary()` 记录各维标准差/有变化维数（审计"视觉是否真的变化"）。
- `training.py`：BC 数据采集开启视觉特征（数据集 56 维），视觉段有真实方差。
- 实验开关：`RunSpec.visual_features` / CLI `--visual/--no-visual`。

### 1.2 真实测试与实测数字

```powershell
python -m unittest tests.test_visual_features tests.test_observation_migration tests.test_v2_integration
```

真实结果（本机）：
- `tests.test_visual_features`：**21 tests OK**（维度/有限性/范围、真实非零方差、
  朝向/位置变化、异常降级、source 字段、低分辨率参数生效）；
- `tests.test_observation_migration`：**11 tests OK**（24→56、40→56 两条迁移链，
  视觉段为 0 时输出逐比特一致）；
- `tests.test_v2_integration`：**10 tests OK**（视觉特征经 Population 写入
  `character.visual_features` 并进入 56 维观测）。

实测成本（`tools/bench_vision_features.py`）：
| 档位 | 单次 observe |
| --- | --- |
| 学习用低分辨率档（默认） | **median 1.623 ms**（满足 ≤2.5ms 目标） |
| GUI 高保真档（1800 视锥 / 3000 视杆 / 32 bins / 120 samples/deg） | 16.174 ms |

BC 训练数据（`collect_dataset`，240 帧）：**56 维**，其中视觉段
**14/16 维有非零方差**（std 最大 0.63）；视觉关时该段全 0。

### 1.3 诚实边界

- 学习用的是**低分辨率视网膜档**（`n_cones`/`n_bins` 降低、可注入参数），
  不是 GUI 高保真档；这是为了把感知成本压到可训练范围，报告实测 ms。
- 视觉特征每 `perception_stride` 帧刷新一次（默认 4，实验 8），不是逐帧；
  行为反应有 0.2–0.4 秒滞后。
- 视觉段权重从零初始化开始学，短预算 BC 训练**不足以**让视觉信息改变行为；
  这由实验数字说话（见 §4），不预设"感觉很重要"。

---

## 2. 低层动作策略：BC → 真实 REINFORCE

### 2.1 改了什么

- 新增 `newworld/rewards.py`：**只奖励真实世界变化**（Δ已验证技术 +1.0、
  Δ活跃设施 +0.5、净资源 +0.2/单位、Δ掌握知识 +0.2、Δ合作 +0.1，
  −0.001·dt 时间成本，死亡 −2.0，单步 clamp ±3）。
  反刷分测试：原地站立 ≤0、粮仓旁挂机 ≤0（能量恢复不算奖励）、
  重复失败动作 ≤0。
- 新增 `newworld/lowlevel_rl.py`：`LowLevelPolicy` 把现有 MLP 变成可采样随机策略
  （mode/move softmax、jump/speak Bernoulli，**语言头保持 argmax 不参与 RL**）；
  手写 log-softmax+BCE 反向；`LowLevelTrainer` 用真实 env 奖励做 REINFORCE，
  按 agent 折扣回报 + 指数滑动基线。
- `Population`：`low_level_mode="rl"` 时采样并记录
  `(agent_id, features, action, reward)`；`attach_lowlevel_policy` /
  `drain_lowlevel_transitions`；死亡时用固定负奖励结算。
- `simulation.make_lowlevel_rl_env`：与文明实验同规则的训练/评估环境；
  `tools/train_lowlevel_rl.py` 真实训练 CLI。

### 2.2 真实训练与对照

**正式训练（真实 env，30 episodes × 4000 steps）**：
```
python tools/train_lowlevel_rl.py --episodes 30 --steps 4000 --seeds 0 \
  --eval-seeds 100,101,102,103,104,105,106,107,108,109 --eval-steps 4000 \
  --out assets/lowlevel_rl.npz --json experiments/results/lowlevel_rl_training.json
```
- 120,000 世界步、**墙钟 735.1s**；**真实转移 583,340 条**、**updates=30**、
  `params_changed=True`、**max|Δw|=9.86e-5**、`||Δw||=7.66e-4`；
- checkpoint `assets/lowlevel_rl.npz` 63,642 B；独立评估（10 seeds）：
  mean_return **+48.68 ± 38.80**（min 15.56 / max 127.58），alive 8、tech 4.2、
  facilities 3.1、lowlevel_decisions 19,762；
- 训练 return first5 +52.02 → last5 +59.44（**无单调上升**，方差很大）。

**10 seed 初始权重对照**（同策略采样、固定 policy seed、训练/评估 seed 不重叠）：

| 指标 | BC 初始权重 | RL 训练后 | 配对差 [95% CI] |
| --- | --- | --- | --- |
| return | +65.61±35.55 | +60.47±38.84 | **−5.13 [−22.93, +12.66]** |
| tech_verified | 5.00±1.00 | 4.80±1.08 | −0.20 [−0.94, +0.54] |
| facilities_active | 3.90±0.94 | 3.90±1.04 | 0.00 [−0.95, +0.95] |
| alive_count | 8.0 | 8.0 | 0 |
| lowlevel_decisions | 19,169.5±1,208.5 | 19,660.4±1,176.3 | **+490.9 [+53.5, +928.3]（+2.6%，8/10 wins）** |

**结论（必须如实引用）**：REINFORCE **真实发生了**（583k 转移、30 次更新、权重确实改变），
但训练后相对 BC 初始权重**没有可测的世界级提升**；唯一显著差异是低层决策占比略增
（行为变了，但结果被 seed 间噪声支配）。原因之一已定位：每回合只在回合末更新一次、
梯度按约 1.9 万条转移平均、lr=0.01 → 30 次更新后 max|Δw|≈1e-4，学习信号太弱。
**不能声称"RL 优于 BC"**。

**机制验证（更早的冒烟预算，仍真实）**：`tools/train_lowlevel_rl.py --episodes 2
--steps 600 --seeds 0 --eval-seeds 100,101 --eval-steps 600`
- 2 个训练回合、**2102 条真实 agent-步转移**、**2 次参数更新**、
  `max|Δw|=3.0e-4`、`||Δw||=2.56e-3`（证明梯度真的改了权重）；
- 独立评估 seed 100/101：mean_return **+20.84 ± 4.70**，tech 4、facilities 2、
  未灭绝；评估回合约 73–75% 的 agent-帧由低层 RL 策略驱动；
- 语言头对照：language `w3` 列与 BC checkpoint **逐比特相同**，
  RL 头 `w3` 列已改变 → 语言头确实未参与 RL。

**正式训练与 10 seed 对照**（待填：30 episodes × 4000 steps 输出、bc vs rl 表）。

---

## 3. 人口上限对照：从"衰亡微缩模型"到可观察人口动态

- `max_agents_created` 现在可由 `RunSpec` / CLI `--cap` 覆盖，硬约束语义不变
  （累计创建，死亡不退还；超过上限的创建仍返回 `None`）。
- 对照实验：cap 8 vs cap 32（无限世界）+ cap 32（有限世界），观察
  人口时间序列、崩溃后恢复、灭绝率。
- **未新增任何绕过上限的代码路径**；实验 CSV 每行记录 `total_agents_created` 与
  `cap_violation`，聚合里统计违规数。

长期对照实验（无限世界、900 世界秒、3 seeds；`experiments/records/v2_cap/`）：

| 配置 | 累计创建 | 死亡 | 末存活 | 存活范围 | 恢复（存活增加）次数 | 技术 | 设施 | 灭绝 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cap 8 | 8/8 | 2 | **6** | 2..8 | 5–6 | 4 | 2–4 | 否 |
| cap 64 | 64/64 | 2 | **62** | 2..64 | 25–35 | 4 | 2–4 | 否 |

- 上限放开后，人口从 2 恢复/增长到 64（创建预算用满），**没有出现灭绝**；
- 900 秒内只发生了 2 次死亡（寿命 900 世界秒，死亡集中在末期），因此**没有观察到
  完整的"崩溃→恢复"周期**——需要远长于寿命的运行才能看到代际周转；
- 两组的**技术/设施相同**（4 项 / 2–4 个）：更多人口没有带来更多技术，
  瓶颈在知识/材料/执行，而不是人口数量（与上一轮结论一致）。

---

## 4. 无限世界：屏幕只是窗口

### 4.1 设计

```
InfiniteScene（newworld/infinite_scene.py）
  确定性生成：全局原子材料/对象/池塘/浆果由 seed+全局坐标决定
  修改持久化：set_atom/clear_atom/采果按全局坐标存入稀疏字典
  recenter(delta)：窗口平移，重建局部 grid/objects/berries/pond
        ↓ 与 PixelScene 同构的窗口 API
World._maybe_recenter()
  人口接近窗口边缘（<25% 宽或 >75%）时平移 25% 窗口
  同时把所有动态 x（agents/facilities/community）反向平移 → 局部坐标体系不变
  记录 window_origin_px（全局偏移，可审计）
```

- GUI/CLI：`--infinite`（GUI 与 simulation），默认仍是有限世界（既有 477 个测试
  的默认行为不变）。
- 屏幕 = 当前窗口；随着人口迁移，世界自动滚动，可以一直走下去
  （真正的"不局限于屏幕"的工程最小实现）。

### 4.2 真实验证

- 模块测试：`tests.test_infinite_scene` **29 tests OK**；`tests.test_v2_integration`
  含无限世界窗口跟随/渲染尺寸用例。
- 确定性：同 seed 重叠区**逐原子一致**（不同窗口宽度、不同 recenter 路径都验证）；
  修改（set_atom/clear_atom/采果/踩踏）按全局坐标持久化，切走再切回一致。
- recenter 实测：**1.02–1.42 ms/次**（640×400、256×80 窗口；契约目标是 ≤30ms）。
- World 集成压测：连续 120 次真实平移，`window_origin_px=-14400`、
  `infinite_fallback_reason=""`、agent.x 全程在屏内（约 1.97 ms/step 含 population.step）。
- GUI 真机验证：`python -m newworld --infinite --windowed --size 960x600
  --screenshot .tmp/ui_infinite2.png` 成功；截图确认 1:1 像素渲染、地形/树/池塘
  为程序化生成、状态栏计数正常。默认窗口宽度 = 屏幕原子宽度（128 原子 = 640px），
  避免整幅 resize 造成的横向压缩。
- 生成规则摘要：材料/地形由 splitmix64(seed, 全局坐标) 决定；池塘每 128 原子一个
  （默认窗口必含水）；树/岩石/浆果按 48/32/28 原子格点确定性放置；浆果约
  14.5 颗/窗口（有限 640×400 世界只有 2 丛）。

---

## 5. 真实实验结果

数据：`experiments/records/v2_final/`（runs.csv / aggregate.json / report.md）；
预算：**6 seeds（200–205）× 6000 步（300 世界秒）、640×400、perception_stride=8**、
共 42 个 run、墙钟 554s；每行都断言累计创建 ≤ cap。

| 配置 | 技术 | 设施 | 存活 | 累计创建 | 视觉维数 | 低层转移 | 灭绝 | 上限违规 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c_bc_vision_off | 4.17±0.90 | 0 | 8.0 | 8.0 | 0/16 | 0 | 0/6 | 0 |
| c_bc_vision_on | 4.17±0.90 | 0 | 8.0 | 8.0 | 16/16 | 0 | 0/6 | 0 |
| c_rl_policy | 4.17±1.21 | 0 | 8.0 | 8.0 | 15.7/16 | 35,198 | 0/6 | 0 |
| c_rl_random | 4.00±1.00 | 0 | 8.0 | 8.0 | 14.0/16 | 0 | 0/6 | 0 |
| cap8_infinite | 4.00±1.00 | 0 | 8.0 | 8.0 | 16/16 | 0 | 0/6 | 0 |
| cap32_infinite | 4.00±1.00 | 0 | 29.8±2.7 | 29.8 | 16/16 | 0 | 0/6 | 0 |
| cap32_finite | 4.33±0.94 | 0 | 32.0 | 32.0 | 16/16 | 0 | 0/6 | 0 |

### 5.1 感知开 vs 关：**干净的空结果**

6 个 seed 的逐指标配对比较：`tech_verified`、`facilities_active`、`alive_count`、
`goal_completion_rate`、`cooperation_index` **全部 6/6 平局，差值 0**；
唯一差异是 `visual_active_dims` 16/16（开）vs 0/16（关）。

原因（已定位并修复一个混淆源）：现有 BC checkpoint 是 legacy 24 维，迁移到 56 维时
**视觉列权重为 0**，且 BC/RL 训练没有把视觉列训练出有效权重 → 视觉输入乘 0，
决策不变。**"感知进入观测"已实现且可测（16/16 维有方差、进入 56 维输入），
但"策略学会用视觉"没有发生**；要证明后者必须重训策略（下一步工作，见 §6.3）。
本报告不把"视觉开/关无差异"说成"视觉有用/没用"，而是明确：**接线完成、学习未完成**。

### 5.2 BC vs RL vs 随机

| 对比（同 seed） | 技术差 [95% CI] | 目标完成率差 | 存活差 |
| --- | --- | --- | --- |
| RL 低层 − BC | +0.00 [−1.05, +1.05]，2 胜/1 负/3 平 | +0.004 [−0.008, +0.016] | 0 |
| 随机低层 − BC | −0.17 [−1.11, +0.78]，1 胜/1 负/4 平 | −0.015 [−0.042, +0.012] | 0 |

与 §2.2 的 10 seed 初始权重对照一致：**RL 机制真实、可测行为变化（低层决策占比
+2.6%），但没有文明级收益**；随机低层在短预算下也不差（与上一轮"规则/规划主导"一致）。

### 5.3 cap 8 vs 32 / 有限 vs 无限

- `cap32_infinite − cap8_infinite`：存活 **+24.0 [+24.0, +24.0]**（6/6 显著），
  技术差 0.0，目标完成率 −0.045 [−0.070, −0.020]（人更多，人均目标完成率略低）。
- `cap32_finite − cap8_infinite`：存活 +21.8 [+19.0, +24.7]，技术 0.0，
  目标完成率 −0.045（同向）。
- **39 个 run 无任何上限违规**；`total_agents_created` 均 ≤ cap。
- 300 秒内所有配置设施都是 0（时间太短；900 秒长期实验里达到 2–4 个设施，
  见 §3），因此这一节只用于回答"放开上限是否改变人口规模/存活"，不用于技术结论。

---

## 6. 已完成 / 未充分验证 / 只有设计 / 未完成

### 6.1 已完成且测试通过

| 项目 | 证据 |
| --- | --- |
| 视觉特征提取（真实视网膜路径） | `tests.test_visual_features` 23 tests OK + benchmark 1.06–1.75 ms |
| 观测 40→56 双旧格式迁移 | `tests.test_observation_migration` |
| 奖励函数与反刷分 | `tests.test_rewards` |
| 低层 REINFORCE（手写梯度 + 有限差分） | `tests.test_lowlevel_rl`（19 项，max_abs_err 3.6e-11） |
| RL 转移/挂载/解码集成 | `tests.test_v2_integration` |
| 无限世界 + 窗口平移 | `tests.test_infinite_scene` 29 OK + `tests.test_v2_integration` |
| cap 覆盖语义 | `tests.test_v2_integration.test_population_cap_override`；6+39 个 run 全部 ≤cap |
| 可复现性（society 不落盘、视觉列不随机变异） | `tests.test_v2_integration.TestReproducibilityGuards`；视觉开/关 6/6 完全相同 |

### 6.2 已实现但未充分验证

- 视觉特征对文明级结果的影响（短预算 + 零初始化视觉权重，实验结论以数字为准）。
- 低层 RL 相对 BC 的收益（小预算、单 seed 训练；10 seed 评估是唯一证据）。
- 无限世界的长期稳定性（长跑/大人口、窗口频繁平移的墙钟成本）。
- RL 训练后的语言头行为：权重未变但共享隐藏层会变，语言 argmax 可能间接变化。

### 6.3 只有设计 / 未完成

- 感知的**端到端可微**（视觉直接进 RL 梯度）——当前是"特征拼接 + 各自训练"。
- 真实人口学（生育/年龄结构/资源承载）——cap 只是放开创建预算，不是人口模型。
- 无限世界的纵向/多区块、跨区块地形融合（当前纵向固定、横向分块）。
- 多智能体集中式训练/分布式执行（CTDE）等高级 RL。

---

## 6.4 过程中发现并修复的缺陷（诚实记录）

| 缺陷 | 影响 | 修复 |
| --- | --- | --- |
| `Population` 默认读/写 `assets/society.npz` | 不同实验 run 互相污染，破坏可复现性；视觉开/关的差异一度来自 run 顺序而非视觉 | 改为 `persist_society=False` 默认（内存网络）；GUI 可选开启；新增测试 |
| 遗传给孩子网络加随机突变时**包括视觉列** | 未训练的随机投影会改变行为，容易被误读成"视觉能力" | 视觉列不做随机变异（只随 BC/RL 学习更新）；新增测试证明孩子视觉列仍为 0 |
| `eye.defocus_diopters` 对贴脸/嵌入几何体不设下限 | 失焦 29804 D → 模糊 σ 3415° → FFT 单帧 130–175 ms（GUI 偶发卡顿） | 按人眼近点 0.1 m 与 20 D 上限截断（真实生理约束）；学习路径另有近点护栏 |

这三项都是**先复现、再修复、再验证**：修复后视觉开/关在 6 个 seed 上逐指标完全一致
（说明之前的"差异"确实是混淆），全量测试保持通过（**516 tests / OK，skipped=1**）。

## 7. 复现命令

```powershell
cd newworldpro
$env:TEMP = (Resolve-Path .tmp).Path

# 测试
python -m unittest discover -s tests

# 训练低层 RL
python tools/train_lowlevel_rl.py --episodes 30 --steps 4000 --seeds 0 `
  --eval-seeds 100,101,102,103,104,105,106,107,108,109 --eval-steps 4000 `
  --out assets/lowlevel_rl.npz --json experiments/results/lowlevel_rl_training.json

# V2 实验矩阵（感知开/关、BC/RL/随机、cap/无限世界）
python experiments/run_v2_experiments.py --seed-start 200 --num-seeds 6 --tag final

# GUI 无限世界
python -m newworld --infinite --windowed --size 960x600
```

---

## 8. 风险与回归

- 既有 477+ 测试保持通过（有限世界为默认；新增能力都是显式开关）。
- 感知 stride > 1 会让"感知开关"的行为影响更滞后；实验以配置值记录。
- 无限世界会平移设施/社区坐标；若未来加入"设施位置历史"，需要同时平移。
- 低层 RL 的奖励只覆盖四类真实变化，仍有被"资源囤积"利用的空间
  （库存满后不再增益，已测）；大规模训练前应做奖励敏感性实验。
