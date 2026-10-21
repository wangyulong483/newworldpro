# P6-B 去 LLM 生成式记忆与反思：配对种子消融报告

本报告由协议 memory_ablation 自动生成；数字来自真实训练+评估，
不是占位符。所有臂都从相同 policy seed 的相同初始权重开始，
在同一批 eval seed（同地图/同初始人口）上配对评估，每臂 n=3。

## 设计

- 实验名：p6_memory
- 世界：128×80，food_regrow_per_s=5e-05，初始人口=120，metabolism_per_s=0.04
- 策略：neural，hidden=32，lr=0.02；训练 3 回合 × 40s（train seeds [1000, 1001, 1002]），policy seed=7
- 配对评估：3 个 seed [0, 1, 2]，每回合 40s；每臂 n=3（同一批 seed）
- 记忆参数：capacity=24，top_k=3，halflife=30.0s，反思间隔=20.0s，反思最少事件=6
- 臂：off（9 维）/ off_padded（15 维恒零记忆对照）/ recency / full / full_reflect；总墙钟时间 127.4s；最佳记忆臂：full_reflect；全臂最佳：full_reflect

## 1. 各臂绝对指标（mean ± std，每臂 n=3）

| 臂 | obs_dim | eat_events | eat_rate | satiation_mean | satiation_std | population | mean_life | starvation_deaths | births | total_reward | gini_hunger |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 9 | 3212.333 ± 216.082 | 0.669 ± 0.045 | 0.680 ± 0.028 | 0.262 ± 0.012 | 113.000 ± 2.944 | 35.156 ± 0.710 | 7.000 ± 2.944 | 0.000 ± 0.000 | 641.538 ± 54.630 | 0.216 ± 0.014 |
| off_padded | 15 | 3538.667 ± 127.878 | 0.737 ± 0.027 | 0.687 ± 0.039 | 0.275 ± 0.011 | 113.333 ± 0.471 | 34.184 ± 0.497 | 6.667 ± 0.471 | 0.000 ± 0.000 | 717.380 ± 37.520 | 0.222 ± 0.027 |
| recency | 15 | 3777.000 ± 121.885 | 0.787 ± 0.025 | 0.680 ± 0.029 | 0.273 ± 0.007 | 113.000 ± 2.944 | 36.754 ± 1.147 | 7.000 ± 2.944 | 0.000 ± 0.000 | 770.531 ± 44.470 | 0.225 ± 0.012 |
| full | 15 | 3704.000 ± 120.535 | 0.772 ± 0.025 | 0.565 ± 0.060 | 0.293 ± 0.012 | 101.667 ± 2.357 | 35.372 ± 0.902 | 18.333 ± 2.357 | 0.000 ± 0.000 | 636.273 ± 5.613 | 0.302 ± 0.040 |
| full_reflect | 15 | 3931.000 ± 117.340 | 0.819 ± 0.024 | 0.574 ± 0.037 | 0.305 ± 0.010 | 104.000 ± 4.243 | 35.355 ± 0.807 | 16.000 ± 4.243 | 0.000 ± 0.000 | 712.888 ± 32.670 | 0.306 ± 0.015 |

## 2. 配对差值（treatment − control，同 seed 一一配对；效应量）

| 比较 | 指标 | mean_delta | relative | wins/losses/ties |
|---|---|---|---|---|
| recency_vs_off | eat_rate | +0.1176 | +17.58% | 3/0/0 |
| recency_vs_off | satiation_mean | -0.0006 | -0.08% | 1/2/0 |
| recency_vs_off | population | +0.0000 | +0.00% | 2/1/0 |
| recency_vs_off | total_reward | +128.9927 | +20.11% | 3/0/0 |
| recency_vs_off | gini_hunger | +0.0088 | +4.06% | 3/0/0 |
| full_vs_off | eat_rate | +0.1024 | +15.31% | 3/0/0 |
| full_vs_off | satiation_mean | -0.1156 | -17.00% | 0/3/0 |
| full_vs_off | population | -11.3333 | -10.03% | 0/3/0 |
| full_vs_off | total_reward | -5.2650 | -0.82% | 1/2/0 |
| full_vs_off | gini_hunger | +0.0857 | +39.71% | 3/0/0 |
| full_reflect_vs_off | eat_rate | +0.1497 | +22.37% | 3/0/0 |
| full_reflect_vs_off | satiation_mean | -0.1067 | -15.68% | 0/3/0 |
| full_reflect_vs_off | population | -9.0000 | -7.96% | 0/3/0 |
| full_reflect_vs_off | total_reward | +71.3500 | +11.12% | 3/0/0 |
| full_reflect_vs_off | gini_hunger | +0.0899 | +41.66% | 3/0/0 |
| full_reflect_vs_full | eat_rate | +0.0473 | +6.13% | 3/0/0 |
| full_reflect_vs_full | satiation_mean | +0.0089 | +1.58% | 2/1/0 |
| full_reflect_vs_full | population | +2.3333 | +2.30% | 2/1/0 |
| full_reflect_vs_full | total_reward | +76.6150 | +12.04% | 3/0/0 |
| full_reflect_vs_full | gini_hunger | +0.0042 | +1.39% | 1/2/0 |
| recency_vs_off_padded | eat_rate | +0.0497 | +6.74% | 3/0/0 |
| recency_vs_off_padded | satiation_mean | -0.0076 | -1.10% | 1/2/0 |
| recency_vs_off_padded | population | -0.3333 | -0.29% | 1/2/0 |
| recency_vs_off_padded | total_reward | +53.1513 | +7.41% | 3/0/0 |
| recency_vs_off_padded | gini_hunger | +0.0024 | +1.08% | 2/1/0 |
| full_vs_off_padded | eat_rate | +0.0344 | +4.67% | 3/0/0 |
| full_vs_off_padded | satiation_mean | -0.1226 | -17.84% | 0/3/0 |
| full_vs_off_padded | population | -11.6667 | -10.29% | 0/3/0 |
| full_vs_off_padded | total_reward | -81.1063 | -11.31% | 0/3/0 |
| full_vs_off_padded | gini_hunger | +0.0793 | +35.71% | 3/0/0 |
| full_reflect_vs_off_padded | eat_rate | +0.0817 | +11.09% | 3/0/0 |
| full_reflect_vs_off_padded | satiation_mean | -0.1137 | -16.54% | 0/3/0 |
| full_reflect_vs_off_padded | population | -9.3333 | -8.24% | 0/3/0 |
| full_reflect_vs_off_padded | total_reward | -4.4913 | -0.63% | 2/1/0 |
| full_reflect_vs_off_padded | gini_hunger | +0.0835 | +37.60% | 3/0/0 |

### 2b. obs_dim 混淆控制解读

off 臂 obs_dim=9，记忆臂 obs_dim=15；NeuralSwarmPolicy 的 W1 按 1/sqrt(obs_dim) 缩放初始化，观测维数本身就会改变初始行为。off_padded 注册 6 维恒零观测（obs_dim=15，不记录/不检索/不反思），把这条混淆路径单独控制。判断记忆信息收益应以 full vs off_padded 为准。

- recency_vs_off_padded：eat_rate Δ=+0.0497（W/L/T=3/0/0）；satiation_mean Δ=-0.0076（W/L/T=1/2/0）；population Δ=-0.3333（W/L/T=1/2/0）；total_reward Δ=+53.1513（W/L/T=3/0/0）；gini_hunger Δ=+0.0024（W/L/T=2/1/0）
- full_vs_off：eat_rate Δ=+0.1024（W/L/T=3/0/0）；satiation_mean Δ=-0.1156（W/L/T=0/3/0）；population Δ=-11.3333（W/L/T=0/3/0）；total_reward Δ=-5.2650（W/L/T=1/2/0）；gini_hunger Δ=+0.0857（W/L/T=3/0/0）
- full_vs_off_padded：eat_rate Δ=+0.0344（W/L/T=3/0/0）；satiation_mean Δ=-0.1226（W/L/T=0/3/0）；population Δ=-11.6667（W/L/T=0/3/0）；total_reward Δ=-81.1063（W/L/T=0/3/0）；gini_hunger Δ=+0.0793（W/L/T=3/0/0）
- full_reflect_vs_off_padded：eat_rate Δ=+0.0817（W/L/T=3/0/0）；satiation_mean Δ=-0.1137（W/L/T=0/3/0）；population Δ=-9.3333（W/L/T=0/3/0）；total_reward Δ=-4.4913（W/L/T=2/1/0）；gini_hunger Δ=+0.0835（W/L/T=3/0/0）
- full_reflect_vs_full：eat_rate Δ=+0.0473（W/L/T=3/0/0）；satiation_mean Δ=+0.0089（W/L/T=2/1/0）；population Δ=+2.3333（W/L/T=2/1/0）；total_reward Δ=+76.6150（W/L/T=3/0/0）；gini_hunger Δ=+0.0042（W/L/T=1/2/0）

注意：eat_rate 是“进食事件频率”（微吃也计数），total_reward 才是净摄入代理；若 eat_rate 上升而 total_reward/satiation/population 不升，则只能说记忆改变了行为频率，不能宣称找食效率或福利收益。福利方向按实质阈值判定（satiation 0.02、population 2、total_reward 10、gini 0.02），低于阈值视为无实质变化。

## 3. 记忆机制活跃度（评估回合均值，每臂 n=3）

| 臂 | updates | events_recorded | retrievals | reflections | insights | mean_memories | obs_food_dir_abs_mean | obs_food_dir_nonzero_frac | obs_success_mean | obs_danger_mean | obs_reflect_food_mean | obs_reflect_risk_mean | reindexed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | — | — | — | — | — | — | — | — | — | — | — | — | — |
| off_padded | 800.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| recency | 800.000 | 3408.000 | 95472.333 | 0.000 | 0.000 | 22.726 | 0.470 | 0.941 | 0.560 | 0.031 | 0.000 | 0.000 | 7.000 |
| full | 800.000 | 3331.333 | 94294.667 | 0.000 | 0.000 | 22.762 | 0.482 | 0.964 | 0.906 | 0.026 | 0.000 | 0.000 | 16.667 |
| full_reflect | 800.000 | 3687.333 | 94432.667 | 2.000 | 204.000 | 23.105 | 0.541 | 1.000 | 0.706 | 0.026 | 0.783 | 0.000 | 15.000 |

## 4. 逐 seed 评估（每臂 n=3，主指标 + 存活）

| 臂 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| off | eat_rate=0.606, sat=0.662, pop=110, starv=10 | eat_rate=0.691, sat=0.720, pop=112, starv=8 | eat_rate=0.710, sat=0.658, pop=117, starv=3 |
| off_padded | eat_rate=0.700, sat=0.656, pop=113, starv=7 | eat_rate=0.757, sat=0.743, pop=113, starv=7 | eat_rate=0.755, sat=0.662, pop=114, starv=6 |
| recency | eat_rate=0.753, sat=0.700, pop=112, starv=8 | eat_rate=0.793, sat=0.701, pop=117, starv=3 | eat_rate=0.814, sat=0.639, pop=110, starv=10 |
| full | eat_rate=0.745, sat=0.539, pop=105, starv=15 | eat_rate=0.765, sat=0.648, pop=100, starv=20 | eat_rate=0.805, sat=0.507, pop=100, starv=20 |
| full_reflect | eat_rate=0.788, sat=0.574, pop=107, starv=13 | eat_rate=0.821, sat=0.619, pop=98, starv=22 | eat_rate=0.848, sat=0.527, pop=107, starv=13 |

## 5. 结论（honest verdict）

记忆效应依赖检索权重，不能判为整体净收益（配对 n=3，仅效应量+胜负计数，不做显著性声称）：所有记忆臂 eat_rate 都高于 off_padded，但福利/结局方向分裂——recency_vs_off_padded（eat_rate +6.7%，W/L=3/0；福利改善：satiation_mean Δ=-0.0076（W/L=1/2），population Δ=-0.3333（W/L=1/2），total_reward Δ=+53.1513（W/L=3/0），gini_hunger Δ=+0.0024（越低越好，W/L=2/1））；full_reflect_vs_off_padded（eat_rate +11.1%，W/L=3/0；福利恶化：satiation_mean Δ=-0.1137（W/L=0/3），population Δ=-9.3333（W/L=0/3），total_reward Δ=-4.4913（W/L=2/1），gini_hunger Δ=+0.0835（越低越好，W/L=3/0））。相对 off 的原始比较：recency_vs_off: eat_rate Δ=+0.118（rel=+17.6%，W/L=3/0）；sat Δ=-0.001（W/L=1/2）；full_vs_off: eat_rate Δ=+0.102（rel=+15.3%，W/L=3/0）；sat Δ=-0.116（W/L=0/3）；full_reflect_vs_off: eat_rate Δ=+0.150（rel=+22.4%，W/L=3/0）；sat Δ=-0.107（W/L=0/3）。也就是说：recency-only 在 obs_dim 对照下有干净的净摄入收益；full(1/1/1) 与 full_reflect 则把策略推向'高频进食/局部抢占'，总摄入不升（甚至下降）、satiation/population 变差、饥饿基尼上升。记忆的价值取决于权重设计，而不是'开了记忆就更聪明'。最佳记忆臂：full_reflect。 反思相对 full 有可检出增量，福利多数指标改善：full_reflect vs full: eat_rate Δ=+0.047（rel=+6.1%，W/L=3/0）；sat Δ=+0.009（W/L=2/1）；福利：satiation_mean Δ=+0.0089（W/L=2/1），population Δ=+2.3333（W/L=2/1），total_reward Δ=+76.6150（W/L=3/0），gini_hunger Δ=+0.0042（越低越好，W/L=1/2）。

**诊断**：记忆机制确实在运行：full_reflect 臂评估回合平均 3687 条事件、2.0 次反思、204.0 条洞察、23.1 条/个体记忆。记忆方向特征并非全零（|food_dir| 均值=0.541，非零占比=100.0%，成功率均值=0.71，危险率均值=0.03，反思风险偏置=0.00：机制有信号，问题在策略能否兑现）。指标口径提示：eat_events/eat_rate 计的是“该帧吃到东西的个体数”（微吃也计数），是行为频率指标；total_reward（累计进食量-时间成本）才是净摄入代理，satiation/population/starvation/gini 是福利/结局。最强臂 vs off_padded 的 total_reward Δ=-4.49（W/L=2/1）、satiation Δ=-0.114、population Δ=-9.3；与 eat_rate 反向/未升 → 在 obs_dim 对照下记忆没有带来净摄入收益。 饥饿基尼：off=0.216、off_padded=0.222、recency=0.225、full=0.302、full_reflect=0.306——记忆特征（full/full_reflect）明显加剧饥饿不平等，与 population 下降、starvation_deaths 上升一致；反思提高了 total_reward/population，但没有降低不平等。分权重看（off_padded 对照，n=3）：recency-only 的 total_reward Δ=+53.15（W/L=3/0）、sat Δ=-0.008、pop Δ=-0.33、gini Δ=+0.0024；full 的 total_reward Δ=-81.11（W/L=0/3）、sat Δ=-0.123、pop Δ=-11.67、gini Δ=+0.0793。即控制 obs_dim 后只有 recency-only 保留了净摄入收益（且 gini 几乎不变），全权重记忆与反思版都没有。

**诚实性说明**：
- 配对设计：每个臂用相同 train_seeds=[1000, 1001, 1002]、相同 policy seed=7 从同初始化训练；评估用相同 eval seeds=[0, 1, 2]（同地图/同初始人口），每臂 n=3。
- obs_dim 混淆控制：off 臂 obs_dim=9，记忆臂 obs_dim=15，W1 初始化尺度按 1/sqrt(obs_dim) 缩放——off_padded 用 6 维恒零观测把这条混淆路径单独控制，因此判断记忆信息收益应以 full vs off_padded 为准（full vs off 只作参考）。
- n 很小，本报告只给效应量（mean_delta / relative）与 wins-loses-ties，不做 p 值或显著性声称；结论不外推到其他预算/世界。
- 福利方向判定使用实质阈值（satiation 0.02、population 2 个个体、total_reward 10、gini 0.02）：|mean_delta| 低于阈值视为无实质变化，避免把 n=3 的微小抖动当成福利效应。
- eat_events/eat_rate 是行为频率指标（该帧吃到东西的个体数，微吃也计数），不等于摄入量；total_reward 才是净摄入代理。只凭 eat_rate 上升不能宣称“记忆有用”——必须同时看 total_reward、satiation、population、starvation。
- 记忆特征只来自已存事件 + 作为检索 query 的当前感知，无硬编码答案特征。
- 事件节流：小于一口 5% 的微吃不入记忆；濒死仅在进入 <0.2 时记录一次；休息事件每 5 世界秒最多一条。这些只影响采样率，不影响检索/反思逻辑。
- off_padded 是空记忆对照（NullMemoryLayer）：走 memory.enabled 构造路径但观测恒 0、不记录事件、不检索、不反思，与 off 的唯一差别是 obs_dim=15。

## 6. 产物

- summary.json（完整数字与本文件同目录）
- report.md（本文件）
- assets/zoo/p6-memory（最佳记忆臂 full_reflect，tag p6-memory）
