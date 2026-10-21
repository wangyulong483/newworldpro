# P5 标度律扫描报告 — p5_scaling

- 协议：scaling_sweep（单维扫描 + log-log 幂律拟合）
- 配置：p5_scaling，config hash 7681ad28ec342e26
- 总墙钟：224.6 s
- 产物目录：C:\Users\27826\Desktop\vscode_win_ws\ds_ws\newworld\newworldpro\experiments\results\p5_scaling
- 复现：python -m newworld --config configs/p5_scaling.yaml

## 判决

Sweep A：吞吐随人口亚线性下降（b=-0.302, R2=0.926, n=5），固定每步开销与随人口增长的个体成本并存。 Sweep B：性能-算力幂律几乎不解释数据（b=0.008, R2=0.227, n=5）；配对差（neural-rule, eat_events）从 +2582.3 变到 +2643.0（单调递增=False，斜率=+47.4/log算力，R2=0.225）。在 1~16 episodes 的窄区间内无法声称学习标度律。

## Sweep A：人口 → 仿真吞吐（规则策略，其余维度冻结）

固定：地图 256×160（seed 0）、food_regrow=0.0005/s、dt=0.05、规则策略、每点 20.0s 世界时间；吞吐 = engine steps / wall_s。

| 初始 N | 平均人口 | 末人口 | steps | wall (s) | 吞吐 (steps/s) | step (ms) | µs/agent-step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 68.25 | 97 | 400 | 0.3998 | 1001 | 0.999 | 14.6 |
| 120 | 130.7 | 158 | 400 | 0.4456 | 897.6 | 1.11 | 8.53 |
| 240 | 240 | 240 | 400 | 0.4984 | 802.6 | 1.25 | 5.19 |
| 480 | 480 | 480 | 400 | 0.6491 | 616.3 | 1.62 | 3.38 |
| 960 | 960 | 960 | 400 | 0.9432 | 424.1 | 2.36 | 2.46 |

**拟合**：throughput = a · N^b，a=3747, b=-0.3019, R²(log)=0.9262, R²(linear)=0.9108, n=5, b_stderr=0.04922, 残差 RMS(log)=0.08357。

逐点残差（log 空间）：N=60: -0.08413，N=120: 0.01654，N=240: 0.1139，N=480: 0.05902，N=960: -0.1054

## Sweep B：学习算力 → 觅食性能（神经策略）

固定：初始人口 120、9 维观测、隐藏层 32、lr=0.02、训练 seed [1000, 1001]、评估 seed [0, 1, 2] × 30.0s；每档 episodes 从零训练，算力 C = episodes × episode_seconds × 人口（agent·秒）。

| episodes | 算力 C (agent·s) | 训练 wall (s) | 平均 eat_events | std | eat_rate | 配对差 vs 规则 | wins/n |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3600 | 6.368 | 5949 | 395 | 1.65 | 2582 | 3/3 |
| 2 | 7200 | 12.71 | 6012 | 437 | 1.67 | 2646 | 3/3 |
| 4 | 1.44e+04 | 23.4 | 6133 | 548 | 1.7 | 2766 | 3/3 |
| 8 | 2.88e+04 | 49.03 | 6220 | 522 | 1.73 | 2853 | 3/3 |
| 16 | 5.76e+04 | 116.5 | 6009 | 437 | 1.67 | 2643 | 3/3 |

规则基线（同 seed 同窗口）：eat_events 均值 3366.3 ± 136（n=3），eat_rate 0.9351。

**拟合**：performance = a · C^b，a=5626, b=0.007819, R²(log)=0.2271, R²(linear)=0.2233, n=5, b_stderr=0.008327。

**配对差趋势**（neural − rule, eat_events）：首档 2582.3 → 末档 2643，单调递增=否，斜率 47.42 eat_events / log(算力)，R²=0.2255。

**模型存档**：p5_scaling_c16 → C:\Users\27826\Desktop\vscode_win_ws\ds_ws\newworld\newworldpro\assets\zoo\p5_scaling_c16（tag=['p5', 'scaling_sweep'], sha256 校验=是）。

## 这说明什么 / 不说明什么

**说明**：

- 在这个固定世界与实现下，人口从 60 到 960 时单机吞吐如何变化（上表与 Sweep A 拟合）；
- 在 1~16 个 30.0s 训练回合的窄算力区间内，REINFORCE 训练出的神经策略相对规则基线的觅食表现如何变化（Sweep B 表与配对差）；
- 两个轴的幂律拟合参数 b、R² 与逐点残差都如实列出，便于判断幂律是否只是事后叙事。

**不说明（限制）**：

- 不是普适标度律：每个轴只有 5 个点，且只扫一个维度；地图 seed、世界参数、网络结构、优化超参全部冻结，换地图/机制后 b 可能完全改变；
- Sweep A 的吞吐是当前纯 numpy 单进程实现的墙钟量，含指标计算、生灭回收、观测组装等固定开销，不能当作理论复杂度或跨语言结论；
- Sweep B 每档策略从零训练，训练 seed 仅 1~2 个、评估仅 3 个 seed × 30s，方差大；窄算力区间内学习曲线常呈饱和/波动，外推指数没有依据；
- Sweep B 在窄区间内已饱和：8 episodes 处 eat_events 均值最高（6219.7），16 episodes 回落到 6009.3（非单调），但每一档都显著高于规则基线（3366.3）；b=0.00782、R2=0.227 只表示“已饱和/噪声主导”，不能外推；
- Sweep A 的墙钟吞吐对机器负载敏感：同一配置另一次完整运行得到 b=-0.317、R2=0.955（本报告为 b=-0.302、R2=0.926），单次运行的 b 末位不宜精确解读；
- eat_events 是离散进食事件，受地图食物承载力与初始人口影响，不能直接等同“智能”或“适应度”；eat_rate 已按初始人口归一，但口径仍是本项目的局部指标；
- 因此本报告只给“该配置下两个单维扫描的描述性幂律”，不是“scaling law 论文复现”。

## 复现

    python -m newworld --config configs/p5_scaling.yaml

产物：summary.json（机器可读，含 sweeps/fit/points）、config.yaml（配置快照）、
report.md（本文件）、模型 zoo 目录 C:\Users\27826\Desktop\vscode_win_ws\ds_ws\newworld\newworldpro\assets\zoo\p5_scaling_c16。

