# experiments/records — 最终实验归档

这里保存**跑过并写进文档**的实验产物（config 快照 + summary + report），
与 `experiments/results/` 的区别是：results/ 是工作目录（被 .gitignore
忽略，可随时重跑覆盖），records/ 是入库的最终版本。

| 目录 | 协议 | 说明 |
|---|---|---|
| `p1_scarce/` | `learning_compare` | P1 复核：trained/untrained/random/rule 四臂 + 密度梯度 |
| `p3_social/` | `social_ablation` | 交换/教学/继承 五配置配对消融 |
| `p4_evolution/` | `evolution_compare` | rule / 技能分布演化 / REINFORCE 学习 三臂 |
| `p5_scaling/` | `scaling_sweep` | 人口→吞吐、算力→表现 两条单维幂律拟合 |
| `p6_communication/` | `communication` | 指称博弈（正结果）+ 世界内通道 off/on（null） |
| `p6_memory/` | `memory_ablation` | off / off_padded / recency / full / full_reflect |

每个目录的 `config.yaml` 是产生该结果的完整配置快照（含 `_base`
展开），`summary.json` 是机器可读指标与结论，`report.md`（若有）是
人读报告。重跑：

    python -m newworld --config configs/<name>.yaml

注意：重跑有随机性（训练/评估 seed 相同则分布可比，权重不会逐位相同）；
文档中的数字以本目录归档的 summary 为准。模型在 `assets/zoo/`。
