# 技术调研纪要：人工生命 / 人工社会 / MARL / 实验平台

> P0 交付物。所有链接均已在线核验（2026-10）。
> 结论服务于本项目：Python + numpy 纯手写网络（无 PyTorch）、
> Windows 开发环境、千级智能体目标。

---

## 1. 人工生命与数字演化

### Avida（https://github.com/devosoft/avida，活跃，LGPL-3.0+）

**解决什么**：数字有机体为自复制程序（环形基因组指令集），复制
发生点变异，完成逻辑任务换取 CPU 时间配额（选择压力），有成熟的
fitness landscape 测量。是"开放式演化"研究的标准平台。

**适合本项目的思想**：
* **能量配额式选择**——个体生存预算与其"贡献"挂钩（我们已有
  hunger 代谢，可扩展为"觅食产出换生存预算"的选择压力）；
* **变异算子**（拷贝错误率）——P4 演化阶段的参数扰动设计参考；
* **基因型距离多样性指标**——种群策略多样性测量。

**不适合**：基因组=可执行代码的范式与本项目"个体=策略网络参数"
根本不同；C++/CMake 工程嵌入纯 Python 库成本约 2-4 人月，不值得。

**结论：不引入代码，仅机制借鉴。**

### Tierra（http://life.ou.edu/——已超时不可达，视为历史项目）

仅历史意义，无活跃维护，不采纳。

## 2. 人工社会与复杂系统

### Sugarscape（Epstein & Axtell《Growing Artificial Societies》1996）

**解决什么**：糖资源梯度 + 个体视野 + 代谢 + 死亡继承四规则，
被证明产生**财富分化、迁徙波、承载容量涌现**——与本项目
swarm 的食物梯度+代谢结构同构。

**可运行转化**：Mesa 官方示例 sugarscape_g1mt 复现了其第 4 章
"交易者"规则并得到去中心化价格均衡
（https://mesa.readthedocs.io/latest/examples.html）——这是
P3 交易机制的现成现象学基线。

### Axelrod 合作与规范（Axelrod-Python：https://github.com/Axelrod-Python/Axelrod）

200+ 策略、锦标赛、Moran 演化、道德度量。**注意**：GitHub 许可证
未自动分类，引入前需人工核验（风险已知）。P3 的声誉/合作机制
先参考其模型结构，不急于引依赖。

### Mesa（https://github.com/projectmesa/mesa，Apache-2.0）

**采用其接口约定，不整体迁移**：
* `DataCollector`（model_reporters/agent_reporters/tables →
  pandas 导出，https://mesa.readthedocs.io/latest/apis/datacollection.html）
  的**接口设计**直接抄给我们的 telemetry 扩展；
* 其 Time/Scheduler 已废弃（3.1 移除）、`batch_run` 已不在 main
  分支——参数扫描需自建，**不要依赖 Mesa 跑实验**。

## 3. 多智能体强化学习

**勘误**（调研中发现的常见编号错误，已核验）：
MAPPO 正确论文是 https://arxiv.org/abs/2103.01955（Yu et al.,
NeurIPS 2022）——**不是** 2103.17119（那是自动驾驶论文）；
QMIX 正确编号 https://arxiv.org/abs/1803.11485（Rashid et al.,
ICML 2018）——**不是** 1706.05296（那是 VDN, Sunehag et al.）。

### PettingZoo（https://pettingzoo.farama.org/，MIT）

AEC 接口明确支持"agents can be added or removed"
（https://pettingzoo.farama.org/api/aec/）——**生灭生命周期是
一等公民**（AEC 循环 + `action=None` 模式）。本项目的
`drain_transitions`/`attach_policy` 模式与其对齐成本低。

### 范式判断（本项目约束下）

| 范式 | 千级个体 | 纯 numpy | 生灭支持 | 判断 |
|---|---|---|---|---|
| IQL + 参数共享 | ✓（一个网络批量前向） | ✓ | ✓（个体只是索引） | **采用** |
| QMIX/MAPPO（CTDE） | ✗（每步集中前向/反向 O(N)） | ✗（需 PyTorch） | 部分 | 不可行 |
| 独立网络每个体 | ✗（内存/算力爆炸） | ✗ | ✓ | 不可行 |

**开放式演化参照**：AlphaStar 联赛三角色+PFSP
（https://www.nature.com/articles/s41586-019-1724-z）、
PSRO_rN 博弈论 niching（https://arxiv.org/abs/1901.08106）——
P4 若做"历史最优个体档案"（防策略遗忘）可借鉴其 exploiter 思想。

## 4. 可视化与实验平台

| 维度 | 自建 FastAPI+PixiJS/React | Mesa 生态（Solara） | Godot 4 |
|---|---|---|---|
| 千级散点渲染 | 强（WebGL/WebGPU 批渲染，https://pixijs.com/，MIT） | 弱（Solara/Matplotlib 路径） | 强 |
| 事件回放 | 自定义 WS 帧协议，完全可控 | 无内置 | 需自定 IPC |
| Python 训练集成 | 原生（同进程/localhost） | 原生 | **硬伤**：官方仅 GDExtension，无 Python/JSON-RPC 支持 |
| 实施成本 | ~1-2 人月 | 1-2 周 | 中高（双语言+绑定） |
| 许可证 | MIT（PixiJS）+ 自有 | Apache-2.0 | MIT |

**决策（用户已选）**：自建 FastAPI+WebSocket+PixiJS/React 主线，
Mesa/Solara 仅作快速原型参照，Godot 排除。

## 5. 采纳清单（汇总）

| 来源 | 采纳 | 形式 |
|---|---|---|
| PettingZoo | 环境接口约定 | P1 的 obs/act/drain 命名对齐 |
| Mesa | DataCollector 接口设计 | telemetry 扩展参考 |
| Sugarscape | 交易/财富分化基线 | P3 现象学对照 |
| Avida | 变异算子/选择压力/多样性 | P4 机制设计 |
| V2 教训 | 个体级秒级奖励 | P1 奖励设计（见 REVIEW §5） |
| 自建栈 | FastAPI+PixiJS | P2 实现 |

**全部不引入**：Avida/Tierra/Mesa/Axelrod 的代码依赖（许可证与
集成成本见各节）。
