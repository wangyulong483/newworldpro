# 研究架构审查：从上帝模拟器到可训练的人工生命平台

> P0 交付物。全部结论基于源码证据（file:line）与已提交实验数据
> （experiments/results/*.json），不从模块名推断功能。
> 调研纪要见 `RESEARCH_REFERENCES.md`；路线图见 `RESEARCH_ROADMAP.md`。

---

## 1. 审查范围与当前状态

仓库包含两条世界线：

| 世界线 | 入口 | 模块 | 状态 |
|---|---|---|---|
| 旧世界（侧视文明） | `python -m newworld --classic` | population/goals/knowledge/memory/neural/highlevel/lowlevel_rl/… | 525 测试在跑，V2 报告完整 |
| 新世界（俯视地图） | `python -m newworld`（默认） | map_engine/swarm/catastrophes/map_render/app_map | 19 测试，V3 报告 |

本审查回答提案 §2.2 的 12 个问题。

## 2. 核心诊断（带证据）

### 2.1 新世界个体没有决策能力（问题 1/2/3）

**个体状态**：`swarm.py:78-88` 的 `SWARM_DTYPE` 只有 7 列
（x/y/hunger/age/breed_cd/tribe/alive）——无记忆、无技能、无知识、
无目标字段。

**行为来源全部是硬编码数值规则 + numpy 随机抽样**：

* 进食：所在 cell 扣 `bite_size`（swarm.py:252-263）——规则；
* 移动：4 方向食物探测 × 随机权重 + 高斯抖动（swarm.py:265-306）——规则+随机；
* 繁衍：年龄/冷却/饱足阈值 + `rng.random < p*dt` 概率抽样（swarm.py:311-322）——规则+随机；
* 死亡：饥饿/老龄危险率抽样/水/火（swarm.py:222-236）——规则+随机。

**结论：V3 不存在任何策略学习。** 无 attach/hook、无转移记录、
无奖励定义；BC/RL 基础设施全部悬空在旧世界。V3 报告如实声明
了"个体是数据不是对象"的设计取舍（UPGRADE_V3_REPORT.md §9.2），
但按研究平台标准，这正是提案禁止的"没有独立决策能力的粒子"。

### 2.2 旧世界学习闭环真实但被结构性边缘化（问题 3/4）

学习链路完整存在且已修复：

* 低层 REINFORCE：训练器 minibatch/优势标准化/梯度裁剪已提交
  （lowlevel_rl.py:596-599，V2.1 修复使 Δw 从 1e-4 升至 3e-3，+100 倍）；
* 奖励：rewards.py 按真实世界前后差值（128-164），反刷分测试覆盖；
* 转移：population.py:487 采样登记 → :501-508 每帧结算 → 死亡 :345-351。

**但决策占比失衡**（experiments/results/goal_policy_ablation.json，
trained 条件 10 seed 均值）：

| 决策来源 | 帧占比 |
|---|---|
| 规则教师 teacher | 63.3% |
| 神经网络（BC） | 34.2% |
| 规则规划器 goal | 1.3% |
| **学习型高层策略 policy_goal** | **0.36%** |
| 基础设施反射 | 0.5% |

学习策略只在"agent 无活跃目标时"被询问（population.py:672），
关键动作份额 <1%。**这解释了 V2 报告"学习策略无文明级收益"的
空结果——不是训练不足，是被测面太小。**

### 2.3 知识系统是固定目录（问题 8）

`RECIPE_CATALOG` 是模块级固定 7 条（knowledge.py:388-452：
stone_axe/fiber_rope/basket/campfire/well/hut/granary）。
`KnowledgeStore(catalog=...)` 可注入自定义目录（:499-515）——
**可扩展但默认固定**。"发现"路径是假设→验证状态机
（hypothesize :620-642 登记 + attempt 交 simulate_recipe 验证），
不是学习——是规则的随机触发。

### 2.4 无社会交互载体（问题 6/7）

新世界 `tribe` 字段恒 0（swarm.py:21-22），无传播/继承/交易机制。
旧世界有真实的社会学习（population._social_learning，170px 范围
教一条知识/次）与继承（note_birth 转移已验证知识+记忆），
但两者都服务 8 人小世界，且不改变个体策略。

### 2.5 实验基础设施是强项（问题 10/11）

* 多 seed 对照：RunSpec/SocietyRunner 完整（simulation.py:56-99、
  :357-428），v2_final 已有 6 seed × 7 配置的 aggregate.json
  （含 mean/std/min/max/n）；
* telemetry 19 指标、duck-typed 世界无关（telemetry.py:147-169、:204）；
* **UI 瓶颈证据**：app_map.py:337 是 `canvas.delete("agent")` 后全量
  逐个体 create_rectangle（:343-349），map_render.py:10-11 注释声称
  的 diff 更新**未实现**；>6000 个体隔点渲染（:341-342）。瓶颈在
  渲染路径（Tk Canvas 全量重建），不在仿真（8000 个体 1.9ms/帧）。

### 2.6 训练与仿真解耦（问题 9）

已解耦：训练器通过 env_factory 拿环境（lowlevel_rl.py:662-679），
策略挂载/卸载幂等（population.py:453-461）。这个模式可直接推广。

## 3. 学习闭环缺口清单（按提案 §1.1 的目标）

| 缺口 | 现状 | P1 计划 |
|---|---|---|
| V3 无策略接口 | swarm 行为内联硬编码 | 抽出可注入 SwarmPolicy |
| V3 无奖励/转移 | 无 | 个体级奖励 + drain 缓冲 |
| 学习信号稀疏 | 高层策略 0.36% 帧 | P1 全部动作帧由策略控制 |
| 信用分配过长 | V2 奖励=文明级 Δ（分钟跨度） | P1 奖励=个体级秒级事件 |
| 个体无记忆/技能 | 7 个数值字段 | P2+ 加（P1 先验证觅食学习） |
| 传播/继承载体缺失 | tribe 恒 0 | P3 |
| 渲染挡路 | 全量重建、无 diff | P2 换浏览器平台 |

## 4. 可复用资产（迁移建议）

**直接复用**：telemetry 19 指标框架 / RunSpec+JSONL /
lowlevel_rl 训练器全套（minibatch+标准化+裁剪）/ rewards 反刷分
设计思想 / knowledge 假设-验证状态机 / Catastrophes 圆形区域干预
（降级为实验干预）/ MapEngine 确定性地形 + food 网格 / Swarm
向量化骨架（SoA dtype + 压实）。

**小改复用**：Swarm 行为段抽策略接口；RECIPE_CATALOG 移入配置。

**退役候选**：app_map 全量重建路径（P2 换浏览器前端后退役）、
probe_* 旧诊断脚本（已被 run_v2_experiments 取代）。

**不迁移**：app_tk（旧世界专用，随旧世界保留）；Avida/Mesa 代码
（见调研报告——只借机制不引代码）。

## 5. 对 §2.3 实验审查三问的回答

1. **训练目标过简单？** 是——高层 8 目标分类 + 奖励是文明级 Δ，
   个体动作与奖励间隔分钟级。
2. **奖励有捷径/错误激励？** rewards.py 反刷分测试覆盖了站立/挂机/
   重复失败，但"协作分 0.1 × share_food 增量"可被两人对刷
   （未测试对刷场景——记录为已知风险）。
3. **训练/评估环境一致？** 一致（同 factory），但评估 seed 数太少
   （goal_policy_ablation 用 10 seed，泛化检验不足）。

P1 的奖励设计直接吸取：(a) 个体级、(b) 秒级信用、(c) 对刷场景
进测试。

## 6. 结论

现有代码库的**世界规则引擎、实验框架、RL 训练器**三块地基质量
高且可复用；缺的是把三者接到同一条世界线上——V3 世界有引擎没学习，
旧世界有学习没世界规模。P1 的任务不是新建，而是**缝合**：给
Swarm 装策略接口，复用 lowlevel_rl 训练器，在 MapEngine 的
确定性地形上做配对对照实验。
