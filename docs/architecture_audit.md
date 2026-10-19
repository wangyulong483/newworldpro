# 现有项目架构审计报告（P0）

- 审计对象：`C:\Users\27826\Desktop\vscode_win_ws\ds_ws\newworld\newworldpro`（`newworld` v0.8.0）
- 审计方式：只读源码 + 运行基线探针（`experiments/probe_baseline.py`、`experiments/probe_cap.py`）
- 审计时点：改造前，git 基线提交 `926af6e`（"baseline: pre-civilization snapshot"）
- 语言/依赖：Python 3.14，numpy + pillow（运行时），tkinter（GUI）；无 torch/gymnasium
- 代码规模：`newworld/` 39 个模块，项目 105 个文件；测试 111 个用例

**结论摘要（先看这里）**

1. 本项目**没有强化学习算法**。所谓"RL 神经网络"实际是**监督式行为克隆（BC）**：
   用 24 维手工特征 → 64 → 32 → 58 的 MLP（共 5594 个参数）模仿手写稳态状态机，
   没有奖励、没有策略梯度、没有价值函数、没有回合回报。
2. `env.py` 里的 Gymnasium 风格环境（8 维观测 / 3 动作 / 4 种奖励函数）**从未被主程序使用**，
   只出现在单元测试和 `examples/hello_env.py` 中。
3. 主循环 `Population.step` 中 **58% 的决策由手写的 `SocialLife` 教师规则接管**，
   神经网络只负责其余基础行为——"自主性"主要来自硬编码规则。
4. 人口上限 `MAX_POPULATION = 8` 约束的是**当前存活数**；死亡时个体从 `world.agents`
   和 `population.minds` 中被**物理删除**，名额随即释放。实测的一次完整生命周期里，
   累计创建数恰好到达 8（没有突破），但那只是因为幸存者在死亡前已饿到无法再生育；
   **代码路径允许"死亡释放名额 → 再次生育"突破 8**（§6.2 用受控测试证明）。
   硬约束必须改成"历史累计创建数"，这正是 `AgentRegistry` 的职责。
5. 系统**没有**目标/计划数据结构、没有知识库、没有技术依赖图、没有个体记忆、
   没有死亡档案、没有知识传承；技术层只有一条硬编码配方（2 木 + 1 石 → 斧），
   公共设施只有屏幕中央固定的一个"社区站点"。
6. 实测（决定性发现）：2 个初始个体在 **~100 世界秒**内出生 6 人并建成社区，
   此后**再无任何积累**；社区建成后手写教师 `SocialLife` 的求偶分支接管所有成年个体，
   **连进食都不再执行**，全员能量单调下降到 0，**t=537.55s 全族饿死灭绝**。
   整段生命周期中 `social` 状态占 82.7% 的个体系-步（55826/67469），
   个体 0 有 2318/2400 步、个体 1 有 2229/2400 步处于同一行为状态，
   最长连续同状态 **2227 步**。这就是"重复简单动作"的量化证据，也是"文明不会持续"的证据。

---

## 1. 当前架构与模块依赖

### 1.1 主程序数据流（GUI / 交互路径，`app_tk.py`）

```
app_tk.run_app
  └─ World(config, seed)                   world.py        纯仿真：几何/物理/生态/声/热
       ├─ PixelScene                       pixelworld.py   512x320 原子网格地形/树/石/浆果/池塘
       ├─ Technology                       technology.py   Inventory / 采集 / 配方 / 社区站点
       ├─ DayCycle / Sun / Air             daynight.py, environment.py
       └─ Character*                       character.py    单个体：运动/体温/库存/生理/社会字段
  └─ Population(world, seed)               population.py   多智能体集成层
       ├─ AgentMind*  (每人一个)           population.py
       │    ├─ NeuralPolicy                neural.py       MLP 24-64-32-58（BC）
       │    ├─ NeuralInnerLife             neural.py       把网络输出翻译成 Decision
       │    ├─ SocialLife                  social.py       手写技能教师（采集/制作/建造/求偶）
       │    └─ Perceiver                   perception.py   视觉/听觉/眼动（每步跑，占总耗时 95%）
       ├─ Society                          society.py      关系图 + 图网络 + 生育/衰老/死亡
       └─ train_round → training.py         behavior cloning（F2 / 后台自动训练）
  └─ render / debug 面板                   render.py, app_tk.py
```

### 1.2 并行但未接入的 RL 路径（`env.py`）

```
NewWorldEnv
  ├─ World（同一世界）
  ├─ 观测：Character.observation() = 8 维（x, vx, facing, 到左墙距离, 到右墙距离,
  │       跳跃高度, vy, on_ground）
  ├─ 动作：3 个离散动作（-1 左 / 0 站 / +1 右），不含跳、不含交互、不含语言
  └─ 奖励：ZeroReward / AliveReward / ExplorationReward / CenterReward（可插拔）
```

**该路径只被 `tests/test_world.py` 与 `examples/hello_env.py` 使用**，`app_tk.py` / `cli.py`
均不引用 `NewWorldEnv`；也没有任何训练循环消费 `step()` 返回的 reward。

### 1.3 依赖关系

- 依赖方向基本分层：`config/art/materials` → `character/physiology` → `world` → `population/app_tk`
- `neural.py` 依赖 `behavior.py`（`Decision`）与 `materials.py`，不依赖 `world`（通过 duck typing）
- `technology.py` 只依赖 `materials.py`（轻量），`society.py` 依赖 `materials.py` + numpy
- **循环依赖风险点**：`world.py` import `technology.py`；`technology.py` 不反向 import，安全。
  本设计新增模块必须遵守同样规则（见 `docs/civilization_design.md` §1）。

---

## 2. 强化学习算法、网络输入与输出（实测确认，非猜测）

### 2.1 算法：**监督式行为克隆，不是 RL**

`training.py` 的实际流程：

1. `collect_dataset()`：用**手写教师** `behavior.InnerLife`（走/觅食/吃/喝/睡）在 7 个起始时刻
   （8/11/14/18/21/2/5 点）× 每时刻 300 步采集 (特征, 行为模式, 移动, 跳, 说话, 词) 数据集；
2. `train_supervised()`：手工反向传播 + momentum 0.9 + lr 0.03 + batch 128，
   交叉熵（模式/移动/语言）+ BCE（跳/说话）多任务损失；
3. `pretrain()` 从零训练；`fine_tune()` 从 checkpoint 继续训练。

**缺失的 RL 要素**：奖励函数（主路径无）、回报/折扣、价值函数、策略梯度、优势估计、
探索策略（ε/熵）、经验回放、终止/截断语义（`terminated` 在 `env.py` 中恒为 `False`）。

### 2.2 网络结构与输入

- 输入 `FEATURE_DIM = 24`：x、vx、facing、on_ground、vy、能量、水分、睡压、警觉、
  昼夜 sin/cos、日照、最近浆果 dx/dy/flag、水池 dx/flag、降雨、土壤湿度、体温、
  跳跃计数、说话计数、时刻、水位。
- 隐藏层：`HIDDEN1 = 64`（tanh）→ `HIDDEN2 = 32`（tanh）。
- 输出 `OUTPUT_DIM = 58 = 5 + 3 + 2 + 3×16`：5 个行为模式（explore/forage/drink/sleep/rest）、
  3 个移动方向、跳/说 2 个辅助位、3 个音节 × 16 词表（原始语言头）。
- 参数量：24×64+64 + 64×32+32 + 32×58+58 = **5594**；`assets/neural_policy.npz` 46674 B 一致。
- **输入里没有任何**：目标、计划、记忆、失败经验、知识、其他 agent 的可观察状态、
  候选行动成本/风险/预期收益、关系上下文。`society.py` 算出的 6 维 `social_context`
  被写入 `character.social_context` 后**没有任何决策路径读取**（只有测试断言长度）。

### 2.3 训练与持久化

- 每 agent 独立权重 `assets/agents/agent_<id>.npz`；社会网络 `assets/society.npz`；
  基础 checkpoint `assets/neural_policy.npz`。
- 子代通过父/母权重交叉 + 0.03 高斯变异继承（`Population._inherit_policy`）——这是**权重遗传**，
  不携带任何经验/知识。
- 加载时只校验 `hidden1`（`NeuralPolicy.load`），**没有输入维度迁移机制**：一旦扩维，
  旧 checkpoint 会因 reshape 失败而不可用（本改造必须补零初始化迁移，见设计 §7）。

---

## 3. 状态空间、动作空间、奖励函数的具体定义

### 3.1 两套互不相通的定义

| 维度 | 主路径（population/neural） | 并行路径（env.py） |
| --- | --- | --- |
| 观测 | 24 维（§2.2） | 8 维（x/vx/facing/墙距/高度/vy/着地） |
| 动作 | `Decision(move∈{-1,0,1}, jump: bool, speak: bool, word)`，外加教师直接调用
`tech.harvest/craft/contribute`、`physiology.eat/drink/sleep/wake` | 3 个离散整数 |
| 交互动作 | 采集/制作/贡献/吃喝睡——**不在动作空间里**，由规则在 `decide()` 内直接执行 | 无 |
| 奖励 | **无** | 4 个可选类，未使用 |
| 回合语义 | 无终止条件（世界永不 end） | `terminated` 恒 False |

### 3.2 基线实测（原始系统，未修改）

**探针 A**（`experiments/results/baseline_probe_pre_seed0.json`）
配置：800×500、dt=0.05、2400 步 = **120 世界秒**、seed=0、2 个初始个体、BC 权重。
运行耗时 54.4 s（44.1 步/s）。

| 指标 | 实测值 |
| --- | --- |
| 出生 / 死亡 / 期末存活 | 4 / 0 / 6 |
| 教师接管 vs 神经自主 | 5153 步 vs 3783 步（**58% 决策来自手写规则**） |
| behavior_state 分布 | explore 42.4%、social 30.9%、rest 24.9%、gather 1.2%、forage 0.58%、craft 0.02%、build 0.02% |
| 最长连续同状态 | **2227 步**（≈111 世界秒） |
| 个体 0 主导状态 | social：**2318 / 2400 步（96.6%）** |
| 个体 1 主导状态 | rest：**2229 / 2400 步（92.9%）** |
| 采集成功 | 10 次（木 6 / 石 4），尝试 17 次（失败 7 次，失败无任何学习或记录） |
| 制作 / 贡献 | 2 次 / 2 次；社区建成 1 次后**再无积累** |
| 每步标量奖励 | **0.0**（主循环无奖励） |
| 目标完成率 | 不适用——**没有目标/计划数据结构** |

**探针 B**（`experiments/results/baseline_long_pre_seed0.json`）
配置：800×500、dt=0.05、seed=0、2 初始个体、关闭感知（见下的行为中性说明）、
最多 40000 步（2000 世界秒），实际在 537.55s 全族灭绝后停止。**实测结果：**

| 指标 | 实测值 |
| --- | --- |
| 历史累计创建 | **8**（t=0,0,32.3,56.3,80.3,104.3,128.3,152.3s） |
| 同时存活峰值 | 8 |
| 社区建成时间 | ≈ t=100s |
| 首次饥饿死亡 | **t=431.15s**（个体 3，能量 0.49） |
| 全族灭绝 | **t=537.55s**（8/8 全部 `starved`） |
| 行为状态-步分布 | social 55826（**82.7%**）、explore 8505、rest 2874、gather 107、forage 52、craft 2、build 2 |
| 能量轨迹 | 全族在社区建成后单调下降：t=100 平均 60 → t=200 平均 41 → t=300 平均 26 → t=400 平均 13 → 灭绝 |

> 行为中性说明：探针关闭了 `Perceiver.perceive` 以提速（4933 步/s vs 44 步/s）。
> 这是**逐比特无影响**的：`NeuralInnerLife.decide(perception=...)` 把 perception
> 传给 `neural.features()`，而该函数**完全忽略** perception 参数；感知结果只用于调试面板。
> 关闭后同一 seed 的决策与动作序列不变（改造中会把这一点做成配置项并加测试）。

**探针 B 的结论比预期更严重**：原始系统的"文明"不是停滞，而是**自我灭绝**——
社区建成后 `SocialLife` 只返回求偶/站立动作，从不返回 `None`，神经网络因此永远得不到
控制权，饥饿的个体不会去吃浆果。系统没有食物储备、没有维护、没有崩溃恢复机制。

**探针 C**（`experiments/probe_cap.py`，粗步长 dt=0.5）
注：dt=0.5 会破坏教师规则的近距离导航，导致**测不到真实生育节奏**（实测累计创建仅 2、
t=534.5s 全族饿死、0 次出生）。因此该探针**不作为人口结论依据**，仅保留为
"时间步长敏感性"的反例证据：行为依赖精细移动，实验 dt 必须固定在 1/20 或更小。

### 3.3 这些数字说明什么

- **重复动作的根因不是网络太小，而是"没有理由做别的"**：社区建成后 `SocialLife.decide`
  的求偶分支（`community.built` → 逼近异性 → 站立）对成年个体**恒成立**，
  而且它从不返回 `None`，于是神经网络被永久静音。这不是探索收敛，
  而是**状态机没有后续目标，同时剥夺了生存行为**。
- **饥饿死亡验证了"闭环缺失"**：能量只出不进（`physiology.update` 每步消耗、
  只有浆果/池塘能补），教师接管后没人吃 → 全员能量 78→0 → 灭绝。
  系统没有食物储备、没有公共库存、没有"低能量优先级高于求偶"的仲裁机制。
- **奖励漏洞不存在，因为根本没有奖励**——但也因此没有任何机制驱动"长期积累"。
  唯一的"进步"信号是手写规则里的硬编码配方。
- **失败不可学习**：`tech.harvest()` 返回 `False` 时（实测 17 次尝试中 7 次失败），
  调用方直接忽略，不进库存、不记日志、不进网络输入、不改变教师策略。

---

## 4. 当前 Agent 为什么会重复简单动作（分层归因）

| 层 | 机制 | 证据 |
| --- | --- | --- |
| 生理层 | 能量/水分只需走到浆果/池塘；`eat()` +26、`drink()` +30，消耗极慢 | physiology.py 0.145/s vs +26/次 |
| 教师层 | 社区建成后 `SocialLife.decide` 的求偶分支恒成立（`community.built` → 逼近异性 → 站立） | social.py:38-59；实测 social/rest 占 56% |
| 神经层 | BC 教师的训练数据里 `explore` 占多数，网络学会"没事就探索"；教师接管时网络输出被完全丢弃 | neural.py:234 后无 fallback 学习 |
| 规划层 | **不存在**：没有目标、没有计划、没有子任务、没有进度 | 全库 grep 无 goal/plan |
| 奖励层 | **不存在**：无信号区分"有效进展"与"原地站立" | §2.1 |
| 记忆层 | **不存在**：失败 7 次采集后行为不变 | §3.2 |
| 社会层 | 只有 affinity 数值积累，不产生合作/分工/交易 | society.py |

**结论**：重复动作是**结构性缺陷**（缺少目标—奖励—记忆闭环），不是超参数问题。
增大网络、换 GRU/LSTM、调学习率都不会改变这一点。

---

## 5. 多步骤规划 / 长期记忆 / 生产闭环 / 死亡丢失 / 知识传承

### 5.1 多步骤规划：**不具备**

- 无 Goal / Plan / PlanStep / 任务依赖 / 前置条件检查 / 进度监测 / 重规划。
- 现有"多步行为"是**隐式写死在 `SocialLife.decide` 里的单线程规则**：
  「材料够 → 去社区 → 制作 → 贡献」；材料不够 → 采集。只有一条路径，
  没有目标选择、没有失败重规划、没有放弃、没有成本估计。
- 无法表达"先学知识、再备料、再制作、再验证"这类依赖链；也没有"检查知识库是否存在配方"这一步
  （配方是代码里的 if，不在数据里）。

### 5.2 长期记忆：**不具备**

- 唯一跨步状态是神经网络权重（每 30 s 后台 BC 训练一次覆盖保存）与社会网络权重。
- 没有情景记忆（观察/动作/结果/失败原因）、没有程序性记忆（如何完成任务）、
  没有公共文明记录。死亡个体的权重文件**保留在磁盘**，但不再被任何活着个体读取，
  也不记录"谁掌握过什么"。

### 5.3 技术与资源能否形成可持续生产闭环：**只能形成一次性闭环**

- 现状链路：采集木/石 → 制作石斧（提高采集 1→2）→ 贡献社区 → 建成小屋+篝火 → **结束**。
- 闭环缺口：
  - 工具只影响采集产量，不影响任何新行动能力；没有工具组合/技术前置依赖图；
  - 没有材料再生约束（树/石原子被移除后不可再生；浆果可再生）——资源稀缺不可控；
  - 没有食物储存、没有公共库存、没有维护成本（建筑永久存在，不损耗）；
  - 没有交易/分工（每个个体都能独立完成全部链路）；
  - 没有"技术改变世界规则"的机制：斧头是唯一效果，且 `craft()` 是硬编码 if。
- 因此"文明"在社区建成后进入**吸收态**（absorbing state），实测 120 s 内到达。

### 5.4 Agent 死亡后会丢失什么：**几乎全部**

死亡路径（`society.py:125-131`）：

```python
if agent.age >= LIFESPAN_S or agent.physiology.energy <= 0.5:
    agent.alive = False
    world.remove_agent(agent.agent_id)      # ← 从 world.agents 永久移除
    self.deaths += 1
    self.events.append(SocietyEvent("death", f"{agent.agent_id}号死亡（{agent.age:.0f}s）"))
```

随后 `Population.sync()`（`population.py:123-128`）执行 `del self.minds[agent_id]`，
销毁 `AgentMind`（神经网络对象、感知器、社会技能状态）。

丢失清单：
- 世界中的身体与可观察状态（位置/库存/工具）→ 立即丢失；
- 神经网络对象（磁盘 npz 仍在，但不再被加载；子代只继承父母权重，不读死者权重）；
- 库存（木/石/工具）→ 丢弃，不进入任何公共库存；
- 没有 `died_at`、没有死亡原因字段（只有一条中文事件字符串）；
- **没有历史注册表**：`world.agents` 里查不到死者，`_next_id` 继续自增，
  无法回答"总共创建过几个个体"——只能从磁盘 `assets/agents/*.npz` 文件数间接猜；
- 没有知识/经验/技能可继承，因为这些东西本身不存在。

### 5.5 是否支持知识传承与代际演化：**不支持**

- 代际只有：代数计数器 `generation`、体型 0.65、父母权重交叉+变异、`parents` 元组、affinity 提升。
- 没有：知识/技能/配方/记忆的继承或教学；没有亲子投资；没有"死亡时把知识移交公共库"；
  没有发现—验证—传播—使用的可追踪链路（因为知识不是数据对象）。

---

## 6. 现有实现与文明演化目标的差距（逐项）

| 目标能力 | 现状 | 差距等级 |
| --- | --- | --- |
| 个体生存与资源管理 | ✅ 有（能量/水分/睡压/浆果/池塘） | 保留 |
| 多步骤目标规划与执行 | ❌ 无 | **新建（P1）** |
| 个体经验记忆与失败利用 | ❌ 无 | **新建（P3）** |
| 工具制作与技术组合 | ⚠️ 仅 1 条硬编码配方，无组合、无前置依赖 | **重写为数据驱动（P2）** |
| 知识发现/验证/保存/传播 | ❌ 无 | **新建（P2/P3）** |
| 合作/模仿/交易/竞争 | ⚠️ 只有 affinity 数值与"靠近"；无交易/教学/分工 | **扩展（P3/P4）** |
| 死亡后知识遗存与代际传承 | ❌ 死亡即删除 | **新建（P3）** |
| 公共设施/共享资源/社会规则 | ⚠️ 单个固定位置社区站点，永久有效，无维护成本 | **扩展（P4）** |
| 环境变化/稀缺/适应 | ⚠️ 昼夜、降雨、水位、土壤湿度有；无干旱/灾害/资源枯竭事件 | **扩展（P4）** |
| 持续发展/停滞/崩溃/恢复 | ❌ 无指标、无状态机；实测进入停滞吸收态 | **新建（P4）** |

### 6.2 人口硬约束实测（任务书 §2.1 的核心违背）

`MAX_POPULATION = 8` 在两个创建点被检查：`society.py:144`（怀孕）与 `society.py:159`（分娩），
判据都是 `len(world.agents) >= MAX_POPULATION`（**当前存活数**）。死亡会立即
`world.remove_agent()`，列表变短 → 名额释放 → **同一份代码随后允许再生育**。

实测（`baseline_long_pre_seed0.json`）：

| 指标 | 实测 |
| --- | --- |
| 历史累计创建 `cumulative_created` | **8**（本次生命周期没有超过 8） |
| 同时存活峰值 `max_alive_ever` | 8 |
| 死亡原因 | 8/8 全部 `starved`，首次 t=431.15s，灭绝 t=537.55s |

**必须区分两个结论，不能混淆：**

1. **本次实跑没有突破 8**：所有 6 次生育都发生在首次死亡（431s）之前；
   431s 后幸存者能量已 < 40，永远达不到怀孕条件，所以第 9 次创建没有发生。
2. **代码路径仍然允许突破 8**：上限判据只是 `len(world.agents)`。只要"死亡发生"与
   "两个能量充足、亲密足够的成年个体存活"同时成立，就会创建第 9 个个体。
   **已用受控复现脚本 `experiments/probe_cap_path.py` 实测证明**
   （`experiments/results/baseline_cap_path.json`）：

   ```
   创建 8 人 → 0 号老死（存活 7）→ 一对成年个体怀孕分娩 →
   agent_ids_seen = [1,2,3,4,5,6,7,8,9]
   cumulative_created_upper_bound = 9，cap_broken = true
   ```

   即：**原始系统在受控条件下会创建第 9 个个体，违反任务书 §2.1 的硬约束。**
   改造后同一场景必须被拒绝，且 `total_created` 恒 ≤ 8（见
   `tests/test_population_limits.py`）。

**这正是本改造必须引入 `AgentRegistry`（历史注册表）的原因**：
"当前存活人口"与"历史累计创建人口"必须是两个独立指标（任务书 §2.1.9）。

此外 `World.agent` 属性在 `world.agents` 为空时会**自动 `spawn()` 一个新个体**
（`world.py:98-103`），这意味着灭绝后任何一次 `world.agent` 访问都可能"复活"人口——
必须改为显式创建路径并受 8 人总额约束。

---

## 7. 应保留 / 应修改 / 确实缺失

### 7.1 必须保留（有效实现，改造不得破坏）

| 模块 | 保留理由 |
| --- | --- |
| `pixelworld.py` / `ecology.py` | 原子世界、地形、资源、生长/破坏、土壤水分——真实世界规则的好底子 |
| `physics.py` / `daynight.py` / `environment.py` / `physiology.py` | 物理/昼夜/热学/生理闭环，可直接作为生存压力来源 |
| `character.py` 的运动与观测 | 运动学、跳跃、8 维观测（env.py 用）稳定且有测试 |
| `neural.py` 的 MLP + 语言头 | 5594 参数的小网络，BC 训练通路完整；保留并**零初始化扩维** |
| `training.py` | 数据集采集 + 手写反向传播；保留，P3 只扩展特征 |
| `society.py::SocietyNetwork` | 图聚合社会上下文有实现；保留，只是目前没人用 |
| `env.py` | 8 维/3 动作/奖励钩子的标准接口；保留给 RL 研究，不删除 |
| 全部 111 个测试 | 基线回归保护 |

### 7.2 必须修改（修复造成停滞/违背约束的机制）

| 位置 | 修改 | 阶段 |
| --- | --- | --- |
| `society.py` 死亡分支 | 不删除历史：登记 `AgentRecord`（died_at/cause/final_state），从存活索引移除但保留档案 | P3 |
| `population.py::sync` | 不再 `del minds[agent_id]` 丢弃一切；改为归档 mind 指纹 + 记忆/知识摘要，且死者不再参与 `step` | P3 |
| 人口上限 | 用 `AgentRegistry.total_created <= 8` 约束**创建**，而非存活数 | P3 |
| `world.py::agent` 属性 | 去掉"访问即 spawn"的隐式复活；受约束创建走 `Population.create_agent()` | P3 |
| `technology.py` | 配方从代码 if 改为数据（`Recipe` + 前置知识）；采集/制造成果经 `simulate_recipe` 验证；新增材料 `fiber` 与技术效果（负重/恢复/取水半径） | P2 |
| `social.py::SocialLife` | 降级为"技能原语 + 教师"，不再决定长期目标；目标选择交给 `GoalManager` | P1 |
| `neural.py::features` | 24 → 40 维（记忆 8 + 目标 4 + 文明 4），旧 checkpoint 零初始化迁移 | P3 |
| `render.py::debug_lines_zh` | 加人口/累计创建/知识/目标/设施行 | P1-P4 |
| `cli.py` | 新增 `--society` headless 文明模式与实验入口 | P0/P4 |

### 7.3 确实缺失（需要新建，接口见 `docs/civilization_design.md`）

| 新模块 | 职责 | 阶段 |
| --- | --- | --- |
| `goals.py` | Goal / Plan / PlanStep / Planner / GoalManager（多步规划、失败重规划、防卡死） | P1 |
| `knowledge.py` | Knowledge / Recipe / RECIPE_CATALOG / KnowledgeStore（发现、验证、依赖图、失效、保留率） | P2 |
| `memory.py` | MemoryEntry / Procedure / MemoryStore（情景/程序记忆、失败利用、传承、8 维摘要） | P3 |
| `civilization.py` | AgentRecord / AgentRegistry / Facility / SocialContract / CivilizationState（历史、上限、设施、崩溃状态机、指标） | P3/P4 |
| `telemetry.py` | 指标采集与日志（动作分布、重复、目标完成、技术、留存、创新、合作、灭绝） | P0 |
| `simulation.py` + `experiments/` | headless 文明运行器、基线/消融实验（A/B/C/D × 多 seed） | P0/P4 |

---

## 8. 改造前基线复现方法

```powershell
cd newworldpro
$env:TEMP = (Resolve-Path .tmp).Path      # 沙箱下临时目录（TemporaryDirectory 清理需要）

# 1) 基线测试（111 用例）
python -m unittest discover -s tests

# 2) 动作/奖励/状态变化基线
python experiments/probe_baseline.py --steps 2400 --dt 0.05 --tag pre

# 3) 人口累计创建上限基线（长时程，真实 dt）
python experiments/probe_long.py --steps 40000 --dt 0.05 --tag pre

# 4) 上限代码路径的受控复现（改造前：死亡释放名额后创建第 9 个个体）
python experiments/probe_cap_path.py
```

产物：
- `experiments/results/baseline_probe_pre_seed0.json`
- `experiments/results/baseline_long_pre_seed0.json`
- `experiments/results/baseline_cap_path.json`
- `experiments/results/baseline_cap_seed0.json`（粗步长反例，见 §3.2 探针 C）

### 8.1 基线测试的 6 个环境错误（不隐瞒）

`python -m unittest discover -s tests` 结果：**Ran 111 tests, FAILED (errors=6)**。
6 个错误全部是 `tempfile.TemporaryDirectory` 在 DSH 沙箱下清理时
`PermissionError: [WinError 5]`（`tempfile._resetperms`），
涉及 `test_neural.py`、`test_training.py`、`test_world.py`（各 2 个）——
**测试体本身通过，报错发生在 `with` 退出清理阶段**，与项目代码无关。
README 声称的 "111 tests OK" 在无沙箱环境下成立；本改造新增
`tests/_env_compat.py` 兼容垫片，使沙箱环境也能得到干净的测试结果
（不改变测试语义）。

---

## 9. 改造后复核：审计结论是否被修复（P0 闭环）

| 审计结论 | 改造后实测 | 证据 |
| --- | --- | --- |
| 无目标/计划/记忆 | 已补齐，A→B 灭绝时间 419.6→597.9s，最长重复动作 4377→54.7 步 | `experiments/results/ablation_final/` |
| 死亡即删除、无历史 | 死亡先归档（库存/知识/技能/记忆/亲子），累计创建只增不减 | `tests/test_population_limits.py`、`test_inheritance.py` |
| 上限按存活数、可破 8 | 受约束创建：第 9 次返回 None 并记录拒绝，跨组跨 seed `total_agents_created==8` | `test_total_created_never_exceeds_eight`；消融表 |
| 技术只有一条硬编码配方 | 7 条数据驱动技术 + 依赖图 + 模拟器唯一验证 | `tests/test_knowledge_integration.py` |
| 小世界无食物、浆果不可再生 | 每个尺寸 ≥2 个浆果丛；浆果可反复再生 | 8 个尺寸生成检查 + `test_validated_axe_changes_harvest_yield` |
| 教师饿死全员（537.55s 灭绝） | 教师生存优先；C/D 组存活到老年（1031.3±1.1s），8/8 死因为 old_age | 消融表 |
| 无奖励信号 | 仍无奖励（未伪造 RL）；规划/知识用规则+代价搜索 | `docs/final_report.md` §8 |
| 无法解释的"重复动作" | 不重复本身不是目标：B 组重复动作 4377→54.7 是规划层的直接效果 | 消融表 `repeat_run_max` |

**结论**：审计中列出的 6 类结构性问题全部有对应修复与测试；
未修复的是"没有奖励学习"——这一点被明确保留为已知限制，而不是用指标掩盖。

### 9.1 改造后测试

`python -m unittest discover -s tests` → **Ran 297 tests ... OK**（111 原始 + 186 新增）。
