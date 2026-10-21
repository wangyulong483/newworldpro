# newworld 架构

> 当前平台只有一条世界线：**确定性俯视地图 + 规则驱动群伙 + 可插拔机制层**。
> 本文件描述模块职责、每步数据流、关键契约（观测 / 压实 / 注册 / 配置 /
> zoo / lab API）。所有行号锚点对应本文件编写时的源码。

---

## 1. 模块总览

| 模块 | 职责 |
|---|---|
| `newworld/map_engine.py` | 地形 + 食物网格；确定性噪声生成；食物再生、森林蔓延；地形/食物只读查询 |
| `newworld/swarm.py` | 群伙 SoA 数组；人口动态（饥饿/死亡/进食/移动/繁衍/压实）；观测组装；转移与奖励结算 |
| `newworld/swarm_policy.py` | `SwarmPolicy` 协议（`act` 批量接口）；`rule` / `random` / `neural` 三种策略；`SwarmTrainer`（REINFORCE） |
| `newworld/social_mech.py` | P3 社会机制层：交换 / 教学 / 继承；技能在世界规则层生效（`swarm.skill_effect` 钩子） |
| `newworld/communication.py` | P6-A 通信层：符号通道（观测追加 K 维"听到的符号直方图"）+ sender REINFORCE；受控指称博弈 `ReferentialGame` |
| `newworld/agent_memory.py` | P6-B 记忆层：逐个体记忆流、新近×重要×相关检索、周期反思；6 维记忆观测 |
| `newworld/experiment.py` | 唯一实验运行器：配置 -> 构造 -> 逐步/整段推进 -> 指标 -> 产物 |
| `newworld/settings.py` | YAML 配置 schema（严格字段校验、`_base` 继承、`--set` 覆盖、sha256 指纹） |
| `newworld/registry.py` | `POLICIES` / `PROTOCOLS` / `MECHANISMS` 命名注册表 + 装饰器 |
| `newworld/zoo.py` | 模型 zoo：`policy.npz + card.json + metrics.json + config.yaml` 的读写与 sha256 校验 |
| `newworld/protocols/` | 实验协议：`common`（训练/评估/配对统计）+ p1/p3/p4/p5/p6 协议模块 |
| `newworld/lab_server.py` | 浏览器实验平台后端（FastAPI + WebSocket）；仿真 daemon 线程是唯一状态权威 |
| `newworld/app_map.py` | 桌面 GUI（tkinter）：地图画布、群伙散点、统计面板、开始/暂停/单步/速度/重置 |
| `newworld/map_render.py` | 底图渲染（地形色块 + 食物明暗）与群伙散点数据 |
| `newworld/viewport.py` | 世界↔屏幕坐标变换、缩放/平移、边界约束（纯逻辑，无 tkinter，可单测） |
| `newworld/cli.py` | 统一命令行入口：GUI / lab / 配置化实验 / `--info` / `--list-components` |
| `newworld/__init__.py` | 包导出（`MapEngine/Swarm/SocialLayer` 等）与 `__version__` |

导入方向是单向的：`map_engine` 不依赖群伙；`swarm` 不 import 社会/通信/
记忆层，机制层通过**观测 provider / skill_effect 钩子**注入（依赖倒置，
避免循环 import）。

---

## 2. 每步数据流

`Simulation.step()`（`newworld/experiment.py:74`）严格按以下顺序推进
一个物理步：

```text
engine.update(dt, daylight)       # 食物再生、森林蔓延、时间前进
swarm.update(dt)                  # 饥饿/年龄 -> 死亡 -> 进食 -> 移动/策略 -> 奖励 -> 繁衍 -> 压实
social.on_death_pass(...)         # 本帧死者的技能转移/失传（P3 继承）
social.update(dt)                 # 交换、教学（P3）
comm.update(dt)                   # 重排 -> 群体奖励 -> sender REINFORCE -> 重采样符号 -> 统计（P6-A）
memory.update(dt)                 # 记录事件 -> 反思 -> 检索 -> 刷新 6 维观测（P6-B）
```

要点：

* 社会层先于通信/记忆层：通信的"群体奖励"读取 `swarm.eat_events`，
  记忆的事件记录读取 `swarm.last_eat_rows` / `_eat_amount`（本帧进食；
  两者都是 swarm 压实前的旧行号，由各层按 `compact_keep` 映射）。
* 机制层只依赖世界规则的输出，不反向改写 `MapEngine` 网格；所有效果
  通过真实数值通路发生（技能乘数、能量转移、符号观测、记忆观测）。
* GUI（`app_map.py`）按同样的顺序手写推进（世界/群伙/社会层，不构建 P6 层）；
  lab（`lab_server.py`）与无窗口实验走 `Simulation.step()`（含 P6 层）。
  三者共用同一批世界规则实现，不各写一套。

---

## 3. 观测组装契约

基础观测 **9 维**由 `Swarm._base_observations`（`newworld/swarm.py:206`）
冻结定义：

| 索引 | 含义 |
|---|---|
| 0 | 自身饱足（hunger，0..1） |
| 1 | 年龄/100（截断到 1） |
| 2 | 脚下食物 |
| 3–6 | 东 / 西 / 南 / 北 3 格处食物（不可走为 -1） |
| 7 | 邻水警示（0/1） |
| 8 | bias（恒 1） |

扩展列由机制层通过 `Swarm.register_obs_provider(provider, dim)`
（`newworld/swarm.py:172`）注册；`provider(view, rows)` 必须返回
`(len(rows), dim)` 的有限浮点数组。`Swarm._observations` 按注册顺序
`hstack` 基础 9 维与各 provider 输出，`Swarm.obs_dim`（`swarm.py:166`）
= `9 + Σdim`。

当前 provider：

| 层 | 维度 | 内容 | 注册入口 |
|---|---|---|---|
| `CommunicationLayer` | `comm.n_symbols`（默认 4） | 上一帧 `radius` 内听到的其它个体符号直方图（无则全 0） | `communication.py:257` |
| `MemoryLayer` | 6 | `[remembered_food_dx, remembered_food_dy, recent_success_rate, recent_danger_rate, reflection_food_bias, reflection_risk_bias]` | `agent_memory.py:447` |

构造顺序契约（`build_simulation`，`newworld/experiment.py:181`）：
先建 swarm，再建并 `attach` social / comm / memory（注册观测），**最后**
按 `swarm.obs_dim` 构造策略并 `swarm.attach_policy(policy)`。因此：
通信/记忆开关改变观测维度时，策略会按新维度初始化；`attach_policy`
（`swarm.py:148`）会校验 `policy.obs_dim == swarm.obs_dim`，不匹配直接报错。
`make_swarm_policy(checkpoint=...)`（`swarm_policy.py:486`）也会拒绝
obs_dim 不一致的 checkpoint（防止把 9 维模型挂到 13 维观测上）。

---

## 4. 压实 / 重排契约（row-indexed state）

`Swarm` 用固定容量结构化数组 + 死亡行压实复用槽位。机制层若按行号
维护状态，必须响应压实：

* `Swarm._compact()`（`newworld/swarm.py:407`）：
  * `compact_keep` = **压实前存活行的旧行号**（按新顺序排列）；
  * `compact_generation` 只在真正发生压实（有死亡）时 +1；
  * 无死亡时 `compact_keep = arange(count)`，generation 不变。
* `last_eat_rows`（本帧吃到食物的旧行号）与 `_eat_amount`（旧行号 -> 实际吃到的量）
  都在压实前记录，供记忆层记录事件。
* 机制层的标准做法（`communication._maybe_reindex`、
  `agent_memory._reindex_if_needed`）：

```python
if swarm.compact_generation != last_seen:
    last_seen = swarm.compact_generation
    keep = swarm.compact_keep          # 旧行号，新顺序
    new_state[:len(keep)] = old_state[keep]
```

通信层在同一次压实后清空"上一步 (obs, symbol)"的训练配对（行号语义已失效）；
记忆层把本帧事件（旧行号）经同一映射改写到新行号。

---

## 5. 策略与训练

* 策略协议：`act(obs_batch) -> (N,) actions`，动作 5 个
  （0 右 / 1 左 / 2 下 / 3 上 / 4 原地）。`newworld/swarm_policy.py` 注册：

| 注册名 | 类 | 说明 |
|---|---|---|
| `rule` | `RuleSwarmPolicy` | 4 方向食物梯度探测 × 随机权重 + wander（无学习的规则基线） |
| `random` | `RandomSwarmPolicy` | 均匀随机动作（下界） |
| `neural` | `NeuralSwarmPolicy` | numpy MLP：obs_dim→hidden→n_actions，tanh + softmax/Gumbel 采样，参数共享 |

* 奖励由 `Swarm.update` 结算（`swarm.py:353` / `swarm.py:503`）：
  进食按实际吃到的食物量、每步时间成本 `REWARD_TIME_PER_S·dt`、
  死亡惩罚 `REWARD_DEATH`；转移累积在 `swarm.transitions`，
  训练器用 `drain_swarm_transitions()` 取走。
* `SwarmTrainer`（`swarm_policy.py:363`）：minibatch REINFORCE，
  批内优势标准化 + 全局梯度范数裁剪；`logprob_and_grads` 手写反向传播。
* checkpoint：`NeuralSwarmPolicy.save/load` 存 `W1/b1/W2/b2 + obs_dim/
  n_actions/hidden/seed/version` 的 `.npz`；zoo 负责加元数据。

---

## 6. 注册表插件模型

`newworld/registry.py` 提供三个命名注册表（`registry.py:140`）：

* `POLICIES` —— 群伙策略（`rule` / `random` / `neural` / ...）；
* `PROTOCOLS` —— 实验协议（`learning_compare` / `social_ablation` /
  `evolution_compare` / `scaling_sweep` / `communication` / `memory_ablation`）；
* `MECHANISMS` —— 机制组件（预留给后续机制注册）。

约定：

* 注册：`@register_policy("name", note="...") / @register_protocol(...) /
  @register_mechanism(...)`；同名重复注册在 import 期直接报错
  （`RegistryError`），测试可用 `overwrite=True`。
* 构造：`registry.create(name, *args, **kwargs)` 原样转发参数；
  未注册名字报错并列出可用名字。
* 新增组件 = 新文件 + 装饰器；CLI、实验入口、lab 后端都不改。
* 协议包 `newworld/protocols/__init__.py` 导入各协议模块以触发注册
  （`experiment._import_protocols` 幂等调用）。

---

## 7. 配置系统

```text
configs/*.yaml
   │  _base: base.yaml（可多层继承，深合并）
   ▼
load_config(path)        -> ExperimentSettings（严格 dataclass，未知字段报错）
apply_overrides(...)     -> 按点号 --set 覆盖后重新校验
   ▼
ExperimentSettings.hash() -> sha256(规范化 JSON)，写入 summary 与 zoo
```

配置根对象是 `ExperimentSettings`（`newworld/settings.py:390`）：

| 段 | 数据类 | 作用 |
|---|---|---|
| `world` | `WorldSettings` | 地图生成 / 生态参数（-> `MapConfig`） |
| `swarm` | `SwarmSettings` | 人口/代谢/繁衍参数（-> `SwarmConfig`） |
| `policy` | `PolicySettings` | 策略种类与训练超参 |
| `social` | `SocialSettings` | P3 三机制开关与参数（-> `SocialConfig`） |
| `comm` | `CommSettings` | P6-A 符号数/半径/消息成本/训练开关 |
| `memory` | `MemorySettings` | P6-B 容量/检索权重/反思周期 |
| `run` | `RunSettings` | 预算（seconds/dt/daylight/seeds）、产物目录、是否存模型 |
| `protocol_params` | `dict` | 协议私有参数（各协议自行解析并校验） |

范围校验在各 dataclass 的 `__post_init__`（如 `0 <= sea < mountain < snow <= 1`、
`top_k <= capacity`、`n_symbols >= 2`）；`load_config`（`settings.py:462`）与
`apply_overrides`（`settings.py:468`）分别负责加载与覆盖，`config_hash`
（`settings.py:522`）给实验指纹。

---

## 8. 实验协议

协议 = 注册表里的一个函数 `f(settings) -> dict`，由 `run_experiment`
（`newworld/experiment.py:265`）调起，产物写到 `output_dir/name/`。
`protocols/common.py` 统一了"造世界-训练-评估-配对统计"的工具：
`simulate`、`train_neural_policy`、`evaluate_policy/evaluate_seeds`、
`mean_std`、`paired_summary`、`summarize_runs`、`write_json`、`timed`。

| 协议名 | 阶段 | 设计 |
|---|---|---|
| `learning_compare` | P1 | 稀缺地图上 random/rule/untrained/trained 四臂配对 + 密度梯度；预注册胜场门槛；模型入 zoo（tag p1） |
| `social_ablation` | P3 | off_all / barter_only / teach_only / inherit_only / all_on 五配置配对对照（规则策略） |
| `evolution_compare` | P4 | rule / 技能分布 θ 变异+选择 / 策略梯度学习 三臂配对；学习臂模型入 zoo（tag p4） |
| `scaling_sweep` | P5 | 人口→吞吐 与 学习算力→性能 两个单维扫描 + 幂律拟合；产 `report.md`，最大算力模型入 zoo（tag p5） |
| `communication` | P6-A | 指称博弈（准确率 vs chance、MI、熵、映射稳定性）+ 世界内通道 off/on 配对消融；on 臂模型入 zoo（tag p6-comm） |
| `memory_ablation` | P6-B | off / off_padded（同维恒零观测，控 obs_dim 混淆）/ recency / full / full_reflect 五臂配对消融；最优记忆臂入 zoo（tag p6-memory） |

协议名、配置与最新结果见 [EXPERIMENTS.md](EXPERIMENTS.md)。

---

## 9. 模型 zoo

`newworld/zoo.py` 把训练产物定义为一个**自描述目录**：

```text
assets/zoo/<name>/
├── policy.npz     # NeuralSwarmPolicy 权重（W1/b1/W2/b2 + 维度/seed/version）
├── card.json      # ModelCard：架构/维度/seed/创建时间/git commit/config hash/weights sha256
├── metrics.json   # 协议写入的评估指标
└── config.yaml    # 训练/评估配置快照（可选）
```

API：`ModelZoo(root=assets/zoo)`；`save(name, policy, metrics=, config=, tags=,
description=, extra=)`（`zoo.py:142`）写目录与 card；`load_card(name)`
（`zoo.py:208`）读元数据；`load(name)`（`zoo.py:218`）载入策略并校验
维度与 card 一致；`verify(name)`（`zoo.py:251`）重算 `policy.npz` 的
sha256 与 card 比对；`names()/cards()/manifest()` 枚举与汇总。
模型名只允许 `[A-Za-z0-9_.-]` 且以字母数字开头（防路径穿越）。

---

## 10. lab 后端

`newworld/lab_server.py`（FastAPI）：

* `LabSession` 持有一个 daemon 仿真线程，以固定 `dt=0.05` 推进；
  速度档 = 每 tick 物理步数 `(1, 2, 4, 8, 16)`，帧流 `STREAM_HZ=15`；
* **仿真线程是唯一状态权威**：所有读写经 `threading.Lock`；API 层只读
  快照 / 发命令；
* 一切干预走 `LabSession.command(kind, **params)` 并 append 到
  `interventions`（时间/类型/参数），支持 `pause / resume / speed / step /
  reset`；未知命令返回 `ok=false` 且不入审计；
* 默认配置是 P1.7 稀缺环境（`default_lab_settings`：256×160、`rule`、
  `food_regrow_per_s=0.0005`），也可 `--config YAML` 用任意实验配置启动；
* 前端是 `lab_static/index.html`（PixiJS），静态挂载在 `/`。

API 契约：

| 方法 | 路径 | 返回 |
|---|---|---|
| GET | `/api/snapshot` | 完整只读快照（地形 hex、人口/食物/饱足、series、interventions、social/comm/memory 统计、config_hash） |
| WS | `/api/stream` | 约 15Hz 增量帧：`t, running, x, y, h, pop, food, sat`（+ comm/memory 统计） |
| GET | `/api/agent/{index}` | 个体详情（位置/饱足/年龄/观测/技能/信号/记忆条数）；越界 `ok=false` |
| POST | `/api/command` | `{kind, ...}` 实验控制，返回 `{ok, record}` |
| GET | `/` | lab 前端页面 |

---

## 11. 已知边界（架构层面）

* numpy MLP 的规模上限：hidden 默认 32，无 GPU/批量并行训练框架；
  千级个体时仿真仍是向量化批处理，但训练只支持"一条轨迹一段回报"的 REINFORCE。
* 单进程单世界：GUI / lab / 实验各自一个进程，没有分布式 rollout 或参数服务器。
* 机制层的工作量随 N 增长（通信的半径直方图 O(N²/步)，记忆的检索
  O(N·capacity)）；当前规模（几百到几千个体）可接受，规模上去需要空间索引。
* lab 的 `series` 采样与 UI 刷新是"可观测性压缩"，不是高保真记录；
  正式实验请用配置化协议产出 `summary.json`。
