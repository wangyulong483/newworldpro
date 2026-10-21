# newworld — 开放式人工社会研究平台

> 一张确定性生成的俯视地图 + 千级规则驱动群伙。世界动态全部来自规则
> （食物再生、森林蔓延、代谢、繁衍、死亡、社会机制）——作为设计原则，
> 平台没有"上帝之手"式干预：不存在改写地图或直接修改个体的入口；
> 人对平台的干预只限于实验控制（暂停 / 继续 / 单步 / 调速 / 重置），
> 且在实验平台中全部入审计记录。
>
> 纯 Python + numpy 实现（无 PyTorch、无 LLM），目标是让"个体策略 /
> 社会机制 / 实验协议"都能被配置、复现和测量，而不是做视觉演示。

---

## 1. 平台定位

这是一个**可复现的人工社会实验台**，不是游戏：

* **世界规则唯一权威**：地形与食物网格只由 `MapEngine` 按规则推进，
  界面只读；没有任何点击改写地图的交互。
* **个体是被测量的对象**：群伙个体是结构化数组里的一行，行为由可注入
  策略决定（规则基线 / 随机基线 / numpy 神经网络），所有转移与奖励可审计。
* **机制独立开关**：交换、教学、继承、通信、记忆各有一套参数与开关，
  支持同 seed 的配对消融实验。
* **实验即配置**：一份 YAML 描述世界、群伙、策略、机制与预算，走
  `load_config -> build_simulation -> protocol -> summary.json` 单一代码路径。
* **诚实的指标**：实验报告必须区分"机制已被触发"与"机制产生了统计上
  可辨识的效应"；没有效应就写没有效应。

研究问题（详见 [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md)）：
学习策略能否在无剧本环境胜过规则策略；社会机制能否产生可测的宏观差异；
演化与学习两条适应路径的差距；平台自身的吞吐/学习标度关系；通信协议
与记忆机制能否在受控任务中被测量。

---

## 2. 世界是什么

### 2.1 地图与生态（`newworld/map_engine.py`）

* 默认地图 **512×320** cell（GUI 默认；配置化实验默认 256×160），
  值噪声 + 双尺度叠加确定性生成：**深海 / 浅海 / 沙滩 / 草地 / 森林 /
  山地 / 雪原**七种地形；同 seed 必得同地图。
* 食物是叠加在地形上的 `food[y, x]` 网格：草地/森林/沙滩各有密度
  上限，按 `food_regrow_per_s` 每秒再生（夜间 `daylight` 越低再生越慢）。
* 森林蔓延：邻接森林的草地以 `forest_spread_per_s` 的小概率长树，
  这是地形唯一的自发变化。
* 群伙只能站上陆地；踏入深/浅水会溺亡（`drowned` 计数）。

### 2.2 群伙与人口动态（`newworld/swarm.py`）

* 全体个体保存在一个 numpy 结构化数组（SoA）里，逐帧整批更新；
  死者的行会被压实复用（compact），阵列容量由 `capacity_hard` 给定。
* 每步顺序：**饥饿/年龄 → 死亡判定 → 进食 → 移动 → 繁衍 → 压实**。
* 饥饿按 `metabolism_per_s` 下降，进食从脚下 cell 扣食物并回补饱足；
  繁殖需要成年、饱足、冷却结束，且人口低于**涌现承载力**估计；
  死亡原因分饥饿 / 老龄 / 溺亡。
* **承载力是涌现的**：人口增长到"食物再生流量 ≈ 消耗流量"附近自然
  收敛，不是硬编码上限（`capacity_hard` 只是数组保险丝）。
* 确定性：同 seed（地图 + 个体初始化 + 策略采样随机数）得到同一轨迹。

### 2.3 策略层（`newworld/swarm_policy.py`）

* 协议：批量接口 `act(obs_batch) -> actions`，5 个离散动作
  （右 / 左 / 下 / 上 / 原地）。
* 内置三种策略（注册表名）：
  * `rule` —— 4 方向食物梯度探测 + 随机抖动（规则基线）；
  * `random` —— 均匀随机动作（下界基线）；
  * `neural` —— numpy MLP（默认 9→32→5，softmax 采样），全体个体共享
    参数，REINFORCE 训练（minibatch + 优势标准化 + 梯度裁剪）。
* **基础观测 9 维**：饱足、年龄/100、脚下食物、东南西北 3 格食物
  （不可走为 -1）、邻水警示、bias。通信/记忆等机制层通过
  `register_obs_provider` 追加观测列，`swarm.obs_dim = 9 + Σprovider`。
* 奖励由世界规则结算：进食按实际吃到的食物量、每步时间成本、死亡惩罚；
  `drain_swarm_transitions()` 给训练器取走转移。

### 2.4 社会机制（`newworld/social_mech.py`，P3）

三个可独立开关的机制，效果落在世界规则层（不是标签）：

| 机制 | 规则 | 参数 |
|---|---|---|
| 交换 | 饱足差足够的邻近个体间转移能量（守恒） | `barter_gap / transfer / radius / interval_s` |
| 教学 | 有技能个体把技能传给半径内无技能邻居 | `teach_radius / chance_per_s / interval_s` |
| 继承 | 个体死亡时技能传给最近存活邻居，无继承人则失传 | `inherit_radius` |

技能三种，真实改变个体参数：`efficient_biter`（进食 ×1.5）、
`long_sight`（移动 ×1.4）、`slow_metabolism`（代谢 ×0.75）。
统计字段（交换/教学/继承/失传/技能人口）同时出现在 GUI、实验平台与
`summary.json` 中。

### 2.5 工程化（P5）

* `newworld/settings.py`：严格 YAML 配置（未知字段报错、`_base` 继承、
  `--set a.b=value` 覆盖、sha256 配置指纹）；
* `newworld/registry.py`：`POLICIES / PROTOCOLS / MECHANISMS` 注册表，
  新增策略/协议 = 新文件 + 装饰器；
* `newworld/experiment.py`：唯一的世界构造 / 推进 / 指标 / 产物路径；
* `newworld/zoo.py`：模型 zoo（自描述目录：权重 + card + metrics + config）；
* `newworld/protocols/`：P1 / P3 / P4 / P5 / P6 实验协议（注册名调用）。

### 2.6 通信与记忆（P6）

P6-A 通信：群伙带一个**离散信号通道**（K 个符号、半径内广播），发送者
策略与移动策略分离（发送者用群体奖励训练），接收者观测追加"上一帧听到
的符号直方图"；语义不硬编码，用互信息 / 对齐率测量是否形成。另有一个
受控**指称博弈**（发送者看隐藏线索发符号、听者据符号猜线索）作为协议
形成的干净证据。

P6-B 记忆：每个体维护固定容量**记忆流**（吃到 / 濒死 / 休息 / 反思
事件），按"新近 × 重要 × 相关"检索 top-k，周期性把事件聚合成更高层
反思；检索结果以 6 维观测 provider 接入移动策略。全部是 numpy 机制层，
**没有 LLM**。

> 具体模块与配置见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)；
> 实验协议与最新数字见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

---

## 3. 快速开始

依赖：Python 3.10+，运行时依赖见 `requirements.txt`
（numpy / pillow / pyyaml / fastapi / uvicorn）。

```powershell
cd C:\Users\27826\Desktop\vscode_win_ws\ds_ws\newworld\newworldpro
python -m pip install -r requirements.txt
```

### 3.1 桌面 GUI

```powershell
python -m newworld                         # 512×320 地图，窗口 1280×800
python -m newworld --seed 7 --size 256x160 # 指定 seed 与地图尺寸
python -m newworld --cell-px 2             # 每个 cell 的屏幕像素
```

界面：顶部实时社会机制统计；中间地图 + 群伙散点；右侧人口/地形/社会
明细；底部开始/暂停、单步、速度、重置；鼠标拖动平移、滚轮缩放；空格
开始/暂停，`+`/`-` 调速，Esc 退出。**界面只观察与控制，不提供任何
改写世界的工具。** 完整说明见 [docs/USER_GUIDE.md](docs/USER_GUIDE.md)。

### 3.2 浏览器实验平台（lab）

```powershell
python -m newworld --lab                   # 打开 http://127.0.0.1:8600
python -m newworld --lab --port 8601 --seed 3
```

平台与无窗口实验共用 `build_simulation` 同一条代码路径；默认是 P1.7
的稀缺环境（`food_regrow_per_s=0.0005`）。支持 `/api/snapshot`、
`/api/stream`（WebSocket）、`/api/agent/{index}`、`/api/command`
（pause / resume / speed / step / reset，全部入审计）。也可以直接启动
后端：`python -m newworld.lab_server --port 8600`，或双击
`start_world.bat`（打开浏览器并启动 8600 端口服务）。

### 3.3 配置化实验

一份 YAML = 一次可复现实验（`configs/base.yaml` 是公共基线，其余配置
用 `_base` 继承它）：

| 配置 | 协议 | 内容 |
|---|---|---|
| `configs/p1_scarce.yaml` | `learning_compare` | P1.7 稀缺世界 random/rule/untrained/trained 四臂配对 + 密度梯度 |
| `configs/p3_social.yaml` | `social_ablation` | 交换/教学/继承五配置配对消融 |
| `configs/p4_evolution.yaml` | `evolution_compare` | 演化（技能分布 θ 变异+选择）vs 学习（策略梯度）三臂配对 |
| `configs/p5_scaling.yaml` | `scaling_sweep` | 人口→吞吐 / 学习算力→性能 单维扫描 + 幂律拟合 |
| `configs/p6_communication.yaml` | `communication` | 指称博弈 + 世界内信号通道 off/on 消融 |
| `configs/p6_memory.yaml` | `memory_ablation` | 记忆流 / 检索 / 反思五臂配对消融（含同维空对照，P6-B） |

```powershell
python -m newworld --config configs/p1_scarce.yaml          # 完整实验（数分钟）
python experiments/run.py --config configs/p1_scarce.yaml --json
python -m newworld --info                  # 版本 / 依赖 / 已注册组件
python -m newworld --list-components       # 策略与协议注册表
```

调试时用 `--set protocol_params.<...>` 缩短训练/评估（见
[docs/USER_GUIDE.md](docs/USER_GUIDE.md) §4.2），产物只用于链路验证。

产物写到 `<run.output_dir 或 --out>/<name>/`：`config.yaml`（配置快照）+
`summary.json`（指标与结论）；训练类协议还会把模型写入 `assets/zoo/`。
配置字段、覆盖语法与输出口径见 [docs/USER_GUIDE.md](docs/USER_GUIDE.md) §4；
各协议的实测数字见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

### 3.4 测试

```powershell
python -m unittest discover -s tests
```

当前 **165 个测试全部通过**（`python -m unittest discover -s tests`，
约 15s）：地图/食物/森林/确定性、群伙人口动态、三种策略与训练器、
社会机制、配置层、注册表、实验运行器/协议、lab API 契约、模型 zoo、
通信通道/指称博弈、记忆流/反思。
若前一次运行把临时目录指向了 `.tmp`，可先
`$env:TEMP=(Resolve-Path .tmp).Path`（受限环境下的临时目录问题）。

---

## 4. 仓库结构

```text
newworldpro/
├── newworld/                 # 平台代码（唯一世界线：俯视地图 + 群伙）
│   ├── map_engine.py         # 地形 + 食物网格（世界规则唯一权威）
│   ├── swarm.py              # 群伙 SoA、人口动态、观测组装、转移/奖励
│   ├── swarm_policy.py       # 策略协议 + rule/random/neural + REINFORCE 训练器
│   ├── social_mech.py        # P3 交换 / 教学 / 继承（技能在世界规则层生效）
│   ├── communication.py      # P6-A 信号通道 + 指称博弈（实验层接入）
│   ├── agent_memory.py       # P6-B 记忆流 / 检索 / 反思（无 LLM）
│   ├── experiment.py         # 唯一实验运行器：配置 -> 仿真 -> 指标 -> 产物
│   ├── settings.py           # YAML 配置 schema / 继承 / 覆盖 / 指纹
│   ├── registry.py           # POLICIES / PROTOCOLS / MECHANISMS 注册表
│   ├── zoo.py                # 模型 zoo 读写与校验
│   ├── protocols/            # P1/P3/P4/P5/P6 实验协议
│   ├── lab_server.py         # 浏览器实验平台后端（FastAPI + WebSocket）
│   ├── app_map.py            # 桌面 GUI
│   ├── map_render.py         # 地图底图与群伙散点渲染
│   ├── viewport.py           # 世界↔屏幕变换、缩放/平移（纯逻辑）
│   └── cli.py                # 统一命令行入口
├── configs/                  # 实验 YAML（base.yaml 为公共基线）
├── experiments/run.py        # 配置化实验统一入口
├── tests/                    # 单元测试（unittest）
├── assets/zoo/               # 模型 zoo（实验产出，按需生成）
├── lab_static/               # lab 前端页面
├── docs/                     # 架构 / 用户指南 / 路线图 / 调研
├── tools/                    # 图标生成与快捷方式工具
├── requirements.txt
└── start_world.bat           # Windows 启动器：打开浏览器 + 启动 lab 服务
```

---

## 5. 当前限制（如实）

* **策略规模小**：numpy 手写 MLP（默认 hidden=32）、REINFORCE；没有
  PyTorch、没有 CTDE/QMIX、没有 LLM。适合验证机制，不适合大规模表示学习。
* **单进程**：GUI / lab / 实验各自只跑一个世界，没有分布式 rollout；
  千级个体时渲染是主要瓶颈（仿真本身远快于渲染）。
* **环境与指标是"可观测性压缩"**：时间尺度、寿命、食物再生都是平台
  参数，不是人口学标定；实验结论只对当前参数族有效。
* **短程评估的效应量**：P4 的经验是"弱表型 + 短程评估"容易读出无效应，
  实验报告必须给点估计与配对胜负，不给显著性宣称（无统计检验）。
* **本轮复核改写了三条旧结论**（详见 `docs/EXPERIMENTS.md`）：
  P1.7 的"稀缺环境学习优势"**撤回**（`eat_rate` 是进食事件计数伪影，
  trained 的饱足/寿命反而更差）；P4 的 learning"显著正效应"降级为
  **事件计数增加、适应度三项更差**（旧结果还存在同 seed 训练泄漏）；
  P6-A 受控指称博弈正结果成立，但**世界内通道 null**；P6-B 记忆
  机制中 **recency-only 有净摄入正结果**，full/反思改变行为频率但
  **不改善福利**。不做"智能涌现"式表述。
* 模型 zoo 里的 checkpoint 只在对应实验跑过之后才存在；未跑过的协议
  不会凭空出现模型，`assets/zoo/` 不完整是正常状态。

---

## 6. 文档索引

| 文档 | 内容 |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 模块职责、每步数据流、观测/压实契约、配置、注册表、zoo、lab API |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | GUI / lab / 配置化实验操作与排错 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 实验协议、运行方式与最新结果（由实验负责人维护） |
| [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md) | P0–P6 研究路线、假设、验收门槛与诚实记录 |
| [docs/RESEARCH_REFERENCES.md](docs/RESEARCH_REFERENCES.md) | 人工生命 / 人工社会 / MARL / 实验平台调研 |
| [assets/MODELS.md](assets/MODELS.md) | 模型 zoo 规范与当前全部 checkpoint 清单 |

外部参考（选读）：Epstein & Axtell《Growing Artificial Societies》；
Bonabeau, *Agent-based modeling*, PNAS 2002；
Lazaridou et al. 涌现通信综述。完整链接与勘误见
[docs/RESEARCH_REFERENCES.md](docs/RESEARCH_REFERENCES.md)。
