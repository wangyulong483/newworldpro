# 文明实验协议（Experiment Protocol v1）

> 本文档固定 **P0–P4 消融实验**的指标定义、分组配置、预算、报告格式与失败处理，
> 保证任何人在同一台机器上能复现同一组数字，并让"差异"可归因。
>
> 接口契约见 `docs/civilization_design.md`；实现见 `newworld/telemetry.py`。
> **本文档只定义协议，不包含本模块作者的实验结果**。文中出现的实测数字
> 均来自 Lead 的 P0 探针（原始系统、未打补丁），逐条标注来源文件。

---

## 0. 目的与适用范围

- 对象：`newworld` 微型文明（人口硬上限 8 个**创建名额**，死亡保留档案）。
- 目标：用**同一套指标**对比 A/B/C/D 四个消融组，回答：
  1. 规划（P1）是否提高目标完成率、降低无进展重复；
  2. 持久化知识/传承（P2/P3）是否让技术积累与保留率上升；
  3. 合作/公共设施/扰动（P4）是否改变生存与恢复曲线；
  4. 原始系统（A 组）的基线行为是什么（**必须能复现 Lead 记录的自灭绝现象**）。
- 非目标：**不**把"动作熵高"或"技术条目多"当作"更智能/更文明"的单一证据。

---

## 1. 环境与硬性约束

| 项 | 固定值/要求 |
| --- | --- |
| Python | 3.14（`python --version` 必须一致后写入报告） |
| 依赖 | 仅标准库 + `numpy`（+ 项目已有 `pillow`/`tkinter`，headless 实验不用 GUI） |
| 工作目录 | `newworldpro/`（所有命令在此目录执行） |
| 临时目录 | **运行测试前必须** `$env:TEMP = (Resolve-Path .tmp).Path`（DSH 沙箱下 Windows `tempfile.mkdtemp` 目录不可写，见 `tests/_env_compat.py`） |
| 人口上限 | `max_agents_created = 8`（硬约束；`total_agents_created <= 8` 必须成立） |
| 随机性 | 所有随机源必须来自 `seed`：`World(seed=...)`、`Population(seed=...)`、`GoalManager(seed=...)`、`KnowledgeStore(seed=...)`；禁止未播种的 `random`/`np.random` |
| 训练预算 | 各组必须相同：`pretrain(steps_per_hour=160, epochs=10)`（F2/重置路径）、`fine_tune(steps_per_hour=120, epochs=4)`（增量路径），自动训练间隔按现有主循环（约 30 墙钟秒）不变；模型结构 `MLP 24→64→32→58`（P3 扩维后输入 40，新增列权重初始化为 0） |
| 代码快照 | 报告必须记录 git commit（若仓库无提交则记录文件 mtime 清单）与 `assets/*.npz` 的 SHA-256 |

---

## 2. 指标定义表

`newworld/telemetry.py::METRIC_NAMES` 为**冻结指标**，`EXTRA_METRIC_NAMES` 为
诊断附加指标。`summary()` 返回全程聚合，`window_summary()` 返回 60 世界秒窗口，
每个窗口自动追加到 `Telemetry.windows`（供 `CivilizationState.history` 使用）。

### 2.1 冻结指标（契约 §8，逐项公式）

| 指标 | 公式与统计窗口 | 类型/范围 |
| --- | --- | --- |
| `survival_rate` | 个体-秒生存率：`Σ(每步存活数×dt) / (窗口开始存活数 × 窗口时长)`，截断 `[0,1]`；分母按个体-秒计，出生抬高分子 | float 0..1 |
| `action_entropy` | 动作类别 `f"{behavior_state}:{move:+d}"` 分布的归一化熵 `H/ln k`（`k`=观测到的类别数；`k<=1` 记 0）；窗口内累积 | float 0..1 |
| `repeat_run_max` | 同一 agent 连续相同动作类别的最长步数，跨 agent 取最大；窗口内累积 | int ≥0 |
| `no_progress_repeat_ratio` | 分子 = **动作类别与上一步相同** 且以下进展信号**全部**未变的 agent-步；分母 = 总 agent-步（见 §2.2 定义） | float 0..1 |
| `goal_completion_rate` | `Δ完成目标数 / Δ生成目标数`（窗口起点计数为基线；Δ生成≤0 记 0 并截断 `[0,1]`） | float 0..1 |
| `plan_cost_error` | `GoalManager.stats()["cost_estimate_error"]`，即 `mean(abs(actual_steps-expected_cost)/max(1,expected_cost))` | float ≥0 |
| `tech_verified` | `KnowledgeStore.all(status="verified")` 数量；无知识库时退化为 `CivilizationState.innovation_count` | int ≥0 |
| `tech_graph_depth` | `KnowledgeStore.depth()`：已验证技术依赖图最长链 | int ≥0 |
| `tech_graph_edges` | `len(KnowledgeStore.edges())`：已验证技术依赖图边数 | int ≥0 |
| `knowledge_retention` | `KnowledgeStore.retention(存活个体 ∪ 死亡档案掌握的知识 id)`；无知识库时近似为 `|并集| / max(1,已验证数)`，截断 `[0,1]` | float 0..1 |
| `innovation_efficiency` | `KnowledgeStore.innovation_efficiency()` = 已验证数 / 总实验尝试数 | float 0..1 |
| `cooperation_benefit` | `clamp(0.5·ln((合作产出+1)/(独立产出+1)), -1, 1)`。合作产出 = 公共库存正增量（社区木材/石头/结构/篝火、活跃设施、活跃契约）+ `event("coop_output", amount=…)`；独立产出 = 个体私有库存正增量 + `event("solo_output", amount=…)` | float -1..1 |
| `recovery_after_disturbance` | 收到 `event("disturbance")` 时记录基线（存活数/已验证知识/活跃设施数）；之后 `mean_i clamp(current_i/baseline_i,0,1)`，到 **60 世界秒**冻结；从未扰动记 1.0 | float 0..1 |
| `total_agents_created` | `AgentRegistry.total_created`（**必须 ≤8**）；无注册表时退化为观测期内见过的 `agent_id` 数 | int 0..8 |
| `extinction_time` | 首次 `alive_count==0` 的**世界时间**；无灭绝记 **-1.0** | float |
| `extinction_cause` | `"starvation"` / `"old_age"` / `"collapse"` / `""`；来源优先级：`event("extinction", cause=…)` > `CivilizationState.extinction_cause` > 最后一条死亡事件原因 | str |

### 2.2 "有意义的进展"信号（`no_progress_repeat_ratio` 的分子定义）

以下任一发生变化 ⇒ **该步不算无进展**：

1. 库存任意字段变化（`wood/stone/tool` 及未来扩展材料，按 `Inventory.__dict__` 签名比较）；
2. 知识状态变化（`known_knowledge_ids` 并集大小，或已验证知识数变化）；
3. 公共设施/社区进度变化（`community.wood/stone/structures`、篝火、活跃设施、活跃契约）；
4. 目标进度变化（该 agent 活跃计划 `(plan_id, current, version)`，或全局目标生成/完成/失败/放弃计数）；
5. 位置位移超过 `0.5 × world.config.speed × dt`（"真的移动了"，滤掉摩擦抖动）；
6. `behavior_state` 变更。

> **为什么要重定义**：Lead 的 P0 短探针实测发现，若用 `(x, energy, hydration, 库存, …)`
> 快照完全相等判定"无进展"，会因为能量/水分每步连续衰减而得到
> `state_change_step_ratio = 1.0`（即无进展恒为 0），指标毫无诊断价值。
> 因此 `telemetry.py` 同时输出两个口径，**引用时必须注明用的是哪一个**。

### 2.3 附加诊断指标（不属于 §8 冻结表，但报告必须附带）

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| `state_identical_ratio` | 相邻两步严格快照（含 `behavior_state`、能量/水分 2 位小数）完全相等的 agent-步比例 | **仅作对照**；因生理量连续变化通常≈0，不可当作"有进展"的证据 |
| `dominant_state_ratio` | 出现最多的 `behavior_state` 占全部 agent-步比例 | 识别"全族锁死在 social/rest" |
| `tech_actions_per_1000_steps` | **材料单位增量代理** ×1000 / 采样步数：存活个体私有材料（wood/stone/等数值字段）正增量 + 工具获得 + 社区公共材料（wood/stone/structures）正增量 | 粗诊断"文明是否还在积累"；**代理口径**：一次 harvest≈1 单位、一次 contribute≈3 单位、失败的 experiment 不产生材料而不计数。真实次数需与 `Technology` 计数器/`event("harvest"/"craft"/…)` 交叉核对 |

---

## 3. 消融分组与固定预算

分组定义与 `civilization_design.md` §2 `ABLATIONS` 完全一致，**不得**在实验里临时改开关：

| 开关 | A（原始系统） | B（+规划） | C（+知识/传承） | D（完整） |
| --- | --- | --- | --- | --- |
| `enable_planning` | False | True | True | True |
| `enable_knowledge` | False | False | True | True |
| `enable_inheritance` | False | False | True | True |
| `enable_social` | False | False | False | True |
| `enable_disturbance` | False | False | False | True |
| `expanded_observation` | False | False | False | True |

固定预算（各组必须一致，否则不可比）：

| 项 | 固定值 |
| --- | --- |
| 世界尺寸 | `800 × 500`（`make_config(800, 500)`；最小允许 320×240 仅用于调试） |
| 步长 `dt` | `0.1` 世界秒/步（与 §9 `RunSpec` 默认一致；**不得**与 0.05 混比） |
| 步数上限 | `20000` 步（= 2000 世界秒；实际提前终止于灭绝或达到上限） |
| 采样窗口 `window_s` | `60.0` 世界秒 |
| 随机种子 | `seed ∈ {0,1,2,3,4}`（每组件 **≥5 个 seed**；建议 10 个用于正式报告） |
| 人口上限 | `max_agents_created = 8`（硬约束，报告 `total_agents_created` 验证） |
| 训练预算 | 见 §1（各组相同，禁止只给 D 组更多训练） |
| 感知 | 默认开启 `enable_perception=True`；为节省预算可关闭，但**所有组必须一致**且报告中注明 |

---

## 4. 基线实验协议

### 4.1 配置快照（每次运行必须写入报告）

```json
{
  "protocol_version": "v1",
  "group": "A",
  "seed": 0,
  "world": {"width": 800, "height": 500, "dt": 0.1, "steps": 20000, "window_s": 60.0},
  "civilization": {"max_agents_created": 8, "lifespan_s": 900.0, "maturity_s": 70.0, "pregnancy_s": 24.0},
  "ablation": {"enable_planning": false, "enable_knowledge": false, "enable_inheritance": false,
               "enable_social": false, "enable_disturbance": false, "expanded_observation": false},
  "model": {"architecture": "MLP 24-64-32-58 (BC)", "checkpoint": "<sha256>", "training": "pretrain 160/10, fine_tune 120/4"},
  "python": "3.14.x",
  "git_commit": "<hash or file-mtime-manifest>"
}
```

### 4.2 原始系统（A 组）基线：应复现的定性-定量结论

A 组是**未打补丁的原始系统**，其基线由 Lead 的两个 P0 探针测得。以下数字**不是**
本模块作者的实验结果，只作为复现验收目标；引用时保留来源标注。

**短探针**（来源：`experiments/results/baseline_probe_pre_seed0.json`，
2 初始个体、800×500、`dt=0.05`、2400 步 = 120 世界秒）：

- 出生 4、死亡 0、存活 6、累计创建 6；
- 决策来源：手写教师接管 5153 步 vs 神经自主 3783 步（**约 58% 的决策来自手写规则**）；
- `behavior_state` 分布：explore 42.4%、social 30.9%、rest 24.9%、gather 1.2%、
  forage 0.58%、craft 0.02%、build 0.02%；单个个体最长连续同状态 **2227 步**
  （个体 0 有 2318/2400 步是 social，个体 1 有 2229/2400 步是 rest）；
- 120 世界秒内技术动作共 17 次 harvest 尝试（成功 10）、2 次 craft、2 次 contribute；
  社区建成 1 次后不再积累；
- 主循环每步标量奖励恒为 0（`env.py` 的 `RewardFunction` 未被使用）。

**未打补丁原始系统的长时程探针**（来源：`experiments/results/baseline_long_pre_seed0.json`，
真实 `dt=0.05`，40000 步上限）：

- **537.55 世界秒全族灭绝（8/8 饥饿）**；
- 累计创建 8（t=0、0、32.3、56.3、80.3、104.3、128.3、152.3 秒），存活峰值 8，
  社区约 100 秒建成，首次饥饿死亡 t=431.15 秒；
- 行为状态-步：social 55826（82.7%）、explore 8505、rest 2874、gather 107、
  forage 52、craft 2、build 2；
- 平均能量轨迹：t=100 → 60；t=200 → 41；t=300 → 26；t=400 → 13；随后灭绝；
- 归因：社区建成后手写教师 `SocialLife` 的求偶分支接管成年个体且**从不返回
  `None`**，个体不再进食。

**A 组的正式定义（与上面的"未打补丁版本"不同，必须分清）**

消融实验中的 A 组 = **原始机制 + P0 机制缺陷修复**：
- 修复 1：教师生存优先（能量/水分过低时交回下层，不再饿死全员）；
- 修复 2：世界食物链（小世界至少生成 2 个浆果丛；浆果被采后可再生）；
- 修复 3：创建预算（累计创建 ≤ 8，不再用存活数判定）；
- 修复 4：世界不再因 `world.agent` 访问而隐式复活人口。

不修复这些，A 组根本不是一个可比较的基线（它会在 537s 全族饿死且没有技术）。

**A 组验收口径**（`800×500`、`dt=0.05`、24000 步）：
- `total_agents_created == 8`（硬约束成立）；
- `tech_verified == 0`（无知识机制）；
- 约 **390~440 世界秒饥饿灭绝**（跨 seed 0/1/2 实测 430.95 / 388.65 / 439.10 s）。
  A 组比未打补丁版本更早灭绝，因为它不再"站在原地饿死"，而是把 8 个名额更快用满，
  随后在夜间食物短缺中崩溃——具体归因见 `experiments/results/ablation_final/report.md`。
- 若 A 组出现了技术或规划指标 > 0，说明消融开关被绕过，应先修开关而不是调指标。

> 注意：`dt=0.1` 的实验网格是 P1–P4 对比用的固定网格；A 组 `dt=0.05` 的
> 537.55 s 来自 P0 探针，只能作为"现象应复现"的参照，不能在报告里与 `dt=0.1`
> 的运行直接平均。

### 4.3 指标报告格式（每次运行一个 JSON + 一个 JSONL）

`Telemetry.to_json()` 顶层即 `summary()` 全部键（19 个指标 + 元数据），另含
`windows` / `action_distribution` / `state_distribution` / `events_count`。
正式报告在其外层再包一层运行元数据：

```json
{
  "run": {"group": "A", "seed": 0, "protocol_version": "v1", "wall_s": 0.0, "failures": []},
  "config": {"...": "§4.1 快照"},
  "metrics": {"extinction_time": 0.0, "survival_rate": 0.0, "total_agents_created": 0},
  "extra": {"action_distribution": {}, "state_distribution": {}, "windows": []},
  "artifacts": {"summary_json": "<path>", "events_jsonl": "<path>"}
}
```

事件 JSONL 由 `Telemetry.event(kind, **fields)` 写入，必须至少覆盖：
`birth` / `death` / `discover` / `verify` / `build` / `contract` / `extinction` /
`disturbance` / `coop_output` / `solo_output`。每条事件自带 `t`、`step`、`seed`、`group`。

---

## 5. 运行流程（集成层伪代码）

```python
telemetry = Telemetry(window_s=60.0, log_path=events_path, seed=seed, group=group)
for step in range(steps):
    population.step(dt)            # 规划→教师→神经→world.step→社会/人口
    telemetry.step(world, population, dt)
    if world.t >= next_window_t:   # 集成层也可以只用 Telemetry 自动结算
        telemetry.window_summary()  # 读取当前窗口，供 CivilizationState.history
telemetry.to_json(summary_path)
telemetry.close()
```

约束：

- `telemetry.step()` **只读**，不得改变世界；有专项测试
  （`tests/test_telemetry.py::TestRealSmallWorld::test_telemetry_does_not_mutate_world`）。
- 死亡/出生/建造/验证/扰动/合作产出等**文明事件**必须显式 `event(...)`，
  否则 `extinction_cause`、`cooperation_benefit`、`recovery_after_disturbance`
  只能退化到默认值。
- 训练奖励与文明指标分开记录：本系统无奖励函数，目标收益用 `event("utility", value=…)`
  记录，**不参与**训练。

---

## 6. 失败案例与波动范围报告要求

1. **不得只报成功运行**：每组 ≥5 个 seed，报告每个 seed 的原始指标 + 汇总
   （中位数、最小/最大值、四分位距）；禁止只贴一次"好看"的运行。
2. **禁止把单次结果写成结论**：`8` 人上限 + 随机出生/死亡意味着种群曲线方差大，
   单次运行只能作为个案（case），不能论证组间差异。
3. **必须报告失败/异常分类**：
   - `extinct_before_400s`（早于预期灭绝）；
   - `not_extinct_within_budget`（20000 步未灭绝）；
   - `population_limit_mismatch`（`total_agents_created > 8` 视为实现 bug，不是实验结论）；
   - `exception`（任何 traceback、NaN 指标、JSON 不可序列化）；
   - `knowledge_lost`（已验证知识存在但 `knowledge_retention == 0`，说明失传）。
4. **波动范围**：至少给出 `extinction_time`、`survival_rate`、`tech_verified`、
   `goal_completion_rate`、`no_progress_repeat_ratio` 五项跨 seed 的区间；
   区间重叠时，结论必须写成"未观察到差异"，不得写成"无影响"。
5. **不可比标记**：`dt`、世界尺寸、训练预算、感知开关、代码 commit 任一不同，
   两份运行必须在报告里标 `comparable: false`，不得合并统计。

---

## 7. 已知混淆因素（写入最终报告，不隐藏）

1. **动作熵 ≠ 智能**：动作熵只是行为类别分布的归一化熵。随机策略、卡死状态机、
   教师接管都能产生"高熵"。本系统是行为克隆（BC），没有奖励学习信号，
   **不能**用动作熵或生存率单独证明"更智能"。必须与
   `tech_verified` / `goal_completion_rate` / `knowledge_retention` /
   `no_progress_repeat_ratio` 联合解释。
2. **技术数量 ≠ 文明水平**：`tech_verified` 是"已验证条目数"，不代表掌握人数、
   能否复用、会不会失传。需同时看 `knowledge_retention`、`tech_graph_depth`、
   以及"掌握者集合"（死亡档案中的 `known_knowledge_ids`）。
3. **8 人小样本不可单次论证**：创建上限 8 使种群统计功效极低；灭绝/存续在
   种子间可剧烈摆动。任何"某组更好"的结论都需要 ≥5 seeds 且报告区间。
4. **教师接管比例**：Lead 短探针记录约 58% 的决策来自手写 `SocialLife`。
   教师接管比例高时，系统的"行为"其实是规则写的；报告必须单独记录教师接管步数
   比例（`teacher_override`）并把文明指标解释限定在"规则 + 规划 + 知识"层。
5. **"无进展"两口径不可混用**：严格快照口径（`state_identical_ratio`）在生理量
   连续变化时恒≈0；主口径（`no_progress_repeat_ratio`）才是诊断停滞的依据。
   混淆两者会得出"系统从不停滞"的错误结论。
6. **`tech_actions_per_1000_steps` 只是代理**：它统计"材料单位正增量"（私有库存
   获得 + 社区公共材料积累），不是真实动作尝试次数；一次 contribute 可能记 3 个
   单位、失败的 experiment 不产生材料而不计数。引用时必须说明这一局限，
   并与 `event("harvest"/"craft"/…)`、`Technology` 计数器交叉核对。
   之所以不用 `behavior_state` 上升沿：个体可以一直停留在 gather/craft 等技术
   状态内反复动作，实测中状态转移全部发生在技术状态之间，上升沿恒为 0。
7. **时间尺度被压缩**：寿命 900 世界秒、妊娠 24 秒；与真实文明时间不可类比。
8. **感知开关影响速度不影响决策**：Lead 实测关闭感知后约 4933 步/s（感知约占 95%
   耗时），且对决策逐比特无影响（`neural.features` 忽略 perception 参数）。
   可用于节省预算，但报告中必须注明，且所有组一致。
9. **初始种群的性别/年龄结构**会显著影响前 200 秒的出生与灭绝路径；
   不同 seed 的初始位置也可能改变社区建成时刻。
10. **奖励缺口**：主循环每步标量奖励恒为 0；任何"奖励提升"表述都是错误的。

---

## 8. 复现命令

### 8.1 指标模块自测（本仓库当前可运行）

```powershell
cd C:\Users\27826\Desktop\vscode_win_ws\ds_ws\newworld\newworldpro
$env:TEMP = (Resolve-Path .tmp).Path
python -m unittest tests.test_telemetry -v
```

### 8.2 消融实验（已落地，可直接运行）

```powershell
cd C:\Users\27826\Desktop\vscode_win_ws\ds_ws\newworld\newworldpro
$env:TEMP = (Resolve-Path .tmp).Path

# 单组单 seed（写 events JSONL + summary JSON）
python -m newworld.simulation --group D --seed 0 --steps 24000 --dt 0.05 `
    --log-dir experiments/results/smoke --json experiments/results/smoke/D_seed0.json

# 全消融网格（A/B/C/D × seed 0,1,2；输出 runs.csv / aggregate.json / report.md）
python experiments/run_ablation.py --groups A B C D --seeds 0 1 2 `
    --steps 24000 --dt 0.05 --tag final
```

> **与本节初稿的差异（已按实际落地修订）**：
> 1. 实际入口是 `python -m newworld.simulation` 与 `experiments/run_ablation.py`
>    （不是 `newworld.experiments.run`）。
> 2. `dt` 由 **0.1 改为 0.05**：P0 实测证明 `dt=0.5` 会使教师规则的近距离导航失效，
>    0.1 也偏离原始行为；0.05 与 Lead 基线探针一致（见 §4.2）。
> 3. 步数 `24000`（= 1200 世界秒），覆盖 900s 寿命的完整生命周期。
> 4. **A 组的定义**：A 组是"原始机制 + P0 机制缺陷修复"（食物链修复、教师生存优先、
>    受约束创建），因为未修复的原始实现根本不能形成可比较的基线。
>    未打补丁版本的实测基线（t=537.55s 全族饥饿灭绝）由
>    `experiments/probe_long.py` 单独保存，作为"修复前"证据，不与 A 组结果混用。

### 8.3 原始系统基线探针（Lead 已落地，可直接运行）

```powershell
cd C:\Users\27826\Desktop\vscode_win_ws\ds_ws\newworld\newworldpro
python experiments/probe_baseline.py --steps 2400 --dt 0.05 --tag pre
python experiments/probe_long.py     # 输出 experiments/results/baseline_long_pre_seed0.json
```

---

## 9. 变更记录

| 日期/版本 | 变更 |
| --- | --- |
| v1 | 冻结 §8 指标表；固定 A/B/C/D 预算；加入两个"停滞"口径与三个附加诊断指标；加入 Lead P0 基线（短探针/长时程）作为 A 组复现验收口径 |
