# 架构说明：模块关系与数据流（v0.9.1）

> 当前架构说明（v0.9.1：鼠标交互 + 学习型决策 + 组合式创新）。
> 项目审计见 `docs/PROJECT_AUDIT.md`；实施记录见 `docs/IMPLEMENTATION_REPORT.md`；
> 实验数据见 `docs/EXPERIMENT_REPORT.md`；鼠标操作见 `docs/USER_GUIDE.md`。
---

## 1. 总览：五层结构

```
┌──────────────────────────────────────────────────────────────────────┐
│ 观测/交互层   cli.py · app_tk.py · render.py · preview.py             │
├──────────────────────────────────────────────────────────────────────┤
│ 实验层        simulation.py (SocietyRunner / EnvironmentDisturbance)   │
│               experiments/run_ablation.py · probe_*.py · telemetry.py   │
├──────────────────────────────────────────────────────────────────────┤
│ 文明层        civilization.py  AgentRegistry · CivilizationState        │
│                               Facility · SocialContract                 │
│               knowledge.py     Knowledge · Recipe · KnowledgeStore     │
│               memory.py        MemoryStore · Procedure                 │
├──────────────────────────────────────────────────────────────────────┤
│ 决策层        goals.py         GoalManager · Planner · Goal/Plan        │
│               social.py        SocialLife（技能教师，生存优先）          │
│               neural.py        NeuralPolicy 40→64→32→58（BC）           │
│               behavior.py      InnerLife（稳态反应式行为）               │
├──────────────────────────────────────────────────────────────────────┤
│ 世界层        world.py         World · Character 注册 · 存档            │
│               technology.py    Inventory · Recipe 执行 · 公共设施效果    │
│               society.py       SocietyNetwork · 生育/衰老/死亡           │
│               pixelworld.py    PixelScene（原子地形/资源）              │
│               ecology.py       soil-water / growth / destruction        │
│               physiology.py · physics.py · daynight.py · environment.py │
│               acoustics.py · optics.py · eye.py · retina.py · perception│
└──────────────────────────────────────────────────────────────────────┘
```

依赖方向严格自上而下（下层不 import 上层）。`population.py` 是唯一的集成点：
它把文明层服务挂到 `World` 上，让决策层每帧读取，并把结果写回文明层。

---

## 2. 每帧数据流（`Population.step(dt)`）

```
for each living agent:
   ① infrastructure reflex   ← 能量低时前往活跃的粮仓/篝火/小屋休息
   ② GoalManager.update()    ← 目标选择 → 计划推进 → Decision（P1）
   ③ SocialLife.decide()     ← 手写技能教师（只有生存无虞时才接管）
   ④ NeuralInnerLife.decide()← 40 维观测 → MLP → Decision（BC 策略）
        ↓ 返回 Decision(move, jump, speak, word)
world.step(actions, dt)       ← 物理/生理/生态推进（世界规则唯一权威）
    ↓
speak / perceive（可选，实验关闭）
    ↓
Society.update(world, dt)     ← 亲密关系、怀孕、出生、衰老、死亡
    ├─ on_death → Population._on_death → AgentRegistry.mark_dead（档案）
    └─ birth   → Population.create_agent → World.create_agent（8 人预算）
    ↓
Population.sync()             ← 新个体建脑；死者脑对象销毁（不可再行动）
    ↓
CivilizationState.update()    ← 设施损耗、collapse 状态、灭绝检测
    ↓
technology.pending_facilities → CivilizationState.build_facility（设施落地）
    ↓
technology.apply_facility_effects（能量/水/休息）
    ↓
memory.record（情景记忆）、_social_learning（知识传播）
    ↓
Telemetry.step(world, population, dt)  ← 只读指标快照
```

关键不变量：

1. **世界规则唯一权威**：材料/技术/设施的变化只经过 `Technology`
   （`harvest` / `execute_recipe` / `simulate_recipe` / `apply_facility_effects`）；
   规划器与知识库只能"请求"。
2. **创建预算**：唯一受约束创建路径是 `World.create_agent → AgentRegistry.register`，
   累计创建 > `max_agents_created`(8) 时返回 `None` 并记录拒绝。
3. **死亡不可逆**：死亡先 `AgentRegistry.mark_dead` 归档（inventory/knowledge/skills/
   memory/lineage），再从 `world.agents` 移除；`Population.sync` 销毁其脑对象。
4. **观测扩维不改旧行为**：`FEATURE_DIM = 40`，旧 24 维 checkpoint 通过
   `NeuralPolicy.from_legacy_weights` 零初始化迁移，输出逐比特一致。

---

## 3. 模块职责表（新增/修改）

| 模块 | 状态 | 职责 | 关键接口 |
| --- | --- | --- | --- |
| `config.py` | 修改 | `CivilizationConfig` + `ABLATIONS`（A/B/C/D）+ `make_civilization_config` | `max_agents_created=8` |
| `character.py` | 修改 | Agent 状态（含 `parent_id/created_at/died_at/known_knowledge_ids/skills/current_goal_id`） | `to_dict()` |
| `world.py` | 修改 | 世界 + 注册表/知识/记忆/文明服务；受约束创建；存档/加载 | `create_agent` `adopt_agent` `save/load` `NoAgentsError` |
| `technology.py` | 修改 | 数据驱动配方执行、模拟器验证、设施效果、新材料（fiber/cord/basket） | `simulate_recipe` `execute_recipe` `can_afford` `apply_facility_effects` |
| `society.py` | 修改 | 生育/衰老/死亡；上限改查注册表；死亡回调 | `Society(on_death=...)` |
| `population.py` | 重写 | 集成层：决策顺序、创建/死亡/继承/教学、设施队列、telemetry | `Population(world, seed, telemetry=...)` |
| `goals.py` | 新增 | 目标/计划/规划器/执行器/反卡死 | `GoalManager.update(world, agent, dt)` |
| `knowledge.py` | 新增 | 技术目录、验证状态机、依赖图、保留率 | `KnowledgeStore.attempt(...)` |
| `memory.py` | 新增 | 情景/程序记忆、8 维摘要、传承、死亡归档 | `MemoryStore.feature_vector` |
| `civilization.py` | 新增 | 历史注册表、设施、契约、collapse/灭绝、合作指数 | `AgentRegistry` `CivilizationState` |
| `telemetry.py` | 新增 | 只读指标采集（19 指标）+ 事件 JSONL | `Telemetry.step/event/summary` |
| `simulation.py` | 新增 | headless 文明运行器 + 环境扰动 + CLI | `SocietyRunner(spec).run()` |
| `neural.py` | 修改 | 24→40 输入、旧 checkpoint 零初始化迁移 | `FEATURE_DIM` `from_legacy_weights` |
| `pixelworld.py` | 修改 | 保证世界有食物（最少浆果丛） | `_ensure_food_supply` |
| `ecology.py` | 修改 | 浆果可再生（修复被采空后不可再生的缺陷） | `Ecology._grow` |
| `social.py` | 修改 | 生存优先于求偶/建造（修复教师饿死全员） | `SocialLife.decide` |

---

## 4. 两套神经网络/环境的关系（保留原状）

- **主路径**：`population.py → neural.py`：40 维输入、MLP 64/32、58 输出、
  行为克隆训练（`training.py`）。这是应用与文明实验实际使用的策略。
- **并行路径**：`env.py::NewWorldEnv`：8 维观测 / 3 动作 / 可插拔奖励，
  供 RL 研究使用，未被主程序使用。改造**没有删除**它，也没有把它伪装成 RL
  训练循环——审计已说明主路径本来就没有奖励信号。
- 两条路径共享 `World` 与 `Character.observation`，互不破坏。

---

## 5. 数据对象与持久化

| 对象 | 位置 | 生命周期 |
| --- | --- | --- |
| 活体 `Character` | `world.agents` | 出生 → 死亡（随后移除，档案保留） |
| 历史 `AgentRecord` | `world.registry` | 永久（含 died_at/cause/inventory/knowledge/skills/lineage） |
| `Knowledge` | `world.knowledge` | 永久（hypothesis/verified/invalidated + validation_history） |
| `MemoryEntry` / `Procedure` | `world.memory` | 个体死亡后仍保留（archive） |
| `Facility` / `SocialContract` | `world.civilization` | 建造 → 损耗 → 失效（档案保留） |
| 指标 / 事件 | `telemetry` + `experiments/results/*` | 实验产物 |

存档：`World.save(path)` 写 JSON（state + registry + knowledge + memory +
civilization）；`World.load(path)` 重建，死亡个体不会被复活。
（进化 checkpoint：`assets/agents/agent_<id>.npz`，兼容旧 24 维格式。）

---

## 6. 消融分组与代码开关

| 组 | 规划 | 知识 | 传承/记忆 | 社会/设施/扰动 | 观测扩维 |
| --- | --- | --- | --- | --- | --- |
| A 原始 | ✗ | ✗ | ✗ | ✗ | ✗ |
| B +规划 | ✓ | ✗ | ✗ | ✗ | ✗ |
| C +知识传承 | ✓ | ✓ | ✓ | ✗ | ✓ |
| D 完整 | ✓ | ✓ | ✓ | ✓ | ✓ |

硬约束（8 个累计创建上限）**不属于**消融开关：所有组都必须满足。
关闭机制时，行为回退到原始实现（A 组仍保留 P0 的缺陷修复：食物再生、
教师生存优先、受约束创建），差异在 `docs/EXPERIMENT_REPORT.md` 中说明。

---

## 7. v0.9.1 升级后的模块关系（鼠标交互 + 学习型决策）

### 7.1 交互层（新增，鼠标主导）

```
app_tk.run_app
  ├─ ui_state.Layout            区域隔离：top / world(画布) / panel / bottom
  ├─ ui_state.Viewport          屏幕↔世界坐标、缩放锚点、平移 clamp、resize
  ├─ ui_state.HitTester         包围盒优先→最近邻（死人不选；缩放平移后仍正确）
  ├─ ui_state.SelectionModel    选中/切换/取消/死亡清理
  ├─ ui_state.ControlState      开始/暂停/继续/单步/速度档位（单步仅暂停时)
  ├─ ui_state.build_agent_detail / build_stats / build_event_log
  │                             ↑ 全部只读真实世界状态（缺失显示"暂无数据"）
  └─ 画布：世界背景 + 个体精灵 + 选中高亮环 + 任务标记（有活跃 Plan 的个体）
```

数据流：`Population.step()` → `World` 状态变化 → UI 每帧只更新画布几何
（`_refresh_frame` 有 key 去重）、面板 0.25s 刷新、背景 0.8s 刷新；
鼠标事件只改 `SelectionModel/ControlState/Viewport`，**不写世界状态**。

### 7.2 决策层（三层职责，本轮明确化）

```
高层：highlevel.GoalPolicy（REINFORCE，可选）+ goals.GoalManager 的目标生成
       ↓ 目标类型 / 目标生命周期（完成/失败/放弃/重规划）
中层：goals.Planner → 结构化计划步骤；explore 步骤标记为 delegated
       ↓
底层：Population._decide
        ├─ infrastructure 反射（低能量时去设施休息）
        ├─ 规划步骤动作（gather/craft/experiment/build/...）
        ├─ delegated → 直接调用 neural.NeuralInnerLife（BC 网络）
        └─ 其它 → social.SocialLife 教师（生存优先）
```

- `Population.decision_counts` 记录每一帧由哪层做出决定，写入实验结果的
  `decisions` / `share_*`，避免把规划器/教师的能力误报成"神经网络学会的能力"。
- `low_level_mode ∈ {full, random, off}` 与 `goal_policy_mode ∈ {heuristic, neural, random}`
  是决策链路的消融开关（同世界规则，仅切换决策模块）。

### 7.3 创新层

```
population._maybe_innovate（每 5 世界秒，enable_innovation=True）
  → innovation.InnovationLab.proposals(agent, knowledge, affordable_only=True)
      （仅 verified 技术、tag 相容对、材料可负担）
  → InnovationLab.attempt(agent, proposal, world)
      真实扣材料 → 组合难度判定成败 → 成功时
      technology.register_custom_effect(tool, effects)
  → 效果由 Technology 消费：harvest/capacity/facility effects/craft 成功率
```

### 7.4 新增模块职责

| 模块 | 职责 | 测试 |
| --- | --- | --- |
| `ui_state.py` | 视口/命中/选择/控制/布局/详情数据（纯逻辑） | 34 |
| `app_tk.py`（重写） | 鼠标主导界面 + 状态栏 + 列表 + 详情 + 控制 + 日志 + `--ui-test` | 7（GUI，可 skip） |
| `highlevel.py` | 30 维特征、GoalPolicy（MLP 30→32→8）、REINFORCE trainer、独立评估 | 23 |
| `innovation.py` | 属性/规则表、提案、真实材料成本、效果注册、上限 3 | 42 |
| `simulation.py`（扩展） | `GoalPolicyEnv` / `make_goal_policy_env`、策略模式/创新开关 | 决策链路 11 |
| `training.py`（修复） | BC 数据在 Population 内采集（上下文特征非零方差） | 决策链路 11 |
| `goals.py`（扩展） | `request_goal`/`has_active_goal`/`consume_delegation` | 37 |
| `population.py`（扩展） | 决策来源计量、策略挂载/转移/奖励、创新集成、低层消融 | 41 + 决策链路 |

### 7.5 硬约束仍然成立

无论鼠标选择、策略挂载、创新、重置，创建智能体只有一条受约束路径
（`World.create_agent → AgentRegistry.register`），实验 CSV 每行都记录
`total_agents_created`，aggregate 检查 ≤8；UI 选择不创建个体；重置 = 新运行（新注册表）。

---

## 8. v0.9.2（V2）新增结构：感知进入决策、低层 RL、无限世界

### 8.1 观测：40 → 56 维

```
neural.features(world, agent)
  ├─ 24 维：位置/速度/朝向/生理/昼夜/最近浆果/池塘/温度/跳跃/说话（v0.8）
  ├─ 16 维：文明上下文（记忆摘要/目标计划/知识/设施/人口/合作，P3）
  └─ 16 维：真实视网膜特征（V2.1）
        ↑ character.visual_features（Population 每 perception_stride 帧刷新）
        ↑ visual_features.VisualFeatureExtractor
            → VisualSystem.perceive（真实 optics/eye/retina/gaze）
```

迁移链：24 →（零初始化 16 上下文）→ 40 →（零初始化 16 视觉）→ 56；
`NeuralPolicy.from_legacy_weights` / `from_v2_weights` 分别处理旧 checkpoint，
新增权重为 0 时输出逐比特一致。

### 8.2 低层动作策略的 RL 回路

```
Population._low_level_decision
  mode="rl" → LowLevelPolicy.sample(56 维特征)
        → Decision(move=MOVE_VALUES[move_index], jump, speak)
        → 本帧待结算在 _pending_lowlevel
  step 结束 → LowLevelReward.observe(真实世界前后差值)
        → (agent_id, features, action, reward) 进入 lowlevel_transitions
        → 训练器 drain → 折扣回报 + 基线 → 手写梯度更新
        → 语言头 argmax 不变（不参与 RL）
```

奖励只来自真实变化（Δ技术/Δ设施/净资源/Δ知识/Δ合作 − 时间成本 − 死亡），
原地站立/粮仓挂机/重复失败动作 ≤0（见 `newworld/rewards.py` 与测试）。

### 8.3 无限世界：窗口化场景

```
World._make_scene → InfiniteScene（config.infinite=True）
    确定性分块：材料/对象/池塘/浆果 = f(seed, 全局原子坐标)
    修改持久化：全局坐标稀疏字典
World._maybe_recenter（每帧 step 末尾）
    人口进入窗口边缘 25% → 平移 25% 窗口
    → InfiniteScene.recenter(delta)   （重建局部 grid/objects/berries/pond）
    → World._shift_world_x(-delta_px) （agents/facilities/community 同步平移）
    → window_origin_px 记录全局偏移
```

局部坐标体系保持不变，因此 `render`/`ui_state`/`Technology` 无需感知无限世界；
有限世界（默认）路径完全不变。

### 8.4 新增模块与测试

| 模块 | 职责 | 测试 |
| --- | --- | --- |
| `visual_features.py` | 真实视网膜 16 维特征（低分辨率学习档） | 21 |
| `rewards.py` | 低层 RL 真实奖励 + 反刷分 | 11 |
| `lowlevel_rl.py` | 策略采样/手写策略梯度/训练器 | 19 |
| `infinite_scene.py` | 无限世界确定性分块与窗口平移 | 27 |
| `simulation.py`（扩展） | RunSpec visual/infinite/cap/lowlevel 策略 + RL env | 10 |
| `population.py`（扩展） | 视觉刷新、RL 采样/转移/奖励、无限 recenter 联动 | — |
