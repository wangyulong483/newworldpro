# 架构说明：模块关系与数据流（改造后 v0.9）

> 改造前的审计见 `docs/architecture_audit.md`；机制设计见 `docs/civilization_design.md`；
> 分阶段实施记录见 `docs/implementation_plan.md`；指标与实验协议见
> `docs/experiment_protocol.md`。

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
教师生存优先、受约束创建），差异在 `docs/experiment_protocol.md` 中说明。
