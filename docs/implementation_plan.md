# 改造实施计划（分阶段，可回滚）

> 本文件记录 **实际执行** 的分阶段改造计划与状态，不是事后润色的愿望清单。
> 依据：`docs/architecture_audit.md` 的审计结论。每个阶段都可单独运行、单独验证；
> 所有新机制由 `config.CivilizationConfig` 的开关控制，关闭后行为回到原始系统
> （A 组），因此可以随时回滚。
>
> 基线快照：git 提交 `926af6e`（改造前 v0.8.0，111 个测试）。

---

## 0. 审计结论驱动的改造顺序

审计发现的问题按"阻塞文明演化"的程度排序，改造也按此顺序进行：

| # | 问题（审计证据） | 对应阶段 | 处理方式 |
| --- | --- | --- | --- |
| 1 | 主循环无奖励、无目标、无记忆；教师接管 58% 决策 | P1 | 新建目标/计划层，插在教师与神经之上 |
| 2 | 死亡即删除，历史/知识全部丢失；累计创建可破 8 | P3 | 历史注册表 + 死亡档案 + 总创建预算 |
| 3 | 技术只有一条硬编码配方 | P2 | 数据驱动 recipe + 依赖图 + 模拟器唯一验证 |
| 4 | 世界没有食物（小世界 0 个浆果丛）、浆果不可再生、教师饿死全员 | P0 修复 | 修 `pixelworld`/`ecology`/`social` 的机制缺陷 |
| 5 | 无公共设施/无维护/无环境扰动/无崩溃恢复判定 | P4 | 设施 + 维护 + 扰动 + collapse 状态机 |
| 6 | 观测缺记忆/目标/社会上下文 | P3 | 24→40 零初始化扩维 + 旧 checkpoint 迁移 |
| 7 | 没有指标与实验入口 | P0/P4 | telemetry + headless runner + 消融脚本 |

---

## 1. P0：审计、日志、基线、机制修复

**目标**：可复现地测量原始系统，找出"重复动作/不能持续"的真实机制原因。

| 工作项 | 产物 | 状态 |
| --- | --- | --- |
| 只读代码审计（11 项任务书问题） | `docs/architecture_audit.md` | ✅ |
| 动作分布/奖励/状态变化/重复动作探针 | `experiments/probe_baseline.py` + `experiments/results/*.json` | ✅ |
| 长时程人口/文明探针（真实 dt） | `experiments/probe_long.py` | ✅ |
| 上限代码路径受控复现（改造前破 8） | `experiments/probe_cap_path.py` | ✅（证明旧代码创建了第 9 个） |
| 指标采集器 | `newworld/telemetry.py`（34 测试） | ✅ |
| 世界食物链缺陷修复 | `pixelworld.py`（保证最少浆果丛）、`ecology.py`（浆果可再生） | ✅ |
| 教师优先级缺陷修复 | `social.py`（生存优先于求偶/建造） | ✅ |
| 沙箱临时目录兼容 | `tests/_env_compat.py`（消除 6 个环境假错误） | ✅ |

**P0 验收（实测）**

- 原始系统基线：2 初始个体 → 8 累计创建 → **t=537.55s 全族饥饿灭绝**；
  `social` 行为占 82.7% 个体-步；主循环每步奖励恒 0；累计创建确实可突破 8。
- 探针可复现（固定 seed/配置/模型版本，结果落盘 JSON）。

---

## 2. P1：目标管理器 / 计划执行 / 重规划

**目标**：至少一个多动作任务可自动完成；能识别失败与无进展；环境变化可重规划。

| 工作项 | 文件 | 状态 |
| --- | --- | --- |
| Goal / PlanStep / Plan / Planner / GoalManager | `newworld/goals.py`（37 测试） | ✅ |
| 目标生成（需求/机会/设施/知识候选驱动） | `goals.py::_candidate_goals` | ✅ |
| 计划执行（结构化 verb → `Decision`） | `goals.py::_run_verb` | ✅ |
| 无进展检测（45 世界秒 / 240 步）→ replan | `goals.py` | ✅ |
| 连续 3 次 replan → abandon + 60s 冷却 | `goals.py` | ✅ |
| 实验计划先补采材料（Lead 补丁） | `goals.py::_plan_goal(experiment)` | ✅ |
| 候选技术轮转（Lead 补丁，避免卡在难题上） | `goals.py::_knowledge_goals` | ✅ |
| 工具/社区等世界入口 | `technology.py` 扩展 | ✅ |

**P1 验收（实测，group B/D）**

- 多步任务自动完成：`gather ×N → craft/experiment → build` 链在真实世界跑通；
- 实测 group D 生命周期：`goals_created=1178`、`completed=1093`、`abandoned=77`、
  `replans=233`、`plan_success_rate≈0.82`、`cost_estimate_error≈0.54`；
- 见 `tests/test_goals.py`（37 用例）与 `experiments/results/ablation_final/report.md`。

---

## 3. P2：持久化知识、技术依赖图、工具组合

**目标**：合法动作取得材料 → 尝试制作 → 模拟器验证；已验证知识可复用；
新技术真实改变世界数值。

| 工作项 | 文件 | 状态 |
| --- | --- | --- |
| Recipe / Knowledge / ValidationRecord / KnowledgeStore | `newworld/knowledge.py`（40 测试） | ✅ |
| 技术依赖图（prerequisites + recipe 前置） | `knowledge.py::edges/verified_edges/depth` | ✅ |
| 状态机 hypothesis→verified→invalidated | `knowledge.py` | ✅ |
| 模拟器唯一验证入口 | `technology.py::simulate_recipe` | ✅ |
| 前置失败不计为技术缺陷（Lead 补丁） | `knowledge.py::PRECONDITION_REASONS` | ✅ |
| 技术真实效果：采集产量/负重/恢复/取水 | `technology.py` | ✅ |
| 新增"粮仓"食物技术（让群体超过自然承载力） | `knowledge.RECIPE_CATALOG` + `civilization.FACILITY_KINDS` | ✅ |
| 知识集成测试（前置/失效/效果/序列化） | `tests/test_knowledge_integration.py` | ✅ |

**P2 验收（实测）**

- group D 在一个生命周期内验证 4 条技术（石斧 / 纤维绳 / 篝火 / 粮仓），
  依赖图 depth=1、verified edges=5（跨种子略有差异）；
- 技术效果可测：石斧使单次采集 1→2；粮仓/篝火在半径内恢复能量；
- 只有 `verified` 的知识进入 `craft/build` 计划，`hypothesis` 只能走 `experiment`。

---

## 4. P3：个体记忆、社会学习、代际传承、人口上限

**目标**：创建总数严格 ≤8；死亡不可行动；知识可按规则跨个体/跨代保存并可追踪。

| 工作项 | 文件 | 状态 |
| --- | --- | --- |
| 情景/程序记忆、8 维摘要、传承、死亡归档 | `newworld/memory.py`（14 测试） | ✅ |
| 历史注册表（total_created 只增不减、死亡档案） | `newworld/civilization.py`（28 测试） | ✅ |
| 受约束创建路径（第 9 个返回 None 并记录拒绝） | `world.create_agent` + `population.create_agent` | ✅ |
| 死亡先归档再移除，死者脑对象销毁（不可再行动） | `population._on_death` / `society.update` | ✅ |
| 出生继承（程序记忆 + 已验证知识） | `population.note_birth` | ✅ |
| 邻近社会学习（教学/模仿，有距离与概率规则） | `population._social_learning` | ✅ |
| 观测扩维 24→40 + 旧 checkpoint 零初始化迁移 | `neural.py` | ✅ |
| 硬约束/死亡/传承/存档测试 | `tests/test_population_limits.py`、`test_inheritance.py`、`test_world_state.py`、`test_observation_migration.py` | ✅ |

**P3 验收（实测）**

- `tests/test_population_limits.py` 8 用例：累计创建 = 8，第 9 次返回 None 且拒绝被记录；
  死亡后 `steps`/`x` 不再变化；死亡档案含 inventory/knowledge/skills/lineage；
- `tests/test_inheritance.py`：子代继承已验证知识；关闭 `enable_inheritance` 后不继承；
  死者知识仍可从注册表读出；
- 迁移测试：旧 24 维 checkpoint 加载后新增列权重为 0，输出逐比特一致。

---

## 5. P4：合作、公共设施、社会规则、环境扰动、崩溃与恢复

**目标**：规则驱动下出现持续技术积累或可解释协作；扰动影响发展路线；允许灭绝；
消融实验能说明新增机制的实际差异。

| 工作项 | 文件 | 状态 |
| --- | --- | --- |
| 公共设施（损耗/维护/失效/效果半径） | `civilization.py` + `technology.apply_facility_effects` | ✅ |
| 设施使用行为（低能量时前往粮仓/篝火/小屋休息） | `population._infrastructure_directive` | ✅ |
| 环境扰动（干旱/风暴/资源枯竭，seed 驱动、可记录） | `simulation.EnvironmentDisturbance` | ✅ |
| collapse 五态机 + 灭绝时间/原因 | `civilization.py` + `population._enforce_extinction_rules` | ✅ |
| headless 文明运行器 | `newworld/simulation.py` | ✅ |
| 消融脚本（A/B/C/D × 多 seed，CSV+JSON+报告） | `experiments/run_ablation.py` | ✅ |
| 实验协议文档 | `docs/experiment_protocol.md` | ✅（命令段由 Lead 按实况更新） |

**P4 验收（实测，见 `experiments/results/ablation_final/report.md`）**

- 规则驱动的持续积累：D 组由 agent 自己发现并验证技术、建造设施、围绕设施定居；
- 环境扰动真实生效（scarcity 减少 18% 木/石原子；drought 停止降雨并加速蒸发；
  storm 破坏植物）并写入事件日志；
- 允许失败：A 组无技术 → 饥饿崩溃；D 组也会因 8 人封顶在老年期全灭；
- 消融差异：A（无规划/知识）tech=0、早崩；D 有 3–4 项已验证技术、设施与知识传播。

---

## 6. 回滚与兼容策略

1. **git 基线**：改造前快照 `926af6e`；每个阶段的改动都在工作区，可按文件回滚。
2. **配置开关**：`CivilizationConfig` 关闭全部机制 = A 组（原始机制 + 缺陷修复）。
3. **模型兼容**：24 维旧 checkpoint 自动零初始化迁移，`NeuralPolicy.load` 兼容 v2/v3。
4. **不删除旧接口**：`Society`/`Technology` 的旧方法（`craft`、`contribute`、
   `MAX_POPULATION` 常量）保留；`World.spawn` 保留为几何原语（不做预算检查），
   受约束创建走 `World.create_agent`。
5. **测试保护**：111 个原始测试全部保留（仅 6 处 tempdir 环境适配），
   新增 186 个测试；`python -m unittest discover -s tests` → **Ran 297 tests ... OK**。

---

## 7. 未完成 / 下一步

- 规划器为规则 + 代价搜索，**未**训练高层策略；BC 无奖励信号，记忆/目标特征的
  习得依赖规划层（已在最终报告列为已知限制）。
- `deliver` verb 目前是 contribute 别名；没有独立公共仓库/契约执行逻辑。
- 维护目标只在设施失效后触发（仍未实现"预防性维护"）。
- 环境扰动只有三种；没有气候长期趋势（变冷/变暖）。
- 8 人硬上限使单个生命周期必然灭绝（约 900–1050s），"跨代文明"只能通过
  知识档案+观察者视角呈现，不能通过人口延续呈现——这是任务书硬约束的直接后果。
