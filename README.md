# newworld — 像素世界盒子（V3）+ 微型文明（v0.9.2）

> **V3 重构（当前默认）**：世界盒子风格的俯视地图世界——
> 一张 512×320 的岛屿地图（值噪声确定性生成：深海/浅海/沙滩/草地/森林/山地/雪原），
> **千人群伙**（numpy 结构化数组，纯数据粒子）在上面繁衍、迁徙、觅食，
> 人口自然收敛到**动态承载力**（食物再生 vs 消耗的平衡，实测 ≈5000）。
> 你是神：工具栏选 **陨石 / 闪电 / 洪水 / 治愈之雨**，点击地图施放。
>
> ```powershell
> python -m newworld              # 新世界（俯视地图 + 上帝之手）
> python -m newworld --classic    # 旧世界（侧视像素小人 + 8 人文明）
> ```
>
> 旧的"微型文明"世界（目标规划/知识/记忆/神经策略）仍完整保留，
> 加 `--classic` 进入——两条世界线共用一套测试与实验框架。
>
> **v0.9.1 本轮升级**：
> - **鼠标主导交互**：点击个体选中/查看详情、右侧面板、底部控制栏（开始/暂停/继续/
>   单步/重置/速度）、智能体列表、事件日志、滚轮缩放、拖动平移；键盘仅作可选辅助。
>   鼠标选择**只观察**，不暂停仿真、不改变行为。
> - **真实学习型决策**：新增高层目标选择策略（REINFORCE，真实奖励/采样/参数更新/
>   checkpoint/独立评估）；修复神经网络在逐帧决策中只占 0.3% 的问题（现约 66%）；
>   修复 BC 训练数据 16 维文明上下文全零的"死特征"。
> - **有限组合式创新**：属性规则驱动的两技术组合 → 真实材料成本 → 真实世界效果
>   （采集/负重/恢复），上限 3 项/世界。
>
> 底层动作策略仍是行为克隆 MLP（输入 40 → 64 → 32 → 58）；旧 24 维 checkpoint
> **零初始化迁移**，行为逐比特一致。
>
> **v0.9.2 V2 升级**（回应"感知与决策断开 / BC 天花板 / 8 人上限 / 屏幕即世界"）：
> - **感知进入决策**：观测 **40 → 56 维**，新增 16 维**真实视网膜特征**
>   （低分辨率学习档，可注入视网膜参数）；`perception_stride` 控制刷新；
>   旧 24/40 维 checkpoint 双链零初始化迁移、逐比特一致。
> - **低层动作策略从 BC 升级为真实 RL**：新增 `rewards.py`（只奖励真实世界变化，
>   含反刷分测试）与 `lowlevel_rl.py`（REINFORCE，手写梯度经有限差分校验），
>   语言头保持 argmax 不参与 RL；训练脚本 `tools/train_lowlevel_rl.py`。
> - **人口上限可控**：`--cap` / `RunSpec.max_agents_created` 可放开做对照实验
>   （硬约束语义不变：累计创建、死亡不退还）。
> - **无限世界**：`--infinite` 开启确定性分块 `InfiniteScene`，屏幕只是局部窗口，
>   人口迁移时世界自动滚动（`window_origin_px` 可审计）。

![社会世界](assets/gui_pixel_world.png)

---

## 1. 快速开始

```powershell
# 依赖：Python 3.11+，numpy 与 pillow（见 requirements.txt）
cd C:\Users\27826\Desktop\vscode_win_ws\ds_ws\newworld\newworldpro
python -m pip install -r requirements.txt

# 新世界（默认）：俯视地图 + 千人群伙 + 上帝之手（陨石/闪电/洪水/治愈）
python -m newworld

# 实验平台（P2 推荐）：浏览器界面 + 实时统计 + 个体详情 + 干预审计
python -m newworld.lab_server          # 打开 http://127.0.0.1:8600

# 旧世界：侧视像素小人文明（V1/V2 机制全部保留）
python -m newworld --classic --windowed --size 1280x800

# 旧世界：无人工鼠标自检（13 项断言，输出 JSON）
python -m newworld --classic --ui-test .tmp\ui_test.json --windowed --size 1280x800

# 旧世界：无窗口单个体仿真 / 无窗口文明实验（A/B/C/D 消融）
python -m newworld --classic --headless --steps 600 --log run.jsonl
python -m newworld.simulation --group D --seed 0 --steps 24000 --dt 0.05

# 训练高层目标策略（真实 REINFORCE）
python tools/train_goal_policy.py --episodes 30 --steps 4000 `
    --eval-seeds 100 101 102 103 104 105 106 107 108 109 --out assets/goal_policy.npz

# 本轮升级的 6 配置 × 10 seed 正式实验
python experiments/run_upgrade_experiments.py --seed-start 100 --num-seeds 10

# V2：低层 RL 训练 + 感知/RL/cap/无限世界实验矩阵
python tools/train_lowlevel_rl.py --episodes 30 --steps 4000 --seeds 0 \
    --eval-seeds 100,101,102,103,104,105,106,107,108,109 --eval-steps 4000 \
    --out assets/lowlevel_rl.npz --json experiments/results/lowlevel_rl_training.json
python experiments/run_v2_experiments.py --seed-start 200 --num-seeds 6 --tag final

# GUI 无限世界（旧世界线：屏幕只是窗口，人口迁移时自动滚动）
python -m newworld --classic --infinite --windowed --size 960x600

# 测试（沙箱环境请先设置可写 TEMP；见 §9）
python -m unittest discover -s tests
```

桌面快捷方式：`newworld 像素世界`（`start_world.bat`）。
图标由角色蓝图 `ref/blueprint.pdf` 生成（16–256 px 多尺寸 `.ico`）：

```powershell
# 1) 从蓝图 PDF 重新生成图标（assets/newworld.ico + 预览 PNG）
python tools/make_icon.py

# 2) 创建/修复桌面快捷方式并绑定该图标（写入桌面，需要相应权限）
powershell -ExecutionPolicy Bypass -File tools/update_shortcut.ps1
```

> 当前桌面上的 `newworld 像素世界.lnk` 已使用该图标；若项目目录移动，
> 重新执行上面的第 1、2 步即可修复。
鼠标操作完整说明见 `docs/USER_GUIDE.md`；决策架构见 `docs/IMPLEMENTATION_REPORT.md`；
实验结果见 `docs/EXPERIMENT_REPORT.md`。

---

## 2. 硬约束：最多创建 8 个 Agent（旧世界 · --classic）

> V3 新世界没有此约束（人口由食物承载力决定，实测可达数千）。
> 以下内容适用于 `--classic` 的旧世界。

- `CivilizationConfig.max_agents_created = 8` 约束的是**历史累计创建数**，
  不是当前存活数。`world.registry.total_created` 只增不减。
- 唯一受约束的创建路径是 `World.create_agent()` →
  `AgentRegistry.register()`；第 9 次创建直接返回 `None` 并记录拒绝
  （`world.creation_denied`）。
- `World.spawn()` 保留为几何/测试原语（不做预算检查），模拟内部不得使用。
- 死亡个体：先归档 `AgentRecord`（`died_at` / `cause` / 库存 / 掌握知识 / 技能 /
  记忆 / 亲子关系），再从存活列表移除，脑对象销毁 → **不能再行动**。
- 8 个名额用完后，世界不会再生新个体；**即使存活为 0 也不会复活人口**。
- "当前存活人口"（`registry.alive_count()`）与"历史累计创建人口"
  （`registry.total_created`）是两个独立指标，调试层与 telemetry 分别展示。

## 3. 决策层：规划 → 教师 → 神经网络（旧世界 · --classic）

每个个体每帧按顺序尝试：

1. **基础设施反射**：能量低时前往活跃的粮仓/篝火/小屋休息（技术真实改变生存）；
2. **目标规划层** `goals.GoalManager`：由需求/库存/知识候选/设施损耗/种群名额
   生成目标 → 计划（gather→craft/build/experiment）→ 结构化步骤执行；
   无进展 45 世界秒/240 步 → 重规划；连续 3 次失败 → 放弃 + 冷却；
3. **社会技能教师** `social.SocialLife`：采集/制作/建造/求偶；
   **生存优先**——能量/水分过低时返回 `None`，把控制权交回下层；
4. **神经网络** `neural.NeuralPolicy`：40 维观测 → 64 → 32 → 58（行为克隆）。

> 原始实现的缺陷修复：教师不再在社区建成后永久接管（原实现导致全员饿死，
> 实测 t=537s 全族灭绝）；浆果被采空后能再生；小世界不再缺少浆果丛。

## 4. 文明层：知识、技术、设施、传承（旧世界 · --classic）

- **技术目录（世界规则，非剧情）**：石斧、纤维绳、背篓、篝火、水井、小屋、粮仓。
  每条都有前置技术与真实数值效果（采集×2、负重+4、夜间/休息恢复、取水半径、
  食物储存）。发现顺序由个体行为与随机实验决定，允许永远不发明。
- **知识状态机**：`hypothesis → verified → invalidated`；
  只有 `verified` 可进入 `craft/build` 计划；每次验证都走
  `Technology.simulate_recipe`（真实扣材料、产产物、按难度可能失败）。
- **依赖图**：`KnowledgeStore.edges()/depth()`；只有已验证子图计入技术深度。
- **知识传播**：邻近教学（距离 + 概率规则）、出生继承（已验证知识 + 程序记忆）、
  死亡后档案永久保留（可追踪"谁发现/谁掌握/何时遗失"）。
- **公共设施**：建造后按半径提供真实效果（能量/水/休息），随时间损耗；
  维护消耗材料。设施失效会影响 `collapse_state`。
- **社会合作**：饱食个体把能量分享给附近濒危个体（真实资源转移，计入
  `cooperation_benefit` 与 `cooperation_index`）。

## 5. 观测与记忆（神经网络扩维 · 旧世界 --classic）

- 输入 24 → **40**：原 24 维感知/生理 + 8 维记忆摘要 + 4 维目标/计划 + 4 维文明上下文。
- 旧 24 维 checkpoint 自动迁移：新增输入列权重为 0，输出**逐比特一致**；
  `expanded_observation=False`（A/B 组）时上下文恒为 0。
- 个体情景记忆 / 程序记忆保存在 `MemoryStore`，死亡后仍保留（`archive`）。

## 6. 环境、昼夜、物理（旧世界 --classic）

- 昼夜（600 s/天）、太阳高度、气温、阴影；原子级破坏/生长、土壤水循环。
- 力学（重力 9.80665、弹道跳跃）、声学（343.2 m/s、语言气泡）、热学、电磁学/黑体太阳、视觉（瞳孔/视网膜/扫视）、听觉。
- **环境扰动（P4，seed 驱动）**：干旱（土壤湿度骤降、停止降雨）、风暴（摧毁浆果丛）、
  资源枯竭（移除整棵树/整块岩石）。事件写入 JSONL，可影响发展路线。

## 7. 代码结构

```text
newworld/
├── 【V3 新世界：俯视地图 · 世界盒子】
│   ├── map_engine.py     ← 地形真相：terrain/food/burning 网格（值噪声确定性生成）
│   ├── swarm.py          ← 千人群伙：SoA 结构化数组，全向量化（承载力涌现）
│   ├── catastrophes.py   ← 上帝之手：陨石/闪电/洪水/治愈（作用真实网格）
│   ├── map_render.py     ← 俯视渲染：底图 PIL + 饱足度渐变散点
│   └── app_map.py        ← 新 GUI：工具栏 + 整图画布 + 右侧统计面板
├── 【旧世界：侧视像素小人文明（--classic 进入）】
│   ├── population.py    ← 集成层：决策顺序、创建/死亡/继承/教学/设施/指标
│   ├── goals.py         ← P1 目标/计划/规划器（反卡死、重规划）
│   ├── knowledge.py     ← P2 技术目录、验证状态机、依赖图
│   ├── memory.py        ← P3 情景/程序记忆、传承、死亡归档
│   ├── civilization.py  ← P3 历史注册表、设施、契约、collapse/灭绝
│   ├── telemetry.py     ← P0/P4 指标采集（19 指标）+ 事件 JSONL
│   ├── simulation.py    ← headless 文明运行器 + 环境扰动 + CLI
│   ├── world.py         ← 世界 + 注册表/知识/记忆/文明服务 + 存档
│   ├── technology.py    ← 数据驱动配方执行、验证、设施效果
│   ├── society.py       ← 关系图网络、生育/衰老/死亡（死亡回调归档）
│   ├── neural.py        ← 56 维输入 MLP（24 生理 + 16 文明 + 16 视觉）+ 旧迁移
│   ├── training.py      ← 行为克隆训练（保留原流程）
│   ├── social.py        ← 技能教师（生存优先）
│   ├── visual_features.py← V2：真实视网膜 16 维特征（光学/眼睛/视网膜路径）
│   ├── rewards.py       ← V2：低层 RL 的真实世界奖励（含反刷分）
│   ├── lowlevel_rl.py   ← V2：低层动作 REINFORCE（minibatch/优势标准化/裁剪）
│   ├── infinite_scene.py← V2：无限世界确定性分块 + 窗口平移
│   └── pixelworld.py / ecology.py / physiology.py / physics.py / daynight.py …
├── app_tk.py / render.py  ← 旧世界交互与渲染
└── cli.py                 ← 入口分流：默认新世界，--classic 旧世界
experiments/
├── probe_baseline.py / probe_long.py / probe_cap_path.py  ← P0 诊断
├── run_ablation.py                                        ← A/B/C/D × 多 seed
└── results/                                               ← 实验产物（JSON/CSV/报告）
docs/
├── UPGRADE_V3_REPORT.md     ← V3 重构报告（决策/实测数据/已知边界）
├── USER_GUIDE_V3.md          ← 新世界操作指南（工具栏/灾害/面板）
├── architecture.md           ← 双世界架构与数据流（V3 §0/§9 + 旧世界 §1-8）
├── IMPLEMENTATION_REPORT.md  ← V1/V2 实施记录：鼠标交互 / 决策链路 / 组合式创新
├── EXPERIMENT_REPORT.md      ← V1/V2 10 seed × 6 配置实验与诚实结论
├── USER_GUIDE.md             ← 旧世界鼠标操作与面板说明（--classic）
└── UPGRADE_V2_REPORT.md      ← V2 感知/RL/上限/无限世界报告（含真实实验）
（V1 接口契约与 P0 审计报告已删除，历史见 git log）
```

## 8. 测试

```powershell
cd newworldpro
$env:TEMP = (Resolve-Path .tmp).Path   # 沙箱/受限环境下需要（见 §9）
python -m unittest discover -s tests
```

当前 **544 个测试 / OK（skipped=1）**：
V3 新世界 19 个（地形/群伙/灾害，含千人性能门槛）+ RL 小批量训练 9 个 +
原有世界/物理/生态/神经/训练/视觉/社会测试 + 目标规划、知识库、记忆、
文明注册表/设施、指标、硬约束/死亡、存档、观测迁移（24→40→56 双链）、
决策链、感知特征、低层 RL、奖励、无限世界、V2 集成等（旧世界线全保留）。

## 9. 沙箱 / Windows 临时目录说明

在受限沙箱中 `tempfile.mkdtemp()` 创建的目录可能不可写（Windows ACL），
会让 6 个与临时目录有关的测试出现假错误。项目提供 `tests/_env_compat.py`
垫片（把 `mkdtemp` 替换为普通创建，语义不变），相关测试模块已引入；
运行时建议把 `TEMP` 指到工作区内（如上）。

## 10. 查看实验结果（旧世界文明实验）

```powershell
# 单次运行：完整指标 + 事件 + 知识/设施状态
python -m newworld.simulation --group D --seed 0 --steps 24000 --dt 0.05 --json run_D0.json

# 消融：4 组 × 3 seed，输出 runs.csv / aggregate.json / report.md
python experiments/run_ablation.py --seeds 0 1 2 --steps 24000 --dt 0.05
Get-Content experiments\results\ablation_final\report.md
```

指标定义、实验数据与局限见 `docs/EXPERIMENT_REPORT.md`；实施与已知问题见
`docs/IMPLEMENTATION_REPORT.md`；V2（感知/RL/无限世界）见 `docs/UPGRADE_V2_REPORT.md`。

```powershell
# V2 矩阵：感知开/关、BC/RL/随机、cap 8/32、有限/无限世界
python experiments/run_v2_experiments.py --seed-start 200 --num-seeds 6 --tag final
Get-Content experiments\results\v2_final\report.md
```

## 11. 已知限制（如实）

**V3 新世界**：全管线 41FPS@8000 人（瓶颈在散点绘制，仿真本身 1.9ms/帧）；
整图可视暂无缩放平移；部落/战争/瘟疫未实现（二期）；详见
`docs/UPGRADE_V3_REPORT.md` §6。以下是旧世界的限制：

- 低层 RL 只训练 mode/move/jump/speak 四个决策头（语言头保持 argmax；
  规划/知识层仍用规则 + 代价搜索）。奖励只覆盖四类真实变化，仍有被"资源囤积"
  利用的空间（库存满后不再增益，已测）；大规模训练前应做奖励敏感性实验。
- 8 人累计创建上限是**默认值**，现在可用 `--cap` 放开做对照；但放开只有创建预算，
  不是人口学模型（没有生育压力/年龄结构/资源承载的标定）。
- 视觉特征是**低分辨率学习档**（不是 GUI 高保真档），且每 `perception_stride`
  帧刷新一次（默认 4，实验 8）；短预算训练**不足以**证明视觉显著改变行为。
- 无限世界是"横向无界 + 窗口跟随"的最小实现：纵向固定，不跨窗口保存远距离
  设施历史；长跑/大人口的墙钟成本未充分验证。
- `deliver` 目前是 contribute 别名；维护仅在设施失效后触发；环境扰动只有三种。
- 时间尺度为可观测性压缩（寿命 900 世界秒），不是人口学标定。

## 12. 参考

- Bonabeau, *Agent-based modeling*, PNAS 2002: <https://www.pnas.org/doi/full/10.1073/pnas.082080899>
- Epstein & Axtell, *Growing Artificial Societies*: <https://books.google.com.sg/books?id=8sXENe8QrmYC>
- *Multi-Agent, Human-Agent and Beyond: A Survey on Cooperation in Social Dilemmas*:
  <http://export.arxiv.org/pdf/2402.17270v2>
- 既有人眼/物理/声学与角色蓝图参考见 `ref/`。
