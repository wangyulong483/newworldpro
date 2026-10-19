# newworld 观察窗口用户指南（鼠标主导交互）

> 对应代码：`newworld/app_tk.py`（界面与交互）+ `newworld/ui_state.py`（纯逻辑）。
> 契约：`docs/upgrade_contracts_v1.md` §1。本文只描述**已经实现并验证过**的行为；
> 未验证或受限的项目在 §8「已知限制」中明确标出。

## 1. 启动

```powershell
cd newworldpro

# 全屏（默认，世界尺寸=屏幕物理分辨率）
python -m newworld

# 窗口化（推荐开发/观察）
python -m newworld --windowed --size 1280x800

# 指定随机种子（个体行为/地形不同）
python -m newworld --windowed --size 1280x800 --seed 7

# 启动约 0.9 秒后截图并退出（用于验证渲染）
python -m newworld --windowed --size 1280x800 --screenshot .tmp\ui_shot.png

# 自动关闭（开发用）
python -m newworld --windowed --size 1280x800 --autoclose-ms 2500
```

桌面快捷方式「newworld 像素世界」的图标由 `ref/blueprint.pdf` 生成
（`python tools/make_icon.py`），可用 `tools/update_shortcut.ps1` 重新绑定；
目标仍是 `start_world.bat --debug`。
窗口以**物理像素**布局（`enable_dpi_awareness()`），高 DPI 屏上像素画不会模糊。
窗口大小变化时通过 `<Configure>` 重建布局（`ui_state.Layout`），不会出现控件错位。

## 2. 界面分区

```
┌──────────────────────────────────────────────────────────────┐
│ 顶部状态栏：第 N 天 时:分 · 世界时间 · 运行状态 · 速度 ·     │
│             存活 · 累计创建/上限 · 死亡 · 剩余可创建 · 缩放/fps│
├───────────────────────────────────────────┬──────────────────┤
│                                           │ 个体列表（可点击）│
│        中央世界画布（可缩放/平移）         │ 选中个体详情      │
│                                           │ 世界统计（可折叠）│
├───────────────────────────────────────────┴──────────────────┤
│ 底部控制栏：开始 暂停 继续 单步 重置 速度- 速度+ 统计 面板 训练│
│ 事件日志（真实世界事件 + 操作记录）                            │
└──────────────────────────────────────────────────────────────┘
```

- 顶部状态栏的计数全部读 `world.registry`（`alive_count()` / `total_created` /
  `total_deaths` / `remaining()`）。达到 8 个累计创建名额后显示
  `⚠ 已达累计创建上限，无法继续创建`。
- 右侧「世界统计」面板可折叠（底部「统计」按钮或 `F1`）。
- 右侧整个面板可折叠（底部「面板」按钮）；折叠后画布变宽。

## 3. 鼠标操作（核心入口）

| 操作 | 效果 | 实现要点 |
| --- | --- | --- |
| 左键点击个体 | 选中该个体，画布出现黄色高亮环 | 命中测试 `ui_state.HitTester`：先精灵包围盒、再 60 屏幕像素内最近邻 |
| 左键点击另一个体 | 切换选中 | 选择模型 `SelectionModel.toggle` |
| 再次左键点击同一个个体的精灵 | 取消选中 | 同上（切换语义） |
| 左键点击空白处 | 取消选中 | 无命中 → `clear()` |
| 点击右侧「个体列表」某一行 | 选中该个体（与场景选择双向同步） | `<<ListboxSelect>>` → `SelectionModel` |
| 滚轮 | 以光标位置为锚点缩放（0.25×–4.0×） | `Viewport.zoom_at`：光标下的世界点保持不动 |
| 右键按住拖动 / 中键按住拖动 | 平移视图 | `Viewport.pan` + `clamp`（世界不会被拖出视图） |
| 底部按钮 | 开始 / 暂停 / 继续 / 单步 / 重置 / 减速 / 加速 / 统计 / 面板 / 训练 | 每个按钮都有对应处理函数 |

要点：

- **选中只改显示，不改仿真**：不暂停、不改变个体行为；仿真继续按实时步进。
- **单步只在暂停时生效**，并且严格推进当前世界配置的
  `world.config.fixed_dt`（默认 1/60 s）**恰好一步**；运行中按单步会被拒绝并写日志。
- **暂停仍可查看**：暂停后世界时间冻结，但详情/统计/日志/缩放/平移/选择都可用。
- **重置有确认对话框**（`tkinter.messagebox.askyesno`）：确认后构造一个全新的
  世界与人口（新种子），清空选择、清空列表并按新存活个体重建、恢复运行。
  取消则什么都不做。
- **UI 点击与世界点击隔离**：布局由 `ui_state.Layout.region_at` 划分为
  `top / world / panel / bottom`。画布控件只覆盖 `world` 区域；画布点击处理函数
  还会把坐标换算成窗口坐标再用 `region_at` 复核，落在 UI 区域的事件直接忽略。
  因此点击顶部状态栏、右侧面板、底部按钮/日志不会误选个体。

## 4. 键盘辅助（可选，不是唯一入口）

| 按键 | 行为 |
| --- | --- |
| `空格` | 暂停 / 继续（等价于点「暂停/继续」按钮，走同一个 `ControlState.toggle`） |
| `F1` | 显示 / 折叠「世界统计」面板 |
| `F2` | 从零训练所有个体的神经网络（后台线程，完成后写入日志） |
| `Q` / `Esc` | 关闭窗口 |

键盘与鼠标**共用处理函数**，不存在两条逻辑：例如空格和「暂停」按钮都会调用
`ControlState.pause()/toggle()`，不会重复推进仿真。

## 5. 面板数据来源（全部真实，缺失显示「暂无数据」）

`ui_state.build_agent_detail(world, population, agent_id)` 逐项读取：

| 面板字段 | 数据来源 |
| --- | --- |
| ID / 存活 / 年龄 / 代数 / 父母 | `world.registry.record(id)`（`AgentRecord`）+ `Character` 字段 |
| 能量 / 水分 / 睡眠压力 / 心率 / 呼吸 | `agent.physiology` |
| 健康 | 当前世界模型**没有** health 字段 → 固定显示「暂无数据」（不编造） |
| 库存 / 工具 / 容量 | `agent.inventory`（wood/stone/fiber/cord/tool/basket/capacity） |
| 当前目标 / 当前计划 / 目标进度 / 最近目标结果 | `population.goals.goals_of(id)`、`active_plan(id)`、`agent.current_goal_id`、`Goal.status/params` |
| 已掌握知识 / 技术状态 | `agent.known_knowledge_ids` + `world.knowledge.get(id)` 的 `name_zh/status` |
| 技能 | `agent.skills` |
| 记忆摘要 | `world.memory.entries(id, limit=5)` |
| 社会关系（前 3） | `population.society.affinity_between(id, other)` |

「世界统计」来自 `world.registry`、`world.knowledge.verified_ids()`、
`world.civilization.facilities/cooperation_index/collapse_state/metrics()`、
`population.goals.stats()`、`world.memory`。「事件日志」来自
`population.society.events`、每个存活个体的记忆条目、死亡档案、已验证技术、
公共设施建造记录；世界刚启动没有事件时，退回由 `registry` 计数组成的
「世界运行中」一行——**不会编造事件**。

刷新策略：详情/统计/日志/状态栏每 **0.25 s** 刷新一次（且内容不变时不重写
Text）；画布物体坐标每帧更新；状态未变时整帧跳过，不会每帧重建 widget。

## 6. 程序化自检 `--ui-test`

需要真实显示环境（tkinter 能创建窗口）。无需人工点击：

```powershell
cd newworldpro
$env:TEMP=(Resolve-Path .tmp).Path

python -m newworld --ui-test .tmp\ui_test.json --windowed --size 1280x800
# 打印 13 项 PASS/FAIL，写 JSON，全部通过 exit code = 0，否则 = 1
```

JSON 结构（`ensure_ascii=False, indent=2`）：

```json
{
  "ok": true, "total": 13, "passed": 13, "failed": 0,
  "checks": [{"name": "world_click_selects_agent", "ok": true, "detail": "点击 0 号精灵中心 → 选中，高亮环 显示"}],
  "world": {"size": "1280x800", "alive": 2, "total_created": 2, "total_deaths": 0,
            "t": 0.0, "selection": null, "playing": true, "speed": 1.0},
  "elapsed_s": 2.8, "python": "3.14.0", "tk": "8.6", "ui_test_path": ".tmp/ui_test.json"
}
```

13 项断言（给事件注入的真实路径，见 `NewWorldApp.run_ui_test`）：

1. `layout_regions_and_click_isolation` — 四区域判定正确；UI 区域坐标喂给世界点击处理必须被忽略；
2. `viewport_transform_and_zoom_anchor` — 世界↔屏幕往返误差 < 1e-6；缩放时光标锚点不动；
3. `world_click_selects_agent` — 用 `event_generate("<Button-1>")` 点精灵中心 → 选中；
4. `world_click_switches_selection` — 点另一个体 → 选择切换；
5. `world_click_toggles_off` — 再点同一个体 → 取消选中；
6. `world_click_empty_clears_selection` — 点远离个体的空白处 → 取消选中；
7. `wheel_zoom_and_drag_pan_hit_consistency` — 注入 `<MouseWheel>` 放大、右键 `<ButtonPress-3>`+`<Motion>`+`<ButtonRelease-3>` 拖动平移，之后按新屏幕位置点击仍命中同一世界个体；
8. `pause_freezes_time_but_panel_visible` — 点「暂停」后持续 `update()` 0.24 s，`world.t` 不变；暂停时详情/统计仍有数据；
9. `single_step_advances_exactly_one_frame` — 暂停下单步两次各恰好 `fixed_dt`；运行中单步被拒绝；
10. `list_click_syncs_selection` — 点击个体列表第 2 项 → 场景选择同步；
11. `dead_agent_clears_selection` — 走主循环同一条 `_apply_alive_ids` 路径：存活集合为空时选中者被自动清理、列表同步，恢复后列表正确；
12. `detail_and_stats_use_real_data` — 详情读真实能量、健康显示「暂无数据」、统计与 `registry` 一致、事件日志非空；
13. `reset_clears_selection_and_list` — 程序化重置（跳过对话框）后新世界 `t=0`、选择清空、列表项数=存活数、恢复运行。

`tests/test_ui_app.py` 在无显示环境会自动 `skipTest`；有显示时它会重复执行上面的
自检并断言 exit code 与 JSON。

## 7. 测试

```powershell
cd newworldpro
$env:TEMP=(Resolve-Path .tmp).Path
python -m unittest tests.test_ui_state tests.test_ui_app -v   # 34 + 7 用例
python -m unittest tests.test_world -v                        # 既有回归
```

`tests/test_ui_state.py` 34 个纯逻辑用例（无 tkinter）：视口往返/缩放锚点/平移
clamp/resize、命中（bbox/最近/重叠/无命中/死人不选/缩放一致性）、选择切换/取消/
死亡清理、控制状态机、布局区域、详情真实字段与「暂无数据」、统计/事件日志非空、
异常输入不抛错、`ui_state` 不引用 tkinter。

## 8. 已知限制（如实说明）

1. **帧率约 18–21 fps**（1280×800、2 个个体）。瓶颈在仿真本身（每个体每帧都做
   感知/神经网络/目标规划，项目文档称感知约占 95% 成本），不是渲染（实测
   `render_world` ≈ 4 ms/帧，角色图层 ≈ 0.16 ms/个）。观察窗口以 16 ms 定时器
   驱动，实际帧率受 `population.step(dt)` 限制。
2. **重置 = 新世界**：旧世界的人口/知识/设施不会保留（`messagebox` 会说明）。
   这与 `World.reset()`（只复位时间与位置、保留档案）不同，是刻意的 UI 语义。
3. **F2「训练」在后台线程**调用 `population.train_round(reset=True)`，会写
   `assets/agents/*.npz`；训练期间仿真仍在推进（沿用旧版行为），极端情况下存在
   数据竞争风险。日常观察不建议在仿真运行中训练。
4. **截图依赖桌面捕获权限**：`ImageGrab.grab` 失败时只打印「截图失败：…」，
   不会让程序崩溃；无桌面的服务器环境请用 `--headless`。
5. **无显示环境无法验证 GUI**：`tests/test_ui_app.py` 会 skip；`--ui-test`
   在没有可见桌面时返回 `2`（无法创建窗口）。本机（Windows 10/11 + tkinter 8.6）
   已真实跑通，见 §6 输出。
6. 面板的「健康」列来自世界模型没有的字段，恒为「暂无数据」——这是契约要求的
   「缺失显示暂无数据」，不是 bug；如需健康值应先在世界/生理模型里增加该字段。
7. 详情面板中的「最近目标结果」只在 `Goal` 携带 `reason`/`rationale`/`params`
   时显示原因；当前 `GoalManager` 未在 `Goal` 上写失败原因时显示计划层的
   `failure_reason`，再缺就显示「暂无数据」。

## 9. 代码结构（二次开发）

- `newworld/ui_state.py`（纯逻辑，**禁止 import tkinter**）：
  `Viewport / AgentBox / AgentHit / HitTester / SelectionModel / ControlState /
  Layout / DetailRow / AgentDetail / build_agent_detail / build_stats /
  build_event_log`。新增交互逻辑优先放这里，并补 `tests/test_ui_state.py`。
- `newworld/app_tk.py`：`NewWorldApp` 组装 tkinter 控件、画布渲染与事件绑定；
  `run_app(config, *, fullscreen, debug, screenshot, autoclose_ms, seed, policy,
  policy_path, ui_test_path)` 是唯一入口（旧关键字参数全部保留）。
  新增鼠标交互时：在 `_bind_events` 绑定 → 方法内只更新 `ui_state` 状态与画布 →
  在 `run_ui_test` 增加一条断言，保证可程序化验证。
- 选中高亮直接用 Canvas 的 `create_oval`（`_ring_item`），**不需要**修改
  `render.py`。
```

---

## 9. 本轮升级补充（v0.9.1）

- **任务标记**：正在执行计划（有活跃 `Plan`）的个体，头顶会显示一个橙色小圆点；
  圆点随缩放缩放、随个体移动。没有活跃计划的个体不显示。
- **决策来源**：选中个体后，详情面板的「当前目标 / 当前任务及进度」来自真实
  `GoalManager`（`goals_of` / `active_plan`）；面板不会显示"暂无数据"以外的占位数字。
- **学习型高层策略（可选）**：默认界面使用规则目标（heuristic）。若想观察训练后的
  高层策略，请用无窗口实验入口（GUI 不加载 goal policy，避免切换策略影响观察）：

  ```powershell
  python -m newworld.simulation --group C --steps 16000 --dt 0.05 `
      --width 640 --height 400 --policy-mode neural --policy-path assets/goal_policy.npz
  ```

- **组合式创新（可选）**：加 `--innovation` 观察有限组合式创新（自然触发、上限 3 项）：

  ```powershell
  python -m newworld.simulation --group C --steps 14000 --dt 0.05 `
      --width 640 --height 400 --innovation --json .tmp\innov.json
  ```

- **训练按钮 / F2**：手动触发一次「所有个体从零训练行为克隆网络」，
  会写 `assets/agents/*.npz`；不建议在长观察期间反复训练。

- **截图语义**：`--screenshot` 抓整个窗口（包含面板与状态栏），便于验收布局。
