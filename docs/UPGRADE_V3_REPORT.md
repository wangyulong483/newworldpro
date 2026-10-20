# V3 升级报告：世界盒子式俯视地图世界

> 提交范围：`map_engine.py` / `swarm.py` / `catastrophes.py` / `map_render.py` /
> `app_map.py`（全新）+ `cli.py`（默认入口切换）+ `start_world.bat` +
> `tests/test_map_world.py`（19 个新测试）。
> 本文只记录**已经实现并实测验证**的内容；未做与边界在 §6 如实列出。
> 旧世界（侧视像素小人文明）**完整保留**，见 §5。

---

## 0. 一句话

V3 把默认世界从"一块屏幕里的 8 个像素小人"换成世界盒子（WorldBox）风格的
**俯视岛屿地图 + 千人群伙 + 上帝之手**：个体不再携带神经网络与感知栈，
而是 numpy 数组里的一行；智能完全从规则与统计中涌现。

## 1. 动机与决策

V2 复盘暴露的三个结构性矛盾：

1. **规模矛盾**：per-agent 决策链（感知占一帧 95% 开销、每人一个 MLP）
   在原理上撑不过百人；而"文明"现象需要千级人口才显现。
2. **视角矛盾**：侧视"屏幕即世界"与地图式玩法（看疆域、放灾害）不兼容。
3. **投入错位**：最精细的子系统（类人视觉）不在决策回路里（V2 已证实
   BC 权重视觉列为零；RL 信号修复后才具备训练前提）。

因此 V3 按"渐进改造"路线做**彻底替换**（用户决策）：
保留实验框架/测试模式/旧世界入口，人口引擎与渲染层全新。

## 2. 新世界架构（五模块数据流）

```
MapEngine (map_engine.py)         地形真相：terrain/food/burning 三个网格
   ↑ update(dt, daylight)             食物再生 · 森林蔓延 · 火蔓延（亚临界）
   │
Swarm (swarm.py)                   千人群伙：结构化数组（x/y/hunger/age/
   ↑ update(dt)                        breed_cd/tribe/alive），全向量化
   │                                   饥饿→进食→移动→繁衍→死亡→压实
   │
Catastrophes (catastrophes.py)     上帝之手：meteor/lightning/flood/heal
   │                                   作用真实网格与个体，事件史可审计
   ↓
MapRenderer (map_render.py)        渲染：底图 PIL（地形+食物明暗+火色）
                                      + 散点（饱足度渐变色）
   ↓
MapWorldApp (app_map.py)           GUI：整图适配画布 + 工具栏 + 面板
```

### 2.1 地形（MapEngine）

* 生成：双尺度值噪声（格点 splitmix64 哈希 + smootherstep 插值）→
  海拔/温度双阈值分层 → 7 种地形（深海/浅海/沙滩/草地/森林/山地/雪原）；
  四周大洋框。**同 seed 逐 cell 一致**（测试断言）。
* 食物：草地/森林/沙滩有各自容量上限，白昼按速率再生，夜间减半。
* 火：`ignite` 点燃森林/草地，3 秒燃烧后焦土化（森林→草地、食物清零）；
  蔓延率压在 **0.55/s 亚临界**——最初 1.6/s 导致分支过程超临界，
  小图烧穿不止（`test_fire_ignites_spreads_and_burns_out` 抓出，已修）。

### 2.2 群伙（Swarm）

* SoA 结构化数组（`SWARM_DTYPE`），容量 4000（可配 8000），死亡槽位压实复用。
* 每帧行为：代谢 → 死亡判定（饿/老/溺/焚，分别计数）→ 就地进食
  （从 food 网格扣 bite）→ 梯度+噪声移动（避水、饱足≥0.95 懒得动）→
  繁衍（成年+饱足+食物充分+概率 0.1/s+冷却）→ 压实。
* **承载力是涌现的**：`carrying_capacity_estimate() = 食物再生流量 / 个体
  消耗`，仅作软约束与 UI 展示；真实人口由动态平衡决定。

### 2.3 灾害与神迹（Catastrophes）

| 名称 | 半径效果 | 实测（seed=3 冒烟） |
|---|---|---|
| 陨石 meteor | 中心焦土+击杀，边缘 25% 点燃 | 杀 9 人、点燃 48 格 |
| 闪电 lightning | 单点点燃可燃地形 | 引发 280 格燃烧事件 |
| 洪水 flood | 低地概率淹没为浅水 | 淹 239 格 |
| 治愈 heal | 灭火+食物回满+饥饿回满 | 灭火即时生效 |

全部事件进 `history`（可 JSON 化），面板滚动展示最近 8 条。

## 3. 实测验收数据

全部来自真实运行（开发机，非理论值）：

| 指标 | 数值 |
|---|---|
| 人口动态（256×160 图） | 220 初始 → 60s 586 → 120s 1500 → **480s 稳定 ≈5050**（=承载力估计 5054） |
| 仿真性能（512×320 图） | **8000 个体 1.9 ms/帧**（515 FPS 余量） |
| 全渲染管线（含散点转换） | 8000 个体 24.5 ms/帧（41 FPS） |
| 底图全量渲染 | 256×160 图 10.2 ms（512×320 约 40ms，仅初始化/重置时） |
| 陨石施放延迟 | 9.8 ms |
| GUI 截图验证 | 画面 91.1% 填充：海洋 47.8% + 植被 38.2%（自动化像素统计断言） |
| 回归 | **544 tests OK**（525 旧全保留 + 19 新） |

## 4. 开发中抓住并修掉的真实缺陷

1. **繁衍冷却从未流逝**：cooldown 只在"吃到食物"分支递减，断粮个体
   永远不冷却 → 首测全族绝育灭绝。修：冷却对所有存活个体流逝。
2. **繁衍食物阈值过高**：0.35 的阈值高于个体觅食后残余（实测 0.12），
   即使饱足也不繁殖 → 220→0。修：阈值降至 0.12 + 繁衍概率 0.1/s 密度自限。
3. **火蔓延超临界**（见 §2.1）。
4. **Canvas 布局竞态**：首帧 `winfo_width()==1` 导致地图缩放异常、
   截图 13.7% 黑屏。修：整图 contain 适配 + 未布局时用请求尺寸。
5. **任务演算中数值溢出**：噪声 seed 乘法超 uint64 → Python int 溢出。
   修：种子先混合到 32 位。

## 5. 旧世界的处置

* `python -m newworld --classic` 进入完整的 V1/V2 侧视世界
  （目标规划/知识/记忆/神经策略/无限世界/8 人文明），**代码与测试零删减**；
* `newworld.simulation`（文明实验框架）与全部 `experiments/` 脚本不受影响；
* V2 的 RL 小批量修复（`lowlevel_rl.py`：minibatch/优势标准化/梯度裁剪，
  学习信号增强 100-210 倍）在同批交付，详见该模块 docstring。

## 6. 已知限制与未做（如实）

1. **渲染 41 FPS ≠ 60**：瓶颈在散点绘制（5000+ Canvas 矩形）而非仿真
   （1.9ms）。二期计划：散点改为位图叠加（PIL draw → PhotoImage）。
2. **视口缩放/平移未接**：`Viewport` 类已复用进来但一期选择整图可视；
   滚轮当前映射为速度调节。
3. **远距离地形不持久**：地图是有限整张（512×320），无 V2 无限世界
   的滚动窗口概念——那是旧世界线的特性。
4. **tribe 列已预留但未使用**：部落/国家/战争是二期内容
   （swarm 数据结构的 `tribe` 字段已就位）。
5. **GUI 截图验证是像素统计断言**，不是人眼审查；`--ui-test`
   程序化自检尚未对新世界实现（旧世界的 13 项自检仍在 `--classic` 下可用）。
6. **昼夜只是食物系数**：`daylight` 参数传常量 0.8；DayCycle 未接入
   （地图世界的昼夜渲染留给二期）。

## 7. 测试

`tests/test_map_world.py`（19 个）分三组：

* **MapEngineTest**（7）：确定性/合法地形/大洋边界/食物再生/火点熄周期/
  脏区登记/水域不可行走；
* **SwarmTest**（7）：落地合法/人口增长/断粮饿死/溺水/槽位压实/
  轨迹可复现/千人性能门槛（<8ms）；
* **CatastrophesTest**（5）：四种灾害真实效果 + 事件史。

```powershell
cd newworldpro; $env:TEMP=(Resolve-Path .tmp).Path
python -m unittest tests.test_map_world -v     # 仅新世界
python -m unittest discover -s tests           # 全套 544
```

## 8. 复现验收

```powershell
# GUI（截图 + 自动关闭）
python -m newworld --screenshot .tmp\v3.png --autoclose-ms 3500

# headless 人口动态（本报告 §3 的数据来源脚本思想）
python -X utf8 -c "from newworld.map_engine import *; from newworld.swarm import *; \
e=MapEngine(MapConfig(seed=0,width=256,height=160)); s=Swarm(e,SwarmConfig(),seed=0); \
[ (e.update(.05,.8), s.update(.05)) for _ in range(9600) ]; print(s.population, s.births, s.deaths)"
```

## 9. 二期路线（已与需求对齐，未开工）

部落形成（亲缘聚簇 → territory 染色）→ 边界冲突/战争/同化 →
瘟疫/地形笔刷 → 视口缩放平移 → 散点位图化（60FPS）→ 昼夜接入。
