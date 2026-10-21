# newworld 用户指南

本指南覆盖三条入口：**桌面 GUI**、**浏览器实验平台（lab）**、
**配置化实验**。架构细节见 [ARCHITECTURE.md](ARCHITECTURE.md)，
实验协议与结果见 [EXPERIMENTS.md](EXPERIMENTS.md)。

---

## 1. 环境与安装

要求 Python 3.10+（开发环境为 3.14）。

```powershell
cd C:\Users\27826\Desktop\vscode_win_ws\ds_ws\newworld\newworldpro
python -m pip install -r requirements.txt
```

检查依赖与组件注册表：

```powershell
python -m newworld --info
python -m newworld --version
```

`--info` 会打印版本、numpy / pillow / pyyaml / fastapi 版本，以及
当前已注册的策略（`rule` / `random` / `neural`）和实验协议
（`PROTOCOLS`，由 `newworld/protocols/` 导入时注册）。
`python -m newworld --list-components` 只列注册表。

---

## 2. 桌面 GUI

### 2.1 启动

```powershell
python -m newworld
python -m newworld --seed 7 --size 256x160
python -m newworld --seed 1 --size 96x64 --cell-px 2
```

| 参数 | 含义 | 默认 |
|---|---|---|
| `--seed` | 世界随机种子（同 seed 同地图/同轨迹） | 0 |
| `--size WxH` | 地图尺寸（cell 数，最小 32×32） | 512x320 |
| `--cell-px` | 每个 cell 的屏幕像素 | 3 |

窗口固定 1280×800 启动；地图大于视口时可拖动/缩放。

### 2.2 界面与操作

* **顶部**：社会机制实时统计（交换次数与转移能量、教学、继承、
  失传、技能人口）。
* **中部**：地图底图（七种地形色块 + 食物明暗）+ 群伙散点
  （颜色由饱足度决定：暗红=饥饿 → 亮青=吃饱）。
* **右侧面板**：当前人口与承载力估计、累计出生/死亡、死因
  （饿/老/溺）、地形分布、食物总量、社会机制明细。
* **底部**：开始/暂停、单步、速度（×1/×2/×4/×8/×16）、重置；
  状态栏显示人口、出生、死亡、食物、速度、FPS。

鼠标：左键拖动平移；滚轮以光标为锚点缩放。
键盘：空格 开始/暂停，`+` / `-` 调速，Esc 退出。

界面**只观察与控制**：没有改写地图或直接修改个体的工具。
"开始/暂停/单步/重置"只影响仿真推进，不改变世界规则。

### 2.3 无显示/自动化验证

GUI 提供两个钩子（不弹窗交互时也可验证渲染链路）：

```powershell
python -m newworld --screenshot .tmp\gui.png --autoclose-ms 1200
```

它会先推进约 200 步让种群动起来，再对画布截图并自动关闭；
`--screenshot` 需要桌面会话（PIL `ImageGrab`）。无 tkinter 时
GUI 会打印"无法启动窗口：缺少 tkinter"并以退出码 2 结束。

---

## 3. 浏览器实验平台（lab）

### 3.1 启动

```powershell
python -m newworld --lab                      # http://127.0.0.1:8600
python -m newworld --lab --port 8601 --seed 3
python -m newworld.lab_server --port 8600 --seed 0        # 只启动后端
python -m newworld.lab_server --config configs/base.yaml  # 用配置启动
```

Windows 下也可双击 `start_world.bat`：打开浏览器并启动 8600 端口服务。
lab 与无窗口实验共用 `newworld.experiment.build_simulation` 同一条
代码路径；**默认世界是 P1.7 的稀缺环境**（`food_regrow_per_s=0.0005`，
256×160，rule 策略，P3 社会机制开启）。

### 3.2 界面

* 主画布：PixiJS 渲染地形纹理 + 群伙散点；拖动平移、滚轮缩放；
  点击个体显示详情（纯观察，不干扰世界）。
* 顶栏：社会机制实时统计（交换/教学/继承/技能人口）。
* 右侧：人口/食物/饱足曲线、世界状态（seed、人口、出生/死亡、
  食物总量、平均饱足）、个体详情（位置/饱足/年龄/技能/观测）、
  干预记录（实验审计）。
* 底部：开始/暂停、单步（一次 50 个物理步）、速度 −/+（×1/×2/×4/×8/×16）、
  重置（seed+1 重建世界）。

所有命令都走 `/api/command`，并写入 `interventions`（时间/类型/参数），
保证"人工干预不计入学习成果"。

### 3.3 HTTP / WebSocket API

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/snapshot` | 只读快照：t、running、speed_index、seed、map、terrain（hex）、population、births、deaths、satiation、food_total、interventions、series、config_hash、social |
| GET | `/api/agent/{index}` | 个体详情：当前位置/饱足/年龄/观测；越界返回 `ok=false` 而不是 500 |
| POST | `/api/command` | 实验控制，body `{"kind": "pause"}` 等 |
| WS | `/api/stream` | 约 15Hz 的帧流：`t, running, x, y, h, pop, food, sat`（启用 P6 时附 comm/memory 统计） |
| GET | `/` | lab 前端页面（`lab_static/index.html`） |

`/api/command` 支持的命令：

| kind | 参数 | 说明 |
|---|---|---|
| `pause` | — | 暂停推进 |
| `resume` | — | 继续推进 |
| `speed` | `index` (0–4) | 每 tick 物理步数 = ×1/×2/×4/×8/×16 |
| `step` | `steps`（默认 1） | 手动推进 N 个物理步（暂停态可用） |
| `reset` | `seed`（可选） | 重建世界；缺省 seed+1 |

PowerShell 调用示例（已实际运行验证）：

```powershell
# 快照
Invoke-RestMethod http://127.0.0.1:8600/api/snapshot

# 暂停 / 单步 / 调速 / 重置（PowerShell 一行一条）
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8600/api/command -ContentType 'application/json' -Body '{"kind":"pause"}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8600/api/command -ContentType 'application/json' -Body '{"kind":"step","steps":10}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8600/api/command -ContentType 'application/json' -Body '{"kind":"speed","index":3}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8600/api/command -ContentType 'application/json' -Body '{"kind":"reset","seed":42}'

# 个体详情（index = 帧流 x 数组下标）
Invoke-RestMethod http://127.0.0.1:8600/api/agent/0
```

WebSocket 帧示例（`/api/stream` 每帧）：

```json
{"t": 1.25, "running": true, "x": [12.3], "y": [40.1], "h": [0.82],
 "pop": 120, "food": 9582.0, "sat": 0.70}
```

注意：`GET /api/snapshot` 的 `series` 只有在仿真运行时（resume 之后）
才会按秒采样增长；暂停态的新会话一开始可能为空。

---

## 4. 配置化实验

### 4.1 一份 YAML = 一次实验

入口有两个（等价，后者面向脚本）：

```powershell
python -m newworld --config configs/p1_scarce.yaml
python experiments/run.py --config configs/p1_scarce.yaml
```

`configs/base.yaml` 是公共基线：世界 256×160、初始人口 120、
容量上限 3000、run 120 秒 / dt 0.05 / seeds [0..4]、社会机制开启、
通信与记忆关闭。各实验配置用 `_base: base.yaml` 继承它，只写差异：

| 配置 | 协议 | 内容 |
|---|---|---|
| `configs/p1_scarce.yaml` | `learning_compare` | 四臂学习对照（稀缺世界 + 密度梯度） |
| `configs/p3_social.yaml` | `social_ablation` | 交换/教学/继承五配置消融 |
| `configs/p4_evolution.yaml` | `evolution_compare` | 演化 vs 学习三臂对照 |
| `configs/p5_scaling.yaml` | `scaling_sweep` | 人口/算力两个单维扫描 + 幂律拟合 |
| `configs/p6_communication.yaml` | `communication` | 指称博弈 + 信号通道 off/on 消融 |
| `configs/p6_memory.yaml` | `memory_ablation` | 记忆/反思五臂消融（含同维空对照） |

用 `python -m newworld --list-components` 查看当前实际注册的协议。
完整实验通常要跑数分钟；调试时可用 `--set protocol_params.<...>` 缩短
训练/评估回合（产物只用于链路验证，不是正式结果）。

完整配置结构（严格 schema，未知字段直接报错）：

| 段 | 关键字段 |
|---|---|
| 顶层 | `name`、`protocol`（注册表名）、`description`、`_base` |
| `world` | `width/height/seed/ocean_margin/noise_scale/sea_level/mountain_level/snow_level/food_regrow_per_s/forest_spread_per_s` |
| `swarm` | `initial_population/capacity_hard/metabolism_per_s/bite_size/mature_age_s/breed_*/old_age_*/move_speed/wander/forage_hunger` |
| `policy` | `kind`（`rule`/`random`/`neural`）、`seed/hidden/checkpoint/lr/gamma/batch_size/epochs_per_episode/normalize_advantage/grad_clip` |
| `social` | `enabled/enable_barter/barter_*/enable_teaching/teach_*/enable_inheritance/inherit_radius/initial_skill_rate` |
| `comm` | `enabled/n_symbols/radius/message_cost/seed/train/lr/batch_size/epochs` |
| `memory` | `enabled/capacity/recency_halflife_s/weight_*/top_k/reflection_*/seed` |
| `run` | `seconds/dt/daylight/seeds/eval_seconds/output_dir/save_models` |
| `protocol_params` | 协议私有参数（各协议自己解析，如 `train_episodes/eval_seeds/densities`） |

### 4.2 覆盖字段（--set）

`--set` 可重复，值按 YAML 解析（数字、布尔、列表、行内映射都能写），
键必须存在于 schema 中（防拼写错误静默生效）。下面是一条**缩短版冒烟
运行**示例（完整实验去掉 `protocol_params.*` 与 `--out` 即可）：

```powershell
python -m newworld --config configs/p6_memory.yaml --set world.width=96 --set world.height=60 --set run.seeds=[0,1,2] --set protocol_params.train_episodes=1 --set protocol_params.train_seeds=[1] --set protocol_params.eval_seeds=[0] --set protocol_params.train_seconds=2 --set protocol_params.eval_seconds=2 --set run.save_models=false --out experiments/results/override_demo
```

含空格/花括号的值用单引号包住，例如
`--set 'protocol_params.densities=[{name: normal, population: 20}]'`。

拼错键会直接报错并列出可用字段，例如
`--set world.widht=128` →
`错误：world: 未知字段 ['widht']；可用: [...]`。

### 4.3 新建一份实验配置

1. 复制 `configs/base.yaml` 为 `configs/<你的实验>.yaml`；
2. 把 `protocol` 改成已注册协议名（`python -m newworld --list-components`
   查看），只写与基线不同的字段（也可用 `_base: base.yaml` 继承）；
3. 运行：

```powershell
python -m newworld --config configs/<你的实验>.yaml
python -m newworld --config configs/<你的实验>.yaml --out experiments/results/my_run
```

### 4.4 产物

默认输出目录 = `run.output_dir`（`experiments/results`）/`name`；
`--out DIR` 会改为 `DIR/name`。每次实验写：

* `config.yaml` —— 生效配置的完整快照（含 `--set` 覆盖后的结果）；
* `summary.json` —— 协议返回的指标、结论、caveats、`config_hash`；
* 协议可另写 `report.md`；训练类协议会把模型写进 `assets/zoo/`。

`python experiments/run.py --config ... --json` 会把完整 summary JSON
打印到 stdout（便于管道处理）。

### 4.5 模型 zoo

训练类协议（P1/P4/P5/P6）在 `save_models: true` 时把策略保存到
`assets/zoo/<name>/`（`policy.npz` + `card.json` + `metrics.json` +
`config.yaml`）。查看/校验方式与当前模型清单见
[../assets/MODELS.md](../assets/MODELS.md)。

---

## 5. 测试

```powershell
python -m unittest discover -s tests
```

指定模块：

```powershell
python -m unittest tests.test_map_world tests.test_swarm_policy
python -m unittest tests.test_experiment tests.test_registry tests.test_zoo
python -m unittest tests.test_lab_server tests.test_social_mech
```

测试覆盖地图/食物/确定性、群伙人口动态、策略与训练器、社会机制、
配置层、注册表、实验运行器与协议、lab API 契约、模型 zoo。
测试不依赖网络（lab 用 FastAPI TestClient）。

在受限/沙箱环境里，临时目录可能不可写（Windows ACL），可把
`TEMP` 指到工作区内再跑：

```powershell
if (-not (Test-Path .tmp)) { New-Item -ItemType Directory .tmp | Out-Null }
$env:TEMP = (Resolve-Path .tmp).Path
python -m unittest discover -s tests
```

---

## 6. 排错

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: fastapi / uvicorn` | 只影响 lab；`python -m pip install -r requirements.txt` |
| GUI 提示"缺少 tkinter" | 当前 Python 未带 Tk；改用 lab（`--lab`），或安装带 Tk 的 Python |
| 窗口打开后不推进 | 点"开始"或按空格；默认是暂停态 |
| 端口被占用 | `--port` 换一个端口（如 8601） |
| `错误：配置文件不存在` | 检查路径相对项目根目录；`--config` 会先试项目根目录 |
| `错误：... 未知字段 [...]` | 配置字段拼写错误；按错误里的"可用字段"修正（严格模式） |
| `错误：... 需要整数/数值` | `--set` 的值类型不对；按 YAML 语法写（`true`/`[0,1]`/数字） |
| 实验产物找不到 | 看 `summary.json` 的 `_artifact_dir`，或 `--out` 指定的目录 |
| lab 快照 `series` 为空 | 先 `resume`（开始）推进一段时间，再取快照 |

更多环境/机制细节见 [ARCHITECTURE.md](ARCHITECTURE.md)。
