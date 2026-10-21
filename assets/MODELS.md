# assets/ 模型与资源清单

> 每个 checkpoint 的架构、用途、生成方式。P4 整理（用户要求：
> "不要连训练的模型是什么都不知道"）。

## 训练产物（模型 checkpoint）

| 文件 | 架构 | 输入→输出 | 用途 | 生成方式 |
|---|---|---|---|---|
| `neural_policy.npz` | MLP 24→64→32→58（BC，v2 格式） | 24 维旧世界观测 → 5 mode+3 move+2 aux+48 语言 | 旧世界低层行为策略（行为克隆） | `tools/train_neural_policy.py --pretrain` |
| `lowlevel_rl.npz` | 同上但 56 维输入（v4 格式，REINFORCE） | 56 维（含 16 视网膜特征）→ 58 | V2 低层 RL 策略（minibatch REINFORCE） | `tools/train_lowlevel_rl.py` |
| `goal_policy.npz` | MLP 30→32→8 | 30 维高层特征 → 8 目标动作 | V1 高层目标策略 | `tools/train_goal_policy.py` |
| `society.npz` | 11→12→6 双层 | 社会特征 → 6 维上下文 | 旧世界社会网络（未接决策——P0 审查发现的死特征） | 旧版 Population 自动保存 |
| `agents/agent_*.npz` | MLP 24→64→32→58 | 同 neural_policy（按个体） | 旧世界每个体独立策略（遗传+变异） | `Population.train_round` / 繁衍继承 |

## 新世界（V3/P1+）模型

新世界的 `NeuralSwarmPolicy`（9→32→5）checkpoint 由
`experiments/run_p1_baseline.py` 按需训练，**不默认保存到 assets/**
（训练-评估在同一脚本内闭环；P5 引入 model zoo 时统一入册）。

## 图像/图标资源

| 文件 | 用途 |
|---|---|
| `newworld.ico` / `newworld_icon_256.png` | 桌面快捷方式图标（`tools/make_icon.py` 从蓝图生成） |
| `gui_pixel_world.png` | README 截图（旧世界） |
| `character.png` / `character_x12.png` | 蓝图角色提取（`tools/extract_character.py`） |
| `world_*.png` / `family/society/neural_speech_preview.png` | 文档预览图（`tools/make_preview.py`） |
