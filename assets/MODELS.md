# assets/ 模型清单与 zoo 规范

> 本文件只描述**当前平台**的训练产物。旧世界/旧命名下的 checkpoint 已在
> 本轮清理中删除，这里不再记录；`assets/` 下除图标资源外只有
> `assets/zoo/`（实验训练产出，按需生成）。

模型不是一个裸 `.npz`，而是一个**自描述目录**：架构、维度、seed、训练
配置指纹、权重 sha256、评估指标都在目录里，回答"这个模型是什么、怎么来
的、指标多少"。

---

## 1. 目录结构

```text
assets/zoo/<name>/
├── policy.npz     # 权重：W1/b1/W2/b2 + obs_dim/n_actions/hidden/seed/version
├── card.json      # 元数据（ModelCard，见 §2）
├── metrics.json   # 该 checkpoint 的评估指标（由协议写入，等于 card.metrics）
└── config.yaml    # 训练/评估所用配置快照（含 _base 展开与 --set 覆盖后的完整配置）
```

`<name>` 只允许 `[A-Za-z0-9_.-]` 且以字母数字开头（`newworld/zoo.py` 的
`_safe_name`；防路径穿越）。

## 2. card.json 字段

| 字段 | 含义 |
|---|---|
| `name` | 模型目录名（唯一） |
| `kind` | 策略种类；当前都是 `neural`（zoo 只接受可 `load` 的神经策略） |
| `obs_dim` / `n_actions` / `hidden` | 网络输入维度 / 动作数 / 隐层宽度 |
| `seed` | 策略初始化/采样 seed |
| `format_version` | checkpoint 格式版本（当前 `1`） |
| `created_at` | UTC 创建时间（ISO 8601） |
| `git_commit` | 保存时的 git commit（无 git 时为空串） |
| `config_hash` | 训练配置的 sha256 指纹（对应 `config.yaml`） |
| `weights_sha256` | `policy.npz` 的 sha256（`verify()` 校验用） |
| `description` / `tags` | 人读说明与标签（协议阶段、实验名） |
| `metrics` | 协议写入的评估指标（与 `metrics.json` 相同） |
| `extra` | 协议额外元数据（如 P5 的算力/训练 seed/评估 seed） |

## 3. 如何检查

以下命令均在项目根目录实际运行验证（PowerShell；Python 单行命令内部
用**单引号**字符串，避免 PowerShell 吞引号）：

```powershell
# 列出全部模型目录
python -c "from newworld.zoo import ModelZoo; print(ModelZoo().names())"

# 一行摘要：名字 / 种类 / 观测维 / 动作数 / 标签
python -c "from newworld.zoo import ModelZoo; print([(c.name, c.kind, c.obs_dim, c.n_actions, c.tags) for c in ModelZoo().cards()])"

# 校验权重 sha256 与 card 一致（防手改/损坏）
python -c "from newworld.zoo import ModelZoo; z=ModelZoo(); print([(n, z.verify(n)) for n in z.names()])"

# 看单个模型的完整元数据
python -c "from newworld.zoo import ModelZoo; print(ModelZoo().load_card('p1_scarce_neural'))"

# 载入策略（校验维度和 card 一致）
python -c "from newworld.zoo import ModelZoo; p=ModelZoo().load('p1_scarce_neural'); print(type(p).__name__, p.obs_dim, p.n_actions)"

# 汇总 manifest（给前端/文档读取；不写盘时只返回 dict）
python -c "from newworld.zoo import ModelZoo; m=ModelZoo().manifest(); print(m['format_version'], len(m['models']))"

# 直接读 npz（调试用；不做 card 校验）
python -c "from newworld.swarm_policy import NeuralSwarmPolicy; p=NeuralSwarmPolicy.load('assets/zoo/p1_scarce_neural/policy.npz'); print(p.obs_dim, p.n_actions, p.hidden)"
```

把 checkpoint 接回仿真（`make_swarm_policy` 的 `checkpoint` 路径优先于
`kind`，且会拒绝 `obs_dim` 不一致的模型）：

```python
import dataclasses
from newworld.settings import load_config
from newworld.experiment import build_simulation

s = load_config("configs/p3_social.yaml")
s = dataclasses.replace(
    s,
    policy=dataclasses.replace(
        s.policy, kind="neural", checkpoint="assets/zoo/p1_scarce_neural/policy.npz"
    ),
)
sim = build_simulation(s, seed=0)
print(sim.policy.obs_dim, sim.swarm.obs_dim)   # 必须一致，否则构造时抛错
```

## 4. 当前模型清单

> 下表由 `assets/zoo/` 在本次文档写入时枚举得到（`ModelZoo().names()`）。
> 训练/评估指标随实验重跑会更新，**以各目录的 `card.json` / `metrics.json`
> 为准**，下表只固定"怎么产出的"。

| 模型 | 结构（obs→hidden→动作） | 产出配置 / 协议 | 训练 seed / 关键参数 | 说明 |
|---|---|---|---|---|
| `p1_scarce_neural` | 9→32→5 | `configs/p1_scarce.yaml` / `learning_compare` | seed 0；训练 10 回合 × 60 世界秒，train_seeds 500–503 | P1.7 稀缺世界 REINFORCE 策略（tag: `p1`, `scarce`） |
| `p4_evolution_learning` | 9→32→5 | `configs/p4_evolution.yaml` / `evolution_compare` | seed 0；训练 8 回合 × 45 世界秒，train_seeds 500–501 | P4 "learning 臂"策略（tag: `p4`, `scarce`），评估 seed 1100–1104 |
| `p5_scaling_c16` | 9→32→5 | `configs/p5_scaling.yaml` / `scaling_sweep` | seed 0；episodes=16，算力 C=57600 agent·秒 | P5 最大算力档策略（tag: `p5`, `scaling_sweep`），评估 seed 0–2 |
| `p6-comm-on` | 13→32→5 | `configs/p6_communication.yaml` / `communication` | seed 7；训练回合 6 | P6-A 世界内通道 on 臂的移动策略（obs = 9 + 4 维符号直方图；tag: `p6-comm`, `on-arm`） |
| `p6-comm-sender` | 9→32→4 | `configs/p6_communication.yaml` / `communication` | seed 12345；n_symbols=4 | P6-A 符号发送者策略（观测 9 维、动作 4 个符号；tag: `p6-comm`, `sender`） |
| `p6-memory` | 15→32→5 | `configs/p6_memory.yaml` / `memory_ablation` | seed 7；训练 3 回合 × 40 世界秒，eval_seeds 0–2 | P6-B 记忆/反思消融的最佳记忆臂策略（obs = 9 + 6 维记忆特征；tag: `p6-memory`，当前 card 记录 `arm=full_reflect`） |

补充：

* `obs_dim` 不总是 9：通信开启时 +K（K=`comm.n_symbols`，默认 4），
  记忆开启时 +6；因此 `p6-comm-on`（13）与 `p6-memory`（15）不能直接挂到
  无机制的世界，反之亦然（`attach_policy` 会拒绝维度不一致）。
* `p6-comm-sender` 不是移动策略（4 个动作 = 4 个符号），用于通信协议内部
  发送符号；把它当移动策略会因动作空间语义不同而行为无意义。
* 每个模型目录里的 `config.yaml` 是**产生它的完整配置快照**，复现实验时
  直接用它对应的 `configs/<name>.yaml` 重跑即可（训练本身有随机性，
  权重不会逐位相同；seed 与训练预算相同则分布可比）。

## 5. 产出与复现

| 协议 | 写入的 zoo 名 | 触发条件 |
|---|---|---|
| `learning_compare`（P1） | `<settings.name>_neural`（如 `p1_scarce_neural`） | `run.save_models: true` |
| `evolution_compare`（P4） | `p4_evolution_learning` | `run.save_models: true` |
| `scaling_sweep`（P5） | 配置里的 `protocol_params.model_name`（如 `p5_scaling_c16`） | `protocol_params.save_model: true` |
| `communication`（P6-A） | `p6-comm-on`、`p6-comm-sender` | on 臂训练后 |
| `memory_ablation`（P6-B） | `p6-memory` | 消融结束后（最佳记忆臂） |

重新训练并写入 zoo 的标准做法：

```powershell
python -m newworld --config configs/p1_scarce.yaml
python -m newworld --config configs/p4_evolution.yaml
python -m newworld --config configs/p5_scaling.yaml
python -m newworld --config configs/p6_communication.yaml
python -m newworld --config configs/p6_memory.yaml
python -c "from newworld.zoo import ModelZoo; z=ModelZoo(); print(z.names()); print([(n, z.verify(n)) for n in z.names()])"
```

## 6. 规则（写入与校验）

1. **只通过协议写 zoo**：`ModelZoo.save()` 统一写 `policy.npz/card.json/
   metrics.json/config.yaml`，不要手工往 `assets/zoo/` 里放文件。
2. **不要手改 card 或权重**：`verify()` 用 sha256 比对，手改会让校验失败；
   改权重就重新训练。
3. **名字即身份**：同名保存会覆盖原目录（协议按实验名/固定名保存）；
   重跑实验前注意这会更新已有模型。
4. **配置可追溯**：`config_hash` 必须与目录里 `config.yaml` 的配置一致；
   复现实验用该配置而不是记忆里的参数。
5. **不含旧世界产物**：当前 zoo 只有单一平台（俯视地图 + 群伙）的策略；
   任何"旧世界/旧命名"的 checkpoint 已删除，不应再被引用。
