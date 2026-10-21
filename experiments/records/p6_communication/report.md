# P6-A 涌现通信实验报告（newworld / newworldpro）

生成方式：本文件由 experiments/results/p6_communication/summary.json 的实测数字生成（不是手抄）。
对应共享任务：task-4（P6-A emergent communication）。

## 0. 运行信息

- 命令： python -m newworld --config configs/p6_communication.yaml
- 产物目录： experiments/results/p6_communication/（summary.json / config.yaml / report.md）
- 配置指纹 config_hash： 27fe3de8fa1c0278e93478b78ec841c6aaa976cbc241592cb859f7a69a087d01
- 总运行时长： 151.2 s（Part A 7.9 s + Part B 143.4 s），远低于 ~10 min 预算。
- 环境： Python 3.14.0 / numpy 2.4.2 / pyyaml 6.0.3；git commit fc502ff。
- 稀缺世界：256x160，food_regrow_per_s=0.006（base 的一半），initial_population=120，social 关闭（隔离通信通道）。

## 1. 结论（TL;DR，先给可复现的数字）

**(i) 控制指称博弈（Part A）确实涌现了指称协议。**
- accuracy = 0.9964 ± 0.0007（5 个 game_seeds x 400 episodes，每个 seed 评估 n=2000）；
- 随机机会水平 = 1/cue_dim = 0.25；未训练基线实测 accuracy = 0.2472 ± 0.0256；
- 符号-线索互信息 MI = 1.981 ± 0.003 bits（上限 log2(4)=2.00 bits）；
- 符号熵 = 1.998 ± 0.002 bits；
- 5/5 个 seed 的 cue->symbol 映射都是可辨识双射（injective fraction = 1.00）；允许符号重命名后跨 seed 一致率 = 1.00（原始映射完全相同的 pair 比例 = 0.10，符号标签本身任意，这个低值不构成反例）。
- 判定：满足事先设定的判据（mean accuracy > chance + 0.10 且 > 0.5）=> protocol_formed = true。

**(ii) 世界内信号通道（Part B）在本设定下是 null / negative，不是正结果。**
- 通道内容：评估 rollout 内 MI = 0.0037 ± 0.0014 bits（通道上限 2 bits，即满容量的 0.18%），有限样本偏置约 0.0008 bits；食物对齐率 = 0.229 ± 0.052（随机 = 0.25，并没有高于随机）；sender 符号熵只有 0.511 ± 0.067 bits（几乎坍缩成常量符号，见 3.3 的 symbol_counts）。
- 行为配对差（on 相对 off，5 个配对评估 seed）：eat_events Δ = -9944.8 ± 3350.8（-14.3%，wins/losses/ties = 0/5/0），satiation_mean Δ = -0.0488 ± 0.0255（-5.1%，0/5/0）。
- 判定：通道没有学到可辨指称语义（MI 与有限样本偏置同量级、对齐不超过随机、sender 坍缩），且 on 臂在全部 5 个评估 seed 上都更差。**我们报告这个负结果，不把它包装成"通信帮助了群体"。**

**明确不宣称的事情**：不宣称"通信在群体中普遍有害"；不宣称 sender 学到了任何语义（Part B）；不宣称指称博弈的 4-symbol 双射等价于组合性/系统性语言。Part B 每臂只有 1 个训练 replicate（5 个评估 seed 共享同一策略），所以负向差只能说明"本预算/本通道实现下这条通道既无学习信号、又拖累了觅食"，不能外推为一般结论。

## 2. Part A：控制指称博弈（方法）

- sender：观测 = cue_dim=4 的 one-hot cue（obs_dim=4），动作 = 从 4 个符号中采样 1 个；
- listener：观测 = 符号的 one-hot（obs_dim=4），动作 = 预测 cue（n_actions=4）；
- 两人共享 0/1 奖励（listener 预测正确 = 1）；两个 NeuralSwarmPolicy（hidden=32）用批内均值基线的 REINFORCE 联合训练；
- 每个 game_seed 独立初始化；训练 400 episodes，每 episode 64 个博弈样本，lr=0.5；训练后用 n=2000 的新样本评估；
- 度量：accuracy（vs 机会 1/cue_dim）、cue-symbol 互信息（bits）、符号熵、cue->symbol 映射及其跨 seed 稳定性（允许符号重命名，用最优置换对齐）。

### 2.1 每个 game_seed 的结果

| seed | untrained acc | accuracy | MI (bits) | entropy (bits) | mapping cue->symbol | n_eval |
|---|---|---|---|---|---|---|
| 0 | 0.200 | 0.995 | 1.979 | 1.999 | {"0":2,"1":1,"2":3,"3":0} | 2000 |
| 1 | 0.268 | 0.997 | 1.979 | 2.000 | {"0":1,"1":3,"2":2,"3":0} | 2000 |
| 2 | 0.264 | 0.997 | 1.988 | 1.998 | {"0":1,"1":0,"2":3,"3":2} | 2000 |
| 3 | 0.240 | 0.997 | 1.980 | 2.000 | {"0":2,"1":1,"2":3,"3":0} | 2000 |
| 4 | 0.264 | 0.997 | 1.980 | 1.995 | {"0":1,"1":2,"2":0,"3":3} | 2000 |

### 2.2 汇总

- accuracy： 0.9964 ± 0.0007（n=5 seeds；min 0.9950 / max 0.9970）
- 高于机会的幅度： +0.7464
- MI： 1.981 ± 0.003 / 2.00 bits
- 熵： 1.998 ± 0.002 bits
- 映射可辨识比例： 1.00；跨 seed 重命名对齐一致率： 1.00
- 判据（mean accuracy > chance + 0.10 and > 0.5）： true

**解读**：4 个 cue 被稳定地映射到 4 个不同符号（MI 接近上限 2 bits、符号熵接近 2 bits），且每个 seed 都是双射、跨 seed 在符号重命名下完全一致（一致率 1.00）——这正是"指称协议形成"的可测含义：符号不是随机的，而是可被 listener 解码回 cue。注意这只证明 4-cue 指称可辨识性，不证明组合性/系统性语言。

## 3. Part B：世界内信号通道 off/on 配对消融（方法）

- off 臂：comm.enabled=false，观测 9 维（基础观测）；
- on 臂：comm.enabled=true（n_symbols=4，radius=6.0），观测 9+4=13 维（听到的邻居符号归一化直方图）；训练时 comm.train=true，sender（NeuralSwarmPolicy，obs=9 基础观测）用 1 步群体奖励 Δeat_events/人口 做内部 REINFORCE；
- 两臂训练预算完全相同：train_seeds=[1000,1001,1002]，6 episodes x 30.0 s，运动策略 seed=7；on 臂 sender seed=12345；
- 评估：同一运动策略在 eval_seeds=[2000,2001,2002,2003,2004] 上配对评估 60.0 s；on 臂评估时 comm.train=false，sender 被冻结为训练后的权重（sender_frozen=true，eval train_updates 全为 0）；
- 训练量：off 578826 / on 578733 transitions；训练 wall 38.0 s / 62.2 s；on 臂 sender 权重最大变化 = 0.2633（说明 sender 确实在被训练，不是随机网络）。

### 3.1 配对结果（on - off，同 seed；n=5 个评估 seed）

| metric | off mean ± std | on mean ± std | paired delta mean ± std | relative | wins / losses / ties |
|---|---|---|---|---|---|
| eat_events | 69349.800 ± 3941.400 | 59405.000 ± 5658.473 | -9944.8000 ± 3350.8210 | -14.3% | 0 / 5 / 0 |
| eat_rate | 9.632 ± 0.547 | 8.251 ± 0.786 | -1.3812 ± 0.4654 | -14.3% | 0 / 5 / 0 |
| satiation_mean | 0.950 ± 0.010 | 0.901 ± 0.026 | -0.0488 ± 0.0255 | -5.1% | 0 / 5 / 0 |
| population | 643.000 ± 12.775 | 624.000 ± 11.189 | -19.0000 ± 13.1909 | -3.0% | 0 / 5 / 0 |
| mean_life | 30.331 ± 7.361 | 27.199 ± 0.065 | -3.1313 ± 7.3340 | -10.3% | 4 / 1 / 0 |

### 3.2 每个评估 seed 的明细

| eval seed | off eat_events | on eat_events | delta | off satiation | on satiation | on channel MI | on entropy | on food alignment |
|---|---|---|---|---|---|---|---|---|
| 2000 | 72969 | 64493 | -8476 | 0.9647 | 0.9245 | 0.0044 | 0.513 | 0.291 |
| 2001 | 68290 | 53074 | -15216 | 0.9503 | 0.8611 | 0.0061 | 0.611 | 0.153 |
| 2002 | 63975 | 52256 | -11719 | 0.9409 | 0.8783 | 0.0022 | 0.549 | 0.192 |
| 2003 | 74669 | 65576 | -9093 | 0.9553 | 0.9166 | 0.0026 | 0.419 | 0.232 |
| 2004 | 66846 | 61626 | -5220 | 0.9374 | 0.9241 | 0.0031 | 0.461 | 0.278 |

### 3.3 on 臂评估 rollout 的通道统计（sender 冻结）

- messages： 340070 ± 5927 条/rollout（滚动统计缓冲 8192 条）
- MI（symbol vs 局部食物梯度桶）： 0.0037 ± 0.0014 bits；有限样本偏置 0.0008 bits => 观测值约为偏置的 4.6 倍，但只有 2-bit 通道容量的 0.18%。
- symbol_entropy： 0.511 ± 0.067 bits（上限 2 bits）=> sender 几乎坍缩到常量符号。
- food_alignment： 0.229 ± 0.052，随机水平 1/K = 0.25 => 未高于随机。
- 每个评估 seed 的 symbol_counts（4 个符号的滚动计数）： [[223,246,7559,164],[280,318,7396,198],[222,256,7503,211],[187,171,7704,130],[194,182,7642,174]]
  （例如第一个 seed 的计数约为 [223, 246, 7559, 164]，绝大部分消息集中在同一个符号上。）

### 3.4 诚实解读（这部分是实验结论，不是免责声明）

1. **通道没有承载可辨语义。** MI 与有限样本偏置同量级（约 5 倍偏置、但仅 0.18% 通道容量），食物对齐不高于随机，且 sender 熵坍缩到 0.51 bits——发送者基本在重复一个常量符号。这不是"协议形成了但度量没测到"，而是"符号分布本身退化"：常量符号不可能传递 cue 信息。
2. **on 臂更差，但原因不能归因于"策略学会了用信号"。** 没有证据显示 on 臂在战略性使用消息（若是，应该看到 MI/对齐上升或至少行为差异与通道内容相关）；更可信的机制是：在个体奖励 + 参数共享运动策略的设定下，通信通道给观测空间增加了 4 维与任务无关、且在训练中非平稳（sender 同时在变）的输入，增加了短训练预算下的优化噪声。这与"通信对群体有害"是两回事。
3. **负结果不能外推。** 每臂只有 1 个训练 replicate（5 个评估 seed 共享同一运动策略，配对只在"同一策略 x 不同世界"意义上成立），训练预算只有 6 x 30 s；因此 -14.3% 的 eat_events 下降只能说明"本通道/本预算下没有收益、且可能拖累"，不能说明通信机制普遍无益或有害。
4. **为什么世界通道难学（机制层面的解释）。** sender 的奖励是"1 步群体奖励"（Δeat_events/人口），所有 sender 共享同一个标量奖励、彼此之间没有个体信用分配；且运动策略（listener 的角色）是按个体进食奖励训练的，并不因"听懂符号"而获得额外收益。因此 sender 的梯度里几乎不包含"哪个符号对该个体所处局部环境有用"的信息，语义涌现缺少必要的学习信号——这正是本实验想诚实展示的边界条件。

## 4. 局限与后续

- Part A：只覆盖 4 cue / 4 symbol 的指称可辨识性；不涉及组合性、系统性、泛化到未见 cue 组合。
- Part B：每臂 1 个训练 replicate、5 个评估 seed、6 x 30 s 训练；没有做显著性检验（n 太小），负向差只能定性描述。
- Part B 的 off/on 观测维度不同（9 vs 13），两臂策略初始化无法逐位相同；配对只在评估 seed 上成立（同世界 seed、同评估时长）。
- 后续可做：给 listener 侧加"听懂即获益"的机制（例如按消息决定行动并给共享奖励）、或让 sender 获得个体级奖励/成本，再看 MI 是否超过有限样本偏置；也可以把 sender 训练改成集中式 critic 以提供 per-sender 信用分配。

## 5. 产物、测试与复现

- 产物：
  - experiments/results/p6_communication/summary.json（全部原始数字，含 per-seed / per-run 明细）
  - experiments/results/p6_communication/config.yaml（config hash 27fe3de8fa1c0278e93478b78ec841c6aaa976cbc241592cb859f7a69a087d01）
  - experiments/results/p6_communication/report.md（本文件）
- 模型 zoo（assets/zoo，均 verify=true，kind=neural，可直接 ModelZoo().load）：
  - p6-comm-on：on 臂运动策略（obs_dim=13, n_actions=5, hidden=32），tags=[p6-comm, emergent-communication, on-arm]
  - p6-comm-sender：训练后的 sender（obs_dim=9, n_actions=4, hidden=32），tags=[p6-comm, sender]
- 测试：
  - python -m unittest tests.test_communication -q => 15 tests OK（2.78 s）
  - python -m unittest discover -s tests -q => 165 tests OK（22.05 s）
- 复现命令（PowerShell，Windows）：
      cd newworldpro
      $env:PYTHONIOENCODING='utf-8'
      python -X utf8 -m newworld --config configs/p6_communication.yaml

## 6. 配置与种子（精确记录）

- 配置： configs/p6_communication.yaml（继承 configs/base.yaml；protocol=communication；comm enabled=true, n_symbols=4, radius=6.0, message_cost=0.0, train lr=0.05, batch_size=128, epochs=1）
- Part A seeds： game_seeds=[0,1,2,3,4]（每个 400 episodes x 64 batch, lr=0.5）
- Part B seeds： train_seeds=[1000,1001,1002] x 6 episodes；train_policy_seed=7；sender_seed=12345；eval_seeds=[2000,2001,2002,2003,2004]（每个 60 s，配对）
- 训练/评估固定 seeds：world 用 eval seed（evaluate_seeds 每个 seed 独立造世界），策略与 sender 用上面的固定 seed。
