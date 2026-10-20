"""群伙策略层（P1）：把 V3 的硬编码行为抽成可注入、可训练的策略。

定位
====
P0 审查（`docs/RESEARCH_ARCHITECTURE_REVIEW.md` §2.1）确认：V3 的
个体行为全部内联在 ``Swarm.update`` 里（梯度+wander 规则+随机抽样），
无策略接口、无转移、无奖励——不满足"可训练人工生命"的定义。

本模块做三件事：

1. **SwarmPolicy 协议**：策略只需要 ``act(obs_batch) -> actions``——
   批量（向量化）接口，与 :class:`~newworld.swarm.Swarm` 的 SoA 数组
   对齐，也和 PettingZoo AEC 的"个体索引+统一动作空间"精神一致
   （调研见 `docs/RESEARCH_REFERENCES.md` §3）；
2. **RuleSwarmPolicy**：把 V3 原有的"4 方向食物探测 × 随机权重 +
   wander 抖动"原样抽出，作为**规则基线**（不是删除，是搬家）；
3. **NeuralSwarmPolicy**：numpy MLP（9→32→5，softmax 采样），
   参数共享 IQL 范式——全部个体共享一套参数、各自独立产生经验，
   这是千级智能体 + 生灭生命周期在纯 numpy 约束下唯一现实的
   MARL 落点（CTDE/QMIX 不可行的论证见 REFERENCES §3）。

动作空间（P1 冻结）
-------------------
5 个离散动作：0=右移、1=左移、2=下移、3=上移、4=原地。
食/繁衍/死亡保持规则层（P1 只学"去哪找食物"，见 ROADMAP §P1）。

观测空间（P1 冻结，9 维，部分可观测）
--------------------------------------
0  hunger          自身饱足 [0,1]
1  age/100         年龄归一（截断）
2  food_here       所在格食物 [0,1]
3-6 food_e/w/s/n   东/西/南/北 3 格处食物（不可走=-1）
7  水警示           所在格是否邻水（0/1）
8  bias            恒 1（网络偏置列，避免全零观测）

奖励（个体级、秒级信用——吸取 V2 教训，REVIEW §5）
--------------------------------------------------
进食 +1.0；死亡 -10.0；每步 -0.01·dt 时间成本。
由 ``Swarm`` 在世界规则执行时结算（本模块只定义常数）。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

#: 动作数（右/左/下/上/原地）
N_ACTIONS = 5
#: 动作 → (dx, dy)
ACTION_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 0), (-1, 0), (0, 1), (0, -1), (0, 0),
)

#: 观测维度
OBS_DIM = 9

#: 奖励常数（Swarm 结算时使用；集中定义防散落）
REWARD_EAT = 1.0
REWARD_DEATH = -10.0
REWARD_TIME_PER_S = -0.01


# ---------------------------------------------------------------------------
# 协议
# ---------------------------------------------------------------------------
class SwarmPolicy:
    """群伙策略协议：批量观测 → 批量动作。

    实现者可以是确定性的（规则）或随机的（网络采样）；随机实现
    自己持有 RNG 并保证同 seed 复现。``act`` 必须接受 (N, OBS_DIM)
    返回 (N,) int 数组，值域 [0, N_ACTIONS)。
    """

    def act(self, obs_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # 可选：训练接口（无之则该策略不可训练，仅作基线）
    def parameters(self) -> dict[str, np.ndarray]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 规则基线：V3 原行为，原样搬家
# ---------------------------------------------------------------------------
class RuleSwarmPolicy(SwarmPolicy):
    """V3 原有觅食规则：4 方向食物探测 × 随机权重 + wander 抖动。

    行为与改造前的 ``Swarm.update`` 逐条对应（梯度探测：观察 4 方向
    3 格食物 × 随机系数，取最高分；不可走 -1；吃饱(hunger≥0.95)
    倾向原地）。作为 P1 对照实验的规则基线。
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs_batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(obs_batch, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != OBS_DIM:
            raise ValueError(f"obs_batch 形状 {batch.shape} != (N, {OBS_DIM})")
        hunger = batch[:, 0]
        food_here = batch[:, 2]
        east = batch[:, 3]
        west = batch[:, 4]
        south = batch[:, 5]
        north = batch[:, 6]
        n = batch.shape[0]
        rand = self.rng.random((n, 4))
        # 四个探测方向的食物得分 × 随机权重（与原实现一致）
        scores = np.stack(
            [east * (0.5 + rand[:, 0]),
             west * (0.5 + rand[:, 1]),
             south * (0.5 + rand[:, 2]),
             north * (0.5 + rand[:, 3])],
            axis=1,
        )  # (n, 4)，索引 0..3 恰好对应动作 0..3
        actions = np.argmax(scores, axis=1).astype(np.int64)
        # 吃饱或脚下有食物：原地（动作 4）
        stay = (hunger >= 0.95) | (food_here >= 0.35)
        actions = np.where(stay, 4, actions)
        # wander：完全随机替换一部分决策（与原实现的抖精神一致）
        wander_mask = self.rng.random(n) < 0.25
        actions = np.where(
            wander_mask,
            self.rng.integers(0, N_ACTIONS, size=n),
            actions,
        )
        return actions


# ---------------------------------------------------------------------------
# 神经策略：参数共享 IQL
# ---------------------------------------------------------------------------
class NeuralSwarmPolicy(SwarmPolicy):
    """numpy MLP（9→32→5，softmax 采样），全体个体共享参数。

    * 前向：``logits = tanh(obs@W1+b1) @ W2 + b2``；
    * 采样：softmax + Gumbel-max（向量化，无需逐个体循环）；
    * 训练：``logprob_and_grads`` 手写策略梯度（REINFORCE，
      推导同 lowlevel_rl——只有单 softmax 头，更简单）。

    同 seed 复现：采样 RNG 由 ``seed`` 决定。
    """

    HIDDEN = 32

    def __init__(self, seed: int = 0, weights: dict[str, np.ndarray] | None = None) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        rng = np.random.default_rng(seed)
        if weights is None:
            scale1 = 1.0 / math.sqrt(float(OBS_DIM))
            scale2 = 1.0 / math.sqrt(float(self.HIDDEN))
            self.W1 = rng.normal(0.0, scale1, (OBS_DIM, self.HIDDEN))
            self.b1 = np.zeros(self.HIDDEN)
            self.W2 = rng.normal(0.0, scale2, (self.HIDDEN, N_ACTIONS))
            self.b2 = np.zeros(N_ACTIONS)
        else:
            for name in ("W1", "b1", "W2", "b2"):
                if name not in weights:
                    raise ValueError(f"weights 缺少 {name}")
            self.W1 = np.asarray(weights["W1"], dtype=np.float64).copy()
            self.b1 = np.asarray(weights["b1"], dtype=np.float64).copy()
            self.W2 = np.asarray(weights["W2"], dtype=np.float64).copy()
            self.b2 = np.asarray(weights["b2"], dtype=np.float64).copy()
            self._validate_shapes()

    def _validate_shapes(self) -> None:
        if self.W1.shape != (OBS_DIM, self.HIDDEN):
            raise ValueError(f"W1 {self.W1.shape} != {(OBS_DIM, self.HIDDEN)}")
        if self.b1.shape != (self.HIDDEN,):
            raise ValueError(f"b1 {self.b1.shape} != {(self.HIDDEN,)}")
        if self.W2.shape != (self.HIDDEN, N_ACTIONS):
            raise ValueError(f"W2 {self.W2.shape} != {(self.HIDDEN, N_ACTIONS)}")
        if self.b2.shape != (N_ACTIONS,):
            raise ValueError(f"b2 {self.b2.shape} != {(N_ACTIONS,)}")

    # -- 前向 ----------------------------------------------------------
    def _forward(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """单个体前向：返回 (hidden, logits)。"""
        vector = np.asarray(obs, dtype=np.float64)
        if vector.shape != (OBS_DIM,):
            raise ValueError(f"obs 形状 {vector.shape} != ({OBS_DIM},)")
        if not np.all(np.isfinite(vector)):
            raise ValueError("obs 含非有限值")
        hidden = np.tanh(vector @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        return hidden, logits

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        total = float(exp.sum())
        if not math.isfinite(total) or total <= 0.0:  # pragma: no cover
            return np.full_like(logits, 1.0 / logits.size)
        return exp / total

    # -- SwarmPolicy 协议 ------------------------------------------------
    def act(self, obs_batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(obs_batch, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != OBS_DIM:
            raise ValueError(f"obs_batch 形状 {batch.shape} != (N, {OBS_DIM})")
        # 批量前向
        hidden = np.tanh(batch @ self.W1 + self.b1)      # (N, H)
        logits = hidden @ self.W2 + self.b2              # (N, A)
        # Gumbel-max 采样（向量化 categorical 采样）
        u = self._rng.random(logits.shape)
        gumbel = -np.log(-np.log(np.clip(u, 1e-12, 1.0 - 1e-12)))
        sampled = np.argmax(logits + gumbel, axis=1)
        return sampled.astype(np.int64)

    def parameters(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    # -- 训练（REINFORCE，逐转移）----------------------------------------
    def logprob_and_grads(
        self, obs: np.ndarray, action: int, advantage: float
    ) -> tuple[float, dict[str, np.ndarray]]:
        """``-A·logπ(a|s)`` 及其梯度（手写反向）。

        dL/dlogits = A·(p - onehot(a))；再 tanh 链式回传到 W2/b2/W1/b1。
        """
        vector = np.asarray(obs, dtype=np.float64)
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action {action} 越界")
        advantage = float(advantage)
        if not math.isfinite(advantage):
            raise ValueError("advantage 非有限")
        hidden, logits = self._forward(vector)
        probs = self._softmax(logits)
        logp = math.log(max(float(probs[action]), 1e-12))
        loss = -advantage * logp

        d_logits = advantage * probs.copy()
        d_logits[action] -= advantage
        dW2 = np.outer(hidden, d_logits)
        db2 = d_logits
        d_hidden = d_logits @ self.W2.T
        d_pre = d_hidden * (1.0 - hidden * hidden)
        dW1 = np.outer(vector, d_pre)
        db1 = d_pre
        return float(loss), {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def apply_grads(self, grads: dict[str, np.ndarray], lr: float) -> None:
        """SGD 一步：``param -= lr·grad``。"""
        lr = float(lr)
        if not math.isfinite(lr) or lr < 0.0:
            raise ValueError("lr 必须非负有限")
        if lr == 0.0:
            return
        for name, param in self.parameters().items():
            grad = grads.get(name)
            if grad is None:
                continue
            array = np.asarray(grad, dtype=np.float64)
            if array.shape != param.shape:
                raise ValueError(f"grad {name} 形状 {array.shape} != {param.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"grad {name} 非有限")
            param -= lr * array

    # -- 序列化 ----------------------------------------------------------
    def save(self, path) -> Path:
        target = Path(path)
        if target.suffix.lower() != ".npz":
            target = target.with_name(target.name + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            obs_dim=np.int64(OBS_DIM),
            n_actions=np.int64(N_ACTIONS),
            hidden=np.int64(self.HIDDEN),
            seed=np.int64(self.seed),
            version=np.asarray(1),
        )
        return target

    @classmethod
    def load(cls, path, seed: int = 0) -> "NeuralSwarmPolicy":
        source = Path(path)
        if not source.exists():
            source = source.with_name(source.name + ".npz")
        if not source.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {path}")
        with np.load(source, allow_pickle=False) as data:
            policy = cls(
                seed=seed,
                weights={
                    "W1": np.asarray(data["W1"], dtype=np.float64),
                    "b1": np.asarray(data["b1"], dtype=np.float64),
                    "W2": np.asarray(data["W2"], dtype=np.float64),
                    "b2": np.asarray(data["b2"], dtype=np.float64),
                },
            )
        return policy

    def flat_weights(self) -> np.ndarray:
        """全部参数拍平（比较训练前后变化用）。"""
        return np.concatenate(
            [self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()]
        )


# ---------------------------------------------------------------------------
# 训练器：minibatch REINFORCE（复用 lowlevel_rl 的修复思想）
# ---------------------------------------------------------------------------
class SwarmTrainer:
    """从 ``drain_swarm_transitions()`` 的批量转移训练神经策略。

    参数化上与 :class:`~newworld.lowlevel_rl.LowLevelTrainer` 同构
    （minibatch + 批内优势标准化 + 全局梯度范数裁剪——V2.1 修复
    的三件套），但适配本策略的 4 参数/5 动作空间，是独立轻量实现
    （不复用是为了避免跨 dtype 耦合；两边的修复思想保持一致）。
    """

    def __init__(
        self,
        policy: NeuralSwarmPolicy,
        *,
        lr: float = 0.02,
        gamma: float = 0.96,
        batch_size: int = 512,
        epochs_per_episode: int = 2,
        normalize_advantage: bool = True,
        grad_clip: float = 0.5,
    ) -> None:
        for name in ("act", "logprob_and_grads", "apply_grads", "parameters"):
            if not callable(getattr(policy, name, None)):
                raise TypeError(f"policy 缺少方法 {name}")
        self.policy = policy
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.epochs_per_episode = int(epochs_per_episode)
        self.normalize_advantage = bool(normalize_advantage)
        self.grad_clip = float(grad_clip)
        self.updates = 0
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    @staticmethod
    def discounted_returns(rewards, gamma: float) -> np.ndarray:
        """``G_t = r_t + γ·G_{t+1}``（倒序累积）。"""
        values = [float(r) for r in rewards]
        out = np.zeros(len(values), dtype=np.float64)
        running = 0.0
        for i in range(len(values) - 1, -1, -1):
            running = values[i] + float(gamma) * running
            out[i] = running
        return out

    def _normalized(self, advantages: np.ndarray) -> np.ndarray:
        """批内标准化（std<1e-8 时只减均值）。"""
        if not self.normalize_advantage or advantages.size < 2:
            return advantages
        mean = float(advantages.mean())
        std = float(advantages.std())
        if std < 1e-8:
            return advantages - mean
        return (advantages - mean) / (std + 1e-8)

    def _global_norm(self, grads: dict[str, np.ndarray]) -> float:
        total = 0.0
        for value in grads.values():
            total += float(np.sum(np.asarray(value) ** 2))
        return math.sqrt(max(0.0, total))

    def train_on_transitions(
        self, transitions: list[tuple[np.ndarray, int, float]]
    ) -> dict[str, float]:
        """一批 ``(obs, action, reward)``（按个体轨迹拼接）做一次训练。

        奖励先按轨迹算折扣回报（调用方保证同一条个体轨迹连续），
        再批内标准化后小批量多 epoch 更新。返回本回合统计。
        """
        if not transitions:
            return {"updates": 0.0, "transitions": 0.0, "mean_return": 0.0,
                    "weight_delta": 0.0}
        before = self.policy.flat_weights()
        # 折扣回报（整批作为一条流——个体在数组里连续排列）
        rewards = [t[2] for t in transitions]
        returns = self.discounted_returns(rewards, self.gamma)
        # 转成 (obs, action, advantage)
        items = [(t[0], int(t[1]), float(r)) for t, r in zip(transitions, returns)]
        count = len(items)
        batch = max(1, self.batch_size if self.batch_size > 0 else count)
        rng = np.random.default_rng(self.updates * 31 + 7)
        updates_done = 0
        for _ in range(self.epochs_per_episode):
            order = rng.permutation(count)
            for start in range(0, count, batch):
                index = order[start:start + batch]
                chunk = [items[i] for i in index]
                adv = self._normalized(
                    np.asarray([c[2] for c in chunk], dtype=np.float64)
                )
                grad_sum = {
                    name: np.zeros_like(value)
                    for name, value in self.policy.parameters().items()
                }
                for (obs, action, _r), a in zip(chunk, adv):
                    _loss, grads = self.policy.logprob_and_grads(obs, action, a)
                    for name in grad_sum:
                        grad_sum[name] += grads[name]
                scale = 1.0 / len(chunk)
                averaged = {name: g * scale for name, g in grad_sum.items()}
                if self.grad_clip > 0.0:
                    norm = self._global_norm(averaged)
                    if norm > self.grad_clip > 0.0:
                        factor = self.grad_clip / norm
                        averaged = {name: g * factor for name, g in averaged.items()}
                self.policy.apply_grads(averaged, self.lr)
                updates_done += 1
                self.updates += 1
        after = self.policy.flat_weights()
        delta = float(np.max(np.abs(after - before)))
        stats = {
            "updates": float(updates_done),
            "transitions": float(count),
            "mean_return": float(np.mean(returns)),
            "weight_delta": delta,
        }
        self.history.append(stats)
        return stats
