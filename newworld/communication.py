"""P6-A 涌现通信：信号通道（CommunicationLayer）+ 控制指称博弈（ReferentialGame）。

科学定位
========
1) **控制指称博弈**（Lazaridou et al. 2017 式）：这是能真正演示"指称协议
   涌现"的实验——sender 看到 one-hot cue（cue_dim 选一）发出 K 选一符号，
   listener 看到符号预测 cue，两者共享 0/1 奖励。用准确率 vs 随机基线
   (1/cue_dim)、符号-线索互信息(bits)、符号熵、cue->symbol 映射来度量
   "协议是否形成"。
2) **世界内信号通道**：CommunicationLayer 把符号作为群伙的额外观测列
   （听到的邻居符号直方图），sender 策略用"1 步群体奖励"（Δeat_events/人口）
   做 REINFORCE。注意：在"个体奖励 + 参数共享策略"的设定下，符号本身没有
   个体收益，也没有专用 listener 侧学习信号——群体奖励对单个 sender 的
   信用分配几乎全是噪声。因此**世界通道很可能学不出可辨语义（MI≈0）**；
   这是可报告的诚实结论，不做调参硬造的正结果。语义没有硬编码：符号只
   通过观测列进入世界，协议完全由学习产生（本模块不定义 symbol->meaning）。

与 Lead 冻结 API 的对齐
-----------------------
* build_communication_layer(swarm, cfg, seed=0)；
* attach_to_swarm()：注册 obs provider，维度 = n_symbols；
  provider 返回每个个体"上一步在 radius 内听到的其它个体符号的归一化
  直方图"（没听到 = 全 0），观测总维度 = 9 + n_symbols；
* update(dt)：在 swarm/social 之后调用——(a) 按 compact_keep 重排
  逐行状态，(b) 算 1 步群体奖励，(c) REINFORCE 训练 sender，
  (d) 为全部存活个体重新采样符号，(e) 更新滚动统计；
* stats dict / summary_zh() / symbol_of(row) 供 experiment 与
  lab_server 使用；obs_dim == n_symbols。

跨回合 sender 权重传递（为什么需要）
------------------------------------
common.train_neural_policy 每个训练回合都重建世界，因此也会重建
CommunicationLayer（sender 会回到随机初始化）。为了让 sender 的权重跨回合
持续学习、并让评估阶段"冻结"的 sender 是训练后的 sender（而不是随机网络），
提供 use_communication_sender 上下文：构造层时优先使用注入的 sender。
comm.train=True 时直接共享该对象（梯度持续累积）；comm.train=False
时复制其权重（冻结）。这是实验协议层的显式接线，不是隐藏状态
（仅支持顺序执行，模块级栈在 finally 中弹出）。

统计口径
--------
* symbol_entropy：滚动缓冲内符号分布的经验熵（bits）；
* mutual_information：滚动缓冲内 符号 vs 局部食物梯度桶 的互信息
  （bits）。桶 = 东/西/南/北各 3 格处食物 argmax，再映射到 [0,K-1]
  （K=4 时就是 4 个方向）；这是"符号是否编码个体局部环境"的基本检验，
  本身不假定任何语义；
* food_alignment：symbol == bucket 的比例（无信息时 ≈ 1/K）；
* 有限样本互信息估计有正偏置，约 (K-1)^2/(2N ln2) bits（见
  mi_expected_bias_bits；N=8192、K=4 时约 8e-4 bits）。
"""

from __future__ import annotations

import itertools
import math
from contextlib import contextmanager
from typing import Any

import numpy as np

from .map_engine import WALKABLE
from .swarm_policy import OBS_DIM, NeuralSwarmPolicy

#: 滚动统计缓冲上限（符号/食物桶样本；控制内存与每步统计开销）
DEFAULT_ROLLING = 8192
#: sender/listener 网络隐层宽度默认值（NeuralSwarmPolicy.HIDDEN 同值）
DEFAULT_HIDDEN = 32
#: 4 个探测方向：东 / 西 / 南 / 北（与 swarm_policy 的基础观测顺序一致）
_PROBE_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
#: sender 注入栈（协议层显式接线；顺序使用，不支持并发训练）
_SENDER_STACK: list[Any] = []


# ---------------------------------------------------------------------------
# 信息论小工具（无外部依赖，测试可单独验证）
# ---------------------------------------------------------------------------
def entropy_bits(values: Any, n_states: int | None = None) -> float:
    """离散经验熵（bits）；空输入返回 0。"""
    array = np.asarray(values, dtype=np.int64).ravel()
    if array.size == 0:
        return 0.0
    if n_states is None:
        n_states = int(array.max()) + 1
    n_states = max(1, int(n_states))
    counts = np.bincount(np.clip(array, 0, n_states - 1), minlength=n_states).astype(np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def mutual_information_bits(
    x: Any,
    y: Any,
    n_x: int | None = None,
    n_y: int | None = None,
) -> float:
    """两个离散变量的经验互信息（bits），独立时≈0、完全相同=H(x)。"""
    xx = np.asarray(x, dtype=np.int64).ravel()
    yy = np.asarray(y, dtype=np.int64).ravel()
    n = int(min(xx.size, yy.size))
    if n == 0:
        return 0.0
    xx = xx[:n]
    yy = yy[:n]
    if n_x is None:
        n_x = int(xx.max()) + 1 if xx.size else 1
    if n_y is None:
        n_y = int(yy.max()) + 1 if yy.size else 1
    n_x = max(1, int(n_x))
    n_y = max(1, int(n_y))
    flat = np.clip(xx, 0, n_x - 1) * n_y + np.clip(yy, 0, n_y - 1)
    joint = np.bincount(flat, minlength=n_x * n_y).reshape(n_x, n_y).astype(np.float64)
    joint /= float(n)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = joint > 0.0
    if not mask.any():
        return 0.0
    ratio = joint[mask] / np.maximum(expected[mask], 1e-300)
    mi = float(np.sum(joint[mask] * np.log2(ratio)))
    return max(0.0, mi)


def mi_expected_bias_bits(n_samples: int, n_x: int, n_y: int) -> float:
    """独立变量下插件式互信息估计的近似正偏置（bits，Miller-Madow 近似）。

    保守用 (n_x-1)(n_y-1)/(2 N ln2)；用于判断观测到的 MI 是否只是有限样本
    噪声（通信通道的"机会水平"）。
    """
    n = max(1, int(n_samples))
    bias = float((max(1, int(n_x)) - 1) * (max(1, int(n_y)) - 1)) / (2.0 * n * math.log(2.0))
    return max(0.0, bias)


# ---------------------------------------------------------------------------
# sender 注入（协议层显式接线）
# ---------------------------------------------------------------------------
@contextmanager
def use_communication_sender(policy: NeuralSwarmPolicy):
    """在 with 块内构造的 CommunicationLayer 使用给定 sender 策略。

    * comm.train=True（训练）：层与注入对象共享权重，跨回合持续学习；
    * comm.train=False（评估）：层复制一份权重并冻结（sender 不再更新）。

    仅用于顺序实验；块结束时出栈，不影响块外构造的层。
    """
    if not isinstance(policy, NeuralSwarmPolicy):
        raise TypeError("sender 必须是 NeuralSwarmPolicy")
    _SENDER_STACK.append(policy)
    try:
        yield policy
    finally:
        _SENDER_STACK.pop()


def _current_injected_sender() -> NeuralSwarmPolicy | None:
    return _SENDER_STACK[-1] if _SENDER_STACK else None


def _clone_sender(source: NeuralSwarmPolicy, *, n_symbols: int, seed: int) -> NeuralSwarmPolicy:
    """复制权重构造一个新 sender（独立采样 RNG，用于冻结评估）。"""
    return NeuralSwarmPolicy(
        seed=int(seed),
        obs_dim=OBS_DIM,
        n_actions=int(n_symbols),
        hidden=int(source.hidden),
        weights=source.parameters(),
    )


# ---------------------------------------------------------------------------
# 世界内信号通道
# ---------------------------------------------------------------------------
class CommunicationLayer:
    """把"符号"接入群伙观测的通信层（语义不硬编码，全部由学习产生）。

    行状态：symbols[row]（上一帧发出的符号），随 swarm 压缩按
    compact_keep 重排。符号本身不影响世界规则（message_cost=0 时）；
    它只作为其它个体的额外观测列（听到的符号直方图）进入策略。
    """

    def __init__(
        self,
        swarm: Any,
        config: Any,
        *,
        seed: int = 0,
        sender: NeuralSwarmPolicy | None = None,
    ) -> None:
        self.swarm = swarm
        self.config = config
        self.n_symbols = int(config.n_symbols)
        if self.n_symbols < 2:
            raise ValueError("comm.n_symbols >= 2")
        self.radius = float(config.radius)
        if self.radius <= 0.0:
            raise ValueError("comm.radius > 0")
        self.seed = int(seed)
        self.obs_dim = self.n_symbols
        self.rng = np.random.default_rng(self.seed)
        capacity = int(swarm.data.shape[0])
        #: 逐行符号（int16；0..K-1），随 swarm 压实重排
        self.symbols = np.zeros(capacity, dtype=np.int16)
        #: sender 策略：状态 -> 符号（REINFORCE，1 步群体奖励）
        if sender is not None:
            if int(sender.obs_dim) != OBS_DIM:
                raise ValueError(f"注入 sender 的 obs_dim={sender.obs_dim} != {OBS_DIM}")
            if int(sender.n_actions) != self.n_symbols:
                raise ValueError(
                    f"注入 sender 的 n_actions={sender.n_actions} != n_symbols={self.n_symbols}"
                )
            self.policy = sender
        else:
            self.policy = NeuralSwarmPolicy(
                seed=self.seed,
                obs_dim=OBS_DIM,
                n_actions=self.n_symbols,
                hidden=DEFAULT_HIDDEN,
            )
        # -- 训练状态 ------------------------------------------------------
        self._pending_obs: list[np.ndarray] = []   # 上一步发送符号时的基础观测（9 维）
        self._pending_symbols: list[int] = []      # 上一步发出的符号
        self._train_buffer: list[tuple[np.ndarray, int, float]] = []
        self._baseline = 0.0
        self._reward_var = 1.0
        self._baseline_alpha = 0.05
        self._prev_eat_events = int(getattr(swarm, "eat_events", 0))
        self._last_compact_generation = int(getattr(swarm, "compact_generation", 0))
        # -- 滚动统计缓冲（环形） ------------------------------------------
        self._rolling = DEFAULT_ROLLING
        self._sym_buffer = np.zeros(self._rolling, dtype=np.int16)
        self._food_buffer = np.zeros(self._rolling, dtype=np.int16)
        self._buffer_count = 0
        self._buffer_pos = 0
        # -- 累计量 --------------------------------------------------------
        self.messages = 0
        self.train_updates = 0
        self._group_reward_sum = 0.0
        self._group_reward_n = 0
        self._message_cost_total = 0.0
        self.stats: dict[str, Any] = {
            "messages": 0,
            "symbol_entropy": 0.0,
            "mutual_information": 0.0,
            "food_alignment": 0.0,
            "train_updates": 0,
            "group_reward_mean": 0.0,
            "symbol_counts": [0] * self.n_symbols,
        }

    # ------------------------------------------------------------------
    # 接线
    # ------------------------------------------------------------------
    def attach_to_swarm(self) -> int:
        """注册观测 provider 并初始化符号；返回 swarm.obs_dim。"""
        total = self.swarm.register_obs_provider(self._heard_histogram, dim=self.n_symbols)
        self._last_compact_generation = int(getattr(self.swarm, "compact_generation", 0))
        self._resample_symbols()
        # 初始化采样不产生"上一步听到的符号"奖励配对，清掉 pending
        self._pending_obs = []
        self._pending_symbols = []
        return total

    # ------------------------------------------------------------------
    # 观测 provider：上一步听到的符号直方图
    # ------------------------------------------------------------------
    def _heard_histogram(self, view: Any, rows: np.ndarray) -> np.ndarray:
        """(len(rows), K) 的归一化"听到的符号"直方图；没听到则全 0。"""
        rows = np.asarray(rows, dtype=np.int64)
        n = int(rows.size)
        out = np.zeros((n, self.n_symbols), dtype=np.float64)
        if n == 0:
            return out
        alive = np.nonzero(view["alive"])[0]
        if alive.size < 2:
            return out
        tx = view["x"][rows].astype(np.float64)
        ty = view["y"][rows].astype(np.float64)
        ax = view["x"][alive].astype(np.float64)
        ay = view["y"][alive].astype(np.float64)
        dx = tx[:, None] - ax[None, :]
        dy = ty[:, None] - ay[None, :]
        within = (dx * dx + dy * dy) <= self.radius * self.radius
        # 排除自己（同一行号）
        within &= rows[:, None] != alive[None, :]
        heard_symbols = self.symbols[alive].astype(np.int64)
        counts = np.zeros((n, self.n_symbols), dtype=np.float64)
        for symbol in range(self.n_symbols):
            counts[:, symbol] = (within & (heard_symbols[None, :] == symbol)).sum(axis=1)
        totals = counts.sum(axis=1)
        nonempty = totals > 0.0
        out[nonempty] = counts[nonempty] / totals[nonempty, None]
        return out

    # ------------------------------------------------------------------
    # 每步更新（在 swarm/social 之后调用）
    # ------------------------------------------------------------------
    def update(self, dt: float) -> dict[str, Any]:
        """推进通信一帧：重排 -> 群体奖励 -> 训练 -> 采样 -> 统计。"""
        swarm = self.swarm
        eat_now = int(getattr(swarm, "eat_events", 0))
        delta = eat_now - self._prev_eat_events
        self._prev_eat_events = eat_now
        if swarm.count == 0:
            self.stats.update({
                "messages": int(self.messages),
                "train_updates": int(self.train_updates),
                "group_reward_mean": self._group_reward_mean(),
                "symbol_counts": [0] * self.n_symbols,
            })
            return dict(self.stats)

        # (a) 压实后逐行状态重排（symbols 与 pending 的行号对应关系失效）
        reindexed = self._maybe_reindex()
        if reindexed:
            self._pending_obs = []
            self._pending_symbols = []

        # (b) 1 步群体奖励：本步人均进食事件增量
        population = max(1, int(swarm.population))
        group_reward = float(delta) / float(population)
        self._group_reward_sum += group_reward
        self._group_reward_n += 1

        # (c) REINFORCE 训练 sender（用上一步的 (state, symbol) 配本步群体奖励）
        self._train_step(group_reward)

        # (d) 为所有存活个体采样新符号（作为下一帧"听到的符号"）
        self._resample_symbols()

        # (e) 滚动统计
        self._update_stats()

        self.stats.update({
            "messages": int(self.messages),
            "train_updates": int(self.train_updates),
            "group_reward_mean": self._group_reward_mean(),
            "symbol_counts": self._rolling_symbol_counts(),
        })
        return dict(self.stats)

    # ------------------------------------------------------------------
    def _maybe_reindex(self) -> bool:
        """swarm 压实后按 compact_keep 重排 symbols；返回是否发生重排。"""
        generation = int(getattr(self.swarm, "compact_generation", 0))
        if generation == self._last_compact_generation:
            return False
        self._last_compact_generation = generation
        keep = np.asarray(getattr(self.swarm, "compact_keep", np.arange(0)), dtype=np.int64)
        fresh = np.zeros_like(self.symbols)
        if keep.size:
            fresh[: keep.size] = self.symbols[keep]
        self.symbols = fresh
        return True

    # ------------------------------------------------------------------
    def _base_observations_for_alive(self) -> tuple[np.ndarray, np.ndarray]:
        """存活个体的 (rows, 基础 9 维观测)，不触发扩展 provider（防递归）。"""
        swarm = self.swarm
        view = swarm.data[: swarm.count]
        rows = np.nonzero(view["alive"])[0]
        if rows.size == 0:
            return rows, np.zeros((0, OBS_DIM), dtype=np.float64)
        obs = np.asarray(swarm._base_observations(view, rows), dtype=np.float64)
        return rows, obs

    def _train_step(self, group_reward: float) -> None:
        """把上一步的 (obs, symbol) 与本步群体奖励配对；够 batch 就更新一次。"""
        if not self.config.train:
            return
        if self._pending_symbols:
            # running baseline（EMA）与 running std（优势归一，避免奖励量级 0.1
            # 时梯度小到不可见）；两者都只影响训练，不影响符号采样语义。
            delta = group_reward - self._baseline
            self._baseline += self._baseline_alpha * delta
            self._reward_var = (1.0 - self._baseline_alpha) * self._reward_var + (
                self._baseline_alpha * delta * delta
            )
            scale = 1.0 / math.sqrt(max(self._reward_var, 1e-12))
            advantage = delta * scale
            for obs, symbol in zip(self._pending_obs, self._pending_symbols):
                self._train_buffer.append((obs, int(symbol), float(advantage)))
            self._pending_obs = []
            self._pending_symbols = []
        batch_size = max(1, int(self.config.batch_size))
        if len(self._train_buffer) < batch_size:
            return
        batch = self._train_buffer[:batch_size]
        del self._train_buffer[:batch_size]
        params = self.policy.parameters()
        epochs = max(1, int(self.config.epochs))
        for _ in range(epochs):
            grads = {name: np.zeros_like(value) for name, value in params.items()}
            for obs, symbol, advantage in batch:
                _loss, sample_grads = self.policy.logprob_and_grads(obs, symbol, advantage)
                for name in grads:
                    grads[name] += sample_grads[name]
            scale = 1.0 / float(len(batch))
            averaged = {name: value * scale for name, value in grads.items()}
            self.policy.apply_grads(averaged, float(self.config.lr))
        self.train_updates += 1

    def _resample_symbols(self) -> None:
        """用 sender 策略为全部存活个体采样新符号（并记录训练配对）。"""
        rows, obs = self._base_observations_for_alive()
        if rows.size == 0:
            self._pending_obs = []
            self._pending_symbols = []
            return
        symbols = np.asarray(self.policy.act(obs), dtype=np.int64).ravel()
        if symbols.shape[0] != rows.size:
            raise RuntimeError("sender.act 返回数量与存活个体数不一致")
        if not np.all((symbols >= 0) & (symbols < self.n_symbols)):
            raise RuntimeError("sender 采样出越界符号")
        self.symbols[rows] = symbols.astype(np.int16)
        # 通信成本（可选，cfg.message_cost=0 时无副作用）：真实从发送者扣能量
        cost = float(getattr(self.config, "message_cost", 0.0) or 0.0)
        if cost > 0.0:
            view = self.swarm.data[: self.swarm.count]
            before = view["hunger"][rows].astype(np.float64)
            after = np.maximum(0.0, before - cost)
            view["hunger"][rows] = after
            self._message_cost_total += float(np.sum(before - after))
        self.messages += int(rows.size)
        # 训练配对（下一步结算群体奖励）
        self._pending_obs = [obs[i] for i in range(rows.size)]
        self._pending_symbols = symbols.astype(np.int64).tolist()

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def _food_buckets(self, view: Any, rows: np.ndarray) -> np.ndarray:
        """局部食物梯度桶：4 方向 3 格食物 argmax -> [0, K-1]。"""
        engine = self.swarm.engine
        width, height = int(engine.width), int(engine.height)
        xs = np.clip(view["x"][rows].astype(np.int32), 0, width - 1)
        ys = np.clip(view["y"][rows].astype(np.int32), 0, height - 1)
        probes = np.zeros((rows.size, 4), dtype=np.float64)
        for k, (dx, dy) in enumerate(_PROBE_DIRECTIONS):
            nx = np.clip(xs + dx * 3, 0, width - 1)
            ny = np.clip(ys + dy * 3, 0, height - 1)
            walkable = np.isin(engine.terrain[ny, nx], tuple(WALKABLE))
            food_there = engine.food[ny, nx].astype(np.float64)
            probes[:, k] = np.where(walkable, food_there, -1.0)
        best = np.argmax(probes, axis=1)  # 0..3
        bucket = (best * self.n_symbols) // 4
        return np.clip(bucket, 0, self.n_symbols - 1).astype(np.int16)

    def _update_stats(self) -> None:
        """把本帧符号与食物桶写入环形缓冲，重算滚动熵/MI/对齐率。"""
        view = self.swarm.data[: self.swarm.count]
        rows = np.nonzero(view["alive"])[0]
        symbols = self.symbols[rows].astype(np.int16)
        buckets = self._food_buckets(view, rows)
        self._append_rolling(symbols, buckets)
        count = self._buffer_count
        if count <= 0:
            return
        sym = self._sym_buffer[:count]
        food = self._food_buffer[:count]
        self.stats["symbol_entropy"] = entropy_bits(sym, self.n_symbols)
        self.stats["mutual_information"] = mutual_information_bits(
            sym, food, self.n_symbols, self.n_symbols
        )
        self.stats["food_alignment"] = float(np.mean(sym == food))

    def _append_rolling(self, symbols: np.ndarray, buckets: np.ndarray) -> None:
        n = int(symbols.size)
        if n <= 0:
            return
        if n >= self._rolling:
            self._sym_buffer[:] = symbols[-self._rolling:]
            self._food_buffer[:] = buckets[-self._rolling:]
            self._buffer_count = self._rolling
            self._buffer_pos = 0
            return
        index = (self._buffer_pos + np.arange(n)) % self._rolling
        self._sym_buffer[index] = symbols
        self._food_buffer[index] = buckets
        self._buffer_pos = int((self._buffer_pos + n) % self._rolling)
        self._buffer_count = min(self._rolling, self._buffer_count + n)

    def _rolling_symbol_counts(self) -> list[int]:
        count = self._buffer_count
        if count <= 0:
            return [0] * self.n_symbols
        counts = np.bincount(self._sym_buffer[:count].astype(np.int64), minlength=self.n_symbols)
        return [int(value) for value in counts[: self.n_symbols].tolist()]

    def _group_reward_mean(self) -> float:
        if self._group_reward_n <= 0:
            return 0.0
        return float(self._group_reward_sum / self._group_reward_n)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def symbol_of(self, row: int) -> int:
        """某一行个体上一次发出的符号（lab_server 个体详情用）。"""
        row = int(row)
        if row < 0 or row >= self.symbols.size:
            raise IndexError(f"row {row} 越界（capacity={self.symbols.size}）")
        return int(self.symbols[row])

    def summary_zh(self) -> str:
        s = self.stats
        return (
            f"消息 {s.get('messages', 0)} 熵 {s.get('symbol_entropy', 0.0):.2f}bit "
            f"MI {s.get('mutual_information', 0.0):.3f}bit "
            f"对齐 {s.get('food_alignment', 0.0):.2f} "
            f"更新 {s.get('train_updates', 0)} "
            f"群体奖励 {s.get('group_reward_mean', 0.0):.3f}"
        )


def build_communication_layer(swarm: Any, config: Any, *, seed: int = 0) -> CommunicationLayer:
    """按配置构造通信层（experiment._build_comm 的唯一入口）。"""
    if int(config.n_symbols) < 2:
        raise ValueError("comm.n_symbols >= 2")
    if float(config.radius) <= 0.0:
        raise ValueError("comm.radius > 0")
    sender: NeuralSwarmPolicy | None = None
    injected = _current_injected_sender()
    if injected is not None:
        if int(injected.obs_dim) != OBS_DIM:
            raise ValueError(f"注入 sender 的 obs_dim={injected.obs_dim} != {OBS_DIM}")
        if int(injected.n_actions) != int(config.n_symbols):
            raise ValueError(
                f"注入 sender 的 n_actions={injected.n_actions} != comm.n_symbols={config.n_symbols}"
            )
        if bool(config.train):
            sender = injected  # 训练：共享对象，权重跨回合持续学习
        else:
            sender = _clone_sender(injected, n_symbols=int(config.n_symbols), seed=int(seed))
    return CommunicationLayer(swarm, config, seed=int(seed), sender=sender)


# ---------------------------------------------------------------------------
# 控制指称博弈（Lazaridou-style referential game）
# ---------------------------------------------------------------------------
class ReferentialGame:
    """控制指称博弈：sender(cue)->symbol，listener(symbol)->cue，共享 0/1 奖励。

    观测均为 one-hot：sender 输入 cue_dim 维（cue），listener 输入
    n_symbols 维（symbol）；两个 NeuralSwarmPolicy 各自 softmax 采样，
    用**批内均值基线**的 REINFORCE 联合训练（同一团队奖励）。
    这是"指称协议能否涌现"的最小可测舞台：训练后看准确率、互信息、
    符号熵与 cue->symbol 映射，而不是看任何人工定义的语义。
    """

    def __init__(
        self,
        *,
        cue_dim: int = 4,
        n_symbols: int = 4,
        hidden: int = DEFAULT_HIDDEN,
        seed: int = 0,
        lr: float = 0.1,
    ) -> None:
        self.cue_dim = int(cue_dim)
        self.n_symbols = int(n_symbols)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.seed = int(seed)
        if self.cue_dim < 2 or self.n_symbols < 2:
            raise ValueError("cue_dim/n_symbols 都必须 >= 2")
        self.sender = NeuralSwarmPolicy(
            seed=self.seed, obs_dim=self.cue_dim, n_actions=self.n_symbols, hidden=self.hidden
        )
        self.listener = NeuralSwarmPolicy(
            seed=self.seed + 1, obs_dim=self.n_symbols, n_actions=self.cue_dim, hidden=self.hidden
        )
        self.rng = np.random.default_rng(self.seed + 2)
        self.updates = 0
        self.history: list[dict[str, float]] = []
        self._cue_onehot = np.eye(self.cue_dim, dtype=np.float64)
        self._symbol_onehot = np.eye(self.n_symbols, dtype=np.float64)

    # ------------------------------------------------------------------
    def train(self, episodes: int, batch: int = 64) -> list[dict[str, float]]:
        """训练 episodes 轮；每轮 batch 个博弈样本做一次联合更新。"""
        episodes = int(episodes)
        batch = int(batch)
        if episodes < 1 or batch < 1:
            raise ValueError("episodes/batch 必须 >= 1")
        for _ in range(episodes):
            stats = self._train_batch(batch)
            stats["episode"] = float(self.updates)
            self.updates += 1
            self.history.append(stats)
        return self.history

    def _train_batch(self, batch: int) -> dict[str, float]:
        cues = self.rng.integers(0, self.cue_dim, size=batch)
        cue_obs = self._cue_onehot[cues]
        symbols = np.asarray(self.sender.act(cue_obs), dtype=np.int64)
        symbol_obs = self._symbol_onehot[symbols]
        predictions = np.asarray(self.listener.act(symbol_obs), dtype=np.int64)
        rewards = (predictions == cues).astype(np.float64)
        advantage = rewards - float(rewards.mean())  # 批内均值基线
        sender_grads = {name: np.zeros_like(value) for name, value in self.sender.parameters().items()}
        listener_grads = {name: np.zeros_like(value) for name, value in self.listener.parameters().items()}
        for i in range(batch):
            _loss, grads = self.sender.logprob_and_grads(cue_obs[i], int(symbols[i]), float(advantage[i]))
            for name in sender_grads:
                sender_grads[name] += grads[name]
            _loss, grads = self.listener.logprob_and_grads(
                symbol_obs[i], int(predictions[i]), float(advantage[i])
            )
            for name in listener_grads:
                listener_grads[name] += grads[name]
        scale = 1.0 / float(batch)
        self.sender.apply_grads({name: value * scale for name, value in sender_grads.items()}, self.lr)
        self.listener.apply_grads({name: value * scale for name, value in listener_grads.items()}, self.lr)
        return {
            "train_accuracy": float(rewards.mean()),
            "symbol_entropy": entropy_bits(symbols, self.n_symbols),
            "cue_symbol_mi": mutual_information_bits(cues, symbols, self.cue_dim, self.n_symbols),
        }

    # ------------------------------------------------------------------
    def evaluate(self, n: int = 2000) -> dict[str, Any]:
        """在 n 个随机 cue 上评估指称成功（确定性：同 seed 同训练结果）。"""
        n = int(n)
        if n < 1:
            raise ValueError("n >= 1")
        cues = self.rng.integers(0, self.cue_dim, size=n)
        cue_obs = self._cue_onehot[cues]
        symbols = np.asarray(self.sender.act(cue_obs), dtype=np.int64)
        symbol_obs = self._symbol_onehot[symbols]
        predictions = np.asarray(self.listener.act(symbol_obs), dtype=np.int64)
        accuracy = float(np.mean(predictions == cues))
        mapping: dict[int, int] = {}
        for cue in range(self.cue_dim):
            mask = cues == cue
            if not mask.any():
                mapping[cue] = -1
                continue
            counts = np.bincount(symbols[mask], minlength=self.n_symbols)
            mapping[cue] = int(np.argmax(counts))
        return {
            "accuracy": accuracy,
            "chance": float(1.0 / self.cue_dim),
            "mutual_information_bits": mutual_information_bits(
                cues, symbols, self.cue_dim, self.n_symbols
            ),
            "symbol_entropy": entropy_bits(symbols, self.n_symbols),
            "mapping": mapping,
            "n": n,
        }


def mapping_similarity(
    left: dict[int, int], right: dict[int, int], n_symbols: int
) -> float:
    """两个 cue->symbol 映射在**符号重命名下**的最大一致率（0..1）。

    符号标签本身是任意的（哪个符号叫 0 并无意义），所以跨 seed 比较协议
    稳定性必须允许符号置换；n_symbols <= 7 时暴力枚举，否则退化为贪心。
    """
    cues = sorted(set(left) & set(right))
    if not cues:
        return 0.0
    best = 0.0
    if n_symbols <= 7:
        for permutation in itertools.permutations(range(int(n_symbols))):
            score = sum(1 for cue in cues if permutation[int(left[cue])] == int(right[cue]))
            best = max(best, score / len(cues))
    else:  # pragma: no cover - 本项目 K=4，仅防御性分支
        used: set[int] = set()
        hits = 0
        for cue in cues:
            if left[cue] not in used:
                used.add(int(left[cue]))
                hits += 1
        best = hits / len(cues)
    return float(best)


__all__ = [
    "DEFAULT_ROLLING",
    "CommunicationLayer",
    "ReferentialGame",
    "build_communication_layer",
    "use_communication_sender",
    "entropy_bits",
    "mutual_information_bits",
    "mi_expected_bias_bits",
    "mapping_similarity",
]
