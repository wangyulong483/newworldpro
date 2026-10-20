"""低层动作策略的真实强化学习（REINFORCE）——把行为克隆网络变成可采样策略。

背景
====
现有 ``neural.NeuralPolicy`` 是**监督式行为克隆（BC）**：用手写教师的标签训练
40/56→64→32→58 的 MLP，没有奖励、没有回报、没有策略梯度（见
``docs/architecture_audit.md``）。本模块把它包装成一个**随机策略**，并用真正的
策略梯度训练：

* 状态：``neural.features(world, perception, character)`` 的输出（40 维，V2 视觉
  接入后为 56 维；本模块从权重形状自适应，不硬编码维度）；
* 动作：mode（5 类 softmax）× move（3 类 softmax）× jump（Bernoulli）×
  speak（Bernoulli）四个**决策头**的联合动作；
* 奖励：完全由集成层按真实世界状态变化计算（``population`` 内），本模块不做
  奖励塑形、不使用任何 BC 标签；
* 学习：采样动作 → 累积折扣回报（按 agent 分别累计，带指数滑动基线）→ numpy
  手写反向传播 → 更新参数。语言头（每音节 16 词表 × 3 音节）**不参与 RL**：
  它的 logits 梯度恒为 0，保持 argmax 解码（``NeuralPolicy.act`` 不变）。

动作索引约定（集成层必须按此映射）
----------------------------------
``ActionSample.move`` 是 **NeuralPolicy move 头的原始下标** ``0..2``，解码表为
:data:`MOVE_VALUES` = ``(1, 0, -1)``：0→右(+1)、1→站(0)、2→左(-1)，与
``NeuralPolicy.act`` 的 ``(1,0,-1)[move_index]`` 完全一致。集成层可用
:meth:`LowLevelPolicy.decode_move` 或 :data:`MOVE_VALUES` 转成 ``Decision`` 的
move 值。``mode`` 是 :data:`MODE_NAMES` 的下标（explore/forage/drink/sleep/rest）。

契约（``docs/upgrade_contracts_v2.md`` §2.2）
----------------------------------------------
``ActionSample / LowLevelPolicy / LowLevelTrainer`` 的签名按冻结契约实现；
``LowLevelPolicy`` 额外提供 ``feature_dim / logprob / probability / flat_weights /
decode_move`` 等只读诊断接口（不影响契约语义）。训练器的 env 由 Lead 提供
（``simulation.make_lowlevel_rl_env``），构造 env 后自动调用
``env.population.attach_lowlevel_policy(self.policy)``，每步从
``env.population.drain_lowlevel_transitions()`` 取
``(agent_id, features, action: ActionSample, reward)``；按 agent 分别计算折扣回报
与指数滑动基线，累加梯度后每回合做一次更新。

**不得**把本模块的结果说成"已训练出更聪明的策略"：单测只验证机制（梯度正确、
采样正确、更新方向正确、序列化正确）；训练效果由 Lead 的对照实验给出。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .neural import (
    AUX_DIM,
    HIDDEN1,
    HIDDEN2,
    LANG_TOKENS,
    MODES,
    MOVE_DIM,
    VOCAB,
    NeuralPolicy,
)

# ---------------------------------------------------------------------------
# 冻结的动作空间 / 网络输出切片
# ---------------------------------------------------------------------------
#: 行为模式名（与 ``neural.MODES`` 一致）
MODE_NAMES: tuple[str, ...] = MODES
#: move 头的下标到实际移动值的解码表（与 ``NeuralPolicy.act`` 一致）
MOVE_VALUES: tuple[int, ...] = (1, 0, -1)

#: 输出头切片（从 ``NeuralPolicy`` 的 58 维输出里解析）
MODE_SLICE = slice(0, len(MODES))
MOVE_SLICE = slice(len(MODES), len(MODES) + MOVE_DIM)
AUX_START = len(MODES) + MOVE_DIM
JUMP_INDEX = AUX_START
SPEAK_INDEX = AUX_START + 1
LANG_START = AUX_START + AUX_DIM
#: 期望的输出维度（模式+移动+辅助+语言）
EXPECTED_OUTPUT_DIM = len(MODES) + MOVE_DIM + AUX_DIM + LANG_TOKENS * len(VOCAB)
#: 数值下限，避免 log(0)
_EPS = 1e-12


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _sigmoid(value: float) -> float:
    """数值稳定的 sigmoid。"""
    value = float(value)
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax（输入一维）。"""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if not math.isfinite(total) or total <= 0.0:  # pragma: no cover - 保护
        return np.full_like(logits, 1.0 / float(logits.size))
    return exp / total


def _finite_array(value, name: str) -> np.ndarray:
    """转成 float64 数组并检查有限性。"""
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 含非有限值（NaN/inf）")
    return array


@dataclass
class ActionSample:
    """一次低层动作采样（四个决策头的联合动作）。

    ``mode`` / ``move`` 是 softmax 头的下标；``jump`` / ``speak`` 是 Bernoulli
    头的结果；``logprob`` 是**联合**对数概率
    ``log p(mode) + log p(move) + log p(jump) + log p(speak)``。

    语言头不在本结构里：它是 argmax 解码的确定性输出，不参与 RL。
    """

    mode: int
    move: int
    jump: bool
    speak: bool
    logprob: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """转为可 JSON 序列化的 dict（日志/诊断用）。"""
        return {
            "mode": int(self.mode),
            "move": int(self.move),
            "jump": bool(self.jump),
            "speak": bool(self.speak),
            "logprob": float(self.logprob),
        }


# ---------------------------------------------------------------------------
# 低层策略：包装 NeuralPolicy 的可采样策略
# ---------------------------------------------------------------------------
class LowLevelPolicy:
    """把 :class:`~newworld.neural.NeuralPolicy` 变成可采样的随机策略。

    * mode / move：对对应 logits 做 softmax 后采样；
    * jump / speak：对 logits 做 sigmoid 得到 Bernoulli 概率后采样；
    * 语言头：**不参与 RL**，保持 ``NeuralPolicy.act`` 的 argmax 解码。

    ``seed`` 只决定内部采样 RNG（实验里固定它即可复现采样序列）；网络权重来自
    传入的 ``neural_policy``。
    """

    def __init__(self, neural_policy, seed: int = 0) -> None:
        if neural_policy is None or not hasattr(neural_policy, "forward"):
            raise TypeError("neural_policy 必须是 NeuralPolicy（或等价对象）")
        self._policy = neural_policy
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._feature_dim = int(np.asarray(neural_policy.w1).shape[0])
        self._output_dim = int(np.asarray(neural_policy.w3).shape[1])
        if self._output_dim < LANG_START:
            raise ValueError(
                f"NeuralPolicy 输出维度 {self._output_dim} 太小，至少需要 {LANG_START}"
            )

    # -- 只读属性 ------------------------------------------------------
    @property
    def neural_policy(self):
        """被包装的 :class:`~newworld.neural.NeuralPolicy`。"""
        return self._policy

    @property
    def feature_dim(self) -> int:
        """输入特征维度（由权重形状决定；当前 40，V2 视觉接入后 56）。"""
        return self._feature_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def num_modes(self) -> int:
        return len(MODE_NAMES)

    def decode_move(self, move_index: int) -> int:
        """move 头下标 → 实际移动值（``MOVE_VALUES``）。"""
        index = int(move_index)
        if index < 0 or index >= MOVE_DIM:
            raise ValueError(f"move 下标必须在 0..{MOVE_DIM - 1}，得到 {move_index}")
        return int(MOVE_VALUES[index])

    def flat_weights(self) -> np.ndarray:
        """全部网络权重（扁平数组，便于比较参数是否更新）。"""
        return np.asarray(self._policy.flat_weights(), dtype=np.float64).copy()

    # -- 输入 / 前向 ---------------------------------------------------
    def _as_features(self, features) -> np.ndarray:
        vector = np.asarray(features, dtype=np.float64).ravel()
        if vector.shape != (self._feature_dim,):
            raise ValueError(
                f"features 维度必须为 ({self._feature_dim},)，得到 {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("features 含非有限值（NaN/inf）")
        return vector

    def _forward(self, features) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 ``(logits, h1, h2)``。"""
        vector = self._as_features(features)
        logits, h1, h2 = self._policy.forward(vector)
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        if logits.shape != (self._output_dim,):
            raise ValueError(
                f"网络输出维度 {logits.shape} != 期望 ({self._output_dim},)"
            )
        if not np.all(np.isfinite(logits)):
            raise ValueError("网络输出含非有限值")
        return logits, np.asarray(h1, dtype=np.float64), np.asarray(h2, dtype=np.float64)

    @staticmethod
    def _head_logits(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        """从 58 维输出解析出四个 RL 头的 logits。"""
        return (
            logits[MODE_SLICE],
            logits[MOVE_SLICE],
            float(logits[JUMP_INDEX]),
            float(logits[SPEAK_INDEX]),
        )

    def _validate_action(self, action: ActionSample) -> ActionSample:
        if isinstance(action, dict):
            action = ActionSample(
                mode=int(action.get("mode", 0)),
                move=int(action.get("move", 0)),
                jump=bool(action.get("jump", False)),
                speak=bool(action.get("speak", False)),
                logprob=float(action.get("logprob", 0.0)),
            )
        if not isinstance(action, ActionSample):
            raise TypeError("action 必须是 ActionSample（或等价 dict）")
        mode = int(action.mode)
        move = int(action.move)
        if mode < 0 or mode >= len(MODE_NAMES):
            raise ValueError(f"mode 必须在 0..{len(MODE_NAMES) - 1}，得到 {mode}")
        if move < 0 or move >= MOVE_DIM:
            raise ValueError(f"move 必须在 0..{MOVE_DIM - 1}，得到 {move}")
        return ActionSample(
            mode=mode,
            move=move,
            jump=bool(action.jump),
            speak=bool(action.speak),
            logprob=float(action.logprob),
        )

    @staticmethod
    def _draw_float(rng) -> float:
        """从任意兼容 RNG 取一个 [0,1) 均匀数。"""
        if rng is None:
            raise ValueError("rng 不能为 None（内部 RNG 由调用方决定）")
        draw_fn = getattr(rng, "random", None)
        if not callable(draw_fn):
            draw_fn = getattr(rng, "random_sample", None)
        if not callable(draw_fn):
            raise TypeError("rng 必须提供 random()/random_sample() 方法")
        value = float(draw_fn())
        if not math.isfinite(value):
            value = 0.0
        return min(max(value, 0.0), 1.0 - _EPS)

    @staticmethod
    def _categorical(probabilities: np.ndarray, draw: float) -> int:
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += float(probability)
            if draw < cumulative:
                return int(index)
        return int(len(probabilities) - 1)

    # -- 对数概率 ------------------------------------------------------
    def _action_logprob_from_heads(
        self,
        mode_probs: np.ndarray,
        move_probs: np.ndarray,
        jump_prob: float,
        speak_prob: float,
        action: ActionSample,
    ) -> float:
        logp = math.log(max(float(mode_probs[action.mode]), _EPS))
        logp += math.log(max(float(move_probs[action.move]), _EPS))
        if action.jump:
            logp += math.log(max(jump_prob, _EPS))
        else:
            logp += math.log(max(1.0 - jump_prob, _EPS))
        if action.speak:
            logp += math.log(max(speak_prob, _EPS))
        else:
            logp += math.log(max(1.0 - speak_prob, _EPS))
        return float(logp)

    def _heads_and_probs(self, features):
        logits, h1, h2 = self._forward(features)
        mode_logits, move_logits, jump_logit, speak_logit = self._head_logits(logits)
        mode_probs = _softmax(mode_logits)
        move_probs = _softmax(move_logits)
        jump_prob = _sigmoid(jump_logit)
        speak_prob = _sigmoid(speak_logit)
        return logits, h1, h2, mode_probs, move_probs, jump_prob, speak_prob

    def logprob(self, features, action: ActionSample) -> float:
        """给定特征与动作的联合对数概率（诊断/测试用）。"""
        action = self._validate_action(action)
        _, _, _, mode_probs, move_probs, jump_prob, speak_prob = self._heads_and_probs(
            features
        )
        return self._action_logprob_from_heads(
            mode_probs, move_probs, jump_prob, speak_prob, action
        )

    def probability(self, features, action: ActionSample) -> float:
        """联合动作概率 ``exp(logprob)``（诊断/测试用）。"""
        return float(math.exp(self.logprob(features, action)))

    # -- 采样 / 确定性解码 --------------------------------------------
    def sample(self, features, rng=None) -> ActionSample:
        """按策略分布采样一帧动作。

        ``rng`` 可为 ``np.random.Generator`` / ``np.random.RandomState`` /
        ``random.Random``；``None`` 时使用策略内部 RNG（``policy.sample(features)``
        因此成立，集成层无需关心 RNG）。
        """
        generator = self._rng if rng is None else rng
        logits, h1, h2 = self._forward(features)
        mode_logits, move_logits, jump_logit, speak_logit = self._head_logits(logits)
        mode_probs = _softmax(mode_logits)
        move_probs = _softmax(move_logits)
        jump_prob = _sigmoid(jump_logit)
        speak_prob = _sigmoid(speak_logit)

        mode = self._categorical(mode_probs, self._draw_float(generator))
        move = self._categorical(move_probs, self._draw_float(generator))
        jump = self._draw_float(generator) < jump_prob
        speak = self._draw_float(generator) < speak_prob
        action = ActionSample(mode=mode, move=move, jump=jump, speak=speak, logprob=0.0)
        action.logprob = self._action_logprob_from_heads(
            mode_probs, move_probs, jump_prob, speak_prob, action
        )
        return action

    def deterministic(self, features) -> ActionSample:
        """argmax 解码（评估用，不在训练路径上）。"""
        logits, h1, h2 = self._forward(features)
        mode_logits, move_logits, jump_logit, speak_logit = self._head_logits(logits)
        mode = int(np.argmax(mode_logits))
        move = int(np.argmax(move_logits))
        jump = bool(jump_logit > 0.0)
        speak = bool(speak_logit > 0.0)
        mode_probs = _softmax(mode_logits)
        move_probs = _softmax(move_logits)
        action = ActionSample(mode=mode, move=move, jump=jump, speak=speak, logprob=0.0)
        action.logprob = self._action_logprob_from_heads(
            mode_probs, move_probs, _sigmoid(jump_logit), _sigmoid(speak_logit), action
        )
        return action

    # -- 手写反向（REINFORCE）-----------------------------------------
    def logprob_and_grads(
        self, features, action: ActionSample, advantage: float
    ) -> tuple[float, dict[str, np.ndarray]]:
        """返回 ``-advantage * logprob`` 及其对全部权重的梯度。

        数学推导（四个头独立）::

            loss   = -A * [log p(mode) + log p(move) + log p(jump) + log p(speak)]
            dL/d(mode_logits) =  A * (p(mode) - onehot(mode))
            dL/d(move_logits) =  A * (p(move) - onehot(move))
            dL/d(jump_logit)  =  A * (sigmoid(z_jump)  - 1{jump})
            dL/d(speak_logit) =  A * (sigmoid(z_speak) - 1{speak})

        语言头 logits 的梯度恒为 0（不参与 RL）。随后用链式法则反传到
        ``w3/b3 → w2/b2 → w1/b1``（tanh 隐藏层）。
        """
        action = self._validate_action(action)
        advantage = float(advantage)
        if not math.isfinite(advantage):
            raise ValueError("advantage 必须是有限数值")
        (
            logits,
            h1,
            h2,
            mode_probs,
            move_probs,
            jump_prob,
            speak_prob,
        ) = self._heads_and_probs(features)
        logp = self._action_logprob_from_heads(
            mode_probs, move_probs, jump_prob, speak_prob, action
        )
        loss = -advantage * logp

        # 输出层梯度：只在四个 RL 头上有值，语言头严格为 0
        d_logits = np.zeros_like(logits)
        d_mode = mode_probs.copy()
        d_mode[action.mode] -= 1.0
        d_logits[MODE_SLICE] = advantage * d_mode
        d_move = move_probs.copy()
        d_move[action.move] -= 1.0
        d_logits[MOVE_SLICE] = advantage * d_move
        d_logits[JUMP_INDEX] = advantage * (jump_prob - (1.0 if action.jump else 0.0))
        d_logits[SPEAK_INDEX] = advantage * (speak_prob - (1.0 if action.speak else 0.0))

        # w3/b3
        dw3 = np.outer(h2, d_logits)
        db3 = d_logits
        # w2/b2（tanh 链式）
        dh2 = d_logits @ np.asarray(self._policy.w3, dtype=np.float64).T
        dz2 = dh2 * (1.0 - h2 * h2)
        dw2 = np.outer(h1, dz2)
        db2 = dz2
        # w1/b1
        dh1 = dz2 @ np.asarray(self._policy.w2, dtype=np.float64).T
        dz1 = dh1 * (1.0 - h1 * h1)
        features_vector = self._as_features(features)
        dw1 = np.outer(features_vector, dz1)
        db1 = dz1
        grads = {
            "w1": dw1,
            "b1": db1,
            "w2": dw2,
            "b2": db2,
            "w3": dw3,
            "b3": db3,
        }
        return float(loss), grads

    def apply_grads(self, grads: dict[str, np.ndarray], lr: float) -> None:
        """梯度下降一步：``param -= lr * grad``（语言头列天然不受影响）。"""
        step = float(lr)
        if not math.isfinite(step) or step < 0.0:
            raise ValueError("lr 必须是有限非负数")
        if step == 0.0:
            return
        if not isinstance(grads, dict):
            raise TypeError("grads 必须是 dict")
        for name, parameter in (
            ("w1", self._policy.w1),
            ("b1", self._policy.b1),
            ("w2", self._policy.w2),
            ("b2", self._policy.b2),
            ("w3", self._policy.w3),
            ("b3", self._policy.b3),
        ):
            grad = grads.get(name, None)
            if grad is None:
                continue
            array = _finite_array(grad, f"梯度 {name}")
            target = np.asarray(parameter, dtype=np.float64)
            if array.shape != target.shape:
                raise ValueError(
                    f"梯度 {name} 形状 {array.shape} != 参数形状 {target.shape}"
                )
            parameter -= step * array

    # -- 序列化 --------------------------------------------------------
    @staticmethod
    def _normalize_path(path) -> Path:
        resolved = Path(path)
        if resolved.suffix.lower() != ".npz":
            resolved = resolved.with_name(resolved.name + ".npz")
        return resolved

    def save(self, path) -> Path:
        """保存为 ``.npz`` checkpoint（缺失 ``.npz`` 后缀时自动补上）。"""
        target = self._normalize_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            weights=self._policy.flat_weights(),
            feature_dim=np.int64(self._feature_dim),
            hidden1=np.int64(np.asarray(self._policy.w1).shape[1]),
            hidden2=np.int64(np.asarray(self._policy.w2).shape[1]),
            output_dim=np.int64(self._output_dim),
            vocab=np.asarray(VOCAB),
            version=np.asarray(4),
            format=np.asarray("newworld.lowlevel_rl/v1"),
            seed=np.int64(self.seed),
        )
        return target

    @classmethod
    def load(cls, path, seed: int = 0) -> "LowLevelPolicy":
        """从 checkpoint 还原；输入维度不同的旧 checkpoint 零初始化迁移。

        V2 视觉把输入从 40 扩到 56 时，旧 checkpoint 的新输入列补 0（与
        ``NeuralPolicy.from_legacy_weights`` 同一迁移思路），因此旧策略在新网络
        上行为逐比特一致，直到继续训练改变权重。
        """
        source = Path(path)
        if not source.exists() and cls._normalize_path(source).exists():
            source = cls._normalize_path(source)
        if not source.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {source}")
        with np.load(source, allow_pickle=False) as data:
            for key in ("weights", "feature_dim", "hidden1", "hidden2", "output_dim"):
                if key not in data.files:
                    raise ValueError(f"checkpoint 缺少字段: {key}")
            flat = np.asarray(data["weights"], dtype=np.float64).reshape(-1)
            feature_dim = int(np.asarray(data["feature_dim"]).reshape(-1)[0])
            hidden1 = int(np.asarray(data["hidden1"]).reshape(-1)[0])
            hidden2 = int(np.asarray(data["hidden2"]).reshape(-1)[0])
            output_dim = int(np.asarray(data["output_dim"]).reshape(-1)[0])
            checkpoint_seed = (
                int(np.asarray(data["seed"]).reshape(-1)[0])
                if "seed" in data.files
                else 0
            )
        policy = NeuralPolicy(seed=0)
        current_feature_dim = int(np.asarray(policy.w1).shape[0])
        current_output_dim = int(np.asarray(policy.w3).shape[1])
        if hidden1 != HIDDEN1 or hidden2 != HIDDEN2:
            raise ValueError(
                f"checkpoint 隐层 ({hidden1},{hidden2}) != 当前 ({HIDDEN1},{HIDDEN2})"
            )
        if output_dim != current_output_dim:
            raise ValueError(
                f"checkpoint 输出维度 {output_dim} != 当前 {current_output_dim}"
            )
        expected_flat = (
            feature_dim * hidden1 + hidden1
            + hidden1 * hidden2 + hidden2
            + hidden2 * output_dim + output_dim
        )
        if flat.size != expected_flat:
            raise ValueError(
                f"checkpoint 权重数 {flat.size} != 按元数据计算的 {expected_flat}"
            )
        if feature_dim == current_feature_dim:
            policy.set_flat_weights(flat)
        elif feature_dim < current_feature_dim:
            # 旧 checkpoint（例如 40 维）→ 新网络：新输入列零初始化迁移
            index = 0
            old_w1 = flat[index : index + feature_dim * hidden1].reshape(
                feature_dim, hidden1
            )
            index += feature_dim * hidden1
            old_b1 = flat[index : index + hidden1]
            index += hidden1
            old_w2 = flat[index : index + hidden1 * hidden2].reshape(hidden1, hidden2)
            index += hidden1 * hidden2
            old_b2 = flat[index : index + hidden2]
            index += hidden2
            old_w3 = flat[index : index + hidden2 * output_dim].reshape(
                hidden2, output_dim
            )
            index += hidden2 * output_dim
            old_b3 = flat[index : index + output_dim]
            new_w1 = np.zeros((current_feature_dim, hidden1), dtype=np.float64)
            new_w1[:feature_dim] = old_w1
            policy.w1 = new_w1
            policy.b1 = old_b1.copy()
            policy.w2 = old_w2.copy()
            policy.b2 = old_b2.copy()
            policy.w3 = old_w3.copy()
            policy.b3 = old_b3.copy()
        else:
            raise ValueError(
                f"checkpoint 输入维度 {feature_dim} 大于当前网络 {current_feature_dim}"
            )
        # seed 参数优先（实验里固定 policy seed）；checkpoint 里的 seed 仅供诊断
        return cls(policy, seed=int(seed))


# ---------------------------------------------------------------------------
# 训练器：REINFORCE（按 agent 分别累计折扣回报 + 指数滑动基线）
# ---------------------------------------------------------------------------
class LowLevelTrainer:
    """用真实的 ``(agent_id, features, action, reward)`` 转移训练低层策略。

    奖励由集成层（``rewards.LowLevelReward`` 等）按真实世界状态变化计算，本类
    只负责：按 agent 分组 → 折扣回报 → 减基线得到 advantage → 手写反向 →
    每回合一次参数更新 → 独立 seed 评估。
    """

    def __init__(
        self,
        policy: LowLevelPolicy,
        env_factory,
        *,
        seed: int = 0,
        lr: float = 0.01,
        gamma: float = 0.95,
        baseline_decay: float = 0.9,
    ) -> None:
        if not callable(env_factory):
            raise TypeError("env_factory 必须可调用")
        for method in ("sample", "deterministic", "logprob_and_grads", "apply_grads"):
            if not callable(getattr(policy, method, None)):
                raise TypeError(f"policy 缺少方法 {method}")
        lr = float(lr)
        gamma = float(gamma)
        baseline_decay = float(baseline_decay)
        if not math.isfinite(lr) or lr < 0.0:
            raise ValueError("lr 必须是有限非负数")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma 必须在 [0, 1]")
        if not 0.0 <= baseline_decay < 1.0:
            raise ValueError("baseline_decay 必须在 [0, 1)")
        self.policy = policy
        self.env_factory = env_factory
        self.seed = int(seed)
        self.lr = lr
        self.gamma = gamma
        self.baseline_decay = baseline_decay
        #: 每个 agent 的回报指数滑动基线（首次出现时用该 agent 的回报均值初始化）
        self._baselines: dict[int, float] = {}
        self._baseline_samples: dict[int, int] = {}
        self._updates = 0
        self.history: list[dict[str, float]] = []
        self.eval_history: list[dict[str, object]] = []
        self.last_eval_details: list[dict[str, float]] = []

    # -- 环境 / 转移读取 ----------------------------------------------
    def _attach_policy(self, env) -> bool:
        """把当前策略挂到 env.population（与高层策略同一套约定）。"""
        population = getattr(env, "population", None)
        candidates = (
            (population, "attach_lowlevel_policy"),
            (env, "attach_lowlevel_policy"),
            (env, "attach_policy"),
        )
        for owner, name in candidates:
            attach = getattr(owner, name, None)
            if not callable(attach):
                continue
            try:
                attach(self.policy)
                return True
            except Exception:
                continue
        return False

    def _make_env(self, seed: int, steps: int, dt: float):
        """构造训练环境：优先带关键字参数，签名不支持时回退只传 seed。"""
        try:
            env = self.env_factory(int(seed), steps=int(steps), dt=float(dt))
        except TypeError:
            env = self.env_factory(int(seed))
        self._attach_policy(env)
        return env

    @staticmethod
    def _ended(env) -> bool:
        ended = getattr(env, "ended", None)
        if not callable(ended):
            return False
        try:
            return bool(ended())
        except Exception:
            return False

    def _feature_dim(self) -> int:
        dimension = getattr(self.policy, "feature_dim", None)
        if dimension is not None:
            return int(dimension)
        neural_policy = getattr(self.policy, "neural_policy", None)
        w1 = getattr(neural_policy, "w1", None)
        if w1 is None:
            raise TypeError("无法确定策略特征维度（policy 缺少 feature_dim/neural_policy）")
        return int(np.asarray(w1).shape[0])

    @staticmethod
    def _normalize_action(action) -> ActionSample:
        """把集成层传来的 action 规范化为 :class:`ActionSample`。"""
        if isinstance(action, ActionSample):
            return action
        if isinstance(action, dict):
            return ActionSample(
                mode=int(action.get("mode", 0)),
                move=int(action.get("move", 0)),
                jump=bool(action.get("jump", False)),
                speak=bool(action.get("speak", False)),
                logprob=float(action.get("logprob", 0.0)),
            )
        if isinstance(action, (tuple, list)) and len(action) >= 4:
            return ActionSample(
                mode=int(action[0]),
                move=int(action[1]),
                jump=bool(action[2]),
                speak=bool(action[3]),
                logprob=float(action[4]) if len(action) > 4 else 0.0,
            )
        raise TypeError("action 必须是 ActionSample / dict / tuple")

    def _drain(self, env) -> list[tuple[int, np.ndarray, ActionSample, float]]:
        """从 ``env.population.drain_lowlevel_transitions()`` 取走并规范化转移。"""
        population = getattr(env, "population", None)
        drain = getattr(population, "drain_lowlevel_transitions", None)
        if not callable(drain):
            return []
        try:
            raw = drain() or []
        except Exception:
            return []
        dimension = self._feature_dim()
        transitions: list[tuple[int, np.ndarray, ActionSample, float]] = []
        for item in raw:
            try:
                agent_id, features, action, reward = item
                agent_id = int(agent_id)
                vector = np.asarray(features, dtype=np.float64).ravel()
                action_sample = self._normalize_action(action)
                reward_value = float(reward)
            except (TypeError, ValueError, IndexError):
                continue
            if vector.shape != (dimension,):
                continue
            if not np.all(np.isfinite(vector)) or not math.isfinite(reward_value):
                continue
            if not (0 <= action_sample.mode < len(MODE_NAMES)):
                continue
            if not (0 <= action_sample.move < MOVE_DIM):
                continue
            transitions.append((agent_id, vector, action_sample, reward_value))
        return transitions

    @staticmethod
    def _env_metrics(env, metrics: dict[str, float]) -> None:
        """把 ``env.result()`` 里的数值型标量并入回合指标（不覆盖已有键）。"""
        result_fn = getattr(env, "result", None)
        if not callable(result_fn):
            return
        try:
            result = result_fn()
        except Exception:
            return
        if not isinstance(result, dict):
            return
        for key, value in result.items():
            name = str(key)
            if name in metrics:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                metrics[name] = number

    # -- 折扣回报 ------------------------------------------------------
    @staticmethod
    def discounted_returns(rewards, gamma: float) -> np.ndarray:
        """按时间顺序计算折扣回报 ``G_t = r_t + γ r_{t+1} + …``。"""
        values = [float(reward) for reward in rewards]
        returns = np.zeros(len(values), dtype=np.float64)
        running = 0.0
        for index in range(len(values) - 1, -1, -1):
            running = values[index] + float(gamma) * running
            returns[index] = running
        return returns

    def _baseline_for(self, agent_id: int) -> float:
        return float(self._baselines.get(agent_id, 0.0))

    def _update_baseline(self, agent_id: int, returns: np.ndarray) -> None:
        """用该 agent 本回合平均回报更新指数滑动基线。"""
        target = float(np.mean(returns)) if returns.size else 0.0
        if self._baseline_samples.get(agent_id, 0) == 0:
            self._baselines[agent_id] = target
        else:
            self._baselines[agent_id] = (
                self.baseline_decay * self._baselines.get(agent_id, 0.0)
                + (1.0 - self.baseline_decay) * target
            )
        self._baseline_samples[agent_id] = self._baseline_samples.get(agent_id, 0) + 1

    # -- 单回合：采样 + 更新 -------------------------------------------
    def run_episode(self, *, seed: int, steps: int, dt: float = 0.05) -> dict[str, float]:
        """跑一个训练回合（采样，并用该回合转移做一次策略梯度更新）。

        返回真实指标：``return``（奖励之和）、``steps``（实际世界步数）、
        ``transitions``（低层样本数）、``agents``（有转移的 agent 数）、
        ``mean_advantage``、``baseline``，以及 ``env.result()`` 的数值快照。
        """
        budget = int(steps)
        if budget <= 0:
            raise ValueError("steps 必须是正整数")
        env = self._make_env(seed, budget, dt)
        transitions: list[tuple[int, np.ndarray, ActionSample, float]] = []
        step_count = 0
        for _ in range(budget):
            step_fn = getattr(env, "step", None)
            if not callable(step_fn):
                raise TypeError("env 缺少 step() 方法")
            step_fn()
            step_count += 1
            transitions.extend(self._drain(env))
            if self._ended(env):
                break
        # 最后一帧之后可能才结算：再取一次
        transitions.extend(self._drain(env))

        total_reward = float(sum(item[3] for item in transitions))
        mean_advantage = 0.0
        agent_count = 0.0
        if transitions:
            grouped: dict[int, list[tuple[np.ndarray, ActionSample, float]]] = {}
            for agent_id, features, action, reward in transitions:
                grouped.setdefault(agent_id, []).append((features, action, reward))
            agent_count = float(len(grouped))
            grad_sum = {
                "w1": np.zeros_like(self.policy.neural_policy.w1),
                "b1": np.zeros_like(self.policy.neural_policy.b1),
                "w2": np.zeros_like(self.policy.neural_policy.w2),
                "b2": np.zeros_like(self.policy.neural_policy.b2),
                "w3": np.zeros_like(self.policy.neural_policy.w3),
                "b3": np.zeros_like(self.policy.neural_policy.b3),
            }
            advantage_values: list[float] = []
            for agent_id in sorted(grouped):
                items = grouped[agent_id]
                rewards = [reward for _, _, reward in items]
                returns = self.discounted_returns(rewards, self.gamma)
                baseline = self._baseline_for(agent_id)
                advantages = returns - baseline
                advantages = np.clip(advantages, -50.0, 50.0)
                advantage_values.extend(float(value) for value in advantages)
                for (features, action, _reward), advantage in zip(items, advantages):
                    _loss, grads = self.policy.logprob_and_grads(
                        features, action, float(advantage)
                    )
                    for name in grad_sum:
                        grad_sum[name] += grads[name]
                self._update_baseline(agent_id, returns)
            mean_advantage = (
                float(np.mean(advantage_values)) if advantage_values else 0.0
            )
            scale = 1.0 / float(len(transitions))
            averaged = {name: value * scale for name, value in grad_sum.items()}
            self.policy.apply_grads(averaged, self.lr)
            self._updates += 1

        baseline_mean = (
            float(np.mean([self._baselines[agent] for agent in self._baselines]))
            if self._baselines
            else 0.0
        )
        metrics: dict[str, float] = {
            "return": total_reward,
            "steps": float(step_count),
            "transitions": float(len(transitions)),
            "agents": agent_count,
            "mean_advantage": mean_advantage,
            "baseline": baseline_mean,
        }
        self._env_metrics(env, metrics)
        return metrics

    # -- 评估（不更新参数）--------------------------------------------
    def evaluate(self, *, seeds, steps: int, dt: float = 0.05) -> dict[str, float]:
        """在给定种子上评估当前策略（不更新参数）。"""
        seed_list = [int(seed) for seed in seeds]
        if not seed_list:
            raise ValueError("seeds 不能为空")
        budget = int(steps)
        if budget <= 0:
            raise ValueError("steps 必须是正整数")
        runs: list[dict[str, float]] = []
        for seed in seed_list:
            env = self._make_env(seed, budget, dt)
            step_count = 0
            for _ in range(budget):
                step_fn = getattr(env, "step", None)
                if not callable(step_fn):
                    raise TypeError("env 缺少 step() 方法")
                step_fn()
                step_count += 1
                if self._ended(env):
                    break
            transitions = self._drain(env)
            metrics: dict[str, float] = {
                "seed": float(seed),
                "return": float(sum(item[3] for item in transitions)),
                "steps": float(step_count),
                "transitions": float(len(transitions)),
                "agents": float(len({item[0] for item in transitions})),
            }
            self._env_metrics(env, metrics)
            runs.append(metrics)
        self.last_eval_details = runs
        returns = np.asarray([run["return"] for run in runs], dtype=np.float64)
        aggregate: dict[str, float] = {
            "episodes": float(len(runs)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "mean_steps": float(np.mean([run["steps"] for run in runs])),
            "total_transitions": float(sum(run["transitions"] for run in runs)),
        }
        common_keys = set(runs[0]) - {"seed", "return", "steps", "transitions", "agents"}
        for run in runs[1:]:
            common_keys &= set(run)
        for key in sorted(common_keys):
            aggregate[f"mean_{key}"] = float(np.mean([run[key] for run in runs]))
        return aggregate

    # -- 完整训练循环 --------------------------------------------------
    def train(
        self,
        *,
        episodes: int,
        steps: int,
        dt: float = 0.05,
        eval_seeds: list[int] | None = None,
        eval_steps: int | None = None,
        eval_every: int | None = None,
        out_path=None,
    ) -> dict[str, object]:
        """真实训练循环：采样 → 真实奖励 → 折扣回报/基线 → 策略梯度更新 → 评估。

        ``eval_seeds`` 为 ``None`` 时自动生成两个与训练种子不重叠的评估种子
        （``seed+100000+7919*k``）；给出时用调用方列表（必须与训练种子不重叠）。
        """
        episode_count = int(episodes)
        if episode_count <= 0:
            raise ValueError("episodes 必须是正整数")
        step_budget = int(steps)
        if step_budget <= 0:
            raise ValueError("steps 必须是正整数")
        eval_budget = step_budget if eval_steps is None else int(eval_steps)
        if eval_seeds is None:
            eval_seed_list = [
                self.seed + 100_000 + 7919 * index for index in range(2)
            ]
        else:
            eval_seed_list = [int(seed) for seed in eval_seeds]
        if eval_seed_list and eval_budget <= 0:
            raise ValueError("eval_steps 必须是正整数")
        every = max(1, episode_count // 4) if eval_every is None else int(eval_every)
        if every <= 0:
            raise ValueError("eval_every 必须是正整数")
        self.history = []
        self.eval_history = []
        for episode in range(episode_count):
            train_seed = self.seed + 101 * episode
            metrics = self.run_episode(seed=train_seed, steps=step_budget, dt=float(dt))
            metrics["episode"] = float(episode + 1)
            self.history.append(metrics)
            if eval_seed_list and (
                (episode + 1) % every == 0 or (episode + 1) == episode_count
            ):
                evaluation = self.evaluate(
                    seeds=eval_seed_list, steps=eval_budget, dt=float(dt)
                )
                self.eval_history.append(
                    {"episode": float(episode + 1), "eval": evaluation}
                )
        final_eval: dict[str, float] = {}
        if eval_seed_list:
            final_eval = self.evaluate(
                seeds=eval_seed_list, steps=eval_budget, dt=float(dt)
            )
        saved = None
        if out_path is not None:
            saved = str(self.policy.save(out_path))
        return {
            "config": {
                "episodes": float(episode_count),
                "steps": float(step_budget),
                "dt": float(dt),
                "eval_seeds": [float(seed) for seed in eval_seed_list],
                "eval_steps": float(eval_budget),
                "eval_every": float(every),
                "lr": float(self.lr),
                "gamma": float(self.gamma),
                "baseline_decay": float(self.baseline_decay),
                "seed": float(self.seed),
            },
            "train": [dict(record) for record in self.history],
            "train_returns": [float(record["return"]) for record in self.history],
            "train_transitions": [float(record["transitions"]) for record in self.history],
            "eval": [dict(record) for record in self.eval_history],
            "final_eval": dict(final_eval),
            "updates": float(self._updates),
            "baselines": {str(key): float(value) for key, value in sorted(self._baselines.items())},
            "out": saved,
        }


__all__ = [
    "ActionSample",
    "LowLevelPolicy",
    "LowLevelTrainer",
    "MODE_NAMES",
    "MOVE_VALUES",
    "MODE_SLICE",
    "MOVE_SLICE",
    "JUMP_INDEX",
    "SPEAK_INDEX",
    "LANG_START",
    "EXPECTED_OUTPUT_DIM",
]
