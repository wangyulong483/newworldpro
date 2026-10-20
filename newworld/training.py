"""神经网络行为 + 原语言策略的训练（行为克隆）。

两种模式：

* ``pretrain`` —— 先观察手写的稳态行为收集数据集，再用监督学习拟合
  MLP（行为模仿）。语言头为每个内在状态学一个专属的 3 音节词。
* ``fine_tune`` —— 从已有 checkpoint 出发在新数据集上**继续**训练。
  这就是 F2 触发的操作；它从不重置权重，因此学习跨会话、跨运行
  连续（checkpoint 会被覆盖）。

这里只有训练辅助；GUI 在按 F2 之前从不训练。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .behavior import InnerLife
from .neural import (
    AUX_DIM,
    FEATURE_DIM,
    LANG_TOKENS,
    MODE_WORDS,
    MODES,
    MOVE_DIM,
    OUTPUT_DIM,
    VOCAB,
    NeuralPolicy,
    features,
)
from .world import World

DEFAULT_POLICY = Path(__file__).resolve().parent.parent / "assets" / "neural_policy.npz"
MODE_INDEX = {mode: index for index, mode in enumerate(MODES)}
#: 数据采集的起始时刻（覆盖一天的不同时段，包括夜间）
START_HOURS = (8.0, 11.0, 14.0, 18.0, 21.0, 2.0, 5.0)


# ---------------------------------------------------------------------------
# 数据集采集（教师 = 稳态行为 / 真实多种群控制器）
# ---------------------------------------------------------------------------
def collect_dataset(
    seed: int = 0,
    steps_per_hour: int = 300,
    hours: tuple[float, ...] = START_HOURS,
    *,
    use_population: bool = True,
) -> dict[str, np.ndarray]:
    """采集行为克隆数据集。

    ``use_population=True``（默认，本轮升级固定）让数据采集跑在
    真实的 :class:`~newworld.population.Population` 里，因此 16 个
    文明上下文输入（记忆/目标计划/文明）在训练数据中有**真实的
    非零方差**。以前数据集在裸 ``World`` + 稳态教师里采集，16 个
    上下文列恰好全为 0——即那些输入永远学不到（审计：16/16 列
    标准差为零）。标签仍然是控制器实际执行的动作，因此这依然是
    行为克隆，不是 RL。
    """
    from .config import make_config

    config = make_config(800, 500)
    dt = config.fixed_dt
    xs: list[np.ndarray] = []
    modes: list[int] = []
    moves: list[int] = []
    jumps: list[bool] = []
    speaks: list[bool] = []
    words: list[np.ndarray] = []

    for hour in hours:
        if use_population:
            frames = _collect_hour_population(seed, hour, steps_per_hour, config, dt)
        else:  # 旧稳态教师路径（保留用于对比）
            frames = _collect_hour_teacher(seed, hour, steps_per_hour, config, dt)
        for frame in frames:
            xs.append(frame[0])
            modes.append(frame[1])
            moves.append(frame[2])
            jumps.append(frame[3])
            speaks.append(frame[4])
            words.append(frame[5])
    return {
        "x": np.asarray(xs, dtype=np.float64),
        "mode": np.asarray(modes, dtype=np.int64),
        "move": np.asarray(moves, dtype=np.int64),
        "jump": np.asarray(jumps, dtype=np.float64),
        "speak": np.asarray(speaks, dtype=np.float64),
        "word": np.asarray(words, dtype=np.int64),
    }


def _frame(agent, decision) -> tuple:
    """一条 (features, mode, move, jump, speak, word) 训练样本。"""
    mode = getattr(agent, "behavior_state", "explore")
    if mode not in MODE_INDEX:
        mode = "explore"
    move = 0
    jump = False
    speak = False
    if decision is not None:
        move = (1, 0, -1).index(decision.move) if decision.move in (-1, 0, 1) else 1
        jump = bool(decision.jump)
        speak = bool(decision.speak)
    # 语言标签 = 该行为状态的目标词（每个状态一个专属词）
    word = np.asarray([VOCAB.index(token) for token in MODE_WORDS[mode]], dtype=np.int64)
    return (None, MODE_INDEX[mode], move, jump, speak, word)


def _collect_hour_population(seed: int, hour: float, steps: int, config, dt: float) -> list[tuple]:
    """从真实的多种群控制器采集帧（文明上下文非零）。"""
    from .config import CivilizationConfig
    from .population import Population

    civ = CivilizationConfig(
        seed=seed,
        enable_perception=False,
        enable_visual_features=True,   # V2.1：视觉特征进入训练数据
        perception_stride=4,
    )
    world = World(config, seed=seed, civ_config=civ)
    world.reset(seed=seed + int(hour))
    # 把时钟拨到目标时刻
    world.t = ((hour - world.day.start_hour) % 24.0) / 24.0 * world.day.day_length_s
    world._update_environment()
    population = Population(world, seed=seed)
    frames: list[tuple] = []
    for _ in range(steps):
        population.step(dt)
        if not world.agents:
            break
        agent = world.agents[0]
        mind = population.minds.get(agent.agent_id)
        decision = mind.last_decision if mind is not None else None
        frame = _frame(agent, decision)
        frames.append((features(world, character=agent),) + frame[1:])
    return frames


def _collect_hour_teacher(seed: int, hour: float, steps: int, config, dt: float) -> list[tuple]:
    """旧路径：裸世界 + 手写稳态教师。"""
    world = World(config, seed=seed)
    world.reset(seed=seed + int(hour))
    world.t = ((hour - world.day.start_hour) % 24.0) / 24.0 * world.day.day_length_s
    world._update_environment()
    teacher = InnerLife(seed + int(hour))
    frames: list[tuple] = []
    for _ in range(steps):
        decision = teacher.decide(world, dt)
        frame = _frame(world.agent, decision)
        frames.append((features(world),) + frame[1:])
        world.step(actions=decision, dt=dt)
        if decision.speak:
            world.speak(world.agent, decision.word)
    return frames


# ---------------------------------------------------------------------------
# 监督训练（手写反向传播，仅 numpy）
# ---------------------------------------------------------------------------
def _softmax(logits: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax。"""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。"""
    return 1.0 / (1.0 + np.exp(-values))


def train_supervised(
    policy: NeuralPolicy,
    data: dict[str, np.ndarray],
    *,
    epochs: int = 24,
    learning_rate: float = 0.03,
    batch_size: int = 128,
    seed: int = 0,
) -> dict[str, float]:
    """在教师数据集上拟合 MLP（从当前权重继续）。

    多任务联合损失：模式/移动/语言用 softmax 交叉熵，跳跃/说话用
    BCE 式梯度；优化器为带动量（0.9）的小批量 SGD。
    """
    rng = np.random.default_rng(seed)
    x = data["x"]
    mode_targets = data["mode"]
    move_targets = data["move"]
    jump_targets = data["jump"]
    speak_targets = data["speak"]
    word_targets = data["word"]
    count = len(x)
    if count == 0:
        return {"mode_accuracy": 0.0, "loss": float("nan")}

    mode_slice = slice(0, len(MODES))
    move_slice = slice(len(MODES), len(MODES) + MOVE_DIM)
    aux_start = len(MODES) + MOVE_DIM
    lang_start = aux_start + AUX_DIM

    # 动量法的速度缓存
    velocity = {
        name: np.zeros_like(getattr(policy, name))
        for name in ("w1", "b1", "w2", "b2", "w3", "b3")
    }
    last_loss = float("nan")
    for epoch in range(epochs):
        order = rng.permutation(count)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, count, batch_size):
            index = order[start : start + batch_size]
            xb = x[index]
            h1 = np.tanh(xb @ policy.w1 + policy.b1)
            h2 = np.tanh(h1 @ policy.w2 + policy.b2)
            logits = h2 @ policy.w3 + policy.b3
            batch = len(index)

            delta = np.zeros_like(logits)
            # 模式：softmax 交叉熵梯度
            mode_prob = _softmax(logits[:, mode_slice])
            onehot = np.eye(len(MODES))[mode_targets[index]]
            delta[:, mode_slice] = (mode_prob - onehot) / batch
            # 移动方向
            move_prob = _softmax(logits[:, move_slice])
            onehot = np.eye(MOVE_DIM)[move_targets[index]]
            delta[:, move_slice] = (move_prob - onehot) / batch
            # 跳跃 / 说话（BCE 梯度）
            aux = _sigmoid(logits[:, aux_start : aux_start + AUX_DIM])
            target_aux = np.stack([jump_targets[index], speak_targets[index]], axis=1)
            delta[:, aux_start : aux_start + AUX_DIM] = 0.5 * (aux - target_aux) / batch
            # 语言（3 个 16 音节分类器）
            for token in range(LANG_TOKENS):
                begin = lang_start + token * len(VOCAB)
                block = logits[:, begin : begin + len(VOCAB)]
                prob = _softmax(block)
                onehot = np.eye(len(VOCAB))[word_targets[index, token]]
                delta[:, begin : begin + len(VOCAB)] = 0.7 * (prob - onehot) / batch

            # 反向传播（手写）
            grad_w3 = h2.T @ delta
            grad_b3 = delta.sum(axis=0)
            dh2 = (delta @ policy.w3.T) * (1.0 - h2 * h2)
            grad_w2 = h1.T @ dh2
            grad_b2 = dh2.sum(axis=0)
            dh1 = (dh2 @ policy.w2.T) * (1.0 - h1 * h1)
            grad_w1 = xb.T @ dh1
            grad_b1 = dh1.sum(axis=0)

            momentum = 0.9
            for name, grad in (
                ("w1", grad_w1),
                ("b1", grad_b1),
                ("w2", grad_w2),
                ("b2", grad_b2),
                ("w3", grad_w3),
                ("b3", grad_b3),
            ):
                velocity[name] = momentum * velocity[name] - learning_rate * grad
                setattr(policy, name, getattr(policy, name) + velocity[name])

            # 汇报模式损失以观察进度
            mode_loss = -np.log(np.clip(mode_prob[np.arange(batch), mode_targets[index]], 1e-9, 1.0)).mean()
            epoch_loss += float(mode_loss)
            batches += 1
        last_loss = epoch_loss / max(1, batches)

    # 在数据集上做最终精度评估（便宜的训练信号）
    logits, _, _ = policy.forward(x)
    predicted = np.argmax(logits[:, mode_slice], axis=1)
    accuracy = float(np.mean(predicted == mode_targets))
    return {"mode_accuracy": accuracy, "loss": last_loss}


# ---------------------------------------------------------------------------
# 公开训练入口
# ---------------------------------------------------------------------------
def pretrain(
    out: str | Path = DEFAULT_POLICY,
    *,
    seed: int = 0,
    steps_per_hour: int = 300,
    epochs: int = 24,
) -> tuple[NeuralPolicy, dict[str, float]]:
    """从零开始：采数据 + 训练 + 保存。"""
    data = collect_dataset(seed=seed, steps_per_hour=steps_per_hour)
    policy = NeuralPolicy(seed=seed)
    metrics = train_supervised(policy, data, epochs=epochs, seed=seed)
    policy.save(out)
    return policy, metrics


def fine_tune(
    path: str | Path = DEFAULT_POLICY,
    *,
    seed: int = 0,
    steps_per_hour: int = 220,
    epochs: int = 10,
) -> tuple[NeuralPolicy, dict[str, float]]:
    """从 checkpoint 继续训练；只有用户按 F2 才会调用。"""
    path = Path(path)
    if path.exists():
        try:
            policy = NeuralPolicy.load(path)
        except ValueError:
            policy = NeuralPolicy(seed=seed)
    else:
        policy = NeuralPolicy(seed=seed)
    data = collect_dataset(seed=seed, steps_per_hour=steps_per_hour)
    metrics = train_supervised(policy, data, epochs=epochs, seed=seed)
    policy.save(path)
    return policy, metrics


def evaluate_behaviour(policy: NeuralPolicy, seed: int = 0, steps: int = 360) -> dict[str, float]:
    """短程神经 rollout，用于状态/测试：距离、词、动作。"""
    from .autonomy import Mind
    from .config import make_config

    world = World(make_config(800, 500), seed=seed)
    world.reset(seed=seed)
    mind = Mind(world, seed=seed, policy=policy)
    words = set()
    for _ in range(steps):
        mind.step(1.0 / 60.0)
        if mind.will.last_output is not None:
            words.add(mind.will.last_output.word)
    return {
        "distance": world.agent.distance_walked,
        "jumps": float(world.agent.jump_count),
        "speech": float(world.agent.speech_count),
        "unique_words": float(len(words)),
        "berries": float(len(world.pixel_scene.berries)),
    }
