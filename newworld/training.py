"""Training for the neural behaviour + proto-language policy.

Two modes:

* ``pretrain`` – collect a dataset by watching the hand-written homeostatic
  behaviour, then fit the MLP by supervised learning (behaviour imitation).
  The language head learns one distinct 3-syllable word per internal state.
* ``fine_tune`` – **continue** from an existing checkpoint on a fresh dataset.
  This is what F2 triggers; it never resets the weights, so learning is
  continuous across the session and across runs (the checkpoint is overwritten).

Only the training helpers are here; the GUI never trains unless F2 is pressed.
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
START_HOURS = (8.0, 11.0, 14.0, 18.0, 21.0, 2.0, 5.0)


# ---------------------------------------------------------------------------
# dataset collection (teacher = homeostatic behaviour)
# ---------------------------------------------------------------------------
def collect_dataset(
    seed: int = 0,
    steps_per_hour: int = 300,
    hours: tuple[float, ...] = START_HOURS,
) -> dict[str, np.ndarray]:
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
        world = World(config, seed=seed)
        world.reset(seed=seed + int(hour))
        world.t = ((hour - world.day.start_hour) % 24.0) / 24.0 * world.day.day_length_s
        world._update_environment()
        teacher = InnerLife(seed + int(hour))
        for _ in range(steps_per_hour):
            decision = teacher.decide(world, dt)
            mode = world.agent.behavior_state
            if mode not in MODE_INDEX:
                mode = "explore"
            xs.append(features(world))
            modes.append(MODE_INDEX[mode])
            moves.append((1, 0, -1).index(decision.move) if decision.move in (-1, 0, 1) else 1)
            jumps.append(bool(decision.jump))
            speaks.append(bool(decision.speak))
            target_word = MODE_WORDS[mode]
            words.append(np.asarray([VOCAB.index(token) for token in target_word], dtype=np.int64))
            world.step(actions=decision, dt=dt)
            if decision.speak:
                world.speak(world.agent, decision.word)
    return {
        "x": np.asarray(xs, dtype=np.float64),
        "mode": np.asarray(modes, dtype=np.int64),
        "move": np.asarray(moves, dtype=np.int64),
        "jump": np.asarray(jumps, dtype=np.float64),
        "speak": np.asarray(speaks, dtype=np.float64),
        "word": np.asarray(words, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# supervised training (manual backprop, numpy only)
# ---------------------------------------------------------------------------
def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
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
    """Fit the MLP on a teacher dataset (continues from current weights)."""
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
            # mode
            mode_prob = _softmax(logits[:, mode_slice])
            onehot = np.eye(len(MODES))[mode_targets[index]]
            delta[:, mode_slice] = (mode_prob - onehot) / batch
            # move
            move_prob = _softmax(logits[:, move_slice])
            onehot = np.eye(MOVE_DIM)[move_targets[index]]
            delta[:, move_slice] = (move_prob - onehot) / batch
            # jump / speak (BCE gradients)
            aux = _sigmoid(logits[:, aux_start : aux_start + AUX_DIM])
            target_aux = np.stack([jump_targets[index], speak_targets[index]], axis=1)
            delta[:, aux_start : aux_start + AUX_DIM] = 0.5 * (aux - target_aux) / batch
            # language (3 classifiers of 16 syllables)
            for token in range(LANG_TOKENS):
                begin = lang_start + token * len(VOCAB)
                block = logits[:, begin : begin + len(VOCAB)]
                prob = _softmax(block)
                onehot = np.eye(len(VOCAB))[word_targets[index, token]]
                delta[:, begin : begin + len(VOCAB)] = 0.7 * (prob - onehot) / batch

            # backprop
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

            # report loss for progress
            mode_loss = -np.log(np.clip(mode_prob[np.arange(batch), mode_targets[index]], 1e-9, 1.0)).mean()
            epoch_loss += float(mode_loss)
            batches += 1
        last_loss = epoch_loss / max(1, batches)

    # final accuracy on the dataset (a cheap training signal)
    logits, _, _ = policy.forward(x)
    predicted = np.argmax(logits[:, mode_slice], axis=1)
    accuracy = float(np.mean(predicted == mode_targets))
    return {"mode_accuracy": accuracy, "loss": last_loss}


# ---------------------------------------------------------------------------
# public training entry points
# ---------------------------------------------------------------------------
def pretrain(
    out: str | Path = DEFAULT_POLICY,
    *,
    seed: int = 0,
    steps_per_hour: int = 300,
    epochs: int = 24,
) -> tuple[NeuralPolicy, dict[str, float]]:
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
    """Continue training from the checkpoint; only called when the user presses F2."""
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
    """Short neural rollout for status/testing: distances, words, actions."""
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
