"""协议共用工具（P5）：训练/评估/配对统计的唯一实现。

所有实验协议（P1 学习对照、P3 社会消融、P4 演化 vs 学习、P5 标度律、
P6 通信/记忆）都复用这里的函数，避免每个脚本各写一套"造世界-跑评测"
逻辑（旧代码的问题之一就是 12 个 run_*.py 各有一份重复实现）。

约定：
* 训练/评估都通过 :func:`newworld.experiment.build_simulation` 构造；
* 每个 (seed) 实验独立造世界（配对对照要求同 seed 同地图同初始人口）；
* 指标口径统一：eat_events / eat_rate / mean_life / satiation 等。
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ..experiment import Simulation, build_simulation
from ..settings import ExperimentSettings
from ..swarm_policy import NeuralSwarmPolicy, SwarmTrainer


# ---------------------------------------------------------------------------
# 基础仿真
# ---------------------------------------------------------------------------
def simulate(
    settings: ExperimentSettings,
    *,
    seed: int,
    seconds: float | None = None,
    policy: Any | None = None,
) -> tuple[Simulation, dict[str, Any]]:
    """造世界 + 跑 seconds 秒，返回 (sim, metrics)。"""
    sim = build_simulation(settings, seed=int(seed), policy=policy)
    metrics = sim.run(seconds=seconds)
    metrics["seed"] = int(seed)
    metrics.update(_life_metrics(sim, metrics))
    return sim, metrics


def _life_metrics(sim: Simulation, metrics: dict[str, Any]) -> dict[str, float]:
    """P1.6 口径：精确平均寿命 / 90s 存活率下界 / 灭绝时刻。"""
    swarm = sim.swarm
    seconds = float(metrics.get("t") or 0.0)
    if swarm.death_ages:
        mean_life = float(np.mean(swarm.death_ages))
    else:
        view = swarm.data[: swarm.count]
        ages = view["age"][view["alive"]] if swarm.count else np.zeros(0)
        mean_life = float(ages.mean()) if ages.size else 0.0
    initial = float(max(1, sim.settings.swarm.initial_population))
    return {
        "mean_life": mean_life,
        "eat_rate": float(swarm.eat_events / max(1.0, seconds * initial)),
        "extinct_seconds": float(seconds if swarm.population == 0 else -1.0),
        "survived_90s": float(1.0 if seconds >= 90.0 and swarm.population > 0 else 0.0),
    }


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train_neural_policy(
    settings: ExperimentSettings,
    *,
    episodes: int,
    episode_seconds: float = 60.0,
    train_seeds: Iterable[int] | None = None,
    seed: int = 0,
    hidden: int = NeuralSwarmPolicy.HIDDEN,
    trainer_kwargs: dict[str, Any] | None = None,
    on_episode: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[NeuralSwarmPolicy, list[dict[str, float]]]:
    """在训练 seed 上做 REINFORCE（每回合一个 60s rollout + 一次更新）。

    与 P1.7 完全同构：每回合重建世界（同 seed 段），跑 episode_seconds，
    drain 转移，用 SwarmTrainer 更新一次。返回 (policy, history)。
    """
    episodes = int(episodes)
    if episodes < 1:
        raise ValueError("episodes >= 1")
    seeds = list(train_seeds) if train_seeds is not None else [1000 + i for i in range(6)]
    if not seeds:
        raise ValueError("train_seeds 不能为空")
    defaults = dict(
        lr=settings.policy.lr,
        gamma=settings.policy.gamma,
        batch_size=settings.policy.batch_size,
        epochs_per_episode=settings.policy.epochs_per_episode,
        normalize_advantage=settings.policy.normalize_advantage,
        grad_clip=settings.policy.grad_clip,
    )
    if trainer_kwargs:
        defaults.update(trainer_kwargs)
    policy = NeuralSwarmPolicy(
        seed=int(seed),
        obs_dim=_probe_obs_dim(settings),
        hidden=int(hidden),
    )
    trainer = SwarmTrainer(policy, **defaults)
    history: list[dict[str, float]] = []
    for episode in range(episodes):
        train_seed = int(seeds[episode % len(seeds)])
        sim = build_simulation(settings, seed=train_seed, policy=policy)
        sim.run(seconds=float(episode_seconds))
        transitions = sim.swarm.drain_swarm_transitions()
        stats = trainer.train_on_transitions(transitions)
        stats = {
            "episode": float(episode + 1),
            "train_seed": float(train_seed),
            **{key: float(value) for key, value in stats.items()},
        }
        history.append(stats)
        if on_episode is not None:
            on_episode(episode, stats)
    return policy, history


def _probe_obs_dim(settings: ExperimentSettings) -> int:
    """构造一次（不跑）仿真，拿到当前机制组合下的观测维度。"""
    sim = build_simulation(settings, seed=int(settings.world.seed))
    return int(sim.swarm.obs_dim)


# ---------------------------------------------------------------------------
# 评估与配对统计
# ---------------------------------------------------------------------------
def evaluate_policy(
    settings: ExperimentSettings,
    policy: Any,
    *,
    seed: int,
    seconds: float | None = None,
) -> dict[str, Any]:
    """在给定 seed 上评估策略（独立造世界，配对对照）。"""
    _sim, metrics = simulate(settings, seed=int(seed), seconds=seconds, policy=policy)
    return metrics


def evaluate_seeds(
    settings: ExperimentSettings,
    policy: Any,
    *,
    seeds: Iterable[int],
    seconds: float | None = None,
) -> list[dict[str, Any]]:
    return [
        evaluate_policy(settings, policy, seed=int(seed), seconds=seconds)
        for seed in seeds
    ]


def mean_std(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0.0}
    return {"mean": float(array.mean()), "std": float(array.std()), "n": float(array.size)}


def paired_summary(
    treatment: list[dict[str, Any]],
    control: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[str, float]:
    """配对差：treat-control（同 seed 一一对应），含胜场数。"""
    if len(treatment) != len(control):
        raise ValueError(f"配对样本数不等: {len(treatment)} vs {len(control)}")
    deltas = []
    for treat, base in zip(treatment, control):
        if int(treat.get("seed", -1)) != int(base.get("seed", -2)):
            raise ValueError("配对样本的 seed 不一致")
        deltas.append(float(treat[metric]) - float(base[metric]))
    array = np.asarray(deltas, dtype=np.float64)
    return {
        "metric": metric,
        "n": float(array.size),
        "mean_delta": float(array.mean()) if array.size else 0.0,
        "std_delta": float(array.std()) if array.size else 0.0,
        "wins": float((array > 0).sum()),
        "losses": float((array < 0).sum()),
        "ties": float((array == 0).sum()),
        "relative": float(array.mean() / (abs(float(np.mean([b[metric] for b in control]))) + 1e-12))
        if array.size else 0.0,
    }


def summarize_runs(runs: list[dict[str, Any]], keys: Iterable[str]) -> dict[str, dict[str, float]]:
    return {key: mean_std(run.get(key, 0.0) for run in runs) for key in keys}


# ---------------------------------------------------------------------------
# 产物
# ---------------------------------------------------------------------------
def results_dir(settings: ExperimentSettings, name: str | None = None) -> Path:
    from ..experiment import output_dir

    base = output_dir(settings)
    return base if name is None else base.parent / str(name)


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def timed() -> Callable[[], float]:
    """返回一个"距创建时刻的秒数"计时器（报告记录 wall time）。"""
    started = time.perf_counter()
    return lambda: time.perf_counter() - started


def with_world_seed(settings: ExperimentSettings, seed: int) -> ExperimentSettings:
    """返回 world/policy seed 都被替换的配置副本。"""
    return dataclasses.replace(
        settings,
        world=dataclasses.replace(settings.world, seed=int(seed)),
        policy=dataclasses.replace(settings.policy, seed=int(seed)),
    )


__all__ = [
    "simulate",
    "train_neural_policy",
    "evaluate_policy",
    "evaluate_seeds",
    "mean_std",
    "paired_summary",
    "summarize_runs",
    "results_dir",
    "write_json",
    "timed",
    "with_world_seed",
]
