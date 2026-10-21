"""统一实验运行器（P5）：配置 -> 世界 -> 协议 -> 产物。

一切实验都走这一条代码路径：

    settings = load_config("configs/xxx.yaml")
    summary = run_experiment(settings)

协议（protocol）由 :data:`newworld.registry.PROTOCOLS` 注册，配置里写
名字即可。本模块负责：

1. 按配置构造 world/swarm/social/comm/memory/policy（组件全部可插拔）；
2. 提供逐步 step 与整段 run 的统一循环（实验室后端也复用它）；
3. 统一指标口径（人口/进食/饱足/基尼/奖励/社会/通信/记忆）；
4. 把 config.yaml + summary.json 写进 output_dir/name/，记录配置 hash。

实验协议只关心"怎么设计对照/扫参"，不再重复世界搭建代码。
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .settings import (
    PROJECT_ROOT,
    ExperimentSettings,
    apply_overrides,
    load_config,
    save_config,
)

#: 协议产物统一文件名
SUMMARY_FILE = "summary.json"
CONFIG_FILE = "config.yaml"


# ---------------------------------------------------------------------------
# 指标辅助
# ---------------------------------------------------------------------------
def gini(values: np.ndarray | list[float]) -> float:
    """基尼系数（0=完全平均，1=一人独占）；空/全零输入返回 0。"""
    array = np.sort(np.asarray(values, dtype=np.float64))
    n = int(array.size)
    total = float(array.sum())
    if n == 0 or total <= 1e-12:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * array)) / (n * total) - (n + 1.0) / n)


# ---------------------------------------------------------------------------
# 仿真组合
# ---------------------------------------------------------------------------
@dataclass
class Simulation:
    """一个可推进的世界组合（engine + swarm + 可选机制层 + policy）。"""

    settings: ExperimentSettings
    engine: Any
    swarm: Any
    policy: Any
    social: Any = None
    comm: Any = None
    memory: Any = None
    seed: int = 0

    # -- 推进 -----------------------------------------------------------
    def step(self, dt: float | None = None) -> dict[str, float]:
        """推进一个物理步（顺序：环境 -> 个体 -> 社会 -> 通信 -> 记忆）。"""
        step_dt = float(self.settings.run.dt if dt is None else dt)
        daylight = float(self.settings.run.daylight)
        deaths_before = len(self.swarm.death_ages)
        self.engine.update(step_dt, daylight)
        self.swarm.update(step_dt)
        if self.social is not None:
            self.social.on_death_pass(deaths_before)
            self.social.update(step_dt)
        if self.comm is not None:
            self.comm.update(step_dt)
        if self.memory is not None:
            self.memory.update(step_dt)
        return {
            "population": float(self.swarm.population),
            "births": float(self.swarm.births),
            "deaths": float(self.swarm.deaths),
        }

    def run(self, seconds: float | None = None) -> dict[str, Any]:
        """连续推进 seconds 世界秒，返回指标摘要。"""
        duration = float(self.settings.run.seconds if seconds is None else seconds)
        steps = max(1, int(round(duration / float(self.settings.run.dt))))
        started = time.perf_counter()
        for _ in range(steps):
            self.step()
        wall = time.perf_counter() - started
        metrics = self.metrics()
        metrics["wall_s"] = round(wall, 4)
        metrics["steps_per_s"] = round(steps / wall, 1) if wall > 0 else 0.0
        return metrics

    # -- 指标 -----------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        swarm = self.swarm
        view = swarm.data[: swarm.count]
        alive = view["alive"] if swarm.count else np.zeros(0, dtype=bool)
        hungers = view["hunger"][alive] if swarm.count else np.zeros(0)
        payload: dict[str, Any] = {
            "t": round(float(self.engine.t), 3),
            "steps": int(self.engine.steps),
            "population": int(swarm.population),
            "births": int(swarm.births),
            "deaths": int(swarm.deaths),
            "starvation_deaths": int(swarm.starvation_deaths),
            "old_age_deaths": int(swarm.old_age_deaths),
            "drowned": int(swarm.drowned),
            "eat_events": int(swarm.eat_events),
            "satiation_mean": round(float(hungers.mean()), 4) if hungers.size else 0.0,
            "satiation_std": round(float(hungers.std()), 4) if hungers.size else 0.0,
            "gini_hunger": round(gini(hungers), 4),
            "total_reward": round(float(swarm.total_reward), 3),
            "food_total": round(float(self.engine.food_capacity()), 1),
            "seed": int(self.seed),
        }
        if self.social is not None:
            payload["social"] = {key: _plain(value) for key, value in self.social.stats.items()}
        if self.comm is not None:
            payload["comm"] = {key: _plain(value) for key, value in dict(self.comm.stats).items()}
        if self.memory is not None:
            payload["memory"] = {key: _plain(value) for key, value in dict(self.memory.stats).items()}
        return payload


def _plain(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# ---------------------------------------------------------------------------
# 构造
# ---------------------------------------------------------------------------
def _build_social(settings: ExperimentSettings, swarm, seed: int):
    if not settings.social.enabled:
        return None
    from .social_mech import SocialLayer

    layer = SocialLayer(swarm, settings.social.to_social_config(), seed=seed)
    layer.inject_initial_skills()
    layer.attach_to_swarm()
    return layer


def _build_comm(settings: ExperimentSettings, swarm, seed: int):
    if not settings.comm.enabled:
        return None
    from .communication import build_communication_layer

    layer = build_communication_layer(swarm, settings.comm, seed=seed)
    layer.attach_to_swarm()
    return layer


def _build_memory(settings: ExperimentSettings, swarm, seed: int):
    if not settings.memory.enabled:
        return None
    from .agent_memory import build_memory_layer

    layer = build_memory_layer(swarm, settings.memory, seed=seed)
    layer.attach_to_swarm()
    return layer


def build_simulation(
    settings: ExperimentSettings,
    *,
    seed: int | None = None,
    policy: Any | None = None,
) -> Simulation:
    """按配置构造一个仿真组合。

    seed 缺省用 settings.world.seed；policy 缺省按 settings.policy 构造。
    观测扩展（comm/memory）必须在构造 policy 之前注册，这样 policy 的
    obs_dim 能自动匹配。
    """
    from .map_engine import MapEngine
    from .swarm import Swarm
    from .swarm_policy import make_swarm_policy

    run_seed = int(settings.world.seed if seed is None else seed)
    world = dataclasses.replace(settings.world, seed=run_seed)
    engine = MapEngine(world.to_map_config())
    swarm = Swarm(engine, settings.swarm.to_swarm_config(), seed=run_seed)
    social = _build_social(settings, swarm, run_seed)
    comm = _build_comm(settings, swarm, run_seed)
    memory = _build_memory(settings, swarm, run_seed)
    if policy is None:
        policy = make_swarm_policy(
            kind=settings.policy.kind,
            obs_dim=swarm.obs_dim,
            seed=run_seed if settings.policy.seed == 0 else settings.policy.seed,
            hidden=settings.policy.hidden,
            checkpoint=settings.policy.checkpoint,
        )
    swarm.attach_policy(policy)
    return Simulation(
        settings=settings,
        engine=engine,
        swarm=swarm,
        policy=policy,
        social=social,
        comm=comm,
        memory=memory,
        seed=run_seed,
    )


# ---------------------------------------------------------------------------
# 协议执行与产物
# ---------------------------------------------------------------------------
def _import_protocols() -> None:
    """import 协议包以触发注册（幂等）。"""
    from . import protocols  # noqa: F401


def output_dir(settings: ExperimentSettings, override: str | Path | None = None) -> Path:
    root = Path(settings.run.output_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if override is not None:
        root = Path(override)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
    return root / settings.name


def write_artifacts(
    settings: ExperimentSettings,
    summary: dict[str, Any],
    *,
    directory: Path | None = None,
) -> Path:
    """把 config + summary 写进实验目录，返回目录。"""
    target = directory or output_dir(settings)
    target.mkdir(parents=True, exist_ok=True)
    save_config(settings, target / CONFIG_FILE)
    payload = dict(summary)
    payload.setdefault("name", settings.name)
    payload.setdefault("protocol", settings.protocol)
    payload.setdefault("config_hash", settings.hash())
    (target / SUMMARY_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def run_experiment(
    settings: ExperimentSettings,
    *,
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """执行配置指定的协议，写产物，返回 summary。"""
    from .registry import PROTOCOLS

    _import_protocols()
    try:
        factory: Callable[[ExperimentSettings], dict[str, Any]] = PROTOCOLS.get(settings.protocol)
    except Exception as exc:
        raise ValueError(f"未知实验协议 {settings.protocol!r}: {exc}") from exc
    summary = factory(settings)
    if not isinstance(summary, dict):
        raise TypeError(f"协议 {settings.protocol} 必须返回 dict，得到 {type(summary)}")
    target = Path(directory) if directory else output_dir(settings)
    result = dict(summary)
    result["_artifact_dir"] = str(target)
    write_artifacts(settings, result, directory=target)
    return result


def run_from_file(
    config_path: str | Path,
    *,
    overrides: list[str] | tuple[str, ...] = (),
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """实验 CLI 的入口：载入 YAML + 应用 --set 覆盖 + 执行。"""
    settings = apply_overrides(load_config(config_path), list(overrides))
    return run_experiment(settings, directory=directory)


__all__ = [
    "Simulation",
    "build_simulation",
    "gini",
    "run_experiment",
    "run_from_file",
    "write_artifacts",
    "output_dir",
    "SUMMARY_FILE",
    "CONFIG_FILE",
]
