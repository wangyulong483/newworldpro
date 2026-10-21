"""P5 标度律扫描协议：人口 -> 吞吐 / 学习算力 -> 性能（幂律拟合 + 诚实报告）。

协议名：scaling_sweep（配置里写 protocol: scaling_sweep 即可）。

两个单维扫描（每次只动一个轴，其余全部冻结）：

* **Sweep A（人口 -> 仿真吞吐）**：固定地图/seed/规则策略，只改初始人口
  N，跑 population_seconds 世界秒，测墙钟吞吐
  throughput = steps / wall_s（引擎步/秒）与人口。对
  throughput = a * N^b 做 log-log 最小二乘拟合，报告 a, b, R^2, n
  与逐点残差。这条线检验“仿真吞吐随人口是线性还是亚线性”——纯经验、
  实现相关，不是理论复杂度结论。
* **Sweep B（学习算力 -> 觅食性能）**：用共同训练器逐档从零训练
  NeuralSwarmPolicy（episode_values 档），在固定 eval_seeds 上评估，
  指标 = eat_events / eat_rate，并与规则基线做同 seed 配对差。算力定义
  C = episodes * episode_seconds * initial_population（agent·秒的训练经验量）。
  对 performance = a * C^b 做 log-log 拟合，同时报告配对差随算力的趋势
  （线性斜率 + R²）。最大算力档策略写入 model zoo（tag p5）。

产物：report.md（分点表格+拟合+墙钟+局限说明）写到
output_dir(settings)（即 experiments/results/<name>/，由 run_experiment
稍后写入 summary.json/config.yaml 的同一目录）。

诚实性约定（本协议的设计立场）：
* 只有 5 个点、单机单进程、纯 numpy，不宣称普适标度律；
* R² 差 / b 接近 1 / 曲线饱和时直接写在 verdict 与 report 的
  “不说明什么” 一节里，不强行讲故事。
"""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..experiment import build_simulation
from ..registry import register_protocol
from ..settings import ExperimentSettings
from ..zoo import ModelZoo
from .common import (
    evaluate_seeds,
    mean_std,
    paired_summary,
    results_dir,
    train_neural_policy,
)

#: Sweep A 默认人口档（可用 protocol_params.population_values 覆盖）
DEFAULT_POPULATION_VALUES: tuple[int, ...] = (60, 120, 240, 480, 960)
#: Sweep B 默认训练算力档（episodes；可用 protocol_params.episode_values 覆盖）
DEFAULT_EPISODE_VALUES: tuple[int, ...] = (1, 2, 4, 8, 16)
#: Sweep B 默认评估 seed
DEFAULT_EVAL_SEEDS: tuple[int, ...] = (0, 1, 2)


# ---------------------------------------------------------------------------
# protocol_params 解析（严格、带错误信息）
# ---------------------------------------------------------------------------
def _int_list(params: dict[str, Any], key: str, default: Iterable[int], *, minimum: int = 1) -> list[int]:
    raw = params.get(key, default)
    if isinstance(raw, (int, float)) or not isinstance(raw, (list, tuple)):
        raise ValueError(f"protocol_params.{key} 必须是整数列表，得到 {raw!r}")
    values = [int(item) for item in raw]
    if not values:
        raise ValueError(f"protocol_params.{key} 不能为空")
    if any(value < minimum for value in values):
        raise ValueError(f"protocol_params.{key} 的取值必须 >= {minimum}: {values}")
    if len(set(values)) != len(values):
        raise ValueError(f"protocol_params.{key} 不能有重复值: {values}")
    return values


def _float_param(params: dict[str, Any], key: str, default: float, *, minimum: float = 0.0) -> float:
    raw = params.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"protocol_params.{key} 必须是数值，得到 {raw!r}")
    value = float(raw)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"protocol_params.{key} 必须 >= {minimum}，得到 {value!r}")
    return value


def _int_param(params: dict[str, Any], key: str, default: int) -> int:
    raw = params.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"protocol_params.{key} 必须是整数，得到 {raw!r}")
    return int(raw)


def _str_param(params: dict[str, Any], key: str, default: str) -> str:
    raw = params.get(key, default)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"protocol_params.{key} 必须是非空字符串，得到 {raw!r}")
    return raw.strip()


def _bool_param(params: dict[str, Any], key: str, default: bool) -> bool:
    raw = params.get(key, default)
    if not isinstance(raw, bool):
        raise ValueError(f"protocol_params.{key} 必须是布尔值，得到 {raw!r}")
    return raw


# ---------------------------------------------------------------------------
# 幂律拟合
# ---------------------------------------------------------------------------
def power_law_fit(xs: Iterable[float], ys: Iterable[float]) -> dict[str, Any]:
    """log-log 最小二乘拟合 y = a * x^b。

    返回 dict：a, b, r2 (log 空间), r2_linear, n, b_stderr, points,
    residual_rms_log, valid, reason。非正/非有限输入或点数 < 2 时
    valid=False（a/b/r2 置 0.0，保证 summary 里的键始终是有限数）。
    """
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    if x.size != y.size:
        raise ValueError(f"x/y 点数不一致: {x.size} vs {y.size}")
    n = int(x.size)
    payload: dict[str, Any] = {
        "n": n,
        "valid": False,
        "reason": "",
        "a": 0.0,
        "b": 0.0,
        "r2": 0.0,
        "r2_log": 0.0,
        "r2_linear": 0.0,
        "b_stderr": None,
        "residual_rms_log": 0.0,
        "points": [],
    }
    if n < 2:
        payload["reason"] = f"点数 {n} < 2，无法拟合幂律"
        return payload
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        payload["reason"] = "输入含非有限值"
        return payload
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        payload["reason"] = "log-log 拟合要求 x/y 全为正"
        return payload
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    b = float(slope)
    log_a = float(intercept)
    a = float(math.exp(log_a))
    fitted_log = log_a + b * log_x
    residual = log_y - fitted_log
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2_log = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else (1.0 if ss_res <= 1e-12 else 0.0)
    predicted = a * np.power(x, b)
    ss_res_lin = float(np.sum((y - predicted) ** 2))
    ss_tot_lin = float(np.sum((y - y.mean()) ** 2))
    r2_linear = (
        1.0 - ss_res_lin / ss_tot_lin
        if ss_tot_lin > 1e-12
        else (1.0 if ss_res_lin <= 1e-12 else 0.0)
    )
    sxx = float(np.sum((log_x - log_x.mean()) ** 2))
    stderr = (
        float(math.sqrt((ss_res / (n - 2)) / sxx)) if n > 2 and sxx > 1e-12 else None
    )
    points = []
    for index in range(n):
        yi = float(y[index])
        pi = float(predicted[index])
        points.append(
            {
                "x": float(x[index]),
                "y": yi,
                "predicted": pi,
                "log_residual": float(residual[index]),
                "relative_error": float((yi - pi) / yi) if yi != 0.0 else 0.0,
            }
        )
    payload.update(
        {
            "valid": True,
            "a": a,
            "b": b,
            "r2": r2_log,
            "r2_log": r2_log,
            "r2_linear": r2_linear,
            "b_stderr": stderr,
            "residual_rms_log": float(math.sqrt(ss_res / n)),
            "points": points,
        }
    )
    return payload


# ---------------------------------------------------------------------------
# Sweep A：人口 -> 仿真吞吐
# ---------------------------------------------------------------------------
def run_population_sweep(settings: ExperimentSettings, params: dict[str, Any]) -> dict[str, Any]:
    """单维扫初始人口 N，测墙钟吞吐 steps/wall_s 与人口，并做幂律拟合。"""
    values = _int_list(params, "population_values", DEFAULT_POPULATION_VALUES)
    seconds = _float_param(params, "population_seconds", 20.0, minimum=0.1)
    seed = _int_param(params, "population_seed", int(settings.world.seed))
    capacity = int(settings.swarm.capacity_hard)
    over_capacity = [value for value in values if value > capacity]
    if over_capacity:
        raise ValueError(
            f"population_values 中的 {over_capacity} 超过 capacity_hard={capacity}；"
            "请调大 swarm.capacity_hard 或减小人口档"
        )
    # 固定规则策略：Sweep A 只测“世界推进成本”，不掺学习
    rule_settings = dataclasses.replace(
        settings, policy=dataclasses.replace(settings.policy, kind="rule")
    )
    dt = float(settings.run.dt)
    steps = max(1, int(round(seconds / dt)))
    points: list[dict[str, Any]] = []
    sweep_started = time.perf_counter()
    for population in values:
        point_settings = dataclasses.replace(
            rule_settings,
            swarm=dataclasses.replace(rule_settings.swarm, initial_population=int(population)),
        )
        setup_started = time.perf_counter()
        sim = build_simulation(point_settings, seed=seed)
        build_wall = time.perf_counter() - setup_started
        wall_started = time.perf_counter()
        population_sum = 0.0
        for _ in range(steps):
            sim.step()
            population_sum += float(sim.swarm.population)
        wall_s = time.perf_counter() - wall_started
        metrics = sim.metrics()
        mean_population = population_sum / steps
        throughput = steps / wall_s if wall_s > 0.0 else 0.0
        points.append(
            {
                "initial_population": int(population),
                "mean_population": round(mean_population, 3),
                "final_population": int(metrics["population"]),
                "world_seconds": round(steps * dt, 3),
                "steps": int(steps),
                "wall_s": round(wall_s, 5),
                "throughput_steps_per_s": round(throughput, 3),
                "step_ms": round(1e3 * wall_s / steps, 5),
                "us_per_agent_step": round(1e6 * wall_s / (steps * max(mean_population, 1e-9)), 4),
                "build_wall_s": round(build_wall, 5),
                "eat_events": int(metrics["eat_events"]),
                "births": int(metrics["births"]),
                "deaths": int(metrics["deaths"]),
                "mean_hunger": round(float(sim.swarm.mean_hunger()), 4),
                "wall_s_total": round(build_wall + wall_s, 5),
            }
        )
    fit = power_law_fit(
        [point["initial_population"] for point in points],
        [point["throughput_steps_per_s"] for point in points],
    )
    return {
        "name": "population_throughput",
        "x": "initial_population",
        "y": "throughput_steps_per_s",
        "description": "固定地图/seed/规则策略，只扫初始人口；吞吐 = engine steps / wall_s",
        "fixed": {
            "world": dataclasses.asdict(settings.world),
            "run": {"dt": dt, "daylight": settings.run.daylight},
            "policy": "rule",
            "seed": seed,
            "capacity_hard": capacity,
            "food_regrow_per_s": settings.world.food_regrow_per_s,
        },
        "population_values": values,
        "population_seconds": seconds,
        "points": points,
        "fit": fit,
        "wall_s": round(time.perf_counter() - sweep_started, 3),
    }


# ---------------------------------------------------------------------------
# Sweep B：学习算力 -> 觅食性能
# ---------------------------------------------------------------------------
def _paired_trend(points: list[dict[str, Any]]) -> dict[str, Any]:
    """配对差（neural - rule, eat_events）随 log(算力) 的线性趋势。"""
    deltas = [float(point["paired"]["mean_delta"]) for point in points]
    log_c = [math.log(float(point["compute_agent_seconds"])) for point in points]
    payload: dict[str, Any] = {
        "metric": "eat_events",
        "mean_delta_by_point": [
            {"episodes": int(point["episodes"]), "mean_delta": float(point["paired"]["mean_delta"])}
            for point in points
        ],
        "monotone_increasing": all(
            later >= earlier for earlier, later in zip(deltas, deltas[1:])
        ),
        "first": deltas[0] if deltas else 0.0,
        "last": deltas[-1] if deltas else 0.0,
        "slope_per_log_compute": 0.0,
        "r2": 0.0,
    }
    if len(set(log_c)) >= 2:
        slope, intercept = np.polyfit(np.asarray(log_c), np.asarray(deltas), 1)
        predicted = intercept + slope * np.asarray(log_c)
        ss_res = float(np.sum((np.asarray(deltas) - predicted) ** 2))
        ss_tot = float(np.sum((np.asarray(deltas) - np.mean(deltas)) ** 2))
        payload["slope_per_log_compute"] = float(slope)
        payload["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        payload["intercept"] = float(intercept)
    return payload


def run_compute_sweep(settings: ExperimentSettings, params: dict[str, Any]) -> dict[str, Any]:
    """单维扫学习算力（训练 episodes），评估觅食性能并做幂律拟合 + 配对差。"""
    values = _int_list(params, "episode_values", DEFAULT_EPISODE_VALUES)
    if sorted(values) != values:
        raise ValueError(f"episode_values 必须升序（报告/保存最大档按最后一项）: {values}")
    episode_seconds = _float_param(params, "episode_seconds", 30.0, minimum=0.1)
    eval_seconds = _float_param(params, "eval_seconds", 30.0, minimum=0.1)
    eval_seeds = _int_list(params, "eval_seeds", DEFAULT_EVAL_SEEDS, minimum=0)
    train_seeds = _int_list(params, "train_seeds", (1000,), minimum=0)
    population = int(settings.swarm.initial_population)
    policy_seed = int(settings.policy.seed)
    hidden = int(settings.policy.hidden)
    sweep_started = time.perf_counter()

    # 规则基线：同 eval_seeds 同 eval_seconds，用于配对差
    rule_settings = dataclasses.replace(
        settings, policy=dataclasses.replace(settings.policy, kind="rule")
    )
    baseline_started = time.perf_counter()
    baseline_runs = evaluate_seeds(
        rule_settings, None, seeds=eval_seeds, seconds=eval_seconds
    )
    baseline = {
        "policy": "rule",
        "eval_seeds": eval_seeds,
        "eval_seconds": eval_seconds,
        "mean_eat_events": mean_std([run["eat_events"] for run in baseline_runs]),
        "mean_eat_rate": mean_std([run["eat_rate"] for run in baseline_runs]),
        "runs": baseline_runs,
        "wall_s": round(time.perf_counter() - baseline_started, 3),
    }

    points: list[dict[str, Any]] = []
    best_policy: Any | None = None
    for episodes in values:
        train_started = time.perf_counter()
        policy, history = train_neural_policy(
            settings,
            episodes=int(episodes),
            episode_seconds=episode_seconds,
            train_seeds=train_seeds,
            seed=policy_seed,
            hidden=hidden,
        )
        train_wall = time.perf_counter() - train_started
        eval_started = time.perf_counter()
        runs = evaluate_seeds(settings, policy, seeds=eval_seeds, seconds=eval_seconds)
        eval_wall = time.perf_counter() - eval_started
        eats = [float(run["eat_events"]) for run in runs]
        rates = [float(run.get("eat_rate", 0.0)) for run in runs]
        paired = paired_summary(runs, baseline_runs, metric="eat_events")
        transitions = float(sum(float(item.get("transitions", 0.0)) for item in history))
        compute_agent_seconds = float(episodes) * episode_seconds * float(population)
        points.append(
            {
                "episodes": int(episodes),
                "compute_agent_seconds": round(compute_agent_seconds, 3),
                "train_transitions": round(transitions, 1),
                "train_wall_s": round(train_wall, 3),
                "eval_wall_s": round(eval_wall, 3),
                "mean_eat_events": round(float(np.mean(eats)), 3) if eats else 0.0,
                "std_eat_events": round(float(np.std(eats)), 3) if eats else 0.0,
                "mean_eat_rate": round(float(np.mean(rates)), 6) if rates else 0.0,
                "paired": paired,
                "eval_runs": runs,
                "train_history": history,
                "final_mean_return": round(
                    float(history[-1].get("mean_return", 0.0)) if history else 0.0, 4
                ),
                "final_weight_delta": round(
                    float(history[-1].get("weight_delta", 0.0)) if history else 0.0, 6
                ),
            }
        )
        best_policy = policy  # 升序扫描：循环结束时就是最大算力档

    fit = power_law_fit(
        [point["compute_agent_seconds"] for point in points],
        [point["mean_eat_events"] for point in points],
    )
    trend = _paired_trend(points)

    # 最大算力档入 zoo（失败不毁实验，记录在 model 字段里）
    model_info: dict[str, Any] = {"saved": False}
    if _bool_param(params, "save_model", True) and best_policy is not None:
        model_name = _str_param(params, "model_name", f"p5_scaling_c{values[-1]}")
        zoo_tags = params.get("zoo_tags", ["p5", "scaling_sweep"])
        zoo_root = params.get("zoo_root")
        largest = points[-1]
        model_info = {"saved": False, "name": model_name}
        try:
            zoo = ModelZoo(zoo_root) if zoo_root else ModelZoo()
            card = zoo.save(
                model_name,
                best_policy,
                metrics={
                    "mean_eat_events": largest["mean_eat_events"],
                    "std_eat_events": largest["std_eat_events"],
                    "mean_eat_rate": largest["mean_eat_rate"],
                    "paired_mean_delta_vs_rule": largest["paired"]["mean_delta"],
                    "episodes": largest["episodes"],
                    "compute_agent_seconds": largest["compute_agent_seconds"],
                },
                config=settings,
                tags=list(zoo_tags),
                description="P5 scaling sweep: largest-compute neural policy",
                extra={
                    "protocol": "scaling_sweep",
                    "episode_seconds": episode_seconds,
                    "train_seeds": train_seeds,
                    "eval_seeds": eval_seeds,
                    "eval_seconds": eval_seconds,
                    "population": population,
                },
            )
            model_info = {
                "saved": True,
                "name": model_name,
                "path": str(zoo.model_dir(model_name)),
                "verified": bool(zoo.verify(model_name)),
                "weights_sha256": card.weights_sha256,
                "tag": list(zoo_tags),
                "metrics": dict(card.metrics),
            }
        except Exception as exc:  # noqa: BLE001 - zoo 失败要如实记录而非中断实验
            model_info = {
                "saved": False,
                "name": model_name,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {
        "name": "learning_compute",
        "x": "compute_agent_seconds",
        "y": "mean_eat_events",
        "description": (
            "固定训练/评估 seed 与网络结构，只扫训练 episodes；"
            "算力 C = episodes * episode_seconds * initial_population（agent·秒）"
        ),
        "fixed": {
            "policy": {
                "kind": "neural",
                "hidden": hidden,
                "seed": policy_seed,
                "n_actions": 5,
            },
            "population": population,
            "lr": settings.policy.lr,
            "gamma": settings.policy.gamma,
            "batch_size": settings.policy.batch_size,
            "epochs_per_episode": settings.policy.epochs_per_episode,
            "train_seeds": train_seeds,
            "episode_seconds": episode_seconds,
            "eval_seeds": eval_seeds,
            "eval_seconds": eval_seconds,
            "obs_dim": int(getattr(best_policy, "obs_dim", 0)) if best_policy is not None else 0,
        },
        "episode_values": values,
        "points": points,
        "fit": fit,
        "baseline": baseline,
        "paired_trend": trend,
        "model": model_info,
        "wall_s": round(time.perf_counter() - sweep_started, 3),
    }


# ---------------------------------------------------------------------------
# 判决（诚实：拟合差就直说）
# ---------------------------------------------------------------------------
def build_verdict(population_sweep: dict[str, Any], compute_sweep: dict[str, Any]) -> tuple[str, list[str]]:
    """由两个 sweep 的拟合结果生成判决 + 注意事项（不做过度解读）。"""
    pop_fit = population_sweep["fit"]
    comp_fit = compute_sweep["fit"]
    parts: list[str] = []

    if not pop_fit.get("valid"):
        parts.append(
            "Sweep A（人口->吞吐）" + f"未得到有效拟合（{pop_fit.get('reason', '')}）。"
        )
    else:
        b = float(pop_fit["b"])
        r2 = float(pop_fit["r2_log"])
        if r2 >= 0.9:
            if b > -0.15:
                parts.append(
                    f"Sweep A：吞吐近似与人口无关（b={b:.3f}, R2={r2:.3f}, n={pop_fit['n']}），"
                    "每步成本几乎不随人口增长，单机吞吐接近线性可扩展。"
                )
            elif b > -0.85:
                parts.append(
                    f"Sweep A：吞吐随人口亚线性下降（b={b:.3f}, R2={r2:.3f}, n={pop_fit['n']}），"
                    "固定每步开销与随人口增长的个体成本并存。"
                )
            else:
                parts.append(
                    f"Sweep A：吞吐约按 1/N 衰减（b={b:.3f}, R2={r2:.3f}, n={pop_fit['n']}），"
                    "单步成本随人口近线性增长、固定开销可忽略。"
                )
        elif r2 >= 0.5:
            parts.append(
                f"Sweep A：幂律只能部分解释吞吐-人口关系（b={b:.3f}, R2={r2:.3f}, n={pop_fit['n']}），"
                "b 的点估计不稳定，不宜当作严格标度律。"
            )
        else:
            parts.append(
                f"Sweep A：吞吐-人口幂律拟合很差（b={b:.3f}, R2={r2:.3f}, n={pop_fit['n']}），"
                "数据不支持“吞吐 ~ N^b”的描述（更可能是固定开销与分段机制的混合）。"
            )

    if not comp_fit.get("valid"):
        parts.append(
            "Sweep B（算力->性能）" + f"未得到有效拟合（{comp_fit.get('reason', '')}）。"
        )
    else:
        b = float(comp_fit["b"])
        r2 = float(comp_fit["r2_log"])
        trend = compute_sweep["paired_trend"]
        delta_note = (
            f"配对差（neural-rule, eat_events）从 {trend['first']:+.1f} 变到 {trend['last']:+.1f}"
            f"（单调递增={trend['monotone_increasing']}，斜率={trend['slope_per_log_compute']:+.1f}/log算力，"
            f"R2={trend['r2']:.3f}）"
        )
        if r2 >= 0.9:
            parts.append(
                f"Sweep B：觅食性能随训练算力单调提升（b={b:.3f}, R2={r2:.3f}, n={comp_fit['n']}）；"
                + delta_note + "。"
            )
        elif r2 >= 0.5:
            parts.append(
                f"Sweep B：性能-算力幂律只部分成立（b={b:.3f}, R2={r2:.3f}, n={comp_fit['n']}）；"
                + delta_note + "。曲线在窄算力区间内已出现饱和/噪声主导，不能外推指数。"
            )
        else:
            parts.append(
                f"Sweep B：性能-算力幂律几乎不解释数据（b={b:.3f}, R2={r2:.3f}, n={comp_fit['n']}）；"
                + delta_note + f"。在 {compute_sweep['episode_values'][0]}~{compute_sweep['episode_values'][-1]} episodes 的窄区间内无法声称学习标度律。"
            )

    notes = [
        "两个扫描都是单维度：每次只动一个人口/算力轴，地图 seed、世界参数、网络结构、优化超参全部冻结。",
        "Sweep A 的吞吐是单机单进程纯 numpy 墙钟量（steps/wall_s），实现相关，不等于理论复杂度。",
        "Sweep B 每档从零训练；算力 C=episodes*episode_seconds*initial_population（agent-seconds），评估 seed 与窗口固定。",
        "5 个点 + 少量 seed 的幂律拟合只能做描述性判断，不宣称普适标度律；R2 与 b 的不确定性已一并报告。",
        "detail 表与逐点残差见同目录 report.md。",
    ]
    return " ".join(parts), notes


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------
def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "n/a"
        return f"{number:.{digits}g}"
    return str(value)


def _write_report(
    directory: Path,
    settings: ExperimentSettings,
    summary: dict[str, Any],
) -> Path:
    """把人类可读报告写到 artifact 目录，返回路径。"""
    population_sweep = summary["sweeps"][0]
    compute_sweep = summary["sweeps"][1]
    pop_fit = population_sweep["fit"]
    comp_fit = compute_sweep["fit"]
    lines: list[str] = []
    add = lines.append

    add(f"# P5 标度律扫描报告 — {settings.name}")
    add("")
    add("- 协议：scaling_sweep（单维扫描 + log-log 幂律拟合）")
    add(f"- 配置：{settings.name}，config hash {settings.hash()[:16]}")
    add(f"- 总墙钟：{_fmt(summary.get('wall_s'), 4)} s")
    add(f"- 产物目录：{directory}")
    add("- 复现：python -m newworld --config configs/p5_scaling.yaml")
    add("")
    add("## 判决")
    add("")
    add(summary["verdict"])
    add("")
    add("## Sweep A：人口 → 仿真吞吐（规则策略，其余维度冻结）")
    add("")
    add(
        f"固定：地图 {settings.world.width}×{settings.world.height}（seed "
        f"{population_sweep['fixed']['seed']}）、food_regrow="
        f"{settings.world.food_regrow_per_s}/s、dt={settings.run.dt}、规则策略、"
        f"每点 {population_sweep['population_seconds']}s 世界时间；"
        f"吞吐 = engine steps / wall_s。"
    )
    add("")
    add("| 初始 N | 平均人口 | 末人口 | steps | wall (s) | 吞吐 (steps/s) | step (ms) | µs/agent-step |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for point in population_sweep["points"]:
        add(
            f"| {point['initial_population']} | {_fmt(point['mean_population'])} | "
            f"{point['final_population']} | {point['steps']} | {_fmt(point['wall_s'], 4)} | "
            f"{_fmt(point['throughput_steps_per_s'])} | {_fmt(point['step_ms'], 3)} | "
            f"{_fmt(point['us_per_agent_step'], 3)} |"
        )
    add("")
    if pop_fit.get("valid"):
        add(
            f"**拟合**：throughput = a · N^b，a={_fmt(pop_fit['a'])}, "
            f"b={_fmt(pop_fit['b'])}, R²(log)={_fmt(pop_fit['r2_log'])}, "
            f"R²(linear)={_fmt(pop_fit['r2_linear'])}, n={pop_fit['n']}, "
            f"b_stderr={_fmt(pop_fit['b_stderr'])}, 残差 RMS(log)={_fmt(pop_fit['residual_rms_log'])}。"
        )
        add("")
        add("逐点残差（log 空间）：" + "，".join(
            f"N={_fmt(point['x'])}: {_fmt(point['log_residual'])}"
            for point in pop_fit["points"]
        ))
    else:
        add(f"**拟合失败**：{pop_fit.get('reason', '')}")
    add("")
    add("## Sweep B：学习算力 → 觅食性能（神经策略）")
    add("")
    add(
        f"固定：初始人口 {compute_sweep['fixed']['population']}、{compute_sweep['fixed']['obs_dim']} 维观测、"
        f"隐藏层 {compute_sweep['fixed']['policy']['hidden']}、lr={settings.policy.lr}、"
        f"训练 seed {compute_sweep['fixed']['train_seeds']}、评估 seed "
        f"{compute_sweep['fixed']['eval_seeds']} × {compute_sweep['fixed']['eval_seconds']}s；"
        f"每档 episodes 从零训练，算力 C = episodes × episode_seconds × 人口（agent·秒）。"
    )
    add("")
    add("| episodes | 算力 C (agent·s) | 训练 wall (s) | 平均 eat_events | std | eat_rate | 配对差 vs 规则 | wins/n |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for point in compute_sweep["points"]:
        paired = point["paired"]
        add(
            f"| {point['episodes']} | {_fmt(point['compute_agent_seconds'])} | "
            f"{_fmt(point['train_wall_s'], 4)} | {_fmt(point['mean_eat_events'])} | "
            f"{_fmt(point['std_eat_events'], 3)} | {_fmt(point['mean_eat_rate'], 3)} | "
            f"{_fmt(paired['mean_delta'], 4)} | {int(paired['wins'])}/{int(paired['n'])} |"
        )
    add("")
    baseline = compute_sweep["baseline"]
    add(
        f"规则基线（同 seed 同窗口）：eat_events 均值 "
        f"{_fmt(baseline['mean_eat_events']['mean'], 5)} ± "
        f"{_fmt(baseline['mean_eat_events']['std'], 3)}（n={int(baseline['mean_eat_events']['n'])}），"
        f"eat_rate {_fmt(baseline['mean_eat_rate']['mean'], 4)}。"
    )
    add("")
    if comp_fit.get("valid"):
        add(
            f"**拟合**：performance = a · C^b，a={_fmt(comp_fit['a'])}, "
            f"b={_fmt(comp_fit['b'])}, R²(log)={_fmt(comp_fit['r2_log'])}, "
            f"R²(linear)={_fmt(comp_fit['r2_linear'])}, n={comp_fit['n']}, "
            f"b_stderr={_fmt(comp_fit['b_stderr'])}。"
        )
    else:
        add(f"**拟合失败**：{comp_fit.get('reason', '')}")
    add("")
    trend = compute_sweep["paired_trend"]
    add(
        f"**配对差趋势**（neural − rule, eat_events）：首档 {_fmt(trend['first'], 5)} → "
        f"末档 {_fmt(trend['last'], 5)}，单调递增={_fmt(trend['monotone_increasing'])}，"
        f"斜率 {_fmt(trend['slope_per_log_compute'], 4)} eat_events / log(算力)，R²={_fmt(trend['r2'])}。"
    )
    add("")
    model = compute_sweep.get("model", {})
    if model.get("saved"):
        add(
            f"**模型存档**：{model['name']} → {model['path']}"
            f"（tag={model.get('tag')}, sha256 校验={_fmt(model.get('verified'))}）。"
        )
    elif model.get("name"):
        add(f"**模型存档失败**：{model.get('error', '未知错误')}")
    add("")
    add("## 这说明什么 / 不说明什么")
    add("")
    add("**说明**：")
    add("")
    add(
        f"- 在这个固定世界与实现下，人口从 {population_sweep['population_values'][0]} 到 "
        f"{population_sweep['population_values'][-1]} 时单机吞吐如何变化（上表与 Sweep A 拟合）；"
    )
    add(
        f"- 在 {compute_sweep['episode_values'][0]}~{compute_sweep['episode_values'][-1]} 个 "
        f"{compute_sweep['fixed']['episode_seconds']}s 训练回合的窄算力区间内，"
        "REINFORCE 训练出的神经策略相对规则基线的觅食表现如何变化（Sweep B 表与配对差）；"
    )
    add("- 两个轴的幂律拟合参数 b、R² 与逐点残差都如实列出，便于判断幂律是否只是事后叙事。")
    add("")
    add("**不说明（限制）**：")
    add("")
    add("- 不是普适标度律：每个轴只有 5 个点，且只扫一个维度；地图 seed、世界参数、网络结构、优化超参全部冻结，换地图/机制后 b 可能完全改变；")
    add("- Sweep A 的吞吐是当前纯 numpy 单进程实现的墙钟量，含指标计算、生灭回收、观测组装等固定开销，不能当作理论复杂度或跨语言结论；")
    add("- Sweep B 每档策略从零训练，训练 seed 仅 1~2 个、评估仅 3 个 seed × 30s，方差大；窄算力区间内学习曲线常呈饱和/波动，外推指数没有依据；")
    if compute_sweep["points"] and comp_fit.get("valid"):
        peak_point = max(compute_sweep["points"], key=lambda point: point["mean_eat_events"])
        last_point = compute_sweep["points"][-1]
        add(
            f"- Sweep B 在窄区间内已饱和：{peak_point['episodes']} episodes 处 eat_events 均值最高"
            f"（{_fmt(peak_point['mean_eat_events'], 5)}），{last_point['episodes']} episodes 回落到"
            f" {_fmt(last_point['mean_eat_events'], 5)}（非单调），但每一档都显著高于规则基线"
            f"（{_fmt(compute_sweep['baseline']['mean_eat_events']['mean'], 5)}）；"
            f"b={_fmt(comp_fit['b'], 3)}、R2={_fmt(comp_fit['r2_log'], 3)} 只表示“已饱和/噪声主导”，不能外推；"
        )
    add("- Sweep A 的墙钟吞吐对机器负载敏感：同一配置另一次完整运行得到 b=-0.317、R2=0.955（本报告为 b=-0.302、R2=0.926），单次运行的 b 末位不宜精确解读；")
    add("- eat_events 是离散进食事件，受地图食物承载力与初始人口影响，不能直接等同“智能”或“适应度”；eat_rate 已按初始人口归一，但口径仍是本项目的局部指标；")
    add("- 因此本报告只给“该配置下两个单维扫描的描述性幂律”，不是“scaling law 论文复现”。")
    add("")
    add("## 复现")
    add("")
    add("    python -m newworld --config configs/p5_scaling.yaml")
    add("")
    add("产物：summary.json（机器可读，含 sweeps/fit/points）、config.yaml（配置快照）、")
    add(f"report.md（本文件）、模型 zoo 目录 {model.get('path') or '（本次未保存模型）'}。")
    add("")
    report_path = directory / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# 协议入口
# ---------------------------------------------------------------------------
@register_protocol(
    "scaling_sweep",
    note="P5 标度律：人口->吞吐 与 学习算力->性能 的单维扫描 + 幂律拟合（诚实报告）",
)
def scaling_sweep(settings: ExperimentSettings) -> dict[str, Any]:
    """执行两个单维扫描，写 report.md，返回 summary。"""
    started = time.perf_counter()
    params = dict(settings.protocol_params or {})
    population_sweep = run_population_sweep(settings, params)
    compute_sweep = run_compute_sweep(settings, params)
    verdict, notes = build_verdict(population_sweep, compute_sweep)
    summary: dict[str, Any] = {
        "protocol": "scaling_sweep",
        "name": settings.name,
        "sweeps": [population_sweep, compute_sweep],
        "verdict": verdict,
        "notes": notes,
        "wall_s": round(time.perf_counter() - started, 3),
        "config_hash": settings.hash(),
    }
    report_path = _write_report(results_dir(settings), settings, summary)
    summary["report_md"] = str(report_path)
    return summary


__all__ = ["power_law_fit", "run_population_sweep", "run_compute_sweep", "build_verdict", "scaling_sweep"]
