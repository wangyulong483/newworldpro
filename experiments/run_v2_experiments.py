"""V2 升级实验矩阵：感知进入决策、低层 BC→RL、放开人口上限、无限世界。

真实运行、真实数据、不挑 seed。输出：
``experiments/results/v2_<tag>/`` 下的 runs.csv / aggregate.json / report.md。

用法::

    cd newworldpro
    $env:TEMP=(Resolve-Path .tmp).Path
    python experiments/run_v2_experiments.py --seed-start 200 --num-seeds 6 --tag final
    python experiments/run_v2_experiments.py --seed-start 200 --num-seeds 1 --tag smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from newworld.simulation import RunSpec, SocietyRunner  # noqa: E402

ASSETS = ROOT / "assets"


@dataclass(frozen=True)
class Config:
    name: str
    description: str
    visual: bool
    low_level: str = "full"
    infinite: bool = False
    cap: int = 8
    steps: int = 6000


CONFIGS: tuple[Config, ...] = (
    Config(
        "c_bc_vision_off",
        "C 组（规划+知识），感知特征关闭，底层 BC —— 对照基线",
        visual=False,
    ),
    Config(
        "c_bc_vision_on",
        "同上 + 真实视网膜特征（16 维）进入 56 维观测",
        visual=True,
    ),
    Config(
        "c_rl_policy",
        "感知开 + 低层 RL 策略（assets/lowlevel_rl.npz，真实 REINFORCE 训练）",
        visual=True,
        low_level="rl",
    ),
    Config(
        "c_rl_random",
        "感知开 + 随机低层动作（RL 的对照）",
        visual=True,
        low_level="random",
    ),
    Config("cap8_infinite", "无限世界 + 8 人上限", visual=True, infinite=True, cap=8),
    Config("cap32_infinite", "无限世界 + 32 人上限（观察人口动态/恢复）", visual=True, infinite=True, cap=32),
    Config("cap32_finite", "有限 640x400 世界 + 32 人上限（空间/资源约束对照）", visual=True, infinite=False, cap=32),
)

NUMERIC_KEYS = (
    "steps_run",
    "world_time_s",
    "total_agents_created",
    "total_deaths",
    "alive_count",
    "tech_verified",
    "tech_graph_depth",
    "facilities_active",
    "innovations",
    "goal_completion_rate",
    "repeat_run_max",
    "no_progress_repeat_ratio",
    "knowledge_retention",
    "cooperation_index",
    "survival_rate",
    "extinction_time",
    "wall_s",
    "visual_active_dims",
    "lowlevel_transitions",
    "lowlevel_mean_reward",
)


def run_config(config: Config, seed: int, *, dt: float = 0.05,
               width: int = 640, height: int = 400,
               window_atoms: int = 0, perception_stride: int = 8) -> dict[str, Any]:
    """运行一个配置 × 一个种子，返回一行真实结果。"""
    spec = RunSpec(
        group="C",
        seed=seed,
        steps=config.steps,
        dt=dt,
        width=width,
        height=height,
        perception=False,
        disturbance=False,
        low_level_mode=config.low_level,
        lowlevel_policy_path=str(ASSETS / "lowlevel_rl.npz")
        if config.low_level == "rl" and (ASSETS / "lowlevel_rl.npz").exists()
        else None,
        visual_features=config.visual,
        infinite=config.infinite,
        window_atoms=window_atoms,
        max_agents_created=config.cap,
        perception_stride=perception_stride,
    )
    runner = SocietyRunner(spec)
    started = time.perf_counter()
    result = runner.run()
    wall = time.perf_counter() - started
    metrics = result.get("metrics") or {}
    civilization = result.get("civilization") or {}
    visual = result.get("visual") or {}
    lowlevel = result.get("lowlevel") or {}
    decisions = result.get("decisions") or {}
    created = int(civilization.get("total_agents_created", 0) or 0)
    row: dict[str, Any] = {
        "config": config.name,
        "seed": int(seed),
        "steps_run": int(result.get("steps_run", 0) or 0),
        "world_time_s": float(result.get("world_time_s", 0.0) or 0.0),
        "wall_s": round(wall, 3),
        "extinct": bool(civilization.get("extinct", False)),
        "extinction_cause": str(civilization.get("extinction_cause", "") or ""),
        "total_agents_created": created,
        "total_deaths": int(civilization.get("total_deaths", 0) or 0),
        "alive_count": int(civilization.get("alive_count", 0) or 0),
        "tech_verified": int(metrics.get("tech_verified", 0) or 0),
        "tech_graph_depth": int(metrics.get("tech_graph_depth", 0) or 0),
        "facilities_active": int(metrics.get("facilities_active", 0) or 0),
        "innovations": int(metrics.get("innovations", 0) or 0),
        "goal_completion_rate": float(metrics.get("goal_completion_rate", 0.0) or 0.0),
        "repeat_run_max": int(metrics.get("repeat_run_max", 0) or 0),
        "no_progress_repeat_ratio": float(metrics.get("no_progress_repeat_ratio", 0.0) or 0.0),
        "knowledge_retention": float(metrics.get("knowledge_retention", 0.0) or 0.0),
        "cooperation_index": float(metrics.get("cooperation_index", 0.0) or 0.0),
        "survival_rate": float(metrics.get("survival_rate", 0.0) or 0.0),
        "extinction_time": float(metrics.get("extinction_time", -1.0) or -1.0),
        "visual_active_dims": int(visual.get("active_dims", 0) or 0),
        "visual_samples": int(visual.get("samples", 0) or 0),
        "lowlevel_transitions": float(lowlevel.get("transitions", 0.0) or 0.0),
        "lowlevel_mean_reward": float(lowlevel.get("mean_reward", 0.0) or 0.0),
        "window_origin_px": float(result.get("window_origin_px", 0.0) or 0.0),
        "cap_limit": int(config.cap),
        "cap_violation": bool(created > config.cap),
    }
    for name in ("neural", "teacher", "goal", "infrastructure", "lowlevel_random",
                 "lowlevel_rl", "policy_goal"):
        row[f"share_{name}"] = float(decisions.get(name, 0.0) or 0.0)
    return row


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _t_critical(count: int) -> float:
    """双尾 95% t 临界值（df=count-1；覆盖常用小样本）。"""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return table.get(max(1, count - 1), 1.96)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    configs = sorted({row["config"] for row in rows})
    result: dict[str, Any] = {}
    for name in configs:
        group = [row for row in rows if row["config"] == name]
        entry: dict[str, Any] = {}
        for key in NUMERIC_KEYS:
            entry[key] = _stats([float(row.get(key, 0.0) or 0.0) for row in group])
        extinct = [row for row in group if row["extinct"]]
        causes: dict[str, int] = {}
        for row in extinct:
            cause = str(row.get("extinction_cause") or "unknown")
            causes[cause] = causes.get(cause, 0) + 1
        entry["extinct_runs"] = len(extinct)
        entry["n_runs"] = len(group)
        entry["extinction_rate"] = len(extinct) / len(group) if group else 0.0
        entry["extinction_causes"] = causes
        entry["cap_violations"] = sum(1 for row in group if row.get("cap_violation"))
        result[name] = entry
    return result


def paired(rows: list[dict[str, Any]], a: str, b: str, key: str) -> dict[str, float] | None:
    left = {row["seed"]: float(row.get(key, 0.0) or 0.0) for row in rows if row["config"] == a}
    right = {row["seed"]: float(row.get(key, 0.0) or 0.0) for row in rows if row["config"] == b}
    diffs = [left[seed] - right[seed] for seed in sorted(left) if seed in right]
    if not diffs:
        return None
    mean = statistics.fmean(diffs)
    std = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
    se = std / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    margin = _t_critical(len(diffs)) * se
    return {
        "mean": mean, "low": mean - margin, "high": mean + margin,
        "wins": sum(1 for d in diffs if d > 0),
        "losses": sum(1 for d in diffs if d < 0),
        "ties": sum(1 for d in diffs if d == 0),
        "n": len(diffs),
    }


PAIRS: tuple[tuple[str, str, str], ...] = (
    ("c_bc_vision_on", "c_bc_vision_off", "感知开 vs 关"),
    ("c_rl_policy", "c_bc_vision_on", "RL 低层 vs BC"),
    ("c_rl_random", "c_bc_vision_on", "随机低层 vs BC"),
    ("cap32_infinite", "cap8_infinite", "32 人上限 vs 8 人（无限世界）"),
    ("cap32_finite", "cap8_infinite", "32 人上限（有限世界） vs 8 人（无限世界）"),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]],
                 aggregates: dict[str, Any], spec: dict[str, Any],
                 paired_rows: dict[str, Any]) -> None:
    lines = ["# V2 实验原始报告（自动生成）", ""]
    for key, value in spec.items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## 每配置汇总（均值 ± 标准差 [min, max]）", ""]
    keys = ("tech_verified", "facilities_active", "alive_count", "total_agents_created",
            "innovations", "goal_completion_rate", "cooperation_index",
            "visual_active_dims", "lowlevel_mean_reward")
    header = "| 配置 | " + " | ".join(keys) + " | 灭绝率 | 上限违规 |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---" for _ in keys) + " | --- | --- |")
    for name in sorted(aggregates):
        entry = aggregates[name]
        cells = []
        for key in keys:
            stats = entry[key]
            cells.append(f"{stats['mean']:.2f}±{stats['std']:.2f} [{stats['min']:.2f},{stats['max']:.2f}]")
        lines.append(
            f"| {name} | " + " | ".join(cells)
            + f" | {entry['extinction_rate']:.2f} | {entry['cap_violations']} |"
        )
    lines += ["", "## 配对差（同 seed，95% CI）", ""]
    for key, data in paired_rows.items():
        lines.append(f"- {key}: {data}")
    lines += ["", "## 逐 run", ""]
    lines.append("| config | seed | tech | fac | alive | created | extinct | cause | visual_dims | wall_s |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['seed']} | {row['tech_verified']} | "
            f"{row['facilities_active']} | {row['alive_count']} | {row['total_agents_created']} | "
            f"{row['extinct']} | {row['extinction_cause']} | {row['visual_active_dims']} | "
            f"{row['wall_s']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V2 experiment matrix")
    parser.add_argument("--seed-start", type=int, default=200)
    parser.add_argument("--num-seeds", type=int, default=6)
    parser.add_argument("--tag", default="final")
    parser.add_argument("--steps", type=int, default=None, help="override per-config steps")
    parser.add_argument("--perception-stride", type=int, default=8)
    parser.add_argument("--only", default="", help="逗号分隔的配置名（调试用）")
    args = parser.parse_args(argv)

    seeds = [args.seed_start + index for index in range(args.num_seeds)]
    only = {name.strip() for name in (args.only or "").split(",") if name.strip()}
    configs = [config for config in CONFIGS if not only or config.name in only]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for config in configs:
        if args.steps:
            config = Config(**{**config.__dict__, "steps": int(args.steps)})
        for seed in seeds:
            row = run_config(
                config, seed,
                window_atoms=128,
                perception_stride=max(1, int(args.perception_stride)),
            )
            rows.append(row)
            print(
                f"[{config.name} seed {seed}] t={row['world_time_s']:.0f}s "
                f"tech={row['tech_verified']} fac={row['facilities_active']} "
                f"alive={row['alive_count']}/{row['total_agents_created']} "
                f"innov={row['innovations']} visual={row['visual_active_dims']}/16 "
                f"lowlevel={row['lowlevel_transitions']:.0f} extinct={row['extinct']}"
                f"({row['extinction_cause'] or '-'}) wall={row['wall_s']}s"
            )
            if row["cap_violation"]:
                print(f"  !! CAP VIOLATION: created {row['total_agents_created']} > {config.cap}")
    elapsed = time.perf_counter() - started

    out = ROOT / "experiments" / "results" / f"v2_{args.tag}"
    write_csv(out / "runs.csv", rows)
    aggregates = aggregate(rows)
    paired_rows: dict[str, Any] = {}
    for a, b, label in PAIRS:
        if not any(row["config"] == a for row in rows) or not any(row["config"] == b for row in rows):
            continue
        for key in ("tech_verified", "facilities_active", "alive_count",
                    "goal_completion_rate", "visual_active_dims", "cooperation_index"):
            data = paired(rows, a, b, key)
            if data is not None:
                paired_rows[f"{label} :: {key}"] = {
                    k: round(v, 4) for k, v in data.items()
                }
    aggregate_payload = {
        "spec": {
            "seeds": seeds,
            "steps": args.steps or "config default (6000)",
            "dt": 0.05,
            "size": "640x400",
            "infinite_window_atoms": 128,
            "perception_stride": args.perception_stride,
            "elapsed_s": round(elapsed, 1),
            "runs": len(rows),
        },
        "configs": aggregates,
        "paired": paired_rows,
    }
    (out / "aggregate.json").write_text(
        json.dumps(aggregate_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    spec = {
        "配置数": len(configs),
        "种子": f"{seeds[0]}..{seeds[-1]}（{len(seeds)} 个）",
        "步数": args.steps or "6000（300 世界秒）",
        "总运行": len(rows),
        "墙钟": f"{elapsed:.1f}s",
        "感知刷新 stride": args.perception_stride,
    }
    write_report(out / "report.md", rows, aggregates, spec, paired_rows)
    print(f"\n结果目录：{out}（{len(rows)} runs，墙钟 {elapsed:.1f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
