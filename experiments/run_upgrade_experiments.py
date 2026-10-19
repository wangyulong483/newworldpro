"""本轮升级的实验矩阵：6 配置 × 多随机种子（任务书 §7.3）。

配置（同一世界规则、同一预算，仅切换决策模块）：

| 配置 | 高层目标选择 | 低层动作 | 知识/传承 | 创新 |
| --- | --- | --- | --- | --- |
| baseline | 无规划（A 组启发式关闭） | BC 网络 | 关闭 | 关闭 |
| planning | 规则启发式（C 组） | BC 网络 | 开启 | 关闭 |
| learned | 训练的高层策略 | BC 网络 | 开启 | 关闭 |
| random_policy | 均匀随机目标（对照） | BC 网络 | 开启 | 关闭 |
| lowlevel_random | 规则启发式 | 随机动作（对照） | 开启 | 关闭 |
| innovation | 训练的高层策略 | BC 网络 | 开启 | 开启 |

默认用 **100 起的 10 个 seed**（与高层策略训练/评估 seed 不重叠时才算"未见布局"，
命令行会打印训练脚本建议的 seed 划分）。输出：runs.csv / aggregate.json / report.md。

运行::

    cd newworldpro
    python experiments/run_upgrade_experiments.py --seeds 100 101 ... 109
    python experiments/run_upgrade_experiments.py --configs baseline,planning --seed-start 100 --num-seeds 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newworld.simulation import RunSpec, SocietyRunner  # noqa: E402

DEFAULT_POLICY = ROOT / "assets" / "goal_policy.npz"

#: config name -> RunSpec keyword overrides
CONFIGS: dict[str, dict[str, object]] = {
    "baseline": dict(group="A"),
    "planning": dict(group="C"),
    "learned": dict(group="C", goal_policy_mode="neural"),
    "random_policy": dict(group="C", goal_policy_mode="random"),
    "lowlevel_random": dict(group="C", low_level_mode="random"),
    "innovation": dict(group="C", goal_policy_mode="neural", enable_innovation=True),
}

NUMERIC_KEYS = (
    "total_agents_created",
    "total_deaths",
    "tech_verified",
    "tech_graph_depth",
    "facilities_active",
    "innovations",
    "goal_completion_rate",
    "plan_cost_error",
    "repeat_run_max",
    "no_progress_repeat_ratio",
    "knowledge_retention",
    "innovation_efficiency",
    "cooperation_benefit",
    "cooperation_index",
    "extinction_time",
    "survival_rate",
    "action_entropy",
)


def _extinction_cause(runner, result: dict) -> str:
    """Reason the run ended; empty when the run did not go extinct."""
    civ = runner.world.civilization
    if not bool(getattr(civ, "extinct", False)):
        return ""
    cause = str(getattr(civ, "extinction_cause", "") or "")
    if cause and cause != "unknown":
        return cause
    records = (result.get("archive") or {}).get("records") or []
    causes = [str(record.get("cause", "")) for record in records if not record.get("alive", False)]
    if not causes:
        return "unknown"
    starved = causes.count("starvation")
    return "starvation" if starved >= causes.count("old_age") else "old_age"


def run_config(config: str, seed: int, *, steps: int, dt: float, width: int,
               height: int, policy_path: str | None) -> dict[str, object]:
    overrides = dict(CONFIGS[config])
    if overrides.get("goal_policy_mode") == "neural" and policy_path:
        overrides["goal_policy_path"] = policy_path
    spec = RunSpec(
        seed=seed,
        steps=steps,
        dt=dt,
        width=width,
        height=height,
        perception=False,
        **overrides,  # type: ignore[arg-type]
    )
    runner = SocietyRunner(spec)
    result = runner.run()
    metrics = result["metrics"]
    civilization = result["civilization"]
    decisions = result.get("decisions", {})
    stats = result.get("goals", {})
    row: dict[str, object] = {
        "config": config,
        "seed": seed,
        "steps_run": result["steps_run"],
        "world_time_s": result["world_time_s"],
        "wall_s": result["wall_s"],
        "extinct": bool(civilization.get("extinct")),
        "extinction_cause": _extinction_cause(runner, result),
        "goal_policy_mode": result.get("goal_policy_mode", ""),
        "policy_attached": runner.population.goal_policy is not None,
    }
    for key in NUMERIC_KEYS:
        if key == "facilities_active":
            row[key] = float(civilization.get("facilities_active", 0.0))
        elif key == "cooperation_index":
            row[key] = float(civilization.get("cooperation_index", 0.0))
        elif key == "tech_verified":
            row[key] = float(metrics.get("tech_verified", len(result["knowledge"]["verified"])))
        elif key == "innovations":
            row[key] = float(len(result.get("innovations", [])))
        elif key == "extinction_time":
            row[key] = float(metrics.get("extinction_time", -1.0))
        else:
            row[key] = float(metrics.get(key, 0.0))
    # decision-source shares (fraction of per-frame decisions)
    for key in ("goal", "teacher", "neural", "infrastructure", "policy_goal",
                "policy_infeasible", "lowlevel_off", "lowlevel_random"):
        row[f"share_{key}"] = float(decisions.get(key, 0.0))
    row["goals_created"] = float(stats.get("goals_created", 0.0))
    row["goals_completed"] = float(stats.get("goals_completed", 0.0))
    row["goals_failed"] = float(stats.get("goals_failed", 0.0))
    row["policy_decisions"] = float(runner.population.decision_counts.get("policy_goal", 0))
    return row


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for config in sorted({str(row["config"]) for row in rows}):
        group_rows = [row for row in rows if row["config"] == config]
        keys = [
            key for key in group_rows[0]
            if isinstance(group_rows[0][key], (int, float)) and not isinstance(group_rows[0][key], bool)
            and key != "seed"
        ]
        result[config] = {}
        for key in keys:
            values = [float(row[key]) for row in group_rows if isinstance(row.get(key), (int, float))]
            if not values:
                continue
            result[config][key] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "n": float(len(values)),
            }
        result[config]["_failure_rate"] = {
            # failure = went extinct by starvation (not merely some deaths)
            "mean": statistics.fmean(
                1.0 if bool(row.get("extinct")) and str(row.get("extinction_cause")) == "starvation"
                else 0.0
                for row in group_rows
            ),
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": float(len(group_rows)),
        }
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    lines = [",".join(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, rows: list[dict[str, object]],
                 aggregates: dict[str, dict[str, dict[str, float]]],
                 spec: dict[str, object]) -> None:
    configs = sorted({str(row["config"]) for row in rows})
    lines = ["# 本轮升级实验报告（自动生成）", ""]
    lines.append("> 由 `experiments/run_upgrade_experiments.py` 生成；所有数字来自真实运行。")
    lines.append("")
    lines.append("## 固定预算")
    for key, value in spec.items():
        lines.append(f"- `{key}` = {value}")
    lines.append("")
    lines.append("## 跨种子统计（均值 ± 标准差 [最小, 最大]）")
    lines.append("")
    metrics = ("total_agents_created", "tech_verified", "facilities_active", "innovations",
               "goal_completion_rate", "repeat_run_max", "cooperation_index",
               "extinction_time", "extinction_starvation_rate")
    header = "| 指标 | " + " | ".join(configs) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---" for _ in configs) + " |")
    for metric in metrics:
        cells = []
        for config in configs:
            if metric == "extinction_starvation_rate":
                stats = aggregates.get(config, {}).get("_failure_rate")
            else:
                stats = aggregates.get(config, {}).get(metric)
            if stats is None:
                cells.append("-")
            elif metric in ("extinction_time",):
                cells.append(f"{stats['mean']:.0f} [{stats['min']:.0f}, {stats['max']:.0f}]")
            else:
                cells.append(f"{stats['mean']:.3f}±{stats['std']:.3f} [{stats['min']:.3f}, {stats['max']:.3f}]")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 决策来源占比（均值，按逐帧决策数归一）")
    lines.append("")
    shares = ("share_goal", "share_teacher", "share_neural", "share_policy_goal",
              "share_policy_infeasible", "share_lowlevel_off", "share_lowlevel_random")
    lines.append("| 配置 | " + " | ".join(shares) + " | policy_decisions |")
    lines.append("| --- | " + " | ".join("---" for _ in shares) + " | --- |")
    for config in configs:
        cells = [f"{aggregates[config].get(key, {}).get('mean', 0.0):.3f}" for key in shares]
        policy_decisions = aggregates[config].get("policy_decisions", {}).get("mean", 0.0)
        lines.append(f"| {config} | " + " | ".join(cells) + f" | {policy_decisions:.1f} |")
    lines.append("")
    lines.append("## 每次运行")
    lines.append("")
    lines.append("| config | seed | steps | t(s) | extinct | cause | tech | facilities | innovations | decisions(neural/teacher/goal) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        decisions = f"{row['share_neural']:.2f}/{row['share_teacher']:.2f}/{row['share_goal']:.2f}"
        lines.append(
            f"| {row['config']} | {row['seed']} | {row['steps_run']} | {row['world_time_s']} | "
            f"{'yes' if row['extinct'] else 'no'} | {row['extinction_cause'] or '-'} | "
            f"{row['tech_verified']:.0f} | {row['facilities_active']:.0f} | {row['innovations']:.0f} | {decisions} |"
        )
    lines.append("")
    lines.append("## 复现")
    lines.append("")
    lines.append("```powershell")
    lines.append("cd newworldpro")
    lines.append("$env:TEMP = (Resolve-Path .tmp).Path")
    lines.append("python experiments/run_upgrade_experiments.py --seed-start 100 --num-seeds 10")
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY))
    parser.add_argument("--out", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)

    configs = [item.strip() for item in args.configs.split(",") if item.strip()]
    seeds = args.seeds or [args.seed_start + index for index in range(args.num_seeds)]
    policy_path = str(Path(args.policy_path)) if args.policy_path else None
    policy_ready = bool(policy_path and Path(policy_path).exists())
    for config in configs:
        if config not in CONFIGS:
            raise SystemExit(f"unknown config {config!r}; available: {sorted(CONFIGS)}")
        if CONFIGS[config].get("goal_policy_mode") == "neural" and not policy_ready:
            raise SystemExit(
                f"config {config!r} needs a trained checkpoint; {policy_path} not found."
            )

    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else ROOT / "experiments" / "results" / f"upgrade_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_dict = {
        "configs": ",".join(configs),
        "seeds": ",".join(str(seed) for seed in seeds),
        "steps": args.steps,
        "dt": args.dt,
        "size": f"{args.width}x{args.height}",
        "perception": False,
        "max_agents_created": 8,
        "policy_path": policy_path if policy_ready else "(none)",
        "policy_checkpoint_exists": policy_ready,
        "world_version": __import__("newworld").WORLD_VERSION,
    }
    rows: list[dict[str, object]] = []
    for config in configs:
        for seed in seeds:
            row = run_config(
                config,
                seed,
                steps=args.steps,
                dt=args.dt,
                width=args.width,
                height=args.height,
                policy_path=policy_path,
            )
            rows.append(row)
            print(
                f"[{config} seed {seed}] t={row['world_time_s']}s tech={row['tech_verified']:.0f} "
                f"fac={row['facilities_active']:.0f} innov={row['innovations']:.0f} "
                f"extinct={row['extinct']}({row['extinction_cause'] or '-'}) "
                f"decisions neural/teacher/goal={row['share_neural']:.2f}/"
                f"{row['share_teacher']:.2f}/{row['share_goal']:.2f}"
            )
    aggregates = aggregate(rows)
    write_csv(out_dir / "runs.csv", rows)
    (out_dir / "aggregate.json").write_text(
        json.dumps(aggregates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(out_dir / "report.md", rows, aggregates, spec_dict)
    print(f"\n结果目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
