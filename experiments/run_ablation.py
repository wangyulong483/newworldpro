"""消融实验运行器：A/B/C/D × 多随机种子。

任务书 §7 要求：
- A 组：原始系统；B 组：+目标规划；C 组：+持久化知识与传承；D 组：完整系统。
- 各组保持相同初始条件、创建数上限（8）、模拟预算与训练预算。
- 每组使用多个随机种子，报告均值、波动范围和失败案例。

运行::

    cd newworldpro
    python experiments/run_ablation.py --seeds 0 1 2 --steps 24000 --dt 0.05
    python experiments/run_ablation.py --groups A,D --seeds 0 --steps 3000 --tag smoke

输出（默认 ``experiments/results/ablation_<tag>``）：
- ``summary_<group>_seed<seed>.json``：单次运行的完整结果（由 SocietyRunner 写出）
- ``runs.csv``：每次运行一行（关键指标）
- ``aggregate.json``：分组 × 指标的 mean/std/min/max
- ``report.md``：人类可读的对照表 + 失败案例 + 复现命令
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

#: metrics copied from the run result into the flat CSV
KEY_METRICS: tuple[str, ...] = (
    "total_agents_created",
    "survival_rate",
    "action_entropy",
    "repeat_run_max",
    "no_progress_repeat_ratio",
    "goal_completion_rate",
    "plan_cost_error",
    "tech_verified",
    "tech_graph_depth",
    "tech_graph_edges",
    "knowledge_retention",
    "innovation_efficiency",
    "cooperation_benefit",
    "recovery_after_disturbance",
    "extinction_time",
)
CIV_METRICS: tuple[str, ...] = (
    "alive_count",
    "total_deaths",
    "facilities_total",
    "facilities_active",
    "innovation_count",
    "cooperation_index",
    "collapse_state_code",
    "avg_energy",
    "extinct",
)


def _row(group: str, seed: int, result: dict) -> dict[str, object]:
    metrics = result.get("metrics", {})
    civilization = result.get("civilization", {})
    row: dict[str, object] = {
        "group": group,
        "seed": seed,
        "steps_run": result.get("steps_run"),
        "world_time_s": result.get("world_time_s"),
        "wall_s": result.get("wall_s"),
        "extinct": civilization.get("extinct", metrics.get("extinction_time", -1) >= 0),
        "extinction_time": metrics.get("extinction_time", -1.0),
    }
    for name in KEY_METRICS:
        row[name] = metrics.get(name, 0.0)
    for name in CIV_METRICS:
        row[name] = civilization.get(name, 0.0)
    row["goals_created"] = result.get("goals", {}).get("goals_created", 0.0)
    row["goals_completed"] = result.get("goals", {}).get("goals_completed", 0.0)
    row["goals_failed"] = result.get("goals", {}).get("goals_failed", 0.0)
    row["knowledge_blocked"] = result.get("knowledge", {}).get("blocked_attempts", 0)
    return row


def _aggregate(rows: list[dict[str, object]], group: str) -> dict[str, dict[str, float]]:
    group_rows = [row for row in rows if row["group"] == group]
    if not group_rows:
        return {}
    numeric_keys = [key for key, value in group_rows[0].items() if isinstance(value, (int, float)) and key != "seed"]
    result: dict[str, dict[str, float]] = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in group_rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        result[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    lines = [",".join(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    rows: list[dict[str, object]],
    aggregates: dict[str, dict[str, dict[str, float]]],
    spec: dict[str, object],
) -> None:
    lines: list[str] = []
    lines.append("# 消融实验报告（自动生成）\n")
    lines.append("> 本文件由 `experiments/run_ablation.py` 生成；所有数字来自实际运行，未做人工调整。\n")
    lines.append("## 固定预算\n")
    for key, value in spec.items():
        lines.append(f"- `{key}` = {value}")
    lines.append("")
    lines.append("## 关键指标对照（跨种子 均值 ± 标准差 [最小, 最大]）\n")
    groups = sorted({str(row["group"]) for row in rows})
    headline = (
        "tech_verified",
        "tech_graph_depth",
        "knowledge_retention",
        "innovation_efficiency",
        "goal_completion_rate",
        "repeat_run_max",
        "no_progress_repeat_ratio",
        "cooperation_index",
        "extinction_time",
        "total_agents_created",
        "alive_count",
    )
    header = "| 指标 | " + " | ".join(groups) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---" for _ in groups) + " |")
    for metric in headline:
        cells = []
        for group in groups:
            stats = aggregates.get(group, {}).get(metric)
            if stats is None:
                cells.append("-")
            elif metric in ("extinction_time",):
                cells.append(f"{stats['mean']:.0f} [{stats['min']:.0f}, {stats['max']:.0f}]")
            else:
                cells.append(f"{stats['mean']:.3f}±{stats['std']:.3f} [{stats['min']:.3f}, {stats['max']:.3f}]")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 每次运行\n")
    lines.append("| group | seed | steps | t(s) | extinct | extinction_t | tech | depth | retention | goals_done |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['seed']} | {row['steps_run']} | {row['world_time_s']} | "
            f"{'yes' if row.get('extinct') else 'no'} | {row.get('extinction_time'):.0f} | "
            f"{row.get('tech_verified'):.0f} | {row.get('tech_graph_depth'):.0f} | "
            f"{row.get('knowledge_retention'):.2f} | {row.get('goals_completed'):.0f} |"
        )
    lines.append("")
    lines.append("## 失败案例\n")
    failures = [row for row in rows if row.get("extinct")]
    if not failures:
        lines.append("本批次没有运行灭绝。")
    else:
        for row in failures:
            lines.append(
                f"- group {row['group']} seed {row['seed']}：在 {row.get('extinction_time'):.0f}s 灭绝，"
                f"累计创建 {row.get('total_agents_created'):.0f}，验证技术 {row.get('tech_verified'):.0f}。"
            )
    lines.append("")
    lines.append("## 复现\n")
    lines.append("```powershell")
    lines.append("cd newworldpro")
    lines.append("$env:TEMP = (Resolve-Path .tmp).Path")
    lines.append("python experiments/run_ablation.py --seeds 0 1 2 --steps 24000 --dt 0.05")
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--perception", action="store_true")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else ROOT / "experiments" / "results" / f"ablation_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_dict = {
        "groups": ",".join(args.groups),
        "seeds": ",".join(str(seed) for seed in args.seeds),
        "steps": args.steps,
        "dt": args.dt,
        "size": f"{args.width}x{args.height}",
        "perception": args.perception,
        "max_agents_created": 8,
        "model": "MLP 24->40 / 64 / 32 / 58 (behaviour cloning, no reward)",
        "world_version": __import__("newworld").WORLD_VERSION,
    }
    rows: list[dict[str, object]] = []
    aggregates: dict[str, dict[str, dict[str, float]]] = {}
    for group in args.groups:
        for seed in args.seeds:
            spec = RunSpec(
                group=group,
                seed=seed,
                steps=args.steps,
                dt=args.dt,
                width=args.width,
                height=args.height,
                perception=args.perception,
                log_dir=out_dir,
            )
            runner = SocietyRunner(spec)
            result = runner.run()
            row = _row(group, seed, result)
            rows.append(row)
            summary_path = out_dir / f"summary_{group}_seed{seed}.json"
            summary_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            print(
                f"[{group} seed {seed}] t={result['world_time_s']}s tech={row['tech_verified']} "
                f"depth={row['tech_graph_depth']} extinct={row['extinct']} "
                f"retention={row['knowledge_retention']:.2f}"
            )
        aggregates[group] = _aggregate(rows, group)

    _write_csv(out_dir / "runs.csv", rows)
    (out_dir / "aggregate.json").write_text(
        json.dumps(aggregates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out_dir / "report.md", rows, aggregates, spec_dict)
    print(f"\n结果目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
