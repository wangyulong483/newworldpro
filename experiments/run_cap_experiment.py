"""放开人口上限的长期对照实验：cap 8 vs 64（无限世界，900 世界秒）。

目的：回答"同样的机制在人口可再生时能否产生真实的复苏/周期"。
记录人口时间序列（存活/累计创建/死亡）、技术、设施、灭绝与恢复事件。
真实运行、不挑 seed；产物写入 experiments/results/v2_cap/。

用法::

    cd newworldpro
    $env:TEMP=(Resolve-Path .tmp).Path
    python experiments/run_cap_experiment.py --seeds 300 301 302 --steps 18000 --cap 64
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from newworld.simulation import RunSpec, SocietyRunner  # noqa: E402


def run_one(seed: int, cap: int, steps: int, dt: float, sample_every: int = 100) -> dict:
    """跑一个长期 run，记录人口时间序列。"""
    runner = SocietyRunner(
        RunSpec(
            group="C",
            seed=seed,
            steps=steps,
            dt=dt,
            width=640,
            height=400,
            perception=False,
            disturbance=False,
            infinite=True,
            window_atoms=128,
            max_agents_created=cap,
            visual_features=False,
            perception_stride=8,
        )
    )
    world = runner.world
    population = runner.population
    started = time.perf_counter()
    series: list[dict] = []
    result = None
    for step in range(steps):
        population.step(dt)
        if step % sample_every == 0 or step == steps - 1:
            alive = len([a for a in world.agents if a.alive])
            series.append(
                {
                    "t": round(float(world.t), 2),
                    "alive": alive,
                    "created": int(world.registry.total_created),
                    "deaths": int(world.registry.total_deaths),
                    "tech": int(len(world.knowledge.verified_ids())),
                    "facilities": int(
                        len(
                            [
                                f
                                for f in world.civilization.facilities
                                if getattr(f, "active", False)
                            ]
                        )
                    ),
                    "cooperation": round(float(getattr(world.civilization, "cooperation_index", 0.0)), 4),
                }
            )
        if world.civilization.extinct:
            break
    wall = time.perf_counter() - started
    result = {
        "seed": seed,
        "cap": cap,
        "steps_requested": steps,
        "steps_run": len(series) * sample_every if series else 0,
        "world_time_s": round(float(world.t), 2),
        "wall_s": round(wall, 1),
        "created": int(world.registry.total_created),
        "deaths": int(world.registry.total_deaths),
        "alive_end": len([a for a in world.agents if a.alive]),
        "extinct": bool(world.civilization.extinct),
        "extinction_cause": str(getattr(world.civilization, "extinction_cause", "") or ""),
        "tech": int(len(world.knowledge.verified_ids())),
        "facilities": int(
            len([f for f in world.civilization.facilities if getattr(f, "active", False)])
        ),
        "alive_min": min((point["alive"] for point in series), default=0),
        "alive_max": max((point["alive"] for point in series), default=0),
        "recoveries": sum(
            1
            for previous, current in zip(series, series[1:])
            if previous["alive"] > 0 and current["alive"] > previous["alive"]
        ),
        "series": series,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="population cap experiment")
    parser.add_argument("--seeds", type=int, nargs="+", default=[300, 301, 302])
    parser.add_argument("--steps", type=int, default=18000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--caps", type=int, nargs="+", default=[8, 64])
    parser.add_argument("--tag", default="v2_cap")
    args = parser.parse_args(argv)
    out = ROOT / "experiments" / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for cap in args.caps:
        for seed in args.seeds:
            result = run_one(seed, cap, args.steps, args.dt)
            rows.append(result)
            print(
                f"[cap {cap} seed {seed}] t={result['world_time_s']}s "
                f"created={result['created']} deaths={result['deaths']} "
                f"alive_end={result['alive_end']} alive {result['alive_min']}..{result['alive_max']} "
                f"recoveries={result['recoveries']} tech={result['tech']} "
                f"fac={result['facilities']} extinct={result['extinct']} "
                f"wall={result['wall_s']}s"
            )
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[key for key in rows[0].keys() if key != "series"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "series"})
    (out / "series.json").write_text(
        json.dumps([{"seed": r["seed"], "cap": r["cap"], "series": r["series"]} for r in rows],
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果目录：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
