"""P3 社会机制配对对照实验：交换 / 教学 / 继承 逐机制消融。

设计（docs/RESEARCH_ROADMAP.md §P3）
=====================================
同地图、同 seed、同资源预算下的**配对对照**：每 seed 跑
"全关（对照）/ 仅机制 X / 全开" 六组（实际五个配置 + 全关基线），
逐 seed 比较机制对社会指标的影响。

假设（预注册）
--------------
H-barter：交换降低种群内饱足分化（方差 ↓）；
H-teach：教学提升技能人口 → 觅食效率上升（技能真实生效）；
H-inherit：继承降低技能失传率 → 知识跨代保留。

验收门槛：目标机制的直接指标（交换数/教学数/继承数）在全关
组为 0、在开启组 >0（机制有效性）；机制对社会指标的影响报告
配对差值与均值±标准差（不做 8/10 门槛——P3 是机制验证阶段，
效应量报告优先于显著性判断）。

用法::

    python experiments/run_p3_social.py --quick       # 冒烟
    python experiments/run_p3_social.py --seeds 10     # 完整
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from newworld.map_engine import MapConfig, MapEngine  # noqa: E402
from newworld.swarm import Swarm, SwarmConfig  # noqa: E402
from newworld.social_mech import SocialConfig, SocialLayer  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "p3_social"

MAP_W, MAP_H = 256, 160
DT = 0.05
RUN_SECONDS = 120.0
SCARCE_REGROW = 0.0005

#: 五个实验配置（对照 + 三机制独立 + 全开）
CONFIGS: dict[str, dict[str, bool]] = {
    "off_all": {"enable_barter": False, "enable_teaching": False, "enable_inheritance": False},
    "barter_only": {"enable_barter": True, "enable_teaching": False, "enable_inheritance": False},
    "teach_only": {"enable_barter": False, "enable_teaching": True, "enable_inheritance": False},
    "inherit_only": {"enable_barter": False, "enable_teaching": False, "enable_inheritance": True},
    "all_on": {"enable_barter": True, "enable_teaching": True, "enable_inheritance": True},
}


def run_one(config_name: str, flags: dict[str, bool], seed: int) -> dict[str, float]:
    """跑一个 (config, seed) 组合；返回指标。"""
    map_config = dataclasses.replace(
        MapConfig(seed=seed, width=MAP_W, height=MAP_H),
        food_regrow_per_s=SCARCE_REGROW,
    )
    engine = MapEngine(map_config)
    swarm = Swarm(
        engine,
        SwarmConfig(initial_population=120, capacity_hard=3000),
        seed=seed,
    )
    social = SocialLayer(swarm, SocialConfig(**flags), seed=seed)
    social.inject_initial_skills()
    for _ in range(int(RUN_SECONDS / DT)):
        deaths_before = len(swarm.death_ages)
        engine.update(DT, 0.8)
        swarm.update(DT)
        social.on_death_pass(deaths_before)
        social.update(DT)
    # 饱足分化（H-barter 的因变量）
    view = swarm.data[: swarm.count]
    alive = view["alive"]
    hungers = view["hunger"][alive]
    s = social.stats
    return {
        "population": float(swarm.population),
        "births": float(swarm.births),
        "deaths": float(swarm.deaths),
        "eat_events": float(swarm.eat_events),
        "hunger_mean": float(hungers.mean()) if hungers.size else 0.0,
        "hunger_std": float(hungers.std()) if hungers.size else 0.0,
        "barter_events": float(s["barter_events"]),
        "barter_total": float(s["barter_transfer_total"]),
        "teach_events": float(s["teach_events"]),
        "inherit_events": float(s["inherit_events"]),
        "knowledge_lost": float(s["knowledge_lost"]),
        "skilled_population": float(s["skilled_population"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P3 社会机制配对对照实验")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=700)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds = 3

    seed_list = [args.seed_start + i for i in range(args.seeds)]
    print(f"P3 社会机制消融：{len(CONFIGS)} 配置 × {len(seed_list)} seeds（配对）")

    results: dict[str, list[dict[str, float]]] = {name: [] for name in CONFIGS}
    started = time.perf_counter()
    for seed in seed_list:
        for name, flags in CONFIGS.items():
            metrics = run_one(name, flags, seed)
            metrics["seed"] = float(seed)
            results[name].append(metrics)
        print(f"  seed={seed}: " + "  ".join(
            f"{n}(pop={results[n][-1]['population']:.0f},"
            f"skill={results[n][-1]['skilled_population']:.0f})"
            for n in CONFIGS))
    wall = time.perf_counter() - started

    # 汇总
    def summarize(runs: list[dict[str, float]]) -> dict[str, float]:
        keys = [k for k in runs[0] if k != "seed"]
        return {k: float(np.mean([r[k] for r in runs])) for k in keys} | {
            f"{k}_std": float(np.std([r[k] for r in runs])) for k in keys
        }

    summary = {name: summarize(runs) for name, runs in results.items()}

    # 配对差值（各机制组 vs 全关对照）
    paired = {}
    for name in ("barter_only", "teach_only", "inherit_only", "all_on"):
        deltas = {
            "hunger_std": [],
            "skilled_population": [],
            "eat_events": [],
            "population": [],
        }
        for run_t, run_c in zip(results[name], results["off_all"]):
            deltas["hunger_std"].append(run_t["hunger_std"] - run_c["hunger_std"])
            deltas["skilled_population"].append(run_t["skilled_population"] - run_c["skilled_population"])
            deltas["eat_events"].append(run_t["eat_events"] - run_c["eat_events"])
            deltas["population"].append(run_t["population"] - run_c["population"])
        paired[name] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in deltas.items()
        }

    # 机制有效性检查（预注册门槛：全关=0，开启>0）
    validity = {
        "barter_works": summary["barter_only"]["barter_events"] > 0
                        and summary["off_all"]["barter_events"] == 0,
        "teach_works": summary["teach_only"]["teach_events"] > 0
                       and summary["off_all"]["teach_events"] == 0,
        "inherit_works": summary["inherit_only"]["inherit_events"] > 0
                         and summary["off_all"]["inherit_events"] == 0,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"s{args.seeds}"
    out = {
        "configs": CONFIGS, "seeds": seed_list,
        "summary": summary, "paired_vs_control": paired,
        "validity": validity, "wall_s": round(wall, 1),
        "run_seconds": RUN_SECONDS,
    }
    (RESULTS_DIR / f"summary_{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 简报
    lines = [
        "# P3 社会机制消融结果", "",
        f"- {len(seed_list)} seeds × {len(CONFIGS)} 配置，各 {RUN_SECONDS:.0f}s 稀缺世界，配对",
        f"- 机制有效性（预注册）：barter={'✓' if validity['barter_works'] else '✗'}"
        f" teach={'✓' if validity['teach_works'] else '✗'}"
        f" inherit={'✓' if validity['inherit_works'] else '✗'}", "",
        "| 配置 | 交换 | 教学 | 继承 | 失传 | 技能人口 | 饱足σ | 末人口 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in CONFIGS:
        s = summary[name]
        lines.append(
            f"| {name} | {s['barter_events']:.0f} | {s['teach_events']:.0f} "
            f"| {s['inherit_events']:.0f} | {s['knowledge_lost']:.0f} "
            f"| {s['skilled_population']:.1f}±{s['skilled_population_std']:.1f} "
            f"| {s['hunger_std']:.3f}±{s['hunger_std_std']:.3f} "
            f"| {s['population']:.0f}±{s['population_std']:.0f} |"
        )
    lines += ["", "## 配对差值 vs 全开对照（逐 seed）"]
    for name, deltas in paired.items():
        lines.append(f"- **{name}**：技能人口 Δ={deltas['skilled_population']['mean']:+.1f}"
                     f"±{deltas['skilled_population']['std']:.1f}，"
                     f"饱足σ Δ={deltas['hunger_std']['mean']:+.3f}"
                     f"±{deltas['hunger_std']['std']:.3f}，"
                     f"进食 Δ={deltas['eat_events']['mean']:+.0f}")
    lines += ["", "局限：P3 为机制验证阶段，报告配对效应量；显著性判断"
              "（8/10 门槛）留给机制效应确立后的扩展实验。"]
    (RESULTS_DIR / f"report_{tag}.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n结果: {RESULTS_DIR / f'summary_{tag}.json'}")
    print(f"简报: {RESULTS_DIR / f'report_{tag}.md'}")
    print(f"有效性: {validity}  耗时 {wall:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
