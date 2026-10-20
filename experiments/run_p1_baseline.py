#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1 基线对照实验：学习策略 vs 随机 vs 规则（配对 seed）。

设计（docs/RESEARCH_ROADMAP.md §P1）：
=====================================
同一 MapEngine 地图、同 seed、同资源预算下四组配对对照：

1. ``random``：均匀随机 5 动作（下界基线）；
2. ``rule``：RuleSwarmPolicy（V3 原规则，含梯度探测+wander）；
3. ``untrained``：随机初始化的 NeuralSwarmPolicy（训练前）；
4. ``trained``：在**训练 seed** 上训练后的 NeuralSwarmPolicy。

指标（每项计算定义见 ROADMAP §P1）：
累计奖励 / 进食次数 / 进食效率 / 平均寿命 / 90s 存活率 / 种群规模。

验收门槛（预注册）：trained 在 ≥8/10 个**训练外** seed 上，
进食效率或平均寿命配对优于 rule。达不到则如实报告差距。

用法::

    python experiments/run_p1_baseline.py                    # 默认完整实验
    python experiments/run_p1_baseline.py --quick             # 冒烟（少量）
    python experiments/run_p1_baseline.py --train-episodes 20 --eval-seeds 10

输出：``experiments/results/p1_baseline/``（summary.json + report.md）。
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
from newworld.swarm_policy import (  # noqa: E402
    N_ACTIONS,
    NeuralSwarmPolicy,
    RuleSwarmPolicy,
    SwarmTrainer,
)

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "p1_baseline"

#: 地图与仿真参数（四组共享，配对实验的关键）
MAP_W, MAP_H = 256, 160
DT = 0.05
EVAL_SECONDS = 120.0          # 评估时长（世界秒）
TRAIN_SECONDS = 60.0          # 每训练回合时长
INITIAL_POP = 120
CAPACITY_HARD = 3000
#: P1.7：稀缺环境（食物再生 0.012 → 0.0005）。P1.6 发现默认环境
#: 承载过剩（食物再生 ≈ 消耗的 10 倍），任何活法都饿不死 → 无觅食
#: 选择压力。稀缺下站桩饿死（饱足 0.14、193 饿死）、移动觅食存活
#: ——这才是"觅食策略可以被学习优选"的环境条件。
FOOD_REGROW_PER_S = 0.0005
#: P1.6：竞争压力档位（多组对照——低密度站桩占优是环境事实，
#: 高密度下觅食移动的价值应显现：3× 与 6× 初始人口两档）
DENSITY_LEVELS: tuple[tuple[str, int], ...] = (
    ("normal", 120),
    ("dense3x", 360),
    ("dense6x", 720),
)


# ---------------------------------------------------------------------------
class RandomPolicy:
    """均匀随机基线（无状态）。"""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs_batch: np.ndarray) -> np.ndarray:
        return self.rng.integers(0, N_ACTIONS, size=obs_batch.shape[0])


# ---------------------------------------------------------------------------
def make_map_config(seed: int) -> MapConfig:
    """实验统一地图配置（含 P1.7 稀缺参数）。"""
    return dataclasses.replace(
        MapConfig(seed=seed, width=MAP_W, height=MAP_H),
        food_regrow_per_s=FOOD_REGROW_PER_S,
    )


def build(engine_seed: int, policy, swarm_seed: int, population: int = INITIAL_POP) -> tuple[MapEngine, Swarm]:
    """构造一个 (engine, swarm) 对；注入给定策略。"""
    engine = MapEngine(make_map_config(engine_seed))
    swarm = Swarm(
        engine,
        SwarmConfig(initial_population=population, capacity_hard=CAPACITY_HARD),
        seed=swarm_seed,
    )
    if policy is not None:
        swarm.attach_policy(policy)
    return engine, swarm


def evaluate(policy, engine_seed: int, swarm_seed: int,
             population: int = INITIAL_POP) -> dict[str, float]:
    """在给定 seed 下评估一个策略；返回指标字典。"""
    engine, swarm = build(engine_seed, policy, swarm_seed, population)
    steps = int(EVAL_SECONDS / DT)
    for _ in range(steps):
        engine.update(DT, 0.8)
        swarm.update(DT)
        if swarm.population == 0:
            break
    survived_seconds = engine.t
    # P1.6：精确平均寿命（Swarm 逐个体记录 death_ages）；
    # 无人死亡 = 全员活过评估（用存活均值，存活个体 age 为真实年龄）
    if swarm.death_ages:
        mean_life = float(np.mean(swarm.death_ages))
    else:
        alive_view = swarm.data[: swarm.count]
        alive_ages = alive_view["age"][alive_view["alive"]] if swarm.count else np.array([0.0])
        mean_life = float(alive_ages.mean()) if alive_ages.size else 0.0
    # 存活个体的期望寿命下界 = 当前平均年龄 + 剩余期望（存活=好）
    return {
        "population": float(swarm.population),
        "births": float(swarm.births),
        "deaths": float(swarm.deaths),
        "eat_events": float(swarm.eat_events),
        "total_reward": float(swarm.total_reward),
        # 进食效率：进食次数 / (存活个体·秒)
        "eat_rate": float(swarm.eat_events / max(1.0, survived_seconds * INITIAL_POP)),
        # P1.6：精确平均寿命（死者的真实年龄；无死亡=存活者当前年龄）
        "mean_life": mean_life,
        "extinct_seconds": float(survived_seconds if swarm.population == 0 else -1.0),
        "world_seconds": float(survived_seconds),
    }


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1 基线对照实验")
    parser.add_argument("--quick", action="store_true", help="冒烟模式（少 seed）")
    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--train-seeds", type=int, default=6)
    parser.add_argument("--eval-seeds", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)

    if args.quick:
        args.train_episodes, args.train_seeds, args.eval_seeds = 3, 2, 3

    train_seed_list = [500 + i for i in range(args.train_seeds)]
    eval_seed_list = [900 + i for i in range(args.eval_seeds)]  # 与训练不重叠
    print(f"P1 基线对照：训练 {args.train_episodes} 回合 × seeds {train_seed_list}")
    print(f"评估 seeds（训练外）：{eval_seed_list}")

    # -- 训练 -----------------------------------------------------------
    policy = NeuralSwarmPolicy(seed=0)
    trainer = SwarmTrainer(policy, lr=args.lr, batch_size=512,
                           epochs_per_episode=2, grad_clip=0.5)
    train_log = []
    started = time.perf_counter()
    for episode in range(args.train_episodes):
        seed = train_seed_list[episode % len(train_seed_list)]
        engine, swarm = build(seed, policy, seed)
        for _ in range(int(TRAIN_SECONDS / DT)):
            engine.update(DT, 0.8)
            swarm.update(DT)
        transitions = swarm.drain_swarm_transitions()
        stats = trainer.train_on_transitions(transitions)
        stats["episode"] = float(episode + 1)
        stats["train_seed"] = float(seed)
        train_log.append(stats)
        print(f"  ep{episode + 1:2d} seed={seed} trans={stats['transitions']:.0f} "
              f"mean_return={stats['mean_return']:+.2f} Δw={stats['weight_delta']:.2e}")
    wall_train = time.perf_counter() - started

    # -- 评估（四组 × 全部 eval seeds × 密度档位，配对） -----------------
    def make_policies():
        return {
            "random": RandomPolicy(seed=1),
            "rule": RuleSwarmPolicy(seed=1),
            "untrained": NeuralSwarmPolicy(seed=0),  # 同 seed=0 未训练
            "trained": policy,
        }

    #: 密度档位结果 {density: {policy_name: [runs]}}
    dense_results: dict[str, dict[str, list[dict[str, float]]]] = {}
    started = time.perf_counter()
    for density_name, population in DENSITY_LEVELS:
        results: dict[str, list[dict[str, float]]] = {"random": [], "rule": [],
                                                      "untrained": [], "trained": []}
        for eval_seed in eval_seed_list:
            for name, pol in make_policies().items():
                metrics = evaluate(pol, eval_seed, eval_seed, population)
                metrics["seed"] = float(eval_seed)
                results[name].append(metrics)
        dense_results[density_name] = results
        row = {k: results[k][-1] for k in results}
        print(f"  [{density_name} pop={population}] seed={eval_seed_list[-1]}: " + "  ".join(
            f"{k}(eat={v['eat_events']:.0f},life={v['mean_life']:.1f})"
            for k, v in row.items()))
    wall_eval = time.perf_counter() - started

    # -- 汇总与配对比较（主档位 = normal） ------------------------------
    results = dense_results["normal"]

    def summarize(runs: list[dict[str, float]]) -> dict[str, float]:
        keys = [k for k in runs[0] if k != "seed"]
        return {k: float(np.mean([r[k] for r in runs])) for k in keys} | {
            f"{k}_std": float(np.std([r[k] for r in runs])) for k in keys
        }

    summary = {name: summarize(runs) for name, runs in results.items()}
    # 配对胜负（trained vs rule，逐 seed）
    wins_eat = wins_life = 0
    diffs_eat, diffs_life = [], []
    for run_t, run_r in zip(results["trained"], results["rule"]):
        de = run_t["eat_rate"] - run_r["eat_rate"]
        dl = run_t["mean_life"] - run_r["mean_life"]
        diffs_eat.append(de)
        diffs_life.append(dl)
        wins_eat += de > 0
        wins_life += dl > 0
    # 密度梯度分析：每个密度档位的 trained-vs-rule 进食效率差
    density_analysis = {}
    for density_name, runs in dense_results.items():
        deats, dlifes = [], []
        de_wins = dl_wins = 0
        for run_t, run_r in zip(runs["trained"], runs["rule"]):
            de = run_t["eat_rate"] - run_r["eat_rate"]
            dl = run_t["mean_life"] - run_r["mean_life"]
            deats.append(de)
            dlifes.append(dl)
            de_wins += de > 0
            dl_wins += dl > 0
        density_analysis[density_name] = {
            "eat_rate_diff_mean": float(np.mean(deats)),
            "eat_rate_diff_std": float(np.std(deats)),
            "eat_wins": int(de_wins),
            "mean_life_diff_mean": float(np.mean(dlifes)),
            "life_wins": int(dl_wins),
            "trained_mean_life": float(np.mean([r["mean_life"] for r in runs["trained"]])),
            "rule_mean_life": float(np.mean([r["mean_life"] for r in runs["rule"]])),
        }
    paired = {
        "trained_vs_rule_eat_wins": wins_eat,
        "trained_vs_rule_life_wins": wins_life,
        "n_eval_seeds": len(eval_seed_list),
        "eat_rate_diff_mean": float(np.mean(diffs_eat)),
        "eat_rate_diff_std": float(np.std(diffs_eat)),
        "mean_life_diff_mean": float(np.mean(diffs_life)),
        "mean_life_diff_std": float(np.std(diffs_life)),
        "gate_passed": bool(wins_eat >= 8 or wins_life >= 8),
        "density_analysis": density_analysis,
        "density_sweep_hypothesis": (
            "低密度下站桩吃再生流占优是环境事实；高密度（竞争压力）下"
            "扫掠觅食应反超——若 dense6x 的 eat_rate_diff 显著高于 normal，"
            "则习得的扫掠策略在资源竞争中具有真实适应价值"
        ),
        "train_wall_s": round(wall_train, 1),
        "eval_wall_s": round(wall_eval, 1),
        "train_log": train_log,
        "train_seeds": train_seed_list,
        "eval_seeds": eval_seed_list,
        "params": {
            "map": f"{MAP_W}x{MAP_H}", "dt": DT, "eval_s": EVAL_SECONDS,
            "train_s_per_ep": TRAIN_SECONDS, "initial_pop": INITIAL_POP,
            "density_levels": {k: v for k, v in DENSITY_LEVELS},
            "lr": args.lr, "episodes": args.train_episodes,
        },
    }

    # -- 写盘 -----------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"e{args.train_episodes}_s{args.eval_seeds}"
    out_json = RESULTS_DIR / f"summary_{tag}.json"
    out_json.write_text(
        json.dumps({"summary": summary, "paired": paired}, ensure_ascii=False,
                   indent=2, default=str),
        encoding="utf-8",
    )
    # 简报
    lines = [
        "# P1 基线对照实验结果", "",
        f"- 训练：{args.train_episodes} 回合（seeds {train_seed_list}），"
        f"lr={args.lr}，minibatch=512×2epoch",
        f"- 评估：{len(eval_seed_list)} 个训练外 seed × {EVAL_SECONDS:.0f} 世界秒，配对四组", "",
        "| 组 | 进食次数 | 进食效率(1/s·人) | 平均寿命(s) | 累计奖励 | 末种群 |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("random", "rule", "untrained", "trained"):
        s = summary[name]
        lines.append(
            f"| {name} | {s['eat_events']:.0f}±{s['eat_events_std']:.0f} "
            f"| {s['eat_rate']:.5f}±{s['eat_rate_std']:.5f} "
            f"| {s['mean_life']:.1f}±{s['mean_life_std']:.1f} "
            f"| {s['total_reward']:+.0f}±{s['total_reward_std']:.0f} "
            f"| {s['population']:.0f}±{s['population_std']:.0f} |"
        )
    lines += [
        "", "## 配对结果（trained vs rule，逐 seed）",
        f"- 进食效率胜 {wins_eat}/{len(eval_seed_list)}，"
        f"Δ = {paired['eat_rate_diff_mean']:+.5f} ± {paired['eat_rate_diff_std']:.5f}",
        f"- 平均寿命胜 {wins_life}/{len(eval_seed_list)}，"
        f"Δ = {paired['mean_life_diff_mean']:+.1f} ± {paired['mean_life_diff_std']:.1f}",
        f"- **验收门槛（≥8/{len(eval_seed_list)}）：{'通过' if paired['gate_passed'] else '未通过'}**",
        "",
        "## 密度梯度（P1.6：竞争压力下习得策略的价值检验）",
        "",
        "| 密度 | 初始人口 | trained-rule 进食效率差 | 进食胜 | 寿命差 | 寿命胜 |",
        "|---|---|---|---|---|---|",
    ]
    for density_name, pop in DENSITY_LEVELS:
        da = density_analysis[density_name]
        lines.append(
            f"| {density_name} | {pop} | {da['eat_rate_diff_mean']:+.5f}±{da['eat_rate_diff_std']:.5f} "
            f"| {da['eat_wins']}/{len(eval_seed_list)} "
            f"| {da['mean_life_diff_mean']:+.1f} "
            f"| {da['life_wins']}/{len(eval_seed_list)} |"
        )
    lines += [
        "",
        "指标定义（P1.6 修订）：mean_life 为逐个体记录的死亡年龄均值"
        "（Swarm.death_ages；无死亡时用存活者当前年龄）；"
        "进食效率分母用初始人口（四组同口径，配对比较有效）。",
    ]
    out_md = RESULTS_DIR / f"report_{tag}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果: {out_json}")
    print(f"简报: {out_md}")
    print(f"门槛（trained ≥8/{len(eval_seed_list)} 胜 rule）: "
          f"{'通过 ✓' if paired['gate_passed'] else '未通过 ✗'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
