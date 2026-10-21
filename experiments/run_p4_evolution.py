"""P4 技能接线 + 演化 vs 学习对照实验。

设计（RESEARCH_ROADMAP §P4 + 2026-10 调研纪要）
=================================================
**问题**：适应的来源对照——"演化（变异+选择，Avida/OpenAI-ES 精神）"
与"个体学习（策略梯度，P1.7 路径）"哪条路径在同环境下
带来的觅食适应增益更大、来得更快？

**方法**（三臂配对，稀缺环境）：
1. ``rule``：规则基线（无适应）；
2. ``evolution``：种群级 ES——把"技能组合概率向量"当作可变异
   参数：每代按携带技能个体的存活表现加权，更新种群技能分布
   （变异 σ + 选择压力都来自真实死亡率，不是硬编码剧本）；
3. ``learning``：P1.7 的 NeuralSwarmPolicy 策略梯度训练。

**指标**：进食效率（配对逐 seed）、技能携带率（演化臂）、
训练后动作分布变化（学习臂）。

这是"机制验证"性质的 P4 第一阶段：演化臂用轻量分布演化
（与 OpenAI ES 同构：只通信标量适应度），学习臂复用 P1.7 全套。
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
    NeuralSwarmPolicy,
    RuleSwarmPolicy,
    SwarmTrainer,
)
from newworld.social_mech import SKILLS, SocialConfig, SocialLayer  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "p4_evolution"

MAP_W, MAP_H = 256, 160
DT = 0.05
RUN_SECONDS = 120.0
GEN_SECONDS = 60.0
SCARCE_REGROW = 0.0005
INITIAL_POP = 120


def build(seed: int, policy, skill_config: SocialConfig | None):
    """构造 (engine, swarm, social)——social 接线技能效果。"""
    map_config = dataclasses.replace(
        MapConfig(seed=seed, width=MAP_W, height=MAP_H),
        food_regrow_per_s=SCARCE_REGROW,
    )
    engine = MapEngine(map_config)
    swarm = Swarm(
        engine,
        SwarmConfig(initial_population=INITIAL_POP, capacity_hard=3000),
        seed=seed,
    )
    if policy is not None:
        swarm.attach_policy(policy)
    social = None
    if skill_config is not None:
        social = SocialLayer(swarm, skill_config, seed=seed)
        social.attach_to_swarm()  # 技能效果接线到世界规则
    return engine, swarm, social


def run_episode(engine, swarm, social, seconds) -> dict[str, float]:
    """跑一个回合；返回指标（不训练任何东西）。"""
    for _ in range(int(seconds / DT)):
        engine.update(DT, 0.8)
        before = len(swarm.death_ages)
        swarm.update(DT)
        if social is not None:
            social.on_death_pass(before)
            social.update(DT)
    return {
        "population": float(swarm.population),
        "eats": float(swarm.eat_events),
        "satiation": float(swarm.mean_hunger()),
    }


# ---------------------------------------------------------------------------
# 臂 2：演化（技能分布的变异 + 选择）
# ---------------------------------------------------------------------------
class SkillDistribution:
    """种群技能分布 θ（3 维 simplex 上的软参数）+ 变异更新。

    与 OpenAI ES 同构的"只通信标量适应度"路径：
    * 技能不是个体的硬属性，而是**出生时从 θ 抽样**——θ 是
      "文明知道怎么获得的技能倾向"；
    * 每代结束：按存活技能个体的表现（饱足度加权）估计每种技能
      的适应度优势，向优势技能移动 θ（选择），并加高斯扰动
      （变异 σ）保持探索——灭绝/垄断的恶性循环会被 σ 打断。
    """

    def __init__(self, seed: int = 0, sigma: float = 0.05, lr: float = 0.3) -> None:
        self.theta = np.full(len(SKILLS), 1.0 / len(SKILLS))
        self.rng = np.random.default_rng(seed)
        self.sigma = float(sigma)
        self.lr = float(lr)
        self.history: list[list[float]] = []

    def sample_skills(self, swarm: Swarm, social: SocialLayer) -> None:
        """出生后按 θ 给新个体抽技能（应用到无技能个体）。"""
        view = swarm.data[: swarm.count]
        alive = np.nonzero(view["alive"])[0]
        unskilled = alive[social.skills[alive] == 0]
        if not unskilled.size:
            return
        picks = self.rng.choice(len(SKILLS), size=unskilled.size, p=self.theta)
        for row, pick in zip(unskilled.tolist(), picks.tolist()):
            social.skills[row] = pick + 1

    def generation_update(self, swarm: Swarm, social: SocialLayer) -> dict[str, float]:
        """一代结束的选择更新：θ 向存活优势技能移动。"""
        view = swarm.data[: swarm.count]
        alive = np.nonzero(view["alive"])[0]
        if not alive.size:
            return {"divergence": 0.0}
        hungers = view["hunger"][alive]
        skills = social.skills[alive]
        # 每种技能的平均饱足（= 适应度代理；无技能个体作基线）
        base_sat = float(hungers[skills == 0].mean()) if (skills == 0).any() else 0.5
        advantage = np.zeros(len(SKILLS))
        for k in range(1, len(SKILLS) + 1):
            mask = skills == k
            if mask.any():
                advantage[k - 1] = float(hungers[mask].mean()) - base_sat
        # softmax(advantage/τ) 作为目标分布的倾向
        temperature = 0.05
        exponent = np.clip(advantage / temperature, -10, 10)
        target = np.exp(exponent - exponent.max())
        target /= target.sum()
        # 变异 + 选择：θ ← (1-lr)·θ + lr·target + N(0, σ)
        self.theta = (1 - self.lr) * self.theta + self.lr * target
        self.theta += self.rng.normal(0.0, self.sigma, self.theta.size)
        self.theta = np.clip(self.theta, 1e-3, None)
        self.theta /= self.theta.sum()
        self.history.append(self.theta.tolist())
        return {"divergence": float(np.abs(advantage).max())}


# ---------------------------------------------------------------------------
# 臂 3：学习（P1.7 策略梯度，照搬）
# ---------------------------------------------------------------------------
def run_learning_arm(train_seeds: list[int], eval_seed: int,
                     episodes: int, lr: float) -> tuple[NeuralSwarmPolicy, list[dict]]:
    """训练神经策略（稀缺环境）；返回 (策略, 训练日志)。"""
    policy = NeuralSwarmPolicy(seed=0)
    trainer = SwarmTrainer(policy, lr=lr, batch_size=512,
                           epochs_per_episode=2, grad_clip=0.5)
    log = []
    for episode in range(episodes):
        seed = train_seeds[episode % len(train_seeds)]
        engine, swarm, _ = build(seed, policy, None)
        for _ in range(int(GEN_SECONDS / DT)):
            engine.update(DT, 0.8)
            swarm.update(DT)
        stats = trainer.train_on_transitions(swarm.drain_swarm_transitions())
        stats["episode"] = float(episode + 1)
        log.append(stats)
    return policy, log


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4 演化 vs 学习对照")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds, args.generations, args.episodes = 2, 2, 3

    eval_seeds = [1100 + i for i in range(args.seeds)]
    print(f"P4 三臂对照：rule / evolution({args.generations} 代) / "
          f"learning({args.episodes} 回合) × {args.seeds} eval seeds")

    results: dict[str, list[dict]] = {"rule": [], "evolution": [], "learning": []}
    started = time.perf_counter()

    # 训练两个适应臂（每 seed 独立训练避免信息泄漏——配对原则）
    for seed in eval_seeds:
        # 1) rule 基线
        engine, swarm, _ = build(seed, RuleSwarmPolicy(seed=seed), None)
        metrics = run_episode(engine, swarm, None, RUN_SECONDS)
        metrics["seed"] = float(seed)
        results["rule"].append(metrics)

        # 2) evolution 臂：技能分布演化 N 代
        social_cfg = SocialConfig(enable_barter=False, enable_teaching=False,
                                  enable_inheritance=False, initial_skill_rate=0.0)
        engine, swarm, social = build(seed, RuleSwarmPolicy(seed=seed), social_cfg)
        dist = SkillDistribution(seed=seed)
        # 第 0 代先注入初始种群
        social.inject_initial_skills()
        gen_log = []
        for gen in range(args.generations):
            # 该代的个体在其一生中接受技能分配（新生个体）
            run_episode(engine, swarm, social, GEN_SECONDS)
            dist.sample_skills(swarm, social)   # 新生个体按 θ 抽技能
            dist.generation_update(swarm, social)
            gen_log.append(dist.theta.tolist())
            if gen == args.generations - 1:
                pass
        # 评估最终一代（θ 冻结）
        engine2, swarm2, social2 = build(seed, RuleSwarmPolicy(seed=seed), social_cfg)
        social2.inject_initial_skills()
        # 应用最终 θ：直接给个体重抽技能
        alive = np.nonzero(swarm2.data[: swarm2.count]["alive"])[0]
        picks = dist.rng.choice(len(SKILLS), size=alive.size, p=dist.theta)
        for row, pick in zip(alive.tolist(), picks.tolist()):
            social2.skills[row] = pick + 1
        social2.attach_to_swarm()
        metrics = run_episode(engine2, swarm2, social2, RUN_SECONDS)
        metrics["seed"] = float(seed)
        metrics["theta_final"] = dist.theta.tolist()
        results["evolution"].append(metrics)

        # 3) learning 臂（独立训练，防泄漏）
        policy, log = run_learning_arm([seed], seed, args.episodes, lr=0.02)
        engine3, swarm3, _ = build(seed, policy, None)
        metrics = run_episode(engine3, swarm3, None, RUN_SECONDS)
        metrics["seed"] = float(seed)
        metrics["weight_delta_last"] = log[-1]["weight_delta"] if log else 0.0
        results["learning"].append(metrics)

        print(f"  seed={seed}: rule(eats={results['rule'][-1]['eats']:.0f}) "
              f"evo(eats={results['evolution'][-1]['eats']:.0f},"
              f"θ={np.round(dist.theta, 2).tolist()}) "
              f"learn(eats={results['learning'][-1]['eats']:.0f})")
    wall = time.perf_counter() - started

    # 汇总 + 配对
    def summarize(runs):
        keys = [k for k in runs[0] if k not in ("seed", "theta_final")]
        return {k: float(np.mean([r[k] for r in runs])) for k in keys}

    summary = {name: summarize(runs) for name, runs in results.items()}
    paired = {}
    for arm in ("evolution", "learning"):
        de = [r["eats"] - b["eats"] for r, b in zip(results[arm], results["rule"])]
        wins = sum(d > 0 for d in de)
        paired[arm] = {
            "eat_diff_mean": float(np.mean(de)),
            "eat_diff_std": float(np.std(de)),
            "wins_vs_rule": int(wins),
            "n": len(de),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"g{args.generations}_e{args.episodes}_s{args.seeds}"
    out = {"summary": summary, "paired": paired, "wall_s": round(wall, 1),
           "eval_seeds": eval_seeds, "raw": results}
    (RESULTS_DIR / f"summary_{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# P4 演化 vs 学习对照结果", "",
        f"- 三臂 × {len(eval_seeds)} eval seeds，各 {RUN_SECONDS:.0f}s 稀缺世界",
        f"- evolution：技能分布 θ 演化 {args.generations} 代（σ=0.05, lr=0.3）",
        f"- learning：NeuralSwarmPolicy 策略梯度 {args.episodes} 回合（独立训练）", "",
        "| 臂 | 进食 | 末种群 | 饱足 | vs rule 配对 |",
        "|---|---|---|---|---|",
    ]
    for name in ("rule", "evolution", "learning"):
        s = summary[name]
        p = paired.get(name, {})
        vs = (f"{p['wins_vs_rule']}/{p['n']}，Δ={p['eat_diff_mean']:+.0f}"
              if p else "基线")
        lines.append(f"| {name} | {s['eats']:.0f} | {s['population']:.0f} "
                     f"| {s['satiation']:.2f} | {vs} |")
    lines += ["", "说明：演化臂的 θ_final 在 raw 数据中（哪种技能被偏好）。"]
    (RESULTS_DIR / f"report_{tag}.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n结果: {RESULTS_DIR / f'summary_{tag}.json'}")
    print(f"简报: {RESULTS_DIR / f'report_{tag}.md'}")
    for arm, p in paired.items():
        print(f"  {arm} vs rule: {p['wins_vs_rule']}/{p['n']} 胜，"
              f"Δeats={p['eat_diff_mean']:+.0f}±{p['eat_diff_std']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
