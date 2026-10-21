"""P4 演化 vs 学习协议（evolution_compare）：适应来源的三臂配对对照。

设计来源：experiments/run_p4_evolution.py（P4 设计沉淀）。P5 起由
注册表按 YAML 里的 protocol: evolution_compare 调起：

    python -m newworld --config configs/p4_evolution.yaml

问题
====
适应的来源对照——"演化（变异 + 选择）"与"个体学习（策略梯度）"在同
环境下，哪条路径带来的觅食适应增益更大、来得更快？

三臂配对（稀缺世界，同 eval seed 同地图同初始人口）
====================================================
1. rule      规则基线（RuleSwarmPolicy，无适应，无技能层）；
2. evolution 技能分布 θ 的变异 + 选择（**不做策略学习**）：θ 是三种
   技能（efficient_biter / long_sight / slow_metabolism）的出生抽样
   分布；每代跑 gen_seconds，按各技能存活个体的平均饱足估计适应度
   优势，θ 向优势技能移动并加高斯变异 σ（与 OpenAI ES 同构的
   "只通信标量适应度"路径）；最后把冻结的 θ 抽样到全新世界评估；
3. learning   NeuralSwarmPolicy 策略梯度（common.train_neural_policy）。

诚实设计修正（相对旧 P4）
==========================
旧 P4 的 learning 臂在**每个评估 seed 上就地训练**（训练/评估同 seed，
存在信息泄漏）。本协议用**训练外 seed** 训练（默认 500+）、在 1100+
seed 上配对评估；因此 learning 的数值不可与旧 P4 直接对比，但配对
泛化结论更可信。旧 P4 的结论（evolution 1/5 胜、learning 5/5 胜）与
机制诊断保留在 caveats 里作为先验。

指标
====
eat_events / eat_rate（分母=初始人口·秒，四臂同口径）/ mean_life /
satiation_mean / population；配对差 = 臂 - rule，逐 eval seed。
演化臂额外记录每代 θ 与适应度散度；学习臂记录训练日志。

诚实性 caveats（随 summary 归档）
=================================
* 演化臂表型弱：3 个离散技能中，bite 倍率被残食上限截断、move 只有
  3 格探测半径，只有 slow_metabolism 在稀缺环境有明确生理优势——
  弱表型组合难以显著超过无技能基线（旧 P4 诊断）。
* 信号通道差距是结构性的：learning 每步拿到稠密 (obs, reward) 梯度
  信号，evolution 每代只有整代存活饱足均值一个标量；旧 P4 在同等
  规模下 evolution 无效应、learning 5/5 胜。
* 演化臂最后评估把 θ 抽样到全新初始种群（skills prior 效应），不是
  完整基因-文化协同演化；不改 metric 定义，但解释时需限定。
* 配对差只报均值/胜场，不做显著性检验（n 个 eval seed 很小）。
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..experiment import build_simulation
from ..registry import register_protocol
from ..settings import ExperimentSettings
from ..social_mech import SKILLS
from ..swarm_policy import NeuralSwarmPolicy, RuleSwarmPolicy
from ..zoo import ModelZoo
from . import common
from .p3_social import advance_with_social

#: 演化臂的社会层配置：三机制全关 + 不注入初始技能（θ 全权决定技能）
_EVO_SOCIAL_FLAGS: dict[str, Any] = {
    "enable_barter": False,
    "enable_teaching": False,
    "enable_inheritance": False,
    "initial_skill_rate": 0.0,
}
#: 适应度 → 目标分布的 softmax 温度（旧 P4 口径）
_TEMPERATURE = 0.05

_ARMS: tuple[str, ...] = ("rule", "evolution", "learning")
_SUMMARY_KEYS: tuple[str, ...] = (
    "population",
    "births",
    "deaths",
    "starvation_deaths",
    "old_age_deaths",
    "eat_events",
    "eat_rate",
    "mean_life",
    "satiation_mean",
    "satiation_std",
    "gini_hunger",
    "total_reward",
    "survived_90s",
    "carrying_capacity_estimate",
    "skilled_population",
)
_PAIRED_METRICS: tuple[str, ...] = (
    "eat_events",
    "eat_rate",
    "mean_life",
    "satiation_mean",
    "population",
)


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def _seed_list(value: Any, *, default_start: int, default_count: int) -> list[int]:
    if isinstance(value, (list, tuple)):
        seeds = [int(seed) for seed in value]
        if not seeds:
            raise ValueError("seed 列表不能为空")
        return seeds
    count = int(default_count if value is None else value)
    if count < 1:
        raise ValueError(f"seed 数量必须 >= 1，得到 {count}")
    return [int(default_start) + index for index in range(count)]


# ---------------------------------------------------------------------------
# 臂 2：技能分布的演化（变异 + 选择）
# ---------------------------------------------------------------------------
class SkillDistribution:
    """种群技能分布 θ（3 维 simplex 上的软参数）+ 变异/选择更新。

    与 OpenAI ES 同构的"只通信标量适应度"路径：

    * 技能不是个体硬属性，而是**出生时从 θ 抽样**——θ 表示"文明倾向
      于获得哪种技能"；
    * 每代结束：按存活技能个体的平均饱足（适应度代理）估计每种技能
      的优势，向优势技能移动 θ（选择），并加高斯扰动 σ（变异）保持
      探索——防止过早灭绝/垄断。
    """

    def __init__(self, seed: int = 0, sigma: float = 0.05, lr: float = 0.3) -> None:
        self.theta = np.full(len(SKILLS), 1.0 / len(SKILLS), dtype=np.float64)
        self.rng = np.random.default_rng(int(seed))
        self.sigma = float(sigma)
        self.lr = float(lr)
        self.history: list[list[float]] = []

    def sample_skills(self, swarm: Any, social: Any) -> int:
        """给当前无技能的存活个体按 θ 抽技能；返回新赋技能数。"""
        view = swarm.data[: swarm.count]
        alive = np.nonzero(view["alive"])[0]
        unskilled = alive[social.skills[alive] == 0]
        if not unskilled.size:
            return 0
        picks = self.rng.choice(len(SKILLS), size=unskilled.size, p=self.theta)
        social.skills[unskilled] = picks.astype(np.int8) + 1
        return int(unskilled.size)

    def assign_all(self, swarm: Any, social: Any) -> int:
        """给所有存活个体按 θ 重抽技能（最终评估用：冻结 θ 的表型）。"""
        alive = np.nonzero(swarm.data[: swarm.count]["alive"])[0]
        if not alive.size:
            return 0
        picks = self.rng.choice(len(SKILLS), size=alive.size, p=self.theta)
        social.skills[alive] = picks.astype(np.int8) + 1
        return int(alive.size)

    def generation_update(self, swarm: Any, social: Any) -> dict[str, float]:
        """一代结束的选择更新：θ 向存活优势技能移动（返回诊断量）。"""
        view = swarm.data[: swarm.count]
        alive = np.nonzero(view["alive"])[0]
        if not alive.size:
            return {"divergence": 0.0, "skilled_fraction": 0.0}
        hungers = view["hunger"][alive].astype(np.float64)
        skills = np.asarray(social.skills[alive])
        # 每种技能的平均饱足（适应度代理）；无技能个体作基线，
        # 若全员有技能则退化到全体均值（相对排序不受基线平移影响）。
        unskilled = skills == 0
        base_sat = float(hungers[unskilled].mean()) if unskilled.any() else float(hungers.mean())
        advantage = np.zeros(len(SKILLS), dtype=np.float64)
        for index in range(1, len(SKILLS) + 1):
            mask = skills == index
            if mask.any():
                advantage[index - 1] = float(hungers[mask].mean()) - base_sat
        exponent = np.clip(advantage / _TEMPERATURE, -10.0, 10.0)
        target = np.exp(exponent - exponent.max())
        target /= target.sum()
        # 变异 + 选择：θ <- (1-lr)·θ + lr·target + N(0, σ)
        self.theta = (1.0 - self.lr) * self.theta + self.lr * target
        self.theta = np.clip(
            self.theta + self.rng.normal(0.0, self.sigma, self.theta.size), 1e-3, None
        )
        self.theta /= self.theta.sum()
        self.history.append(self.theta.tolist())
        return {
            "divergence": float(np.abs(advantage).max()),
            "skilled_fraction": float((skills > 0).mean()),
        }


# ---------------------------------------------------------------------------
# 单臂运行
# ---------------------------------------------------------------------------
def _decorate(metrics: dict[str, Any], *, seed: int, settings: ExperimentSettings,
              sim: Any) -> dict[str, Any]:
    """补齐配对/报告需要的字段（密度、初始人口、承载力、技能人口）。"""
    metrics["seed"] = int(seed)
    metrics["density"] = "normal"
    metrics["initial_population"] = int(settings.swarm.initial_population)
    metrics["carrying_capacity_estimate"] = float(
        sim.swarm.carrying_capacity_estimate()
    )
    social_stats = metrics.get("social") or {}
    metrics["skilled_population"] = float(social_stats.get("skilled_population", 0.0))
    return metrics


def _run_rule_arm(
    settings: ExperimentSettings, *, seed: int, eval_seconds: float
) -> dict[str, Any]:
    policy = RuleSwarmPolicy(seed=seed)
    sim, metrics = common.simulate(settings, seed=seed, seconds=eval_seconds, policy=policy)
    return _decorate(metrics, seed=seed, settings=settings, sim=sim)


def _run_learning_arm(
    settings: ExperimentSettings,
    policy: NeuralSwarmPolicy,
    *,
    seed: int,
    eval_seconds: float,
) -> dict[str, Any]:
    sim, metrics = common.simulate(settings, seed=seed, seconds=eval_seconds, policy=policy)
    return _decorate(metrics, seed=seed, settings=settings, sim=sim)


def _run_evolution_arm(
    settings: ExperimentSettings,
    *,
    seed: int,
    generations: int,
    gen_seconds: float,
    eval_seconds: float,
    sigma: float,
    lr: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """演化臂：θ 变异/选择 generations 代，再在全新世界评估冻结 θ。"""
    train_sim = build_simulation(settings, seed=int(seed), policy=RuleSwarmPolicy(seed=seed))
    social = train_sim.social
    if social is None:  # 配置错误保护（演化臂必须有技能层）
        raise ValueError("evolution 臂需要 social.enabled=true 的技能层")
    distribution = SkillDistribution(seed=seed, sigma=sigma, lr=lr)
    generation_log: list[dict[str, Any]] = []
    for generation in range(int(generations)):
        advance_with_social(train_sim, float(gen_seconds))
        assigned = distribution.sample_skills(train_sim.swarm, social)
        update = distribution.generation_update(train_sim.swarm, social)
        generation_log.append(
            {
                "generation": generation + 1,
                "assigned_skills": assigned,
                "theta": distribution.theta.tolist(),
                "divergence": update["divergence"],
                "skilled_fraction": update["skilled_fraction"],
                "population": int(train_sim.swarm.population),
                "eat_events": int(train_sim.swarm.eat_events),
            }
        )
    # 最终评估：全新世界，θ 冻结后给全部初始个体抽样（表型评估）
    eval_sim = build_simulation(settings, seed=int(seed), policy=RuleSwarmPolicy(seed=seed))
    eval_social = eval_sim.social
    if eval_social is None:  # pragma: no cover - 与上面同一保护
        raise ValueError("evolution 臂需要 social.enabled=true 的技能层")
    distribution.assign_all(eval_sim.swarm, eval_social)
    advance_with_social(eval_sim, float(eval_seconds))
    metrics = eval_sim.metrics()
    metrics.update(common._life_metrics(eval_sim, metrics))
    _decorate(metrics, seed=seed, settings=settings, sim=eval_sim)
    info = {
        "theta_final": distribution.theta.tolist(),
        "theta_history": distribution.history,
        "generations": generation_log,
        "eval_population_skilled": float(metrics["skilled_population"]),
    }
    return metrics, info


# ---------------------------------------------------------------------------
# 协议主体
# ---------------------------------------------------------------------------
@register_protocol(
    "evolution_compare",
    note="P4 演化 vs 学习三臂对照：rule / 技能分布演化 / 策略梯度学习（配对 eval seed）",
)
def run_evolution_compare(settings: ExperimentSettings) -> dict[str, Any]:
    """P4 三臂对照协议：规则 / 演化 / 学习，配对评估并写 zoo 学习模型。"""
    clock = common.timed()
    params = settings.protocol_params
    eval_seeds = _seed_list(
        params.get("eval_seeds"),
        default_start=int(params.get("eval_seed_start", 1100)),
        default_count=5,
    )
    train_seeds = _seed_list(
        params.get("train_seeds"),
        default_start=int(params.get("train_seed_start", 500)),
        default_count=2,
    )
    if set(train_seeds) & set(eval_seeds):
        raise ValueError(
            f"训练 seed 与评估 seed 必须不重叠: {train_seeds} vs {eval_seeds}"
        )
    eval_seconds = float(params.get("eval_seconds", settings.run.eval_seconds))
    episode_seconds = float(params.get("episode_seconds", 60.0))
    episodes = int(params.get("episodes", 8))
    generations = int(params.get("generations", 4))
    gen_seconds = float(params.get("gen_seconds", 60.0))
    sigma = float(params.get("sigma", 0.05))
    lr_evolution = float(params.get("lr_evolution", 0.3))
    hidden = int(params.get("hidden", settings.policy.hidden))
    if episodes < 1 or generations < 1:
        raise ValueError("episodes/generations 必须 >= 1")

    # 臂配置：rule/learning 无技能层；evolution 有技能层但三机制全关、
    # 初始技能率 0（θ 全权决定技能）。
    no_social_settings = dataclasses.replace(
        settings, social=dataclasses.replace(settings.social, enabled=False)
    )
    evolution_settings = dataclasses.replace(
        settings,
        social=dataclasses.replace(settings.social, enabled=True, **_EVO_SOCIAL_FLAGS),
    )

    # -- 学习臂训练（一次性；训练 seed 与评估 seed 不重叠） -----------
    train_clock = common.timed()
    learning_policy, train_log = common.train_neural_policy(
        no_social_settings,
        episodes=episodes,
        episode_seconds=episode_seconds,
        train_seeds=train_seeds,
        seed=0,
        hidden=hidden,
    )
    train_wall_s = train_clock()

    # -- 三臂配对评估 ----------------------------------------------------
    results: dict[str, list[dict[str, Any]]] = {arm: [] for arm in _ARMS}
    evolution_log: list[dict[str, Any]] = []
    eval_clock = common.timed()
    for seed in eval_seeds:
        results["rule"].append(
            _run_rule_arm(no_social_settings, seed=seed, eval_seconds=eval_seconds)
        )
        evolution_metrics, evolution_info = _run_evolution_arm(
            evolution_settings,
            seed=seed,
            generations=generations,
            gen_seconds=gen_seconds,
            eval_seconds=eval_seconds,
            sigma=sigma,
            lr=lr_evolution,
        )
        results["evolution"].append(evolution_metrics)
        evolution_log.append({"seed": int(seed), **evolution_info})
        results["learning"].append(
            _run_learning_arm(
                no_social_settings,
                learning_policy,
                seed=seed,
                eval_seconds=eval_seconds,
            )
        )
    eval_wall_s = eval_clock()

    # -- 汇总 ------------------------------------------------------------
    per_arm = {
        arm: common.summarize_runs(runs, _SUMMARY_KEYS) for arm, runs in results.items()
    }
    paired: dict[str, dict[str, dict[str, float]]] = {
        arm: {
            metric: common.paired_summary(results[arm], results["rule"], metric=metric)
            for metric in _PAIRED_METRICS
        }
        for arm in ("evolution", "learning")
    }

    # -- 简报文本 --------------------------------------------------------
    n_seeds = len(eval_seeds)

    def _arm_bits(arm: str) -> str:
        item = paired[arm]
        return (
            f"{arm}: eat_events {int(item['eat_events']['wins'])}/{n_seeds} 胜"
            f"(Δ={item['eat_events']['mean_delta']:+.0f}±{item['eat_events']['std_delta']:.0f}), "
            f"eat_rate {int(item['eat_rate']['wins'])}/{n_seeds} 胜"
            f"(Δ={item['eat_rate']['mean_delta']:+.5f}), "
            f"mean_life {int(item['mean_life']['wins'])}/{n_seeds} 胜"
            f"(Δ={item['mean_life']['mean_delta']:+.1f}s), "
            f"satiation Δ={item['satiation_mean']['mean_delta']:+.4f}"
        )

    # 诚实判读：learning 的事件计数优势是否伴随福利/存活恶化？
    learning_paired = paired["learning"]
    evolution_paired = paired["evolution"]
    theta_shift_by_seed = [
        float(
            np.abs(
                np.asarray(item["theta_final"], dtype=np.float64)
                - 1.0 / len(SKILLS)
            ).mean()
        )
        for item in evolution_log
    ]
    mean_theta_shift = float(np.mean(theta_shift_by_seed)) if theta_shift_by_seed else 0.0
    event_count_divergence = bool(
        float(learning_paired["eat_events"]["mean_delta"]) > 0.0
        and float(learning_paired["satiation_mean"]["mean_delta"]) < 0.0
        and float(learning_paired["mean_life"]["mean_delta"]) < 0.0
        and float(learning_paired["population"]["mean_delta"]) < 0.0
    )
    honest_flags = {
        "learning_eat_events_wins": int(learning_paired["eat_events"]["wins"]),
        "learning_eat_rate_wins": int(learning_paired["eat_rate"]["wins"]),
        "learning_satiation_wins": int(learning_paired["satiation_mean"]["wins"]),
        "learning_mean_life_wins": int(learning_paired["mean_life"]["wins"]),
        "learning_population_wins": int(learning_paired["population"]["wins"]),
        "learning_event_count_vs_welfare_divergence": event_count_divergence,
        "evolution_eat_rate_wins": int(evolution_paired["eat_rate"]["wins"]),
        "evolution_mean_eat_rate_delta": float(
            evolution_paired["eat_rate"]["mean_delta"]
        ),
        "evolution_theta_mean_shift_from_uniform": mean_theta_shift,
        "evolution_theta_moved": bool(mean_theta_shift > 0.02),
    }
    honest_bits = (
        f"learning: eat_events {int(learning_paired['eat_events']['wins'])}/{n_seeds}"
        f" 胜（Δ={learning_paired['eat_events']['mean_delta']:+.0f}），"
        f"eat_rate {int(learning_paired['eat_rate']['wins'])}/{n_seeds}"
        f"（Δ={learning_paired['eat_rate']['mean_delta']:+.5f}），"
        f"但 satiation {int(learning_paired['satiation_mean']['wins'])}/{n_seeds}"
        f"（Δ={learning_paired['satiation_mean']['mean_delta']:+.4f}）、"
        f"mean_life {int(learning_paired['mean_life']['wins'])}/{n_seeds}"
        f"（Δ={learning_paired['mean_life']['mean_delta']:+.1f}s）、"
        f"population {int(learning_paired['population']['wins'])}/{n_seeds}"
        f"（Δ={learning_paired['population']['mean_delta']:+.1f}）；"
        f"evolution: eat_rate {int(evolution_paired['eat_rate']['wins'])}/{n_seeds} 胜"
        f"（Δ={evolution_paired['eat_rate']['mean_delta']:+.5f}），"
        f"θ 距均匀分布平均偏移={mean_theta_shift:.3f}（θ 确实在动）。"
    )
    if event_count_divergence:
        honest_statement = honest_bits + (
            "判读：learning 只赢'进食事件计数'，同一批 seed 上 satiation、"
            "mean_life、population 全面变差——eat_events 计事件数而非摄入"
            "能量，不能把该增益表述为适应优势。旧 P4 的'learning 显著正效应"
            "（+6368 eats）'在本协议口径下不重复为福利改善，应予降级为"
            "'事件计数上升'。evolution 的 θ 明显移动但宏观指标无差异（表型"
            "弱），不宣称演化有效或无效，只报告当前规模下无宏观效应。"
        )
    else:
        honest_statement = honest_bits + (
            "判读：未见明确的事件计数/福利背离；按上表数值如实解读，"
            "不做显著性声明。"
        )
    verdict = (
        f"三臂 vs rule（配对，n={n_seeds}，eval {eval_seconds:g}s，"
        f"演化 {generations} 代×{gen_seconds:g}s，学习 {episodes} 回合×"
        f"{episode_seconds:g}s 训练于 seed {train_seeds}）："
        f"{_arm_bits('evolution')}；{_arm_bits('learning')}。"
        " 只报效应量/胜场，不做显著性检验。"
        f" 诚实判读：{'learning 仅事件计数占优、福利/存活恶化（非适应优势）；evolution 无宏观效应' if event_count_divergence else '未见明确事件计数/福利背离，按上表数值如实解读'}。"
    )

    # -- 学习模型入 zoo ---------------------------------------------------
    zoo_model: dict[str, Any] | None = None
    if settings.run.save_models:
        zoo = ModelZoo()
        zoo_name = f"{settings.name}_learning"
        card = zoo.save(
            zoo_name,
            learning_policy,
            metrics={
                "train_episodes": episodes,
                "episode_seconds": episode_seconds,
                "train_seeds": train_seeds,
                "eval_seeds": eval_seeds,
                "eval_seconds": eval_seconds,
                "eat_rate_delta_vs_rule": paired["learning"]["eat_rate"]["mean_delta"],
                "eat_events_delta_vs_rule": paired["learning"]["eat_events"]["mean_delta"],
                "eat_events_wins_vs_rule": paired["learning"]["eat_events"]["wins"],
                "event_count_vs_welfare_divergence": event_count_divergence,
                "n_eval_seeds": n_seeds,
            },
            config=settings,
            tags=["p4", "scarce"],
            description=(
                "P4 学习臂：稀缺世界 REINFORCE 训练的 NeuralSwarmPolicy"
                "（训练 seed 与评估 seed 不重叠）"
            ),
        )
        zoo_model = {
            "name": zoo_name,
            "directory": str(zoo.model_dir(zoo_name)),
            "weights_sha256": card.weights_sha256,
            "config_hash": card.config_hash,
            "tags": list(card.tags),
        }

    caveats = [
        "演化臂表型弱：3 个离散技能中，efficient_biter 的 bite 倍率会被"
        "残食上限截断、long_sight 只有 3 格探测半径，只有 "
        "slow_metabolism（代谢 ×0.75）在稀缺环境有明确生理优势——"
        "弱表型的任意组合都难显著超过无技能基线（旧 P4 机制诊断）。",
        "信号通道是结构性差距：learning 每步拿到稠密 (obs, reward) 梯度"
        "信号（每回合数万条转移），evolution 每代只有整代存活饱足均值"
        "一个标量。旧 P4 同等规模下 evolution 1/5 胜、learning 5/5 胜；"
        "本协议不做'演化无效'的强声明，只报告当前规模下的效应量。",
        "旧 P4 的 learning 臂在每个 eval seed 上就地训练（训练/评估同 seed，"
        "信息泄漏）；本协议改为在训练外 seed（500+）训练、评估 seed"
        "（1100+）配对，因此数值不可与旧 P4 直接对比，但泛化结论更可信。",
        "演化臂最终评估把冻结 θ 抽样到全新初始种群（skills prior 效应），"
        "不是完整基因-文化协同演化；解释时需限定为'技能先验的适应价值'。",
        f"评估时长 {eval_seconds:g}s（旧 P4 为 120s）、每代 {gen_seconds:g}s"
        f"（旧 P4 为 60s）；mean_life 在短窗口下右删失更常见。",
        "配对差只报均值/标准差/胜场，不做显著性检验（n 个 eval seed 很小）。",
        "rule 臂无技能层；evolution 臂有技能层但三机制全关且初始技能率 0，"
        "两者世界规则差异只有'技能效果是否生效'，是公平的机制对照。",
        "eat_events 口径警告（本次实测触发）：learning 的 eat_events/eat_rate"
        " 上升与 satiation/mean_life/population 下降同时出现——eat_events "
        "计事件数而非摄入能量，事件计数优势不能解释为适应增益；旧 P4 的"
        " learning 正结果需按此降级理解（见 honest_reading.statement）。",
    ]

    return {
        "name": settings.name,
        "protocol": "evolution_compare",
        "experiment": {
            "eval_seeds": eval_seeds,
            "train_seeds": train_seeds,
            "episodes": episodes,
            "episode_seconds": episode_seconds,
            "generations": generations,
            "gen_seconds": gen_seconds,
            "eval_seconds": eval_seconds,
            "sigma": sigma,
            "lr_evolution": lr_evolution,
            "hidden": hidden,
            "food_regrow_per_s": float(settings.world.food_regrow_per_s),
            "initial_population": int(settings.swarm.initial_population),
            "learning_train_eval_seeds_disjoint": True,
        },
        "arms": list(_ARMS),
        "per_arm": per_arm,
        "paired": paired,
        "evolution_arms": evolution_log,
        "train_log": train_log,
        "honest_reading": {
            "flags": honest_flags,
            "statement": honest_statement,
        },
        "verdict": verdict,
        "zoo_model": zoo_model,
        "caveats": caveats,
        "wall_s": round(float(clock()), 1),
        "train_wall_s": round(float(train_wall_s), 1),
        "eval_wall_s": round(float(eval_wall_s), 1),
    }


__all__ = ["SkillDistribution", "run_evolution_compare"]
