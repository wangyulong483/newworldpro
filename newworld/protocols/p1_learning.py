"""P1 学习对照协议（learning_compare）：学习策略 vs 随机/规则/未训练基线。

设计来源：experiments/run_p1_baseline.py（P1 / P1.5 / P1.6 / P1.7 的
设计沉淀）。P5 起不再有独立实验脚本，本协议由注册表按 YAML 里的
protocol: learning_compare 调起：

    python -m newworld --config configs/p1_scarce.yaml

实验设计
========
同一稀缺地图、同 seed、同资源预算下的四臂**配对**对照（每 seed 独立
重建世界，四臂共享同一初始地图与人口）：

1. random：均匀随机 5 动作（下界基线）；
2. rule：RuleSwarmPolicy（V3 原规则：食物梯度探测 + wander）；
3. untrained：随机初始化的 NeuralSwarmPolicy（同 seed=0）；
4. trained：在**训练 seed** 上做 REINFORCE 后的 NeuralSwarmPolicy。

训练与评估 seed 集合不重叠（训练 seed 500+，评估 seed 900+），评估
再做 densities 密度梯度（初始人口档位：低密度站桩吃再生流 vs
高密度竞争），因为 P1.6 的诊断表明"学习优势"只有在存在觅食选择
压力时才有意义。

指标（口径全部继承 newworld.protocols.common，不在此重定义）
------------------------------------------------------------------
* eat_events：进食事件总数；eat_rate = 进食 / (初始人口·秒)
  （四臂同分母，可直接配对；不是严格的存活个体·秒 per-capita）；
* mean_life：逐个体死亡年龄均值（Swarm.death_ages）；若评估
  窗口内无人死亡则退化为存活者当前年龄（右删失下界）；
* 其余：population / births / deaths / starvation_deaths /
  satiation_mean / satiation_std / gini_hunger / total_reward /
  survived_90s / carrying_capacity_estimate。

预注册门槛
==========
主密度（densities 第一档）上，trained 在 eat_rate 或 mean_life
的配对胜场 >= ceil(0.8*n_eval_seeds) 即判"门槛通过"；达不到如实报告
差距。配对差只报告均值/胜场，不做显著性检验（小样本纪律）。

P1.6 / P1.7 诚实性备注（随 summary 一起归档）
=============================================
* P1.6 指标伪影教训：旧 mean_life 在无人死亡时的 fallback 逻辑
  曾制造 trained 的假阳性；当前口径已改为逐个体死亡年龄。
* P1.6 承载过剩：默认 world.food_regrow_per_s=0.012 约为总代谢
  消耗的 10 倍，任何活法都饿不死 → 没有觅食选择压力。本协议使用的
  配置必须用 P1.7 稀缺值（0.0005），summary 里会记录实际再生速率
  与各密度初始人口/承载力估计，供读者自行判断竞争是否成立。
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from ..registry import register_protocol
from ..settings import ExperimentSettings
from ..swarm_policy import NeuralSwarmPolicy, RandomSwarmPolicy, RuleSwarmPolicy
from ..zoo import ModelZoo
from . import common

#: 四种对照臂（顺序固定，报告可读）
_ARMS: tuple[str, ...] = ("random", "rule", "untrained", "trained")
#: 逐臂汇总的指标键（两两配对比较的公共口径）
_SUMMARY_KEYS: tuple[str, ...] = (
    "population",
    "births",
    "deaths",
    "starvation_deaths",
    "old_age_deaths",
    "drowned",
    "eat_events",
    "eat_rate",
    "mean_life",
    "satiation_mean",
    "satiation_std",
    "gini_hunger",
    "total_reward",
    "survived_90s",
    "carrying_capacity_estimate",
)
#: 配对比较的指标（任务要求的 eat_events / eat_rate / mean_life）
_PAIRED_METRICS: tuple[str, ...] = ("eat_events", "eat_rate", "mean_life")
#: 次级配对（trained vs random / untrained）：检验"学习优势"是否只是相对
#: rule 基线的口径产物（无学习基线是否同样胜过 rule）
_SECONDARY_METRICS: tuple[str, ...] = (
    "eat_events",
    "eat_rate",
    "mean_life",
    "satiation_mean",
)


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def _seed_list(value: Any, *, default_start: int, default_count: int) -> list[int]:
    """protocol_params 里的 seed 规格：整数=数量，列表=显式 seed。"""
    if isinstance(value, (list, tuple)):
        seeds = [int(seed) for seed in value]
        if not seeds:
            raise ValueError("seed 列表不能为空")
        return seeds
    count = int(default_count if value is None else value)
    if count < 1:
        raise ValueError(f"seed 数量必须 >= 1，得到 {count}")
    return [int(default_start) + index for index in range(count)]


def _densities(settings: ExperimentSettings) -> list[tuple[str, int]]:
    """解析密度档位；缺省用配置里的初始人口作为 normal 档。"""
    raw = settings.protocol_params.get("densities")
    if raw is None:
        return [("normal", int(settings.swarm.initial_population))]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("protocol_params.densities 必须是非空列表")
    parsed: list[tuple[str, int]] = []
    for item in raw:
        if not isinstance(item, dict) or "name" not in item or "population" not in item:
            raise ValueError(f"密度档位需要 {{name, population}}，得到 {item!r}")
        parsed.append((str(item["name"]), int(item["population"])))
    names = [name for name, _ in parsed]
    if len(set(names)) != len(names):
        raise ValueError(f"密度档位重名: {names}")
    return parsed


def _density_settings(settings: ExperimentSettings, population: int) -> ExperimentSettings:
    """把初始人口替换成指定密度档位（容量上限同时保证 >= 人口）。"""
    swarm = dataclasses.replace(
        settings.swarm,
        initial_population=int(population),
        capacity_hard=max(int(settings.swarm.capacity_hard), int(population)),
    )
    return dataclasses.replace(settings, swarm=swarm)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def _evaluate(
    settings: ExperimentSettings,
    *,
    eval_seeds: list[int],
    eval_seconds: float,
    trained_policy: NeuralSwarmPolicy,
    hidden: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """四臂 × eval seeds 配对评估；返回 {density: {arm: [runs]}}。"""
    obs_dim = int(trained_policy.obs_dim)
    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for density_name, population in _densities(settings):
        density_settings = _density_settings(settings, population)
        arm_runs: dict[str, list[dict[str, Any]]] = {arm: [] for arm in _ARMS}
        for eval_seed in eval_seeds:
            policies = {
                # 旧 P1 行为：random/rule/untrained 每评估 seed 以固定
                # seed 重建（同 seed 可复现）；trained 复用同一对象。
                "random": RandomSwarmPolicy(seed=1),
                "rule": RuleSwarmPolicy(seed=1),
                "untrained": NeuralSwarmPolicy(
                    seed=0, obs_dim=obs_dim, hidden=int(hidden)
                ),
                "trained": trained_policy,
            }
            for arm in _ARMS:
                sim, metrics = common.simulate(
                    density_settings,
                    seed=int(eval_seed),
                    seconds=float(eval_seconds),
                    policy=policies[arm],
                )
                metrics["seed"] = int(eval_seed)
                metrics["density"] = density_name
                metrics["initial_population"] = int(population)
                # P1.6 承载过剩诊断：把评分时的承载力估计一起存档，
                # 读者可据此判断该密度档位到底有没有竞争。
                metrics["carrying_capacity_estimate"] = float(
                    sim.swarm.carrying_capacity_estimate()
                )
                metrics["food_regrow_per_s"] = float(
                    density_settings.world.food_regrow_per_s
                )
                arm_runs[arm].append(metrics)
        results[density_name] = arm_runs
    return results


# ---------------------------------------------------------------------------
# 协议主体
# ---------------------------------------------------------------------------
@register_protocol(
    "learning_compare",
    note="P1 学习对照：random/rule/untrained/trained 四臂配对 + 密度梯度",
)
def run_learning_compare(settings: ExperimentSettings) -> dict[str, Any]:
    """P1 学习对照协议：训练后做四臂配对评估并写 zoo 模型。"""
    clock = common.timed()
    params = settings.protocol_params
    train_episodes = int(params.get("train_episodes", 12))
    train_seeds = _seed_list(
        params.get("train_seeds"), default_start=500, default_count=6
    )
    eval_seeds = _seed_list(
        params.get("eval_seeds"), default_start=900, default_count=10
    )
    if set(train_seeds) & set(eval_seeds):
        raise ValueError(
            f"训练 seed 与评估 seed 必须不重叠: {train_seeds} vs {eval_seeds}"
        )
    eval_seconds = float(params.get("eval_seconds", settings.run.eval_seconds))
    episode_seconds = float(params.get("episode_seconds", 60.0))
    hidden = int(params.get("hidden", settings.policy.hidden))
    if train_episodes < 1:
        raise ValueError("train_episodes 必须 >= 1")

    # -- 训练（唯一的学习臂） -------------------------------------------
    train_clock = common.timed()
    trained_policy, train_log = common.train_neural_policy(
        settings,
        episodes=train_episodes,
        episode_seconds=episode_seconds,
        train_seeds=train_seeds,
        seed=0,
        hidden=hidden,
    )
    train_wall_s = train_clock()

    # -- 四臂配对评估（密度梯度） --------------------------------------
    eval_clock = common.timed()
    results = _evaluate(
        settings,
        eval_seeds=eval_seeds,
        eval_seconds=eval_seconds,
        trained_policy=trained_policy,
        hidden=hidden,
    )
    eval_wall_s = eval_clock()

    # -- 汇总 ------------------------------------------------------------
    per_arm: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    paired: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for density_name, arm_runs in results.items():
        per_arm[density_name] = {
            arm: common.summarize_runs(arm_runs[arm], _SUMMARY_KEYS) for arm in _ARMS
        }
        control = arm_runs["rule"]
        paired[density_name] = {
            arm: {
                metric: common.paired_summary(arm_runs[arm], control, metric=metric)
                for metric in _PAIRED_METRICS
            }
            for arm in ("random", "untrained", "trained")
        }

    # -- 次级配对：trained vs random / untrained -------------------------
    # 目的：区分"相对 rule 基线的优势"与"任何移动策略都有的优势"。
    secondary_paired: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        density_name: {
            f"trained_vs_{base_arm}": {
                metric: common.paired_summary(
                    results[density_name]["trained"],
                    results[density_name][base_arm],
                    metric=metric,
                )
                for metric in _SECONDARY_METRICS
            }
            for base_arm in ("random", "untrained")
        }
        for density_name in results
    }

    # -- 预注册门槛（主密度 = 第一档） ---------------------------------
    primary_density = next(iter(results))
    n_seeds = len(eval_seeds)
    needed = max(1, int(math.ceil(0.8 * n_seeds)))
    trained_paired = paired[primary_density]["trained"]
    eat_wins = int(trained_paired["eat_rate"]["wins"])
    life_wins = int(trained_paired["mean_life"]["wins"])
    gate_passed = bool(eat_wins >= needed or life_wins >= needed)
    gate = {
        "primary_density": primary_density,
        "n_eval_seeds": n_seeds,
        "needed_wins": needed,
        "eat_rate_wins": eat_wins,
        "mean_life_wins": life_wins,
        "passed": gate_passed,
        "rule": "trained 在 eat_rate 或 mean_life 的配对胜场 >= ceil(0.8*n)",
    }

    # -- 简报文本（诚实、带数字） --------------------------------------
    eat_delta = trained_paired["eat_rate"]["mean_delta"]
    eat_std = trained_paired["eat_rate"]["std_delta"]
    life_delta = trained_paired["mean_life"]["mean_delta"]
    life_std = trained_paired["mean_life"]["std_delta"]
    density_bits = []
    for density_name, arm_paired in paired.items():
        item = arm_paired["trained"]
        density_bits.append(
            f"{density_name}(pop={results[density_name]['trained'][0]['initial_population']}): "
            f"eat_rate {int(item['eat_rate']['wins'])}/{n_seeds} 胜 "
            f"(Δ={item['eat_rate']['mean_delta']:+.5f}), "
            f"mean_life {int(item['mean_life']['wins'])}/{n_seeds} 胜 "
            f"(Δ={item['mean_life']['mean_delta']:+.1f}s)"
        )
    # 诚实判读：eat-event 计数口径的优势是否也能被"无学习"基线复现？
    random_paired = paired[primary_density]["random"]
    untrained_paired = paired[primary_density]["untrained"]
    secondary = secondary_paired[primary_density]
    trained_vs_random = secondary["trained_vs_random"]["eat_rate"]
    trained_vs_untrained = secondary["trained_vs_untrained"]["eat_rate"]
    random_wins = int(random_paired["eat_rate"]["wins"])
    untrained_wins = int(untrained_paired["eat_rate"]["wins"])
    random_eat_delta = float(random_paired["eat_rate"]["mean_delta"])
    untrained_eat_delta = float(untrained_paired["eat_rate"]["mean_delta"])
    beats_random = int(trained_vs_random["wins"]) >= needed
    arms_eat = {
        arm: per_arm[primary_density][arm]["eat_events"]["mean"] for arm in _ARMS
    }
    arms_sat = {
        arm: per_arm[primary_density][arm]["satiation_mean"]["mean"] for arm in _ARMS
    }
    arms_life = {
        arm: per_arm[primary_density][arm]["mean_life"]["mean"] for arm in _ARMS
    }
    trained_sat = arms_sat["trained"]
    rule_sat = arms_sat["rule"]
    event_count_artifact = bool(
        gate_passed
        and random_wins >= needed
        and not beats_random
        and trained_sat < rule_sat
    )
    honest_flags = {
        "gate_passed": gate_passed,
        "trained_beats_rule_eat_rate": bool(eat_wins >= needed),
        "random_also_beats_rule": bool(random_wins >= needed),
        "untrained_also_beats_rule": bool(untrained_wins >= needed),
        "trained_beats_random_eat_rate": beats_random,
        "trained_welfare_better_than_rule": bool(
            trained_sat > rule_sat and arms_life["trained"] > arms_life["rule"]
        ),
        "event_count_artifact_signature": event_count_artifact,
    }
    honest_bits = (
        f"trained vs rule eat_rate {eat_wins}/{n_seeds}（Δ={eat_delta:+.5f}）；"
        f"无学习成分的 random {random_wins}/{n_seeds} 胜 rule"
        f"（Δ={random_eat_delta:+.5f}）、untrained {untrained_wins}/{n_seeds} 胜"
        f" rule（Δ={untrained_eat_delta:+.5f}）；trained vs random 配对"
        f" Δ={trained_vs_random['mean_delta']:+.5f}"
        f"（{int(trained_vs_random['wins'])}/{n_seeds} 胜），trained vs untrained"
        f" Δ={trained_vs_untrained['mean_delta']:+.5f}"
        f"（{int(trained_vs_untrained['wins'])}/{n_seeds} 胜）；福利口径："
        f"trained satiation={trained_sat:.3f}（rule={rule_sat:.3f}，"
        f"random={arms_sat['random']:.3f}，untrained={arms_sat['untrained']:.3f}），"
        f"mean_life={arms_life['trained']:.1f}s（rule={arms_life['rule']:.1f}s），"
        f"eat_events={arms_eat['trained']:.0f}（random={arms_eat['random']:.0f}，"
        f"untrained={arms_eat['untrained']:.0f}，rule={arms_eat['rule']:.0f}）。"
    )
    if event_count_artifact:
        honest_statement = honest_bits + (
            "判读：'eat_rate 胜 rule' 不能表述为学习到的觅食能力——无学习的"
            " random/untrained 同样（甚至更大程度地）胜 rule，trained 未胜过"
            " random/untrained（配对胜场未达门槛），且 satiation/mean_life 反而"
            "低于 rule。该优势符合'事件计数口径奖励移动频率'的伪影特征"
            "（eat_events 计事件数，不计量摄入能量/适应度）。P1.7 的'学习优势'"
            "在本次复现中不成立；预注册门槛只是'相对 rule'的门槛。"
        )
    elif gate_passed and beats_random:
        honest_statement = honest_bits + (
            "判读：trained 同时胜过 rule 与 random 的事件计数，方向支持学习"
            "效应；但 satiation/mean_life 若不同向改善，仍需按摄入量/存活"
            "口径复核（本 summary 已列全部数值）。"
        )
    else:
        honest_statement = honest_bits + (
            "判读：未通过预注册门槛或未胜过无学习基线，如实报告差距，"
            "不宣称学习优势。"
        )
    verdict = (
        f"trained vs rule（{primary_density} 主密度，n={n_seeds}）："
        f"eat_rate 胜 {eat_wins}/{n_seeds}（Δ={eat_delta:+.5f}±{eat_std:.5f}），"
        f"mean_life 胜 {life_wins}/{n_seeds}（Δ={life_delta:+.1f}±{life_std:.1f}s）；"
        f"预注册门槛 >= {needed}/{n_seeds} → {'通过' if gate_passed else '未通过'}。"
        f" 全部密度：{'；'.join(density_bits)}。"
        f" 诚实判读：{'eat_rate 优势是事件计数口径伪影，不宣称学习优势' if event_count_artifact else ('方向支持学习效应，但需 welfare 口径复核' if gate_passed and beats_random else '未通过/未胜过无学习基线，如实报告差距')}。"
        f" 详细数字见 honest_reading；不做显著性检验。"
    )

    # -- 模型入 zoo（训练产物的可追溯元数据） ---------------------------
    zoo_model: dict[str, Any] | None = None
    if settings.run.save_models:
        zoo = ModelZoo()
        zoo_name = f"{settings.name}_neural"
        trained_eval = per_arm[primary_density]["trained"]
        card = zoo.save(
            zoo_name,
            trained_policy,
            metrics={
                "train_episodes": train_episodes,
                "train_seeds": train_seeds,
                "eval_seeds": eval_seeds,
                "eval_seconds": eval_seconds,
                "density": primary_density,
                "eat_events_mean": trained_eval["eat_events"]["mean"],
                "eat_rate_mean": trained_eval["eat_rate"]["mean"],
                "eat_rate_delta_vs_rule": eat_delta,
                "mean_life_mean": trained_eval["mean_life"]["mean"],
                "mean_life_delta_vs_rule": life_delta,
                "gate_passed": gate_passed,
                "trained_beats_random_eat_rate": beats_random,
                "event_count_artifact_signature": event_count_artifact,
            },
            config=settings,
            tags=["p1", "scarce"],
            description=(
                "P1.7 稀缺世界 REINFORCE 训练的策略（NeuralSwarmPolicy，"
                "obs=9、动作=5、参数共享）"
            ),
        )
        zoo_model = {
            "name": zoo_name,
            "directory": str(zoo.model_dir(zoo_name)),
            "weights_sha256": card.weights_sha256,
            "config_hash": card.config_hash,
            "tags": list(card.tags),
        }

    # -- 诚实性 caveats（P1.6/P1.7 沉淀，必须随结果归档） ---------------
    caveats = [
        "P1.6 指标伪影教训：旧 mean_life 在无人死亡时用 fallback 逻辑（曾"
        "制造 trained 的假阳性）；本协议使用 Swarm.death_ages 逐个体死亡"
        "年龄；若评估窗口内无人死亡，则 mean_life 退化为存活者当前年龄"
        "（右删失下界，跨窗口不可比）。",
        "P1.6 承载过剩诊断：默认 world.food_regrow_per_s=0.012 约为总代谢"
        "消耗的 10 倍，任何活法都饿不死 → 无觅食选择压力。本配置使用 "
        f"food_regrow_per_s={settings.world.food_regrow_per_s}（P1.7 稀缺）。"
        " 但 normal 档位仍可能远低于实际承载力（见每臂 "
        "carrying_capacity_estimate），低密度下站桩吃再生流是环境事实；"
        "只有 dense 档位才构成竞争压力。",
        "eat_rate 分母 = 初始人口 × 模拟秒数（四臂同口径，可配对），"
        "不是存活个体·秒 per-capita；原始 eat_events 同时报告。",
        "eat_events 口径警告（本次实测触发）：无学习的 random/untrained 臂在"
        "稀缺环境中也稳定胜过 rule 的进食事件数——rule 在饱足高时停止移动"
        "（事件少但饱足高），随机/神经策略持续移动会在残食格触发更多"
        "'小口事件'。eat_events 计的是事件数而非摄入能量，因此'eat_rate 胜过"
        " rule'不等于觅食能力/适应度优势；必须同时看 satiation/mean_life/"
        "摄入量级的配对结果（见 honest_reading）。",
        f"评估时长 {eval_seconds:g}s（旧 P1 为 120s）；短窗口下 mean_life "
        "右删失更常见，绝对寿命值不可与旧 120s 结果直接对比。",
        "配对差仅报告均值/标准差/胜场；n 很小，不做显著性检验，"
        "不把方向性差异表述为已确立效应。",
        "trained 臂跨 eval seed 复用同一策略对象（采样 RNG 状态延续，"
        "与旧 P1 一致）；random/rule/untrained 每个 eval seed 以固定 seed "
        "重建（同 seed 可复现）。",
    ]

    return {
        "name": settings.name,
        "protocol": "learning_compare",
        "experiment": {
            "train_episodes": train_episodes,
            "train_seeds": train_seeds,
            "episode_seconds": episode_seconds,
            "eval_seeds": eval_seeds,
            "eval_seconds": eval_seconds,
            "hidden": hidden,
            "food_regrow_per_s": float(settings.world.food_regrow_per_s),
            "policy": settings.policy.kind,
        },
        "densities": {
            name: {"initial_population": int(population)}
            for name, population in _densities(settings)
        },
        "arms": list(_ARMS),
        "per_arm": per_arm,
        "paired": paired,
        "secondary_paired": secondary_paired,
        "gate": gate,
        "honest_reading": {
            "flags": honest_flags,
            "statement": honest_statement,
        },
        "verdict": verdict,
        "train_log": train_log,
        "zoo_model": zoo_model,
        "caveats": caveats,
        "wall_s": round(float(clock()), 1),
        "train_wall_s": round(float(train_wall_s), 1),
        "eval_wall_s": round(float(eval_wall_s), 1),
    }


__all__ = ["run_learning_compare"]
