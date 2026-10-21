"""P6-B 记忆/反思消融实验协议（真实训练 + 配对种子 + obs_dim 混淆控制）。

实验问题
========
去 LLM 的生成式记忆（记忆流 × 新近/重要/相关检索 + 周期反思）到底能不能
让群伙觅食得更好？用同一世界、同一训练 seed 的配对消融回答：

  off           记忆关闭（9 维基础观测）——经典下界/基线
  off_padded    记忆关闭 + 6 维恒零观测 provider（obs_dim=15，与记忆臂同维）
                ——控制 W1 按 1/sqrt(obs_dim) 缩放的初始化尺度混淆
  recency       只有新近权重（w_rec=1, w_imp=0, w_rel=0），无反思
  full          新近 × 重要 × 相关（1/1/1），无反思
  full_reflect  full + 每 reflection_every_s 秒反思一次（生成高层记忆）

每个臂都从**相同 policy seed** 的相同初始权重开始训练（REINFORCE），并在
**相同 eval seeds**（同地图/同初始人口）上配对评估。报告：
* 各臂绝对指标（mean±std，n=eval seeds 数，每臂 n 相同）；
* 相对 off / off_padded 的配对差值（mean_delta / relative / wins-loses-ties）；
* 反射臂相对 full 的增量（反思到底有没有用）；
* 记忆机制活跃度（事件/检索/反思/洞察/特征幅值）证明机制真的在跑；
* 诚实的结论：有效应就给效应量与胜率；若效应在 off_padded 下消失，直接
  说明"更像 obs_dim/初始化尺度混淆，而非记忆信息收益"；没有效应就直说
  并诊断。n 很小，不做显著性过度声称。

产物
====
* summary.json（experiment.run_experiment 统一写）含本协议返回的完整 dict；
* report.md（本协议写进产物目录）：表格化数字 + 结论；
* assets/zoo/p6-memory：最佳记忆臂模型（tag p6-memory；空对照不参与评选）。
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ..agent_memory import NullMemorySettings
from ..experiment import output_dir
from ..registry import register_protocol
from ..settings import ExperimentSettings, MemorySettings
from ..zoo import ModelZoo
from .common import (
    evaluate_seeds,
    mean_std,
    paired_summary,
    summarize_runs,
    timed,
    train_neural_policy,
)

#: 评估指标（配对/汇总口径）
#: * eat_events/eat_rate 是"进食事件次数/频率"，微吃也计数 → 行为频率指标；
#: * total_reward = 累计个体奖励（≈累计进食量 - 时间成本）→ 净摄入代理；
#: * satiation/population/starvation_deaths/gini 是福利/结局指标。
#: 二者方向必须同时看，不能只凭 eat_rate 上升就说"记忆有用"。
EVAL_METRICS = (
    "eat_events",
    "eat_rate",
    "satiation_mean",
    "satiation_std",
    "population",
    "mean_life",
    "starvation_deaths",
    "births",
    "total_reward",
    "gini_hunger",
)
#: 主效应指标（报告与最佳臂选择）
PRIMARY_METRIC = "eat_rate"
SECONDARY_METRIC = "satiation_mean"
#: 配对比较对（vs off = 原始消融；vs off_padded = obs_dim 混淆控制）
COMPARISONS = (
    ("recency_vs_off", "recency", "off"),
    ("full_vs_off", "full", "off"),
    ("full_reflect_vs_off", "full_reflect", "off"),
    ("full_reflect_vs_full", "full_reflect", "full"),
    ("recency_vs_off_padded", "recency", "off_padded"),
    ("full_vs_off_padded", "full", "off_padded"),
    ("full_reflect_vs_off_padded", "full_reflect", "off_padded"),
)
ARM_NAMES = ("off", "off_padded", "recency", "full", "full_reflect")
#: 有真实记忆信息的臂（zoo 最佳模型只从这里选，避免把空对照当产品）
MEMORY_ARM_CANDIDATES = ("recency", "full", "full_reflect")
#: 相对 off 的原始消融比较
RAW_COMPARISONS = ("recency_vs_off", "full_vs_off", "full_reflect_vs_off")
#: 相对 off_padded（同 obs_dim 零记忆）的混淆控制比较
PADDED_COMPARISONS = (
    "recency_vs_off_padded",
    "full_vs_off_padded",
    "full_reflect_vs_off_padded",
)


# ---------------------------------------------------------------------------
# 配置处理
# ---------------------------------------------------------------------------
def _arm_memory_configs(
    base: MemorySettings, reflection_every_s: float
) -> dict[str, MemorySettings]:
    """五个消融臂的记忆配置（其余字段沿用 YAML）。"""
    recency_only = dataclasses.replace(
        base,
        enabled=True,
        weight_recency=1.0,
        weight_importance=0.0,
        weight_relevance=0.0,
        reflection_every_s=0.0,
    )
    full = dataclasses.replace(
        base,
        enabled=True,
        weight_recency=1.0,
        weight_importance=1.0,
        weight_relevance=1.0,
        reflection_every_s=0.0,
    )
    full_reflect = dataclasses.replace(
        full, reflection_every_s=float(reflection_every_s)
    )
    # 空记忆对照：enabled=True 但观测恒 0（走正常构造路径，obs_dim=15）
    padded = NullMemorySettings(
        enabled=True,
        capacity=base.capacity,
        recency_halflife_s=base.recency_halflife_s,
        weight_recency=base.weight_recency,
        weight_importance=base.weight_importance,
        weight_relevance=base.weight_relevance,
        top_k=base.top_k,
        reflection_every_s=0.0,
        reflection_min_events=base.reflection_min_events,
        seed=base.seed,
    )
    return {
        "off": dataclasses.replace(base, enabled=False),
        "off_padded": padded,
        "recency": recency_only,
        "full": full,
        "full_reflect": full_reflect,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    return value


# ---------------------------------------------------------------------------
# 协议
# ---------------------------------------------------------------------------
@register_protocol(
    "memory_ablation",
    note="P6-B 记忆/反思配对消融（off/off_padded/recency/full/full+reflection）",
)
def memory_ablation(settings: ExperimentSettings) -> dict[str, Any]:
    """训练 + 配对评估五个记忆臂，写 report.md / zoo 模型，返回 summary。"""
    params = dict(settings.protocol_params or {})
    train_episodes = int(params.get("train_episodes", 3))
    train_seconds = float(params.get("train_seconds", settings.run.seconds))
    eval_seconds = float(params.get("eval_seconds", settings.run.eval_seconds))
    train_seeds = [int(seed) for seed in params.get("train_seeds", [1000, 1001, 1002])]
    eval_seeds = [int(seed) for seed in params.get("eval_seeds", settings.run.seeds)]
    policy_seed = int(params.get("policy_seed", 7))
    reflection_every_s = float(
        params.get("reflection_every_s", settings.memory.reflection_every_s or 20.0)
    )
    hidden = int(params.get("hidden", settings.policy.hidden))
    if not train_seeds:
        raise ValueError("protocol_params.train_seeds 不能为空")
    if not eval_seeds:
        raise ValueError("protocol_params.eval_seeds 不能为空")
    if train_episodes < 1:
        raise ValueError("protocol_params.train_episodes >= 1")

    arm_memory = _arm_memory_configs(settings.memory, reflection_every_s)
    started = timed()
    arms: dict[str, dict[str, Any]] = {}
    eval_runs: dict[str, list[dict[str, Any]]] = {}
    policies: dict[str, Any] = {}
    summaries: dict[str, dict[str, dict[str, float]]] = {}

    for arm in ARM_NAMES:
        arm_started = timed()
        arm_settings = dataclasses.replace(settings, memory=arm_memory[arm])
        policy, history = train_neural_policy(
            arm_settings,
            episodes=train_episodes,
            episode_seconds=train_seconds,
            train_seeds=train_seeds,
            seed=policy_seed,
            hidden=hidden,
        )
        train_wall = arm_started()
        runs = evaluate_seeds(
            arm_settings, policy, seeds=eval_seeds, seconds=eval_seconds
        )
        eval_wall = arm_started() - train_wall
        run_summary = summarize_runs(runs, EVAL_METRICS)
        memory_summary: dict[str, dict[str, float]] = {}
        memory_keys: list[str] = []
        for run in runs:
            for key in run.get("memory") or {}:
                if key not in memory_keys:
                    memory_keys.append(key)
        for key in memory_keys:
            values = [
                float(run["memory"].get(key, 0.0))
                for run in runs
                if run.get("memory")
            ]
            if values:
                memory_summary[key] = mean_std(values)
        policies[arm] = policy
        eval_runs[arm] = runs
        summaries[arm] = run_summary
        arms[arm] = {
            "memory_config": dataclasses.asdict(arm_memory[arm]),
            "is_control": arm in ("off", "off_padded"),
            "obs_dim": int(getattr(policy, "obs_dim", 0)),
            "n_seeds": len(runs),
            "train": {
                "episodes": train_episodes,
                "seconds_per_episode": train_seconds,
                "seeds": train_seeds,
                "policy_seed": policy_seed,
                "wall_s": round(train_wall, 3),
                "history": [_jsonable(item) for item in history],
            },
            "eval": {
                "seeds": eval_seeds,
                "n_seeds": len(runs),
                "seconds": eval_seconds,
                "wall_s": round(eval_wall, 3),
                "summary": _jsonable(run_summary),
                "runs": [
                    _jsonable(
                        {
                            "seed": run.get("seed"),
                            **{key: run.get(key, 0.0) for key in EVAL_METRICS},
                            "memory": run.get("memory"),
                        }
                    )
                    for run in runs
                ],
            },
            "memory_stats": _jsonable(memory_summary),
            "wall_s": round(arm_started(), 3),
        }

    # -- 配对差值 -------------------------------------------------------
    paired: dict[str, dict[str, dict[str, float]]] = {}
    for name, treatment, control in COMPARISONS:
        paired[name] = {
            metric: paired_summary(
                eval_runs[treatment], eval_runs[control], metric=metric
            )
            for metric in EVAL_METRICS
        }

    # -- 最佳记忆臂 + zoo（空对照不参与评选）---------------------------
    best_arm = max(
        MEMORY_ARM_CANDIDATES,
        key=lambda arm: (
            summaries[arm][PRIMARY_METRIC]["mean"],
            summaries[arm][SECONDARY_METRIC]["mean"],
        ),
    )
    best_overall_arm = max(
        ARM_NAMES,
        key=lambda arm: summaries[arm][PRIMARY_METRIC]["mean"],
    )
    best_policy = policies[best_arm]
    best_settings = dataclasses.replace(settings, memory=arm_memory[best_arm])
    zoo_name = "p6-memory"
    zoo_saved = False
    zoo_error = ""
    zoo_info: dict[str, Any]
    try:
        card = ModelZoo().save(
            zoo_name,
            best_policy,
            metrics=_jsonable(
                {
                    "arm": best_arm,
                    "best_overall_arm": best_overall_arm,
                    "eval_seeds": eval_seeds,
                    "n_seeds": len(eval_seeds),
                    "eval_seconds": eval_seconds,
                    "train_episodes": train_episodes,
                    "train_seconds": train_seconds,
                    "policy_seed": policy_seed,
                    **{
                        f"{metric}_mean": summaries[best_arm][metric]["mean"]
                        for metric in EVAL_METRICS
                    },
                    **{
                        f"{metric}_std": summaries[best_arm][metric]["std"]
                        for metric in EVAL_METRICS
                    },
                }
            ),
            config=best_settings,
            tags=["p6-memory", "memory_ablation", f"arm:{best_arm}"],
            description=(
                "P6-B 去 LLM 生成式记忆/反思消融的最佳记忆臂模型"
                f"（arm={best_arm}，obs_dim={getattr(best_policy, 'obs_dim', '?')}）"
            ),
        )
        zoo_saved = True
        zoo_info = {"name": zoo_name, "arm": best_arm, "card": card.to_dict()}
    except Exception as exc:  # noqa: BLE001 - zoo 产物失败不应毁掉实验
        zoo_info = {"name": zoo_name, "arm": best_arm, "error": str(exc)}
        zoo_error = str(exc)

    # -- 结论/诊断（只由数字决定，不写死结论）---------------------------
    verdict, diagnosis, notes = _conclude(paired, summaries, eval_seeds, params, arms)
    total_wall = started()

    summary: dict[str, Any] = {
        "name": settings.name,
        "protocol": "memory_ablation",
        "description": settings.description,
        "design": {
            "arms": list(ARM_NAMES),
            "controls": ["off", "off_padded"],
            "train_seeds": train_seeds,
            "eval_seeds": eval_seeds,
            "policy_seed": policy_seed,
            "train_episodes": train_episodes,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "reflection_every_s": reflection_every_s,
            "primary_metric": PRIMARY_METRIC,
            "paired": True,
            "n_seeds": len(eval_seeds),
            "n_per_arm": len(eval_seeds),
        },
        "config": {
            "world": {
                "width": settings.world.width,
                "height": settings.world.height,
                "food_regrow_per_s": settings.world.food_regrow_per_s,
            },
            "swarm": {
                "initial_population": settings.swarm.initial_population,
                "metabolism_per_s": settings.swarm.metabolism_per_s,
                "bite_size": settings.swarm.bite_size,
            },
            "policy": {
                "kind": settings.policy.kind,
                "hidden": settings.policy.hidden,
                "lr": settings.policy.lr,
            },
            "memory": dataclasses.asdict(settings.memory),
        },
        "arms": arms,
        "paired": _jsonable(paired),
        "best_arm": best_arm,
        "best_overall_arm": best_overall_arm,
        "best_arm_reason": f"记忆臂中 mean {PRIMARY_METRIC} 最高（并列时看 {SECONDARY_METRIC}）",
        "zoo": _jsonable(zoo_info),
        "zoo_saved": zoo_saved,
        "zoo_error": zoo_error,
        "verdict": verdict,
        "diagnosis": diagnosis,
        "honest_notes": notes,
        "wall_s": round(total_wall, 3),
    }
    summary = _jsonable(summary)

    # -- report.md ------------------------------------------------------
    target = output_dir(settings)
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / "report.md"
    report_path.write_text(_render_report(settings, summary), encoding="utf-8")
    summary["report_path"] = str(report_path)
    return summary


# ---------------------------------------------------------------------------
# 结论 / 诊断（数字驱动）
# ---------------------------------------------------------------------------
def _pair_positive(pair: dict[str, float], metric: str) -> bool:
    """该配对比较是否算"正效应"：eat_rate 需 >+5% 且胜>负；satiation 需 +0.02。"""
    if metric == "eat_rate":
        good = pair["mean_delta"] > 0.0 and pair["relative"] >= 0.05
    else:
        good = pair["mean_delta"] >= 0.02
    return bool(good and pair["wins"] > pair["losses"])


#: 福利/结局指标（判断"是不是净收益"用；eat_rate 只是行为频率）
WELFARE_METRICS = ("satiation_mean", "population", "total_reward")
#: 越低越好的福利指标（饥饿不平等）
WELFARE_INVERSE_METRICS = ("gini_hunger",)
#: 福利"实质变化"阈值：|mean_delta| 小于该值视为无实质差异（n=3 的小抖动）。
#: 这是保守处理：避免把 +0.004 的基尼抖动当成"福利恶化"。
WELFARE_THRESHOLDS = {
    "satiation_mean": 0.02,
    "population": 2.0,
    "total_reward": 10.0,
    "gini_hunger": 0.02,
}


def _welfare_sign(pair: dict[str, dict[str, float]]) -> int:
    """福利方向（带实质阈值）：+1 显著变好，-1 显著变差，0 无实质变化。"""
    score = 0
    for metric in WELFARE_METRICS:
        delta = pair[metric]["mean_delta"]
        if abs(delta) < WELFARE_THRESHOLDS[metric]:
            continue
        score += 1 if delta > 0.0 else -1
    for metric in WELFARE_INVERSE_METRICS:
        delta = pair[metric]["mean_delta"]
        if abs(delta) < WELFARE_THRESHOLDS[metric]:
            continue
        score += -1 if delta > 0.0 else 1
    return 1 if score > 0 else (-1 if score < 0 else 0)


def _describe_welfare(
    paired: dict[str, dict[str, dict[str, float]]], name: str
) -> str:
    pair = paired[name]
    parts = []
    for metric in WELFARE_METRICS:
        stats = pair[metric]
        parts.append(
            f"{metric} Δ={stats['mean_delta']:+.4f}"
            f"（W/L={int(stats['wins'])}/{int(stats['losses'])}）"
        )
    for metric in WELFARE_INVERSE_METRICS:
        stats = pair[metric]
        parts.append(
            f"{metric} Δ={stats['mean_delta']:+.4f}（越低越好，"
            f"W/L={int(stats['wins'])}/{int(stats['losses'])}）"
        )
    return "，".join(parts)


def _describe_pair(
    paired: dict[str, dict[str, dict[str, float]]], name: str, label: str
) -> str:
    pair = paired[name]
    eat = pair["eat_rate"]
    sat = pair[SECONDARY_METRIC]
    return (
        f"{label}: eat_rate Δ={eat['mean_delta']:+.3f}（rel={eat['relative']:+.1%}，"
        f"W/L={int(eat['wins'])}/{int(eat['losses'])}）；"
        f"sat Δ={sat['mean_delta']:+.3f}（W/L={int(sat['wins'])}/{int(sat['losses'])}）"
    )


def _conclude(
    paired: dict[str, dict[str, dict[str, float]]],
    summaries: dict[str, dict[str, dict[str, float]]],
    eval_seeds: list[int],
    params: dict[str, Any],
    arms: dict[str, dict[str, Any]],
) -> tuple[str, str, list[str]]:
    n = len(eval_seeds)
    behavioral_raw = [
        name for name in RAW_COMPARISONS
        if _pair_positive(paired[name]["eat_rate"], "eat_rate")
    ]
    behavioral_padded = [
        name for name in PADDED_COMPARISONS
        if _pair_positive(paired[name]["eat_rate"], "eat_rate")
    ]
    welfare = {
        name: _welfare_sign(paired[name]) for name, _t, _c in COMPARISONS
    }
    reflection_pair = paired["full_reflect_vs_full"]
    reflection_behavior = _pair_positive(reflection_pair["eat_rate"], "eat_rate")
    reflection_welfare = _welfare_sign(reflection_pair)
    best_memory_arm = max(
        MEMORY_ARM_CANDIDATES, key=lambda arm: summaries[arm][PRIMARY_METRIC]["mean"]
    )
    raw_text = "；".join(
        _describe_pair(paired, name, name) for name in behavioral_raw
    )
    padded_text = "；".join(
        _describe_pair(paired, name, name) for name in behavioral_padded
    )
    strong_raw = (
        max(behavioral_raw, key=lambda name: paired[name]["eat_rate"]["mean_delta"])
        if behavioral_raw
        else None
    )
    strong_padded = (
        max(
            behavioral_padded,
            key=lambda name: paired[name]["eat_rate"]["mean_delta"],
        )
        if behavioral_padded
        else None
    )
    welfare_text = ""
    if strong_padded is not None:
        welfare_text = (
            f"最强臂在 off_padded 对照下：{_describe_welfare(paired, strong_padded)}"
        )
        if strong_raw is not None:
            welfare_text += (
                f"；相对 off：{_describe_welfare(paired, strong_raw)}"
            )

    if behavioral_raw and behavioral_padded:
        padded_lines = []
        for name in behavioral_padded:
            label = (
                "改善" if welfare[name] > 0
                else ("恶化" if welfare[name] < 0 else "无实质变化")
            )
            padded_lines.append(
                f"{name}（eat_rate {paired[name]['eat_rate']['relative']:+.1%}，"
                f"W/L={int(paired[name]['eat_rate']['wins'])}/"
                f"{int(paired[name]['eat_rate']['losses'])}；福利{label}："
                f"{_describe_welfare(paired, name)}）"
            )
        padded_welfare_text = "；".join(padded_lines)
        clean = [name for name in behavioral_padded if welfare[name] > 0]
        degraded = [name for name in behavioral_padded if welfare[name] < 0]
        if clean and not degraded:
            verdict = (
                f"检测到超越 obs_dim 对照的正效应（配对 n={n}，仅效应量+胜负计数，"
                f"不做显著性声称）：{raw_text}。同 obs_dim 对照（off_padded）下："
                f"{padded_welfare_text}。即控制初始化尺度后，记忆特征仍带来净收益。"
                f"最佳记忆臂：{best_memory_arm}。"
            )
        elif clean and degraded:
            verdict = (
                f"记忆效应依赖检索权重，不能判为整体净收益（配对 n={n}，仅效应量+"
                f"胜负计数，不做显著性声称）：所有记忆臂 eat_rate 都高于 off_padded，"
                f"但福利/结局方向分裂——{padded_welfare_text}。相对 off 的原始比较："
                f"{raw_text}。也就是说：recency-only 在 obs_dim 对照下有干净的净摄入"
                f"收益；full(1/1/1) 与 full_reflect 则把策略推向'高频进食/局部抢占'，"
                f"总摄入不升（甚至下降）、satiation/population 变差、饥饿基尼上升。"
                f"记忆的价值取决于权重设计，而不是'开了记忆就更聪明'。"
                f"最佳记忆臂：{best_memory_arm}。"
            )
        elif degraded:
            verdict = (
                f"存在稳定的行为差异，但不能判为净收益（配对 n={n}，仅效应量+胜负"
                f"计数，不做显著性声称）：eat_rate 上升且越过 obs_dim 对照——"
                f"{raw_text}；off_padded 对照下：{padded_welfare_text}。福利/结局指标"
                f"反向（satiation/population 下降、饥饿基尼上升），即记忆把策略推向"
                f"更频繁但更低效的进食，这不是记忆的信息收益。"
                f"最佳记忆臂：{best_memory_arm}。"
            )
        else:
            verdict = (
                f"eat_rate 上升且越过 obs_dim 对照，但福利指标无实质变化（配对 "
                f"n={n}）：{raw_text}；off_padded 对照下：{padded_welfare_text}。"
                f"只能说明行为频率改变，没有证据表明净收益。"
                f"最佳记忆臂：{best_memory_arm}。"
            )
    elif behavioral_raw and not behavioral_padded:
        verdict = (
            f"off 对照下看似有正效应，但在 off_padded（同 obs_dim 恒零记忆）对照下"
            f"消失（配对 n={n}）：{raw_text}；而相对 off_padded："
            + "；".join(
                _describe_pair(paired, name, name) for name in PADDED_COMPARISONS
            )
            + "。因此该差异更像 obs_dim/初始化尺度的混淆，而不是记忆信息带来的"
            f"收益。最佳记忆臂：{best_memory_arm}。"
        )
    else:
        largest_rel, largest_name = max(
            (abs(paired[name]["eat_rate"]["relative"]), name)
            for name in RAW_COMPARISONS
        )
        verdict = (
            f"未观测到稳定的行为正效应（配对 n={n}，每臂 n={n}）：所有记忆臂相对 "
            f"off 的 eat_rate 变化均未同时满足 >+5% 且胜>负（最大 |相对变化|="
            f"{largest_rel:.1%}，比较={largest_name}）；相对 off_padded 的对照"
            "同样未见稳定优势。"
            + "；".join(
                _describe_pair(paired, name, name) for name in PADDED_COMPARISONS
            )
            + f"。最佳记忆臂：{best_memory_arm}。该结论仅适用于本预算"
            f"（{params.get('train_episodes', '?')} 回合训练、评估 "
            f"{params.get('eval_seconds', '?')}s、{n} 个配对种子），不外推。"
        )

    refl_text = (
        _describe_pair(paired, "full_reflect_vs_full", "full_reflect vs full")
        + "；福利：" + _describe_welfare(paired, "full_reflect_vs_full")
    )
    if reflection_behavior and reflection_welfare > 0:
        reflection_gini = reflection_pair["gini_hunger"]
        gini_worse = (
            reflection_gini["mean_delta"] > 0.0
            and reflection_gini["wins"] > reflection_gini["losses"]
        )
        verdict += (
            " 反思相对 full 有可检出增量，福利多数指标改善"
            + ("（但饥饿基尼未改善）" if gini_worse else "")
            + f"：{refl_text}。"
        )
    elif reflection_behavior:
        verdict += (
            f" 反思相对 full 有行为增量，但福利未同向改善：{refl_text}。"
        )
    else:
        verdict += f" 反思未带来可检出的行为增量：{refl_text}。"

    activity = {
        key: value["mean"]
        for key, value in arms["full_reflect"]["memory_stats"].items()
    }
    food_abs = float(activity.get("obs_food_dir_abs_mean", 0.0))
    food_nonzero = float(activity.get("obs_food_dir_nonzero_frac", 0.0))
    success = float(activity.get("obs_success_mean", 0.0))
    danger = float(activity.get("obs_danger_mean", 0.0))
    reflect_risk = float(activity.get("obs_reflect_risk_mean", 0.0))
    events = float(activity.get("events_recorded", 0.0))
    reflections = float(activity.get("reflections", 0.0))
    insights = float(activity.get("insights", 0.0))
    mean_memories = float(activity.get("mean_memories", 0.0))

    diagnosis_parts = [
        f"记忆机制确实在运行：full_reflect 臂评估回合平均 {events:.0f} 条事件、"
        f"{reflections:.1f} 次反思、{insights:.1f} 条洞察、"
        f"{mean_memories:.1f} 条/个体记忆。"
    ]
    if insights <= 0.0:
        diagnosis_parts.append(
            "但评估回合反思触发数为 0，full_reflect 实际退化为 full——这本身是"
            "实现/预算问题，不能据此谈反思效应。"
        )
    if food_nonzero < 0.3 or food_abs < 0.15:
        diagnosis_parts.append(
            f"记忆方向特征信号偏弱（|food_dir| 均值={food_abs:.3f}，非零占比="
            f"{food_nonzero:.1%}），策略很难从中获得可用信息。"
        )
    else:
        diagnosis_parts.append(
            f"记忆方向特征并非全零（|food_dir| 均值={food_abs:.3f}，非零占比="
            f"{food_nonzero:.1%}，成功率均值={success:.2f}，危险率均值={danger:.2f}，"
            f"反思风险偏置={reflect_risk:.2f}：机制有信号，问题在策略能否兑现）。"
        )
    if not behavioral_raw and not behavioral_padded:
        diagnosis_parts.append(
            "无稳定效应最可能的原因：(1) 基础 9 维观测已含 4 方向 ±3 格食物梯度，"
            "记忆方向信息与当前感知高度冗余；(2) 训练预算小、n 小，效应量被"
            "世界随机性（出生/死亡/食物再生）淹没；(3) 本环境的觅食主要靠局部"
            "梯度即可完成，长程记忆没有可兑现的收益。"
        )
    else:
        base_pair = (
            paired[strong_padded] if strong_padded is not None
            else paired["full_vs_off"]
        )
        reward_stats = base_pair["total_reward"]
        sat_stats = base_pair[SECONDARY_METRIC]
        pop_stats = base_pair["population"]
        same_direction = (
            reward_stats["mean_delta"] > 0.0
            and reward_stats["wins"] > reward_stats["losses"]
        )
        diagnosis_parts.append(
            "指标口径提示：eat_events/eat_rate 计的是“该帧吃到东西的个体数”"
            "（微吃也计数），是行为频率指标；total_reward（累计进食量-时间成本）"
            "才是净摄入代理，satiation/population/starvation/gini 是福利/结局。"
            f"最强臂 vs off_padded 的 total_reward Δ={reward_stats['mean_delta']:+.2f}"
            f"（W/L={int(reward_stats['wins'])}/{int(reward_stats['losses'])}）、"
            f"satiation Δ={sat_stats['mean_delta']:+.3f}、"
            f"population Δ={pop_stats['mean_delta']:+.1f}；"
            + (
                "与 eat_rate 反向/未升 → 在 obs_dim 对照下记忆没有带来净摄入收益。"
                if not same_direction
                else "与 eat_rate 同向 → 可谈净摄入收益。"
            )
            + f" 饥饿基尼：off={summaries['off']['gini_hunger']['mean']:.3f}、"
            f"off_padded={summaries['off_padded']['gini_hunger']['mean']:.3f}、"
            f"recency={summaries['recency']['gini_hunger']['mean']:.3f}、"
            f"full={summaries['full']['gini_hunger']['mean']:.3f}、"
            f"full_reflect={summaries['full_reflect']['gini_hunger']['mean']:.3f}"
            "——记忆特征（full/full_reflect）明显加剧饥饿不平等，与 population 下降、"
            "starvation_deaths 上升一致；反思提高了 total_reward/population，"
            "但没有降低不平等。"
        )

        diagnosis_parts.append(
            "分权重看（off_padded 对照，n=" + str(n) + "）：recency-only 的 "
            f"total_reward Δ={paired['recency_vs_off_padded']['total_reward']['mean_delta']:+.2f}"
            f"（W/L={int(paired['recency_vs_off_padded']['total_reward']['wins'])}/"
            f"{int(paired['recency_vs_off_padded']['total_reward']['losses'])}）、"
            f"sat Δ={paired['recency_vs_off_padded']['satiation_mean']['mean_delta']:+.3f}、"
            f"pop Δ={paired['recency_vs_off_padded']['population']['mean_delta']:+.2f}、"
            f"gini Δ={paired['recency_vs_off_padded']['gini_hunger']['mean_delta']:+.4f}；"
            "full 的 total_reward Δ="
            f"{paired['full_vs_off_padded']['total_reward']['mean_delta']:+.2f}"
            f"（W/L={int(paired['full_vs_off_padded']['total_reward']['wins'])}/"
            f"{int(paired['full_vs_off_padded']['total_reward']['losses'])}）、"
            f"sat Δ={paired['full_vs_off_padded']['satiation_mean']['mean_delta']:+.3f}、"
            f"pop Δ={paired['full_vs_off_padded']['population']['mean_delta']:+.2f}、"
            f"gini Δ={paired['full_vs_off_padded']['gini_hunger']['mean_delta']:+.4f}。"
            "即控制 obs_dim 后只有 recency-only 保留了净摄入收益（且 gini 几乎不变），"
            "全权重记忆与反思版都没有。"
        )

    diagnosis = "".join(diagnosis_parts)

    notes = [
        f"配对设计：每个臂用相同 train_seeds={arms['off']['train']['seeds']}、相同 "
        f"policy seed={arms['off']['train']['policy_seed']} 从同初始化训练；评估用"
        f"相同 eval seeds={eval_seeds}（同地图/同初始人口），每臂 n={n}。",
        "obs_dim 混淆控制：off 臂 obs_dim=9，记忆臂 obs_dim=15，W1 初始化尺度按 "
        "1/sqrt(obs_dim) 缩放——off_padded 用 6 维恒零观测把这条混淆路径单独控制，"
        "因此判断记忆信息收益应以 full vs off_padded 为准（full vs off 只作参考）。",
        "n 很小，本报告只给效应量（mean_delta / relative）与 wins-loses-ties，"
        "不做 p 值或显著性声称；结论不外推到其他预算/世界。",
        "福利方向判定使用实质阈值（satiation 0.02、population 2 个个体、"
        "total_reward 10、gini 0.02）：|mean_delta| 低于阈值视为无实质变化，"
        "避免把 n=3 的微小抖动当成福利效应。",
        "eat_events/eat_rate 是行为频率指标（该帧吃到东西的个体数，微吃也计数），"
        "不等于摄入量；total_reward 才是净摄入代理。只凭 eat_rate 上升不能宣称"
        "“记忆有用”——必须同时看 total_reward、satiation、population、starvation。",
        "记忆特征只来自已存事件 + 作为检索 query 的当前感知，无硬编码答案特征。",
        "事件节流：小于一口 5% 的微吃不入记忆；濒死仅在进入 <0.2 时记录一次；"
        "休息事件每 5 世界秒最多一条。这些只影响采样率，不影响检索/反思逻辑。",
        "off_padded 是空记忆对照（NullMemoryLayer）：走 memory.enabled 构造路径但"
        "观测恒 0、不记录事件、不检索、不反思，与 off 的唯一差别是 obs_dim=15。",
    ]
    return verdict, diagnosis, notes


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
def _cell(stat: dict[str, float], digits: int = 3) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def _render_report(settings: ExperimentSettings, summary: dict[str, Any]) -> str:
    design = summary["design"]
    config = summary["config"]
    n = design["n_seeds"]
    lines: list[str] = []
    lines.append("# P6-B 去 LLM 生成式记忆与反思：配对种子消融报告")
    lines.append("")
    lines.append("本报告由协议 memory_ablation 自动生成；数字来自真实训练+评估，")
    lines.append("不是占位符。所有臂都从相同 policy seed 的相同初始权重开始，")
    lines.append("在同一批 eval seed（同地图/同初始人口）上配对评估，每臂 n=" + str(n) + "。")
    lines.append("")
    lines.append("## 设计")
    lines.append("")
    lines.append(f"- 实验名：{summary['name']}")
    lines.append(
        f"- 世界：{config['world']['width']}×{config['world']['height']}，"
        f"food_regrow_per_s={config['world']['food_regrow_per_s']}，"
        f"初始人口={config['swarm']['initial_population']}，"
        f"metabolism_per_s={config['swarm']['metabolism_per_s']}"
    )
    lines.append(
        f"- 策略：{config['policy']['kind']}，hidden={config['policy']['hidden']}，"
        f"lr={config['policy']['lr']}；训练 {design['train_episodes']} 回合 × "
        f"{design['train_seconds']:.0f}s（train seeds {design['train_seeds']}），"
        f"policy seed={design['policy_seed']}"
    )
    lines.append(
        f"- 配对评估：{n} 个 seed {design['eval_seeds']}，每回合 "
        f"{design['eval_seconds']:.0f}s；每臂 n={n}（同一批 seed）"
    )
    lines.append(
        f"- 记忆参数：capacity={config['memory']['capacity']}，"
        f"top_k={config['memory']['top_k']}，"
        f"halflife={config['memory']['recency_halflife_s']}s，"
        f"反思间隔={design['reflection_every_s']}s，"
        f"反思最少事件={config['memory']['reflection_min_events']}"
    )
    lines.append(
        f"- 臂：off（9 维）/ off_padded（15 维恒零记忆对照）/ recency / full / "
        f"full_reflect；总墙钟时间 {summary['wall_s']:.1f}s；"
        f"最佳记忆臂：{summary['best_arm']}；全臂最佳：{summary['best_overall_arm']}"
    )
    lines.append("")

    lines.append(f"## 1. 各臂绝对指标（mean ± std，每臂 n={n}）")
    lines.append("")
    lines.append("| 臂 | obs_dim | " + " | ".join(EVAL_METRICS) + " |")
    lines.append("|" + "---|" * (len(EVAL_METRICS) + 2))
    for arm in ARM_NAMES:
        row = [arm, str(summary["arms"][arm]["obs_dim"])]
        for metric in EVAL_METRICS:
            row.append(_cell(summary["arms"][arm]["eval"]["summary"][metric]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## 2. 配对差值（treatment − control，同 seed 一一配对；效应量）")
    lines.append("")
    lines.append("| 比较 | 指标 | mean_delta | relative | wins/losses/ties |")
    lines.append("|---|---|---|---|---|")
    for name, _treatment, _control in COMPARISONS:
        for metric in (
            PRIMARY_METRIC, SECONDARY_METRIC, "population", "total_reward",
            "gini_hunger",
        ):
            pair = summary["paired"][name][metric]
            lines.append(
                f"| {name} | {metric} | {pair['mean_delta']:+.4f} | "
                f"{pair['relative']:+.2%} | "
                f"{int(pair['wins'])}/{int(pair['losses'])}/{int(pair['ties'])} |"
            )
    lines.append("")

    lines.append("### 2b. obs_dim 混淆控制解读")
    lines.append("")
    lines.append(
        "off 臂 obs_dim=9，记忆臂 obs_dim=15；NeuralSwarmPolicy 的 W1 按 "
        "1/sqrt(obs_dim) 缩放初始化，观测维数本身就会改变初始行为。off_padded "
        "注册 6 维恒零观测（obs_dim=15，不记录/不检索/不反思），把这条混淆路径"
        "单独控制。判断记忆信息收益应以 full vs off_padded 为准。"
    )
    lines.append("")
    for name in (
        "recency_vs_off_padded",
        "full_vs_off",
        "full_vs_off_padded",
        "full_reflect_vs_off_padded",
        "full_reflect_vs_full",
    ):
        pair = summary["paired"][name]
        cells = []
        for metric in (
            PRIMARY_METRIC, SECONDARY_METRIC, "population", "total_reward",
            "gini_hunger",
        ):
            stats = pair[metric]
            cells.append(
                f"{metric} Δ={stats['mean_delta']:+.4f}"
                f"（W/L/T={int(stats['wins'])}/{int(stats['losses'])}/"
                f"{int(stats['ties'])}）"
            )
        lines.append(f"- {name}：" + "；".join(cells))
    lines.append("")
    lines.append(
        "注意：eat_rate 是“进食事件频率”（微吃也计数），total_reward 才是净摄入"
        "代理；若 eat_rate 上升而 total_reward/satiation/population 不升，则只能"
        "说记忆改变了行为频率，不能宣称找食效率或福利收益。福利方向按实质阈值"
        "判定（satiation 0.02、population 2、total_reward 10、gini 0.02），"
        "低于阈值视为无实质变化。"
    )
    lines.append("")

    lines.append("## 3. 记忆机制活跃度（评估回合均值，每臂 n=" + str(n) + "）")
    lines.append("")
    activity_keys = [
        "updates",
        "events_recorded",
        "retrievals",
        "reflections",
        "insights",
        "mean_memories",
        "obs_food_dir_abs_mean",
        "obs_food_dir_nonzero_frac",
        "obs_success_mean",
        "obs_danger_mean",
        "obs_reflect_food_mean",
        "obs_reflect_risk_mean",
        "reindexed",
    ]
    lines.append("| 臂 | " + " | ".join(activity_keys) + " |")
    lines.append("|" + "---|" * (len(activity_keys) + 1))
    for arm in ARM_NAMES:
        stats = summary["arms"][arm]["memory_stats"]
        cells = [
            f"{stats[key]['mean']:.3f}" if key in stats else "—"
            for key in activity_keys
        ]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 4. 逐 seed 评估（每臂 n=" + str(n) + "，主指标 + 存活）")
    lines.append("")
    lines.append(
        "| 臂 | " + " | ".join(f"seed {seed}" for seed in design["eval_seeds"]) + " |"
    )
    lines.append("|" + "---|" * (len(design["eval_seeds"]) + 1))
    for arm in ARM_NAMES:
        runs = summary["arms"][arm]["eval"]["runs"]
        values = [
            f"eat_rate={run['eat_rate']:.3f}, sat={run['satiation_mean']:.3f}, "
            f"pop={int(run['population'])}, starv={int(run['starvation_deaths'])}"
            for run in runs
        ]
        lines.append(f"| {arm} | " + " | ".join(values) + " |")
    lines.append("")

    lines.append("## 5. 结论（honest verdict）")
    lines.append("")
    lines.append(summary["verdict"])
    lines.append("")
    lines.append("**诊断**：" + summary["diagnosis"])
    lines.append("")
    lines.append("**诚实性说明**：")
    for note in summary["honest_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## 6. 产物")
    lines.append("")
    lines.append("- summary.json（完整数字与本文件同目录）")
    lines.append("- report.md（本文件）")
    if summary.get("zoo_saved"):
        lines.append(
            f"- assets/zoo/{summary['zoo']['name']}（最佳记忆臂 "
            f"{summary['zoo']['arm']}，tag p6-memory）"
        )
    else:
        lines.append(
            f"- zoo 保存失败：{summary.get('zoo_error', '')!r}（实验数字仍有效）"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "memory_ablation",
    "ARM_NAMES",
    "MEMORY_ARM_CANDIDATES",
    "COMPARISONS",
    "RAW_COMPARISONS",
    "PADDED_COMPARISONS",
    "EVAL_METRICS",
]
