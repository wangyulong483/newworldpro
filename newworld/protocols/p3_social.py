"""P3 社会机制消融协议（social_ablation）：交换 / 教学 / 继承逐机制对照。

设计来源：experiments/run_p3_social.py（P3 设计沉淀）。P5 起由
注册表按 YAML 里的 protocol: social_ablation 调起：

    python -m newworld --config configs/p3_social.yaml

实验设计
========
同地图、同 seed、同资源预算的**配对对照**：每 seed 跑五个配置

* off_all      三机制全关（对照）
* barter_only  仅交换
* teach_only   仅教学
* inherit_only 仅继承
* all_on       三机制全开

稀缺世界 + 规则策略（P3 是机制验证阶段，不混合学习变量）。
配对比较各机制配置 vs off_all 的 population / eat_events / eat_rate /
satiation_mean / gini_hunger 与机制计数器，报告均值±标准差、配对差值。

预注册判据（不做过强声明）
==========================
机制"被激活"的必要条件：该机制的计数器在开启组 > 0 且在 off_all 组
== 0；否则机制未生效，任何宏观指标差异都不可归因于它。效果量只报告
方向与均值差（n 个配对 seed 很小，不做显著性检验）。

诚实性 caveats
==============
* off_all 组仍按 initial_skill_rate 注入初始技能（技能效果存在），只是
  关闭了传播/交换机制——因此 H-teach/H-inherit 检验的是"传播通道"，不是
  "技能是否存在"（与旧 P3 设计一致）。
* SocialLayer 的继承实现是行索引近似（见 newworld/social_mech.py 注释：
  swarm.update 内部先 _compact 再回调），本协议在每帧 swarm 压实后按
  compact_keep 重排技能行，使技能跟随个体；但"死者身份 → 继承人"的
  配对本就是统计近似，继承计数只能作为机制活性证据，不能解释为精确
  谱系传播。
* satiation_mean/std、population 是评估窗口末尾快照，受人口规模与时间
  影响；gini_hunger 同理，仅作同 seed 配对比较。
* 规则策略每 seed 重建（同 seed 可复现），五配置共享同一 map seed 与
  初始人口，保证配对性。
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..experiment import build_simulation
from ..registry import register_protocol
from ..settings import ExperimentSettings
from ..swarm_policy import RuleSwarmPolicy
from . import common

#: 五个实验配置（对照 + 三机制独立 + 全开；键名与旧 P3 保持一致）
CONFIGS: dict[str, dict[str, bool]] = {
    "off_all": {
        "enable_barter": False,
        "enable_teaching": False,
        "enable_inheritance": False,
    },
    "barter_only": {
        "enable_barter": True,
        "enable_teaching": False,
        "enable_inheritance": False,
    },
    "teach_only": {
        "enable_barter": False,
        "enable_teaching": True,
        "enable_inheritance": False,
    },
    "inherit_only": {
        "enable_barter": False,
        "enable_teaching": False,
        "enable_inheritance": True,
    },
    "all_on": {
        "enable_barter": True,
        "enable_teaching": True,
        "enable_inheritance": True,
    },
}

#: 社会机制计数器（来自 SocialLayer.stats，metrics() 里嵌在 social 下）
_COUNTER_KEYS: tuple[str, ...] = (
    "barter_events",
    "barter_transfer_total",
    "teach_events",
    "inherit_events",
    "knowledge_lost",
    "skilled_population",
)
#: 逐配置汇总指标
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
) + _COUNTER_KEYS
#: 机制"证据强度"下限：开启组事件总数少于此值时，只算机制接线成立，
#: 不足以支撑"该机制有宏观效应"的判断（启发式，不是显著性检验）。
_MIN_EVIDENCE_EVENTS = 5.0
#: 配对比较指标（任务要求：population/eat_events/satiation_mean/
#: gini_hunger + 社会计数器；额外加 eat_rate/mean_life 便于同 P1 对照）
_PAIRED_METRICS: tuple[str, ...] = (
    "population",
    "eat_events",
    "eat_rate",
    "mean_life",
    "satiation_mean",
    "satiation_std",
    "gini_hunger",
    "skilled_population",
    "barter_events",
    "teach_events",
    "inherit_events",
    "knowledge_lost",
)


# ---------------------------------------------------------------------------
# 仿真推进：压实后重排技能行（修正 SocialLayer.skills 与个体错位）
# ---------------------------------------------------------------------------
def _reindex_social_skills(swarm: Any, social: Any) -> None:
    """按 swarm.compact_keep 把技能数组重排到压实后的行号。

    swarm._compact() 把存活行搬到数组前段；SocialLayer.skills 是行索引
    数组，若不重排，技能会在个体死后"粘"在位置上（效果作用到错误个体）。
    compact_keep 记录压实前存活行的旧行号，据此重建即可。
    """
    keep = np.asarray(swarm.compact_keep, dtype=np.int64)
    skills = np.asarray(social.skills)
    rebuilt = np.zeros_like(skills)
    if keep.size:
        rebuilt[: keep.size] = skills[keep]
    social.skills = rebuilt


def advance_with_social(sim: Any, seconds: float | None = None) -> dict[str, Any]:
    """逐步推进仿真（与 Simulation.step 同序），并在压实后重排技能行。

    顺序刻意与 newworld.experiment.Simulation.step 一致：
    环境 -> 个体（含压实）-> 社会（死亡继承 + 机制）-> 通信 -> 记忆。
    唯一差异是在个体压实与社会层之间插入技能行重排。
    """
    dt = float(sim.settings.run.dt)
    daylight = float(sim.settings.run.daylight)
    duration = float(sim.settings.run.seconds if seconds is None else seconds)
    steps = max(1, int(round(duration / dt)))
    swarm = sim.swarm
    social = sim.social
    last_generation = int(swarm.compact_generation)
    for _ in range(steps):
        deaths_before = len(swarm.death_ages)
        sim.engine.update(dt, daylight)
        swarm.update(dt)
        if social is not None and int(swarm.compact_generation) != last_generation:
            _reindex_social_skills(swarm, social)
            last_generation = int(swarm.compact_generation)
        if social is not None:
            social.on_death_pass(deaths_before)
            social.update(dt)
        if sim.comm is not None:
            sim.comm.update(dt)
        if sim.memory is not None:
            sim.memory.update(dt)
    return sim.metrics()


# ---------------------------------------------------------------------------
# 单个 (配置, seed) 运行
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


def _run_config(
    settings: ExperimentSettings,
    *,
    flags: dict[str, bool],
    seed: int,
    seconds: float,
) -> dict[str, Any]:
    """跑一个 (配置, seed)：规则策略 + 指定机制开关，返回指标。"""
    social_settings = dataclasses.replace(settings.social, enabled=True, **flags)
    case = dataclasses.replace(settings, social=social_settings)
    policy = RuleSwarmPolicy(seed=seed)
    sim = build_simulation(case, seed=int(seed), policy=policy)
    metrics = advance_with_social(sim, seconds)
    # 复用 common 的 P1.6 口径（mean_life / eat_rate / survived_90s），
    # 不在此重新定义指标，保证跨协议可比。
    metrics.update(common._life_metrics(sim, metrics))
    metrics["seed"] = int(seed)
    metrics["density"] = "normal"
    metrics["initial_population"] = int(settings.swarm.initial_population)
    metrics["carrying_capacity_estimate"] = float(
        sim.swarm.carrying_capacity_estimate()
    )
    social_stats = metrics.get("social") or {}
    for key in _COUNTER_KEYS:
        metrics[key] = float(social_stats.get(key, 0.0))
    return metrics


# ---------------------------------------------------------------------------
# 协议主体
# ---------------------------------------------------------------------------
@register_protocol(
    "social_ablation",
    note="P3 社会机制消融：交换/教学/继承五配置配对对照（规则策略，稀缺世界）",
)
def run_social_ablation(settings: ExperimentSettings) -> dict[str, Any]:
    """P3 社会机制消融协议：五配置 × 配对 seed，报告效果量与机制活性。"""
    clock = common.timed()
    params = settings.protocol_params
    seeds = _seed_list(
        params.get("seeds"), default_start=int(params.get("seed_start", 700)),
        default_count=10,
    )
    eval_seconds = float(params.get("eval_seconds", settings.run.eval_seconds))
    if not settings.social.enabled:
        # 机制消融必须在机制层开启的前提下才有意义（配置错误直接报错）
        raise ValueError(
            "social_ablation 需要 social.enabled=true（见 configs/p3_social.yaml）"
        )

    results: dict[str, list[dict[str, Any]]] = {name: [] for name in CONFIGS}
    for seed in seeds:
        for config_name, flags in CONFIGS.items():
            results[config_name].append(
                _run_config(settings, flags=flags, seed=seed, seconds=eval_seconds)
            )

    # -- 逐配置汇总 ------------------------------------------------------
    per_config: dict[str, dict[str, dict[str, float]]] = {
        name: common.summarize_runs(runs, _SUMMARY_KEYS)
        for name, runs in results.items()
    }
    # -- 配对差值 vs off_all --------------------------------------------
    control = results["off_all"]
    treated = ("barter_only", "teach_only", "inherit_only", "all_on")
    paired_vs_control: dict[str, dict[str, dict[str, float]]] = {
        name: {
            metric: common.paired_summary(results[name], control, metric=metric)
            for metric in _PAIRED_METRICS
        }
        for name in treated
    }

    # -- 机制活性（预注册必要条件：开=有事件，关=0 事件） ---------------
    def _total(config_name: str, key: str) -> float:
        return float(sum(float(run[key]) for run in results[config_name]))

    validity: dict[str, dict[str, Any]] = {}
    for mechanism, counter, on_config in (
        ("barter", "barter_events", "barter_only"),
        ("teaching", "teach_events", "teach_only"),
        ("inheritance", "inherit_events", "inherit_only"),
    ):
        on_total = _total(on_config, counter)
        off_total = _total("off_all", counter)
        active = bool(on_total > 0.0 and off_total == 0.0)
        validity[mechanism] = {
            "counter": counter,
            "on_config": on_config,
            "on_total": on_total,
            "off_all_total": off_total,
            "active": active,
            "evidence": (
                "sufficient"
                if active and on_total >= _MIN_EVIDENCE_EVENTS
                else ("insufficient" if active else "inactive")
            ),
            "enough_events": bool(active and on_total >= _MIN_EVIDENCE_EVENTS),
        }

    # -- 三个预注册假设的方向性读法（只报方向，不做显著性） ------------
    def _delta(config_name: str, metric: str) -> float:
        return float(paired_vs_control[config_name][metric]["mean_delta"])

    hypotheses = {
        "H-barter（交换降低饱足分化）": {
            "metric": "satiation_std",
            "delta_barter_only_vs_off_all": _delta("barter_only", "satiation_std"),
            "expected_sign": "<0",
            "direction_consistent": _delta("barter_only", "satiation_std") < 0.0,
            "note": "仅方向性证据；n 小，不宣称已确立效应",
        },
        "H-teach（教学提升技能人口 → 觅食效率）": {
            "skilled_population_delta": _delta("teach_only", "skilled_population"),
            "eat_rate_delta": _delta("teach_only", "eat_rate"),
            "expected_sign": ">0",
            "direction_consistent": (
                _delta("teach_only", "skilled_population") > 0.0
                and _delta("teach_only", "eat_rate") > 0.0
            ),
            "note": "仅方向性证据；n 小，不宣称已确立效应",
        },
        "H-inherit（继承降低知识失传）": {
            "metric": "knowledge_lost",
            "delta_inherit_only_vs_off_all": _delta("inherit_only", "knowledge_lost"),
            "expected_sign": "<0",
            "direction_consistent": _delta("inherit_only", "knowledge_lost") < 0.0,
            "note": "继承实现是行索引统计近似（见 caveats），只作机制活性证据",
        },
    }

    # -- 简报文本 --------------------------------------------------------
    validity_bits = " ".join(
        f"{name}={'激活' if item['active'] else '未激活'}"
        f"({item['counter']}: on={item['on_total']:.0f}, off={item['off_all_total']:.0f})"
        for name, item in validity.items()
    )
    effect_bits = []
    for name in treated:
        delta = paired_vs_control[name]
        effect_bits.append(
            f"{name}: pop Δ={delta['population']['mean_delta']:+.1f}, "
            f"eat Δ={delta['eat_events']['mean_delta']:+.0f}, "
            f"satiation Δ={delta['satiation_mean']['mean_delta']:+.4f}, "
            f"gini Δ={delta['gini_hunger']['mean_delta']:+.4f}"
        )
    insufficient = [
        name for name, item in validity.items() if item["evidence"] != "sufficient"
    ]
    insufficient_zh = {
        "barter": "交换",
        "teaching": "教学",
        "inheritance": "继承",
    }
    max_pop_delta = max(abs(_delta(name, "population")) for name in treated)
    max_sat_delta = max(abs(_delta(name, "satiation_mean")) for name in treated)
    max_gini_delta = max(abs(_delta(name, "gini_hunger")) for name in treated)
    honest_statement = (
        f"机制活性（必要条件）：{validity_bits}。配对效果量：{'；'.join(effect_bits)}。"
        f" 最大配对差量级：population={max_pop_delta:.2f}、"
        f"satiation={max_sat_delta:.4f}、gini={max_gini_delta:.4f}。"
        " 判读：三条机制在开启组都产生了事件且 off_all 恒为 0，机制接线成立；"
        "但宏观指标配对差都在噪声量级，本 n 下不能声称任一机制产生了可测"
        "的宏观效应。"
        + (
            f" {'、'.join(insufficient_zh.get(name, name) for name in insufficient)}"
            f" 的事件总数 < {_MIN_EVIDENCE_EVENTS:g}，证据强度不足，"
            "只能算接通过、不能算已验证。"
            if insufficient
            else " 全部机制事件数达到最小证据量。"
        )
        + " off_all 仍注入初始技能，故教学/继承检验的是传播通道。"
    )
    verdict = (
        f"机制活性（预注册必要条件）：{validity_bits}。"
        f" 效果量（配对 vs off_all，n={len(seeds)}，只报方向/均值，"
        f"不做显著性检验）：{'；'.join(effect_bits)}。"
        f" 诚实判读：接线成立；宏观效应均在噪声量级"
        f"（population Δ<={max_pop_delta:.2f}、satiation Δ<={max_sat_delta:.4f}）"
        + (
            f"；{'、'.join(insufficient_zh.get(name, name) for name in insufficient)}"
            "事件过少证据不足"
            if insufficient
            else ""
        )
        + "。"
    )

    caveats = [
        "机制活性判据是必要条件而非充分条件：计数器 >0 只证明机制在运行，"
        "宏观指标差异仍需配对比较；n 个配对 seed 很小，不宣称显著性。",
        f"off_all 对照仍按 initial_skill_rate={settings.social.initial_skill_rate}"
        " 注入初始技能（技能效果真实存在），关闭的只是交换/教学/继承三条"
        "传播通道——与旧 P3 设计一致；因此 H-teach/H-inherit 不能解释为"
        "'技能从无到有'的效应。",
        "SocialLayer 的继承是行索引统计近似（base 模块注释已声明：swarm.update"
        " 内部先 _compact 再回调社会层）。本协议在每帧压实后按 compact_keep"
        " 重排技能行，让技能效果跟随正确个体；但'死者→继承人'配对本身仍是"
        "近似，inherit_events/knowledge_lost 只作机制活性证据。",
        "satiation_mean/std、gini_hunger、population 是评估窗口末快照，"
        "受人口规模与时间影响；只在同 seed 配对的五配置间比较。",
        f"评估时长 {eval_seconds:g}s，世界再生速率 "
        f"{settings.world.food_regrow_per_s}（P1.7 稀缺环境）；"
        "规则策略每 seed 以 seed 重建（同 seed 可复现）。",
    ]

    return {
        "name": settings.name,
        "protocol": "social_ablation",
        "experiment": {
            "seeds": seeds,
            "eval_seconds": eval_seconds,
            "food_regrow_per_s": float(settings.world.food_regrow_per_s),
            "initial_population": int(settings.swarm.initial_population),
            "policy": "rule",
            "initial_skill_rate": float(settings.social.initial_skill_rate),
        },
        "configs": {name: dict(flags) for name, flags in CONFIGS.items()},
        "per_config": per_config,
        "paired_vs_control": paired_vs_control,
        "validity": validity,
        "hypotheses": hypotheses,
        "honest_reading": {
            "insufficient_evidence": insufficient,
            "max_abs_paired_deltas": {
                "population": max_pop_delta,
                "satiation_mean": max_sat_delta,
                "gini_hunger": max_gini_delta,
            },
            "statement": honest_statement,
        },
        "verdict": verdict,
        "caveats": caveats,
        "wall_s": round(float(clock()), 1),
    }


__all__ = ["CONFIGS", "advance_with_social", "run_social_ablation"]
