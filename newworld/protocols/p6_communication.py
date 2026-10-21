"""P6-A 涌现通信实验协议：指称博弈演示 + 世界内信号通道诚实消融。

实验设计（对应共享任务 task-4）
==============================
Part A（可控、能真正展示语义协议形成）
    Lazaridou 式指称博弈：cue_dim 个 one-hot cue，sender 发 n_symbols 选一
    符号，listener 由符号预测 cue，共享 0/1 奖励；每个 game_seed 独立初始化
    训练 game_episodes 轮。报告：
    * accuracy 均值±标准差 vs 随机机会水平 1/cue_dim；
    * 符号-线索互信息 MI（bits，上限 log2(n_symbols)）与符号熵；
    * cue->symbol 映射、映射在跨 seed 的稳定性（允许符号重命名，因为符号
      标签本身任意的：mapping_similarity 做最优置换对齐）。

Part B（世界内通道：honest ablation）
    同一训练预算、同一训练 seed 序列，只切换 comm.enabled：
    * off 臂：观测 9 维（无通信）；
    * on  臂：观测 9+K 维（听到的邻居符号直方图），且训练时 sender 用
      "1 步群体奖励 = Δeat_events/人口" 做内部 REINFORCE；评估时 comm.train
      =False 冻结 sender（用训练后的权重复制），同一运动策略在各 eval_seed
      上配对评估。报告 eat_events / eat_rate / satiation_mean 的配对差
      （treat-control，含 wins/losses/n）以及 on 臂评估 rollout 的通道统计
      （MI/熵/对齐率）与有限样本 MI 偏置。

Verification 口径（不做事后调参的"造正结果"）
--------------------------------------------
* Part A 判据事先固定：mean accuracy > chance + 0.10 且 > 0.5 才算"协议形成"；
* Part B 若配对差与 0 不可区分（n=5，未做显著性检验）就写 null；通道 MI 与
  有限样本偏置同量级即判"无可辨语义"。世界通道在"个体奖励+共享策略"下本来
  就缺少 per-sender 信用分配，学习出语义不是必然——负结果同样写入 summary。

产物：summary.json 由 experiment.run_experiment 统一写入
experiments/results/<name>/；on 臂模型/条件 sender 存入 assets/zoo（tag p6-comm）。
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np

from ..communication import (
    DEFAULT_ROLLING,
    ReferentialGame,
    build_communication_layer,  # noqa: F401  (文档/导出用)
    mapping_similarity,
    mi_expected_bias_bits,
    use_communication_sender,
)
from ..registry import register_protocol
from ..settings import ExperimentSettings
from ..swarm_policy import NeuralSwarmPolicy, OBS_DIM
from . import common


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _plain(value: Any) -> Any:
    """numpy 标量/数组 -> JSON 可序列化的 Python 原生类型。"""
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _int_list(values: Any, default: tuple[int, ...]) -> list[int]:
    if values is None:
        return list(default)
    return [int(value) for value in values]


def _arm_settings(settings: ExperimentSettings, *, on: bool, train: bool) -> ExperimentSettings:
    """返回只改 comm 开关的配置副本（其它字段完全一致，配对对照）。"""
    return dataclasses.replace(
        settings,
        comm=dataclasses.replace(settings.comm, enabled=bool(on), train=bool(train)),
    )


def _numeric_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0.0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "n": float(array.size),
    }


# ---------------------------------------------------------------------------
# Part A：控制指称博弈
# ---------------------------------------------------------------------------
def run_referential_part(settings: ExperimentSettings) -> dict[str, Any]:
    params = dict(settings.protocol_params or {})
    cue_dim = int(params.get("cue_dim", 4))
    n_symbols = int(params.get("game_n_symbols", settings.comm.n_symbols))
    game_seeds = _int_list(params.get("game_seeds"), (0, 1, 2, 3, 4))
    episodes = int(params.get("game_episodes", 400))
    batch = int(params.get("game_batch", 64))
    eval_n = int(params.get("game_eval_n", 2000))
    lr = float(params.get("game_lr", 0.5))
    hidden = int(params.get("game_hidden", settings.policy.hidden))

    per_seed: list[dict[str, Any]] = []
    untrained_accuracies: list[float] = []
    started = time.perf_counter()
    for seed in game_seeds:
        game = ReferentialGame(
            cue_dim=cue_dim,
            n_symbols=n_symbols,
            hidden=hidden,
            seed=int(seed),
            lr=lr,
        )
        untrained = game.evaluate(n=min(eval_n, max(200, eval_n // 4)))
        game.train(episodes=episodes, batch=batch)
        result = game.evaluate(n=eval_n)
        entry = {
            "seed": int(seed),
            "untrained_accuracy": float(untrained["accuracy"]),
            "accuracy": float(result["accuracy"]),
            "chance": float(result["chance"]),
            "mutual_information_bits": float(result["mutual_information_bits"]),
            "symbol_entropy_bits": float(result["symbol_entropy"]),
            "mapping": _plain(result["mapping"]),
            "n_eval": int(result["n"]),
            "updates": int(game.updates),
            "train_accuracy_last": float(game.history[-1]["train_accuracy"]) if game.history else 0.0,
        }
        per_seed.append(entry)
        untrained_accuracies.append(float(untrained["accuracy"]))

    accuracy = _numeric_stats([entry["accuracy"] for entry in per_seed])
    mi = _numeric_stats([entry["mutual_information_bits"] for entry in per_seed])
    entropy = _numeric_stats([entry["symbol_entropy_bits"] for entry in per_seed])
    untrained = _numeric_stats(untrained_accuracies)
    chance = float(1.0 / cue_dim)
    margin = accuracy["mean"] - chance
    protocol_formed = bool(margin >= 0.10 and accuracy["mean"] > 0.5)

    # 映射稳定性：符号标签任意 => 允许符号重命名后的最优一致率
    injective = 0
    for entry in per_seed:
        mapping = entry["mapping"]
        values = [int(value) for value in mapping.values()]
        if len(set(values)) == len(values) and min(values) >= 0:
            injective += 1
    similarities: list[float] = []
    for i in range(len(per_seed)):
        for j in range(i + 1, len(per_seed)):
            similarities.append(
                mapping_similarity(
                    per_seed[i]["mapping"], per_seed[j]["mapping"], n_symbols
                )
            )
    raw_identical = 0
    for i in range(len(per_seed)):
        for j in range(i + 1, len(per_seed)):
            if per_seed[i]["mapping"] == per_seed[j]["mapping"]:
                raw_identical += 1
    pairs = max(1, len(similarities))

    return _plain({
        "cue_dim": cue_dim,
        "n_symbols": n_symbols,
        "game_seeds": game_seeds,
        "game_episodes": episodes,
        "game_batch": batch,
        "game_lr": lr,
        "hidden": hidden,
        "chance_accuracy": chance,
        "mi_upper_bound_bits": float(np.log2(max(2, n_symbols))),
        "untrained_accuracy": untrained,
        "accuracy": accuracy,
        "accuracy_margin_over_chance": margin,
        "mutual_information_bits": mi,
        "symbol_entropy_bits": entropy,
        "mapping_injective_fraction": float(injective / max(1, len(per_seed))),
        "mapping_relabeled_agreement": float(np.mean(similarities)) if similarities else 1.0,
        "mapping_raw_identical_fraction": float(raw_identical / pairs),
        "protocol_formed": protocol_formed,
        "criterion": "mean accuracy > chance + 0.10 and > 0.5",
        "per_seed": per_seed,
        "wall_s": round(time.perf_counter() - started, 3),
    })


# ---------------------------------------------------------------------------
# Part B：世界内信号通道 off/on 配对消融
# ---------------------------------------------------------------------------
def _summarize_channel(runs: list[dict[str, Any]], n_symbols: int) -> dict[str, Any]:
    """汇总 on 臂评估 rollout 的通道统计（每个 run 的 comm stats 是滚动量）。"""
    keys = ("mutual_information", "symbol_entropy", "food_alignment", "train_updates", "messages")
    collected = {key: [] for key in keys}
    for run in runs:
        comm = run.get("comm") or {}
        for key in keys:
            if key in comm:
                collected[key].append(float(comm[key]))
    summary = {key: _numeric_stats(values) for key, values in collected.items()}
    summary["mi_bits"] = summary.pop("mutual_information")
    summary["entropy_bits"] = summary.pop("symbol_entropy")
    summary["alignment"] = summary.pop("food_alignment")
    summary["rolling_samples"] = float(DEFAULT_ROLLING)
    summary["mi_expected_bias_bits"] = float(
        mi_expected_bias_bits(DEFAULT_ROLLING, n_symbols, n_symbols)
    )
    summary["alignment_chance"] = float(1.0 / n_symbols)
    # 评估阶段 sender 被冻结：train_updates 必须为 0
    updates = collected.get("train_updates", [])
    summary["sender_frozen"] = bool(all(value == 0.0 for value in updates)) if updates else True
    return _plain(summary)


def run_swarm_ablation_part(settings: ExperimentSettings) -> dict[str, Any]:
    params = dict(settings.protocol_params or {})
    train_seeds = _int_list(params.get("train_seeds"), (1000, 1001, 1002))
    train_episodes = int(params.get("train_episodes", 6))
    train_episode_seconds = float(
        params.get("train_episode_seconds", settings.run.seconds)
    )
    train_policy_seed = int(params.get("train_policy_seed", 7))
    sender_seed = int(params.get("sender_seed", 12345))
    eval_seeds = _int_list(params.get("eval_seeds"), (2000, 2001, 2002, 2003, 2004))
    eval_seconds = float(params.get("eval_seconds", settings.run.eval_seconds))
    hidden = int(settings.policy.hidden)
    n_symbols = int(settings.comm.n_symbols)

    started = time.perf_counter()
    trained: dict[str, Any] = {}
    for arm in ("off", "on"):
        on = arm == "on"
        train_settings = _arm_settings(settings, on=on, train=True)
        sender: NeuralSwarmPolicy | None = None
        sender_before: np.ndarray | None = None
        if on:
            sender = NeuralSwarmPolicy(
                seed=sender_seed, obs_dim=OBS_DIM, n_actions=n_symbols, hidden=hidden
            )
            sender_before = sender.flat_weights().copy()
            context = use_communication_sender(sender)
        else:
            context = None
        arm_started = time.perf_counter()
        if context is not None:
            with context:
                policy, history = common.train_neural_policy(
                    train_settings,
                    episodes=train_episodes,
                    episode_seconds=train_episode_seconds,
                    train_seeds=train_seeds,
                    seed=train_policy_seed,
                    hidden=hidden,
                )
        else:
            policy, history = common.train_neural_policy(
                train_settings,
                episodes=train_episodes,
                episode_seconds=train_episode_seconds,
                train_seeds=train_seeds,
                seed=train_policy_seed,
                hidden=hidden,
            )
        update_stats = _plain({
            "episodes": float(len(history)),
            "transitions": float(sum(float(entry.get("transitions", 0.0)) for entry in history)),
            "mean_episode_return_mean": float(
                np.mean([float(entry.get("mean_return", 0.0)) for entry in history])
            ) if history else 0.0,
            "mean_episode_return_last": float(history[-1].get("mean_return", 0.0)) if history else 0.0,
            "weight_delta_max": float(
                np.max([float(entry.get("weight_delta", 0.0)) for entry in history])
            ) if history else 0.0,
        })
        sender_delta = 0.0
        if sender is not None and sender_before is not None:
            sender_delta = float(np.max(np.abs(sender.flat_weights() - sender_before)))
        trained[arm] = {
            "on": on,
            "policy": policy,
            "sender": sender,
            "sender_weight_delta": sender_delta,
            "train": update_stats,
            "train_wall_s": round(time.perf_counter() - arm_started, 3),
        }

    # -- 配对评估（同一运动策略，各自配置；on 臂 sender 冻结）--------------
    eval_settings_off = _arm_settings(settings, on=False, train=False)
    eval_settings_on = _arm_settings(settings, on=True, train=False)
    runs: dict[str, list[dict[str, Any]]] = {}
    runs["off"] = common.evaluate_seeds(
        eval_settings_off, trained["off"]["policy"], seeds=eval_seeds, seconds=eval_seconds
    )
    with use_communication_sender(trained["on"]["sender"]):
        runs["on"] = common.evaluate_seeds(
            eval_settings_on, trained["on"]["policy"], seeds=eval_seeds, seconds=eval_seconds
        )

    metrics = ("eat_events", "eat_rate", "satiation_mean", "population", "mean_life")
    paired = {
        metric: common.paired_summary(runs["on"], runs["off"], metric=metric)
        for metric in metrics
    }
    arm_summary = {
        arm: common.summarize_runs(arm_runs, metrics) for arm, arm_runs in runs.items()
    }
    channel = _summarize_channel(runs["on"], n_symbols)

    # -- 保存 on 臂模型（tag p6-comm）与训练后的 sender --------------------
    saved_models: dict[str, Any] = {}
    if settings.run.save_models:
        from ..zoo import ModelZoo

        zoo = ModelZoo()
        policy_on = trained["on"]["policy"]
        if getattr(policy_on, "kind", "") != "neural":
            policy_on.kind = "neural"  # zoo.load 需要 kind=neural（NeuralSwarmPolicy 无类属性）
        channel_metrics = _plain({
            "eat_events_mean": arm_summary["on"]["eat_events"]["mean"],
            "eat_rate_mean": arm_summary["on"]["eat_rate"]["mean"],
            "satiation_mean": arm_summary["on"]["satiation_mean"]["mean"],
            "channel_mi_bits": channel["mi_bits"]["mean"],
            "channel_entropy_bits": channel["entropy_bits"]["mean"],
            "channel_alignment": channel["alignment"]["mean"],
            "eat_events_paired_delta": paired["eat_events"]["mean_delta"],
        })
        try:
            card = zoo.save(
                "p6-comm-on",
                policy_on,
                metrics=channel_metrics,
                config=settings,
                tags=["p6-comm", "emergent-communication", "on-arm"],
                description="P6-A on-arm movement policy (9+4 obs, comm channel enabled)",
            )
            saved_models["p6-comm-on"] = card.to_dict()
            if trained["on"]["sender"] is not None:
                sender = trained["on"]["sender"]
                if getattr(sender, "kind", "") != "neural":
                    sender.kind = "neural"
                sender_card = zoo.save(
                    "p6-comm-sender",
                    sender,
                    metrics=_plain({"sender_weight_delta": trained["on"]["sender_weight_delta"]}),
                    config=settings,
                    tags=["p6-comm", "sender"],
                    description="P6-A trained symbol sender (obs=9, n_symbols=K)",
                )
                saved_models["p6-comm-sender"] = sender_card.to_dict()
        except Exception as exc:  # pragma: no cover - 模型保存失败不应毁掉实验
            saved_models["error"] = repr(exc)

    return _plain({
        "train_seeds": train_seeds,
        "train_episodes": train_episodes,
        "train_episode_seconds": train_episode_seconds,
        "train_policy_seed": train_policy_seed,
        "sender_seed": sender_seed,
        "eval_seeds": eval_seeds,
        "eval_seconds": eval_seconds,
        "n_symbols": n_symbols,
        "hidden": hidden,
        "arms": {
            "off": {
                "train": trained["off"]["train"],
                "train_wall_s": trained["off"]["train_wall_s"],
                "eval": arm_summary["off"],
            },
            "on": {
                "train": trained["on"]["train"],
                "train_wall_s": trained["on"]["train_wall_s"],
                "sender_weight_delta": trained["on"]["sender_weight_delta"],
                "eval": arm_summary["on"],
            },
        },
        "paired_treatment_minus_control": paired,
        "eval_runs": {"off": _plain(runs["off"]), "on": _plain(runs["on"])},
        "channel_stats_eval": channel,
        "saved_models": saved_models,
        "wall_s": round(time.perf_counter() - started, 3),
    })


# ---------------------------------------------------------------------------
# 协议入口
# ---------------------------------------------------------------------------
@register_protocol("communication", note="P6-A 涌现通信：指称博弈 + 世界内信号通道 off/on 消融")
def run_communication(settings: ExperimentSettings) -> dict[str, Any]:
    """执行 P6-A 实验，返回可直接写入 summary.json 的字典。"""
    started = time.perf_counter()
    part_a = run_referential_part(settings)
    part_b = run_swarm_ablation_part(settings)

    # -- 诚实结论 ------------------------------------------------------
    accuracy = part_a["accuracy"]["mean"]
    accuracy_std = part_a["accuracy"]["std"]
    chance = part_a["chance_accuracy"]
    mi_bits = part_a["mutual_information_bits"]["mean"]
    mi_upper = part_a["mi_upper_bound_bits"]
    channel_mi = part_b["channel_stats_eval"]["mi_bits"]["mean"]
    channel_bias = part_b["channel_stats_eval"]["mi_expected_bias_bits"]
    channel_entropy = part_b["channel_stats_eval"]["entropy_bits"]["mean"]
    channel_alignment = part_b["channel_stats_eval"]["alignment"]["mean"]
    alignment_chance = part_b["channel_stats_eval"]["alignment_chance"]
    eat_paired = part_b["paired_treatment_minus_control"]["eat_events"]
    sat_paired = part_b["paired_treatment_minus_control"]["satiation_mean"]
    protocol_formed = bool(part_a["protocol_formed"])
    if protocol_formed:
        part_a_text = (
            f"(i) 控制指称博弈：accuracy={accuracy:.3f}±{accuracy_std:.3f} vs chance "
            f"{chance:.3f}（+{accuracy - chance:.3f}），MI={mi_bits:.3f}/{mi_upper:.2f} bits，"
            f"符号熵={part_a['symbol_entropy_bits']['mean']:.3f} bits，映射可辨识比例="
            f"{part_a['mapping_injective_fraction']:.2f} -> 指称协议确实形成。"
        )
    else:
        part_a_text = (
            f"(i) 控制指称博弈：accuracy={accuracy:.3f}±{accuracy_std:.3f} vs chance "
            f"{chance:.3f}（+{accuracy - chance:.3f}），MI={mi_bits:.3f}/{mi_upper:.2f} bits "
            f"-> 未达到事先设定的协议形成判据。"
        )
    near_bias = channel_mi <= max(3.0 * channel_bias, 0.01)
    if near_bias:
        part_b_text = (
            f"(ii) 世界信号通道：通道 MI={channel_mi:.4f} bits 与有限样本偏置 "
            f"{channel_bias:.4f} bits 同量级，符号熵 {channel_entropy:.3f} bits、"
            f"食物对齐 {channel_alignment:.3f}（随机 {alignment_chance:.3f}）=> "
            "未学到可辨指称语义（null，不宣称群体通信收益）。"
        )
    else:
        part_b_text = (
            f"(ii) 世界信号通道：通道 MI={channel_mi:.4f} bits > 偏置 {channel_bias:.4f}"
            f"（{channel_mi / max(channel_bias, 1e-12):.1f}x），符号熵 {channel_entropy:.3f} bits、"
            f"对齐 {channel_alignment:.3f}（随机 {alignment_chance:.3f}）=> 通道含少量非平凡信息，"
            "但需更多 seed 才能确认。"
        )
    behavior_text = (
        f"行为配对差（on-off, n={int(eat_paired['n'])}）：Δeat_events="
        f"{eat_paired['mean_delta']:+.2f}±{eat_paired['std_delta']:.2f}"
        f"（wins={int(eat_paired['wins'])}/losses={int(eat_paired['losses'])}/"
        f"ties={int(eat_paired['ties'])}），Δsatiation="
        f"{sat_paired['mean_delta']:+.4f}±{sat_paired['std_delta']:.4f}；"
        "n 很小、未做显著性检验，若接近 0 应视为中性（可能小幅负向），"
        "不能据此宣称通信提升群体适应度。"
    )
    verdict = " ".join([part_a_text, part_b_text, behavior_text])

    notes = [
        "Part A 是可控对照：随机机会准确率 = 1/cue_dim；MI 上限 = log2(n_symbols) bits。",
        "Part A 映射跨 seed 比较允许符号重命名（mapping_similarity 最优置换），因为符号标签任意。",
        "Part B 奖励是个体级（实际进食量），sender 只从群体标量奖励学习，缺少 per-sender 信用分配；"
        "在个体奖励+参数共享策略下世界通道未必可学，null 是合理科学结论。",
        "消息无个体成本（comm.message_cost=0），符号只通过邻居观测直方图进入世界，语义未硬编码。",
        "on 臂评估时 comm.train=False 冻结 sender（复制训练后权重）；off/on 训练预算与 seed 序列相同。",
        "on/off 观测维度不同（9 vs 9+K），初始化权重无法逐位相同；配对仅在评估 seed 上成立。",
        "有限样本 MI 插值估计有正偏置，约 (K-1)^2/(2N ln2) bits（N=8192, K=4 时约 8e-4）。",
    ]
    limitations = [
        "Part B 每个臂只训练 1 个策略、评估 5 个 seed，样本量不足以做显著性检验。",
        "训练回合短（30s/回合、6 回合），策略尚未收敛到最优觅食策略；对照差异可能被训练不足淹没。",
        "指称博弈只证明 4 个 cue / 4 个符号的指称可辨识性，不等同于组合性/系统性语言。",
        "世界通道的 sender 训练信号（群体奖励 1 步延迟）本身就是弱信号，负结果不能证明"
        "任何通信在群体中都不可能，只证明本设定下该通道未产生可测语义收益。",
    ]

    return {
        "protocol": settings.protocol,
        "config_name": settings.name,
        "part_a_referential": part_a,
        "part_b_swarm_ablation": part_b,
        "verdict": verdict,
        "conclusion": (
            "controlled referential game demonstrates protocol formation"
            if protocol_formed else "referential protocol formation not demonstrated"
        ) + "; world channel " + ("null (no measurable learned semantics)" if near_bias else "weak signal, inconclusive"),
        "notes": notes,
        "limitations": limitations,
        "seeds": {
            "part_a_game_seeds": part_a["game_seeds"],
            "part_b_train_seeds": part_b["train_seeds"],
            "part_b_eval_seeds": part_b["eval_seeds"],
        },
        "wall_s": round(time.perf_counter() - started, 3),
    }


__all__ = ["run_communication", "run_referential_part", "run_swarm_ablation_part"]
