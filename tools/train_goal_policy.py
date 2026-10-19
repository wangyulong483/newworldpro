"""训练高层目标选择策略（REINFORCE）——真实训练入口，不伪造任何结果。

用法::

    python tools/train_goal_policy.py --episodes 30 --steps 4000 \
        --out assets/goal_policy.npz
    python tools/train_goal_policy.py --episodes 2 --steps 800 --seeds 0 \
        --eval-seeds 100,101 --out .tmp/goal_policy_smoke.npz

流程（全部是真实计算）：

1. 用 ``simulation.make_goal_policy_env(seed, ...)`` 构造训练环境；环境内部挂载
   :class:`newworld.highlevel.GoalPolicy`，目标选择由策略采样决定。
2. 每帧推进世界；目标结束时集成层把 ``(features, action, reward)`` 写进
   ``population.drain_policy_transitions()``，本脚本取走并计算折扣回报 - 基线。
3. 用 numpy 手写反向传播做一次 REINFORCE 更新；重复 ``--episodes`` 个回合。
4. 每回合打印真实回报 / 步数 / 转移数；周期性用**独立种子**评估。
5. 训练完在 ``--eval-seeds`` 上做独立评估（不参与训练），并把训练种子中
   训练回报最好的策略保存到 ``--out``。

注意：
* 奖励由集成层按世界真实变化计算（生存/技术/设施/资源/知识/合作/失败/死亡），
  本脚本**不**做奖励塑形，也不使用任何行为克隆标签。
* 若集成钩子尚未落地，脚本以 exit code 2 明确失败，不打印假曲线。
* 若钩子已落地但 ``drain_policy_transitions`` 没有产生任何转移，脚本会如实
  报告 ``transitions=0``（即没有发生策略梯度更新），并仍然保存初始策略。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from newworld.highlevel import (  # noqa: E402
    FEATURE_DIM,
    GOAL_ACTIONS,
    GoalPolicy,
    GoalPolicyTrainer,
)

#: 默认 checkpoint 路径
DEFAULT_OUT = Path("assets") / "goal_policy.npz"


def _parse_seeds(text: str, *, label: str) -> list[int]:
    """解析 ``"0,1,2"`` 形式的种子列表。"""
    seeds: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            seeds.append(int(part))
        except ValueError:
            raise SystemExit(f"错误：--{label} 含非整数 {part!r}")
    if not seeds:
        raise SystemExit(f"错误：--{label} 不能为空")
    return seeds


def _make_env_factory(args):
    """返回 ``env_factory(seed, steps=..., dt=...)``，并检查钩子是否可用。"""
    try:
        from newworld.simulation import make_goal_policy_env  # noqa: WPS433
    except ImportError:
        print(
            "错误：集成钩子尚未落地——newworld.simulation 里没有 "
            "make_goal_policy_env。请等 Lead 完成 population.attach_goal_policy / "
            "drain_policy_transitions / simulation.make_goal_policy_env 后再运行。",
            file=sys.stderr,
        )
        raise SystemExit(2)

    def factory(seed: int, *, steps: int = 4000, dt: float = 0.05):
        """构造真实训练环境；兼容 make_goal_policy_env 的两种签名。"""
        try:
            return make_goal_policy_env(
                int(seed),
                group=args.group,
                steps=int(steps),
                dt=float(dt),
                width=int(args.width),
                height=int(args.height),
            )
        except TypeError:
            return make_goal_policy_env(int(seed), steps=int(steps), dt=float(dt))

    return factory


def _check_hooks(factory, args) -> int:
    """探测集成钩子：构造一个短环境，打印真实接口可用性。"""
    env = factory(int(args.seeds[0]), steps=1, dt=float(args.dt))
    population = getattr(env, "population", None)
    checks = {
        "env.step": callable(getattr(env, "step", None)),
        "env.ended": callable(getattr(env, "ended", None)),
        "env.result": callable(getattr(env, "result", None)),
        "env.population": population is not None,
        "population.drain_policy_transitions": callable(
            getattr(population, "drain_policy_transitions", None)
        ),
        "population.goals": getattr(population, "goals", None) is not None,
        "env.world": getattr(env, "world", None) is not None,
    }
    print("集成钩子探测：")
    for name, ok in checks.items():
        print(f"  {'OK  ' if ok else 'MISS'} {name}")
    try:
        env.step()
    except Exception as exc:  # pragma: no cover - 仅诊断路径
        print(f"  env.step() 抛错: {type(exc).__name__}: {exc}")
        return 1
    try:
        result = env.result()
        print(f"  env.result() = {result}")
    except Exception as exc:  # pragma: no cover
        print(f"  env.result() 抛错: {type(exc).__name__}: {exc}")
        return 1
    return 0 if all(checks.values()) else 1


def _print_train_curve(seed: int, result: dict[str, object]) -> None:
    history = result.get("train", [])
    total = len(history)
    for index, record in enumerate(history, start=1):
        print(
            f"[seed {seed}] ep {index}/{total} "
            f"return={record['return']:+.3f} "
            f"steps={int(record['steps'])} "
            f"transitions={int(record['transitions'])} "
            f"baseline={record['baseline']:+.3f} "
            f"mean_adv={record['mean_advantage']:+.3f}"
        )
    for entry in result.get("eval", []):
        evaluation = entry["eval"]
        print(
            f"[seed {seed}] eval@{int(entry['episode'])} "
            f"mean_return={evaluation['mean_return']:+.3f} "
            f"std={evaluation['std_return']:.3f} "
            f"mean_steps={evaluation['mean_steps']:.1f} "
            f"transitions={int(evaluation.get('total_transitions', 0))}"
        )


def _training_score(result: dict[str, object]) -> float:
    """用**训练**回报（最近 3 回合均值）选择保存哪个种子的策略。"""
    returns = [float(value) for value in result.get("train_returns", [])]
    if not returns:
        return float("-inf")
    recent = returns[-3:]
    return sum(recent) / float(len(recent))


def _format_final_eval(evaluation: dict[str, float]) -> str:
    keys = [
        "episodes",
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
        "mean_steps",
    ]
    extra = sorted(
        key for key in evaluation if key.startswith("mean_") and key not in keys
    )
    parts = [f"{key}={evaluation[key]:+.3f}" for key in keys if key in evaluation]
    parts.extend(f"{key}={evaluation[key]:+.3f}" for key in extra)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# A/B 对照模式：同一 checkpoint 固定 policy 内部 RNG 流，比较三类高层策略
# ---------------------------------------------------------------------------
def _env_ended(env) -> bool:
    ended = getattr(env, "ended", None)
    if not callable(ended):
        return False
    try:
        return bool(ended())
    except Exception:
        return False


def _numeric_result(env) -> dict[str, float]:
    """读取 ``env.result()`` 里的数值型标量。"""
    result_fn = getattr(env, "result", None)
    if not callable(result_fn):
        return {}
    try:
        data = result_fn()
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in data.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out[str(key)] = number
    return out


def _decision_share(env) -> dict[str, float]:
    population = getattr(env, "population", None)
    share_fn = getattr(population, "decision_share", None)
    if not callable(share_fn):
        return {}
    try:
        return {str(key): float(value) for key, value in share_fn().items()}
    except Exception:
        return {}


def _aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "n": float(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _run_compare(args, factory, eval_seeds: list[int]) -> int:
    """真实 A/B：trained checkpoint vs 同结构 untrained vs heuristic（不 attach）。

    三个条件用**同一套 env seed、同一步数、同一个固定 policy seed（=0）**；
    策略实例按 seed 顺序连续使用（内部 RNG 流固定），与正式实验的
    "--policy-mode neural --policy-path ... + 固定 policy seed" 口径一致。
    """
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.out)
    if not checkpoint.exists():
        print(f"错误：checkpoint 不存在: {checkpoint}", file=sys.stderr)
        return 2
    steps = int(args.eval_steps) if args.eval_steps is not None else int(args.steps)
    if steps <= 0:
        print("错误：评估步数必须为正整数", file=sys.stderr)
        return 2
    trained = GoalPolicy.load(checkpoint)
    untrained = GoalPolicy(seed=0)
    conditions: tuple[tuple[str, object | None], ...] = (
        ("trained", trained),
        ("untrained", untrained),
        ("heuristic", None),
    )
    metric_keys = (
        "return",
        "alive_count",
        "tech_verified",
        "facilities_active",
        "goal_decisions",
        "transitions",
        "steps_run",
    )
    share_keys = (
        "neural",
        "teacher",
        "goal",
        "policy_goal",
        "policy_infeasible",
        "infrastructure",
        "policy_error",
    )
    print("=== A/B 对照（同一 checkpoint / 固定 policy 内部 RNG 流）===")
    print(
        f"checkpoint={checkpoint}  size={checkpoint.stat().st_size} bytes  "
        f"seeds={eval_seeds}  steps={steps}  dt={float(args.dt)}  "
        f"world={int(args.width)}x{int(args.height)}  group={args.group}"
    )
    print(
        "trained/untrained: 策略按 seed 顺序连续采样（policy seed=0 固定）；"
        "heuristic: 不 attach（env 自带启发式），因此无策略转移、return 按构造为 0，"
        "不可与其它条件的 return 直接比较。"
    )
    started = time.perf_counter()
    payload: dict[str, object] = {
        "compare": True,
        "checkpoint": str(checkpoint),
        "checkpoint_size": int(checkpoint.stat().st_size),
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "steps": steps,
        "dt": float(args.dt),
        "group": str(args.group),
        "width": int(args.width),
        "height": int(args.height),
        "conditions": {},
    }
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for label, policy in conditions:
        rows: list[dict[str, object]] = []
        for seed in eval_seeds:
            env = factory(int(seed), steps=steps, dt=float(args.dt))
            if policy is not None:
                attach = getattr(
                    getattr(env, "population", None), "attach_goal_policy", None
                )
                if callable(attach):
                    attach(policy)
            step_count = 0
            while step_count < steps and not _env_ended(env):
                step_fn = getattr(env, "step", None)
                if not callable(step_fn):
                    print("错误：env 缺少 step()", file=sys.stderr)
                    return 2
                step_fn()
                step_count += 1
            drain = getattr(
                getattr(env, "population", None), "drain_policy_transitions", None
            )
            transitions = list(drain()) if callable(drain) else []
            rewards = [float(item[2]) for item in transitions]
            result_metrics = _numeric_result(env)
            row: dict[str, object] = {
                "seed": int(seed),
                "return": float(sum(rewards)),
                "transitions": float(len(transitions)),
                "steps_run": float(step_count),
                "decision_share": _decision_share(env),
            }
            for key in ("alive_count", "tech_verified", "facilities_active", "goal_decisions"):
                if key in result_metrics:
                    row[key] = float(result_metrics[key])
            rows.append(row)
        aggregates: dict[str, dict[str, float]] = {}
        for key in metric_keys:
            values = [
                float(row[key]) for row in rows if key in row and row[key] is not None
            ]
            aggregates[key] = _aggregate(values)
        for key in share_keys:
            values = [
                float(row["decision_share"].get(key, 0.0))  # type: ignore[union-attr]
                for row in rows
            ]
            aggregates[f"decision_share.{key}"] = _aggregate(values)
        summary[label] = aggregates
        payload["conditions"][label] = {
            "rows": rows,
            "aggregate": aggregates,
        }  # type: ignore[index]
        print(f"\n--- {label} ---")
        for key in metric_keys:
            agg = aggregates[key]
            print(
                f"  {key:<18} mean={agg['mean']:+9.3f}  std={agg['std']:8.3f}  "
                f"min={agg['min']:+9.3f}  max={agg['max']:+9.3f}  n={int(agg['n'])}"
            )
        for key in share_keys:
            agg = aggregates[f"decision_share.{key}"]
            print(
                f"  share.{key:<12} mean={agg['mean']:.4f}  std={agg['std']:.4f}  "
                f"min={agg['min']:.4f}  max={agg['max']:.4f}"
            )
    elapsed = time.perf_counter() - started
    payload["wall_s"] = round(elapsed, 3)
    print(f"\n对照墙钟: {elapsed:.1f}s")
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"对照结果 JSON: {json_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="训练 newworld 高层目标选择策略（REINFORCE，真实策略梯度）"
    )
    parser.add_argument("--episodes", type=int, default=30, help="训练回合数")
    parser.add_argument("--steps", type=int, default=4000, help="每回合世界步数")
    parser.add_argument("--dt", type=float, default=0.05, help="世界步长（秒）")
    parser.add_argument("--seeds", default="0", help="训练种子，逗号分隔（如 0,1,2）")
    parser.add_argument(
        "--eval-seeds",
        default="100,101",
        help="独立评估种子，逗号分隔；不得与训练种子重叠",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=2,
        help="周期评估使用的种子数（取 --eval-seeds 前 N 个）；最终独立评估使用 --eval-seeds 全部",
    )
    parser.add_argument(
        "--eval-steps", type=int, default=None, help="评估步数（默认同 --steps）"
    )
    parser.add_argument(
        "--eval-every", type=int, default=None, help="每隔多少回合做周期评估"
    )
    parser.add_argument("--lr", type=float, default=0.05, help="策略梯度学习率")
    parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    parser.add_argument(
        "--baseline-decay", type=float, default=0.9, help="基线指数滑动平均衰减"
    )
    parser.add_argument("--group", default="D", help="消融分组（A/B/C/D）")
    parser.add_argument("--width", type=int, default=640, help="世界宽度")
    parser.add_argument("--height", type=int, default=400, help="世界高度")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="checkpoint 输出路径")
    parser.add_argument("--json", default=None, help="把完整结果写入该 JSON 路径")
    parser.add_argument(
        "--check-hooks",
        action="store_true",
        help="只探测集成钩子与 env 接口，不训练",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="A/B 对照模式：trained checkpoint vs untrained vs heuristic",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="--compare 使用的 checkpoint（默认取 --out）",
    )
    args = parser.parse_args(argv)

    if args.episodes <= 0 or args.steps <= 0:
        raise SystemExit("错误：--episodes/--steps 必须是正整数")
    train_seeds = _parse_seeds(args.seeds, label="seeds")
    eval_seeds = _parse_seeds(args.eval_seeds, label="eval-seeds")
    overlap = sorted(set(train_seeds) & set(eval_seeds))
    if overlap:
        raise SystemExit(f"错误：训练种子与评估种子重叠: {overlap}")
    eval_count = int(args.eval_episodes)
    if eval_count < 0:
        raise SystemExit("错误：--eval-episodes 必须 >= 0")
    eval_seed_list = eval_seeds[:eval_count] if eval_count > 0 else []
    eval_steps = args.steps if args.eval_steps is None else int(args.eval_steps)

    factory = _make_env_factory(args)
    if args.check_hooks:
        return _check_hooks(factory, args)
    if args.compare:
        return _run_compare(args, factory, eval_seeds)

    started = time.perf_counter()
    candidates: list[tuple[float, int, GoalPolicy, dict[str, object]]] = []
    all_transitions = 0.0
    for seed in train_seeds:
        policy = GoalPolicy(seed=seed)
        initial_weights = policy.flat_weights().copy()
        trainer = GoalPolicyTrainer(
            policy,
            factory,
            seed=seed,
            lr=float(args.lr),
            gamma=float(args.gamma),
            baseline_decay=float(args.baseline_decay),
        )
        result = trainer.train(
            episodes=int(args.episodes),
            steps=int(args.steps),
            dt=float(args.dt),
            eval_episodes=eval_count,
            eval_steps=eval_steps,
            eval_every=args.eval_every,
            eval_seeds=eval_seed_list if eval_count > 0 else None,
        )
        all_transitions += float(sum(result.get("train_transitions", [])))
        delta = policy.flat_weights() - initial_weights
        params_changed = bool(not np.allclose(policy.flat_weights(), initial_weights))
        max_delta = float(np.abs(delta).max()) if params_changed else 0.0
        delta_l2 = float(np.linalg.norm(delta)) if params_changed else 0.0
        result["params_changed"] = params_changed
        result["max_param_delta"] = max_delta
        result["param_delta_l2"] = delta_l2
        _print_train_curve(seed, result)
        score = _training_score(result)
        print(
            f"[seed {seed}] training score (最近 3 回合平均回报) = {score:+.3f}  "
            f"updates={int(result.get('updates', 0))}"
        )
        print(
            f"[seed {seed}] 真实参数更新: {params_changed}  "
            f"max|Δw|={max_delta:.4f}  ||Δw||={delta_l2:.4f}"
        )
        candidates.append((score, seed, policy, result))

    best_score, best_seed, best_policy, best_result = max(
        candidates, key=lambda item: item[0]
    )
    out_path = Path(args.out)
    saved = best_policy.save(out_path)

    # 用独立评估种子评估"被选中的策略"（选择只依据训练回报，不看评估）
    # 最终评估使用 --eval-seeds 的**全部**种子；周期评估只用前 --eval-episodes 个
    final_eval: dict[str, float] = {}
    if eval_count > 0 and eval_seeds:
        final_trainer = GoalPolicyTrainer(
            best_policy,
            factory,
            seed=best_seed,
            lr=float(args.lr),
            gamma=float(args.gamma),
            baseline_decay=float(args.baseline_decay),
        )
        final_eval = final_trainer.evaluate(
            seeds=eval_seeds, steps=eval_steps, dt=float(args.dt)
        )
        final_details = list(final_trainer.last_eval_details)
    else:
        final_details = []

    elapsed = time.perf_counter() - started
    total_episodes = int(args.episodes) * len(train_seeds)
    print("")
    print("=== 训练完成 ===")
    print(f"训练种子: {train_seeds}  每种子回合数: {int(args.episodes)}  "
          f"每回合步数: {int(args.steps)}  dt={float(args.dt)}")
    print(f"总训练回合: {total_episodes}  总世界步数: {total_episodes * int(args.steps)}  "
          f"墙钟: {elapsed:.1f}s")
    print(f"总策略转移（真实目标级样本）: {int(all_transitions)}")
    if all_transitions <= 0:
        print(
            "警告：没有任何策略转移——集成钩子未产生 (features, action, reward)，"
            "本次运行没有发生策略梯度更新（策略保持初始化状态，未学到东西）。"
        )
    print(f"保存策略的训练种子: {best_seed}  训练 score: {best_score:+.3f}")
    print(
        f"保存策略真实参数更新: {best_result.get('params_changed')}  "
        f"max|Δw|={float(best_result.get('max_param_delta', 0.0)):.4f}  "
        f"||Δw||={float(best_result.get('param_delta_l2', 0.0)):.4f}"
    )
    print(f"checkpoint: {saved}")
    print(f"特征维度: {FEATURE_DIM}  动作数: {len(GOAL_ACTIONS)}  "
          f"参数总数: {best_policy.flat_weights().size}")
    if final_eval:
        print(f"最终独立评估种子（全部 --eval-seeds）: {eval_seeds}")
        print(f"独立评估: {_format_final_eval(final_eval)}")
        for details in final_details:
            print(
                f"  评估 seed={int(details['seed'])} return={details['return']:+.3f} "
                f"steps={int(details['steps'])} transitions={int(details['transitions'])}"
            )
        if final_eval.get("total_transitions", 0.0) <= 0:
            print(
                "警告：评估回合同样没有策略转移——评估没有真正使用策略决策。"
            )
    else:
        print("独立评估: 未执行（--eval-episodes 0）")

    if args.json:
        payload = {
            "args": {
                "episodes": int(args.episodes),
                "steps": int(args.steps),
                "dt": float(args.dt),
                "seeds": train_seeds,
                "eval_seeds": eval_seeds,
                "periodic_eval_seeds": eval_seed_list,
                "eval_steps": int(eval_steps),
                "group": str(args.group),
                "width": int(args.width),
                "height": int(args.height),
                "lr": float(args.lr),
                "gamma": float(args.gamma),
                "baseline_decay": float(args.baseline_decay),
            },
            "runs": [
                {
                    "seed": int(seed),
                    "training_score": float(score),
                    "result": result,
                }
                for score, seed, _policy, result in candidates
            ],
            "best_seed": int(best_seed),
            "best_training_score": float(best_score),
            "final_eval": final_eval,
            "final_eval_details": final_details,
            "checkpoint": str(saved),
            "wall_s": round(elapsed, 3),
        }
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"结果 JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
