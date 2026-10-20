"""训练低层动作策略（REINFORCE）——真实训练入口，不伪造任何结果。

用法::

    python tools/train_lowlevel_rl.py --episodes 30 --steps 4000 \
        --out assets/lowlevel_rl.npz
    python tools/train_lowlevel_rl.py --episodes 2 --steps 600 --seeds 0 \
        --eval-seeds 100,101 --eval-steps 600 --json .tmp/lowlevel_smoke.json

流程（全部是真实计算）：

1. 从 ``--from-checkpoint``（默认 ``assets/neural_policy.npz``，即现有 BC 权重）
   加载 :class:`~newworld.neural.NeuralPolicy`，用 :class:`LowLevelPolicy` 包装成
   可采样策略（``--random-init`` 可改为从随机权重开始）。
2. 用 ``simulation.make_lowlevel_rl_env(seed, ...)`` 构造训练环境；trainer 会自动
   调用 ``env.population.attach_lowlevel_policy(policy)``（``low_level_mode="rl"``
   时 Population 用该策略采样；mode 不是 rl 时不会产生转移，本脚本如实报告）。
3. 每帧推进世界；``population.drain_lowlevel_transitions()`` 给出
   ``(agent_id, features, action: ActionSample, reward)``，按 agent 计算折扣回报
   与指数滑动基线，手写反向做一次 REINFORCE 更新。
4. 每回合打印真实回报/步数/转移数；周期性用独立种子评估；训练完保存训练回报
   最好的策略，并在 ``--eval-seeds`` 上做独立评估。

注意：
* 奖励由集成层（``newworld/rewards.py`` 的 ``LowLevelReward``）按真实世界状态变化
  计算，本脚本不做奖励塑形、不使用任何行为克隆标签。
* 语言头不参与 RL（保持 argmax）。
* 若集成钩子尚未落地，脚本以 exit code 2 明确失败，不打印假曲线；若钩子已落地但
  没有产生任何转移（例如 ``low_level_mode != "rl"``），脚本会如实报告
  ``transitions=0``（没有发生策略梯度更新），不补造数据。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from newworld.lowlevel_rl import (  # noqa: E402
    MOVE_VALUES,
    LowLevelPolicy,
    LowLevelTrainer,
)
from newworld.neural import FEATURE_DIM, MODES, NeuralPolicy  # noqa: E402

#: 默认行为克隆 checkpoint 与输出 checkpoint
DEFAULT_FROM = Path("assets") / "neural_policy.npz"
DEFAULT_OUT = Path("assets") / "lowlevel_rl.npz"


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
    """返回 ``env_factory(seed, steps=..., dt=...)``；钩子缺失时 exit 2。"""
    try:
        from newworld.simulation import make_lowlevel_rl_env  # noqa: WPS433
    except ImportError:
        print(
            "错误：集成钩子尚未落地——newworld.simulation 里没有 "
            "make_lowlevel_rl_env。请等 Lead 完成 population.attach_lowlevel_policy / "
            "drain_lowlevel_transitions / simulation.make_lowlevel_rl_env 后再运行。",
            file=sys.stderr,
        )
        raise SystemExit(2)

    def factory(seed: int, *, steps: int = 4000, dt: float = 0.05):
        """构造真实训练环境；兼容 make_lowlevel_rl_env 的两种签名。"""
        try:
            return make_lowlevel_rl_env(
                int(seed),
                group=args.group,
                steps=int(steps),
                dt=float(dt),
                width=int(args.width),
                height=int(args.height),
            )
        except TypeError:
            return make_lowlevel_rl_env(int(seed), steps=int(steps), dt=float(dt))

    return factory


def _load_neural_policy(args, seed: int) -> NeuralPolicy:
    """按 --from-checkpoint / --random-init 构造初始神经策略。"""
    if args.random_init:
        return NeuralPolicy(seed=seed)
    path = Path(args.from_checkpoint)
    if not path.exists():
        print(
            f"错误：BC checkpoint 不存在: {path}（可用 --random-init 从随机权重开始）",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        return NeuralPolicy.load(path)
    except ValueError as exc:
        print(
            f"错误：无法加载 BC checkpoint {path}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _check_hooks(factory, args) -> int:
    """探测集成钩子：构造一个短环境，打印真实接口可用性。"""
    env = factory(int(args.seeds[0]), steps=1, dt=float(args.dt))
    population = getattr(env, "population", None)
    checks = {
        "env.step": callable(getattr(env, "step", None)),
        "env.ended": callable(getattr(env, "ended", None)),
        "env.result": callable(getattr(env, "result", None)),
        "env.population": population is not None,
        "population.attach_lowlevel_policy": callable(
            getattr(population, "attach_lowlevel_policy", None)
        ),
        "population.drain_lowlevel_transitions": callable(
            getattr(population, "drain_lowlevel_transitions", None)
        ),
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
            f"return={record['return']:+.4f} "
            f"steps={int(record['steps'])} "
            f"transitions={int(record['transitions'])} "
            f"agents={int(record.get('agents', 0))} "
            f"baseline={record['baseline']:+.4f} "
            f"mean_adv={record['mean_advantage']:+.4f}"
        )
    for entry in result.get("eval", []):
        evaluation = entry["eval"]
        print(
            f"[seed {seed}] eval@{int(entry['episode'])} "
            f"mean_return={evaluation['mean_return']:+.4f} "
            f"std={evaluation['std_return']:.4f} "
            f"mean_steps={evaluation['mean_steps']:.1f} "
            f"transitions={int(evaluation.get('total_transitions', 0))}"
        )


def _training_score(result: dict[str, object]) -> float:
    """用**训练**回报（最近 3 回合均值）选择保存哪个种子的策略。"""
    returns = [float(value) for value in result.get("train_returns", [])]
    if not returns:
        return float("-inf")
    return float(np.mean(returns[-3:]))


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
    parts = [f"{key}={evaluation[key]:+.4f}" for key in keys if key in evaluation]
    parts.extend(f"{key}={evaluation[key]:+.4f}" for key in extra)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# A/B 对照模式：BC 初始权重 vs 训练后 checkpoint（同一 low_level_mode=rl 决策链）
# ---------------------------------------------------------------------------
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


def _env_result_numbers(env) -> dict[str, float]:
    result_fn = getattr(env, "result", None)
    if not callable(result_fn):
        return {}
    try:
        data = result_fn()
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    numbers: dict[str, float] = {}
    for key, value in data.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            numbers[str(key)] = number
    return numbers


def _run_compare(args, factory, eval_seeds: list[int]) -> int:
    """真实 A/B：BC 初始权重 vs 训练后 checkpoint（都走 low_level_mode=rl 采样）。

    两个条件使用**同一套 env seed、同一步数、同一个固定 policy seed（=0）**；
    每个条件的策略实例按 seed 顺序连续使用（内部 RNG 流固定）。BC 初始权重从
    ``--from-checkpoint`` 加载（旧的 24 维 BC checkpoint 由 ``NeuralPolicy.load``
    零初始化迁移到当前 56 维输入）；训练后权重从 ``--checkpoint``（默认 ``--out``）
    加载。所有 seed 全部报告，不挑选。
    """
    trained_path = Path(args.checkpoint) if args.checkpoint else Path(args.out)
    if not trained_path.exists():
        print(f"错误：训练后 checkpoint 不存在: {trained_path}", file=sys.stderr)
        return 2
    trained_policy = LowLevelPolicy.load(trained_path, seed=0)
    if args.random_init:
        init_label = "random_init"
        init_policy = LowLevelPolicy(NeuralPolicy(seed=0), seed=0)
        init_path = None
    else:
        init_path = Path(args.from_checkpoint)
        if not init_path.exists():
            print(f"错误：BC 初始 checkpoint 不存在: {init_path}", file=sys.stderr)
            return 2
        init_label = "bc_init"
        try:
            init_policy = LowLevelPolicy(NeuralPolicy.load(init_path), seed=0)
        except ValueError as exc:
            print(f"错误：无法加载 BC checkpoint {init_path}: {exc}", file=sys.stderr)
            return 2
    steps = int(args.eval_steps) if args.eval_steps is not None else int(args.steps)
    if steps <= 0:
        print("错误：评估步数必须为正整数", file=sys.stderr)
        return 2
    conditions = ((init_label, init_policy), ("rl_trained", trained_policy))
    metric_keys = (
        "return",
        "alive_count",
        "tech_verified",
        "facilities_active",
        "lowlevel_decisions",
        "transitions",
        "steps_run",
        "total_agents_created",
        "total_deaths",
        "cooperation_index",
    )
    print("=== 低层 RL A/B 对照（low_level_mode=rl，固定 policy seed=0）===")
    print(
        f"{init_label}={init_path}  size="
        f"{init_path.stat().st_size if init_path is not None else '-'} bytes  |  "
        f"rl_trained={trained_path}  size={trained_path.stat().st_size} bytes"
    )
    print(
        f"seeds={eval_seeds}  steps={steps}  dt={float(args.dt)}  "
        f"world={int(args.width)}x{int(args.height)}  group={args.group}"
    )
    print(
        "两个条件都通过 population.attach_lowlevel_policy 走 low_level_mode='rl' "
        "决策链；策略按 seed 顺序连续采样（内部 RNG 流固定），全部 seed 均报告。"
    )
    started = time.perf_counter()
    payload: dict[str, object] = {
        "compare": True,
        "bc_init_checkpoint": None if init_path is None else str(init_path),
        "bc_init_size": (
            int(init_path.stat().st_size) if init_path is not None else 0
        ),
        "trained_checkpoint": str(trained_path),
        "trained_size": int(trained_path.stat().st_size),
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "steps": steps,
        "dt": float(args.dt),
        "group": str(args.group),
        "width": int(args.width),
        "height": int(args.height),
        "conditions": {},
    }
    for label, policy in conditions:
        rows: list[dict[str, object]] = []
        for seed in eval_seeds:
            env = factory(int(seed), steps=steps, dt=float(args.dt))
            attach = getattr(
                getattr(env, "population", None), "attach_lowlevel_policy", None
            )
            if callable(attach):
                attach(policy)
            step_count = 0
            while step_count < steps:
                ended = getattr(env, "ended", None)
                if callable(ended):
                    try:
                        if ended():
                            break
                    except Exception:
                        pass
                step_fn = getattr(env, "step", None)
                if not callable(step_fn):
                    print("错误：env 缺少 step()", file=sys.stderr)
                    return 2
                step_fn()
                step_count += 1
            drain = getattr(
                getattr(env, "population", None),
                "drain_lowlevel_transitions",
                None,
            )
            transitions = list(drain()) if callable(drain) else []
            rewards = [float(item[3]) for item in transitions]
            numbers = _env_result_numbers(env)
            row: dict[str, object] = {
                "seed": int(seed),
                "return": float(sum(rewards)),
                "transitions": float(len(transitions)),
                "steps_run": float(step_count),
            }
            for key in (
                "alive_count",
                "tech_verified",
                "facilities_active",
                "lowlevel_decisions",
                "total_agents_created",
                "total_deaths",
                "cooperation_index",
            ):
                if key in numbers:
                    row[key] = float(numbers[key])
            rows.append(row)
        aggregates = {
            key: _aggregate(
                [float(row[key]) for row in rows if key in row and row[key] is not None]
            )
            for key in metric_keys
        }
        payload["conditions"][label] = {"rows": rows, "aggregate": aggregates}  # type: ignore[index]
        print(f"\n--- {label} ---")
        for key in metric_keys:
            agg = aggregates[key]
            print(
                f"  {key:<20} mean={agg['mean']:+10.4f}  std={agg['std']:9.4f}  "
                f"min={agg['min']:+10.4f}  max={agg['max']:+10.4f}  n={int(agg['n'])}"
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
        description="训练 newworld 低层动作策略（REINFORCE，真实策略梯度）"
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
        help="周期评估使用的种子数（取 --eval-seeds 前 N 个）；最终评估使用 --eval-seeds 全部",
    )
    parser.add_argument(
        "--eval-steps", type=int, default=None, help="评估步数（默认同 --steps）"
    )
    parser.add_argument(
        "--eval-every", type=int, default=None, help="每隔多少回合做周期评估"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="策略梯度学习率")
    parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    parser.add_argument(
        "--baseline-decay", type=float, default=0.9, help="基线指数滑动平均衰减"
    )
    parser.add_argument("--group", default="D", help="消融分组（A/B/C/D）")
    parser.add_argument("--width", type=int, default=640, help="世界宽度")
    parser.add_argument("--height", type=int, default=400, help="世界高度")
    parser.add_argument(
        "--from-checkpoint",
        default=str(DEFAULT_FROM),
        help="初始 BC checkpoint（默认 assets/neural_policy.npz）",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="忽略 --from-checkpoint，从随机权重开始",
    )
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
        help="A/B 对照模式：BC 初始权重 vs 训练后 checkpoint（low_level_mode=rl）",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="--compare 使用的训练后 checkpoint（默认取 --out）",
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
    periodic_eval_seeds = eval_seeds[:eval_count] if eval_count > 0 else []
    eval_steps = args.steps if args.eval_steps is None else int(args.eval_steps)

    factory = _make_env_factory(args)
    if args.check_hooks:
        return _check_hooks(factory, args)
    if args.compare:
        return _run_compare(args, factory, eval_seeds)

    started = time.perf_counter()
    candidates: list[tuple[float, int, LowLevelPolicy, dict[str, object]]] = []
    all_transitions = 0.0
    for seed in train_seeds:
        neural = _load_neural_policy(args, seed)
        policy = LowLevelPolicy(neural, seed=seed)
        initial_weights = policy.flat_weights()
        trainer = LowLevelTrainer(
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
            eval_seeds=periodic_eval_seeds if eval_count > 0 else [],
            eval_steps=eval_steps,
            eval_every=args.eval_every,
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
            f"[seed {seed}] training score (最近 3 回合平均回报) = {score:+.4f}  "
            f"updates={int(result.get('updates', 0))}"
        )
        print(
            f"[seed {seed}] 真实参数更新: {params_changed}  "
            f"max|Δw|={max_delta:.6f}  ||Δw||={delta_l2:.6f}  "
            f"feature_dim={policy.feature_dim}"
        )
        candidates.append((score, seed, policy, result))

    best_score, best_seed, best_policy, best_result = max(
        candidates, key=lambda item: item[0]
    )
    out_path = Path(args.out)
    saved = best_policy.save(out_path)

    # 最终独立评估使用 --eval-seeds 全部种子（不参与选模型）
    final_eval: dict[str, float] = {}
    final_details: list[dict[str, float]] = []
    if eval_count > 0 and eval_seeds:
        final_trainer = LowLevelTrainer(
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

    elapsed = time.perf_counter() - started
    total_episodes = int(args.episodes) * len(train_seeds)
    print("")
    print("=== 训练完成 ===")
    print(
        f"训练种子: {train_seeds}  每种子回合数: {int(args.episodes)}  "
        f"每回合步数: {int(args.steps)}  dt={float(args.dt)}"
    )
    print(
        f"总训练回合: {total_episodes}  总世界步数: {total_episodes * int(args.steps)}  "
        f"墙钟: {elapsed:.1f}s"
    )
    print(f"总低层策略转移（真实 agent-步样本）: {int(all_transitions)}")
    if all_transitions <= 0:
        print(
            "警告：没有任何低层策略转移——集成钩子未产生 "
            "(agent_id, features, action, reward)（例如 low_level_mode != 'rl'），"
            "本次运行没有发生策略梯度更新（策略保持初始 BC 权重）。"
        )
    print(f"保存策略的训练种子: {best_seed}  训练 score: {best_score:+.4f}")
    print(
        f"保存策略真实参数更新: {best_result.get('params_changed')}  "
        f"max|Δw|={float(best_result.get('max_param_delta', 0.0)):.6f}  "
        f"||Δw||={float(best_result.get('param_delta_l2', 0.0)):.6f}"
    )
    print(f"checkpoint: {saved}")
    print(
        f"特征维度: {best_policy.feature_dim}  模式数: {len(MODES)}  "
        f"move 解码: {MOVE_VALUES}  参数总数: {best_policy.flat_weights().size}"
    )
    if final_eval:
        print(f"最终独立评估种子（全部 --eval-seeds）: {eval_seeds}")
        print(f"独立评估: {_format_final_eval(final_eval)}")
        for details in final_details:
            print(
                f"  评估 seed={int(details['seed'])} return={details['return']:+.4f} "
                f"steps={int(details['steps'])} transitions={int(details['transitions'])}"
            )
        if final_eval.get("total_transitions", 0.0) <= 0:
            print("警告：评估回合同样没有策略转移——评估没有真正使用策略采样。")
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
                "periodic_eval_seeds": periodic_eval_seeds,
                "eval_steps": int(eval_steps),
                "group": str(args.group),
                "width": int(args.width),
                "height": int(args.height),
                "lr": float(args.lr),
                "gamma": float(args.gamma),
                "baseline_decay": float(args.baseline_decay),
                "from_checkpoint": str(args.from_checkpoint),
                "random_init": bool(args.random_init),
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
            "checkpoint_size": int(saved.stat().st_size) if saved.exists() else 0,
            "feature_dim": int(best_policy.feature_dim),
            "move_values": list(MOVE_VALUES),
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
