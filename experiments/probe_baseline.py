"""P0 基线探针：在**未修改**的原始系统上测量重复动作、奖励与状态变化。

用途
----
任务书 §3.2 要求"不要只凭经验判断原因"，必须记录：动作频率、奖励贡献、
动作前后状态变化、连续重复次数、目标完成率、资源获取/消耗、失败识别、
世界感知、探索收敛、奖励漏洞、长期目标稀疏性。

本脚本只读 + 计数（对 `SocialLife.decide` / `Technology.*` 做计数包装，
不改变其行为），输出 JSON 到 `experiments/results/baseline_probe_<tag>.json`。

运行::

    cd newworldpro
    python experiments/probe_baseline.py --steps 2400 --dt 0.05 --tag pre

说明：原始系统**没有奖励函数**被主循环使用（`env.py` 的 reward 只存在于
未接入的 RL 包装器），因此本探针的 "reward" 一栏记录的是**奖励缺口**：
每步实际可用的标量训练信号 = 0。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newworld.config import make_config  # noqa: E402
from newworld.population import Population  # noqa: E402
from newworld.social import SocialLife  # noqa: E402
from newworld.technology import Technology  # noqa: E402
from newworld.world import World  # noqa: E402


def _install_counters() -> dict[str, Counter]:
    """Wrap teacher/technology methods with call counters (behaviour unchanged)."""
    counters: dict[str, Counter] = {
        "social_decision": Counter(),   # "gather"/"build"/... or "none"
        "teacher_override": Counter(),  # yes/no
        "tech": Counter(),
        "craft_ok": Counter(),
    }
    original_decide = SocialLife.decide

    def decide(self, world, character, dt):  # type: ignore[no-untyped-def]
        decision = original_decide(self, world, character, dt)
        counters["social_decision"][character.behavior_state] += 1
        counters["teacher_override"]["yes" if decision is not None else "no"] += 1
        return decision

    SocialLife.decide = decide  # type: ignore[assignment]

    for name in ("harvest", "craft", "contribute"):
        original = getattr(Technology, name)

        def make(fn, label):  # type: ignore[no-untyped-def]
            def wrapped(self, character, *args, **kwargs):  # type: ignore[no-untyped-def]
                result = fn(self, character, *args, **kwargs)
                counters["tech"][f"{label}_attempt"] += 1
                if result:
                    if label == "harvest":
                        counters["tech"][f"harvest_{args[0] if args else kwargs.get('kind', '?')}"] += 1
                    else:
                        counters["tech"][f"{label}_ok"] += 1
                return result

            return wrapped

        setattr(Technology, name, make(original, name))
    return counters


def probe(steps: int, dt: float, width: int, height: int, seed: int) -> dict[str, object]:
    counters = _install_counters()
    cfg = make_config(width, height)
    world = World(cfg, seed=seed)
    world.reset(seed=seed)
    population = Population(world, seed=seed)

    # -- per-step recording ------------------------------------------------
    behavior = Counter()
    moves = Counter()
    modes = Counter()
    per_agent: dict[int, dict[str, object]] = {}
    repeat_run = 0
    repeat_run_max = 0
    last_key = None
    no_progress_steps = 0
    state_change_steps = 0
    prev_snapshot = None
    rewards_seen = 0.0
    deaths = 0
    births = 0
    alive_history = []

    started = time.perf_counter()
    for step in range(steps):
        alive_before = [a for a in world.agents if a.alive]
        population.step(dt)
        alive_after = [a for a in world.agents if a.alive]
        births += max(0, len(alive_after) - len(alive_before))
        deaths += max(0, len(alive_before) - len(alive_after))
        alive_history.append(len(alive_after))
        if not alive_after:
            break
        # record every living agent's decision category
        for agent in alive_after:
            entry = per_agent.setdefault(
                agent.agent_id,
                {"steps": 0, "moves": Counter(), "states": Counter(), "tool": False},
            )
            entry["steps"] += 1
            entry["moves"][agent.facing] += 1
            entry["states"][agent.behavior_state] += 1
            entry["tool"] = bool(agent.inventory.tool)
            behavior[agent.behavior_state] += 1
            moves[agent.behavior_state] += 1
        lead_agent = alive_after[0]
        mind = population.minds.get(lead_agent.agent_id)
        if mind is not None and mind.will.last_output is not None:
            modes[mind.will.last_output.mode] += 1
        # repeat-action run (same facing + jump-ish proxy: behavior_state)
        key = (lead_agent.facing, lead_agent.behavior_state)
        if key == last_key:
            repeat_run += 1
        else:
            repeat_run = 1
            last_key = key
        repeat_run_max = max(repeat_run_max, repeat_run)
        # "no progress": nothing observable changed (x, energy, inventory, tech)
        snapshot = (
            round(lead_agent.x, 3),
            round(lead_agent.physiology.energy, 3),
            round(lead_agent.physiology.hydration, 3),
            lead_agent.inventory.wood,
            lead_agent.inventory.stone,
            lead_agent.inventory.tool,
            world.technology.community.wood,
            world.technology.community.stone,
            world.technology.community.structures,
            lead_agent.behavior_state,
        )
        if snapshot == prev_snapshot:
            no_progress_steps += 1
        else:
            state_change_steps += 1
        prev_snapshot = snapshot
        # the original main loop never reads a reward: record the gap explicitly
        rewards_seen += 0.0

    elapsed = time.perf_counter() - started
    community = world.technology.community
    total_behaviour = sum(behavior.values()) or 1
    return {
        "config": {
            "size": [width, height],
            "steps": steps,
            "dt": dt,
            "seed": seed,
            "world_version": __import__("newworld").WORLD_VERSION,
            "network": "MLP 24-64-32-58 (BC, no reward)",
        },
        "timing": {"wall_s": round(elapsed, 3), "steps_per_s": round(steps / max(elapsed, 1e-9), 1)},
        "population": {
            "alive_start": 2,
            "alive_end": len([a for a in world.agents if a.alive]),
            "births": births,
            "deaths": deaths,
            "total_agents_in_world": len(world.agents),
            "max_alive": max(alive_history) if alive_history else 0,
        },
        "action_distribution": {
            "behavior_state": {k: round(v / total_behaviour, 4) for k, v in behavior.most_common()},
            "neural_mode": dict(modes),
            "teacher_override": dict(counters["teacher_override"]),
        },
        "repetition": {
            "repeat_run_max": repeat_run_max,
            "no_progress_step_ratio": round(no_progress_steps / max(1, steps), 4),
            "state_change_step_ratio": round(state_change_steps / max(1, steps), 4),
        },
        "economy": {
            "tech_calls": dict(counters["tech"]),
            "community_wood": community.wood,
            "community_stone": community.stone,
            "community_structures": community.structures,
            "community_built": community.built,
            "inventory_end": {str(a.agent_id): {"wood": a.inventory.wood, "stone": a.inventory.stone,
                                                "tool": a.inventory.tool}
                              for a in world.agents if a.alive},
        },
        "goal_tracking": {
            "goals_defined": 0,
            "goals_completed": 0,
            "goals_failed": 0,
            "note": "原始系统没有目标/计划数据结构（审计结论 §3.1-5）",
        },
        "reward": {
            "steps": steps,
            "scalar_reward_total": rewards_seen,
            "reward_per_step": 0.0,
            "reward_functions_in_env_py": ["ZeroReward", "AliveReward", "ExplorationReward", "CenterReward"],
            "note": "env.py 的奖励函数未被主循环使用；population/neural 路径是行为克隆，无奖励",
        },
        "per_agent": {
            str(k): {
                "steps": v["steps"],
                "states": dict(v["states"]),
                "tool_ever": v["tool"],
            }
            for k, v in per_agent.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="pre")
    parser.add_argument("--out", default="experiments/results")
    args = parser.parse_args()

    result = probe(args.steps, args.dt, args.width, args.height, args.seed)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"baseline_probe_{args.tag}_seed{args.seed}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[saved] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
