"""P0 长时程人口/文明探针（原始系统，未修改）。

粗时间步长（dt=0.5）会破坏教师规则的近距离导航，导致测不到真实生育节奏，
因此本探针使用真实 dt=1/20，并**关闭感知**提速：
`NeuralInnerLife.decide` 传入的 `perception` 在 `neural.features()` 里被完全忽略，
所以跳过 `Perceiver.perceive` 对决策**逐比特无影响**（只影响调试/日志），
可把每步耗时从 ~14 ms 降到 ~0.5 ms。

输出：`experiments/results/baseline_long_<tag>_seed<seed>.json`

运行::

    cd newworldpro
    python experiments/probe_long.py --steps 30000 --dt 0.05 --tag pre
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

from newworld import perception  # noqa: E402
from newworld.config import make_config  # noqa: E402
from newworld.population import Population  # noqa: E402
from newworld.world import World  # noqa: E402


def _disable_perception() -> None:
    """Behaviour-neutral speed-up (see module docstring)."""
    from newworld.perception import Perceiver

    Perceiver.perceive = lambda self, world, dt, character=None: None  # type: ignore[method-assign]


def run(steps: int, dt: float, width: int, height: int, seed: int, with_perception: bool) -> dict[str, object]:
    if not with_perception:
        _disable_perception()
    cfg = make_config(width, height)
    world = World(cfg, seed=seed)
    world.reset(seed=seed)

    created: dict[int, float] = {a.agent_id: 0.0 for a in world.agents}
    original_spawn = world.spawn

    def spawn_tracking(*args, **kwargs):  # type: ignore[no-untyped-def]
        character = original_spawn(*args, **kwargs)
        created[character.agent_id] = round(world.t, 3)
        return character

    world.spawn = spawn_tracking  # type: ignore[method-assign]
    population = Population(world, seed=seed)

    # death tracking: wrap World.remove_agent
    death_events: list[dict[str, object]] = []
    original_remove = world.remove_agent

    def remove_tracking(agent_id: int) -> None:
        record = next((a for a in world.agents if a.agent_id == agent_id), None)
        if record is not None:
            death_events.append(
                {
                    "t": round(world.t, 2),
                    "agent_id": agent_id,
                    "age": round(record.age, 1),
                    "energy": round(record.physiology.energy, 2),
                    "hydration": round(record.physiology.hydration, 2),
                    "cause": "starved" if record.physiology.energy <= 0.5 else "old_age",
                    "inventory": {"wood": record.inventory.wood, "stone": record.inventory.stone,
                                  "tool": record.inventory.tool},
                }
            )
        original_remove(agent_id)

    world.remove_agent = remove_tracking  # type: ignore[method-assign]

    behavior = Counter()
    snapshots: list[dict[str, object]] = []
    starvation_seen = None
    started = time.perf_counter()
    for step in range(steps):
        population.step(dt)
        alive = [a for a in world.agents if a.alive]
        for agent in alive:
            behavior[agent.behavior_state] += 1
        if starvation_seen is None and any(e["cause"] == "starved" for e in death_events):
            starvation_seen = round(world.t, 2)
        if step % max(1, steps // 20) == 0:
            snapshots.append(
                {
                    "t": round(world.t, 1),
                    "alive": len(alive),
                    "cumulative_created": len(created),
                    "deaths": len(death_events),
                    "community_built": world.technology.community.built,
                    "ages": [round(a.age) for a in alive],
                    "energies": [round(a.physiology.energy) for a in alive],
                }
            )
        if not alive:
            break
    elapsed = time.perf_counter() - started
    total_steps = step + 1
    return {
        "config": {"steps": steps, "dt": dt, "size": [width, height], "seed": seed,
                   "perception": with_perception, "world_version": __import__("newworld").WORLD_VERSION},
        "timing": {"wall_s": round(elapsed, 2), "steps_per_s": round(total_steps / max(elapsed, 1e-9), 1)},
        "cumulative_created": len(created),
        "created_at": created,
        "max_alive_ever": max((s["alive"] for s in snapshots), default=0),
        "deaths": death_events,
        "first_starvation_t": starvation_seen,
        "extinct": not any(a.alive for a in world.agents),
        "extinction_t": round(world.t, 2) if not any(a.alive for a in world.agents) else None,
        "cap_expected": 8,
        "cap_violated": len(created) > 8,
        "final_alive": len([a for a in world.agents if a.alive]),
        "community_built": world.technology.community.built,
        "behavior_state_steps": dict(behavior.most_common()),
        "snapshots": snapshots,
        "note": "原始系统：死亡即删除、无历史注册表、无知识/记忆/目标；上限只限制存活数。",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="pre")
    parser.add_argument("--perception", action="store_true")
    args = parser.parse_args()
    result = run(args.steps, args.dt, args.width, args.height, args.seed, args.perception)
    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"baseline_long_{args.tag}_seed{args.seed}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: v for k, v in result.items() if k not in ("created_at", "behavior_state_steps", "snapshots")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
