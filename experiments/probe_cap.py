"""P0 人口上限探针：证明原始系统"创建数上限"不是历史累计约束。

原始代码 `society.MAX_POPULATION = 8` 只检查 `len(world.agents)`（当前存活数）；
死亡时 `world.remove_agent()` 会把个体移出列表，于是名额被释放，
**历史累计创建数会突破 8**。本脚本用粗时间步长跑足够长的生命周期来实测。

运行::

    cd newworldpro
    python experiments/probe_cap.py --steps 4000 --dt 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newworld.config import make_config  # noqa: E402
from newworld.population import Population  # noqa: E402
from newworld.world import World  # noqa: E402


def run(steps: int, dt: float, width: int, height: int, seed: int) -> dict[str, object]:
    cfg = make_config(width, height)
    world = World(cfg, seed=seed)
    world.reset(seed=seed)

    created: set[int] = {agent.agent_id for agent in world.agents}
    original_spawn = world.spawn

    def spawn_counting(*args, **kwargs):  # type: ignore[no-untyped-def]
        character = original_spawn(*args, **kwargs)
        created.add(character.agent_id)
        return character

    world.spawn = spawn_counting  # type: ignore[method-assign]
    population = Population(world, seed=seed)

    deaths = 0
    alive_peak = 0
    stage_log: list[dict[str, object]] = []
    for step in range(steps):
        before = len([a for a in world.agents if a.alive])
        population.step(dt)
        after = [a for a in world.agents if a.alive]
        deaths += max(0, before - len(after))
        alive_peak = max(alive_peak, len(after))
        if step % max(1, steps // 12) == 0:
            stage_log.append(
                {
                    "t": round(world.t, 1),
                    "alive": len(after),
                    "cumulative_created": len(created),
                    "deaths": deaths,
                    "knowledge_kept": 0,
                }
            )
        if not after:
            break
    return {
        "config": {"steps": steps, "dt": dt, "size": [width, height], "seed": seed},
        "cumulative_created": len(created),
        "max_alive_ever": alive_peak,
        "deaths": deaths,
        "cap_expected": 8,
        "cap_violated": len(created) > 8,
        "final_alive": len([a for a in world.agents if a.alive]),
        "final_world_time_s": round(world.t, 1),
        "stage_log": stage_log,
        "note": "原始系统死亡即删除，无历史注册表；创建数只受存活数限制，故突破 8。",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run(args.steps, args.dt, args.width, args.height, args.seed)
    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"baseline_cap_seed{args.seed}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[saved] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
