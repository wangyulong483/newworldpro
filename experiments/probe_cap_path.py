"""P0 上限路径受控复现：证明原始系统"死亡释放名额 → 创建第 9 个个体"。

不依赖随机长跑：直接构造 8 个已创建个体，让 1 个老死，再让两名能量充足的
成年个体在近距离、高亲密度下触发怀孕→分娩，检查累计创建数是否变成 9。

运行::

    cd newworldpro
    python experiments/probe_cap_path.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newworld.config import make_config  # noqa: E402
from newworld.society import MAX_POPULATION, Society  # noqa: E402
from newworld.world import World  # noqa: E402


def main() -> int:
    cfg = make_config(800, 500)
    world = World(cfg, seed=0)
    world.reset(seed=0)
    world.agents.clear()

    created: list[int] = []
    for index in range(MAX_POPULATION):
        sex = "female" if index % 2 else "male"
        agent = world.spawn(sex=sex, age=200.0, x=300.0 + index * 40.0)
        created.append(agent.agent_id)
    society = Society(seed=0)

    # 让 0 号老死 → 存活 7 人 → 名额被释放
    world.agents[0].age = 10_000.0
    society.update(world, 0.1)
    after_death_created = len(created)
    alive_after_death = [a for a in world.agents if a.alive]

    # 两名成年个体：距离近、亲密度高、能量充足
    male = next(a for a in alive_after_death if a.sex == "male")
    female = next(a for a in alive_after_death if a.sex == "female")
    male.x = female.x + 10.0
    male.physiology.energy = 95.0
    female.physiology.energy = 95.0
    society.relate(male.agent_id, female.agent_id, 1.0)
    # 跑若干步让怀孕→分娩发生（妊娠 24 世界秒，dt=0.5 → 至少 48 步）
    for _ in range(80):
        society.update(world, 0.5)

    cumulative_created = len([a for a in world.agents]) + (MAX_POPULATION - len(alive_after_death))
    # 上一行不可靠（死者已删除），直接用"活着的 + 已死亡的"推导：
    cumulative_created = len(alive_after_death) + 1 + max(0, len(world.agents) - len(alive_after_death))
    # 更直接：记录 total unique ids ever seen
    seen = set(created) | {a.agent_id for a in world.agents}
    result = {
        "max_population_constant": MAX_POPULATION,
        "created_initial": len(created),
        "alive_after_old_age_death": len(alive_after_death),
        "births_after_death": society.births,
        "alive_at_end": len([a for a in world.agents if a.alive]),
        "cumulative_created_upper_bound": len(seen),
        "cap_broken": len(seen) > MAX_POPULATION,
        "agent_ids_seen": sorted(seen),
        "note": "原始代码用 len(world.agents)（存活数）判定上限；死亡释放名额后仍可生育，"
                "历史累计创建数因此可超过 8。本脚本是受控构造，不依赖长时程随机模拟。",
    }
    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "baseline_cap_path.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[saved] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
