"""Headless civilization runner: experiments without a window.

``SocietyRunner`` builds a world + population from a :class:`RunSpec`, runs the
simulation for a fixed budget, applies seeded **environment disturbances** (P4),
collects metrics with :class:`~newworld.telemetry.Telemetry` and returns a
JSON-serialisable result.  This is the entry point used by ``experiments/``.

Design notes
------------
* The runner never modifies world data behind the agents' backs: disturbances go
  through the public :class:`~newworld.pixelworld.PixelScene` / ecology APIs,
  exactly like weather or a foraging agent would.
* The disturbance schedule is **seeded, not scripted**: every run of the
  ``enable_disturbance=True`` groups experiences one drought, one storm and one
  resource-scarcity event, each at a seeded random time in the first minutes, so
  all three kinds actually occur while the development route is still open.
  (Measured: scheduling only a random kind left groups C and D identical,
  because a single late shock cannot change a route that is already decided.)
* Perception costs ~95% of a frame and the neural feature extractor ignores it
  (``neural.features`` never reads ``perception``), so experiments disable it by
  default via ``spec.perception = False`` — behavioural results are identical,
  the wall-clock is ~30x faster.

CLI::

    python -m newworld.simulation --group D --seed 0 --steps 20000 --dt 0.05
    python -m newworld.simulation --group A --seed 1 --steps 5000 --save run.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    ABLATIONS,
    CivilizationConfig,
    WorldConfig,
    make_civilization_config,
    make_config,
)
from .population import Population
from .telemetry import Telemetry
from .world import World

#: kinds of seeded environment disturbance (P4)
DISTURBANCE_KINDS: tuple[str, ...] = ("drought", "storm", "scarcity")


@dataclass
class RunSpec:
    """Everything that defines one experiment run (reproducible by design)."""

    group: str = "D"
    seed: int = 0
    steps: int = 20000
    dt: float = 0.05
    width: int = 800
    height: int = 500
    #: wall-clock / behaviour switches that are experiment-level, not ablations
    perception: bool = False
    disturbance: bool | None = None          # None -> follow the ablation group
    telemetry_window_s: float = 60.0
    log_dir: str | Path | None = None        # events + summary written here
    save_path: str | Path | None = None      # full world state at the end
    #: stop the run at the first extinction (the default: record collapse)
    stop_on_extinction: bool = True
    #: optional budget for the neural behaviour-cloning training round
    train_rounds: int = 0

    def as_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["log_dir"] = None if self.log_dir is None else str(self.log_dir)
        data["save_path"] = None if self.save_path is None else str(self.save_path)
        return data


@dataclass
class EnvironmentDisturbance:
    """Seeded environmental shocks (drought / storm / resource scarcity).

    Each event has a duration and is applied to the *real* world state (rain
    target, plant/forest destruction, soil moisture), so it changes the viable
    development route instead of being a UI flag.
    """

    seed: int = 0
    enabled: bool = True
    #: first event window (world seconds) — early enough to matter
    first_min_s: float = 60.0
    first_max_s: float = 150.0
    #: gap between consecutive events
    interval_min_s: float = 110.0
    interval_max_s: float = 210.0
    duration_s: float = 90.0
    rng: random.Random = field(init=False, repr=False)
    schedule: list[tuple[float, str]] = field(init=False, default_factory=list)
    active_kind: str = ""
    active_until: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed * 7919 + 13)
        if self.enabled:
            kinds = list(DISTURBANCE_KINDS)
            self.rng.shuffle(kinds)
            when = self.rng.uniform(self.first_min_s, self.first_max_s)
            for kind in kinds:
                self.schedule.append((when, kind))
                when += self.rng.uniform(self.interval_min_s, self.interval_max_s)

    # ------------------------------------------------------------------
    def apply(self, world, telemetry: Telemetry | None = None) -> None:
        """Advance the disturbance schedule; called once per simulation step."""
        if not self.enabled:
            return
        now = world.t
        ecology = world.pixel_scene.ecology
        if self.active_kind and now >= self.active_until:
            self.active_kind = ""
        while self.schedule and now >= self.schedule[0][0] and not self.active_kind:
            _, kind = self.schedule.pop(0)
            self.active_kind = kind
            self.active_until = now + self.duration_s
            record = {"t": round(now, 2), "kind": kind, "duration_s": self.duration_s}
            self.history.append(record)
            if telemetry is not None:
                try:
                    telemetry.event("disturbance", **record)
                except Exception:
                    pass
            # one-shot impact at the start of the event: it must be able to
            # change the development route, not just the weather display
            if kind == "scarcity":
                self._remove_resources(world)
            elif kind == "storm":
                self._destroy_plant(world)
            elif kind == "drought":
                self._dry_out(world)
        if self.active_kind == "drought":
            # no rain and fast evaporation: the pond shrinks, growth stops
            ecology.rain_target = 0.0
            ecology.rain = max(0.0, ecology.rain - 0.05)
            ecology.rain_timer = max(ecology.rain_timer, self.duration_s)
        elif self.active_kind == "storm":
            ecology.rain_target = 0.95
            self._storm_damage(world)

    # ------------------------------------------------------------------
    def _remove_resources(self, world) -> None:
        """Scarcity: wipe out one whole tree and one whole rock body.

        Removing scattered atoms had no measurable effect (verified in the first
        ablation batch: groups C and D were identical), so this removes the
        objects the planner actually navigates to — a real supply shock.
        """
        scene = world.pixel_scene
        from . import materials

        for target_kind, material_key in (("tree", "wood"), ("rock", "stone")):
            candidates = [obj for obj in scene.objects if obj.kind == target_kind]
            if not candidates:
                continue
            victim = self.rng.choice(candidates)
            material = materials.MATERIAL_INDEX[material_key]
            leaf = materials.MATERIAL_INDEX["leaf"]
            for y in range(max(0, victim.top), min(scene.ah, victim.bottom + 1)):
                for x in range(max(0, victim.x0), min(scene.aw, victim.x1 + 1)):
                    if int(scene.grid[y, x]) in (material, leaf):
                        scene.clear_atom(x, y)
            if victim in scene.objects:
                scene.objects.remove(victim)
        scene.refresh_caches()

    def _destroy_plant(self, world) -> None:
        """Storm: tear down a whole berry bush (a food supply shock)."""
        scene = world.pixel_scene
        bushes = [obj for obj in scene.objects if obj.kind == "bush"]
        if not bushes:
            return
        victim = self.rng.choice(bushes)
        for slot in list(getattr(victim, "berry_slots", ())):
            if slot in scene.berries:
                scene.berries.remove(slot)
        from . import materials

        leaf = materials.MATERIAL_INDEX["leaf"]
        berry = materials.MATERIAL_INDEX["berry"]
        for y in range(max(0, victim.top), min(scene.ah, victim.bottom + 1)):
            for x in range(max(0, victim.x0), min(scene.aw, victim.x1 + 1)):
                if int(scene.grid[y, x]) in (leaf, berry):
                    scene.clear_atom(x, y)
        if victim in scene.objects:
            scene.objects.remove(victim)
        scene.refresh_caches()

    def _dry_out(self, world) -> None:
        """Drought: soil moisture collapses, so regrowth stops for a while."""
        ecology = world.pixel_scene.ecology
        ecology.moisture = (ecology.moisture * 0.35).astype(ecology.moisture.dtype)

    def _storm_damage(self, world) -> None:
        """Storm: tear soft plant atoms at a few random columns."""
        scene = world.pixel_scene
        for _ in range(2):
            x = self.rng.uniform(0.0, float(world.config.width))
            y = float(world.config.ground_y)
            scene.trample(x, y, force=3.0)


class SocietyRunner:
    """Run one civilization experiment headlessly."""

    def __init__(self, spec: RunSpec | None = None) -> None:
        self.spec = spec if spec is not None else RunSpec()
        civ_config = make_civilization_config(
            self.spec.group,
            seed=self.spec.seed,
            enable_perception=bool(self.spec.perception),
        )
        if self.spec.disturbance is not None:
            civ_config.enable_disturbance = bool(self.spec.disturbance)
        self.civ_config: CivilizationConfig = civ_config
        self.world_config: WorldConfig = make_config(self.spec.width, self.spec.height)
        self.world = World(self.world_config, seed=self.spec.seed, civ_config=self.civ_config)
        self.world.reset(seed=self.spec.seed)
        self.events_path = None
        if self.spec.log_dir is not None:
            log_dir = Path(self.spec.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.events_path = log_dir / f"events_{self.spec.group}_seed{self.spec.seed}.jsonl"
        self.telemetry = Telemetry(
            window_s=self.spec.telemetry_window_s,
            log_path=self.events_path,
            seed=self.spec.seed,
            group=self.spec.group,
        )
        self.population = Population(self.world, self.spec.seed, telemetry=self.telemetry)
        self.disturbance = EnvironmentDisturbance(
            seed=self.spec.seed,
            enabled=bool(self.civ_config.enable_disturbance),
        )
        self.result: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        spec = self.spec
        started = time.perf_counter()
        steps_run = 0
        for step in range(spec.steps):
            self.disturbance.apply(self.world, self.telemetry)
            self.population.step(spec.dt)
            steps_run = step + 1
            if self.world.civilization.extinct and spec.stop_on_extinction:
                break
        for _ in range(spec.train_rounds):
            self.population.train_round(seed=spec.seed)
        summary = self.telemetry.summary()
        elapsed = time.perf_counter() - started
        self.result = {
            "spec": spec.as_dict(),
            "civ_config": self.civ_config.as_dict(),
            "world_config": self.world_config.as_dict(),
            "steps_run": steps_run,
            "world_time_s": round(self.world.t, 3),
            "wall_s": round(elapsed, 3),
            "steps_per_s": round(steps_run / max(elapsed, 1e-9), 1),
            "metrics": summary,
            "civilization": self.world.civilization.metrics(self.world),
            "goals": self.population.goals.stats() if self.population.goals else {},
            "knowledge": {
                "verified": self.world.knowledge.verified_ids(),
                "technologies": {
                    knowledge.knowledge_id: {
                        "status": knowledge.status,
                        "confidence": round(knowledge.confidence, 3),
                        "discovered_by": knowledge.discovered_by,
                        "discovered_at": round(knowledge.discovered_at, 2),
                        "successful_uses": knowledge.successful_uses,
                        "failed_uses": knowledge.failed_uses,
                    }
                    for knowledge in self.world.knowledge.all()
                },
                "graph_depth": self.world.knowledge.depth(),
                "graph_edges": len(self.world.knowledge.verified_edges()),
                "innovation_efficiency": round(self.world.knowledge.innovation_efficiency(), 4),
                "blocked_attempts": getattr(self.world.knowledge, "blocked_attempts", 0),
            },
            "archive": self.population.archive_summary(),
            "disturbances": list(self.disturbance.history),
            "telemetry_windows": getattr(self.telemetry, "windows", []),
            "events_count": len(getattr(self.telemetry, "events", ()) or ()),
            "extra_events": {"creation_denied": list(self.world.creation_denied)},
        }
        self.telemetry.close()
        self._write_summary()
        return self.result

    # ------------------------------------------------------------------
    def _write_summary(self) -> None:
        if self.result is None or self.spec.log_dir is None:
            return
        log_dir = Path(self.spec.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"summary_{self.spec.group}_seed{self.spec.seed}.json"
        path.write_text(
            json.dumps(self.result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def save(self, path: str | Path) -> Path:
        """Persist the full world state (registry, knowledge, memory, civ)."""
        return self.world.save(path)

    @classmethod
    def load(cls, path: str | Path) -> World:
        """Load a world state saved by :meth:`save`."""
        return World.load(path)

    # ------------------------------------------------------------------
    def summary_lines(self) -> list[str]:
        if self.result is None:
            return ["(not run yet)"]
        metrics = self.result["metrics"]
        civilization = self.result["civilization"]
        return [
            f"group={self.spec.group} seed={self.spec.seed} steps={self.result['steps_run']} "
            f"t={self.result['world_time_s']}s wall={self.result['wall_s']}s",
            f"total_agents_created={metrics.get('total_agents_created')} "
            f"alive={civilization.get('alive_count')} deaths={civilization.get('total_deaths')}",
            f"extinct={civilization.get('extinct')} at={civilization.get('extinction_time')} "
            f"cause={civilization.get('extinction_cause')}",
            f"tech_verified={metrics.get('tech_verified')} depth={metrics.get('tech_graph_depth')} "
            f"retention={metrics.get('knowledge_retention')}",
            f"goal_completion_rate={metrics.get('goal_completion_rate')} "
            f"repeat_run_max={metrics.get('repeat_run_max')} "
            f"no_progress={metrics.get('no_progress_repeat_ratio')}",
            f"collapse_state={civilization.get('collapse_state_code')} "
            f"cooperation_index={civilization.get('cooperation_index')}",
            f"disturbances={self.result.get('disturbances')}",
        ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="headless civilization experiment runner")
    parser.add_argument("--group", default="D", choices=sorted(ABLATIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--perception", action="store_true", help="enable visual perception (slow)")
    parser.add_argument("--disturbance", choices=("on", "off"), default=None)
    parser.add_argument("--train-rounds", type=int, default=0)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--json", default=None, help="write the full result JSON here")
    parser.add_argument("--keep-going", action="store_true", help="do not stop at extinction")
    args = parser.parse_args(argv)

    spec = RunSpec(
        group=args.group,
        seed=args.seed,
        steps=args.steps,
        dt=args.dt,
        width=args.width,
        height=args.height,
        perception=args.perception,
        disturbance=None if args.disturbance is None else args.disturbance == "on",
        train_rounds=args.train_rounds,
        log_dir=args.log_dir,
        save_path=args.save,
        stop_on_extinction=not args.keep_going,
    )
    runner = SocietyRunner(spec)
    result = runner.run()
    if args.save:
        runner.save(args.save)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    print("\n".join(runner.summary_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
