"""The world: geometry, inhabitants, physics (mechanics, acoustics, heat).

``World`` is the pure simulation.  It knows nothing about tkinter, PIL or
rendering, which makes it cheap to import, easy to test and safe to run in a
headless training loop.

Besides the inhabitants it carries the physical environment:

* **gravity** — every inhabitant falls with ``g = 9.80665 m/s^2``
* **sun** — a black-body source of light and radiative heat
* **air** — a medium at 20 C that carries sound and applies drag
* **sound field** — pressure waves (speech, future environmental sounds)

Rendering lives in :mod:`newworld.render`; the reinforcement-learning wrapper
lives in :mod:`newworld.env`.
"""

from __future__ import annotations

import random

from . import physics
from .acoustics import SpeechEvent, SoundField, synthesize_speech
from .character import Character, mouth_position
from .civilization import AgentRegistry, CivilizationState, PopulationLimitError
from .config import CivilizationConfig, WorldConfig
from .daynight import DayCycle
from .environment import Air, Sun, step_body_temperature
from .knowledge import KnowledgeStore
from .memory import MemoryStore
from .optics import Scene, Shape
from .physiology import Physiology
from .pixelworld import PixelScene
from .technology import Technology


class NoAgentsError(RuntimeError):
    """Raised when code asks for ``world.agent`` but nobody is alive.

    The original world silently spawned a replacement inhabitant on attribute
    access, which would let a civilization "resurrect" itself after extinction
    and bypass the 8-agent creation budget.  Creation must be explicit now
    (:meth:`World.create_agent`).
    """


class World:
    """A screen-sized physical world inhabited by :class:`Character`\\ s.

    Parameters
    ----------
    config:
        Geometry / physics / palette.  Defaults to :class:`WorldConfig` with
        the current screen resolution.
    seed:
        Seed for the internal :class:`random.Random`, used to place
        inhabitants and to make up speech.
    civ_config:
        Civilization-layer switches and the **hard** creation budget
        (:class:`~newworld.config.CivilizationConfig`).  Defaults to all
        mechanisms on with ``max_agents_created = 8``.
    """

    def __init__(
        self,
        config: WorldConfig | None = None,
        seed: int | None = None,
        *,
        civ_config: CivilizationConfig | None = None,
    ) -> None:
        self.config = config if config is not None else WorldConfig()
        self.rng = random.Random(seed)
        self.seed_value = seed
        self.civ_config = civ_config if civ_config is not None else CivilizationConfig(seed=seed or 0)
        self.agents: list[Character] = []
        #: visible 2-D objects added manually (natural objects live in pixel_scene)
        self.objects: list[Shape] = []
        #: pixel-atom world: terrain, rocks, trees, bushes, pond
        self.pixel_scene = PixelScene(self.config, self.seed_value or 0)
        #: community/tool layer (gathering, crafting, building)
        self.technology = Technology(self.config, self.pixel_scene, seed=self.seed_value or 0)
        #: historical registry: every agent ever created (dead ones are archived)
        self.registry = AgentRegistry(self.civ_config.max_agents_created)
        #: individual episodic / procedural memory (kept after death)
        self.memory = MemoryStore()
        #: persistent, verifiable knowledge (recipes, status, validation history)
        self.knowledge = KnowledgeStore(
            seed=self.seed_value or 0,
            bootstrap=bool(self.civ_config.enable_knowledge),
        )
        #: civilization-level state: facilities, contracts, collapse, extinction
        self.civilization = CivilizationState(
            self.registry,
            knowledge=self.knowledge,
            memory=self.memory,
            config=self.civ_config,
        )
        #: whether the neural observation is expanded to 40 inputs (P3)
        self.expanded_observation = bool(self.civ_config.expanded_observation)
        #: set by Population: the goal manager, so features() can read plan state
        self.goal_manager = None
        #: virtual clock and solar geometry
        self.day = DayCycle()
        #: physical environment
        self.sun = Sun()
        self.air = Air()
        self.sound = SoundField()
        self.hour: float = self.day.start_hour
        self.daylight: float = self.day.daylight(self.hour)
        self.t: float = 0.0
        self.steps: int = 0
        self._next_id = 0
        #: denied creation attempts (budget exhausted), for telemetry
        self.creation_denied: list[dict[str, object]] = []
        #: deliberate, budget-checked creation of the first inhabitant
        self.create_agent()

    # ------------------------------------------------------------------
    # scene / light field
    # ------------------------------------------------------------------
    def scene(self) -> Scene:
        """The visible world for the inhabitant's eyes (pixel objects + extras)."""
        base = self.pixel_scene.optics_scene()
        return Scene(objects=tuple(base.objects) + tuple(self.objects))

    def _update_environment(self) -> None:
        """Advance the solar clock and the air temperature."""
        self.hour = self.day.hour(self.t)
        self.daylight = self.day.daylight(self.hour)
        self.sun.elevation_deg = self.day.sun_elevation_deg(self.hour)
        # simple diurnal air temperature: 10 C at night, 24 C at noon
        self.air.temperature_k = 283.15 + 14.0 * self.daylight

    def add_object(self, obj: Shape) -> Shape:
        """Add a visible object to the world (used by future experiments)."""
        self.objects.append(obj)
        return obj

    def clear_objects(self) -> None:
        self.objects.clear()

    # ------------------------------------------------------------------
    # inhabitants
    # ------------------------------------------------------------------
    @property
    def agent(self) -> Character:
        """First inhabitant.

        Raises :class:`NoAgentsError` when nobody is alive: the world never
        spawns an inhabitant implicitly, because that would let an extinct
        civilization resurrect itself and bypass the creation budget.
        """
        if not self.agents:
            raise NoAgentsError(
                "no inhabitants: creation must go through World.create_agent()"
            )
        return self.agents[0]

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    @property
    def alive_agents(self) -> list[Character]:
        return [character for character in self.agents if character.alive]

    def spawn(
        self,
        x: float | None = None,
        facing: int = 1,
        agent_id: int | None = None,
        *,
        name: str = "",
        sex: str = "male",
        age: float = 0.0,
        generation: int = 0,
        body_scale: float = 1.0,
        parents: tuple[int, ...] = (),
    ) -> Character:
        """Low-level *geometric* spawn (tests, previews, tooling).

        **This does not simulate a creation event**: it bypasses the registry and
        the 8-agent budget.  Everything inside the simulation must use
        :meth:`create_agent` instead.
        """
        cfg = self.config
        if x is None:
            x = self.rng.uniform(cfg.min_x, cfg.max_x)
        character = Character(
            x=float(x),
            facing=facing,
            agent_id=self._next_id if agent_id is None else agent_id,
            name=name,
            sex=sex,
            age=age,
            generation=generation,
            body_scale=body_scale,
            parents=parents,
            parent_id=(parents[0] if parents else None),
        )
        character.attach(cfg)
        self._next_id = max(self._next_id, character.agent_id) + 1
        self.agents.append(character)
        return character

    def create_agent(
        self,
        x: float | None = None,
        facing: int = 1,
        *,
        name: str = "",
        sex: str = "male",
        age: float = 0.0,
        generation: int = 0,
        body_scale: float = 1.0,
        parents: tuple[int, ...] = (),
        parent_id: int | None = None,
        cause: str = "birth",
    ) -> Character | None:
        """Simulation-authoritative creation: respects the **total** budget.

        Returns ``None`` when the historical creation budget
        (``civ_config.max_agents_created``) is exhausted; the denial is recorded
        in :attr:`civilization` so telemetry can report it.  This is the *only*
        legal way to add an inhabitant during a simulation.
        """
        if not self.registry.can_create():
            self.creation_denied.append(
                {"t": round(self.t, 3), "cause": cause, "total_created": self.registry.total_created}
            )
            return None
        character = self.spawn(
            x=x,
            facing=facing,
            name=name,
            sex=sex,
            age=age,
            generation=generation,
            body_scale=body_scale,
            parents=parents,
        )
        if parent_id is None and parents:
            parent_id = parents[0]
        character.created_at = self.t
        character.parent_id = parent_id
        self.registry.register(character, t=self.t, parent_id=parent_id)
        return character

    def adopt_agent(self, character: Character, *, t: float | None = None) -> None:
        """Register a pre-existing (geometrically spawned) inhabitant.

        Used by :class:`~newworld.population.Population` when it takes over a
        world that was built before the registry existed (tests, previews).
        """
        if self.registry.record(character.agent_id) is not None:
            return
        character.created_at = self.t if t is None else t
        self.registry.register(character, t=character.created_at, parent_id=character.parent_id)

    def remove_agent(self, agent_id: int) -> None:
        """Remove an inhabitant from the *live* list (death/emigration).

        The historical record is **not** deleted: call
        :meth:`~newworld.civilization.AgentRegistry.mark_dead` first (or use
        :meth:`Population.report_death`) so the agent stays in the archive.
        """
        self.agents = [character for character in self.agents if character.agent_id != agent_id]

    def agent_by_id(self, agent_id: int) -> Character | None:
        for character in self.agents:
            if character.agent_id == agent_id:
                return character
        return None

    def reset(
        self,
        seed: int | None = None,
        positions: list[float] | None = None,
    ) -> list[tuple[float, ...]]:
        """Reset the world to ``t = 0`` and return the initial observations.

        ``positions`` optionally fixes the x position of every inhabitant
        (length must match the number of inhabitants).  Without it, each
        inhabitant is placed at a random legal location.

        .. note::
           This is a **development / test helper** (re-positioning and re-seeding
           the terrain), not a civilization reset: the historical registry,
           knowledge and memory deliberately survive it.  A fresh experiment
           should construct a new :class:`World` instead.
        """
        if seed is not None:
            self.seed_value = seed
            self.rng.seed(seed)
        if positions is not None and len(positions) != len(self.agents):
            raise ValueError(
                f"positions length {len(positions)} != agent count {len(self.agents)}"
            )
        self.t = 0.0
        self.steps = 0
        self.sound = SoundField()
        if seed is not None:
            self.pixel_scene = PixelScene(self.config, seed)
            self.technology = Technology(self.config, self.pixel_scene, seed=seed)
        self.technology.pending_facilities.clear()
        self._update_environment()
        for index, character in enumerate(self.agents):
            character.attach(self.config)  # config may have been resized
            character.x = (
                float(positions[index])
                if positions is not None
                else self.rng.uniform(self.config.min_x, self.config.max_x)
            )
            character.x = min(max(character.x, self.config.min_x), self.config.max_x)
            character.y = self.config.ground_y
            character.vx = 0.0
            character.vy = 0.0
            character.on_ground = True
            character.phase = 0.0
            character.distance_walked = 0.0
            character.steps = 0
            character.body_temperature_k = physics.HUMAN_BODY_TEMPERATURE_K
            character.utterance = None
            character.utterance_until = 0.0
            character.jump_count = 0
            character.speech_count = 0
            character.behavior_state = "explore"
            character.physiology = Physiology()
        return self.observations()

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    @staticmethod
    def _split_action(action) -> tuple[int, bool]:
        move = getattr(action, "move", action)
        jump = bool(getattr(action, "jump", False))
        return int(move), jump

    def step(
        self,
        dt: float | None = None,
        actions: int | list | None = None,
    ) -> list[tuple[float, ...]]:
        """Advance the world by ``dt`` seconds.

        ``actions`` may be a single action (applied to every inhabitant) or one
        action per inhabitant.  An action is either an integer ``-1/0/+1`` or an
        object with ``.move`` and ``.jump`` attributes
        (:class:`newworld.autonomy.Decision`).  Returns the normalised
        observation of every inhabitant.
        """
        dt = self.config.fixed_dt if dt is None else float(dt)
        count = len(self.agents)
        if count == 0:
            self.t += dt
            self.steps += 1
            self.sound.advance(dt)
            return []
        if actions is None:
            action_list = [0] * count
        elif isinstance(actions, int):
            action_list = [actions] * count
        elif not isinstance(actions, (list, tuple)):
            action_list = [actions] * count
        else:
            action_list = list(actions)
            if len(action_list) == 1 and count > 1:
                action_list = action_list * count
            elif len(action_list) != count:
                raise ValueError(
                    f"expected {count} actions, got {len(action_list)}"
                )
        self.t += dt
        self._update_environment()
        for character, action in zip(self.agents, action_list):
            move, jump = self._split_action(action)
            character.update(dt, move, self.config, jump=jump)
            step_body_temperature(
                character,
                self.air,
                self.sun,
                dt,
                activity=1.2 if move != 0 else 1.0,
            )
            character.physiology.update(
                dt,
                hour=self.hour,
                daylight=self.daylight,
                moving=move != 0,
                jumped=jump,
                body_temperature_c=character.body_temperature_c,
            )
        # ecology: soil-water cycle, growth and atomic destruction
        self.pixel_scene.update_ecology(dt, self.day, self.hour)
        if abs(self.agent.vx) > 40.0 and self.rng.random() < dt * 0.45:
            feet = self.agent.y if self.agent.y is not None else self.config.ground_y
            self.pixel_scene.trample(self.agent.x, feet, force=1.4)
        self.steps += 1
        self.sound.advance(dt)
        return self.observations()

    # ------------------------------------------------------------------
    # acoustics
    # ------------------------------------------------------------------
    def speak(self, character: Character, text: str | None = None) -> SpeechEvent:
        """Make the inhabitant say something (default: a made-up word)."""
        from .acoustics import random_word

        phrase = text if text is not None else random_word(self.rng)
        origin_x, origin_y = mouth_position(character, self.config)
        event = synthesize_speech(
            origin_x,
            origin_y,
            self.t,
            phrase,
            self.rng,
            speaker_id=character.agent_id,
        )
        for wave in event.waves:
            self.sound.add(wave)
        character.utterance = phrase
        character.utterance_until = self.t + event.duration_s + 0.3
        character.speech_count += 1
        return event

    def sound_pressure_at(self, x: float, y: float) -> float:
        """Current sound pressure at a world point (Pa)."""
        return self.sound.pressure_at(x, y, self.t)

    def sound_rms_at(self, x: float, y: float) -> float:
        return self.sound.rms_pressure_at(x, y, self.t)

    # ------------------------------------------------------------------
    # environment snapshot
    # ------------------------------------------------------------------
    def environment_state(self) -> dict[str, float]:
        return {
            "gravity_m_s2": physics.STANDARD_GRAVITY,
            "hour": self.hour,
            "daylight": self.daylight,
            "sun_elevation_deg": self.sun.elevation_deg,
            "solar_irradiance_w_m2": self.sun.horizontal_irradiance,
            "sun_temperature_k": self.sun.temperature_k,
            "visible_irradiance_w_m2": self.sun.visible_irradiance,
            "air_temperature_c": self.air.temperature_c,
            "air_density_kg_m3": self.air.density_kg_m3,
            "speed_of_sound_m_s": self.air.speed_of_sound_m_s,
        }

    def observations(self) -> list[tuple[float, ...]]:
        return [character.observation(self.config) for character in self.agents]

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def state(self) -> dict[str, object]:
        """Serialisable snapshot (handy for JSONL research logs)."""
        return {
            "t": round(self.t, 6),
            "steps": self.steps,
            "seed": self.seed_value,
            "world": self.config.as_dict(),
            "civ_config": self.civ_config.as_dict(),
            "environment": self.environment_state(),
            "agents": [
                {
                    **character.to_dict(),
                    "at_left_wall": character.at_left_wall,
                    "at_right_wall": character.at_right_wall,
                }
                for character in self.agents
            ],
            "population": {
                "alive": len(self.alive_agents),
                "total_agents_created": self.registry.total_created,
                "max_agents_created": self.civ_config.max_agents_created,
                "total_deaths": self.registry.total_deaths,
                "extinct": self.civilization.extinct,
                "collapse_state": self.civilization.collapse_state,
            },
            "technology": self.technology.summary_zh(),
            "knowledge": {
                "verified": len(self.knowledge.verified_ids()),
                "total": len(self.knowledge.all()),
                "graph_depth": self.knowledge.depth(),
            },
            "facilities": [facility.to_dict() for facility in self.civilization.facilities],
        }

    # ------------------------------------------------------------------
    # serialisation (world state save / load)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """Full serialisable world state (used by save/load tests)."""
        # Snapshot living agents into the registry first: otherwise inventory /
        # knowledge / skills of *living* individuals would not be part of the
        # historical record and a save/load round trip would lose them.
        for character in self.agents:
            record = self.registry.record(character.agent_id)
            if record is None:
                continue
            record.known_knowledge_ids = tuple(character.known_knowledge_ids)
            record.skills = {str(key): float(value) for key, value in character.skills.items()}
            record.inventory = {
                "wood": int(character.inventory.wood),
                "stone": int(character.inventory.stone),
                "fiber": int(character.inventory.fiber),
                "cord": int(character.inventory.cord),
                "basket": int(character.inventory.basket),
            }
        return {
            "state": self.state(),
            "registry": self.registry.to_dict(),
            "knowledge": self.knowledge.to_dict(),
            "memory": self.memory.to_dict(),
            "civilization": self.civilization.to_dict(),
        }

    def save(self, path: str | Path) -> Path:
        """Write the world state to a JSON file."""
        import json
        from pathlib import Path as _Path

        path = _Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path, config: WorldConfig | None = None) -> "World":
        """Rebuild a world from :meth:`save` output (registry + knowledge + civ)."""
        import json
        from pathlib import Path as _Path

        data = json.loads(_Path(path).read_text(encoding="utf-8"))
        civ_data = data["state"].get("civ_config", {})
        civ_config = CivilizationConfig(**{
            key: civ_data[key]
            for key in civ_data
            if key in CivilizationConfig.__dataclass_fields__
        })
        world = cls(config, seed=int(data["state"].get("seed") or 0), civ_config=civ_config)
        world.registry = AgentRegistry.from_dict(data["registry"])
        world.knowledge = KnowledgeStore.from_dict(data["knowledge"])
        world.memory = MemoryStore.from_dict(data["memory"])
        world.civilization = CivilizationState.from_dict(
            data["civilization"],
            knowledge=world.knowledge,
            memory=world.memory,
        )
        # live inhabitants: only the ones still registered as alive
        world.agents.clear()
        for record in world.registry.records():
            if not record.alive:
                continue
            agent = world.spawn(
                x=0.0,
                name=record.name,
                sex=record.sex,
                age=0.0,
                generation=record.generation,
                agent_id=record.agent_id,
            )
            agent.created_at = record.born_at
            agent.parent_id = record.parent_id
            agent.inventory.wood = int(record.inventory.get("wood", 0))
            agent.inventory.stone = int(record.inventory.get("stone", 0))
            agent.inventory.fiber = int(record.inventory.get("fiber", 0))
            agent.inventory.cord = int(record.inventory.get("cord", 0))
            agent.inventory.tool = record.inventory.get("tool")  # type: ignore[assignment]
            agent.known_knowledge_ids = list(record.known_knowledge_ids)
            agent.skills = dict(record.skills)
        state = data["state"]
        world.t = float(state.get("t") or 0.0)
        world.steps = int(state.get("steps") or 0)
        return world

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<World {self.config.width}x{self.config.height} "
            f"agents={len(self.agents)} t={self.t:.2f}>"
        )
