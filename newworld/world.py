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
from .config import WorldConfig
from .daynight import DayCycle
from .environment import Air, Sun, step_body_temperature
from .optics import Scene, Shape
from .physiology import Physiology
from .pixelworld import PixelScene
from .technology import Technology


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
    """

    def __init__(self, config: WorldConfig | None = None, seed: int | None = None) -> None:
        self.config = config if config is not None else WorldConfig()
        self.rng = random.Random(seed)
        self.seed_value = seed
        self.agents: list[Character] = []
        #: visible 2-D objects added manually (natural objects live in pixel_scene)
        self.objects: list[Shape] = []
        #: pixel-atom world: terrain, rocks, trees, bushes, pond
        self.pixel_scene = PixelScene(self.config, self.seed_value or 0)
        #: community/tool layer (gathering, crafting, building)
        self.technology = Technology(self.config, self.pixel_scene)
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
        self.spawn()

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
        """First inhabitant (convenience for the single-agent world)."""
        if not self.agents:
            self.spawn()
        return self.agents[0]

    @property
    def n_agents(self) -> int:
        return len(self.agents)

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
        """Add a new inhabitant and return it."""
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
        )
        character.attach(cfg)
        self._next_id = max(self._next_id, character.agent_id) + 1
        self.agents.append(character)
        return character

    def remove_agent(self, agent_id: int) -> None:
        """Remove an inhabitant from the world (death/emigration)."""
        self.agents = [character for character in self.agents if character.agent_id != agent_id]

    def reset(
        self,
        seed: int | None = None,
        positions: list[float] | None = None,
    ) -> list[tuple[float, ...]]:
        """Reset the world to ``t = 0`` and return the initial observations.

        ``positions`` optionally fixes the x position of every inhabitant
        (length must match the number of inhabitants).  Without it, each
        inhabitant is placed at a random legal location.
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
            self.technology = Technology(self.config, self.pixel_scene)
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
            "environment": self.environment_state(),
            "agents": [
                {
                    **character.to_dict(),
                    "at_left_wall": character.at_left_wall,
                    "at_right_wall": character.at_right_wall,
                }
                for character in self.agents
            ],
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<World {self.config.width}x{self.config.height} "
            f"agents={len(self.agents)} t={self.t:.2f}>"
        )
