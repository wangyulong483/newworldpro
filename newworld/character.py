"""The inhabitant of newworld.

The inhabitant is the creature from the user's blueprint: an orange pixel-art
body with a blue thought bubble.  It lives near the bottom of the screen, walks
left/right, **jumps under gravity** (mechanics), has a **body temperature**
(thermodynamics) and **speaks** (acoustics).  Its motion is clipped to the
screen so it can never leave the world.

Actions are integers for the RL/environment interface::

    -1  walk left
     0  stand still (friction decelerates the creature)
    +1  walk right

``update(..., jump=True)`` additionally launches a ballistic jump when the
creature is standing on the ground.  The free (§:mod:`newworld.autonomy`)
creature uses both.

The class is deliberately free of rendering and numpy so it can be copied,
pickled, mutated by evolution experiments and unit-tested cheaply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import physics
from .config import WorldConfig
from .materials import ATOM_SIZE
from .physiology import Physiology
from .technology import Inventory

LEFT = -1
STAY = 0
RIGHT = 1
ACTIONS = (LEFT, STAY, RIGHT)


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


@dataclass
class Character:
    """One inhabitant of the world."""

    x: float
    facing: int = 1
    vx: float = 0.0
    phase: float = 0.0
    agent_id: int = 0
    # bookkeeping (useful for research logs)
    distance_walked: float = 0.0
    steps: int = 0
    # -- mechanics -----------------------------------------------------
    #: feet position in world pixels (``None`` until :meth:`attach`); y grows down
    y: float | None = None
    vy: float = 0.0
    on_ground: bool = True
    mass_kg: float = physics.BODY_MASS_KG
    jump_count: int = 0
    # -- thermodynamics -------------------------------------------------
    body_temperature_k: float = physics.HUMAN_BODY_TEMPERATURE_K
    heat_balance: object | None = None
    # -- acoustics ------------------------------------------------------
    utterance: str | None = None
    utterance_until: float = 0.0
    speech_count: int = 0
    # -- inner life ------------------------------------------------
    behavior_state: str = "explore"
    physiology: Physiology = field(default_factory=Physiology)
    # -- society ------------------------------------------------------
    name: str = ""
    sex: str = "male"          # "male" | "female"
    age: float = 0.0           # seconds of life
    generation: int = 0
    alive: bool = True
    pregnant: bool = False
    pregnancy_timer: float = 0.0
    partner_id: int | None = None
    parents: tuple[int, ...] = ()
    body_scale: float = 1.0
    inventory: Inventory = field(default_factory=Inventory)
    social_context: tuple[float, ...] = ()
    # -- civilization layer (P3): identity, history, knowledge, plans -----
    #: direct parent (``parents[0]`` when present); kept explicit for lineage
    parent_id: int | None = None
    #: world time of creation and of death (None while alive)
    created_at: float = 0.0
    died_at: float | None = None
    death_cause: str = ""
    #: knowledge ids this individual has learned (KnowledgeStore ids)
    known_knowledge_ids: list[str] = field(default_factory=list)
    #: skill -> proficiency (0..1), learned procedurally
    skills: dict[str, float] = field(default_factory=dict)
    #: ids of episodic memories kept for this agent (MemoryStore)
    memory_ids: list[int] = field(default_factory=list)
    #: current goal / plan handles (GoalManager ids)
    current_goal_id: int | None = None
    current_plan_id: int | None = None
    #: contributions to public facilities / cooperation (kind -> amount)
    contributions: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = float(self.x)
        self.facing = 1 if self.facing >= 0 else -1
        self.vx = float(self.vx)
        self.phase = float(self.phase)
        # geometry, attached by the world when the agent is created
        self._sprite_w: int = 0
        self._sprite_h: int = 0
        self._ground: float = 0.0
        # screen-edge limits (left / right), refreshed by :meth:`attach`
        self._min_x: float = -1.0
        self._max_x: float = -1.0

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def attach(self, config: WorldConfig) -> None:
        """Attach sprite geometry / screen-edge limits from the configuration."""
        self._sprite_w = config.sprite_w
        self._sprite_h = config.sprite_h
        self._ground = float(config.ground_y)
        self._min_x = config.min_x
        self._max_x = config.max_x
        if self.y is None:
            self.y = self._ground

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def update(
        self,
        dt: float,
        action: int,
        config: WorldConfig,
        *,
        jump: bool = False,
    ) -> None:
        """Advance the inhabitant by ``dt`` seconds (mechanics with gravity)."""
        if dt <= 0.0:
            return
        if self.y is None:
            self.attach(config)
        action = int(action)
        action = RIGHT if action > 0 else LEFT if action < 0 else STAY

        # -- horizontal locomotion -------------------------------------
        if action != STAY:
            self.vx += action * config.accel * dt
            self.vx = clamp(self.vx, -config.speed, config.speed)
            self.facing = action
            self.phase += dt * math.tau * 1.75 * (abs(self.vx) / config.speed)
        elif self.vx != 0.0:
            drop = config.friction * dt
            if abs(self.vx) <= drop:
                self.vx = 0.0
            else:
                self.vx -= math.copysign(drop, self.vx)

        # air drag (quadratic): a = -0.5 rho Cd A v|v| / m
        drag = physics.air_drag_force(physics.px_to_m(self.vx)) / max(1e-6, self.mass_kg)
        drag_px = drag * physics.PIXELS_PER_METER
        if abs(self.vx) > 1e-9:
            self.vx -= math.copysign(min(abs(self.vx), abs(drag_px) * dt), self.vx)
        else:
            self.vx = 0.0

        previous_x = self.x
        self.x += self.vx * dt
        if self.x < config.min_x:
            self.x = config.min_x
            self.vx = 0.0
        elif self.x > config.max_x:
            self.x = config.max_x
            self.vx = 0.0

        # -- vertical mechanics: gravity + ballistic jump ---------------
        if jump and self.on_ground:
            self.vy = -physics.JUMP_SPEED_PX
            self.on_ground = False
            self.jump_count += 1
        self.vy += physics.GRAVITY_PX * dt
        self.y += self.vy * dt
        if self.y >= self._ground:
            self.y = self._ground
            self.vy = 0.0
            self.on_ground = True
        elif self.y < self._sprite_h:
            self.y = self._sprite_h
            self.vy = max(self.vy, 0.0)

        self.distance_walked += abs(self.x - previous_x)
        self.steps += 1

    def jump(self, config: WorldConfig) -> bool:
        """Launch a jump if standing on the ground; return whether it happened."""
        if not self.on_ground:
            return False
        self.vy = -physics.JUMP_SPEED_PX
        self.on_ground = False
        self.jump_count += 1
        return True

    # ------------------------------------------------------------------
    # geometry / derived state
    # ------------------------------------------------------------------
    @property
    def width(self) -> int:
        return self._sprite_w

    @property
    def height(self) -> int:
        return self._sprite_h

    @property
    def at_left_wall(self) -> bool:
        return self.x <= self._min_x + 0.5

    @property
    def at_right_wall(self) -> bool:
        return self.x >= self._max_x - 0.5

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Left, top, right, bottom of the sprite in world pixels."""
        feet = self._ground if self.y is None else self.y
        left = self.x - self._sprite_w / 2.0
        top = feet - self._sprite_h
        return left, top, left + self._sprite_w, top + self._sprite_h

    @property
    def body_temperature_c(self) -> float:
        return self.body_temperature_k - 273.15

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def observation(self, config: WorldConfig) -> tuple[float, ...]:
        """Normalised feature vector used for RL / logs (8 values).

        ``(x_norm, vx_norm, facing, dist_left_norm, dist_right_norm,
        height_norm, vy_norm, on_ground)``

        * ``x_norm`` / ``dist_left_norm`` / ``dist_right_norm`` as before
        * ``height_norm`` jump height above the ground, normalised to the
          ballistic peak of a full jump
        * ``vy_norm`` vertical speed normalised by the jump speed
        * ``on_ground`` 1.0 when standing on the ground, else 0.0
        """
        span = config.travel_span
        if span <= 0:
            x_norm = dist_left = dist_right = 0.0
        else:
            x_norm = (self.x - config.min_x) / span
            dist_left = (self.x - config.min_x) / span
            dist_right = (config.max_x - self.x) / span
        feet = self._ground if self.y is None else self.y
        peak_px = max(1.0, physics.jump_height_m() * physics.PIXELS_PER_METER)
        height_norm = clamp((self._ground - feet) / peak_px, 0.0, 1.0)
        return (
            clamp(x_norm, 0.0, 1.0),
            clamp(self.vx / config.speed, -1.0, 1.0),
            float(self.facing),
            dist_left,
            dist_right,
            height_norm,
            clamp(self.vy / physics.JUMP_SPEED_PX, -1.0, 1.0),
            1.0 if self.on_ground else 0.0,
        )

    def bob_offset(self, config: WorldConfig) -> int:
        """Vertical bounce (screen px, negative = upwards) while walking."""
        if abs(self.vx) < 1.0 or not self.on_ground:
            return 0
        amount = min(1.0, abs(self.vx) / config.speed)
        amplitude = config.cell * 0.4
        return -int(round(abs(math.sin(self.phase)) * amplitude * amount))

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        return {
            "agent_id": self.agent_id,
            "x": round(self.x, 3),
            "y": round(self.y if self.y is not None else 0.0, 3),
            "vx": round(self.vx, 3),
            "vy": round(self.vy, 3),
            "facing": self.facing,
            "on_ground": self.on_ground,
            "body_temperature_c": round(self.body_temperature_c, 3),
            "utterance": self.utterance,
            "behavior_state": self.behavior_state,
            "name": self.name,
            "sex": self.sex,
            "age": round(self.age, 2),
            "alive": self.alive,
            "pregnant": self.pregnant,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "created_at": round(self.created_at, 3),
            "died_at": None if self.died_at is None else round(self.died_at, 3),
            "death_cause": self.death_cause,
            "current_goal_id": self.current_goal_id,
            "current_plan_id": self.current_plan_id,
            "known_knowledge": list(self.known_knowledge_ids),
            "skills": {key: round(value, 3) for key, value in self.skills.items()},
            "wood": self.inventory.wood,
            "stone": self.inventory.stone,
            "fiber": getattr(self.inventory, "fiber", 0),
            "cord": getattr(self.inventory, "cord", 0),
            "tool": self.inventory.tool,
            "energy": round(self.physiology.energy, 2),
            "hydration": round(self.physiology.hydration, 2),
            "sleep_pressure": round(self.physiology.sleep_pressure, 2),
            "heart_rate": round(self.physiology.heart_rate, 2),
            "sleeping": self.physiology.sleeping,
            "distance_walked": round(self.distance_walked, 3),
            "steps": self.steps,
        }


# ---------------------------------------------------------------------------
# body geometry helpers
# ---------------------------------------------------------------------------
def eye_position(character: Character, config: WorldConfig) -> tuple[float, float]:
    """World position of the eye (head centre of the pixel body)."""
    feet = config.ground_y if character.y is None else character.y
    return character.x, feet - 14.0 * ATOM_SIZE


def mouth_position(character: Character, config: WorldConfig) -> tuple[float, float]:
    """World position of the mouth (front of the head)."""
    feet = config.ground_y if character.y is None else character.y
    return (
        character.x + character.facing * 2.0 * ATOM_SIZE,
        feet - 10.0 * ATOM_SIZE,
    )
