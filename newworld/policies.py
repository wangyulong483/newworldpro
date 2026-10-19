"""Tiny scripted policies used by tests / RL experiments.

**The interactive application does not use these policies.**  The inhabitant is
free and chooses his own actions in :mod:`newworld.autonomy` (``FreeWill``);
these deterministic policies exist for unit tests, for exercising the
environment and as simple baselines for future RL training scripts.

Policies are callables ``policy(world) -> action``.
"""

from __future__ import annotations

import random

from .character import LEFT, RIGHT, STAY, Character
from .world import World


class RandomPolicy:
    """Uniformly random actions."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def reset(self, world: World | None = None) -> None:
        return None

    def __call__(self, world: World) -> int:
        return self.rng.choice((LEFT, STAY, RIGHT))


class WanderPolicy:
    """Smooth random wandering that respects the world bounds.

    The inhabitant keeps a direction for a random while, sometimes pauses, and
    turns around when it reaches a wall.  This produces the full left/right
    range of motion, which is what the demo recordings show.
    """

    def __init__(
        self,
        seed: int | None = None,
        min_hold: float = 0.35,
        max_hold: float = 1.4,
        stand_probability: float = 0.12,
    ) -> None:
        self.rng = random.Random(seed)
        self.min_hold = min_hold
        self.max_hold = max_hold
        self.stand_probability = stand_probability
        self.reset()

    def reset(self, world: World | None = None) -> None:
        self.direction = self.rng.choice((LEFT, RIGHT))
        self.stand_ticks = 0
        self.remaining_ticks = 0

    def __call__(self, world: World) -> int:
        character: Character = world.agent
        # bounce off the screen edges: this is the "invisible wall" behaviour
        if character.at_left_wall and self.direction == LEFT:
            self.direction = RIGHT
        elif character.at_right_wall and self.direction == RIGHT:
            self.direction = LEFT

        if self.remaining_ticks <= 0:
            if self.rng.random() < self.stand_probability:
                self.stand_ticks = self.rng.randint(10, 35)
            hold_seconds = self.rng.uniform(self.min_hold, self.max_hold)
            self.remaining_ticks = max(1, int(hold_seconds / world.config.fixed_dt))
            self.direction = self.rng.choice((LEFT, RIGHT))

        if self.stand_ticks > 0:
            self.stand_ticks -= 1
            self.remaining_ticks -= 1
            return STAY
        self.remaining_ticks -= 1
        return self.direction


class ConstantPolicy:
    """Hold one action forever (useful for boundary tests)."""

    def __init__(self, action: int) -> None:
        self.action = int(action)

    def reset(self, world: World | None = None) -> None:
        return None

    def __call__(self, world: World) -> int:
        return self.action


POLICIES = {
    "wander": WanderPolicy,
    "random": RandomPolicy,
    "left": lambda seed=None: ConstantPolicy(LEFT),
    "right": lambda seed=None: ConstantPolicy(RIGHT),
    "stand": lambda seed=None: ConstantPolicy(STAY),
}


def make_policy(name: str, seed: int | None = None):
    """Build a policy by name (``wander``, ``random``, ``left``, ``right``, ``stand``)."""
    try:
        factory = POLICIES[name]
    except KeyError:
        raise ValueError(
            f"unknown policy {name!r}, available: {', '.join(sorted(POLICIES))}"
        ) from None
    return factory(seed)
