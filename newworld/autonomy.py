"""Free behaviour + mind for the inhabitant.

The creature is **free**: no keyboard control, no reward, no externally imposed
objective.  Its actions come from :class:`newworld.behavior.InnerLife`, a
homeostatic state machine driven by energy, hydration, sleep pressure and the
solar clock.  ``Mind`` couples that behaviour with perception and with the
physical world.
"""

from __future__ import annotations

from .behavior import Decision, InnerLife
from .neural import NeuralInnerLife, NeuralPolicy
from .perception import Perceiver, PerceptionState
from .world import World


class FreeWill(InnerLife):
    """Backwards-compatible name for the free behaviour engine."""

    def act(self, world: World, dt: float) -> int:
        """Return only the horizontal action (legacy helper)."""
        return self.decide(world, dt).move


class Mind:
    """The inhabitant as a whole: free behaviour + perception (all senses)."""

    def __init__(self, world: World, seed: int | None = 0, policy: NeuralPolicy | None = None) -> None:
        self.world = world
        self.neural = policy is not None
        self.will = NeuralInnerLife(policy, seed) if policy is not None else FreeWill(seed)
        self.perceiver = Perceiver(seed)
        self.perception: PerceptionState | None = None
        self.action = 0
        self.last_speech = None

    # ------------------------------------------------------------------
    def step(self, dt: float) -> int:
        """Live one step: decide, move/jump/interact, maybe speak, perceive."""
        if self.neural:
            decision = self.will.decide(self.world, dt, perception=self.perception)  # type: ignore[call-arg]
        else:
            decision = self.will.decide(self.world, dt)
        self.action = decision.move
        self.world.step(actions=decision, dt=dt)
        if decision.speak:
            self.last_speech = self.world.speak(self.world.agent, decision.word)
        self.perception = self.perceiver.perceive(self.world, dt)
        return self.action

    # ------------------------------------------------------------------
    def summary_lines(self) -> list[str]:
        return self.perceiver.summary_lines()
