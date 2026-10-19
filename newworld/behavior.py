"""Free behaviour driven by the creature's own physiology.

Nothing here is controlled from outside: the creature chooses what to do from
internal homeostatic drives (energy, hydration, sleep pressure), the solar clock
and what it perceives around it.  The resulting state machine is:

``explore`` → ``forage`` → ``eat`` / ``drink`` / ``sleep`` / ``rest``

* hungry + berries → walk to a berry bush, reach (or jump for) the berry, eat;
* thirsty + pond → walk to the water, drink;
* sleepy at night → lie down and sleep until dawn;
* otherwise → wander, look around, jump and talk to itself.

Interactions use the body's hand position against the pixel-atom scene, so the
hands really need to reach the berry/water.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .materials import ATOM_SIZE

#: approximate hand position above the feet and reach in front (screen px)
HAND_HEIGHT_PX = 40.0
HAND_REACH_PX = 40.0


@dataclass
class Decision:
    """One self-generated action."""

    move: int = 0
    jump: bool = False
    speak: bool = False
    word: str | None = None


class InnerLife:
    """Homeostatic free behaviour state machine."""

    def __init__(self, seed: int | None = 0) -> None:
        self.rng = random.Random(seed)
        self.walk_dir = self.rng.choice((-1, 1))
        self.walk_timer = self.rng.uniform(0.6, 2.4)
        self.jump_timer = self.rng.uniform(2.0, 5.0)
        self.speak_timer = self.rng.uniform(3.0, 7.0)
        self.eat_timer = 0.0
        self.drink_timer = 0.0
        self.target_x = None
        self.state = "explore"

    # ------------------------------------------------------------------
    def decide(self, world, dt: float) -> Decision:
        character = world.agent
        physiology = character.physiology
        scene = world.pixel_scene
        hour = world.day.hour(world.t)
        daylight = world.day.daylight(hour)

        # -- sleep / wake cycle ----------------------------------------
        if physiology.sleeping:
            character.behavior_state = "sleep"
            if daylight > 0.02 and (physiology.sleep_pressure < 45.0 or hour > 6.8):
                physiology.wake()
            else:
                return Decision(0, False, False)
        elif (daylight < 0.06 and physiology.sleep_pressure > 55.0) or hour >= 22.0 or hour < 5.0:
            if physiology.sleep_pressure > 35.0:
                physiology.sleep()
                character.behavior_state = "sleep"
                return Decision(0, False, False)

        feet_y = character.y if character.y is not None else world.config.ground_y
        hand_x = character.x + character.facing * HAND_REACH_PX
        hand_y = feet_y - HAND_HEIGHT_PX

        # -- ambient self-generated actions (jump / talk) ---------------
        self.jump_timer -= dt
        ambient_jump = False
        if self.jump_timer <= 0.0:
            self.jump_timer = self.rng.uniform(2.5, 6.0)
            ambient_jump = character.on_ground
        self.speak_timer -= dt
        ambient_speak = False
        if self.speak_timer <= 0.0:
            self.speak_timer = self.rng.uniform(3.0, 8.0)
            ambient_speak = True

        # -- drink ------------------------------------------------------
        if physiology.hydration < 45.0 and self._near_water(scene, hand_x, hand_y):
            physiology.drink()
            character.behavior_state = "drink"
            self.drink_timer = 0.8
            return Decision(0, False, self.rng.random() < 0.5)

        # -- eat / forage ----------------------------------------------
        berry = self._nearest_berry(scene, character.x, feet_y)
        if berry is not None and physiology.energy < 92.0:
            berry_x = berry[0] * ATOM_SIZE
            berry_y = berry[1] * ATOM_SIZE
            horizontal = abs(berry_x - character.x)
            # low berries are within arm reach; high berries need a jump
            low = feet_y - 115.0 <= berry_y <= feet_y + 5.0
            reachable = horizontal < 42.0 and low
            high = berry_y < feet_y - 115.0
            if reachable:
                scene.pick_berry(berry)
                physiology.eat()
                character.behavior_state = "eat"
                self.eat_timer = 0.9
                return Decision(0, False, ambient_speak or self.rng.random() < 0.6)
            if horizontal > 18.0:
                move = 1 if berry_x > character.x else -1
                self.walk_dir = move
                character.behavior_state = "forage"
                return Decision(move, ambient_jump, False)
            if high and character.on_ground:
                return Decision(0, True, False)
            character.behavior_state = "forage"
            return Decision(0, ambient_jump, ambient_speak)

        # -- idle states ------------------------------------------------
        if self.eat_timer > 0.0:
            self.eat_timer -= dt
            character.behavior_state = "eat"
            return Decision(0, False, False)
        if self.drink_timer > 0.0:
            self.drink_timer -= dt
            character.behavior_state = "drink"
            return Decision(0, False, False)
        if physiology.energy < 15.0:
            character.behavior_state = "rest"
            return Decision(0, False, self.rng.random() < 0.1)

        # -- explore ----------------------------------------------------
        character.behavior_state = "explore"
        self.walk_timer -= dt
        if self.walk_timer <= 0.0:
            self.walk_timer = self.rng.uniform(0.7, 2.6)
            self.walk_dir = self.rng.choice((-1, 1))
            if self.rng.random() < 0.18:
                self.walk_dir = 0
        return Decision(self.walk_dir, ambient_jump, ambient_speak)

    # ------------------------------------------------------------------
    def _nearest_berry(self, scene, x: float, feet_y: float):
        best = None
        best_distance = math.inf
        for berry in scene.berries:
            bx = berry[0] * ATOM_SIZE
            by = berry[1] * ATOM_SIZE
            if by < feet_y - 190.0:  # way above the head
                continue
            distance = abs(bx - x)
            if distance < best_distance:
                best_distance = distance
                best = berry
        return best

    def _near_water(self, scene, hand_x: float, hand_y: float) -> bool:
        atom_x = int(hand_x / ATOM_SIZE)
        atom_y = int(hand_y / ATOM_SIZE)
        return scene.water_near(atom_x, atom_y, radius_atoms=4.0)
