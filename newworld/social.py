"""Social/technical skills: gathering, tool making, building the community.

These are the *skills* an agent can execute.  They are written as a learnable
teacher policy (``SocialLife``) and are used to collect training data for the
agents' neural behaviour; the neural network decides when to explore/forage/etc.,
while the social layer turns community needs into actions.
"""

from __future__ import annotations

from dataclasses import dataclass

from .behavior import Decision


class SocialLife:
    """Hand-coded teacher for tool use and community construction."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def decide(self, world, character, dt: float) -> Decision | None:
        tech = getattr(world, "technology", None)
        if tech is None:
            return None
        community = tech.community
        inventory = character.inventory

        # -- survival gate (P0 defect fix) ------------------------------
        # The original teacher returned a mate-seeking/standing decision for every
        # adult once the community was built and *never* returned None, which
        # silenced the neural policy and starved the whole population
        # (measured: 8/8 deaths by t=537s in the P0 baseline).  Survival always
        # outranks social/technical work: hand control back to the planner /
        # neural policy while the individual is hungry, thirsty or exhausted.
        physiology = character.physiology
        if physiology.energy < 35.0 or physiology.hydration < 30.0:
            return None
        if physiology.energy < 45.0 and inventory.wood < 2:
            return None

        # pregnant mothers stay near the community and rest
        if character.pregnant:
            character.behavior_state = "rest"
            if abs(character.x - community.x_screen) > 80.0:
                return Decision(1 if community.x_screen > character.x else -1, False, False, None)
            return Decision(0, False, False, None)

        # community built: adults seek the opposite sex (pair bonding / reproduction)
        if community.built:
            if character.age < 70.0:
                return None
            mates = [
                other
                for other in world.agents
                if other.alive
                and other is not character
                and other.age >= 70.0
                and (other.sex != character.sex or character.pregnant)
            ]
            if mates:
                mate = min(mates, key=lambda other: abs(other.x - character.x))
                distance = mate.x - character.x
                if abs(distance) > 55.0:
                    direction = 1 if distance > 0 else -1
                    character.facing = direction
                    character.behavior_state = "social"
                    return Decision(direction, False, False, None)
                character.facing = 1 if distance >= 0 else -1
                character.behavior_state = "social"
                return Decision(0, False, False, None)
            return None

        # if we carry enough material, bring it to the community and build
        if inventory.wood >= 2 and inventory.stone >= 1:
            if tech.near_community(character):
                if inventory.tool is None:
                    tech.craft(character)
                    character.behavior_state = "craft"
                    return Decision(0, False, False, None)
                tech.contribute(character)
                character.behavior_state = "build"
                return Decision(0, False, False, None)
            character.behavior_state = "forage"
            return Decision(1 if community.x_screen > character.x else -1, False, False, None)

        # otherwise gather the missing material from nature
        wanted = "wood" if inventory.wood < 2 else "stone"
        target = tech.nearest_resource(character, wanted)
        if target is None:
            return None
        target_x, _ = target
        character.behavior_state = "gather"
        if abs(target_x - character.x) > 14.0:
            direction = 1 if target_x > character.x else -1
            character.facing = direction
            return Decision(direction, False, False, None)
        character.facing = 1 if target_x >= character.x else -1
        tech.harvest(character, wanted)
        return Decision(0, False, False, None)
