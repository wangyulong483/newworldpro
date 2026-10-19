"""Material economy: inventory, recipes, tools, public facilities.

The original module hard-coded exactly one recipe (2 wood + 1 stone -> axe) and
one public site (the community in the middle of the screen).  This version keeps
those legacy entry points byte-compatible (``craft`` / ``contribute`` still do
exactly what they did) and adds a **data-driven** layer on top:

* :class:`Inventory` now carries ``fiber`` / ``cord`` / ``basket`` and a carry
  capacity (a basket raises it) so "more technology" really changes movement and
  gathering economics;
* :meth:`Technology.execute_recipe` is the **only** legal way to turn materials
  into technology.  It consumes the inputs and applies the outputs regardless of
  who asked (planner, teacher or knowledge experiment), so the world simulator
  stays the single authority — a planner cannot "declare" a new tool into
  existence;
* :meth:`Technology.simulate_recipe` is the validation path used by
  :mod:`newworld.knowledge`: an attempt may *fail* (``recipe.difficulty``) while
  still consuming materials, which is what makes discovery a real experiment
  instead of a scripted unlock;
* :meth:`Technology.apply_facility_effects` gives built facilities real,
  measurable effects (warmth, rest recovery, water access, craft reliability).

Recipes are duck-typed (``recipe_id``, ``inputs``, ``outputs``, ``requires_tool``,
``difficulty``, ...) so this module does **not** import :mod:`newworld.knowledge`
(no circular imports).  The catalogue lives in ``knowledge.RECIPE_CATALOG``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import materials

#: outputs that end up in the raw-material inventory
INVENTORY_OUTPUTS = frozenset({"wood", "stone", "fiber", "cord"})
#: outputs that are carried equipment
TOOL_OUTPUTS = frozenset({"axe", "pickaxe", "basket", "pot"})
#: outputs that are public structures (created by the civilization layer)
FACILITY_OUTPUTS = frozenset({"campfire", "hut", "well", "workshop", "granary"})

#: facility kind -> (radius in px, short Chinese name)
FACILITY_RADIUS = {
    "campfire": 160.0,
    # a granary feeds everyone who lives around it: the first technology that
    # lets a group exceed the natural carrying capacity of the berry bushes
    "granary": 240.0,
    "hut": 200.0,
    "well": 90.0,
    "workshop": 160.0,
}


@dataclass
class Inventory:
    """What an inhabitant carries.

    ``capacity()`` limits raw materials (``wood/stone/fiber/cord``); equipment
    and tools do not count.  A basket (technology) raises the capacity, which is
    the first mechanism where a discovered technique changes future behaviour
    rather than just a number on screen.
    """

    wood: int = 0
    stone: int = 0
    fiber: int = 0
    cord: int = 0
    tool: str | None = None
    basket: int = 0
    capacity_base: int = 8

    # ------------------------------------------------------------------
    def capacity(self) -> int:
        return self.capacity_base + 4 * self.basket

    def load(self) -> int:
        return self.wood + self.stone + self.fiber + self.cord

    def free(self) -> int:
        return max(0, self.capacity() - self.load())

    def add(self, kind: str, amount: int = 1) -> bool:
        """Add raw material if there is room; return whether it fitted."""
        if kind not in INVENTORY_OUTPUTS:
            return False
        amount = int(amount)
        if amount <= 0:
            return True
        if self.load() + amount > self.capacity():
            return False
        setattr(self, kind, int(getattr(self, kind)) + amount)
        return True

    def take(self, kind: str, amount: int = 1) -> bool:
        """Remove raw material if available; return whether it was there."""
        if kind not in INVENTORY_OUTPUTS:
            return False
        current = int(getattr(self, kind, 0))
        if current < int(amount):
            return False
        setattr(self, kind, current - int(amount))
        return True

    def has(self, kind: str, amount: int = 1) -> bool:
        return int(getattr(self, kind, 0)) >= int(amount)

    def to_dict(self) -> dict[str, object]:
        return {
            "wood": self.wood,
            "stone": self.stone,
            "fiber": self.fiber,
            "cord": self.cord,
            "tool": self.tool,
            "basket": self.basket,
            "capacity": self.capacity(),
        }


@dataclass
class CommunitySite:
    """Legacy public site (kept for backward compatibility and tests)."""

    x_screen: float
    wood: int = 0
    stone: int = 0
    structures: int = 0
    campfire: bool = False

    @property
    def built(self) -> bool:
        return self.structures >= 1

    def need(self) -> str:
        if self.wood < 4:
            return "wood"
        if self.stone < 2:
            return "stone"
        return "build"

    def summary_zh(self) -> str:
        return (
            f"社区 木{self.wood}/4 石{self.stone}/2 结构{self.structures} "
            f"{'篝火' if self.campfire else ''}"
        )


class Technology:
    """Gathering, tool making, recipes and public construction on the atom grid."""

    def __init__(self, config, scene, seed: int = 0) -> None:
        self.config = config
        self.scene = scene
        self.rng = random.Random(seed)
        self.community = CommunitySite(x_screen=config.width / 2.0)
        #: facilities whose construction was validated this step, consumed by
        #: the civilization layer (kind, x_screen, agent_id) -> see Population
        self.pending_facilities: list[tuple[str, float, int]] = []
        self.recipes_executed = 0
        self.recipes_failed = 0

    # ------------------------------------------------------------------
    # gathering
    # ------------------------------------------------------------------
    def _hand(self, character) -> tuple[float, float]:
        feet = character.y if character.y is not None else self.config.ground_y
        return character.x + character.facing * 40.0, feet - 40.0

    def nearest_resource(self, character, kind: str):
        """Nearest tree (wood), rock (stone) or plant (fiber) in screen coords."""
        if kind == "fiber":
            # cordage can be gathered from any plant, including the grass the
            # agent is standing on: the target is the nearest tree/bush, or the
            # agent's own position when only ground grass is available.
            best = None
            best_distance = 1e9
            for obj in self.scene.objects:
                if obj.kind not in ("tree", "bush"):
                    continue
                center = (obj.x0 + obj.x1) / 2.0 * materials.ATOM_SIZE
                distance = abs(center - character.x)
                if distance < best_distance:
                    best_distance = distance
                    best = (center, obj)
            return best if best is not None else (character.x, None)
        lookup = "wood" if kind == "wood" else kind
        best = None
        best_distance = 1e9
        for obj in self.scene.objects:
            if lookup == "wood" and obj.kind != "tree":
                continue
            if lookup == "stone" and obj.kind != "rock":
                continue
            center = (obj.x0 + obj.x1) / 2.0 * materials.ATOM_SIZE
            distance = abs(center - character.x)
            if distance < best_distance:
                best_distance = distance
                best = (center, obj)
        return best

    #: material atom for each harvestable kind (first match wins)
    _HARVEST_MATERIAL = {
        "wood": ("wood",),
        "stone": ("stone",),
        # cordage comes from leaves (trees/bushes) or grass
        "fiber": ("leaf", "grass"),
    }

    def harvest(self, character, kind: str) -> bool:
        """Remove a material atom near the hand and put it in the inventory.

        ``fiber`` gathers leaf atoms (cordage) or — when no leaf is in reach —
        grass, ``wood`` trunk atoms, ``stone`` rock atoms.  Returns ``False``
        when out of reach or when the inventory is full; callers must treat that
        as a real failure (the P1 planner does).
        """
        if kind not in self._HARVEST_MATERIAL:
            return False
        hand_x, hand_y = self._hand(character)
        cx = int(hand_x / materials.ATOM_SIZE)
        cy = int(hand_y / materials.ATOM_SIZE)
        targets = [materials.MATERIAL_INDEX[key] for key in self._HARVEST_MATERIAL[kind]]
        for radius in range(1, 10):
            best = None
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x, y = cx + dx, cy + dy
                    if not (0 <= x < self.scene.aw and 0 <= y < self.scene.ah):
                        continue
                    if int(self.scene.grid[y, x]) in targets:
                        distance = dx * dx + dy * dy
                        if best is None or distance < best[0]:
                            best = (distance, x, y)
            if best is not None:
                _, x, y = best
                amount = 2 if character.inventory.tool else 1
                if not character.inventory.add(kind, amount):
                    return False  # inventory full: a genuine failure
                self.scene.clear_atom(x, y)
                self.scene.refresh_caches()
                return True
        return False

    def near_community(self, character) -> bool:
        return abs(character.x - self.community.x_screen) < 90.0

    # ------------------------------------------------------------------
    # recipes (data driven; recipes come from knowledge.RECIPE_CATALOG)
    # ------------------------------------------------------------------
    def can_afford(self, character, recipe) -> bool:
        """Whether the inventory has every input of ``recipe``."""
        for kind, amount in dict(recipe.inputs).items():
            if not character.inventory.has(kind, amount):
                return False
        return True

    def has_tool_for(self, character, recipe) -> bool:
        required = getattr(recipe, "requires_tool", None)
        if required is None:
            return True
        return character.inventory.tool == required or (
            required in ("any", "tool") and character.inventory.tool is not None
        )

    def recipe_blocker(self, character, recipe) -> str | None:
        """Return a short machine-readable reason why the recipe cannot run.

        Public facilities may be built wherever the individual stands (the
        structure is a *place*: whoever is within its radius benefits), so there
        is deliberately no "must be at the community site" precondition — the
        original fixed centre site is legacy and kept only for compatibility.
        """
        if not self.has_tool_for(character, recipe):
            return "missing_tool"
        for kind, amount in dict(recipe.inputs).items():
            if not character.inventory.has(kind, amount):
                return "missing_material"
        return None

    def execute_recipe(self, character, recipe) -> tuple[bool, str]:
        """Simulator-authoritative execution: consume inputs, apply outputs.

        This is the *only* way a recipe may change the world.  It is called by
        planners, teachers and knowledge experiments alike, so a neural network
        or planner cannot bypass the rules.
        """
        blocker = self.recipe_blocker(character, recipe)
        if blocker is not None:
            self.recipes_failed += 1
            return False, blocker
        for kind, amount in dict(recipe.inputs).items():
            character.inventory.take(kind, amount)
        outputs = dict(recipe.outputs)
        for kind in (kind for kind in outputs if kind in FACILITY_OUTPUTS):
            # facility materials are consumed; the structure itself is queued and
            # created by the civilization layer (it owns public facilities)
            self.pending_facilities.append((kind, character.x, character.agent_id))
        for kind, amount in outputs.items():
            if kind in INVENTORY_OUTPUTS:
                character.inventory.add(kind, amount)
            elif kind == "axe":
                character.inventory.tool = "axe"
            elif kind == "pickaxe":
                character.inventory.tool = "pickaxe"
            elif kind == "basket":
                character.inventory.basket += int(amount)
            elif kind in ("pot",):
                if character.inventory.tool is None:
                    character.inventory.tool = kind
        self.recipes_executed += 1
        return True, "ok"

    def simulate_recipe(self, character, recipe) -> tuple[bool, str]:
        """Attempt a recipe as an **experiment** (used by KnowledgeStore).

        Differs from :meth:`execute_recipe` in that the attempt can *fail*:
        with probability ``recipe.difficulty`` the materials are spoiled (still
        consumed) and the attempt returns ``(False, "defect")``.  This is what
        makes knowledge discovery an actual experiment with wasted cost instead
        of a guaranteed unlock.
        """
        blocker = self.recipe_blocker(character, recipe)
        if blocker is not None:
            self.recipes_failed += 1
            return False, blocker
        difficulty = float(getattr(recipe, "difficulty", 0.3))
        if self.rng.random() < difficulty:
            # the experiment consumes the materials and teaches nothing usable
            for kind, amount in dict(recipe.inputs).items():
                character.inventory.take(kind, amount)
            self.recipes_failed += 1
            return False, "defect"
        return self.execute_recipe(character, recipe)

    # ------------------------------------------------------------------
    # legacy crafting / community (kept identical to the original behaviour)
    # ------------------------------------------------------------------
    def craft(self, character) -> bool:
        """Legacy recipe: 2 wood + 1 stone -> axe."""
        inventory = character.inventory
        if inventory.tool or inventory.wood < 2 or inventory.stone < 1:
            return False
        inventory.wood -= 2
        inventory.stone -= 1
        inventory.tool = "axe"
        return True

    def contribute(self, character) -> bool:
        """Legacy contribution: 2 wood + 1 stone -> community progress."""
        inventory = character.inventory
        if not self.near_community(character):
            return False
        if inventory.wood < 2 or inventory.stone < 1:
            return False
        inventory.wood -= 2
        inventory.stone -= 1
        community = self.community
        community.wood += 2
        community.stone += 1
        if community.wood >= 4 and community.stone >= 2 and community.structures < 1:
            community.structures = 1
            community.campfire = True
        return True

    def repair(self, character, facility) -> bool:
        """Spend materials to restore a public facility's condition."""
        cost_wood = int(getattr(facility, "maintenance_wood", 0))
        cost_stone = int(getattr(facility, "maintenance_stone", 0))
        inventory = character.inventory
        if inventory.wood < cost_wood or inventory.stone < cost_stone:
            return False
        inventory.wood -= cost_wood
        inventory.stone -= cost_stone
        facility.maintained(character.agent_id, 0.0)
        return True

    # ------------------------------------------------------------------
    # facility effects: technology must visibly change the world
    # ------------------------------------------------------------------
    def apply_facility_effects(self, character, facilities, dt: float) -> dict[str, float]:
        """Apply the effects of all **active** facilities near ``character``.

        Returns a small dict of what happened (for telemetry/tests):
        ``{"warmth": 1, "rest": 1, "water": 1, "workshop": 1}``.

        Energy recovery from several facilities is **capped** (``0.45/s``) so
        that stacking identical structures cannot make an individual immortal;
        the cap is a world rule, not a balancing knob hidden in the planner.
        """
        effects: dict[str, float] = {}
        physiology = character.physiology
        energy_gain = 0.0
        for facility in facilities or ():
            if not getattr(facility, "active", True):
                continue
            radius = FACILITY_RADIUS.get(getattr(facility, "kind", ""), 0.0)
            if radius <= 0.0:
                continue
            if abs(getattr(facility, "x_screen", 0.0) - character.x) > radius:
                continue
            kind = facility.kind
            if kind == "campfire":
                energy_gain += 0.35 * dt
                effects["warmth"] = effects.get("warmth", 0.0) + dt
            elif kind == "hut":
                if physiology.sleeping or character.behavior_state == "rest":
                    energy_gain += 0.30 * dt
                    physiology.sleep_pressure = max(0.0, physiology.sleep_pressure - 0.30 * dt)
                effects["rest"] = effects.get("rest", 0.0) + dt
            elif kind == "well":
                # water access without walking to the pond
                physiology.hydration = min(100.0, physiology.hydration + 1.2 * dt)
                effects["water"] = effects.get("water", 0.0) + dt
            elif kind == "workshop":
                effects["workshop"] = effects.get("workshop", 0.0) + dt
            elif kind == "granary":
                # stored food: slow passive energy recovery for everyone nearby
                energy_gain += 0.22 * dt
                effects["granary"] = effects.get("granary", 0.0) + dt
        if energy_gain > 0.0:
            physiology.energy = min(100.0, physiology.energy + min(energy_gain, 0.45 * dt))
        return effects

    def craft_failure_scale(self, character, facilities) -> float:
        """Workshop halves experiment failure probability."""
        for facility in facilities or ():
            if getattr(facility, "kind", "") != "workshop":
                continue
            if not getattr(facility, "active", True):
                continue
            if abs(facility.x_screen - character.x) <= FACILITY_RADIUS["workshop"]:
                return 0.5
        return 1.0

    # ------------------------------------------------------------------
    def summary_zh(self) -> str:
        return self.community.summary_zh()
