"""Simple material economy: inventory, tools and a community site."""

from __future__ import annotations

from dataclasses import dataclass

from . import materials


@dataclass
class Inventory:
    wood: int = 0
    stone: int = 0
    tool: str | None = None

    def add(self, kind: str, amount: int = 1) -> None:
        if kind == "wood":
            self.wood += amount
        elif kind == "stone":
            self.stone += amount


@dataclass
class CommunitySite:
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
    """Gathering, tool making and community construction on the atom grid."""

    def __init__(self, config, scene) -> None:
        self.config = config
        self.scene = scene
        self.community = CommunitySite(x_screen=config.width / 2.0)

    # ------------------------------------------------------------------
    def _hand(self, character) -> tuple[float, float]:
        feet = character.y if character.y is not None else self.config.ground_y
        return character.x + character.facing * 40.0, feet - 40.0

    def nearest_resource(self, character, kind: str):
        """Nearest tree (wood) or rock (stone) object, in screen coordinates."""
        best = None
        best_distance = 1e9
        for obj in self.scene.objects:
            if kind == "wood" and obj.kind != "tree":
                continue
            if kind == "stone" and obj.kind != "rock":
                continue
            center = (obj.x0 + obj.x1) / 2.0 * materials.ATOM_SIZE
            distance = abs(center - character.x)
            if distance < best_distance:
                best_distance = distance
                best = (center, obj)
        return best

    def harvest(self, character, kind: str) -> bool:
        """Remove a wood/stone atom near the hand and put it in the inventory."""
        hand_x, hand_y = self._hand(character)
        cx = int(hand_x / materials.ATOM_SIZE)
        cy = int(hand_y / materials.ATOM_SIZE)
        target = materials.MATERIAL_INDEX["wood" if kind == "wood" else "stone"]
        for radius in range(1, 9):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x, y = cx + dx, cy + dy
                    if not (0 <= x < self.scene.aw and 0 <= y < self.scene.ah):
                        continue
                    if int(self.scene.grid[y, x]) != target:
                        continue
                    self.scene.clear_atom(x, y)
                    self.scene.refresh_caches()
                    amount = 2 if character.inventory.tool else 1
                    character.inventory.add(kind, amount)
                    return True
        return False

    def near_community(self, character) -> bool:
        return abs(character.x - self.community.x_screen) < 90.0

    def craft(self, character) -> bool:
        inventory = character.inventory
        if inventory.tool or inventory.wood < 2 or inventory.stone < 1:
            return False
        inventory.wood -= 2
        inventory.stone -= 1
        inventory.tool = "axe"
        return True

    def contribute(self, character) -> bool:
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

    def summary_zh(self) -> str:
        return self.community.summary_zh()
