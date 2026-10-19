"""Pixel atoms and materials.

In newworld **one pixel is one atom**: ``ATOM_SIZE`` screen pixels form one
world-atom (the visible pixel art blocks).  Every material is described by its
elemental composition (atom fractions), density, heat capacity, thermal
conductivity, albedo and mechanical state.  Objects and the inhabitant are
built from these atoms, so their mass, heat capacity, appearance and behaviour
all follow from the same material definitions.

Compositions are *approximations* for simulation purposes, chosen to reflect the
real bulk composition of the material (water = H2O, rock = SiO2, wood/grass =
cellulose, air = N2/O2/Ar, flesh = water + organic matter, bone = calcium
phosphate + collagen).
"""

from __future__ import annotations

from dataclasses import dataclass

#: screen pixels per atom (the visible pixel-art block)
ATOM_SIZE = 5


@dataclass(frozen=True)
class Material:
    """One kind of matter made of pixel atoms."""

    key: str
    name_zh: str
    #: atom fractions, e.g. (("H", 2 / 3), ("O", 1 / 3)) for water
    elements: tuple[tuple[str, float], ...]
    density_kg_m3: float
    specific_heat: float  # J/(kg K)
    thermal_conductivity: float  # W/(m K)
    albedo: tuple[int, int, int]
    hardness: float  # arbitrary 0..10 (stone hard, grass soft)
    phase: str  # "solid" | "liquid" | "gas"
    opacity: float = 1.0
    edible: bool = False
    drinkable: bool = False

    @property
    def formula(self) -> str:
        parts = []
        for symbol, fraction in self.elements:
            parts.append(f"{symbol}{fraction:.2f}")
        return " ".join(parts)

    @property
    def elements_text(self) -> str:
        return "  ".join(
            f"{symbol} {fraction * 100:.0f}%" for symbol, fraction in self.elements
        )

    @property
    def is_solid(self) -> bool:
        return self.phase == "solid"

    @property
    def is_fluid(self) -> bool:
        return self.phase in ("liquid", "gas")

    @property
    def colour(self) -> tuple[int, int, int]:
        return self.albedo


MATERIALS: dict[str, Material] = {
    "air": Material(
        "air", "空气",
        (("N", 0.78), ("O", 0.21), ("Ar", 0.01)),
        1.225, 1005.0, 0.026, (205, 228, 242), 0.0, "gas", opacity=0.0,
    ),
    "water": Material(
        "water", "水",
        (("H", 2 / 3), ("O", 1 / 3)),
        1000.0, 4186.0, 0.60, (48, 96, 152), 0.0, "liquid",
        drinkable=True,
    ),
    "soil": Material(
        "soil", "土壤",
        (("Si", 0.30), ("O", 0.55), ("C", 0.12), ("H", 0.03)),
        1300.0, 840.0, 0.30, (108, 78, 52), 2.0, "solid",
    ),
    "stone": Material(
        "stone", "石头",
        (("Si", 1 / 3), ("O", 2 / 3)),
        2700.0, 790.0, 2.50, (126, 126, 130), 8.0, "solid",
    ),
    "grass": Material(
        "grass", "草",
        (("C", 0.286), ("H", 0.476), ("O", 0.238)),
        700.0, 1300.0, 0.10, (86, 148, 68), 1.0, "solid",
    ),
    "wood": Material(
        "wood", "木头",
        (("C", 0.35), ("H", 0.40), ("O", 0.25)),
        600.0, 1700.0, 0.16, (110, 74, 48), 5.0, "solid",
    ),
    "leaf": Material(
        "leaf", "树叶",
        (("C", 0.30), ("H", 0.44), ("O", 0.26)),
        260.0, 1600.0, 0.08, (66, 138, 62), 0.5, "solid",
    ),
    "berry": Material(
        "berry", "浆果",
        (("C", 0.20), ("H", 0.48), ("O", 0.32)),
        850.0, 3000.0, 0.20, (172, 48, 66), 0.4, "solid", edible=True,
    ),
    "flesh": Material(
        "flesh", "肉体",
        (("H", 0.50), ("O", 0.30), ("C", 0.17), ("N", 0.03)),
        1000.0, 3500.0, 0.45, (226, 126, 81), 0.8, "solid",
    ),
    "bone": Material(
        "bone", "骨骼",
        (("Ca", 0.15), ("P", 0.10), ("O", 0.45), ("H", 0.25), ("C", 0.05)),
        1900.0, 1300.0, 0.30, (222, 210, 186), 7.0, "solid",
    ),
}

#: short lookup used all over the code
M = MATERIALS

#: material id <-> material (0 = air / empty)
MATERIAL_LIST: tuple[Material, ...] = tuple(MATERIALS.values())
MATERIAL_INDEX: dict[str, int] = {material.key: index for index, material in enumerate(MATERIAL_LIST)}
MATERIAL_BY_INDEX: dict[int, Material] = {index: material for index, material in enumerate(MATERIAL_LIST)}


def material(key: str) -> Material:
    return MATERIALS[key]


def material_name_zh(key: str) -> str:
    return MATERIALS[key].name_zh


