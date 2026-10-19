"""Atomic destruction / growth and the soil-water cycle.

Everything happens on the same pixel-atom grid as the rest of the world:

* **destruction** – soft atoms (grass, leaves, berries) can be torn off; they
  become falling debris and disappear when they hit the ground;
* **growth** – berries regrow on their bushes, grass grows on wet soil, driven
  by sunlight and soil moisture;
* **soil-water cycle** – the sun evaporates the pond surface, rain puts water
  back, soil moisture diffuses and feeds plant growth, wet soil is darker.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from . import materials

SOFT = {"grass", "leaf", "berry"}
HARD = {"stone", "wood", "soil", "water"}


@dataclass
class Debris:
    """A detached atom falling under gravity (visual + physics)."""

    x: float
    y: float
    vx: float
    vy: float
    material: str
    life: float = 4.0


class Ecology:
    def __init__(self, scene) -> None:
        self.rng = random.Random(2024)
        self.aw = scene.aw
        self.ah = scene.ah
        self.surface = scene.surface
        self.ground_row = scene.ground_row
        self.moisture = np.zeros((self.ah, self.aw), dtype=np.float32)
        self._init_moisture(scene)
        self.humidity = 0.42
        self.rain = 0.0
        self.rain_target = 0.0
        self.rain_timer = self.rng.uniform(20.0, 45.0)
        self.debris: list[Debris] = []
        self.destroyed = 0
        self.grown = 0
        self._accum = 0.0
        self._initial_water = int((scene.grid == materials.MATERIAL_INDEX["water"]).sum())
        self.water_level = 1.0

    # ------------------------------------------------------------------
    def _init_moisture(self, scene) -> None:
        mask = (scene.grid == materials.MATERIAL_INDEX["soil"]) | (
            scene.grid == materials.MATERIAL_INDEX["grass"]
        )
        self.moisture[mask] = 0.38
        left, right = scene.pond
        for x in range(max(0, left - 10), min(self.aw, right + 11)):
            column = self.moisture[:, x]
            column[mask[:, x]] = np.minimum(1.0, column[mask[:, x]] + 0.35)
        # deeper soil is a little wetter
        for y in range(self.surface, self.ah):
            depth = (y - self.surface) / max(1, self.ah - self.surface)
            self.moisture[y, :] = np.minimum(1.0, self.moisture[y, :] + 0.15 * depth)

    # ------------------------------------------------------------------
    def update(self, dt: float, scene, day, hour: float) -> None:
        # debris always falls (and fades when it hits the ground)
        remaining = []
        for particle in self.debris:
            particle.vy += 9.8 * 5.0 * dt  # atoms/s^2 (atom ~5 cm)
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.life -= dt
            if (
                particle.life > 0.0
                and 0 <= particle.x < self.aw
                and particle.y < self.surface
            ):
                remaining.append(particle)
        self.debris = remaining

        self._accum += dt
        if self._accum < 0.25:
            return
        step = self._accum
        self._accum = 0.0
        daylight = day.daylight(hour)

        # -- weather ----------------------------------------------------
        self.rain_timer -= step
        if self.rain_timer <= 0.0:
            self.rain_target = 0.0 if self.rain_target > 0.1 else 0.85
            self.rain_timer = self.rng.uniform(24.0, 70.0)
        self.rain += (self.rain_target - self.rain) * min(1.0, step * 0.7)
        self.humidity = float(
            np.clip(0.30 + 0.55 * self.rain - 0.10 * daylight + self.rng.uniform(-0.02, 0.02), 0.0, 1.0)
        )

        changed = False
        # -- evaporation from the pond ---------------------------------
        if daylight > 0.12 and self.rng.random() < 0.35 + 0.3 * daylight:
            left, right = scene.pond
            candidates = [x for x in range(left, right + 1) if scene.grid[scene.pond_surface, x] == materials.MATERIAL_INDEX["water"]]
            if candidates:
                x = self.rng.choice(candidates)
                scene.clear_atom(x, scene.pond_surface)
                self.humidity = min(1.0, self.humidity + 0.01)
                changed = True

        # -- rain refills the pond -------------------------------------
        if self.rain > 0.30 and self.rng.random() < 0.55:
            left, right = scene.pond
            candidates = [x for x in range(left, right + 1) if scene.grid[scene.pond_surface, x] == 0]
            if candidates:
                x = self.rng.choice(candidates)
                scene.set_atom(x, scene.pond_surface, "water")
                changed = True

        # -- soil moisture: diffusion, rain, drying --------------------
        mask = (scene.grid == materials.MATERIAL_INDEX["soil"]) | (
            scene.grid == materials.MATERIAL_INDEX["grass"]
        )
        wet = self.moisture * mask
        average = (
            np.roll(wet, 1, axis=1)
            + np.roll(wet, -1, axis=1)
            + np.roll(wet, 1, axis=0)
            + np.roll(wet, -1, axis=0)
        ) / 4.0
        self.moisture = np.clip(
            self.moisture + (average - self.moisture) * 0.02,
            0.0,
            1.0,
        ).astype(np.float32)
        self.moisture *= mask
        surface_slice = self.moisture[self.surface, :]
        surface_slice = np.clip(
            surface_slice + (self.rain * 0.20 - daylight * 0.04) * step,
            0.0,
            1.0,
        )
        self.moisture[self.surface, :] = surface_slice

        # -- growth -----------------------------------------------------
        if self._grow(scene, daylight):
            changed = True
        if changed:
            scene.refresh_caches()
        water_now = int((scene.grid == materials.MATERIAL_INDEX["water"]).sum())
        self.water_level = water_now / max(1, self._initial_water)

    # ------------------------------------------------------------------
    def _grow(self, scene, daylight: float) -> bool:
        changed = False
        # berry bushes regrow their berries on wet, sunlit days.
        #
        # P0 defect fix: the original condition required the slot atom to still be
        # ``leaf``, but eating a berry clears that atom (``pick_berry``), so a bush
        # could only ever produce its initial berries — the documented regrowth
        # rule was unreachable and a finite food supply was eaten to extinction.
        # A slot may regrow when it is leaf (fresh bush) or empty air (picked).
        air = 0
        leaf = materials.MATERIAL_INDEX["leaf"]
        for obj in scene.objects:
            if getattr(obj, "kind", "") != "bush":
                continue
            for slot in getattr(obj, "berry_slots", ()):
                if slot in scene.berries:
                    continue
                x, y = slot
                if scene.grid[y, x] not in (leaf, air):
                    continue
                moisture = float(self.moisture[y, x])
                if moisture > 0.22 and daylight > 0.08 and self.rng.random() < 0.30:
                    if scene.add_berry(x, y):
                        obj.berries.append(slot)
                        self.grown += 1
                        changed = True
        # grass grows taller on wet surface soil
        if self.rng.random() < 0.65:
            x = self.rng.randrange(1, self.aw - 1)
            if scene.grid[self.surface, x] == materials.MATERIAL_INDEX["grass"]:
                height = 0
                while (
                    self.surface - height - 1 >= 0
                    and scene.grid[self.surface - height - 1, x] == materials.MATERIAL_INDEX["grass"]
                ):
                    height += 1
                if height < 3 and self.moisture[self.surface, x] > 0.25:
                    scene.set_atom(x, self.surface - height - 1, "grass")
                    self.grown += 1
                    changed = True
        return changed

    # ------------------------------------------------------------------
    def trample(self, scene, x_screen: float, y_screen: float, force: float = 1.4) -> int:
        """Tear soft atoms around a screen-space point; returns atoms removed."""
        cx = int(x_screen / materials.ATOM_SIZE)
        cy = int(y_screen / materials.ATOM_SIZE)
        removed = 0
        for dy in range(-2, 2):
            for dx in range(-2, 3):
                x, y = cx + dx, cy + dy
                if not (0 <= x < self.aw and 0 <= y < self.ah):
                    continue
                material = materials.MATERIAL_BY_INDEX.get(int(scene.grid[y, x]))
                if material is None:
                    continue
                if material.key not in SOFT or material.hardness > force:
                    continue
                scene.clear_atom(x, y)
                if material.key == "berry" and (x, y) in scene.berries:
                    scene.berries.remove((x, y))
                self.debris.append(
                    Debris(
                        x=float(x),
                        y=float(y),
                        vx=self.rng.uniform(-2.5, 2.5),
                        vy=self.rng.uniform(-7.0, -2.0),
                        material=material.key,
                    )
                )
                removed += 1
        if removed:
            self.destroyed += removed
            scene.refresh_caches()
        return removed

    # ------------------------------------------------------------------
    def summary_zh(self) -> str:
        wet_values = self.moisture[self.moisture > 0.0]
        wet = float(wet_values.mean()) if wet_values.size else 0.0
        return (
            f"土壤湿度 {wet:.2f}  空气湿度 {self.humidity:.2f}  "
            f"降雨 {self.rain:.2f}  水位 {self.water_level:.2f}  "
            f"破坏 {self.destroyed}  生长 {self.grown}"
        )
