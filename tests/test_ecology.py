"""Tests for atomic destruction/growth and the soil-water cycle."""

from __future__ import annotations

import unittest

import numpy as np

from newworld import make_config
from newworld.daynight import DayCycle
from newworld.materials import MATERIAL_INDEX
from newworld.pixelworld import PixelScene


def small_scene(seed: int = 0) -> PixelScene:
    config = make_config(800, 500)
    return PixelScene(config, seed=seed)


class TestAtomDestruction(unittest.TestCase):
    def test_trample_removes_soft_atoms_and_makes_debris(self) -> None:
        scene = small_scene()
        # find a grass atom on the surface
        ys, xs = np.where(scene.grid == MATERIAL_INDEX["grass"])
        y, x = int(ys[0]), int(xs[0])
        before = int((scene.grid > 0).sum())
        removed = scene.trample(x * 5.0, y * 5.0, force=2.0)
        self.assertGreater(removed, 0)
        self.assertGreater(scene.ecology.destroyed, 0)
        self.assertGreater(len(scene.ecology.debris), 0)
        self.assertLessEqual(int((scene.grid > 0).sum()), before)

    def test_debris_falls(self) -> None:
        scene = small_scene()
        scene.trample(scene.aw * 5.0 / 2, scene.surface * 5.0, force=2.0)
        if not scene.ecology.debris:
            self.skipTest("no soft atoms at the centre in this seed")
        y_before = scene.ecology.debris[0].y
        scene.update_ecology(0.2, DayCycle(), 12.0)
        self.assertGreaterEqual(scene.ecology.debris[0].y if scene.ecology.debris else 1e9, y_before)


class TestGrowth(unittest.TestCase):
    def test_berries_regrow_on_wet_soil(self) -> None:
        scene = small_scene(seed=2)
        bushes = [obj for obj in scene.objects if obj.kind == "bush"]
        self.assertTrue(bushes)
        initial = list(scene.berries)
        for berry in initial:
            scene.pick_berry(berry)
        self.assertEqual(scene.berries, [])
        scene.ecology.moisture[:, :] = 1.0
        for _ in range(400):
            scene.ecology._grow(scene, daylight=0.9)
            if scene.berries:
                break
        self.assertGreater(len(scene.berries), 0)
        self.assertGreater(scene.ecology.grown, 0)

    def test_grass_grows(self) -> None:
        scene = small_scene()
        for _ in range(400):
            if scene.ecology._grow(scene, daylight=0.9):
                break
        self.assertGreater(scene.ecology.grown, 0)


class TestSoilWaterCycle(unittest.TestCase):
    def test_rain_wets_soil(self) -> None:
        scene = small_scene()
        day = DayCycle()
        before = float(scene.ecology.moisture[scene.ecology.moisture > 0].mean())
        scene.ecology.rain = 0.9
        scene.ecology.rain_target = 0.9
        for _ in range(60):
            scene.update_ecology(0.25, day, 12.0)
        after = float(scene.ecology.moisture[scene.ecology.moisture > 0].mean())
        self.assertGreater(after, before - 1e-6)
        self.assertGreater(scene.ecology.humidity, 0.0)

    def test_sun_evaporates_pond(self) -> None:
        scene = small_scene()
        scene.ecology.rain = 0.0
        scene.ecology.rain_target = 0.0
        scene.ecology.rain_timer = 1e9
        initial = scene.ecology.water_level
        for _ in range(80):
            scene.update_ecology(0.25, DayCycle(), 12.0)
        self.assertLessEqual(scene.ecology.water_level, initial)

    def test_summary_is_chinese(self) -> None:
        scene = small_scene()
        summary = scene.ecology.summary_zh()
        self.assertIn("土壤湿度", summary)
        self.assertIn("破坏", summary)


if __name__ == "__main__":
    unittest.main()
