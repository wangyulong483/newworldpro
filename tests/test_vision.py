"""视觉管线测试：光学/眼睛/视网膜/感知。

Tests for the inhabitant's visual perception (optics, eye, retina, gaze)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from newworld import make_config
from newworld.autonomy import Mind
from newworld.character import RIGHT
from newworld.eye import (
    EyeParameters,
    angular_gaussian_blur,
    blur_sigma_deg,
    circle_of_confusion_deg,
    diffraction_limit_arcmin,
    pupil_diameter_mm,
)
from newworld.optics import Box, Disc, Panorama, Scene
from newworld.perception import GazeController, GazeParameters, VisualSystem, eye_position
from newworld.retina import L_CONE, M_CONE, S_CONE, Retina, RetinaParameters
from newworld.world import World

SMALL = (800, 500)
LUMA = np.array((0.2126, 0.7152, 0.0722))


def small_config():
    return make_config(*SMALL)


# ---------------------------------------------------------------------------
# light transport
# ---------------------------------------------------------------------------
class TestOptics(unittest.TestCase):
    def test_empty_world_is_uniform_white(self) -> None:
        scene = Scene()
        panorama = scene.panorama(100.0, 100.0, 0.0, 160.0, samples_per_deg=20.0)
        self.assertEqual(len(panorama), 160 * 20)
        self.assertTrue(np.allclose(panorama.radiance, 1.0))
        self.assertTrue(np.all(np.isinf(panorama.distance_px)))

    def test_ray_hits_disc(self) -> None:
        scene = Scene(objects=(Disc(cx=100.0, cy=0.0, radius=10.0, albedo=(0.0, 0.0, 0.0)),))
        colour, distance = scene.trace(0.0, 0.0, 0.0)
        self.assertEqual(colour, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(distance, 90.0, places=6)

    def test_ray_misses_disc(self) -> None:
        scene = Scene(objects=(Disc(cx=100.0, cy=100.0, radius=5.0, albedo=(0.0, 0.0, 0.0)),))
        colour, distance = scene.trace(0.0, 0.0, 0.0)
        self.assertEqual(colour, Scene().environment)
        self.assertTrue(math.isinf(distance))

    def test_nearest_object_occludes_far_object(self) -> None:
        near = Box(60.0, -10.0, 70.0, 10.0, albedo=(0.2, 0.2, 0.2))
        far = Box(100.0, -10.0, 110.0, 10.0, albedo=(0.8, 0.8, 0.8))
        scene = Scene(objects=(far, near))
        colour, distance = scene.trace(0.0, 0.0, 0.0)
        self.assertEqual(colour, (0.2, 0.2, 0.2))
        self.assertAlmostEqual(distance, 60.0, places=6)

    def test_panorama_contains_coloured_target(self) -> None:
        red = Disc(cx=100.0, cy=0.0, radius=20.0, albedo=(1.0, 0.0, 0.0))
        scene = Scene(objects=(red,))
        panorama = scene.panorama(0.0, 0.0, 0.0, 90.0, samples_per_deg=10.0)
        luminance = panorama.radiance @ LUMA
        self.assertLess(float(luminance.min()), 0.5)
        self.assertGreater(float(luminance.max()), 0.9)


# ---------------------------------------------------------------------------
# eye optics
# ---------------------------------------------------------------------------
class TestEyeOptics(unittest.TestCase):
    def setUp(self) -> None:
        self.params = EyeParameters()

    def test_pupil_bright_vs_dark(self) -> None:
        bright = pupil_diameter_mm(1.0, self.params)
        dark = pupil_diameter_mm(1e-3, self.params)
        self.assertAlmostEqual(bright, self.params.pupil_min_mm, places=3)
        self.assertAlmostEqual(dark, self.params.pupil_max_mm, places=3)
        self.assertGreater(dark, bright)

    def test_defocus_grows_with_pupil_and_diopters(self) -> None:
        self.assertLess(circle_of_confusion_deg(2.0, 0.0), circle_of_confusion_deg(2.0, 1.0))
        self.assertLess(circle_of_confusion_deg(2.0, 2.0), circle_of_confusion_deg(8.0, 2.0))

    def test_diffraction_limit(self) -> None:
        # 2 mm pupil: Airy first minimum ~ 1.2 arcmin; 8 mm ~ 0.3 arcmin
        self.assertGreater(diffraction_limit_arcmin(2.0), diffraction_limit_arcmin(8.0))
        self.assertAlmostEqual(diffraction_limit_arcmin(2.0), 1.15, delta=0.2)

    def test_blur_sigma_is_positive_and_grows(self) -> None:
        sigma_sharp = blur_sigma_deg(2.0, 0.0)
        sigma_defocus = blur_sigma_deg(6.0, 2.5)
        self.assertGreater(sigma_sharp, 0.0)
        self.assertGreater(sigma_defocus, sigma_sharp)

    def test_angular_blur_softens_an_edge(self) -> None:
        samples = 400
        angles = np.linspace(-10.0, 10.0, samples)
        radiance = np.ones((samples, 3), dtype=np.float32)
        radiance[: samples // 2] = 0.0
        panorama = Panorama(angles, radiance, np.full(samples, np.inf))
        blurred = angular_gaussian_blur(panorama, 1.0)
        centre = blurred.radiance[samples // 2 - 2 : samples // 2 + 2]
        self.assertTrue(np.all(centre > 0.0))
        self.assertTrue(np.all(centre < 1.0))
        self.assertLess(float(blurred.radiance[0, 0]), 0.1)
        self.assertGreater(float(blurred.radiance[-1, 0]), 0.9)


# ---------------------------------------------------------------------------
# retina
# ---------------------------------------------------------------------------
def uniform_panorama(samples: int = 2000) -> Panorama:
    angles = np.linspace(-80.0, 80.0, samples)
    radiance = np.ones((samples, 3), dtype=np.float32)
    return Panorama(angles, radiance, np.full(samples, np.inf))


class TestRetina(unittest.TestCase):
    def setUp(self) -> None:
        self.params = RetinaParameters()
        self.retina = Retina(self.params, np.random.default_rng(0))

    def test_fovea_has_higher_cone_density(self) -> None:
        fovea = np.sum(np.abs(self.retina.cone_ecc) < 1.0)
        periphery = np.sum((np.abs(self.retina.cone_ecc) > 10.0) & (np.abs(self.retina.cone_ecc) < 11.0))
        self.assertGreater(fovea, periphery * 2)

    def test_rod_peak_matches_human_topography(self) -> None:
        # rods peak around 18 deg and are absent from the fovea
        self.assertEqual(int(np.sum(np.abs(self.retina.rod_ecc) < 1.0)), 0)
        near_peak = np.sum(np.abs(self.retina.rod_ecc - 18.0) < 3.0)
        far = np.sum(np.abs(self.retina.rod_ecc - 60.0) < 3.0)
        self.assertGreater(near_peak, far)

    def test_cone_proportions(self) -> None:
        counts = np.bincount(self.retina.cone_types.astype(int), minlength=3)
        total = counts.sum()
        self.assertAlmostEqual(counts[L_CONE] / total, 0.60, delta=0.06)
        self.assertAlmostEqual(counts[M_CONE] / total, 0.30, delta=0.06)
        self.assertAlmostEqual(counts[S_CONE] / total, 0.10, delta=0.05)

    def test_uniform_white_world(self) -> None:
        observation = self.retina.sample(uniform_panorama(), 0.1)
        self.assertAlmostEqual(observation.mean_luminance, 1.0, places=5)
        self.assertAlmostEqual(observation.foveal_luminance, 1.0, places=5)
        self.assertTrue(0.0 < observation.cone_responses.mean() < 1.0)
        # a uniform field has no contrast: ganglion cells report ~zero
        self.assertLess(float(np.abs(observation.ganglion).max()), 1e-5)

    def test_adaptation_changes_response(self) -> None:
        retina = Retina(self.params, np.random.default_rng(1))
        bright = retina.sample(uniform_panorama(), dt=0.5)
        dark = uniform_panorama()
        dark.radiance *= 0.01
        adapted = bright
        for _ in range(60):
            adapted = retina.sample(dark, dt=0.1)
        self.assertGreater(adapted.adaptation, bright.adaptation)
        self.assertGreater(bright.mean_luminance, adapted.mean_luminance)

    def test_feature_vector_size(self) -> None:
        observation = self.retina.sample(uniform_panorama(), 0.1)
        expected = self.params.n_bins * 3 + self.params.n_bins + self.params.n_bins + 3
        self.assertEqual(observation.feature_vector().size, expected)

    def test_blink_blocks_light(self) -> None:
        retina = Retina(self.params, np.random.default_rng(2))
        observation = retina.sample(uniform_panorama(), 0.1, lid_transmission=0.0)
        self.assertAlmostEqual(observation.mean_luminance, 0.0, places=6)


# ---------------------------------------------------------------------------
# eye movements
# ---------------------------------------------------------------------------
class TestGaze(unittest.TestCase):
    def test_saccade_and_blink_statistics(self) -> None:
        import random

        params = GazeParameters()
        controller = GazeController(params, random.Random(0))
        dt = 1.0 / 240.0
        duration = 60.0
        offsets = []
        for _ in range(int(duration / dt)):
            state = controller.update(dt)
            offsets.append(state.offset_deg)
            self.assertLessEqual(abs(state.offset_deg), params.max_rotation_deg + 1.0)
        saccades_per_second = controller.state.saccade_count / duration
        self.assertGreaterEqual(saccades_per_second, 1.5)
        self.assertLessEqual(saccades_per_second, 7.0)
        self.assertGreater(controller.state.blink_count, 2)
        self.assertGreater(len(set(offsets)), 10)


# ---------------------------------------------------------------------------
# full visual system + autonomy
# ---------------------------------------------------------------------------
class TestVisualSystem(unittest.TestCase):
    def test_perceive_pixel_world(self) -> None:
        config = small_config()
        world = World(config, seed=0)
        system = VisualSystem(seed=0)
        perception = system.perceive(world.scene(), world.agent, config, 1.0 / 60.0)
        # the pixel world has objects now, so the visual field is not uniform
        self.assertGreaterEqual(perception.mean_luminance, 0.0)
        self.assertLessEqual(perception.mean_luminance, 1.0)
        self.assertTrue(0.0 <= perception.pupil_mm <= 8.0)
        self.assertTrue(math.isfinite(perception.gaze_world_deg))
        self.assertIn(perception.mode, ("fixation", "saccade"))
        self.assertGreater(perception.retina.feature_vector().size, 0)

    def test_eye_sits_above_the_ground(self) -> None:
        config = small_config()
        world = World(config, seed=0)
        eye_x, eye_y = eye_position(world.agent, config)
        self.assertAlmostEqual(eye_x, world.agent.x)
        self.assertLess(eye_y, config.ground_y)
        self.assertGreater(eye_y, config.ground_y - config.sprite_h)

    def test_world_objects_are_seen(self) -> None:
        config = small_config()
        world = World(config, seed=0)
        world.reset(positions=[config.min_x + 50.0])
        world.agent.facing = 1
        world.add_object(
            Disc(
                cx=world.agent.x + 400.0,
                cy=config.ground_y - 60.0,
                radius=40.0,
                albedo=(0.0, 0.0, 0.0),
            )
        )
        system = VisualSystem(seed=0)
        perception = system.perceive(world.scene(), world.agent, config, 1.0 / 60.0)
        self.assertGreaterEqual(float(perception.mean_luminance), 0.0)
        self.assertLessEqual(float(perception.mean_luminance), 1.0)


class TestAutonomy(unittest.TestCase):
    def test_free_will_acts_without_input(self) -> None:
        config = small_config()
        world = World(config, seed=0)
        mind = Mind(world, seed=3)
        for _ in range(900):
            mind.step(1.0 / 60.0)
        self.assertIsNotNone(mind.perception)
        self.assertGreater(mind.perception.retina.feature_vector().size, 0)
        # he moves / eats / talks / jumps on his own, driven only by physiology
        self.assertGreater(world.agent.distance_walked, 20.0)
        self.assertIn(world.agent.behavior_state, ("explore", "forage", "eat", "drink", "rest", "sleep"))

    def test_mind_is_deterministic_with_seed(self) -> None:
        config = small_config()
        world_a = World(config, seed=5)
        world_b = World(config, seed=5)
        world_b.reset(seed=5)
        mind_a = Mind(world_a, seed=5)
        mind_b = Mind(world_b, seed=5)
        actions_a = [mind_a.step(1.0 / 60.0) for _ in range(300)]
        actions_b = [mind_b.step(1.0 / 60.0) for _ in range(300)]
        self.assertEqual(actions_a, actions_b)
        self.assertAlmostEqual(world_a.agent.x, world_b.agent.x, places=6)

    def test_free_will_stays_on_screen(self) -> None:
        config = small_config()
        world = World(config, seed=0)
        mind = Mind(world, seed=1)
        for _ in range(2000):
            mind.step(1.0 / 60.0)
            self.assertGreaterEqual(world.agent.x, config.min_x - 1e-9)
            self.assertLessEqual(world.agent.x, config.max_x + 1e-9)


if __name__ == "__main__":
    unittest.main()
