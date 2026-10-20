"""物理常数与单位换算测试（g、声速、声压级、跳跃高度等）。

Tests for the physical laws: mechanics, acoustics, thermodynamics, EM."""

from __future__ import annotations

import math
import random
import unittest

import numpy as np

from newworld import make_config
from newworld import acoustics, electromagnetism, environment, physics
from newworld.autonomy import Decision, FreeWill, Mind
from newworld.character import Character
from newworld.render import character_layer, render_frame
from newworld.world import World

SMALL = (800, 500)


def small_config():
    return make_config(*SMALL)


# ---------------------------------------------------------------------------
# constants & electromagnetic radiation
# ---------------------------------------------------------------------------
class TestConstants(unittest.TestCase):
    def test_values_match_references(self) -> None:
        self.assertAlmostEqual(physics.STANDARD_GRAVITY, 9.80665, places=6)
        self.assertAlmostEqual(physics.STEFAN_BOLTZMANN, 5.670374419e-8, places=18)
        self.assertAlmostEqual(physics.sound_speed(293.15), 343.2, delta=1.0)
        self.assertAlmostEqual(physics.SOLAR_CONSTANT, 1361.0, places=1)

    def test_unit_system(self) -> None:
        self.assertEqual(physics.PIXELS_PER_METER, 100.0)
        self.assertAlmostEqual(physics.m_to_px(1.0), 100.0)
        self.assertAlmostEqual(physics.px_to_m(250.0), 2.5)


class TestElectromagnetism(unittest.TestCase):
    def test_wien_peak_of_the_sun(self) -> None:
        peak_nm = electromagnetism.wien_peak_wavelength_m(5772.0) * 1e9
        self.assertAlmostEqual(peak_nm, 502.0, delta=5.0)

    def test_planck_spectrum_peaks_near_wien(self) -> None:
        wavelengths = np.linspace(200e-9, 1500e-9, 2000)
        radiance = np.array(
            [electromagnetism.planck_spectral_radiance(w, 5772.0) for w in wavelengths]
        )
        peak = wavelengths[int(np.argmax(radiance))]
        self.assertAlmostEqual(peak * 1e9, 502.0, delta=15.0)

    def test_visible_fraction_of_sunlight(self) -> None:
        fraction = electromagnetism.solar_visible_fraction(5772.0)
        self.assertGreater(fraction, 0.30)
        self.assertLess(fraction, 0.55)

    def test_photon_energy(self) -> None:
        self.assertAlmostEqual(electromagnetism.photon_energy_ev(500e-9), 2.48, delta=0.05)


class TestSun(unittest.TestCase):
    def test_irradiance_geometry(self) -> None:
        sun = environment.Sun(elevation_deg=90.0)
        self.assertAlmostEqual(sun.horizontal_irradiance, physics.SOLAR_CONSTANT, places=3)
        sun30 = environment.Sun(elevation_deg=30.0)
        self.assertAlmostEqual(
            sun30.horizontal_irradiance, physics.SOLAR_CONSTANT * 0.5, delta=1.0
        )

    def test_visible_part_and_direction(self) -> None:
        sun = environment.Sun(elevation_deg=55.0)
        self.assertGreater(sun.visible_irradiance, 300.0)
        self.assertLess(sun.visible_irradiance, sun.horizontal_irradiance)
        self.assertAlmostEqual(sun.peak_wavelength_nm, 502.0, delta=5.0)
        dx, dy = sun.light_direction()
        self.assertAlmostEqual(math.hypot(dx, dy), 1.0, places=6)
        self.assertLess(dy, 0.0)  # sun is above: light travels downwards


# ---------------------------------------------------------------------------
# mechanics: gravity + jumping
# ---------------------------------------------------------------------------
class TestMechanics(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()
        self.character = Character(x=(self.config.min_x + self.config.max_x) / 2)
        self.character.attach(self.config)

    def test_gravity_and_landing(self) -> None:
        self.character.y = self.config.ground_y - 100.0
        self.character.on_ground = False
        self.character.vy = 0.0
        steps = 0
        while not self.character.on_ground and steps < 1000:
            self.character.update(1 / 240, 0, self.config)
            steps += 1
        self.assertTrue(self.character.on_ground)
        self.assertAlmostEqual(self.character.y, self.config.ground_y, places=6)
        fall_time = steps / 240.0
        expected = math.sqrt(2 * physics.px_to_m(100.0) / physics.STANDARD_GRAVITY)
        self.assertAlmostEqual(fall_time, expected, delta=0.1)

    def test_jump_height_matches_ballistic_formula(self) -> None:
        ground = self.config.ground_y
        peak = ground
        for _ in range(240):  # one second at 240 Hz
            self.character.update(1 / 240, 0, self.config, jump=self.character.on_ground)
            peak = min(peak, self.character.y)
        expected_px = physics.jump_height_m() * physics.PIXELS_PER_METER
        self.assertAlmostEqual(ground - peak, expected_px, delta=expected_px * 0.05)

    def test_cannot_double_jump(self) -> None:
        self.character.update(1 / 240, 0, self.config, jump=True)
        self.assertFalse(self.character.on_ground)
        self.character.update(1 / 240, 0, self.config, jump=True)
        self.assertEqual(self.character.jump_count, 1)

    def test_observation_contains_vertical_state(self) -> None:
        observation = self.character.observation(self.config)
        self.assertEqual(len(observation), 8)
        self.assertEqual(observation[7], 1.0)  # on the ground
        self.character.update(1 / 240, 0, self.config, jump=True)
        airborne = self.character.observation(self.config)
        self.assertEqual(airborne[7], 0.0)
        self.assertGreater(airborne[5], 0.0)  # jump height
        self.assertLess(airborne[6], 0.0)  # moving upwards


# ---------------------------------------------------------------------------
# thermodynamics
# ---------------------------------------------------------------------------
class TestThermodynamics(unittest.TestCase):
    def test_sun_warms_the_body(self) -> None:
        config = small_config()
        character = Character(x=100.0)
        character.attach(config)
        sun = environment.Sun()
        air = environment.Air()
        for _ in range(3600):
            environment.step_body_temperature(character, air, sun, 1.0)
        warm = character.body_temperature_c
        self.assertAlmostEqual(warm, 37.0, delta=1.5)

        character.body_temperature_k = physics.HUMAN_BODY_TEMPERATURE_K
        sunless = environment.Sun(irradiance=0.0)
        for _ in range(3600):
            environment.step_body_temperature(character, air, sunless, 1.0)
        cold = character.body_temperature_c
        self.assertLess(cold, warm - 0.3)

    def test_hot_air_raises_temperature(self) -> None:
        config = small_config()
        character = Character(x=100.0)
        character.attach(config)
        sun = environment.Sun()
        for _ in range(3600):
            environment.step_body_temperature(character, environment.Air(temperature_k=313.15), sun, 1.0)
        hot = character.body_temperature_c
        character.body_temperature_k = physics.HUMAN_BODY_TEMPERATURE_K
        for _ in range(3600):
            environment.step_body_temperature(character, environment.Air(temperature_k=273.15), sun, 1.0)
        cold = character.body_temperature_c
        self.assertGreater(hot, cold + 0.5)

    def test_heat_balance_terms(self) -> None:
        balance = environment.body_heat_balance(310.15, 293.15, 1000.0)
        self.assertGreater(balance.metabolic_w, 0.0)
        self.assertGreater(balance.solar_w, 0.0)
        self.assertGreater(balance.radiative_loss_w, 0.0)
        self.assertGreater(balance.convective_loss_w, 0.0)


# ---------------------------------------------------------------------------
# acoustics
# ---------------------------------------------------------------------------
class TestAcoustics(unittest.TestCase):
    def test_spherical_spreading_inverse_sqrt_distance(self) -> None:
        wave = acoustics.SoundWave(0.0, 0.0, 0.0, 10.0, 200.0, 0.02)
        # sample the whole wave: peak amplitude at 4 m should be half of 1 m
        moments = np.linspace(0.0, 9.0, 4000)
        values_1m = [abs(wave.pressure(100.0, 0.0, t)) for t in moments]
        values_4m = [abs(wave.pressure(400.0, 0.0, t)) for t in moments]
        ratio = max(values_1m) / max(values_4m)
        self.assertAlmostEqual(ratio, 2.0, delta=0.15)
        self.assertGreater(max(values_1m), max(values_4m))

    def test_sound_travels_at_speed_of_sound(self) -> None:
        wave = acoustics.SoundWave(0.0, 0.0, 0.0, 1.0, 173.0, 0.02)
        distance_px = 343.1 * physics.PIXELS_PER_METER  # 1 second of travel
        self.assertEqual(wave.pressure(distance_px, 0.0, 0.99), 0.0)
        after = [abs(wave.pressure(distance_px, 0.0, t)) for t in (1.005, 1.01, 1.02, 1.04)]
        self.assertGreater(max(after), 0.0)

    def test_spl_reference(self) -> None:
        self.assertAlmostEqual(physics.spl_db(20e-6), 0.0, places=6)
        self.assertAlmostEqual(physics.spl_db(2e-5), 0.0, places=6)
        self.assertAlmostEqual(physics.spl_db(0.02), 60.0, delta=0.1)

    def test_speech_is_audible_from_the_mouth_to_the_ear(self) -> None:
        config = small_config()
        world = World(config, seed=0)
        world.reset(positions=[400.0])
        event = world.speak(world.agent, "ba-da")
        self.assertGreater(len(event.waves), 0)
        world.step(dt=0.1, actions=0)  # let the first syllable reach the ear
        from newworld.character import eye_position

        ear_x, ear_y = eye_position(world.agent, config)
        rms = world.sound.rms_pressure_at(ear_x, ear_y, world.t)
        self.assertGreater(rms, 0.0)
        self.assertGreater(physics.spl_db(rms), 40.0)
        self.assertLess(physics.spl_db(rms), 110.0)

    def test_silence_before_any_sound(self) -> None:
        config = small_config()
        world = World(config, seed=0)
        self.assertEqual(world.sound_pressure_at(400.0, 300.0), 0.0)
        self.assertEqual(world.sound_rms_at(400.0, 300.0), 0.0)


# ---------------------------------------------------------------------------
# integration: free behaviour with all physics
# ---------------------------------------------------------------------------
class TestPhysicsIntegration(unittest.TestCase):
    def test_free_behaviour_jumps_speaks_and_hears(self) -> None:
        config = small_config()
        world = World(config, seed=1)
        world.reset(seed=1)
        mind = Mind(world, seed=1)
        heights = []
        spls = []
        for _ in range(2400):  # 40 s
            mind.step(config.fixed_dt)
            heights.append(world.agent.y)
            if mind.perception is not None:
                spls.append(mind.perception.sound_spl_db)
        self.assertGreater(world.agent.jump_count, 0)
        self.assertGreater(world.agent.speech_count, 0)
        self.assertLess(min(heights), config.ground_y - 20.0)  # he really leaves the ground
        self.assertGreaterEqual(min(heights), 0.0)
        self.assertGreater(max(spls), 20.0)  # he hears his own voice
        self.assertAlmostEqual(world.agent.body_temperature_c, 37.0, delta=3.0)

    def test_decision_object_can_be_passed_to_world_step(self) -> None:
        config = small_config()
        world = World(config, seed=0)
        observations = world.step(dt=1 / 60, actions=Decision(move=1, jump=True))
        self.assertEqual(len(observations), 1)
        self.assertFalse(world.agent.on_ground)

    def test_rendering_follows_the_jump(self) -> None:
        config = small_config()
        world = World(config, seed=0)
        world.reset(positions=[400.0])
        before = world.agent.y
        world.agent.update(1 / 60, 0, config, jump=True)
        for _ in range(12):
            world.agent.update(1 / 60, 0, config)
        self.assertLess(world.agent.y, before)
        layer, anchor = character_layer(world)
        self.assertGreater(layer.height, 0)
        frame = render_frame(world)
        self.assertEqual(frame.size, SMALL)


if __name__ == "__main__":
    unittest.main()
