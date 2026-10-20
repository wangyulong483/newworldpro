"""世界核心测试：几何/物理/生态/观测/默认值。

Tests for newworld.  Run with:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newworld import WorldConfig, make_config, parse_size
from newworld import art
from newworld.character import LEFT, RIGHT, STAY, Character
from newworld.env import (
    AliveReward,
    CenterReward,
    ExplorationReward,
    NewWorldEnv,
    ZeroReward,
)
from newworld.policies import WanderPolicy, make_policy
from newworld.preview import generate_previews
from newworld.render import (
    build_debug_panel,
    character_layer,
    render_frame,
    render_world,
)
from newworld.world import World as WorldClass

# --- test-environment shim (Windows/DSH sandbox tempdir fix) ---------------
try:  # "python -m unittest tests.test_world"
    from tests._env_compat import patch_tempdir
except ImportError:  # "python -m unittest discover -s tests"
    from _env_compat import patch_tempdir
patch_tempdir()
# -------------------------------------------------------------------------

SMALL = (800, 500)


def small_config(**overrides) -> WorldConfig:
    return make_config(*SMALL, **overrides)


class TestArt(unittest.TestCase):
    def test_grid_is_rectangular(self) -> None:
        self.assertEqual(len(art.SPRITE_ROWS), art.SPRITE_H)
        for row in art.SPRITE_ROWS:
            self.assertEqual(len(row), art.SPRITE_W)
        self.assertGreaterEqual(art.SPRITE_W, 10)
        self.assertGreaterEqual(art.SPRITE_H, 10)

    def test_sprite_image_scaling_and_facing(self) -> None:
        image = art.sprite_image(cell=4)
        self.assertEqual(image.size, (art.SPRITE_W * 4, art.SPRITE_H * 4))
        self.assertEqual(image.mode, "RGBA")
        mirrored = art.sprite_image(cell=4, facing=-1)
        self.assertNotEqual(image.tobytes(), mirrored.tobytes())
        # the corners of the blueprint are transparent
        self.assertEqual(image.getpixel((0, 0))[3], 0)

    def test_palette_keys_are_known(self) -> None:
        keys = set("".join(art.SPRITE_ROWS))
        self.assertTrue(keys.issubset(set(art.PALETTE)))


class TestConfig(unittest.TestCase):
    def test_derived_geometry(self) -> None:
        config = small_config()
        self.assertEqual(config.width, SMALL[0])
        self.assertEqual(config.height, SMALL[1])
        self.assertGreater(config.ground_y, 0)
        self.assertLess(config.ground_y, config.height)
        self.assertGreater(config.min_x, 0)
        self.assertGreater(config.max_x, config.min_x)
        self.assertLess(config.max_x, config.width)
        self.assertGreaterEqual(config.cell, 4)
        self.assertGreater(config.sprite_w, 0)
        self.assertLessEqual(config.sprite_w, config.width)
        self.assertLessEqual(config.sprite_h, config.ground_y)
        # the whole world is a plain white canvas; nothing else is painted
        self.assertEqual(config.background, (255, 255, 255))

    def test_walk_range_almost_fills_the_screen(self) -> None:
        # on the target 2560x1600 screen the inhabitant must be able to walk
        # across more than 90% of the screen width
        config = make_config(2560, 1600)
        self.assertGreater(config.travel_ratio, 0.9)
        self.assertLess(config.sprite_h, config.height * 0.15)

    def test_cell_override(self) -> None:
        config = small_config(cell=10)
        self.assertEqual(config.cell, 10)
        self.assertEqual(config.sprite_w, art.SPRITE_W * 10)

    def test_parse_size(self) -> None:
        self.assertEqual(parse_size("2560x1600"), (2560, 1600))
        self.assertEqual(parse_size("768,432"), (768, 432))
        with self.assertRaises(ValueError):
            parse_size("nope")

    def test_rejects_tiny_world(self) -> None:
        with self.assertRaises(ValueError):
            WorldConfig(width=100, height=100)


class TestCharacter(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()
        self.character = Character(x=self.config.min_x)
        self.character.attach(self.config)

    def test_walks_right(self) -> None:
        start = self.character.x
        for _ in range(30):
            self.character.update(1 / 60, RIGHT, self.config)
        self.assertGreater(self.character.x, start)
        self.assertEqual(self.character.facing, 1)
        self.assertGreater(self.character.vx, 0)

    def test_walks_left(self) -> None:
        character = Character(x=(self.config.min_x + self.config.max_x) / 2)
        character.attach(self.config)
        for _ in range(30):
            character.update(1 / 60, LEFT, self.config)
        self.assertEqual(character.facing, -1)
        self.assertLess(character.vx, 0)

    def test_clamped_to_left_wall(self) -> None:
        for _ in range(600):
            self.character.update(1 / 60, LEFT, self.config)
        self.assertAlmostEqual(self.character.x, self.config.min_x, places=6)
        self.assertEqual(self.character.vx, 0.0)
        self.assertTrue(self.character.at_left_wall)

    def test_clamped_to_right_wall(self) -> None:
        character = Character(x=self.config.max_x - 5)
        character.attach(self.config)
        for _ in range(600):
            character.update(1 / 60, RIGHT, self.config)
        self.assertAlmostEqual(character.x, self.config.max_x, places=6)
        self.assertEqual(character.vx, 0.0)
        self.assertTrue(character.at_right_wall)

    def test_idle_friction_stops_the_creature(self) -> None:
        character = Character(x=(self.config.min_x + self.config.max_x) / 2)
        character.attach(self.config)
        character.vx = self.config.speed * 0.5
        for _ in range(120):
            character.update(1 / 60, STAY, self.config)
        self.assertEqual(character.vx, 0.0)
        x_after_stop = character.x
        for _ in range(30):
            character.update(1 / 60, STAY, self.config)
        self.assertEqual(character.x, x_after_stop)

    def test_observation_ranges(self) -> None:
        self.character.x = self.config.min_x
        observation = self.character.observation(self.config)
        self.assertEqual(len(observation), 8)
        self.assertAlmostEqual(observation[0], 0.0)
        self.assertAlmostEqual(observation[3], 0.0)
        self.assertAlmostEqual(observation[4], 1.0)
        for value in observation:
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)


class TestWorld(unittest.TestCase):
    def test_spawn_and_bounds(self) -> None:
        world = WorldClass(small_config(), seed=0)
        self.assertEqual(world.n_agents, 1)
        world.spawn(x=10_000)  # outside: clamps on next reset, not on spawn
        world.reset(seed=0)
        for character in world.agents:
            self.assertGreaterEqual(character.x, world.config.min_x)
            self.assertLessEqual(character.x, world.config.max_x)

    def test_reset_positions(self) -> None:
        world = WorldClass(small_config(), seed=0)
        observations = world.reset(positions=[world.config.min_x])
        self.assertAlmostEqual(world.agent.x, world.config.min_x)
        self.assertEqual(len(observations), 1)

    def test_reset_position_length_mismatch(self) -> None:
        world = WorldClass(small_config(), seed=0)
        with self.assertRaises(ValueError):
            world.reset(positions=[100, 200])

    def test_deterministic_reset(self) -> None:
        world_a = WorldClass(small_config(), seed=42)
        world_b = WorldClass(small_config(), seed=42)
        world_a.reset(seed=42)
        world_b.reset(seed=42)
        self.assertAlmostEqual(world_a.agent.x, world_b.agent.x)

    def test_multi_agent_actions(self) -> None:
        world = WorldClass(small_config(), seed=0)
        second = world.spawn(x=world.config.max_x - 10)
        world.reset(seed=0)
        observations = world.step(dt=1 / 60, actions=[RIGHT, LEFT])
        self.assertEqual(len(observations), 2)
        self.assertGreater(world.agents[0].x, world.config.min_x)
        self.assertLess(second.x, world.config.max_x)

    def test_state_snapshot_is_json_serialisable(self) -> None:
        world = WorldClass(small_config(), seed=0)
        json.dumps(world.state())


class TestEnv(unittest.TestCase):
    def test_reset_and_step_api(self) -> None:
        env = NewWorldEnv(small_config(), seed=0, max_steps=4)
        observation, info = env.reset(seed=0)
        self.assertEqual(observation.shape, (8,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertIsInstance(info, dict)
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(2)
            self.assertEqual(observation.shape, (8,))
            self.assertIsInstance(reward, float)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        observation, reward, terminated, truncated, info = env.step(2)
        self.assertTrue(truncated)
        self.assertFalse(terminated)

    def test_actions_move_the_inhabitant(self) -> None:
        env = NewWorldEnv(small_config(), seed=0)
        env.reset(seed=0)
        start = env.world.agent.x
        for _ in range(60):
            env.step(2)  # right
        self.assertGreater(env.world.agent.x, start)

    def test_exploration_reward_state_resets(self) -> None:
        reward = ExplorationReward(bins=4, new_bin_reward=1.0)
        env = NewWorldEnv(small_config(), reward=reward, seed=0)
        env.reset(seed=0, options={"positions": [env.world.config.min_x]})
        total = 0.0
        for _ in range(120):
            _, step_reward, *_ = env.step(2)
            total += step_reward
        self.assertGreater(total, 0.0)
        self.assertGreater(len(reward.visited), 1)
        env.reset(seed=0)
        self.assertEqual(reward.visited, set())

    def test_reward_variants(self) -> None:
        for reward in (ZeroReward(), AliveReward(), CenterReward()):
            env = NewWorldEnv(small_config(), reward=reward, seed=1)
            env.reset(seed=1)
            _, value, *_ = env.step(1)
            self.assertIsInstance(value, float)

    def test_render_returns_rgb_array(self) -> None:
        env = NewWorldEnv(small_config(), seed=0)
        env.reset(seed=0)
        frame = env.render()
        self.assertEqual(frame.shape, (SMALL[1], SMALL[0], 3))
        self.assertEqual(frame.dtype, np.uint8)


class TestRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = small_config()
        cls.world = WorldClass(cls.config, seed=0)
        cls.world.reset(seed=0)

    def test_world_image(self) -> None:
        image = render_world(self.world)
        self.assertEqual(image.size, SMALL)
        self.assertEqual(image.mode, "RGB")

    def test_character_layer(self) -> None:
        layer, anchor = character_layer(self.world)
        self.assertGreater(layer.width, 0)
        self.assertGreater(layer.height, 0)
        self.assertEqual(layer.mode, "RGBA")
        self.assertTrue(0 <= anchor[0] < layer.width)
        self.assertTrue(0 <= anchor[1] < layer.height)

    def test_render_frame_and_debug(self) -> None:
        frame = render_frame(self.world)
        self.assertEqual(frame.size, SMALL)
        debug_frame = render_frame(
            self.world, debug=True, debug_lines=("测试调试信息", "第二行")
        )
        self.assertEqual(debug_frame.size, SMALL)
        self.assertNotEqual(frame.tobytes(), debug_frame.tobytes())
        panel = build_debug_panel(self.world, ["测试"])
        self.assertGreater(panel.width, 0)

    def test_jump_changes_the_drawn_position(self) -> None:
        world = WorldClass(self.config, seed=0)
        world.reset(seed=0)
        grounded_image, grounded_anchor = character_layer(world)
        world.agent.update(1 / 60, 0, self.config, jump=True)
        for _ in range(12):
            world.agent.update(1 / 60, 0, self.config)
        airborne_image, airborne_anchor = character_layer(world)
        self.assertEqual(airborne_anchor, grounded_anchor)
        frame = render_frame(world)
        self.assertEqual(frame.size, SMALL)


class TestPreview(unittest.TestCase):
    def test_generate_previews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_previews(small_config(), tmp)
            self.assertEqual(
                set(paths),
                {
                    "world_preview",
                    "world_preview_1280",
                    "world_night_preview",
                    "world_debug_preview",
                    "world_daynight_preview",
                },
            )
            for path in paths.values():
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0)
            with_tmp = Path(tmp)
            preview = with_tmp / "world_preview.png"
            from PIL import Image

            with Image.open(preview) as image:
                self.assertEqual(image.size, SMALL)


class TestCli(unittest.TestCase):
    def test_headless_run_writes_log_and_frames(self) -> None:
        from newworld.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.jsonl"
            frames_dir = Path(tmp) / "frames"
            code = cli_main(
                [
                    "--headless",
                    "--size",
                    "800x500",
                    "--steps",
                    "25",
                    "--policy",
                    "wander",
                    "--seed",
                    "3",
                    "--log",
                    str(log_path),
                    "--frames",
                    str(frames_dir),
                    "--frame-every",
                    "10",
                ]
            )
            self.assertEqual(code, 0)
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 25)
            first = json.loads(lines[0])
            for key in ("step", "t", "action", "x", "vx", "obs"):
                self.assertIn(key, first)
            frames = sorted(frames_dir.glob("frame_*.png"))
            self.assertEqual(len(frames), 3)  # steps 0, 10, 20

    def test_headless_free_run_logs_perception(self) -> None:
        from newworld.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "free.jsonl"
            code = cli_main(
                [
                    "--headless",
                    "--size",
                    "800x500",
                    "--steps",
                    "40",
                    "--log",
                    str(log_path),
                ]
            )
            self.assertEqual(code, 0)
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 40)
            record = json.loads(lines[-1])
            for key in ("gaze_deg", "pupil_mm", "mean_luminance", "eye_mode", "blink"):
                self.assertIn(key, record)

    def test_preview_cli(self) -> None:
        from newworld.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmp:
            code = cli_main(["--preview", tmp, "--size", "800x500"])
            self.assertEqual(code, 0)
            self.assertTrue((Path(tmp) / "world_preview.png").exists())

    def test_info_cli(self) -> None:
        from newworld.cli import main as cli_main

        self.assertEqual(cli_main(["--info"]), 0)


class TestPolicies(unittest.TestCase):
    def test_wander_stays_in_bounds(self) -> None:
        world = WorldClass(small_config(), seed=0)
        policy = WanderPolicy(seed=1)
        for _ in range(3000):
            world.step(actions=policy(world))
            self.assertGreaterEqual(world.agent.x, world.config.min_x - 1e-9)
            self.assertLessEqual(world.agent.x, world.config.max_x + 1e-9)

    def test_make_policy(self) -> None:
        for name in ("wander", "random", "left", "right", "stand"):
            policy = make_policy(name, seed=0)
            self.assertTrue(callable(policy))
        with self.assertRaises(ValueError):
            make_policy("does-not-exist")


if __name__ == "__main__":
    unittest.main()
