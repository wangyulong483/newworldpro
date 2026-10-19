"""Tests for the optional neural-network behaviour policy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from newworld import World, make_config
from newworld.autonomy import Mind
from newworld.neural import FEATURE_DIM, MODES, NeuralInnerLife, NeuralPolicy, features


def small_world() -> World:
    world = World(make_config(800, 500), seed=0)
    world.reset(seed=0)
    return world


class TestNeuralPolicy(unittest.TestCase):
    def test_features(self) -> None:
        world = small_world()
        vector = features(world)
        self.assertEqual(vector.shape, (FEATURE_DIM,))
        self.assertTrue(np.all(np.isfinite(vector)))

    def test_policy_outputs_valid_actions(self) -> None:
        policy = NeuralPolicy(seed=1)
        output = policy.act(features(small_world()))
        self.assertIn(output.mode, MODES)
        self.assertIn(output.move, (-1, 0, 1))
        self.assertIsInstance(output.jump, bool)
        self.assertIsInstance(output.speak, bool)

    def test_weights_roundtrip(self) -> None:
        policy = NeuralPolicy(seed=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.npz"
            policy.save(path)
            loaded = NeuralPolicy.load(path)
        a = policy.forward(features(small_world()))[0]
        b = loaded.forward(features(small_world()))[0]
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_neural_life_runs_and_interacts(self) -> None:
        world = small_world()
        policy = NeuralPolicy(seed=3)
        life = NeuralInnerLife(policy, seed=3)
        for _ in range(240):
            decision = life.decide(world, 1.0 / 60.0)
            world.step(actions=decision)
            if decision.speak:
                world.speak(world.agent)
        self.assertIsNotNone(life.last_output)
        self.assertGreater(world.t, 0.0)
        self.assertIsNotNone(world.agent.physiology)

    def test_mind_with_neural_policy(self) -> None:
        world = small_world()
        mind = Mind(world, seed=4, policy=NeuralPolicy(seed=4))
        for _ in range(90):
            mind.step(1.0 / 60.0)
        self.assertIsNotNone(mind.perception)
        self.assertTrue(mind.neural)


if __name__ == "__main__":
    unittest.main()
