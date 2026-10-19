"""Tests for teacher-imitation pretraining and continuous fine-tuning."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from newworld.neural import FEATURE_DIM, MODES, NeuralPolicy
from newworld.training import collect_dataset, fine_tune, train_supervised


class TestTraining(unittest.TestCase):
    def test_dataset(self) -> None:
        data = collect_dataset(seed=0, steps_per_hour=30, hours=(9.0, 21.0))
        count = 60
        self.assertEqual(data["x"].shape, (count, FEATURE_DIM))
        self.assertEqual(data["mode"].shape, (count,))
        self.assertEqual(data["word"].shape, (count, 3))
        self.assertTrue(np.all(np.isfinite(data["x"])))
        self.assertTrue(np.all(data["mode"] < len(MODES)))

    def test_supervised_imitation_learns_modes(self) -> None:
        data = collect_dataset(seed=0, steps_per_hour=80, hours=(8.0, 12.0, 22.0))
        policy = NeuralPolicy(seed=0)
        metrics = train_supervised(policy, data, epochs=12, seed=0)
        self.assertGreater(metrics["mode_accuracy"], 0.5)
        self.assertGreater(metrics["loss"], -1.0)

    def test_fine_tune_continues_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.npz"
            policy = NeuralPolicy(seed=1)
            policy.save(path)
            before = policy.flat_weights().copy()
            updated, metrics = fine_tune(path, seed=1, steps_per_hour=30, epochs=3)
            self.assertTrue(path.exists())
            self.assertFalse(np.allclose(before, updated.flat_weights()))
            self.assertIn("mode_accuracy", metrics)
            loaded = NeuralPolicy.load(path)
            self.assertEqual(loaded.flat_weights().shape, before.shape)


if __name__ == "__main__":
    unittest.main()
