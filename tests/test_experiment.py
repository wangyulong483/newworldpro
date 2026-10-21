"""P5 统一实验运行器单元测试：构建/推进/指标/产物/协议注册。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newworld.experiment import (
    build_simulation,
    gini,
    run_experiment,
    run_from_file,
    write_artifacts,
)
from newworld.registry import PROTOCOLS, register_protocol
from newworld.settings import (
    ExperimentSettings,
    SocialSettings,
    SwarmSettings,
    WorldSettings,
    apply_overrides,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
(PROJECT_ROOT / ".tmp").mkdir(exist_ok=True)


def tiny_settings(**overrides) -> ExperimentSettings:
    base = ExperimentSettings(
        name="tiny",
        world=WorldSettings(width=64, height=48, seed=3, ocean_margin=4, noise_scale=24,
                            food_regrow_per_s=0.0005),
        swarm=SwarmSettings(initial_population=20, capacity_hard=200),
        social=SocialSettings(enabled=False),
    )
    return apply_overrides(base, overrides) if overrides else base


class BuildSimulationTest(unittest.TestCase):
    def test_build_and_run(self) -> None:
        sim = build_simulation(tiny_settings(), seed=3)
        self.assertEqual(sim.swarm.obs_dim, 9)
        metrics = sim.run(seconds=1.0)
        self.assertIn("population", metrics)
        self.assertGreater(metrics["steps"], 0)
        self.assertEqual(metrics["seed"], 3)

    def test_deterministic_same_seed(self) -> None:
        def run() -> int:
            sim = build_simulation(tiny_settings(), seed=11)
            sim.run(seconds=1.0)
            return sim.swarm.eat_events

        self.assertEqual(run(), run())

    def test_policy_obs_dim_mismatch_rejected(self) -> None:
        from newworld.swarm_policy import NeuralSwarmPolicy

        sim = build_simulation(tiny_settings(), seed=3)
        swarm = sim.swarm
        with self.assertRaises(ValueError):
            swarm.attach_policy(NeuralSwarmPolicy(obs_dim=9 + 4, seed=0))

    def test_obs_provider_extends_observations(self) -> None:
        sim = build_simulation(tiny_settings(), seed=3)
        swarm = sim.swarm

        def provider(view, rows):
            return np.ones((rows.shape[0], 3), dtype=np.float64)

        self.assertEqual(swarm.register_obs_provider(provider, 3), 12)
        obs = swarm._observations(swarm.data[: swarm.count], np.arange(swarm.count))
        self.assertEqual(obs.shape[1], 12)
        self.assertTrue(np.all(obs[:, -3:] == 1.0))


class MetricsTest(unittest.TestCase):
    def test_gini_bounds(self) -> None:
        self.assertAlmostEqual(gini([]), 0.0)
        self.assertAlmostEqual(gini([0.5, 0.5, 0.5]), 0.0, places=6)
        self.assertGreater(gini([0.0, 0.0, 1.0]), 0.5)

    def test_metrics_keys(self) -> None:
        sim = build_simulation(tiny_settings(), seed=3)
        metrics = sim.run(seconds=1.0)
        for key in ("population", "births", "deaths", "eat_events", "satiation_mean",
                    "gini_hunger", "total_reward", "wall_s", "steps_per_s"):
            self.assertIn(key, metrics)

    def test_write_artifacts(self) -> None:
        settings = tiny_settings()
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / ".tmp")) as tmp:
            summary = {"verdict": "ok"}
            directory = write_artifacts(settings, summary, directory=Path(tmp) / "out")
            self.assertTrue((directory / "config.yaml").exists())
            payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["verdict"], "ok")
            self.assertEqual(payload["config_hash"], settings.hash())


class ProtocolDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.name = "unit_test_protocol"
        if self.name in PROTOCOLS:
            return

        @register_protocol(self.name, note="测试协议")
        def _run(settings: ExperimentSettings) -> dict:
            return {"name": settings.name, "verdict": "ran", "value": 7}

        self.addCleanup(lambda: None)

    def test_run_registered_protocol(self) -> None:
        settings = apply_overrides(tiny_settings(), ["name=unit", "protocol=" + self.name])
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / ".tmp")) as tmp:
            summary = run_experiment(settings, directory=Path(tmp) / "out")
        self.assertEqual(summary["verdict"], "ran")
        self.assertIn("_artifact_dir", summary)

    def test_unknown_protocol_rejected(self) -> None:
        settings = apply_overrides(tiny_settings(), ["name=unit", "protocol=no_such_protocol"])
        with self.assertRaises(ValueError):
            run_experiment(settings)

    def test_run_from_file(self) -> None:
        from newworld.settings import save_config

        settings = apply_overrides(tiny_settings(), ["name=fromfile", "protocol=" + self.name])
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / ".tmp")) as tmp:
            config_path = Path(tmp) / "cfg.yaml"
            save_config(settings, config_path)
            summary = run_from_file(config_path, overrides=[], directory=Path(tmp) / "out")
            self.assertEqual(summary["verdict"], "ran")
            self.assertTrue((Path(tmp) / "out" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
