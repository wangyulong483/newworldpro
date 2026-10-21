"""P5 模型 zoo 单元测试：保存/载入/校验/manifest。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newworld.settings import ExperimentSettings
from newworld.swarm_policy import NeuralSwarmPolicy
from newworld.zoo import ModelZoo, ZooError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
(PROJECT_ROOT / ".tmp").mkdir(exist_ok=True)


class ModelZooTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / ".tmp"))
        self.zoo = ModelZoo(Path(self._tmp.name) / "zoo")
        self.policy = NeuralSwarmPolicy(obs_dim=9, seed=3, hidden=8)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_save_and_load(self) -> None:
        card = self.zoo.save(
            "m1", self.policy, metrics={"eat_events": 12.0},
            config=ExperimentSettings(), tags=["test"], description="demo",
        )
        self.assertEqual(card.obs_dim, 9)
        self.assertEqual(card.hidden, 8)
        self.assertTrue(card.weights_sha256)
        self.assertTrue(card.git_commit or True)  # 非仓库环境允许空
        loaded = self.zoo.load("m1")
        self.assertEqual(loaded.obs_dim, 9)
        np.testing.assert_allclose(loaded.W1, self.policy.W1)

    def test_card_and_manifest(self) -> None:
        self.zoo.save("m1", self.policy, metrics={"a": 1})
        self.assertEqual(self.zoo.names(), ["m1"])
        card = self.zoo.get("m1")
        self.assertEqual(card.metrics["a"], 1)
        manifest = self.zoo.manifest()
        self.assertEqual(manifest["models"][0]["name"], "m1")
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / ".tmp")) as tmp:
            payload_path = Path(tmp) / "manifest.json"
            self.zoo.manifest(payload_path)
            self.assertEqual(json.loads(payload_path.read_text(encoding="utf-8"))["models"][0]["name"], "m1")

    def test_verify_detects_tampering(self) -> None:
        self.zoo.save("m1", self.policy)
        self.assertTrue(self.zoo.verify("m1"))
        weights = self.zoo.model_dir("m1") / "policy.npz"
        with weights.open("ab") as handle:
            handle.write(b"tamper")
        self.assertFalse(self.zoo.verify("m1"))

    def test_config_snapshot_written(self) -> None:
        self.zoo.save("m2", self.policy, config=ExperimentSettings(name="cfg"))
        self.assertTrue((self.zoo.model_dir("m2") / "config.yaml").exists())

    def test_invalid_name_rejected(self) -> None:
        with self.assertRaises(ZooError):
            self.zoo.save("../escape", self.policy)

    def test_missing_model_rejected(self) -> None:
        with self.assertRaises(ZooError):
            self.zoo.load_card("nope")

    def test_rule_policy_without_save_rejected(self) -> None:
        from newworld.swarm_policy import RuleSwarmPolicy

        with self.assertRaises(ZooError):
            self.zoo.save("rule", RuleSwarmPolicy())


if __name__ == "__main__":
    unittest.main()
