"""验收测试：神经网络输入扩维与旧 checkpoint 的零初始化迁移。

任务书 §6/§4.2 要求"先确认现有网络是否缺少必要输入，再决定是否增加记忆结构"，
并且不得破坏已有模型。本测试证明：

1. 观测维度从 24 增加到 40（8 记忆 + 4 目标 + 4 文明）；
2. 旧的 24 维 checkpoint 可以加载，**新增输入列权重为 0**；
3. 迁移后的网络在新增特征为 0 时输出与旧网络**逐比特一致**；
4. 未启用扩维时（A/B 组）上下文特征恒为 0。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from newworld.config import CivilizationConfig, make_config
from newworld.neural import (
    CONTEXT_DIM,
    FEATURE_DIM,
    HIDDEN1,
    HIDDEN2,
    LEGACY_FEATURE_DIM,
    LEGACY_FLAT_SIZE,
    OUTPUT_DIM,
    NeuralPolicy,
    features,
)
from newworld.world import World

try:  # "python -m unittest tests.test_observation_migration"
    from tests._env_compat import patch_tempdir
except ImportError:  # "python -m unittest discover -s tests"
    from _env_compat import patch_tempdir
patch_tempdir()


def legacy_weights(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.normal(0.0, 0.5, LEGACY_FEATURE_DIM * HIDDEN1),
            np.zeros(HIDDEN1),
            rng.normal(0.0, 0.4, HIDDEN1 * HIDDEN2),
            np.zeros(HIDDEN2),
            rng.normal(0.0, 0.3, HIDDEN2 * OUTPUT_DIM),
            np.zeros(OUTPUT_DIM),
        ]
    )


class TestLegacyMigration(unittest.TestCase):
    def test_dimensions(self) -> None:
        self.assertEqual(LEGACY_FEATURE_DIM, 24)
        self.assertEqual(CONTEXT_DIM, 16)
        self.assertEqual(FEATURE_DIM, 40)
        self.assertEqual(LEGACY_FLAT_SIZE, 24 * 64 + 64 + 64 * 32 + 32 + 32 * 58 + 58)

    def test_zero_initialised_new_columns(self) -> None:
        flat = legacy_weights()
        migrated = NeuralPolicy.from_legacy_weights(flat)
        self.assertEqual(migrated.w1.shape, (FEATURE_DIM, HIDDEN1))
        self.assertTrue(np.all(migrated.w1[LEGACY_FEATURE_DIM:] == 0.0))
        self.assertFalse(np.all(migrated.w1[:LEGACY_FEATURE_DIM] == 0.0))

    def test_outputs_are_bit_identical_on_zero_context(self) -> None:
        flat = legacy_weights(seed=7)
        migrated = NeuralPolicy.from_legacy_weights(flat)
        rng = np.random.default_rng(3)
        legacy_inputs = rng.normal(size=(5, LEGACY_FEATURE_DIM))
        extended = np.concatenate([legacy_inputs, np.zeros((5, CONTEXT_DIM))], axis=1)
        # reconstruct the original 24-input network for comparison
        legacy = NeuralPolicy(seed=0)
        index = 0
        legacy.w1 = flat[index : index + LEGACY_FEATURE_DIM * HIDDEN1].reshape(LEGACY_FEATURE_DIM, HIDDEN1)
        index += LEGACY_FEATURE_DIM * HIDDEN1
        legacy.b1 = flat[index : index + HIDDEN1].copy()
        index += HIDDEN1
        legacy.w2 = flat[index : index + HIDDEN1 * HIDDEN2].reshape(HIDDEN1, HIDDEN2).copy()
        index += HIDDEN1 * HIDDEN2
        legacy.b2 = flat[index : index + HIDDEN2].copy()
        index += HIDDEN2
        legacy.w3 = flat[index : index + HIDDEN2 * OUTPUT_DIM].reshape(HIDDEN2, OUTPUT_DIM).copy()
        index += HIDDEN2 * OUTPUT_DIM
        legacy.b3 = flat[index : index + OUTPUT_DIM].copy()
        np.testing.assert_array_equal(
            migrated.forward(extended)[0], legacy.forward(legacy_inputs)[0]
        )

    def test_legacy_checkpoint_file_loads(self) -> None:
        flat = legacy_weights(seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.npz"
            np.savez(
                path,
                weights=flat,
                feature_dim=LEGACY_FEATURE_DIM,
                hidden1=HIDDEN1,
                hidden2=HIDDEN2,
                output_dim=OUTPUT_DIM,
                version=np.asarray(2),
            )
            policy = NeuralPolicy.load(path)
        self.assertEqual(policy.w1.shape, (FEATURE_DIM, HIDDEN1))
        self.assertTrue(np.all(policy.w1[LEGACY_FEATURE_DIM:] == 0.0))

    def test_saved_policy_roundtrip(self) -> None:
        policy = NeuralPolicy(seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.npz"
            policy.save(path)
            loaded = NeuralPolicy.load(path)
        np.testing.assert_array_equal(loaded.flat_weights(), policy.flat_weights())


class TestFeatureVector(unittest.TestCase):
    def test_feature_dim_is_40(self) -> None:
        world = World(make_config(480, 300), seed=0)
        world.reset(seed=0)
        vector = features(world)
        self.assertEqual(vector.shape, (FEATURE_DIM,))

    def test_context_is_zero_when_expansion_disabled(self) -> None:
        civ = CivilizationConfig(seed=0, expanded_observation=False, enable_perception=False)
        world = World(make_config(480, 300), seed=0, civ_config=civ)
        world.reset(seed=0)
        # a non-zero memory entry must still not leak into the observation
        world.memory.record(world.agent.agent_id, 1.0, "episode", "state:gather", "ok", True)
        vector = features(world, character=world.agent)
        self.assertTrue(np.all(vector[LEGACY_FEATURE_DIM:] == 0.0))

    def test_context_is_populated_when_enabled(self) -> None:
        civ = CivilizationConfig(seed=0, expanded_observation=True, enable_perception=False)
        world = World(make_config(480, 300), seed=0, civ_config=civ)
        world.reset(seed=0)
        for _ in range(5):
            world.memory.record(world.agent.agent_id, 1.0, "episode", "state:gather", "ok", True)
        vector = features(world, character=world.agent)
        self.assertEqual(vector.shape, (FEATURE_DIM,))
        self.assertTrue(np.any(vector[LEGACY_FEATURE_DIM:] != 0.0))
        self.assertTrue(np.all(np.isfinite(vector)))


if __name__ == "__main__":
    unittest.main()
