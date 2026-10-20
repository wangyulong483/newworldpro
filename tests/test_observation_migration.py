"""验收测试：神经网络输入扩维与旧 checkpoint 的零初始化迁移（V2：40 -> 56）。

历史链：24（v0.8 感知/生理）→ 40（v2：+16 文明上下文）→ 56（v2.1：+16 真实视网膜特征）。
本测试证明：

1. 当前输入宽度为 56，视觉段 16 维、上下文段 16 维；
2. 旧的 24 维与 40 维 checkpoint 都能加载，**新增输入列权重为 0**；
3. 迁移后的网络在新增特征全 0 时输出与旧网络**逐比特一致**；
4. 视觉特征未启用（或尚无缓存）时视觉段全为 0。
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
    V2_FEATURE_DIM,
    V2_FLAT_SIZE,
    VISUAL_DIM,
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


def build_v2_network(seed: int = 0) -> NeuralPolicy:
    """构造一个 40 输入（v2）网络的扁平权重。"""
    rng = np.random.default_rng(seed)
    v2 = NeuralPolicy(seed=seed)
    w1 = rng.normal(0.0, 0.5, (V2_FEATURE_DIM, HIDDEN1))
    return np.concatenate(
        [
            w1.reshape(-1),
            np.zeros(HIDDEN1),
            rng.normal(0.0, 0.4, (HIDDEN1, HIDDEN2)).reshape(-1),
            np.zeros(HIDDEN2),
            rng.normal(0.0, 0.3, (HIDDEN2, OUTPUT_DIM)).reshape(-1),
            np.zeros(OUTPUT_DIM),
        ]
    )


class TestDimensions(unittest.TestCase):
    def test_current_dimensions(self) -> None:
        self.assertEqual(LEGACY_FEATURE_DIM, 24)
        self.assertEqual(CONTEXT_DIM, 16)
        self.assertEqual(VISUAL_DIM, 16)
        self.assertEqual(V2_FEATURE_DIM, 40)
        self.assertEqual(FEATURE_DIM, 56)
        self.assertEqual(LEGACY_FLAT_SIZE, 24 * 64 + 64 + 64 * 32 + 32 + 32 * 58 + 58)
        self.assertEqual(V2_FLAT_SIZE, 40 * 64 + 64 + 64 * 32 + 32 + 32 * 58 + 58)


class TestMigrationChain(unittest.TestCase):
    def test_legacy_24_to_56_zero_initialised(self) -> None:
        flat = legacy_weights()
        migrated = NeuralPolicy.from_legacy_weights(flat)
        self.assertEqual(migrated.w1.shape, (FEATURE_DIM, HIDDEN1))
        self.assertTrue(np.all(migrated.w1[LEGACY_FEATURE_DIM:] == 0.0))
        self.assertFalse(np.all(migrated.w1[:LEGACY_FEATURE_DIM] == 0.0))

    def test_v2_40_to_56_visual_columns_zero(self) -> None:
        flat = build_v2_network(seed=5)
        migrated = NeuralPolicy.from_v2_weights(flat)
        self.assertEqual(migrated.w1.shape, (FEATURE_DIM, HIDDEN1))
        self.assertTrue(np.all(migrated.w1[V2_FEATURE_DIM:] == 0.0))
        self.assertFalse(np.all(migrated.w1[:V2_FEATURE_DIM] == 0.0))

    def test_v2_outputs_are_bit_identical_on_zero_visual(self) -> None:
        flat = build_v2_network(seed=7)
        migrated = NeuralPolicy.from_v2_weights(flat)
        rng = np.random.default_rng(3)
        v2_inputs = rng.normal(size=(5, V2_FEATURE_DIM))
        extended = np.concatenate([v2_inputs, np.zeros((5, VISUAL_DIM))], axis=1)

        # reconstruct the original 40-input network for comparison
        v2 = NeuralPolicy(seed=0)
        index = 0
        v2.w1 = flat[index : index + V2_FEATURE_DIM * HIDDEN1].reshape(V2_FEATURE_DIM, HIDDEN1)
        index += V2_FEATURE_DIM * HIDDEN1
        v2.b1 = flat[index : index + HIDDEN1].copy()
        index += HIDDEN1
        v2.w2 = flat[index : index + HIDDEN1 * HIDDEN2].reshape(HIDDEN1, HIDDEN2).copy()
        index += HIDDEN1 * HIDDEN2
        v2.b2 = flat[index : index + HIDDEN2].copy()
        index += HIDDEN2
        v2.w3 = flat[index : index + HIDDEN2 * OUTPUT_DIM].reshape(HIDDEN2, OUTPUT_DIM).copy()
        index += HIDDEN2 * OUTPUT_DIM
        v2.b3 = flat[index : index + OUTPUT_DIM].copy()
        np.testing.assert_array_equal(
            migrated.forward(extended)[0], v2.forward(v2_inputs)[0]
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

    def test_v2_checkpoint_file_loads(self) -> None:
        flat = build_v2_network(seed=13)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v2.npz"
            np.savez(
                path,
                weights=flat,
                feature_dim=V2_FEATURE_DIM,
                hidden1=HIDDEN1,
                hidden2=HIDDEN2,
                output_dim=OUTPUT_DIM,
                version=np.asarray(3),
            )
            policy = NeuralPolicy.load(path)
        self.assertEqual(policy.w1.shape, (FEATURE_DIM, HIDDEN1))
        self.assertTrue(np.all(policy.w1[V2_FEATURE_DIM:] == 0.0))

    def test_saved_policy_roundtrip(self) -> None:
        policy = NeuralPolicy(seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.npz"
            policy.save(path)
            loaded = NeuralPolicy.load(path)
        np.testing.assert_array_equal(loaded.flat_weights(), policy.flat_weights())


class TestFeatureVector(unittest.TestCase):
    def test_feature_dim_is_56(self) -> None:
        world = World(make_config(480, 300), seed=0)
        world.reset(seed=0)
        vector = features(world)
        self.assertEqual(vector.shape, (FEATURE_DIM,))
        self.assertTrue(np.all(np.isfinite(vector)))

    def test_visual_segment_zero_without_cache(self) -> None:
        civ = CivilizationConfig(seed=0, enable_visual_features=True, enable_perception=False)
        world = World(make_config(480, 300), seed=0, civ_config=civ)
        world.reset(seed=0)
        vector = features(world, character=world.agent)
        self.assertTrue(np.all(vector[V2_FEATURE_DIM:] == 0.0))

    def test_visual_segment_reads_cached_features(self) -> None:
        civ = CivilizationConfig(seed=0, enable_visual_features=True, enable_perception=False)
        world = World(make_config(480, 300), seed=0, civ_config=civ)
        world.reset(seed=0)
        agent = world.agent
        agent.visual_features = tuple([0.5] * VISUAL_DIM)
        vector = features(world, character=agent)
        self.assertTrue(np.allclose(vector[V2_FEATURE_DIM:], 0.5))

    def test_visual_segment_zero_when_disabled(self) -> None:
        civ = CivilizationConfig(seed=0, enable_visual_features=False, enable_perception=False)
        world = World(make_config(480, 300), seed=0, civ_config=civ)
        world.reset(seed=0)
        agent = world.agent
        agent.visual_features = tuple([0.5] * VISUAL_DIM)
        vector = features(world, character=agent)
        self.assertTrue(np.all(vector[V2_FEATURE_DIM:] == 0.0))


if __name__ == "__main__":
    unittest.main()
