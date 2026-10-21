"""P5 YAML 配置层单元测试：严格校验/继承/覆盖/指纹。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from newworld.settings import (
    ConfigError,
    ExperimentSettings,
    WorldSettings,
    apply_overrides,
    config_hash,
    load_config,
    save_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
(PROJECT_ROOT / ".tmp").mkdir(exist_ok=True)


def write_temp_yaml(directory: Path, name: str, payload: dict) -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


class DefaultsTest(unittest.TestCase):
    def test_defaults_valid_and_hashable(self) -> None:
        settings = ExperimentSettings()
        self.assertEqual(len(settings.hash()), 64)
        self.assertGreater(settings.run.steps, 0)
        self.assertEqual(config_hash(settings), settings.hash())

    def test_hash_sensitive_to_fields(self) -> None:
        a = ExperimentSettings()
        b = ExperimentSettings(world=WorldSettings(width=128))
        self.assertNotEqual(a.hash(), b.hash())

    def test_dict_roundtrip(self) -> None:
        settings = ExperimentSettings()
        again = ExperimentSettings.__dataclass_fields__  # noqa: F841
        from newworld.settings import _from_dict  # 内部严格构造器

        clone = _from_dict(ExperimentSettings, settings.to_dict())
        self.assertEqual(clone, settings)
        self.assertEqual(clone.hash(), settings.hash())

    def test_invalid_values_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentSettings(world=WorldSettings(width=16))
        with self.assertRaises(ValueError):
            from newworld.settings import SwarmSettings

            SwarmSettings(initial_population=100, capacity_hard=10)


class YamlLoadTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / ".tmp"))
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_load_simple(self) -> None:
        path = write_temp_yaml(self.dir, "a.yaml", {"name": "exp", "world": {"width": 96}})
        settings = load_config(path)
        self.assertEqual(settings.name, "exp")
        self.assertEqual(settings.world.width, 96)
        self.assertEqual(settings.world.height, WorldSettings().height)

    def test_base_inheritance_and_deep_merge(self) -> None:
        write_temp_yaml(self.dir, "base.yaml", {"world": {"width": 96, "height": 64}, "run": {"seeds": [1, 2]}})
        path = write_temp_yaml(
            self.dir, "child.yaml", {"_base": "base.yaml", "name": "child", "world": {"width": 128}}
        )
        settings = load_config(path)
        self.assertEqual(settings.world.width, 128)
        self.assertEqual(settings.world.height, 64)
        self.assertEqual(settings.run.seeds, (1, 2))

    def test_base_cycle_rejected(self) -> None:
        write_temp_yaml(self.dir, "a.yaml", {"_base": "b.yaml"})
        write_temp_yaml(self.dir, "b.yaml", {"_base": "a.yaml"})
        with self.assertRaises(ConfigError):
            load_config(self.dir / "a.yaml")

    def test_unknown_field_rejected(self) -> None:
        path = write_temp_yaml(self.dir, "a.yaml", {"world": {"widht": 96}})
        with self.assertRaises(ConfigError) as ctx:
            load_config(path)
        self.assertIn("widht", str(ctx.exception))

    def test_wrong_type_rejected(self) -> None:
        path = write_temp_yaml(self.dir, "a.yaml", {"world": {"width": "wide"}})
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_base_yaml_in_repo_loads(self) -> None:
        settings = load_config(PROJECT_ROOT / "configs" / "base.yaml")
        self.assertEqual(settings.name, "base")
        self.assertGreater(settings.run.seeds[0], -1)

    def test_save_roundtrip(self) -> None:
        path = self.dir / "saved.yaml"
        settings = ExperimentSettings(name="saved", world=WorldSettings(width=128))
        save_config(settings, path)
        self.assertEqual(load_config(path).hash(), settings.hash())


class OverridesTest(unittest.TestCase):
    def test_dotted_override_with_types(self) -> None:
        settings = apply_overrides(
            ExperimentSettings(),
            ["world.width=128", "run.seeds=[7,8,9]", "policy.kind=neural", "comm.enabled=true"],
        )
        self.assertEqual(settings.world.width, 128)
        self.assertEqual(settings.run.seeds, (7, 8, 9))
        self.assertEqual(settings.policy.kind, "neural")
        self.assertTrue(settings.comm.enabled)

    def test_unknown_path_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            apply_overrides(ExperimentSettings(), ["world.typo=1"])

    def test_malformed_override_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            apply_overrides(ExperimentSettings(), ["world.width"])

    def test_no_overrides_is_identity(self) -> None:
        settings = ExperimentSettings()
        self.assertEqual(apply_overrides(settings, []), settings)


if __name__ == "__main__":
    unittest.main()
