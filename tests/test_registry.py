"""P5 注册表单元测试：注册/查表/重名/诊断。"""

from __future__ import annotations

import unittest

from newworld.registry import Registry, RegistryError


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = Registry("demo")

    def test_register_and_create(self) -> None:
        @self.registry.register("thing")
        class Thing:
            def __init__(self, value: int = 1) -> None:
                self.value = value

        self.assertEqual(self.registry.create("thing", value=7).value, 7)
        self.assertIn("thing", self.registry)
        self.assertEqual(self.registry.names(), ["thing"])

    def test_factory_without_decorator(self) -> None:
        self.registry.register("answer", lambda: 42)
        self.assertEqual(self.registry.create("answer"), 42)

    def test_duplicate_registration_rejected(self) -> None:
        self.registry.register("a", lambda: 1)
        with self.assertRaises(RegistryError):
            self.registry.register("a", lambda: 2)

    def test_overwrite_allowed_explicitly(self) -> None:
        self.registry.register("a", lambda: 1)
        self.registry.register("a", lambda: 2, overwrite=True)
        self.assertEqual(self.registry.create("a"), 2)

    def test_unknown_name_lists_available(self) -> None:
        self.registry.register("known", lambda: 1)
        with self.assertRaises(RegistryError) as ctx:
            self.registry.get("unknown")
        self.assertIn("known", str(ctx.exception))

    def test_invalid_name_and_factory(self) -> None:
        with self.assertRaises(RegistryError):
            self.registry.register("  ", lambda: 1)
        with self.assertRaises(RegistryError):
            self.registry.register("x", 123)  # type: ignore[arg-type]

    def test_notes_and_describe(self) -> None:
        @self.registry.register("noted", note="一行说明")
        def factory():
            return 1

        self.assertEqual(self.registry.notes()["noted"], "一行说明")
        self.assertIn("noted", self.registry.describe())

    def test_iteration_and_len(self) -> None:
        self.registry.register("a", lambda: 1)
        self.registry.register("b", lambda: 2)
        self.assertEqual(list(self.registry), ["a", "b"])
        self.assertEqual(len(self.registry), 2)


class GlobalRegistriesTest(unittest.TestCase):
    def test_policies_registered(self) -> None:
        from newworld.registry import POLICIES

        for name in ("rule", "random", "neural"):
            self.assertIn(name, POLICIES)

    def test_mechanisms_registry_exists(self) -> None:
        from newworld.registry import MECHANISMS

        self.assertEqual(MECHANISMS.kind, "mechanism")


if __name__ == "__main__":
    unittest.main()
