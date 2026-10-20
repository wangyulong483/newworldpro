"""``newworld.app_tk`` 的 GUI 集成测试（无显示环境时整类自动 skip）。

运行：``cd newworldpro; $env:TEMP=(Resolve-Path .tmp).Path; python -m unittest tests.test_ui_app -v``

契约（V1 契约 §1.3，接口见 app_tk/ui_state docstring）：>=6 个用例；若 ``tkinter.Tk()`` 可用
（或环境变量 ``NEWWORLD_UI_TEST=1``）则构造 app 并调用
``run_app(..., ui_test_path=tmp)`` 断言 exit code 0 与 JSON 结果；否则 ``skipTest``。
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path

from newworld.app_tk import HELP, TK_AVAILABLE, run_app
from newworld.config import make_config

# --- test-environment shim (Windows/DSH sandbox tempdir fix) ---------------
try:  # "python -m unittest tests.test_ui_app"
    from tests._env_compat import patch_tempdir
except ImportError:  # "python -m unittest discover -s tests"
    from _env_compat import patch_tempdir

patch_tempdir()
# -------------------------------------------------------------------------

#: 需要进入的本模块交互断言（与 app_tk.NewWorldApp.UI_TEST_CHECKS 对应）
REQUIRED_CHECKS = (
    "layout_regions_and_click_isolation",
    "viewport_transform_and_zoom_anchor",
    "world_click_selects_agent",
    "world_click_switches_selection",
    "world_click_toggles_off",
    "world_click_empty_clears_selection",
    "wheel_zoom_and_drag_pan_hit_consistency",
    "pause_freezes_time_but_panel_visible",
    "single_step_advances_exactly_one_frame",
    "list_click_syncs_selection",
    "dead_agent_clears_selection",
    "detail_and_stats_use_real_data",
    "reset_clears_selection_and_list",
)


def _display_available() -> tuple[bool, str]:
    """能否真正创建 Tk 窗口（无显示/无 tkinter 时返回 False + 原因）。"""
    if not TK_AVAILABLE:
        return False, "缺少 tkinter 或 Pillow"
    if os.environ.get("NEWWORLD_UI_TEST") == "1":
        return True, "NEWWORLD_UI_TEST=1"
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        root.destroy()
        return True, "Tk 可用"
    except Exception as exc:  # pragma: no cover - 无显示环境
        return False, f"无法创建 Tk 窗口：{exc}"


def _run_ui_test(directory: str, *, size: tuple[int, int] = (1280, 800)) -> tuple[int, dict]:
    path = Path(directory) / "ui_test.json"
    code = run_app(make_config(*size), fullscreen=False, ui_test_path=path)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return code, payload


class TestUiApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        available, reason = _display_available()
        if not available:
            raise unittest.SkipTest(f"无显示环境，跳过 GUI 集成测试（{reason}）")
        cls._display_reason = reason

    # -- 1. --ui-test 真实跑通 -----------------------------------------
    def test_ui_test_exit_code_zero_and_json_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, payload = _run_ui_test(tmp)
            self.assertEqual(code, 0, payload)
            self.assertTrue(payload.get("ok"))
            self.assertGreaterEqual(int(payload.get("total", 0)), 10)
            self.assertEqual(payload.get("failed"), 0)
            self.assertEqual(payload.get("passed"), payload.get("total"))

    # -- 2. JSON 结构与 13 项断言名称 ----------------------------------
    def test_ui_test_json_contains_all_interaction_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, payload = _run_ui_test(tmp)
            names = [check["name"] for check in payload.get("checks", ())]
            self.assertEqual(names, list(REQUIRED_CHECKS))
            for check in payload["checks"]:
                self.assertTrue(check["ok"], check)
                self.assertTrue(check["detail"])
            self.assertIn("world", payload)
            self.assertIn("alive", payload["world"])
            self.assertIn("selection", payload["world"])

    # -- 3. 键盘辅助入口仍然存在 ---------------------------------------
    def test_keyboard_helpers_and_help_text(self) -> None:
        import newworld.app_tk as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        for binding in ("<space>", "<F1>", "<F2>", "<KeyPress-q>"):
            self.assertIn(binding, source)
        self.assertIn("鼠标", HELP)
        self.assertIn("按钮", HELP)

    # -- 4. run_app 旧签名兼容 + 新增 ui_test_path ---------------------
    def test_run_app_signature_compatibility(self) -> None:
        signature = inspect.signature(run_app)
        for name in ("fullscreen", "debug", "screenshot", "autoclose_ms", "seed", "policy", "policy_path"):
            self.assertIn(name, signature.parameters)
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIn("ui_test_path", signature.parameters)

    # -- 5. 不同窗口尺寸下 <Configure> 重建 Layout 仍能通过自检 --------
    def test_ui_test_works_with_other_window_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, payload = _run_ui_test(tmp, size=(960, 640))
            self.assertEqual(code, 0, payload)
            self.assertTrue(payload.get("ok"))
            self.assertEqual(payload.get("failed"), 0)

    # -- 6. --screenshot / --autoclose-ms 模式仍可用 -------------------
    def test_screenshot_and_autoclose_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "shot.png"
            code = run_app(
                make_config(960, 640),
                fullscreen=False,
                screenshot=shot,
                autoclose_ms=4000,
            )
            self.assertEqual(code, 0)
            if not shot.exists():  # pragma: no cover - 有些环境不允许抓屏
                self.skipTest("当前环境无法抓屏（ImageGrab 不可用）")
            self.assertGreater(shot.stat().st_size, 0)

    # -- 7. 面板数据构建对真实世界可用（GUI 无关的回归保护） -----------
    def test_detail_uses_absent_field_as_no_data(self) -> None:
        from newworld.population import Population
        from newworld.ui_state import NO_DATA, build_agent_detail, build_stats
        from newworld.world import World

        config = make_config(480, 320)
        world = World(config, seed=5)
        world.reset(seed=5)
        population = Population(world, seed=5)
        agent = world.agents[0]
        detail = build_agent_detail(world, population, agent.agent_id)
        values = {row.label: row.value for row in detail.rows}
        self.assertEqual(values["健康"], NO_DATA)
        self.assertNotEqual(values["能量"], NO_DATA)
        self.assertTrue(build_stats(world, population))


if __name__ == "__main__":
    unittest.main()
