"""``newworld.ui_state`` 的纯逻辑测试（无 tkinter、无显示依赖）。

运行：``cd newworldpro; $env:TEMP=(Resolve-Path .tmp).Path; python -m unittest tests.test_ui_state -v``

契约（docs/upgrade_contracts_v1.md §1.1 / §1.3）要求 >=14 个用例：
视口往返、缩放锚点、平移 clamp、resize；命中（bbox/最近/重叠/无命中/死人不选）；
选择切换/取消/死亡清理；控制状态机（暂停时单步、运行时不单步、速度档位）；
布局区域判定（UI 点击不判为 world）；detail 构建（真实字段 + 缺失显示“暂无数据”）；
事件日志/统计行非空；异常输入不抛错。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from newworld import ui_state as ui
from newworld.config import make_config
from newworld.population import Population
from newworld.world import World

# --- test-environment shim (Windows/DSH sandbox tempdir fix) ---------------
try:  # "python -m unittest tests.test_ui_state"
    from tests._env_compat import patch_tempdir
except ImportError:  # "python -m unittest discover -s tests"
    from _env_compat import patch_tempdir

patch_tempdir()
# -------------------------------------------------------------------------


def _box(agent_id: int, x: float, y: float, w: float = 40.0, h: float = 80.0, alive: bool = True) -> ui.AgentBox:
    return ui.AgentBox(agent_id=agent_id, x=x, y=y, sprite_w=w, sprite_h=h, alive=alive)


_WORLD_CACHE: dict[str, object] = {}


def _shared_world() -> tuple[World, Population]:
    """构造一次真实的 World + Population，并推进少量步数产生目标/记忆。"""
    cached = _WORLD_CACHE.get("pair")
    if cached is not None:
        return cached  # type: ignore[return-value]
    config = make_config(480, 320)
    world = World(config, seed=3)
    world.reset(seed=3, positions=[config.width * 0.4])
    population = Population(world, seed=3)
    for _ in range(25):
        population.step(1.0 / 60.0)
    _WORLD_CACHE["pair"] = (world, population)
    return world, population


# ---------------------------------------------------------------------------
# 视口
# ---------------------------------------------------------------------------
class TestViewport(unittest.TestCase):
    def setUp(self) -> None:
        self.view = ui.Viewport(1000, 800, 400, 300).clamp()

    def test_screen_world_roundtrip(self) -> None:
        for point in ((0.0, 0.0), (123.5, 45.25), (399.0, 299.0)):
            world_point = self.view.screen_to_world(*point)
            screen_point = self.view.world_to_screen(*world_point)
            self.assertAlmostEqual(screen_point[0], point[0], places=6)
            self.assertAlmostEqual(screen_point[1], point[1], places=6)

    def test_zoom_keeps_anchor_fixed(self) -> None:
        anchor = (200.0, 150.0)
        before = self.view.screen_to_world(*anchor)
        zoomed = self.view.zoom_at(anchor[0], anchor[1], 1.5)
        after = zoomed.screen_to_world(*anchor)
        self.assertAlmostEqual(zoomed.scale, 1.5, places=6)
        self.assertAlmostEqual(before[0], after[0], places=6)
        self.assertAlmostEqual(before[1], after[1], places=6)

    def test_zoom_scale_limits(self) -> None:
        self.assertAlmostEqual(self.view.zoom_at(10, 10, 1000.0).scale, ui.MAX_SCALE)
        self.assertAlmostEqual(self.view.zoom_at(10, 10, 1e-6).scale, ui.MIN_SCALE)

    def test_pan_then_clamp_keeps_world_inside(self) -> None:
        view = ui.Viewport(1000, 800, 400, 300, offset_x=100.0, offset_y=100.0)
        moved = view.pan(50.0, 30.0)
        self.assertAlmostEqual(moved.offset_x, 50.0)
        self.assertAlmostEqual(moved.offset_y, 70.0)
        dragged = view.pan(-10_000.0, -10_000.0).clamp()
        self.assertAlmostEqual(dragged.offset_x, 1000.0 - 400.0)
        self.assertAlmostEqual(dragged.offset_y, 800.0 - 300.0)
        back = view.pan(10_000.0, 10_000.0).clamp()
        self.assertAlmostEqual(back.offset_x, 0.0)
        self.assertAlmostEqual(back.offset_y, 0.0)

    def test_clamp_centers_small_world(self) -> None:
        view = ui.Viewport(200, 100, 400, 300).clamp()
        self.assertAlmostEqual(view.offset_x, (200 - 400) / 2.0)
        self.assertAlmostEqual(view.offset_y, (100 - 300) / 2.0)
        x0, y0, x1, y1 = view.visible_world_rect()
        self.assertAlmostEqual(x0, -100.0)
        self.assertAlmostEqual(y1, 200.0)

    def test_resize_keeps_center(self) -> None:
        view = ui.Viewport(1000, 800, 400, 300, offset_x=100.0, offset_y=100.0)
        center_before = view.screen_to_world(200.0, 150.0)
        resized = view.resize(600, 400)
        center_after = resized.screen_to_world(300.0, 200.0)
        self.assertAlmostEqual(center_before[0], center_after[0], places=6)
        self.assertAlmostEqual(center_before[1], center_after[1], places=6)

    def test_visible_world_rect(self) -> None:
        view = ui.Viewport(1000, 800, 400, 300, scale=2.0, offset_x=50.0, offset_y=25.0)
        x0, y0, x1, y1 = view.visible_world_rect()
        self.assertAlmostEqual(x0, 50.0)
        self.assertAlmostEqual(y0, 25.0)
        self.assertAlmostEqual(x1, 50.0 + 200.0)
        self.assertAlmostEqual(y1, 25.0 + 150.0)


# ---------------------------------------------------------------------------
# 命中测试
# ---------------------------------------------------------------------------
class TestHitTester(unittest.TestCase):
    def setUp(self) -> None:
        self.view = ui.Viewport(1000, 800, 500, 400)
        self.tester = ui.HitTester(self.view, max_distance_px=60.0)

    def test_hit_inside_bbox(self) -> None:
        hit = self.tester.hit([_box(7, 100, 200)], 110, 190)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.agent_id, 7)
        self.assertEqual(hit.method, "bbox")

    def test_hit_overlap_prefers_nearest(self) -> None:
        agents = [_box(1, 100, 200), _box(2, 110, 200)]
        hit = self.tester.hit(agents, 109, 200)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.agent_id, 2)

    def test_hit_nearest_fallback(self) -> None:
        hit = self.tester.hit([_box(3, 100, 200)], 140, 200)  # bbox 外，中心距 40
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.agent_id, 3)
        self.assertEqual(hit.method, "nearest")

    def test_hit_miss_returns_none(self) -> None:
        self.assertIsNone(self.tester.hit([_box(3, 100, 200)], 400, 350))
        self.assertIsNone(self.tester.hit([], 0, 0))

    def test_hit_ignores_dead_agents(self) -> None:
        dead = _box(4, 100, 200, alive=False)
        self.assertIsNone(self.tester.hit([dead], 100, 200))
        self.assertIsNone(self.tester.hit([dead], 110, 200))

    def test_hit_uses_viewport_scale(self) -> None:
        view = ui.Viewport(1000, 800, 500, 400, scale=2.0)
        tester = ui.HitTester(view)
        screen = view.world_to_screen(100.0, 200.0)
        hit = tester.hit([_box(9, 100, 200)], screen[0], screen[1])
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.agent_id, 9)
        # 屏幕像素距离判定：世界点再偏移 40 世界像素 = 80 屏幕像素 > 60
        offset = view.world_to_screen(140.0, 200.0)
        self.assertIsNone(tester.hit([_box(9, 100, 200)], offset[0], offset[1]))


# ---------------------------------------------------------------------------
# 选择模型
# ---------------------------------------------------------------------------
class TestSelection(unittest.TestCase):
    def test_select_and_clear(self) -> None:
        selection = ui.SelectionModel()
        self.assertFalse(selection.has_selection)
        self.assertTrue(selection.select(3, alive_ids={3, 4}))
        self.assertEqual(selection.selected_id, 3)
        self.assertTrue(selection.has_selection)
        self.assertFalse(selection.select(3, alive_ids={3, 4}))  # 没变化
        self.assertTrue(selection.clear())
        self.assertFalse(selection.clear())

    def test_select_rejects_dead_or_missing(self) -> None:
        selection = ui.SelectionModel()
        selection.sync({1, 2})
        self.assertFalse(selection.select(99))
        self.assertFalse(selection.select(99, alive_ids={1, 2}))
        self.assertFalse(selection.select(-1))
        self.assertIsNone(selection.selected_id)

    def test_toggle(self) -> None:
        selection = ui.SelectionModel()
        self.assertTrue(selection.toggle(5, alive_ids={5, 6}))
        self.assertEqual(selection.selected_id, 5)
        self.assertTrue(selection.toggle(5, alive_ids={5, 6}))
        self.assertIsNone(selection.selected_id)

    def test_sync_clears_dead_selection(self) -> None:
        selection = ui.SelectionModel()
        selection.select(2, alive_ids={1, 2})
        self.assertFalse(selection.sync({1, 2}))
        self.assertTrue(selection.sync({1, 3}))
        self.assertIsNone(selection.selected_id)


# ---------------------------------------------------------------------------
# 控制状态机
# ---------------------------------------------------------------------------
class TestControlState(unittest.TestCase):
    def test_step_only_when_paused(self) -> None:
        control = ui.ControlState(playing=True)
        self.assertFalse(control.request_step())
        self.assertTrue(control.pause())
        self.assertTrue(control.request_step())
        self.assertEqual(control.pending_steps, 1)
        self.assertTrue(control.resume())
        self.assertFalse(control.request_step())

    def test_consume_step_once(self) -> None:
        control = ui.ControlState(playing=False)
        self.assertFalse(control.consume_step())
        self.assertTrue(control.request_step())
        self.assertTrue(control.consume_step())
        self.assertFalse(control.consume_step())

    def test_speed_levels(self) -> None:
        control = ui.ControlState()
        self.assertEqual(control.speed, 1.0)
        self.assertTrue(control.set_speed(2.0))
        self.assertEqual(control.speed, 2.0)
        self.assertFalse(control.set_speed(3.0))
        self.assertEqual(control.speed, 2.0)
        self.assertAlmostEqual(control.nudge_speed(-1), 1.0)
        self.assertAlmostEqual(control.nudge_speed(+1), 2.0)
        control.set_speed(ui.SPEEDS[0])
        self.assertAlmostEqual(control.nudge_speed(-1), ui.SPEEDS[0])
        control.set_speed(ui.SPEEDS[-1])
        self.assertAlmostEqual(control.nudge_speed(+1), ui.SPEEDS[-1])

    def test_scaled_dt(self) -> None:
        control = ui.ControlState(playing=True, speed=2.0)
        self.assertAlmostEqual(control.scaled_dt(1.0 / 60.0), 2.0 / 60.0)
        control.pause()
        self.assertEqual(control.scaled_dt(1.0 / 60.0), 0.0)

    def test_available_actions(self) -> None:
        control = ui.ControlState(playing=True)
        actions = control.available_actions()
        for key in ("start", "pause", "resume", "step", "slower", "faster"):
            self.assertIn(key, actions)
        self.assertTrue(actions["pause"])
        self.assertFalse(actions["resume"])
        self.assertFalse(actions["step"])
        control.pause()
        self.assertTrue(control.available_actions()["step"])
        self.assertFalse(control.available_actions()["pause"])


# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------
class TestLayout(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = ui.Layout(1280, 800)

    def test_regions_are_distinct(self) -> None:
        rects = self.layout.rects()
        self.assertEqual(set(rects), {"top", "world", "panel", "bottom"})
        self.assertEqual(self.layout.region_at(640, 4), "top")
        self.assertEqual(self.layout.region_at(640, 400), "world")
        self.assertEqual(self.layout.region_at(1270, 400), "panel")
        self.assertEqual(self.layout.region_at(640, 790), "bottom")

    def test_ui_clicks_are_not_world(self) -> None:
        width, height = 1280, 800
        for point in (
            (10, 5),
            (width - 10, 5),          # 状态栏右上角
            (10, height - 5),         # 底部控制栏
            (width // 2, height - 5),
            (width - 10, height // 2),  # 右侧详情面板
        ):
            self.assertNotEqual(self.layout.region_at(*point), "world", point)

    def test_hidden_panel(self) -> None:
        hidden = self.layout.with_panel(False)
        rects = hidden.rects()
        self.assertEqual(rects["panel"][2], 0)
        self.assertEqual(rects["world"][2], 1280)
        self.assertEqual(hidden.region_at(1270, 400), "world")

    def test_resize_and_out_of_bounds(self) -> None:
        resized = self.layout.resize(900, 600)
        self.assertEqual(resized.width, 900)
        self.assertEqual(resized.height, 600)
        self.assertEqual(self.layout.region_at(-100, -100), "top")
        self.assertEqual(self.layout.region_at(10_000, 10_000), "bottom")
        self.assertIsInstance(self.layout.region_at(float("nan"), float("nan")), str)


# ---------------------------------------------------------------------------
# 面板数据（真实 World + Population）
# ---------------------------------------------------------------------------
class TestPanelData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world, cls.population = _shared_world()

    def test_detail_has_real_fields(self) -> None:
        agent = self.world.agent_by_id(0)
        self.assertIsNotNone(agent)
        detail = ui.build_agent_detail(self.world, self.population, 0)
        self.assertEqual(detail.agent_id, 0)
        self.assertIsInstance(detail.title, str)
        self.assertGreaterEqual(len(detail.rows), 10)
        values = {row.label: row.value for row in detail.rows}
        # 真实生理字段
        self.assertAlmostEqual(float(values["能量"]), agent.physiology.energy, places=1)
        self.assertAlmostEqual(float(values["水分"]), agent.physiology.hydration, places=1)
        self.assertIn("木", values["库存"])
        self.assertNotEqual(values["ID"], ui.NO_DATA)

    def test_detail_missing_field_shows_no_data(self) -> None:
        detail = ui.build_agent_detail(self.world, self.population, 0)
        values = {row.label: row.value for row in detail.rows}
        # 世界模型没有 health 字段 → 必须显示“暂无数据”而不是编造
        self.assertEqual(values["健康"], ui.NO_DATA)
        # 没有数据时必须是占位符，不能是空串/None
        for row in detail.rows:
            self.assertIsInstance(row.label, str)
            self.assertIsInstance(row.value, str)
            self.assertNotEqual(row.value, "")

    def test_detail_none_selection(self) -> None:
        detail = ui.build_agent_detail(self.world, self.population, None)
        self.assertIsNone(detail.agent_id)
        self.assertFalse(detail.selected)
        self.assertGreaterEqual(len(detail.rows), 1)
        self.assertIn("点击", "".join(row.value for row in detail.rows))

    def test_detail_unknown_agent(self) -> None:
        detail = ui.build_agent_detail(self.world, self.population, 9999)
        self.assertGreaterEqual(len(detail.rows), 1)

    def test_stats_rows_non_empty_and_real(self) -> None:
        rows = ui.build_stats(self.world, self.population)
        self.assertGreaterEqual(len(rows), 8)
        values = {row.label: row.value for row in rows}
        self.assertEqual(values["存活人口"], str(self.world.registry.alive_count()))
        self.assertIn(str(self.world.registry.total_created), values["累计创建"])
        self.assertIn("秒", values["世界时间"])
        self.assertNotEqual(values["文明状态"], "")

    def test_event_log_non_empty_and_limit(self) -> None:
        log = ui.build_event_log(self.world, self.population)
        self.assertGreaterEqual(len(log), 1)
        self.assertTrue(all(isinstance(line, str) and line for line in log))
        limited = ui.build_event_log(self.world, self.population, limit=3)
        self.assertLessEqual(len(limited), 3)
        self.assertEqual(ui.build_event_log(self.world, self.population, limit=0), [])

    def test_anomalous_inputs_do_not_raise(self) -> None:
        self.assertIsInstance(ui.build_agent_detail(None, None, None), ui.AgentDetail)
        self.assertIsInstance(ui.build_agent_detail(None, None, 5), ui.AgentDetail)
        self.assertIsInstance(ui.build_agent_detail(self.world, None, 0), ui.AgentDetail)
        self.assertGreaterEqual(len(ui.build_stats(None, None)), 1)
        self.assertGreaterEqual(len(ui.build_event_log(None, None)), 1)
        # 视口/布局/状态机的异常输入
        bad = ui.Viewport(0, 0, 0, 0)
        self.assertEqual(len(bad.screen_to_world(1, 2)), 2)
        self.assertEqual(len(bad.zoom_at(0, 0, float("nan")).world_to_screen(1, 1)), 2)
        self.assertEqual(len(ui.Layout(0, 0, panel_visible=True).rects()), 4)
        self.assertIsNone(ui.HitTester(bad, max_distance_px=float("nan")).hit([], 1, 2))
        control = ui.ControlState(speed=float("nan"))
        self.assertEqual(control.speed, 1.0)
        self.assertFalse(control.set_speed("not-a-speed"))
        self.assertEqual(control.nudge_speed(0), control.speed)

    def test_ui_state_does_not_use_tkinter(self) -> None:
        source = Path(ui.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import tkinter", source)
        self.assertNotIn("from tkinter", source)


if __name__ == "__main__":
    unittest.main()
