"""InfiniteScene（无限世界窗口场景）测试。

覆盖契约 §3.4：确定性（同 seed 重叠区逐原子一致）、不同 seed 不同、
窗口平移后全局点材料一致、修改跨 recenter 持久、对象/浆果在重叠区一致、
池塘周期性存在、窗口关键属性、render_atoms 尺寸、update_ecology 可运行、
非法参数/极小窗口降级不崩，以及 recenter 实测性能。

所有用例都使用真世界配置 ``make_config(640, 400)``（不是假场景）。
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from newworld import make_config
from newworld.daynight import DayCycle
from newworld.ecology import Ecology
from newworld.infinite_scene import MIN_WINDOW_ATOMS, InfiniteScene, material_name_zh
from newworld.materials import ATOM_SIZE, MATERIAL_INDEX
from newworld.optics import Box, Disc, Scene

CONFIG = make_config(640, 400)

#: 池塘周期：每 2 个 64 原子的 chunk = 每 128 个原子一个池塘
POND_STEP = 2


def make_scene(seed: int = 0, *, window_atoms: int | None = None) -> InfiniteScene:
    """在真世界配置上建一个无限场景窗口。"""
    return InfiniteScene(CONFIG, seed, window_atoms=window_atoms)


def global_material_map(scene: InfiniteScene) -> dict[tuple[int, int], int]:
    """窗口内每个原子 → (全局原子 x, 原子 y) → 材料编号（含 0 = 空气）。"""
    start = scene.window_start_atom
    grid = scene.grid
    return {
        (start + x, y): int(grid[y, x])
        for y in range(scene.ah)
        for x in range(scene.aw)
    }


def global_berry_set(scene: InfiniteScene) -> set[tuple[int, int]]:
    """窗口内浆果的全局坐标集合。"""
    start = scene.window_start_atom
    return {(start + x, y) for x, y in scene.berries}


def object_key(obj) -> tuple:
    """物体的身份/外观关键字段（坐标已由调用方换算成全局）。"""
    return (obj.kind, obj.x0, obj.x1, obj.top, obj.bottom, obj.atom_count)


def object_key_of(scene: InfiniteScene) -> list[tuple]:
    """窗口内物体的关键字段列表（同窗口可直接比较）。"""
    return [object_key(obj) for obj in scene.objects]


def fully_inside(scene: InfiniteScene, obj) -> bool:
    """物体是否没有贴窗口左右边缘（即没有被裁剪，坐标是精确的）。"""
    return obj.x0 > 0 and obj.x1 < scene.aw - 1


class TestDeterminism(unittest.TestCase):
    """同 seed 一致、不同 seed 不同。"""

    def test_same_seed_is_atom_identical(self) -> None:
        first = make_scene(seed=3)
        second = make_scene(seed=3)
        self.assertTrue(np.array_equal(first.grid, second.grid))
        self.assertEqual(first.berries, second.berries)
        self.assertEqual(object_key_of(first), object_key_of(second))
        self.assertEqual(first.pond, second.pond)
        self.assertTrue(np.array_equal(first.ecology.moisture, second.ecology.moisture))

    def test_same_offset_via_different_paths(self) -> None:
        # 窗口最终落在同一个全局位置时，生成结果与怎么走到那里无关
        first = make_scene(seed=2)
        second = make_scene(seed=2)
        first.recenter(1_000_000)
        second.recenter(999_999)
        second.recenter(1)
        self.assertEqual(first.window_start_atom, second.window_start_atom)
        self.assertTrue(np.array_equal(first.grid, second.grid))
        self.assertEqual(global_berry_set(first), global_berry_set(second))

    def test_different_seed_differs(self) -> None:
        first = make_scene(seed=0)
        second = make_scene(seed=1)
        self.assertFalse(np.array_equal(first.grid, second.grid))
        self.assertNotEqual(global_material_map(first), global_material_map(second))
        # 池塘位置/宽度由 seed 决定，通常也不同
        self.assertNotEqual(
            (first._pond_layout(0), first._pond_layout(2)),
            (second._pond_layout(0), second._pond_layout(2)),
        )


class TestRecenterConsistency(unittest.TestCase):
    """窗口平移后，同一全局原子的材料/物体/浆果不变。"""

    def test_overlap_materials_stable_after_recenter(self) -> None:
        scene = make_scene(seed=11)
        before = global_material_map(scene)
        scene.recenter(37)
        after = global_material_map(scene)
        overlap = [key for key in after if 37 <= key[0] < scene.aw]
        self.assertGreater(len(overlap), 1000)
        mismatches = [key for key in overlap if before[key] != after[key]]
        self.assertEqual(mismatches, [])

    def test_overlap_stable_after_two_opposite_recenters(self) -> None:
        scene = make_scene(seed=12)
        before = global_material_map(scene)
        scene.recenter(37)
        scene.recenter(-100)  # 窗口起点 -63
        self.assertEqual(scene.window_start_atom, -63)
        after = global_material_map(scene)
        overlap = [key for key in after if 0 <= key[0] < scene.aw]
        self.assertGreater(len(overlap), 500)
        self.assertEqual([key for key in overlap if before[key] != after[key]], [])

    def test_two_scenes_overlap_same_seed(self) -> None:
        left = make_scene(seed=7)
        right = make_scene(seed=7)
        right.recenter(64)
        left_map = global_material_map(left)
        right_map = global_material_map(right)
        overlap = [key for key in right_map if 64 <= key[0] < left.aw]
        self.assertGreater(len(overlap), 1000)
        self.assertEqual(
            [key for key in overlap if left_map[key] != right_map[key]], []
        )

    def test_overlap_across_different_window_widths(self) -> None:
        # 窗口宽度不同也不影响重叠区域的逐原子结果（全局确定性）
        wide = make_scene(seed=3, window_atoms=200)
        narrow = make_scene(seed=3, window_atoms=100)
        wide_map = global_material_map(wide)
        narrow_map = global_material_map(narrow)
        self.assertEqual(len(narrow_map), narrow.aw * narrow.ah)
        mismatches = [key for key in narrow_map if wide_map[key] != narrow_map[key]]
        self.assertEqual(mismatches, [])

    def test_objects_and_berries_consistent_in_overlap(self) -> None:
        left = make_scene(seed=5)
        right = make_scene(seed=5)
        right.recenter(80)

        left_berries = {(x, y) for x, y in global_berry_set(left) if 80 <= x < 256}
        right_berries = {(x, y) for x, y in global_berry_set(right) if 80 <= x < 256}
        self.assertTrue(left_berries)  # 这个窗口确实有浆果，比较才有意义
        self.assertEqual(left_berries, right_berries)

        left_objects = {
            object_key(obj) for obj in left.objects if fully_inside(left, obj)
        }
        right_objects = {
            (
                obj.kind,
                80 + obj.x0,
                80 + obj.x1,
                obj.top,
                obj.bottom,
                obj.atom_count,
            )
            for obj in right.objects
            if fully_inside(right, obj)
        }
        # 只比较两边都完整落在重叠区（全局 81..254）里的物体
        left_common = {key for key in left_objects if 81 <= key[1] and key[2] <= 254}
        right_common = {key for key in right_objects if 81 <= key[1] and key[2] <= 254}
        self.assertTrue(left_common)
        self.assertEqual(left_common, right_common)


class TestPersistence(unittest.TestCase):
    """修改、破坏、采果都按全局坐标持久化，recenter 后仍然存在。"""

    def test_set_and_clear_atom_survive_recenter(self) -> None:
        scene = make_scene(seed=2)
        x, y = 100, scene.surface
        gx = scene.global_atom_x(x)
        self.assertEqual(scene.material_at(x, y).key, "grass")
        scene.set_atom(x, y, "wood")
        scene.clear_atom(x + 5, y)
        self.assertEqual(scene.material_at(x, y).key, "wood")
        self.assertFalse(scene.solid_at(x + 5, y))

        scene.recenter(150)
        # 全局坐标 gx 已落在窗口左侧之外（局部 x = -50）
        self.assertEqual(scene.local_atom_x(gx), -50)
        scene.recenter(-150)
        self.assertEqual(scene.window_start_atom, 0)
        self.assertEqual(scene.material_at(x, y).key, "wood")
        self.assertFalse(scene.solid_at(x + 5, y))
        # 与 PixelScene 的实际行为一致：0 号格子是 air 材料，不是 None
        self.assertEqual(scene.atom_material(x + 5, y).key, "air")

    def test_trampled_atom_survives_recenter(self) -> None:
        scene = make_scene(seed=5)
        ys, xs = np.where(scene.grid == MATERIAL_INDEX["grass"])
        self.assertGreater(xs.size, 0)
        x, y = int(xs[0]), int(ys[0])
        gx = scene.global_atom_x(x)
        removed = scene.trample((x + 0.5) * ATOM_SIZE, (y + 0.5) * ATOM_SIZE, force=2.0)
        self.assertGreaterEqual(removed, 1)
        self.assertFalse(scene.solid_at(x, y))
        scene.recenter(200)
        scene.recenter(-200)
        self.assertEqual(scene.global_atom_x(x), gx)
        self.assertFalse(scene.solid_at(x, y))

    def test_picked_berry_stays_picked(self) -> None:
        scene = make_scene(seed=4)
        berries = [berry for berry in scene.berries if 20 <= berry[0] <= 230]
        self.assertTrue(berries)
        x, y = berries[0]
        gx = scene.global_atom_x(x)
        self.assertEqual(scene.material_at(x, y).key, "berry")
        self.assertTrue(scene.pick_berry((x, y)))
        self.assertFalse(scene.pick_berry((x, y)))  # 再摘一次失败
        self.assertEqual(scene.material_at(x, y).key, "leaf")

        scene.recenter(300)
        scene.recenter(-300)
        self.assertEqual(scene.window_start_atom, 0)
        local_x = scene.local_atom_x(gx)
        self.assertEqual(scene.material_at(local_x, y).key, "leaf")
        self.assertNotIn((local_x, y), scene.berries)

    def test_edit_persists_when_visible_again(self) -> None:
        scene = make_scene(seed=9, window_atoms=128)
        x, y = 60, scene.surface
        scene.set_atom(x, y, "stone")
        scene.recenter(5000)
        scene.recenter(-5000)
        self.assertEqual(scene.window_start_atom, 0)
        self.assertEqual(scene.material_at(x, y).key, "stone")


class TestPond(unittest.TestCase):
    """池塘周期性存在，且窗口里的 pond 属性合法。"""

    def test_every_default_window_has_water(self) -> None:
        for start in (0, 1, 63, 127, 128, 129, 200, 511, 1000, -50, -128):
            scene = make_scene(seed=8)
            scene.recenter(start)
            water = int((scene.grid == MATERIAL_INDEX["water"]).sum())
            self.assertGreater(water, 0, f"窗口 start={start} 没有水")
            left, right = scene.pond
            self.assertGreaterEqual(left, 0)
            self.assertLessEqual(right, scene.aw - 1)
            self.assertLessEqual(left, right)
            self.assertEqual(scene.pond_surface, scene.surface)

    def test_pond_layout_periodic_and_in_bounds(self) -> None:
        scene = make_scene(seed=6)
        for chunk in range(0, 20, POND_STEP):
            left, width = scene._pond_layout(chunk)
            self.assertGreaterEqual(width, 18)
            self.assertLessEqual(width, 30)
            chunk_start = chunk * 64
            self.assertGreaterEqual(left, chunk_start + 6)
            self.assertLessEqual(left + width, chunk_start + 64 - 6)
        # 相邻 chunk 的池塘参数由 hash 决定 → 通常不同（不是固定模板）
        self.assertNotEqual(scene._pond_layout(0), scene._pond_layout(2))

    def test_pond_available_near_water_query(self) -> None:
        scene = make_scene(seed=8)
        ys, xs = np.where(scene.grid == MATERIAL_INDEX["water"])
        self.assertGreater(xs.size, 0)
        x, y = int(xs[0]), int(ys[0])
        self.assertTrue(scene.water_near(x, y))
        self.assertTrue(scene.water_near(x, y, radius_atoms=0.0))


class TestWindowGeometry(unittest.TestCase):
    """窗口关键属性与坐标换算。"""

    def test_default_geometry(self) -> None:
        scene = make_scene(seed=0)
        self.assertEqual(scene.aw, 2 * (CONFIG.width // ATOM_SIZE))
        self.assertEqual(scene.aw, 256)
        self.assertEqual(scene.ah, CONFIG.height // ATOM_SIZE)
        self.assertEqual(scene.ah, 80)
        self.assertEqual(scene.ground_row, int(round(CONFIG.ground_y / ATOM_SIZE)))
        self.assertEqual(scene.surface, scene.ground_row - 1)
        self.assertEqual(scene.grid.shape, (scene.ah, scene.aw))
        self.assertEqual(scene.grid.dtype, np.uint8)
        self.assertEqual(scene.terrain_bottom, scene.ah - 1)
        self.assertIsInstance(scene.ecology, Ecology)
        self.assertEqual(scene.window_start_atom, 0)
        self.assertEqual(scene.window_start_px, 0.0)
        self.assertEqual(scene.window_end_atom, scene.aw)

    def test_custom_window_and_coordinate_helpers(self) -> None:
        scene = make_scene(seed=0, window_atoms=100)
        self.assertEqual(scene.aw, 100)
        scene.recenter(40)
        self.assertEqual(scene.window_start_atom, 40)
        self.assertEqual(scene.window_start_px, 40 * ATOM_SIZE)
        self.assertEqual(scene.global_atom_x(5), 45)
        self.assertEqual(scene.local_atom_x(45), 5)
        self.assertEqual(scene.local_atom_x(0), -40)  # 窗口外是合法值

    def test_ecology_state_shape_matches_window(self) -> None:
        scene = make_scene(seed=1)
        self.assertEqual(scene.ecology.moisture.shape, (scene.ah, scene.aw))
        self.assertEqual(scene.ecology.aw, scene.aw)
        self.assertEqual(scene.ecology.ah, scene.ah)
        self.assertEqual(scene.ecology.surface, scene.surface)


class TestApiSurface(unittest.TestCase):
    """契约 §3.2 列出的窗口化 API 必须全部存在且可用。"""

    WINDOW_ATTRS = (
        "aw", "ah", "surface", "ground_row", "pond", "pond_surface",
        "objects", "berries", "grid", "ecology", "window_start_atom",
        "window_start_px",
    )
    METHODS = (
        "global_atom_x", "local_atom_x", "recenter",
        "set_atom", "clear_atom", "atom_material", "refresh_caches",
        "add_berry", "pick_berry", "water_near", "berry_at", "solid_at",
        "material_at", "is_blocked", "surface_screen_y", "refresh_lighting",
        "update_ecology", "trample", "render_atoms", "optics_scene",
        "material_composition_text", "material_name_zh",
    )

    def test_contract_api_present(self) -> None:
        scene = make_scene(seed=0)
        for name in self.WINDOW_ATTRS:
            self.assertTrue(hasattr(scene, name), f"缺少属性 {name}")
        for name in self.METHODS:
            self.assertTrue(callable(getattr(scene, name, None)), f"缺少方法 {name}")
        self.assertIsInstance(scene.aw, int)
        self.assertIsInstance(scene.ah, int)
        self.assertIsInstance(scene.surface, int)
        self.assertIsInstance(scene.ground_row, int)
        self.assertEqual(len(scene.pond), 2)
        self.assertIsInstance(scene.pond_surface, int)
        self.assertIsInstance(scene.objects, list)
        self.assertIsInstance(scene.berries, list)
        self.assertIsInstance(scene.grid, np.ndarray)
        self.assertIsInstance(scene.ecology, Ecology)


class TestRenderingAndEcology(unittest.TestCase):
    """render_atoms 尺寸、optics_scene、生态推进。"""

    def test_render_atoms_shape_and_day_night(self) -> None:
        scene = make_scene(seed=2)
        day = DayCycle()
        noon = scene.render_atoms(day, 12.0)
        self.assertEqual(noon.shape, (scene.ah, scene.aw, 3))
        self.assertEqual(noon.dtype, np.uint8)
        midnight = scene.render_atoms(day, 0.0)
        self.assertEqual(midnight.shape, noon.shape)
        self.assertGreater(
            int(np.abs(noon.astype(np.int16) - midnight.astype(np.int16)).sum()), 0
        )
        self.assertGreater(int(noon.max()), 0)
        # 下雨分支（阴影/雨点绘制）也要能跑
        scene.ecology.rain = 0.6
        rainy = scene.render_atoms(day, 18.0)
        self.assertEqual(rainy.shape, (scene.ah, scene.aw, 3))
        scene.ecology.rain = 0.0

    def test_optics_scene_uses_local_geometry(self) -> None:
        scene = make_scene(seed=3)
        optics = scene.optics_scene()
        self.assertIsInstance(optics, Scene)
        self.assertGreater(len(optics.objects), 0)
        self.assertTrue(any(isinstance(shape, Box) for shape in optics.objects))
        self.assertTrue(any(isinstance(shape, Disc) for shape in optics.objects))
        self.assertIs(scene.optics_scene(), optics)  # 缓存
        scene.set_atom(1, scene.surface - 1, "wood")
        self.assertIsNot(scene.optics_scene(), optics)  # 修改后失效

    def test_update_ecology_runs(self) -> None:
        scene = make_scene(seed=6)
        before = scene.ecology.moisture.copy()
        day = DayCycle()
        for step in range(40):
            scene.update_ecology(1.0 / 60.0, day, 12.0 + step * 0.02)
        self.assertEqual(scene.ecology.moisture.shape, (scene.ah, scene.aw))
        self.assertTrue(np.isfinite(scene.ecology.water_level))
        self.assertGreaterEqual(scene.ecology.rain, 0.0)
        self.assertLessEqual(scene.ecology.rain, 1.0)
        self.assertFalse(np.array_equal(before, scene.ecology.moisture))

    def test_ecology_survives_recenter(self) -> None:
        scene = make_scene(seed=6)
        day = DayCycle()
        for _ in range(10):
            scene.update_ecology(0.1, day, 10.0)
        rain_before = scene.ecology.rain
        scene.recenter(128)
        self.assertEqual(scene.ecology.moisture.shape, (scene.ah, scene.aw))
        self.assertEqual(scene.ecology.water_level, 1.0)
        self.assertAlmostEqual(scene.ecology.rain, rain_before, places=6)
        for _ in range(10):
            scene.update_ecology(0.1, day, 10.0)
        self.assertTrue(np.isfinite(scene.ecology.water_level))


class TestHelpersAndDegradation(unittest.TestCase):
    """碰撞/查询辅助、非法参数与极小窗口降级。"""

    def test_surface_and_collision_helpers(self) -> None:
        scene = make_scene(seed=0)
        x = 7
        expected = float((scene.top_row[x] + 1) * ATOM_SIZE)
        self.assertEqual(scene.surface_screen_y(x * ATOM_SIZE + 1.0), expected)
        self.assertEqual(expected % ATOM_SIZE, 0.0)

        ys, xs = np.where(scene.grid == MATERIAL_INDEX["wood"])
        self.assertGreater(xs.size, 0)
        wx, wy = int(xs[0]), int(ys[0])
        self.assertTrue(
            scene.is_blocked(
                (wx + 0.5) * ATOM_SIZE,
                ATOM_SIZE * 0.8,
                (wy + 1.0) * ATOM_SIZE,
                ATOM_SIZE * 0.8,
            )
        )
        self.assertFalse(scene.solid_at(-1, 0))
        self.assertIsNone(scene.material_at(scene.aw, 0))
        self.assertFalse(scene.water_near(-100, -100))
        # 地平线以下（仰角 <= 1°）的阴影清空分支
        scene.refresh_lighting(DayCycle(), 12.0, 0.0)
        self.assertFalse(scene.shadow_ground.any())

    def test_berry_helpers(self) -> None:
        scene = make_scene(seed=4)
        self.assertTrue(scene.berries)
        berry = scene.berries[0]
        self.assertEqual(scene.berry_at(berry[0], berry[1]), berry)
        self.assertIsNone(scene.berry_at(-100, -100))

        # 在树叶上可以结浆果，在土壤/石头上不行
        ys, xs = np.where(scene.grid == MATERIAL_INDEX["leaf"])
        self.assertGreater(xs.size, 0)
        lx, ly = int(xs[0]), int(ys[0])
        self.assertTrue(scene.add_berry(lx, ly))
        self.assertEqual(scene.material_at(lx, ly).key, "berry")
        ys, xs = np.where(scene.grid == MATERIAL_INDEX["soil"])
        self.assertGreater(xs.size, 0)
        self.assertFalse(scene.add_berry(int(xs[0]), int(ys[0])))
        self.assertFalse(scene.add_berry(-1, 0))

    def test_invalid_recenter_deltas_do_not_crash(self) -> None:
        scene = make_scene(seed=0)
        scene.recenter(0)
        scene.recenter(None)  # 非法 → 按 0
        self.assertEqual(scene.window_start_atom, 0)
        scene.recenter(2.9)  # 浮点 → 2
        self.assertEqual(scene.window_start_atom, 2)
        scene.recenter(-10 ** 9)
        self.assertLess(scene.window_start_atom, 0)
        scene.recenter(10 ** 9)
        self.assertGreater(scene.window_start_atom, 0)
        self.assertEqual(scene.grid.shape, (scene.ah, scene.aw))
        self.assertEqual(scene.ecology.moisture.shape, (scene.ah, scene.aw))

    def test_tiny_window_degrades_gracefully(self) -> None:
        tiny = make_scene(seed=0, window_atoms=1)
        self.assertEqual(tiny.aw, MIN_WINDOW_ATOMS)
        self.assertEqual(tiny.grid.shape, (tiny.ah, tiny.aw))
        left, right = tiny.pond
        self.assertLessEqual(left, right)
        self.assertLess(right, tiny.aw)
        tiny.recenter(7)
        tiny.recenter(-31)
        tiny.update_ecology(0.5, DayCycle(), 12.0)
        self.assertFalse(np.array_equal(tiny.grid, np.zeros_like(tiny.grid)))
        self.assertEqual(tiny.render_atoms(DayCycle(), 12.0).shape, (tiny.ah, tiny.aw, 3))

    def test_material_text_helpers(self) -> None:
        scene = make_scene(seed=0)
        text = scene.material_composition_text()
        self.assertTrue(text)
        self.assertIn("草", text)
        self.assertEqual(scene.material_name_zh("water"), "水")
        self.assertEqual(material_name_zh("stone"), "石头")


class TestPerformance(unittest.TestCase):
    """recenter 实测性能（契约目标 ≤ 30 ms/次；这里放宽上限避免 CI 抖动）。"""

    def test_recenter_performance(self) -> None:
        scene = make_scene(seed=0)
        scene.recenter(40)  # 预热
        count = 20
        started = time.perf_counter()
        for _ in range(count):
            scene.recenter(40)
        per_call_ms = (time.perf_counter() - started) / count * 1000.0
        print(f"\n[bench] recenter: {per_call_ms:.2f} ms/call (default 256x80 window)")
        self.assertLess(per_call_ms, 150.0)


if __name__ == "__main__":
    unittest.main()
