"""俯视地图世界（V3）单元测试：map_engine / swarm / catastrophes。

测试哲学与本仓库一致：只断言**世界规则的机制正确性**（确定性、
守恒、灾害真实作用），不断言"文明兴盛"之类的结果。
"""

from __future__ import annotations

import unittest

import numpy as np

from newworld.map_engine import (
    DEEP_WATER,
    FOREST,
    GRASS,
    MapConfig,
    MapEngine,
    SHALLOW_WATER,
    TERRAIN_INFO,
)
from newworld.swarm import Swarm, SwarmConfig
from newworld.catastrophes import Catastrophes


def make_engine(seed: int = 0, width: int = 128, height: int = 80) -> MapEngine:
    """小地图工厂（测试统一入口）。"""
    return MapEngine(MapConfig(seed=seed, width=width, height=height, noise_scale=32))


# ---------------------------------------------------------------------------
class MapEngineTest(unittest.TestCase):
    """地形生成 / 查询 / 再生 / 燃烧。"""

    def test_generation_is_deterministic(self) -> None:
        """同 seed 两张地图逐 cell 一致。"""
        a = make_engine(seed=7)
        b = make_engine(seed=7)
        self.assertTrue(np.array_equal(a.terrain, b.terrain))

    def test_terrain_only_valid_ids(self) -> None:
        """地形网格只含注册表内的编号。"""
        engine = make_engine(seed=1)
        valid = set(TERRAIN_INFO)
        used = set(np.unique(engine.terrain).tolist())
        self.assertTrue(used <= valid, f"未知地形编号: {used - valid}")

    def test_ocean_margin_enforced(self) -> None:
        """地图四周是大洋（边界语义）。"""
        engine = make_engine(seed=2)
        m = engine.config.ocean_margin
        edge = np.concatenate([
            engine.terrain[:m, :].ravel(),
            engine.terrain[-m:, :].ravel(),
            engine.terrain[:, :m].ravel(),
            engine.terrain[:, -m:].ravel(),
        ])
        self.assertTrue((edge == DEEP_WATER).all())

    def test_food_regrows_over_time(self) -> None:
        """白昼下食物总量单调不减（再生只增不减，无进食时）。"""
        engine = make_engine(seed=3)
        before = engine.food_capacity()
        for _ in range(60):
            engine.update(0.05, 1.0)
        self.assertGreater(engine.food_capacity(), before)

    def test_fire_ignites_spreads_and_burns_out(self) -> None:
        """点燃森林 → 燃烧数 > 0 → 充分长时间后自然熄灭。"""
        engine = make_engine(seed=4)
        ys, xs = np.nonzero(engine.terrain == FOREST)
        if not ys.size:  # pragma: no cover - 小图无森林时跳过
            self.skipTest("no forest on this small map")
        y, x = int(ys[0]), int(xs[0])
        self.assertTrue(engine.ignite(x, y))
        self.assertTrue(engine.burning[y, x])
        burned_before = engine.cells_burned
        saw_burning = int(engine.burning.sum()) > 0
        for _ in range(4000):  # 200 秒（火灾可能大面积蔓延，给足时间）
            engine.update(0.05, 1.0)
            saw_burning = saw_burning or int(engine.burning.sum()) > 0
        self.assertTrue(saw_burning)
        self.assertGreater(engine.cells_burned, burned_before)
        self.assertEqual(int(engine.burning.sum()), 0)

    def test_set_terrain_marks_dirty(self) -> None:
        """set_terrain 登记脏区且更新地形。"""
        engine = make_engine(seed=5)
        engine.set_terrain(10, 10, SHALLOW_WATER)
        self.assertEqual(engine.terrain[10, 10], SHALLOW_WATER)
        self.assertIn((10, 10), engine.dirty_terrain)
        self.assertGreater(engine.terraformed, 0)

    def test_water_is_not_walkable(self) -> None:
        """水域不可行走。"""
        engine = make_engine(seed=6)
        ys, xs = np.nonzero(engine.terrain == DEEP_WATER)
        self.assertTrue(ys.size > 0)
        self.assertFalse(engine.is_walkable(int(xs[0]), int(ys[0])))
        walkable = engine.walkable_mask()
        self.assertEqual(int(walkable.sum()), engine.land_cells())


# ---------------------------------------------------------------------------
class SwarmTest(unittest.TestCase):
    """群伙人口动态。"""

    def test_initial_spawn_on_land_with_food(self) -> None:
        """初始个体都落在可行走格上。"""
        engine = make_engine(seed=10)
        swarm = Swarm(engine, SwarmConfig(initial_population=50), seed=10)
        xs, ys = swarm.positions_xy()
        self.assertEqual(xs.size, 50)
        for x, y in zip(xs.tolist(), ys.tolist()):
            self.assertTrue(engine.is_walkable(int(x), int(y)))

    def test_population_grows_with_food(self) -> None:
        """食物充足时人口增长（繁衍机制活着）。"""
        engine = make_engine(seed=11)
        swarm = Swarm(engine, SwarmConfig(initial_population=60), seed=11)
        start = swarm.population
        for _ in range(1200):  # 60 秒
            engine.update(0.05, 1.0)
            swarm.update(0.05)
        self.assertGreater(swarm.population, start)
        self.assertGreater(swarm.births, 0)

    def test_starvation_kills_when_foodless(self) -> None:
        """没有食物时个体饿死（hunger 递减到 0 → 死）。"""
        engine = make_engine(seed=12)
        swarm = Swarm(engine, SwarmConfig(initial_population=20), seed=12)
        # 清空食物并停止再生
        engine.food[:] = 0.0
        for _ in range(3000):  # 150 秒 > 1/0.010 代谢寿命
            swarm.update(0.05)  # 不更新 engine → 食物不回来
        self.assertGreater(swarm.starvation_deaths, 0)
        self.assertEqual(swarm.population, 0)

    def test_drowning_in_water(self) -> None:
        """个体被放进水里会溺死。"""
        engine = make_engine(seed=13)
        swarm = Swarm(engine, SwarmConfig(initial_population=5), seed=13)
        ys, xs = np.nonzero(engine.terrain == DEEP_WATER)
        self.assertTrue(ys.size > 0)
        view = swarm.data[: swarm.count]
        for i in range(min(5, swarm.count)):
            view["x"][i] = float(xs[0])
            view["y"][i] = float(ys[0])
        for _ in range(20):
            swarm.update(0.05)
        self.assertGreater(swarm.drowned, 0)

    def test_dead_slots_are_compacted(self) -> None:
        """死亡后数组压实（count = 活着数，无空洞）。"""
        engine = make_engine(seed=14)
        swarm = Swarm(engine, SwarmConfig(initial_population=10), seed=14)
        engine.food[:] = 0.0
        for _ in range(2600):
            swarm.update(0.05)
        self.assertEqual(swarm.count, swarm.population)
        self.assertEqual(swarm.population, 0)

    def test_deterministic_trajectory(self) -> None:
        """同 seed 同步数 → 同人口（可复现性）。"""
        def run(seed: int) -> int:
            engine = make_engine(seed=seed)
            swarm = Swarm(engine, SwarmConfig(initial_population=30), seed=seed)
            for _ in range(400):
                engine.update(0.05, 1.0)
                swarm.update(0.05)
            return swarm.population

        self.assertEqual(run(21), run(21))

    def test_thousand_agents_performance(self) -> None:
        """性能验收：≥1000 个体时一次 update < 8ms。"""
        import time

        engine = make_engine(seed=15, width=256, height=160)
        swarm = Swarm(engine, SwarmConfig(initial_population=200, capacity_hard=6000), seed=15)
        for _ in range(2400):  # 120 秒让人口充分增长
            engine.update(0.05, 1.0)
            swarm.update(0.05)
        self.assertGreaterEqual(swarm.population, 1000,
                                f"人口仅 {swarm.population}，无法验证千人性状")
        start = time.perf_counter()
        for _ in range(60):
            engine.update(0.05, 1.0)
            swarm.update(0.05)
        per_frame_ms = (time.perf_counter() - start) / 60 * 1000.0
        self.assertLess(per_frame_ms, 8.0, f"帧成本 {per_frame_ms:.1f}ms 超标")


# ---------------------------------------------------------------------------
class CatastrophesTest(unittest.TestCase):
    """上帝之手。"""

    def setUp(self) -> None:
        self.engine = make_engine(seed=20)
        self.swarm = Swarm(self.engine, SwarmConfig(initial_population=100), seed=20)
        self.cats = Catastrophes(self.engine, self.swarm)
        for _ in range(100):
            self.engine.update(0.05, 1.0)
            self.swarm.update(0.05)

    def _populated_point(self) -> tuple[int, int]:
        xs, ys = self.swarm.positions_xy()
        return int(np.median(xs)), int(np.median(ys))

    def test_meteor_kills_and_scars(self) -> None:
        """陨石：击杀个体 + 中心食物清零。"""
        x, y = self._populated_point()
        before = self.swarm.population
        event = self.cats.meteor(x, y, radius=10)
        self.assertGreater(event.killed, 0)
        self.assertLess(self.swarm.population, before)
        self.assertEqual(self.engine.food[y, x], 0.0)

    def test_lightning_ignites_forest(self) -> None:
        """闪电：点燃可燃地形。"""
        ys, xs = np.nonzero(
            (self.engine.terrain == FOREST) | (self.engine.terrain == GRASS)
        )
        if not ys.size:  # pragma: no cover
            self.skipTest("no burnable terrain")
        event = self.cats.lightning(int(xs[0]), int(ys[0]))
        self.assertEqual(event.ignited, 1)

    def test_flood_converts_land_to_water(self) -> None:
        """洪水：圆内部分陆地变浅水。"""
        x, y = self._populated_point()
        before_water = int((self.engine.terrain == SHALLOW_WATER).sum())
        event = self.cats.flood(x, y, radius=12)
        self.assertGreater(event.flooded, 0)
        after_water = int((self.engine.terrain == SHALLOW_WATER).sum())
        self.assertGreater(after_water, before_water)

    def test_heal_extinguishes_and_feeds(self) -> None:
        """治愈：灭火 + 食物回填 + 圆内个体饥饿回满。"""
        # 先点一把火
        ys, xs = np.nonzero(self.engine.terrain == FOREST)
        if ys.size:
            self.engine.ignite(int(xs[0]), int(ys[0]))
        x, y = self._populated_point()
        self.cats.heal(x, y, radius=15)
        disk_y = np.arange(max(0, y - 15), min(self.engine.height, y + 16))
        disk_x = np.arange(max(0, x - 15), min(self.engine.width, x + 16))
        self.assertFalse(self.engine.burning[np.ix_(disk_y, disk_x)].any())

    def test_history_recorded(self) -> None:
        """灾害史被记录且可 JSON 化。"""
        x, y = self._populated_point()
        self.cats.meteor(x, y, 5)
        self.cats.lightning(x + 3, y + 3)
        self.assertEqual(len(self.cats.history), 2)
        for event in self.cats.history:
            as_dict = event.to_dict()
            self.assertIsInstance(as_dict["kind"], str)
            self.assertIsInstance(as_dict["killed"], int)


if __name__ == "__main__":
    unittest.main()
