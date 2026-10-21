"""俯视地图世界（V3）单元测试：map_engine / swarm / 清理契约。

测试哲学与本仓库一致：只断言**世界规则的机制正确性**（确定性、
守恒、地形变化来源）与**清理契约**（旧世界/上帝之手接口确实不存在），
不断言"文明兴盛"之类的结果。
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from newworld.map_engine import (
    DEEP_WATER,
    FOREST,
    GRASS,
    MapConfig,
    MapEngine,
    TERRAIN_INFO,
)
from newworld.swarm import Swarm, SwarmConfig


def make_engine(seed: int = 0, width: int = 128, height: int = 80) -> MapEngine:
    """小地图工厂（测试统一入口）。"""
    return MapEngine(MapConfig(seed=seed, width=width, height=height, noise_scale=32))


# ---------------------------------------------------------------------------
class MapEngineTest(unittest.TestCase):
    """地形生成 / 查询 / 再生。"""

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

    def test_water_is_not_walkable(self) -> None:
        """水域不可行走。"""
        engine = make_engine(seed=6)
        ys, xs = np.nonzero(engine.terrain == DEEP_WATER)
        self.assertTrue(ys.size > 0)
        self.assertFalse(engine.is_walkable(int(xs[0]), int(ys[0])))
        walkable = engine.walkable_mask()
        self.assertEqual(int(walkable.sum()), engine.land_cells())


# ---------------------------------------------------------------------------
class WorldIsReadOnlyTest(unittest.TestCase):
    """清理契约：地图没有上帝之手入口，地形只由森林蔓延自发变化。"""

    def test_no_god_hand_api(self) -> None:
        """旧版灾害/上帝之手接口必须不存在。"""
        engine = make_engine(seed=30)
        for name in ("ignite", "set_terrain", "burning", "burn_left",
                     "cells_burned", "terraformed"):
            self.assertFalse(hasattr(engine, name),
                             f"MapEngine 仍有上帝之手接口: {name}")

    def test_terrain_changes_only_via_forest_spread(self) -> None:
        """长时间推进后，地形变化只可能是 草地→森林（森林蔓延）。"""
        engine = MapEngine(MapConfig(seed=31, width=96, height=64,
                                     noise_scale=24, forest_spread_per_s=1.0))
        before = engine.terrain.copy()
        for _ in range(96):
            engine.update(0.05, 1.0)
        changed = np.nonzero(before != engine.terrain)
        self.assertGreater(changed[0].size, 0,
                           "森林蔓延没有发生，无法验证地形变化来源")
        for y, x in zip(changed[0].tolist(), changed[1].tolist()):
            self.assertEqual(int(before[y, x]), GRASS, f"({x},{y}) 非草地的地形被改写")
            self.assertEqual(int(engine.terrain[y, x]), FOREST, f"({x},{y}) 变化目标不是森林")
            self.assertIn((y, x), engine.dirty_terrain, f"({x},{y}) 未登记脏区")

    def test_legacy_modules_removed(self) -> None:
        """旧侧视世界模块不得再存在（包内不再有可导入入口）。"""
        for name in ("catastrophes", "ui_state", "world", "population",
                     "pixelworld", "society", "config"):
            self.assertIsNone(importlib.util.find_spec(f"newworld.{name}"),
                              f"newworld.{name} 仍然存在（旧世界残留）")


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


if __name__ == "__main__":
    unittest.main()
