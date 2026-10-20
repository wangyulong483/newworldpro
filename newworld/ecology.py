"""原子级破坏 / 生长与土壤水循环（生态模块）。

一切都发生在与世界其余部分相同的像素原子网格上：

* **破坏** —— 软原子（草、树叶、浆果）可以被踩掉，变成下落的碎屑，
  落地后消失；
* **生长** —— 浆果在浆果丛上重新结果、草在湿润土壤上长高，
  由阳光与土壤湿度驱动；
* **土壤水循环** —— 太阳蒸发池塘水面，降雨把水补回去；土壤湿度
  扩散并滋养植物，湿土壤颜色更暗。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from . import materials

#: 可被踩掉的软材料
SOFT = {"grass", "leaf", "berry"}
#: 踩不动的硬材料
HARD = {"stone", "wood", "soil", "water"}


@dataclass
class Debris:
    """一个在重力下坠落的脱落原子（视觉 + 物理双重存在）。"""

    x: float
    y: float
    vx: float
    vy: float
    material: str
    life: float = 4.0


class Ecology:
    """生态状态机：天气、土壤湿度、池塘水位、生长与破坏计数。

    由 :class:`newworld.pixelworld.PixelScene` 持有，世界每帧调用
    :meth:`update`（内部按 0.25 s 节流，不必每帧都跑全部规则）。
    """

    def __init__(self, scene) -> None:
        """从场景初始化生态状态（湿度/天气/水位）。"""
        self.rng = random.Random(2024)
        self.aw = scene.aw
        self.ah = scene.ah
        self.surface = scene.surface
        self.ground_row = scene.ground_row
        self.moisture = np.zeros((self.ah, self.aw), dtype=np.float32)
        self._init_moisture(scene)
        self.humidity = 0.42
        # 降雨强度 0..1，向 rain_target 平滑过渡；timer 控制旱/雨切换
        self.rain = 0.0
        self.rain_target = 0.0
        self.rain_timer = self.rng.uniform(20.0, 45.0)
        self.debris: list[Debris] = []
        self.destroyed = 0
        self.grown = 0
        self._accum = 0.0
        # 水位 = 当前水原子数 / 初始水原子数（池塘蒸干的度量）
        self._initial_water = int((scene.grid == materials.MATERIAL_INDEX["water"]).sum())
        self.water_level = 1.0

    # ------------------------------------------------------------------
    def _init_moisture(self, scene) -> None:
        """初始化土壤湿度：土壤/草地 0.38，池塘附近 +0.35，越深越湿。"""
        mask = (scene.grid == materials.MATERIAL_INDEX["soil"]) | (
            scene.grid == materials.MATERIAL_INDEX["grass"]
        )
        self.moisture[mask] = 0.38
        left, right = scene.pond
        for x in range(max(0, left - 10), min(self.aw, right + 11)):
            column = self.moisture[:, x]
            column[mask[:, x]] = np.minimum(1.0, column[mask[:, x]] + 0.35)
        # 深层土壤稍湿
        for y in range(self.surface, self.ah):
            depth = (y - self.surface) / max(1, self.ah - self.surface)
            self.moisture[y, :] = np.minimum(1.0, self.moisture[y, :] + 0.15 * depth)

    # ------------------------------------------------------------------
    def update(self, dt: float, scene, day, hour: float) -> None:
        """每帧推进生态；重规则按累积 0.25 s 节流执行。"""
        # 碎屑永远受重力（落地后淡出消失）
        remaining = []
        for particle in self.debris:
            particle.vy += 9.8 * 5.0 * dt  # 原子/s^2（每个原子约 5 cm）
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.life -= dt
            if (
                particle.life > 0.0
                and 0 <= particle.x < self.aw
                and particle.y < self.surface
            ):
                remaining.append(particle)
        self.debris = remaining

        self._accum += dt
        if self._accum < 0.25:
            return
        step = self._accum
        self._accum = 0.0
        daylight = day.daylight(hour)

        # -- 天气：定时在旱/雨之间切换，湿度随之浮动 --------------------
        self.rain_timer -= step
        if self.rain_timer <= 0.0:
            self.rain_target = 0.0 if self.rain_target > 0.1 else 0.85
            self.rain_timer = self.rng.uniform(24.0, 70.0)
        self.rain += (self.rain_target - self.rain) * min(1.0, step * 0.7)
        self.humidity = float(
            np.clip(0.30 + 0.55 * self.rain - 0.10 * daylight + self.rng.uniform(-0.02, 0.02), 0.0, 1.0)
        )

        changed = False
        # -- 池塘蒸发：白天随机蒸掉一个表面水原子 ----------------------
        if daylight > 0.12 and self.rng.random() < 0.35 + 0.3 * daylight:
            left, right = scene.pond
            candidates = [x for x in range(left, right + 1) if scene.grid[scene.pond_surface, x] == materials.MATERIAL_INDEX["water"]]
            if candidates:
                x = self.rng.choice(candidates)
                scene.clear_atom(x, scene.pond_surface)
                self.humidity = min(1.0, self.humidity + 0.01)
                changed = True

        # -- 降雨回填池塘 ----------------------------------------------
        if self.rain > 0.30 and self.rng.random() < 0.55:
            left, right = scene.pond
            candidates = [x for x in range(left, right + 1) if scene.grid[scene.pond_surface, x] == 0]
            if candidates:
                x = self.rng.choice(candidates)
                scene.set_atom(x, scene.pond_surface, "water")
                changed = True

        # -- 土壤湿度：向邻居扩散 + 表层受降雨/日晒增减 -----------------
        mask = (scene.grid == materials.MATERIAL_INDEX["soil"]) | (
            scene.grid == materials.MATERIAL_INDEX["grass"]
        )
        wet = self.moisture * mask
        average = (
            np.roll(wet, 1, axis=1)
            + np.roll(wet, -1, axis=1)
            + np.roll(wet, 1, axis=0)
            + np.roll(wet, -1, axis=0)
        ) / 4.0
        self.moisture = np.clip(
            self.moisture + (average - self.moisture) * 0.02,
            0.0,
            1.0,
        ).astype(np.float32)
        self.moisture *= mask
        surface_slice = self.moisture[self.surface, :]
        surface_slice = np.clip(
            surface_slice + (self.rain * 0.20 - daylight * 0.04) * step,
            0.0,
            1.0,
        )
        self.moisture[self.surface, :] = surface_slice

        # -- 生长 -------------------------------------------------------
        if self._grow(scene, daylight):
            changed = True
        if changed:
            scene.refresh_caches()
        water_now = int((scene.grid == materials.MATERIAL_INDEX["water"]).sum())
        self.water_level = water_now / max(1, self._initial_water)

    # ------------------------------------------------------------------
    def _grow(self, scene, daylight: float) -> bool:
        """植物生长：浆果再生 + 草长高；有任何变化返回 True。"""
        changed = False
        # 浆果丛在湿润、有阳光的日子里重新结果。
        #
        # P0 缺陷修复：原条件要求槽位原子仍是 ``leaf``，但吃掉浆果会把该
        # 原子清成 leaf 之外的值（``pick_berry``），于是丛只能产出初始
        # 浆果——文档宣称的"再生"规则永远不可达，有限食物终被吃绝。
        # 现在槽位是 leaf（新鲜丛）或空（被采过）都可以再结果。
        air = 0
        leaf = materials.MATERIAL_INDEX["leaf"]
        for obj in scene.objects:
            if getattr(obj, "kind", "") != "bush":
                continue
            for slot in getattr(obj, "berry_slots", ()):
                if slot in scene.berries:
                    continue
                x, y = slot
                if scene.grid[y, x] not in (leaf, air):
                    continue
                moisture = float(self.moisture[y, x])
                if moisture > 0.22 and daylight > 0.08 and self.rng.random() < 0.30:
                    if scene.add_berry(x, y):
                        obj.berries.append(slot)
                        self.grown += 1
                        changed = True
        # 潮湿的表层土壤上，草向上长高（最高 3 格）
        if self.rng.random() < 0.65:
            x = self.rng.randrange(1, self.aw - 1)
            if scene.grid[self.surface, x] == materials.MATERIAL_INDEX["grass"]:
                height = 0
                while (
                    self.surface - height - 1 >= 0
                    and scene.grid[self.surface - height - 1, x] == materials.MATERIAL_INDEX["grass"]
                ):
                    height += 1
                if height < 3 and self.moisture[self.surface, x] > 0.25:
                    scene.set_atom(x, self.surface - height - 1, "grass")
                    self.grown += 1
                    changed = True
        return changed

    # ------------------------------------------------------------------
    def trample(self, scene, x_screen: float, y_screen: float, force: float = 1.4) -> int:
        """踩掉屏幕坐标附近硬度低于 ``force`` 的软原子；返回踩掉的原子数。"""
        cx = int(x_screen / materials.ATOM_SIZE)
        cy = int(y_screen / materials.ATOM_SIZE)
        removed = 0
        for dy in range(-2, 2):
            for dx in range(-2, 3):
                x, y = cx + dx, cy + dy
                if not (0 <= x < self.aw and 0 <= y < self.ah):
                    continue
                material = materials.MATERIAL_BY_INDEX.get(int(scene.grid[y, x]))
                if material is None:
                    continue
                if material.key not in SOFT or material.hardness > force:
                    continue
                scene.clear_atom(x, y)
                if material.key == "berry" and (x, y) in scene.berries:
                    scene.berries.remove((x, y))
                # 被踩掉的原子变成一个向上弹起的碎屑粒子
                self.debris.append(
                    Debris(
                        x=float(x),
                        y=float(y),
                        vx=self.rng.uniform(-2.5, 2.5),
                        vy=self.rng.uniform(-7.0, -2.0),
                        material=material.key,
                    )
                )
                removed += 1
        if removed:
            self.destroyed += removed
            scene.refresh_caches()
        return removed

    # ------------------------------------------------------------------
    def summary_zh(self) -> str:
        """生态状态摘要（中文，UI 调试面板展示用）。"""
        wet_values = self.moisture[self.moisture > 0.0]
        wet = float(wet_values.mean()) if wet_values.size else 0.0
        return (
            f"土壤湿度 {wet:.2f}  空气湿度 {self.humidity:.2f}  "
            f"降雨 {self.rain:.2f}  水位 {self.water_level:.2f}  "
            f"破坏 {self.destroyed}  生长 {self.grown}"
        )
