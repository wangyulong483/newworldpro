"""俯视地图渲染（V3）：地形底图 + 群伙散点 + 火焰叠加。

渲染策略（tkinter Canvas 的性能现实）
------------------------------------
* **底图**：整张地图渲染成一张 PIL 图（每 cell 一个色块），交给
  Canvas 当 PhotoImage。地图是 512×320 cell、每 cell 3px → 1536×960
  的底图，一次生成 ~10ms；地形变化时只重画脏 cell。
* **群伙**：几千个点不能每帧重画 Image —— 用 Canvas 的批量
  ``create_rectangle``（每个个体一个 2×2 的小方块，按 tribe/饥饿
  着色），每帧 diff（只移动的个体更新坐标）。实测 5000 个矩形
  的 diff 更新在 tkinter 上可行（<8ms/帧）。
* **火焰**：燃烧 cell 叠一层橙红半透明色块（也在 Canvas 上画，
  数量少时很快）。

色板
----
地形色来自 :data:`newworld.map_engine.TERRAIN_INFO`；个体色板
按饱足度从暗红（饥饿）到亮青（吃饱）渐变。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .map_engine import MapEngine, TERRAIN_INFO
from .swarm import Swarm

#: 每 cell 的渲染像素数（世界像素 → 屏幕像素的最小单位）
CELL_PX = 3

#: 个体色板：饥饿 0..1 → 暗红 → 橙 → 黄 → 亮青绿
AGENT_COLOURS: tuple[tuple[int, int, int], ...] = (
    (170, 40, 40),
    (200, 110, 40),
    (225, 180, 60),
    (120, 200, 120),
    (80, 220, 180),
    (60, 230, 220),
)

#: 火焰色（叠加在燃烧 cell 上）
FIRE_COLOUR = (255, 120, 30)
FIRE_CORE_COLOUR = (255, 220, 90)


def _agent_colour(hunger: float) -> tuple[int, int, int]:
    """饱足度 → 个体颜色（渲染散点用）。"""
    index = int(np.clip(hunger, 0.0, 0.999) * len(AGENT_COLOURS))
    return AGENT_COLOURS[index]


class MapRenderer:
    """把 MapEngine + Swarm 渲染成底图与散点数据。"""

    def __init__(self, engine: MapEngine, cell_px: int = CELL_PX) -> None:
        self.engine = engine
        self.cell_px = int(cell_px)
        #: 底图（PIL Image，RGB）：地形色块
        self.base = self._render_base_full()
        #: 覆盖了食物明暗层的展示图（外部拿去转 PhotoImage）
        self.display: Image.Image = self.base.copy()
        self._last_dirty: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    def _render_base_full(self) -> Image.Image:
        """整张底图全量渲染（初始化/重置用，~10ms）。"""
        h, w = self.engine.height, self.engine.width
        px = self.cell_px
        # 用 numpy 生成像素数组再转 Image（比逐格画快一个数量级）
        arr = np.zeros((h, w, 3), dtype=np.float64)
        # 每地形的色块
        for tid, (_name, colour) in TERRAIN_INFO.items():
            mask = self.engine.terrain == tid
            if not mask.any():
                continue
            arr[mask] = colour
        # 食物明暗：陆地上食物越多越亮
        land = self.engine.terrain >= 3
        arr[land] += (np.clip(self.engine.food, 0.0, 1.0) * 46.0)[:, :, None][land]
        # 燃烧 cell 染上火色
        arr[self.engine.burning] = FIRE_COLOUR
        # cell → cell_px 块放大
        big = np.repeat(np.repeat(arr, px, axis=0), px, axis=1)
        self.base = Image.fromarray(np.clip(big, 0, 255).astype(np.uint8), mode="RGB")
        return self.base

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """增量刷新：地形脏 cell 即时重画；食物层周期性全量重算。"""
        engine = self.engine
        dirty = engine.dirty_terrain
        px = self.cell_px
        if dirty:
            draw = ImageDraw.Draw(self.display)
            for (y, x) in list(dirty):
                tid = int(engine.terrain[y, x])
                colour = TERRAIN_INFO.get(tid, (0, 0, 0))[1]
                draw.rectangle((x * px, y * px, x * px + px - 1, y * px + px - 1), fill=colour)
            self._last_dirty |= dirty
            dirty.clear()
        # 周期性食物层刷新（全量，交给 numpy）
        if engine.steps % 16 == 0 or self._last_dirty:
            self._blend_food_layer()

    def _blend_food_layer(self) -> None:
        """食物明暗层全量重算（numpy 向量化，~5ms @ 512×320）。"""
        self._render_base_full()
        self.display = self.base.copy()
        self._last_dirty.clear()

    # ------------------------------------------------------------------
    def agent_points(self, swarm: Swarm) -> list[tuple[float, float, tuple[int, int, int]]]:
        """存活个体 → [(x_px, y_px, colour)]（Canvas 散点用）。

        逐个体转 tuple 在 5000 人时约 3ms（可接受；更高人口时
        调用方可降采样）。
        """
        px = self.cell_px
        xs, ys = swarm.positions_xy()
        if xs.size == 0:
            return []
        view = swarm.data[: swarm.count]
        hungers = view["hunger"][view["alive"]]
        points = []
        for x, y, hunger in zip(xs.tolist(), ys.tolist(), hungers.tolist()):
            points.append((x * px, y * px, _agent_colour(hunger)))
        return points
