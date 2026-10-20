"""灾难与神迹（V3）：陨石、闪电火灾、洪水、治愈之雨。

世界盒子风格的"上帝之手"：玩家/实验脚本通过 :class:`Catastrophes`
对俯视地图施加区域性灾害。所有灾害都作用于**真实世界状态**——
地形网格、食物网格、燃烧通道、群伙个体的生死——不是视觉效果。

灾害清单
--------
* **陨石**（meteor）：圆形撞击坑——中心地形变焦土（山地/重新分层）、
  食物清零、范围内的个体直接死亡；边缘有几率点燃（火灾）。
* **闪电**（lightning）：单点点燃可燃地形（森林/草地），火按
  ``MapEngine.update`` 的蔓延规则扩散。
* **洪水**（flood）：圆形区域内低海拔陆地被淹为浅水（个体会被
  淹死——由 Swarm 的水域死亡判定处理）。
* **治愈之雨**（heal）：区域内灭火 + 食物瞬间回填到地形上限、
  群伙饥饿恢复（仁慈的神迹，也为实验提供"救援"对照）。

所有事件都进 :attr:`Catastrophes.history`（可审计、可写 JSONL）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .map_engine import (
    DEEP_WATER,
    FOREST,
    GRASS,
    SAND,
    SHALLOW_WATER,
    FOOD_TERRAIN,
    MapEngine,
)
from .swarm import Swarm


@dataclass
class CatastropheEvent:
    """一次灾害记录（可 JSON 化）。"""

    kind: str
    x: int
    y: int
    radius: int
    t: float
    killed: int = 0
    ignited: int = 0
    flooded: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
            "t": round(self.t, 2),
            "killed": self.killed,
            "ignited": self.ignited,
            "flooded": self.flooded,
        }


@dataclass
class Catastrophes:
    """上帝之手的执行器（挂在 MapEngine + Swarm 之上）。"""

    engine: MapEngine
    swarm: Swarm
    history: list[CatastropheEvent] = field(default_factory=list)
    history_limit: int = 512

    def _record(self, event: CatastropheEvent) -> None:
        self.history.append(event)
        if len(self.history) > self.history_limit:
            del self.history[: self.history_limit // 2]

    def _disk(self, x: int, y: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
        """圆形区域的 (ys, xs) 索引（裁剪到地图内）。"""
        y0 = max(0, y - radius)
        y1 = min(self.engine.height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(self.engine.width, x + radius + 1)
        ys, xs = np.meshgrid(
            np.arange(y0, y1), np.arange(x0, x1), indexing="ij"
        )
        inside = (ys - y) ** 2 + (xs - x) ** 2 <= radius * radius
        return ys[inside], xs[inside]

    def _kill_in_disk(self, x: int, y: int, radius: int) -> int:
        """杀死圆内的个体；返回死亡数。"""
        swarm = self.swarm
        if swarm.count == 0:
            return 0
        view = swarm.data[: swarm.count]
        alive = view["alive"]
        if not alive.any():
            return 0
        ax = view["x"].astype(np.int32)
        ay = view["y"].astype(np.int32)
        hit = ((ax - x) ** 2 + (ay - y) ** 2) <= radius * radius
        hit &= alive
        count = int(hit.sum())
        if count:
            view["alive"] = view["alive"] & ~hit
        return count

    # ------------------------------------------------------------------
    # 灾害
    # ------------------------------------------------------------------
    def meteor(self, x: int, y: int, radius: int = 12) -> CatastropheEvent:
        """陨石撞击：焦土 + 击杀 + 边缘点燃。"""
        engine = self.engine
        ys, xs = self._disk(x, y, radius)
        killed = self._kill_in_disk(x, y, radius)
        # 中心三分之一的 cell 焦土化（变沙滩样式的裸地 + 食物清零）
        inner = ((ys - y) ** 2 + (xs - x) ** 2) <= (radius // 2) ** 2
        for ty, tx in zip(ys[inner].tolist(), xs[inner].tolist()):
            if engine.terrain[ty, tx] not in (DEEP_WATER,):
                engine.terrain[ty, tx] = SAND
                engine.dirty_terrain.add((ty, tx))
        engine.food[ys, xs] = 0.0
        # 边缘环带随机点燃（幸存的森林倒霉了）
        ring = ~inner
        ignited = 0
        for ty, tx in zip(ys[ring].tolist(), xs[ring].tolist()):
            if np.random.default_rng((ty * 131 + tx * 31 + int(engine.t * 1000))).random() < 0.25:
                if engine.ignite(tx, ty):
                    ignited += 1
        event = CatastropheEvent("meteor", x, y, radius, engine.t, killed=killed, ignited=ignited)
        self._record(event)
        return event

    def lightning(self, x: int, y: int) -> CatastropheEvent:
        """闪电：点燃落点的可燃地形（可能引发大火）。"""
        engine = self.engine
        ignited = 1 if engine.ignite(x, y) else 0
        event = CatastropheEvent("lightning", x, y, 0, engine.t, ignited=ignited)
        self._record(event)
        return event

    def flood(self, x: int, y: int, radius: int = 18) -> CatastropheEvent:
        """洪水：圆内陆地（非山地）变浅水；山脚形成小湖群。"""
        engine = self.engine
        ys, xs = self._disk(x, y, radius)
        flooded = 0
        for ty, tx in zip(ys.tolist(), xs.tolist()):
            if engine.terrain[ty, tx] in (DEEP_WATER, SHALLOW_WATER):
                continue
            # 山地抗洪：只有当圈内"水位"漫过时才淹（简化为概率随距离衰减）
            distance = math.hypot(tx - x, ty - y)
            probability = 1.0 - distance / max(1.0, radius)
            if np.random.default_rng((ty * 977 + tx * 13 + int(engine.t))).random() < probability:
                engine.terrain[ty, tx] = SHALLOW_WATER
                engine.food[ty, tx] = 0.0
                engine.dirty_terrain.add((ty, tx))
                flooded += 1
        event = CatastropheEvent("flood", x, y, radius, engine.t, flooded=flooded)
        self._record(event)
        return event

    # ------------------------------------------------------------------
    # 神迹（仁慈向）
    # ------------------------------------------------------------------
    def heal(self, x: int, y: int, radius: int = 15) -> CatastropheEvent:
        """治愈之雨：灭火 + 食物回满 + 群伙饥饿恢复。"""
        engine = self.engine
        swarm = self.swarm
        ys, xs = self._disk(x, y, radius)
        # 灭火
        engine.burning[ys, xs] = False
        engine.burn_left[ys, xs] = 0.0
        # 食物回填到地形上限
        for ty, tx in zip(ys.tolist(), xs.tolist()):
            cap = FOOD_TERRAIN.get(int(engine.terrain[ty, tx]), 0.0)
            engine.food[ty, tx] = cap
        # 群伙：圆内个体饥饿回满
        healed = 0
        if swarm.count:
            view = swarm.data[: swarm.count]
            ax = view["x"].astype(np.int32)
            ay = view["y"].astype(np.int32)
            hit = ((ax - x) ** 2 + (ay - y) ** 2) <= radius * radius
            hit &= view["alive"]
            if hit.any():
                view["hunger"][hit] = 1.0
                healed = int(hit.sum())
        event = CatastropheEvent("heal", x, y, radius, engine.t, killed=-healed)
        self._record(event)
        return event

    # ------------------------------------------------------------------
    def summary_zh(self) -> str:
        """灾害史的单行中文摘要（事件日志面板用）。"""
        counts: dict[str, int] = {}
        for event in self.history:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        if not counts:
            return "灾害：无"
        names = {"meteor": "陨石", "lightning": "闪电", "flood": "洪水", "heal": "治愈"}
        parts = [f"{names.get(k, k)}×{v}" for k, v in sorted(counts.items())]
        return "灾害：" + " ".join(parts)
