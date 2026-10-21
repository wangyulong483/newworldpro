"""俯视地图世界引擎（V3）：地形 + 食物网格。

这是新世界的地基：一张由 cell 组成的俯视地图（默认 512×320，每 cell
在屏幕上渲染为固定大小的色块）。地形由 seed 确定性生成（值噪声 →
海拔/温度双阈值分层），食物按地形/湿度/昼夜在网格上再生。

设计原则（延续本项目的"世界规则唯一权威 + 可审计"传统）
----------------------------------------------------------------
1. **确定性**：同 seed 必得同地图（值噪声用整数格点哈希，无 Python
   随机漂移）。
2. **网格即真相**：``terrain``/``food`` 都是与地图同形状的 numpy
   数组，任何模块（群伙/渲染）都只通过 MapEngine 的公开方法读取它们，
   方便审计与测试。
3. **地形语义**（:data:`TERRAIN_IDS`）：深水/浅水/沙滩/草地/森林/
   山地/雪原。
4. **只读世界**：地图本身不会被外部"上帝之手"改写；地形唯一的
   自发变化是森林蔓延（草地邻接森林时小概率长树），其余变化都由
   时间与食物再生驱动。

坐标约定：``terrain[y, x]``，y 向下、x 向右（与图像数组一致）；
cell 索引从 0 开始。群伙（:mod:`newworld.swarm`）只读取地形/食物，
渲染（:mod:`newworld.map_render`）只消费本模块暴露的网格。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# 地形定义
# ---------------------------------------------------------------------------
#: 地形编号（uint8 网格的取值）。0 留给"无效/边界外"。
DEEP_WATER = 1
SHALLOW_WATER = 2
SAND = 3
GRASS = 4
FOREST = 5
MOUNTAIN = 6
SNOW = 7

#: 地形 id -> (中文名, 基础色 RGB)。渲染底图与 UI 共用这张表。
TERRAIN_INFO: dict[int, tuple[str, tuple[int, int, int]]] = {
    DEEP_WATER: ("深海", (24, 46, 88)),
    SHALLOW_WATER: ("浅海", (46, 92, 138)),
    SAND: ("沙滩", (214, 194, 140)),
    GRASS: ("草地", (96, 148, 66)),
    FOREST: ("森林", (46, 104, 52)),
    MOUNTAIN: ("山地", (122, 116, 110)),
    SNOW: ("雪原", (232, 236, 240)),
}

#: 可行走地形（群伙可居住/通过）
WALKABLE = frozenset({SAND, GRASS, FOREST, MOUNTAIN, SNOW})
#: 可再生食物的地形（食物密度上限不同）
FOOD_TERRAIN = {GRASS: 1.0, FOREST: 0.7, SAND: 0.15}
#: 水域（群伙溺亡规则用）
WATER = frozenset({DEEP_WATER, SHALLOW_WATER})


def _hash2(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """整数格点 → [0,1) 均匀伪随机数（splitmix64 风格，向量化）。"""
    mixed_seed = (seed * 0x1F123BB5 + 0x9E3779B9) & 0xFFFFFFFF
    h = (x.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
         + y.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
         + np.uint64(mixed_seed))
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    h = h ^ (h >> np.uint64(31))
    return (h & np.uint64(0xFFFFFFFFFFFF)).astype(np.float64) / float(1 << 48)


def _noise_grid(width: int, height: int, seed: int, scale: int) -> np.ndarray:
    """生成 (height, width) 的平滑值噪声，值域 [0, 1]。

    格点哈希（整数）→ 双线性插值（分数部分平滑）→ 两尺度叠加（倍频），
    得到自然地貌的"大陆块"感。全程无 Python 随机，同 seed 复现。
    """
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    total = np.zeros((height, width), dtype=np.float64)
    for octave_scale, weight in ((scale, 0.65), (max(2, scale // 2), 0.35)):
        gx = xs / octave_scale
        gy = ys / octave_scale
        x0 = np.floor(gx).astype(np.int64)
        y0 = np.floor(gy).astype(np.int64)
        fx = gx - x0
        fy = gy - y0
        # 平滑插值权重（ smootherstep ）
        fx = fx * fx * (3.0 - 2.0 * fx)
        fy = fy * fy * (3.0 - 2.0 * fy)
        corners = []
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            corners.append(_hash2(x0 + dx, y0 + dy, seed + octave_scale * 7919))
        c00, c10, c01, c11 = corners
        top = c00 * (1.0 - fx) + c10 * fx
        bottom = c01 * (1.0 - fx) + c11 * fx
        total += weight * (top * (1.0 - fy) + bottom * fy)
    return total


@dataclass
class MapConfig:
    """俯视地图的生成参数。"""

    width: int = 512
    height: int = 320
    seed: int = 0
    #: 地图四周的水域框宽（cell 数）：岛屿世界，防 agent 走出边界
    ocean_margin: int = 6
    #: 噪声尺度（cell 数）：越大陆地块越整
    noise_scale: int = 64
    #: 海平面阈值（噪声 0..1 归一后高于此值为陆地）
    sea_level: float = 0.42
    #: 山地/雪线阈值（相对噪声范围）
    mountain_level: float = 0.74
    snow_level: float = 0.86
    #: 食物再生速率（每秒、每 cell，满食物为 1.0）
    food_regrow_per_s: float = 0.012
    #: 森林蔓延：邻接草地有小概率长树（每秒每 cell）
    forest_spread_per_s: float = 0.0004


class MapEngine:
    """俯视世界：地形网格 + 食物网格 + 昼夜驱动的再生循环。

    坐标约定：``terrain[y, x]``，y 向下、x 向右（与图像数组一致）；
    cell 索引从 0 开始。群伙（:mod:`newworld.swarm`）只读取地形与食物。
    """

    def __init__(self, config: MapConfig | None = None) -> None:
        self.config = config or MapConfig()
        cfg = self.config
        self.width = int(cfg.width)
        self.height = int(cfg.height)
        self.t: float = 0.0
        self.steps: int = 0
        self.terrain = self._generate_terrain()
        #: 每 cell 的食物量 0..1（陆地才有）
        self.food = self._initial_food()
        #: 修改过的地形 cell（渲染层增量重画用）：set of (y, x)
        self.dirty_terrain: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    def _generate_terrain(self) -> np.ndarray:
        """确定性生成地形：双噪声（大陆块 + 细节）→ 阈值分层。"""
        cfg = self.config
        w, h = self.width, self.height
        elevation = _noise_grid(w, h, cfg.seed, cfg.noise_scale)
        temperature = _noise_grid(w, h, cfg.seed + 424243, max(24, cfg.noise_scale))
        # 归一化到 [0,1]（噪声理论值域约 [0,1]，夹紧即可）
        elevation = np.clip(elevation, 0.0, 1.0)
        temperature = np.clip(temperature, 0.0, 1.0)

        terrain = np.zeros((h, w), dtype=np.uint8)
        # 水陆
        land = elevation >= cfg.sea_level
        shallow = (~land) & (elevation >= cfg.sea_level - 0.06)
        terrain[shallow] = SHALLOW_WATER
        terrain[~land & ~shallow] = DEEP_WATER
        # 陆地按海拔
        beach = land & (elevation < cfg.sea_level + 0.05)
        terrain[beach] = SAND
        high = land & (elevation >= cfg.mountain_level)
        snowy = land & (elevation >= cfg.snow_level)
        plain = land & ~beach & ~high
        # 森林：平原上温度适中的部分
        forest = plain & (temperature > 0.35) & (temperature < 0.75)
        terrain[plain & ~forest] = GRASS
        terrain[forest] = FOREST
        terrain[high & ~snowy] = MOUNTAIN
        terrain[snowy] = SNOW
        # 周边大洋（保证世界是岛屿，也顺带定义了边界）
        m = int(cfg.ocean_margin)
        terrain[:m, :] = DEEP_WATER
        terrain[h - m:, :] = DEEP_WATER
        terrain[:, :m] = DEEP_WATER
        terrain[:, w - m:] = DEEP_WATER
        return terrain

    def _initial_food(self) -> np.ndarray:
        """初始食物：按地形上限随机撒（之后由再生循环维持）。"""
        food = np.zeros((self.height, self.width), dtype=np.float32)
        for tid, cap in FOOD_TERRAIN.items():
            mask = self.terrain == tid
            if not mask.any():
                continue
            noise = _noise_grid(self.width, self.height, self.config.seed + tid * 131, 16)
            food[mask] = (np.clip(noise[mask], 0, 1) * cap).astype(np.float32)
        return food

    # ------------------------------------------------------------------
    # 查询（群伙/渲染共用）
    # ------------------------------------------------------------------
    def terrain_at(self, x: int, y: int) -> int:
        """(x, y) 处的地形编号；越界返回 0。"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.terrain[y, x])
        return 0

    def is_walkable(self, x: int, y: int) -> bool:
        """cell 是否可行走（陆地）。"""
        return self.terrain_at(x, y) in WALKABLE

    def walkable_mask(self) -> np.ndarray:
        """整张可行走掩码（渲染/统计用）。"""
        mask = np.zeros_like(self.terrain, dtype=bool)
        for tid in WALKABLE:
            mask |= self.terrain == tid
        return mask

    def land_cells(self) -> int:
        """可行走 cell 总数（承载力估算用）。"""
        return int(self.walkable_mask().sum())

    def food_capacity(self) -> float:
        """整张地图的食物总容量（承载力的分子）。"""
        return float(self.food.sum())

    def food_at(self, x: int, y: int) -> float:
        """(x, y) 处的食物量；水域/越界为 0。"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.food[y, x])
        return 0.0

    # ------------------------------------------------------------------
    # 每帧推进（世界规则）
    # ------------------------------------------------------------------
    def update(self, dt: float, daylight: float) -> None:
        """推进一个物理步：食物再生 + 森林蔓延。

        Parameters
        ----------
        dt:
            步长（世界秒）。
        daylight:
            白昼强度 0..1（夜间再生减半）。
        """
        cfg = self.config
        self.t += dt
        self.steps += 1

        # -- 食物再生：可食地形按速率回填（夜间 ×0.5） ----------
        regrow = cfg.food_regrow_per_s * dt * (0.5 + 0.5 * daylight)
        for tid, cap in FOOD_TERRAIN.items():
            mask = self.terrain == tid
            if not mask.any():
                continue
            room = cap - self.food[mask]
            grow = np.clip(room, 0.0, None) * min(1.0, regrow * 8.0)
            self.food[mask] += grow.astype(self.food.dtype)

        # -- 森林蔓延（低频事件：草地上小概率长树） ------------------
        if self.steps % 32 == 0 and cfg.forest_spread_per_s > 0.0:
            probability = cfg.forest_spread_per_s * dt * 32.0
            rng = np.random.default_rng(self.steps * 31 + 5)
            ys, xs = np.nonzero(self.terrain == GRASS)
            if ys.size:
                hit = rng.random(ys.size) < probability
                for y, x in zip(ys[hit].tolist(), xs[hit].tolist()):
                    # 只在邻接森林时长树（扩散感）
                    neighbourhood = self.terrain[
                        max(0, y - 2):y + 3, max(0, x - 2):x + 3
                    ]
                    if (neighbourhood == FOREST).any():
                        self.terrain[y, x] = FOREST
                        self.dirty_terrain.add((y, x))

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    def terrain_summary(self) -> dict[str, int]:
        """各地形 cell 数（状态栏/统计面板用）。"""
        counts: dict[str, int] = {}
        for tid, (name, _colour) in TERRAIN_INFO.items():
            counts[name] = int((self.terrain == tid).sum())
        return counts

    def summary_zh(self) -> str:
        """单行中文摘要（UI 状态栏）。"""
        parts = self.terrain_summary()
        return (
            f"陆地 {parts.get('草地', 0) + parts.get('森林', 0) + parts.get('山地', 0)}"
            f"{parts.get('沙滩', 0) and '+' + str(parts['沙滩']) or ''}格"
            f" 食物 {self.food_capacity():.0f}"
        )
