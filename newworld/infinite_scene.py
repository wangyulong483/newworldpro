"""无限世界：横向无界的确定性世界 + 局部窗口场景。

契约出处：V2 契约 §3（infinite-dev，全局坐标持久化 + recenter）。本模块只做
"场景的一半"（地形/物体/水/生态/光学/渲染），不含角色与文明；``World`` /
``Ecology`` / ``render`` / ``Technology`` / ``optics`` 通过
:class:`InfiniteScene` 提供的**窗口化 API**工作，与
:class:`newworld.pixelworld.PixelScene` 同名同义（坐标都是**窗口局部坐标**），
因此集成层可以基本不改地复用现有代码。

设计
----
* **横向无界、纵向固定**：世界在 x 方向没有边界（全局原子 x 可以是任意整数，
  包括负数），y 方向仍由配置高度决定，地表固定在同一行。窗口只是这个无限
  世界上的一个"取景框"：``grid`` 形状 ``(ah, aw)``，``aw`` 默认是
  ``2 × (config.width // ATOM_SIZE)``（比屏幕宽一倍，留出跟随/预取余量）。
* **全局确定性生成**：任意全局原子 ``(gx, y)`` 的材料只由 ``seed``、``gx``、
  ``y`` 与分块 hash 决定，**与窗口位置无关**。两个相同 seed 的窗口在重叠
  区域逐原子一致；不同 seed 不同。hash 用显式 64 位 splitmix 混合，不依赖
  ``hash()`` / ``PYTHONHASHSEED``，跨进程、跨平台可复现。
* **对象/资源按确定性格点放置**：树/岩石/浆果丛各自按固定宽度的全局 chunk
  放置，每个 chunk 由 hash 决定"有没有、在哪、多大"；池塘每
  ``POND_PERIOD_CHUNKS × POND_CHUNK_ATOMS``（默认 2×64 = 128）个原子出现
  一次，保证任意默认宽度（256 原子）的窗口内**至少有一个池塘**。
* **修改持久化**：``set_atom`` / ``clear_atom`` / ``pick_berry`` / ``add_berry``
  按**全局坐标**记录进稀疏字典 ``_edits``；``recenter`` 重建时会把这些修改
  叠加到确定性基底上，所以切走再切回内容一致。
* **窗口平移**：``recenter(delta)`` 执行 ``window_start_atom += delta`` 并
  重建 grid/objects/berries/pond 和派生缓存（solid/blocking/top_row/
  albedo/shadow/optics）。

生态与窗口平移
--------------
``ecology`` 是真实的 :class:`newworld.ecology.Ecology` 实例（它只依赖
``aw/ah/surface/ground_row/pond/pond_surface/grid/objects/berries`` 与
``add_berry/set_atom/clear_atom/refresh_caches``，这些都由本类提供）。

``recenter`` 时 **moisture 数组按窗口平移**：重叠列直接搬运旧数组，新进入
窗口的列用与 :meth:`newworld.ecology.Ecology._init_moisture` 相同的近似规则
重新播种（土壤/草 0.38、池塘左右 10 列 +0.35、越深最多 +0.15·相对深度）。
这是**近似重采样**：湿度扩散的历史不会跨窗口边界传播，窗口边缘的湿度可能
与"世界一直存在"的理想值略有偏差。天气（rain/humidity/rain_timer）、碎屑、
破坏/生长计数保留；``water_level`` 因窗口内容变了而重置为 1.0（
``_initial_water`` 取新窗口的水原子数），所以它只在同一窗口内可比。

已知限制 / 与 PixelScene 的差异
-------------------------------
* 纵向固定：没有 PixelScene 那种整屏地形起伏，地表始终是 ``surface`` 行。
* ``objects`` 只包含**与窗口相交**的物体，``x0/x1`` 是裁剪后的局部坐标，
  所以同一个物体在不同窗口里的 ``x0/x1`` 不同；``atom_count`` 仍是完整的
  全局原子数，可以据此判断是同一个物体。
* ``berries`` 是窗口内**所有**浆果原子，包括 PixelScene 里只作装饰的"花"
  顶端浆果；这样"位置一致 / 可摘 / 持久化"只有一条规则。
* ``render_atoms`` / ``optics_scene`` 只覆盖窗口。把动态 x 坐标平移到窗口
  坐标系是 Lead 的 ``World`` 集成层的工作（见契约 §3.3）。
"""

from __future__ import annotations

import math
import random

import numpy as np

from . import materials
from .daynight import DayCycle
from .ecology import Ecology
from .materials import MATERIAL_BY_INDEX, MATERIAL_INDEX, MATERIAL_LIST, MATERIALS, Material
from .optics import Box, Disc, Scene
from .pixelworld import PixelObject

ATOM = materials.ATOM_SIZE

#: 默认窗口宽度 = 屏幕宽度（原子）× 这个倍数
DEFAULT_WINDOW_FACTOR = 2
#: 窗口最小宽度（原子）。极小窗口会退化成单/双池塘的窄条，但不允许崩溃。
MIN_WINDOW_ATOMS = 16

#: 池塘所在 chunk 的宽度（原子）
POND_CHUNK_ATOMS = 64
#: 每多少个 chunk 出现一个池塘（2 → 每 128 个原子一个池塘）
POND_PERIOD_CHUNKS = 2
#: 池塘宽度范围（原子，闭区间）
POND_MIN_WIDTH = 18
POND_MAX_WIDTH = 30

#: 树/岩石/浆果丛的确定性格点宽度（原子）
TREE_CHUNK_ATOMS = 48
ROCK_CHUNK_ATOMS = 32
BUSH_CHUNK_ATOMS = 28

#: 生成物体时要额外扫描的 chunk 余量（原子）。
#: 最大的物体足迹半径是树（树冠 ≤ 12 格 + 2 格余量），48 足够覆盖
#: "chunk 在窗口外、但物体探进窗口"的情形。
_OBJECT_MARGIN_ATOMS = 48

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15


# ---------------------------------------------------------------------------
# 确定性 hash（不用内置 hash()，避免 PYTHONHASHSEED 影响可复现性）
# ---------------------------------------------------------------------------
def _mix64(value: int) -> int:
    """splitmix64 的一次混合（对任意 Python 整数取低 64 位后混合）。"""
    value &= _MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MASK64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _MASK64
    value ^= value >> 31
    return value & _MASK64


def _hash64(*values: int) -> int:
    """把若干整数折叠成一个 64 位 hash（负数按二补码处理，结果是确定的）。"""
    h = _GOLDEN
    for value in values:
        h = _mix64(h ^ ((int(value) & _MASK64) + _GOLDEN))
    return h


def _unit(value: int) -> float:
    """把 64 位 hash 映射到 [0, 1)。"""
    return (value & _MASK64) / float(1 << 64)


def material_name_zh(key: str) -> str:
    """按 key 取材料中文名（与 :func:`newworld.pixelworld.material_name_zh` 一致）。"""
    return MATERIALS[key].name_zh


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
class InfiniteScene:
    """横向无界的确定性世界上的一个局部窗口（与 PixelScene 兼容）。

    参数
    ----
    config:
        世界配置（只用到 ``width`` / ``height`` / ``ground_y``）。
    seed:
        世界种子。相同 seed 的两个实例在重叠区域逐原子一致。
    window_atoms:
        窗口宽度（原子）。默认 ``2 × (config.width // ATOM_SIZE)``；
        小于 :data:`MIN_WINDOW_ATOMS` 会被夹到该值（极小窗口降级不崩）。
    """

    def __init__(self, config, seed: int = 0, *, window_atoms: int | None = None) -> None:
        """建立窗口并生成确定性世界；随后创建真实 :class:`Ecology`。"""
        self.config = config
        self.seed = int(seed)
        #: 仅为兼容 PixelScene 的同名属性；本模块的生成不依赖它（纯 hash）。
        self.rng = random.Random(self.seed)

        base_atoms = max(MIN_WINDOW_ATOMS, config.width // ATOM)
        if window_atoms is None:
            requested = base_atoms * DEFAULT_WINDOW_FACTOR
        else:
            try:
                requested = int(window_atoms)
            except (TypeError, ValueError):  # 非法参数降级为默认窗口
                requested = base_atoms * DEFAULT_WINDOW_FACTOR
        self.aw = max(MIN_WINDOW_ATOMS, requested)
        self.ah = max(MIN_WINDOW_ATOMS, config.height // ATOM)
        # 地表行（原子）：与 PixelScene 相同的换算，纵向固定
        self.ground_row = int(round(config.ground_y / ATOM))
        self.surface = self.ground_row - 1
        self.terrain_bottom = self.ah - 1

        #: 窗口左边缘的全局原子 x（recenter 修改它）
        self.window_start_atom = 0
        #: 全局坐标 → 材料编号的稀疏修改表（只记录被玩家/生态改过的原子）
        self._edits: dict[tuple[int, int], int] = {}

        self.grid = np.zeros((self.ah, self.aw), dtype=np.uint8)
        self.objects: list[PixelObject] = []
        self.berries: list[tuple[int, int]] = []
        self.pond = (0, max(0, self.aw - 1))
        self.pond_surface = self.surface
        self.shadow_ground = np.zeros(self.aw, dtype=bool)
        self._optics_cache: Scene | None = None

        self._generate_window()
        self.refresh_caches()
        self.ecology = Ecology(self)

    # ------------------------------------------------------------------
    # 全局 <-> 局部坐标
    # ------------------------------------------------------------------
    @property
    def window_start_px(self) -> float:
        """窗口左边缘的全局屏幕像素 x（= ``window_start_atom * ATOM_SIZE``）。"""
        return float(self.window_start_atom * ATOM)

    @property
    def window_end_atom(self) -> int:
        """窗口右边缘（不含）的全局原子 x。"""
        return self.window_start_atom + self.aw

    def global_atom_x(self, local_atom_x: int) -> int:
        """窗口局部原子 x → 全局原子 x。"""
        return int(local_atom_x) + self.window_start_atom

    def local_atom_x(self, global_atom_x: int) -> int:
        """全局原子 x → 窗口局部原子 x（可能落在窗口外）。"""
        return int(global_atom_x) - self.window_start_atom

    # ------------------------------------------------------------------
    # 确定性布局（纯函数：只依赖 seed 与全局坐标）
    # ------------------------------------------------------------------
    def _pond_layout(self, chunk: int) -> tuple[int, int]:
        """第 ``chunk`` 个池塘 chunk 的 (全局左边界, 宽度)。"""
        h = _hash64(self.seed, chunk, 21)
        width = POND_MIN_WIDTH + h % (POND_MAX_WIDTH - POND_MIN_WIDTH + 1)
        free = POND_CHUNK_ATOMS - 12 - width  # 两侧各留 6 个原子的岸
        offset = 6 + (h >> 16) % max(1, free)
        left = chunk * POND_CHUNK_ATOMS + offset
        return left, width

    def _pond_depth(self, left: int, width: int, gx: int) -> int:
        """池塘在全局列 ``gx`` 的水深（原子行数）；正弦洼地 + 确定性抖动。"""
        frac = (gx - left) / max(1, width)
        base = 2 + int(1.4 * math.sin(math.pi * frac))
        jitter = (_hash64(self.seed, gx, 31) % 3) - 1
        return max(1, base + jitter)

    def _pond_intersects(self, gx0: int, gx1: int) -> bool:
        """全局列区间 ``[gx0, gx1]`` 是否与某个池塘重叠（用于避让）。"""
        if gx1 < gx0:
            gx0, gx1 = gx1, gx0
        first = gx0 // POND_CHUNK_ATOMS
        last = gx1 // POND_CHUNK_ATOMS
        for chunk in range(first, last + 1):
            if chunk % POND_PERIOD_CHUNKS:
                continue
            left, width = self._pond_layout(chunk)
            if left <= gx1 and left + width >= gx0:
                return True
        return False

    def _chunk_range(self, chunk_size: int) -> range:
        """覆盖窗口（含 ``_OBJECT_MARGIN_ATOMS`` 余量）的全局 chunk 编号。"""
        start = self.window_start_atom
        first = (start - _OBJECT_MARGIN_ATOMS) // chunk_size
        last = (start + self.aw + _OBJECT_MARGIN_ATOMS) // chunk_size
        return range(first, last + 1)

    # ------------------------------------------------------------------
    # 生成期绘制工具（全局坐标 → 窗口局部；越界部分自动裁剪）
    # ------------------------------------------------------------------
    def _paint_rect(
        self,
        grid: np.ndarray,
        gx0: int,
        gx1: int,
        gy0: int,
        gy1: int,
        index: int,
        start: int,
    ) -> int:
        """把全局矩形写入窗口，返回**完整**矩形的原子数（含窗口外）。"""
        width = max(0, gx1 - gx0 + 1)
        height = max(0, gy1 - gy0 + 1)
        vis_x0 = max(0, gx0 - start)
        vis_x1 = min(self.aw - 1, gx1 - start)
        vis_y0 = max(0, gy0)
        vis_y1 = min(self.ah - 1, gy1)
        if vis_x1 >= vis_x0 and vis_y1 >= vis_y0:
            grid[vis_y0:vis_y1 + 1, vis_x0:vis_x1 + 1] = index
        return width * height

    def _paint_disc(
        self,
        grid: np.ndarray,
        gcx: float,
        gcy: float,
        radius: float,
        index: int,
        start: int,
    ) -> int:
        """把全局实心圆盘写入窗口，返回**完整**圆盘的原子数（含窗口外）。"""
        radius_sq = radius * radius
        x_lo = int(math.floor(gcx - radius))
        x_hi = int(math.ceil(gcx + radius))
        y_lo = int(math.floor(gcy - radius))
        y_hi = int(math.ceil(gcy + radius))
        if x_hi < x_lo or y_hi < y_lo:
            return 0
        gx = np.arange(x_lo, x_hi + 1, dtype=np.float64)
        gy = np.arange(y_lo, y_hi + 1, dtype=np.float64)
        mask = ((gx - gcx) ** 2)[None, :] + ((gy - gcy) ** 2)[:, None] <= radius_sq
        count = int(mask.sum())
        # 可见部分在完整 mask 里的切片
        vis_x0 = max(0, x_lo - start)
        vis_x1 = min(self.aw - 1, x_hi - start)
        vis_y0 = max(0, y_lo)
        vis_y1 = min(self.ah - 1, y_hi)
        if vis_x1 >= vis_x0 and vis_y1 >= vis_y0:
            mx0 = vis_x0 + start - x_lo
            mx1 = vis_x1 + start - x_lo + 1
            my0 = vis_y0 - y_lo
            my1 = vis_y1 - y_lo + 1
            grid[vis_y0:vis_y1 + 1, vis_x0:vis_x1 + 1][mask[my0:my1, mx0:mx1]] = index
        return count

    def _add_object(
        self,
        kind: str,
        name_zh: str,
        gx0: int,
        gx1: int,
        gtop: int,
        gbottom: int,
        albedo: tuple[int, int, int],
        atom_count: int,
        density: float,
        specific_heat: float,
        *,
        blocking: bool = False,
        berries: list[tuple[int, int]] | None = None,
        berry_slots: list[tuple[int, int]] | None = None,
        start: int,
    ) -> PixelObject | None:
        """登记一个与窗口相交的物体（坐标裁剪成窗口局部；不相交则跳过）。"""
        lx0 = max(0, gx0 - start)
        lx1 = min(self.aw - 1, gx1 - start)
        if lx1 < lx0:
            return None
        volume = max(0, atom_count) * (0.05 ** 3)  # 每个原子 = 5 cm 的立方体
        mass = volume * density
        obj = PixelObject(
            kind=kind,
            name_zh=name_zh,
            x0=lx0,
            x1=lx1,
            top=max(0, gtop),
            bottom=min(self.ah - 1, gbottom),
            albedo=albedo,
            atom_count=atom_count,
            thermal_mass_j_k=mass * specific_heat,
            berries=list(berries or ()),
            berry_slots=list(berry_slots if berry_slots is not None else (berries or ())),
            blocking=blocking,
        )
        self.objects.append(obj)
        return obj

    # ------------------------------------------------------------------
    # 窗口生成（确定性；recenter 后整体重建）
    # ------------------------------------------------------------------
    def _generate_window(self) -> None:
        """按全局确定性规则生成当前窗口的 grid/objects/berries/pond。

        顺序：地质分层 → 池塘 → 草簇/花 → 树 → 岩石 → 浆果丛 → 叠加
        ``_edits``（持久修改）→ 汇总浆果。所有绘制都用**全局坐标**计算，
        只把可见部分裁剪进窗口，因此同一全局原子在不同窗口里得到同一材料。
        """
        aw, ah = self.aw, self.ah
        s = self.surface
        start = self.window_start_atom
        self.objects = []
        grid = np.zeros((ah, aw), dtype=np.uint8)

        soil_i = MATERIAL_INDEX["soil"]
        grass_i = MATERIAL_INDEX["grass"]
        stone_i = MATERIAL_INDEX["stone"]
        water_i = MATERIAL_INDEX["water"]
        wood_i = MATERIAL_INDEX["wood"]
        leaf_i = MATERIAL_INDEX["leaf"]
        berry_i = MATERIAL_INDEX["berry"]

        # -- 地质分层（横向均匀，纵向固定） ----------------------------
        if 0 <= s < ah:
            grid[s, :] = grass_i
        if s + 1 < ah:
            grid[max(0, s + 1):min(ah, s + 4), :] = soil_i
        if s + 4 < ah:
            grid[max(0, s + 4):, :] = stone_i

        # -- 池塘（每 POND_PERIOD_CHUNKS 个 chunk 一次，周期 128 原子） --
        first_chunk = start // POND_CHUNK_ATOMS
        last_chunk = (start + aw - 1) // POND_CHUNK_ATOMS
        for chunk in range(first_chunk, last_chunk + 1):
            if chunk % POND_PERIOD_CHUNKS:
                continue
            left, width = self._pond_layout(chunk)
            right = left + width
            for gx in range(max(left, start), min(right, start + aw - 1) + 1):
                lx = gx - start
                depth = self._pond_depth(left, width, gx)
                for y in range(max(0, s), min(ah, s + depth + 1)):
                    grid[y, lx] = water_i
                y_soil = s + depth + 1
                if 0 <= y_soil < ah:
                    grid[y_soil, lx] = soil_i

        # -- 草簇与"花"（按列 hash；只长在没有被池塘淹没的草地上） ------
        for lx in range(aw):
            if not (0 <= s < ah) or grid[s, lx] != grass_i:
                continue
            gx = start + lx
            h = _hash64(self.seed, gx, 11)
            if h % 7 == 0:
                height = 1 + (h >> 8) % 2
                for k in range(1, height + 1):
                    if s - k >= 0:
                        grid[s - k, lx] = grass_i
            if h % 19 == 0:
                if s - 1 >= 0:
                    grid[s - 1, lx] = grass_i
                if s - 2 >= 0:
                    grid[s - 2, lx] = grass_i
                if s - 3 >= 0:
                    grid[s - 3, lx] = berry_i

        # -- 树（树干 + 树冠） -----------------------------------------
        for chunk in self._chunk_range(TREE_CHUNK_ATOMS):
            h = _hash64(self.seed, chunk, 101)
            if h % 100 >= 60:
                continue
            cx = chunk * TREE_CHUNK_ATOMS + 10 + (h >> 8) % 28
            trunk_h = 22 + (h >> 16) % 13
            trunk_w = 3 + (1 if (h >> 24) % 3 == 0 else 0)
            canopy_r = 8.0 + ((h >> 32) % 41) / 10.0
            if self._pond_intersects(cx - 6, cx + 6):
                continue
            gx0 = cx - trunk_w // 2
            gx1 = cx + trunk_w // 2
            trunk_count = self._paint_rect(grid, gx0, gx1, s - trunk_h, s, wood_i, start)
            cy = s - trunk_h - 2
            canopy_count = self._paint_disc(grid, float(cx), float(cy), canopy_r, leaf_i, start)
            self._add_object(
                "tree", "树",
                int(math.floor(cx - canopy_r)), int(math.ceil(cx + canopy_r)),
                int(math.floor(cy - canopy_r)), s,
                (78, 128, 62), trunk_count + canopy_count, 600.0, 1700.0,
                blocking=True, start=start,
            )

        # -- 岩石（落在土壤上，避开池塘） -------------------------------
        for chunk in self._chunk_range(ROCK_CHUNK_ATOMS):
            h = _hash64(self.seed, chunk, 121)
            if h % 100 >= 45:
                continue
            cx = chunk * ROCK_CHUNK_ATOMS + 6 + (h >> 8) % 20
            radius = 2.0 + ((h >> 16) % 23) / 10.0
            if self._pond_intersects(cx - 4, cx + 4):
                continue
            cy = s - radius * 0.55
            count = self._paint_disc(grid, float(cx), float(cy), radius, stone_i, start)
            if count <= 0:
                continue
            self._add_object(
                "rock", "岩石",
                int(math.floor(cx - radius)), int(math.ceil(cx + radius)),
                int(math.floor(cy - radius)), s,
                (126, 126, 130), count, 2700.0, 790.0,
                blocking=True, start=start,
            )

        # -- 浆果丛（树叶 + 可食用浆果；可穿过） ------------------------
        for chunk in self._chunk_range(BUSH_CHUNK_ATOMS):
            h = _hash64(self.seed, chunk, 141)
            if h % 100 >= 45:
                continue
            cx = chunk * BUSH_CHUNK_ATOMS + 6 + (h >> 8) % 16
            radius = 3.0 + ((h >> 16) % 21) / 10.0
            if self._pond_intersects(cx - 6, cx + 6):
                continue
            cy = s - radius * 0.8
            leaf_count = self._paint_disc(grid, float(cx), float(cy), radius, leaf_i, start)
            slots: list[tuple[int, int]] = []
            berry_count = 2 + (h >> 24) % 4
            for k in range(berry_count):
                hb = _hash64(self.seed, chunk, 200 + k)
                bx = int(cx + (_unit(hb) - 0.5) * 2.0 * radius * 0.7)
                by = int(cy + (_unit(hb >> 17) - 0.5) * 2.0 * radius * 0.5)
                if not (0 <= by < ah):
                    continue
                lx = bx - start
                if 0 <= lx < aw and grid[by, lx] == leaf_i:
                    grid[by, lx] = berry_i
                    slots.append((lx, by))
            self._add_object(
                "bush", "浆果丛",
                int(math.floor(cx - radius)), int(math.ceil(cx + radius)),
                int(math.floor(cy - radius)), s,
                (74, 138, 66), leaf_count, 260.0, 1600.0,
                blocking=False, berries=slots, berry_slots=list(slots), start=start,
            )

        # -- 持久修改（全局坐标）叠加在确定性基底之上 ------------------
        for (gx, y), index in self._edits.items():
            lx = gx - start
            if 0 <= lx < aw and 0 <= y < ah:
                grid[y, lx] = index

        # -- 选择窗口的"代表池塘"并汇总浆果 ----------------------------
        self.grid = grid
        self.pond = self._select_pond()
        self.pond_surface = self.surface
        ys, xs = np.nonzero(grid == berry_i)
        self.berries = [(int(x), int(y)) for y, x in zip(ys.tolist(), xs.tolist())]
        # 被修改/踩掉的浆果槽位不再算作"已结果"，但槽位本身保留给生态再生
        for obj in self.objects:
            if obj.berries:
                obj.berries = [p for p in obj.berries if grid[p[1], p[0]] == berry_i]

    def _select_pond(self) -> tuple[int, int]:
        """选距离窗口中心最近的那个池塘，夹到窗口内（保证 left ≤ right）。"""
        center = self.window_start_atom + self.aw / 2.0
        first = (self.window_start_atom - POND_CHUNK_ATOMS) // POND_CHUNK_ATOMS
        last = (self.window_start_atom + self.aw + POND_CHUNK_ATOMS) // POND_CHUNK_ATOMS
        best: tuple[float, int, int] | None = None
        for chunk in range(first, last + 1):
            if chunk % POND_PERIOD_CHUNKS:
                continue
            left, width = self._pond_layout(chunk)
            distance = abs(left + width / 2.0 - center)
            if best is None or distance < best[0]:
                best = (distance, left, width)
        if best is None:  # 理论上不可达：每 128 个原子必有一个池塘
            return (0, max(0, self.aw - 1))
        _, left, width = best
        local_left = left - self.window_start_atom
        local_right = left + width - self.window_start_atom
        local_left = max(0, min(self.aw - 1, local_left))
        local_right = max(0, min(self.aw - 1, local_right))
        if local_right < local_left:
            local_right = local_left
        return (local_left, local_right)

    # ------------------------------------------------------------------
    # 派生缓存（与 PixelScene 同名）
    # ------------------------------------------------------------------
    def refresh_caches(self) -> None:
        """重建 solid/blocking/top_row/albedo 缓存（原子级修改后必须调用）。"""
        self.solid = self.grid > 0
        self.blocking = self._blocking_mask()
        self.top_row = self._top_rows()
        self.albedo_rgb = self._albedo_rgb()

    def _blocking_mask(self) -> np.ndarray:
        """硬阻挡掩码：只有石头和木头挡身体（与 PixelScene 相同）。"""
        mask = np.zeros_like(self.grid, dtype=bool)
        for key in ("stone", "wood"):
            mask |= self.grid == MATERIAL_INDEX[key]
        return mask

    def _top_rows(self) -> np.ndarray:
        """每列最上方实心原子的行号（整列为空时 = ah）。"""
        solid = self.grid > 0
        rows = np.full(self.aw, self.ah, dtype=np.int32)
        if self.aw:
            first = solid.argmax(axis=0).astype(np.int32)
            rows = np.where(solid.any(axis=0), first, self.ah).astype(np.int32)
        return rows

    def _albedo_rgb(self) -> np.ndarray:
        """把材料编号网格翻译成 RGB 图像（ah × aw × 3，uint8）。"""
        table = np.zeros((len(MATERIAL_LIST), 3), dtype=np.uint8)
        for index, material in MATERIAL_BY_INDEX.items():
            table[index] = material.albedo
        return table[self.grid]

    # ------------------------------------------------------------------
    # 窗口平移
    # ------------------------------------------------------------------
    def recenter(self, delta_atoms: int) -> None:
        """把窗口整体平移 ``delta_atoms`` 个原子（>0 向右，内容相对屏幕左移）。

        重建 grid/objects/berries/pond 与派生缓存；``_edits``（全局坐标）
        自动重新叠加，所以修改/采果在平移后仍然存在。生态的 ``moisture``
        随窗口平移（近似重采样，见模块 docstring），天气等状态保留。

        非法 ``delta_atoms``（``None`` / 无法转成整数）按 0 处理，不抛错；
        ``0`` 直接返回。目标性能：单次 ≤ 30 ms（默认 256×80 窗口实测见
        测试 ``test_recenter_performance``）。
        """
        try:
            delta = int(delta_atoms)
        except (TypeError, ValueError):
            delta = 0
        if delta == 0:
            return
        old_moisture = self.ecology.moisture.copy()
        self.window_start_atom += delta
        self._optics_cache = None
        self._generate_window()
        self.refresh_caches()
        self._shift_ecology(delta, old_moisture)

    def _shift_ecology(self, delta: int, old_moisture: np.ndarray) -> None:
        """把生态状态平移到新窗口（moisture 近似重采样，其余状态保留）。"""
        eco = self.ecology
        aw, ah = self.aw, self.ah
        new = np.zeros((ah, aw), dtype=np.float32)
        if 0 < delta < aw:
            new[:, :aw - delta] = old_moisture[:, delta:]
        elif -aw < delta < 0:
            shift = -delta
            new[:, shift:] = old_moisture[:, :aw - shift]
        # 新进入窗口的列（窗口整体换掉时就是全部列）
        if delta > 0:
            new_columns = np.arange(max(0, aw - delta), aw)
        elif delta < 0:
            new_columns = np.arange(0, min(aw, -delta))
        else:  # pragma: no cover - recenter(0) 会提前返回
            new_columns = np.arange(0, 0)
        if new_columns.size:
            self._seed_moisture(new, new_columns)
        mask = (self.grid == MATERIAL_INDEX["soil"]) | (self.grid == MATERIAL_INDEX["grass"])
        new = (new * mask).astype(np.float32)
        eco.moisture = new
        eco.aw = aw
        eco.ah = ah
        eco.surface = self.surface
        eco.ground_row = self.ground_row
        # 碎屑是窗口局部坐标，跟着窗口一起平移；出界的碎屑由 update 自然清掉
        for particle in eco.debris:
            particle.x -= delta
        # 窗口内容变了：水位基准取新窗口的水原子数，water_level 归 1.0
        eco._initial_water = max(1, int((self.grid == MATERIAL_INDEX["water"]).sum()))
        eco.water_level = 1.0

    def _seed_moisture(self, moisture: np.ndarray, columns: np.ndarray) -> None:
        """给新进入窗口的列播种湿度（近似 Ecology._init_moisture 的规则）。"""
        mask = (self.grid == MATERIAL_INDEX["soil"]) | (self.grid == MATERIAL_INDEX["grass"])
        depth = np.clip(
            (np.arange(self.ah, dtype=np.float32) - self.surface)
            / float(max(1, self.ah - self.surface)),
            0.0,
            1.0,
        )
        left, right = self.pond
        pond_near = np.zeros(self.aw, dtype=bool)
        pond_near[max(0, left - 10):min(self.aw, right + 11)] = True
        base = np.where(
            mask,
            0.38 + 0.35 * pond_near[None, :] + 0.15 * depth[:, None],
            0.0,
        )
        moisture[:, columns] = np.minimum(1.0, base[:, columns]).astype(np.float32)

    # ------------------------------------------------------------------
    # 原子级修改（坐标是窗口局部；修改按全局坐标持久化）
    # ------------------------------------------------------------------
    def set_atom(self, x: int, y: int, material: str) -> None:
        """把原子 (x, y) 改成某材料（越界忽略；记录到全局修改表）。"""
        if not (0 <= x < self.aw and 0 <= y < self.ah):
            return
        index = MATERIAL_INDEX[material]
        self.grid[y, x] = index
        self._edits[(self.window_start_atom + x, y)] = index
        self._optics_cache = None

    def clear_atom(self, x: int, y: int) -> None:
        """清空原子 (x, y)（变成空气；记录到全局修改表）。"""
        if not (0 <= x < self.aw and 0 <= y < self.ah):
            return
        self.grid[y, x] = 0
        self._edits[(self.window_start_atom + x, y)] = 0
        self._optics_cache = None

    def atom_material(self, x: int, y: int) -> Material | None:
        """查询单个原子的材料（越界返回 None；空气格返回 air 材料）。

        空气格返回 ``MATERIAL_BY_INDEX[0]``（air）而不是 None，与
        :meth:`newworld.pixelworld.PixelScene.atom_material` 的实际行为一致。
        """
        if 0 <= x < self.aw and 0 <= y < self.ah:
            return MATERIAL_BY_INDEX.get(int(self.grid[y, x]))
        return None

    def add_berry(self, x: int, y: int) -> bool:
        """在树叶原子上结一颗浆果（生态再生用），成功返回 True。

        与 PixelScene 的差异：成功时把"这里有一颗浆果"记进全局修改表，
        所以生态在窗口 A 里再生的浆果，切到窗口 B 再切回来仍然在。
        """
        if not (0 <= x < self.aw and 0 <= y < self.ah):
            return False
        if self.grid[y, x] != MATERIAL_INDEX["leaf"]:
            return False
        self.grid[y, x] = MATERIAL_INDEX["berry"]
        if (x, y) not in self.berries:
            self.berries.append((x, y))
        self._edits[(self.window_start_atom + x, y)] = MATERIAL_INDEX["berry"]
        return True

    def pick_berry(self, berry: tuple[int, int]) -> bool:
        """摘走一颗浆果：原子变回树叶，等待生态再生（按全局坐标持久化）。"""
        if berry not in self.berries:
            return False
        x, y = berry
        self.grid[y, x] = MATERIAL_INDEX["leaf"]
        self.berries.remove(berry)
        for obj in self.objects:
            if berry in obj.berries:
                obj.berries.remove(berry)
        self._edits[(self.window_start_atom + x, y)] = MATERIAL_INDEX["leaf"]
        self.solid = self.grid > 0
        self.blocking = self._blocking_mask()
        self.albedo_rgb = self._albedo_rgb()
        self._optics_cache = None
        return True

    # ------------------------------------------------------------------
    # 查询（与 PixelScene 同名同义，坐标均为窗口局部）
    # ------------------------------------------------------------------
    def solid_at(self, atom_x: int, atom_y: int) -> bool:
        """原子 (x, y) 是否实心（非空气）。"""
        if 0 <= atom_x < self.aw and 0 <= atom_y < self.ah:
            return bool(self.grid[atom_y, atom_x])
        return False

    def material_at(self, atom_x: int, atom_y: int) -> Material | None:
        """查原子 (x, y) 的材料（越界返回 None；与 atom_material 同义）。"""
        if 0 <= atom_x < self.aw and 0 <= atom_y < self.ah:
            return MATERIAL_BY_INDEX.get(int(self.grid[atom_y, atom_x]))
        return None

    def is_blocked(
        self,
        x_screen: float,
        body_width_screen: float,
        feet_y_screen: float,
        body_height_screen: float,
    ) -> bool:
        """一个身体大小的盒子是否与石头/木头重叠（参数全为屏幕像素）。"""
        half = body_width_screen / 2.0
        x0 = int((x_screen - half) / ATOM)
        x1 = int((x_screen + half) / ATOM)
        y0 = int((feet_y_screen - body_height_screen) / ATOM)
        y1 = int((feet_y_screen - 2.0) / ATOM)
        x0 = max(0, min(self.aw - 1, x0))
        x1 = max(0, min(self.aw - 1, x1))
        y0 = max(0, min(self.ah - 1, y0))
        y1 = max(0, min(self.ah - 1, y1))
        return bool(self.blocking[y0:y1 + 1, x0:x1 + 1].any())

    def berry_at(self, atom_x: int, atom_y: int, radius_atoms: float = 2.5) -> tuple[int, int] | None:
        """找半径内最近的一颗浆果（窗口局部原子坐标），没有返回 None。"""
        best = None
        best_distance = radius_atoms * radius_atoms
        for berry in self.berries:
            distance = (berry[0] - atom_x) ** 2 + (berry[1] - atom_y) ** 2
            if distance <= best_distance:
                best_distance = distance
                best = berry
        return best

    def water_near(self, atom_x: int, atom_y: int, radius_atoms: float = 3.0) -> bool:
        """半径内是否有水原子（喝水前的检测；窗口局部坐标）。"""
        x0 = max(0, atom_x - int(radius_atoms))
        x1 = min(self.aw - 1, atom_x + int(radius_atoms))
        y0 = max(0, atom_y - int(radius_atoms))
        y1 = min(self.ah - 1, atom_y + int(radius_atoms))
        if x1 < x0 or y1 < y0:
            return False
        return bool((self.grid[y0:y1 + 1, x0:x1 + 1] == MATERIAL_INDEX["water"]).any())

    def surface_screen_y(self, x_screen: float) -> float:
        """屏幕 x 处地表的屏幕 y 坐标（站立/落点计算）。"""
        x = int(x_screen / ATOM)
        x = max(0, min(self.aw - 1, x))
        return float((self.top_row[x] + 1) * ATOM)

    # ------------------------------------------------------------------
    # 光照 / 生态 / 踩踏
    # ------------------------------------------------------------------
    def refresh_lighting(self, day: DayCycle, hour: float, elevation_deg: float) -> None:
        """计算地面阴影掩码（与 PixelScene 相同的逐列高度投影）。"""
        if elevation_deg <= 1.0:
            self.shadow_ground[:] = False
            return
        shadow = np.zeros(self.aw, dtype=bool)
        sun_dir = -1 if day.sun_horizontal_frac(hour) >= 0.5 else 1
        tan_elev = math.tan(math.radians(min(80.0, elevation_deg)))
        surface = self.surface
        for x in range(self.aw):
            top = int(self.top_row[x])
            height = surface - top
            if height <= 1:  # 草簇太矮，不产生可见阴影
                continue
            length = min(self.aw, int(height / max(0.08, tan_elev)))
            for step in range(1, length + 1):
                target = x + sun_dir * step
                if 0 <= target < self.aw:
                    shadow[target] = True
        self.shadow_ground = shadow

    def update_ecology(self, dt: float, day: DayCycle, hour: float) -> None:
        """推进生态（土壤水分、草/树生长、降雨、碎屑）；world.step 每帧调用。"""
        self.ecology.update(dt, self, day, hour)

    def trample(self, x_screen: float, y_screen: float, force: float = 1.4) -> int:
        """踩踏破坏原子（角色落地的效果），返回破坏的原子数。

        ``Ecology.trample`` 会调用本类的 ``clear_atom``，因此破坏同样按全局
        坐标持久化。
        """
        return self.ecology.trample(self, x_screen, y_screen, force)

    # ------------------------------------------------------------------
    # 渲染 / 光学 / 文本
    # ------------------------------------------------------------------
    def render_atoms(self, day: DayCycle, hour: float) -> np.ndarray:
        """渲染窗口为原子 RGB 图像 (ah, aw, 3)，与 PixelScene 同形同义。"""
        elevation = day.sun_elevation_deg(hour)
        daylight = day.daylight(hour)
        ambient = day.ambient_light(hour)
        sky = np.array(day.sky_colour(hour), dtype=np.float64)
        self.refresh_lighting(day, hour, elevation)

        rgb = self.albedo_rgb.astype(np.float64) / 255.0
        solid = self.solid
        shadowed = np.zeros((self.ah, self.aw), dtype=bool)
        if 0 <= self.surface < self.ah:
            shadowed[self.surface, self.shadow_ground] = True
        light = np.where(solid, ambient + daylight * 0.9, 1.0)
        light = np.where(shadowed, light * 0.45, light)
        lit = rgb * light[:, :, None]
        # 湿土壤/湿草更暗
        moisture = self.ecology.moisture
        wet = np.clip(moisture, 0.0, 1.0)[:, :, None]
        lit = lit * (1.0 - 0.22 * wet)
        # 雨云压暗天空
        rain = self.ecology.rain
        if rain > 0.05:
            sky = sky * (1.0 - 0.35 * rain)
        out = np.where(solid[:, :, None], lit, sky / 255.0)
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

        # 太阳 / 月亮（星星与雨点用固定种子，保证帧间稳定不闪烁）
        x_frac, y_frac = day.sun_screen_fraction(hour)
        cx = int(x_frac * (self.aw - 1))
        cy = int(y_frac * (self.ah - 1))
        if elevation > 0:
            self._draw_disc(out, cx, cy, 3, (255, 232, 150))
            self._draw_disc(out, cx, cy, 2, (255, 248, 200))
        elif rain <= 0.05:
            self._draw_disc(out, cx, cy, 3, (208, 214, 226))
            self._draw_disc(out, cx - 1, cy - 1, 2, (170, 178, 196))
            rng = np.random.default_rng(7)
            for _ in range(14):
                sx = int(rng.integers(0, self.aw))
                sy = int(rng.integers(0, max(4, int(self.ah * 0.35))))
                if not solid[sy, sx]:
                    out[sy, sx] = (200, 205, 225)
        if rain > 0.05:
            rng = np.random.default_rng(11)
            for _ in range(int(rain * 120)):
                rx = int(rng.integers(0, max(1, self.aw - 2)))
                ry = int(rng.integers(0, max(1, self.ah - 2)))
                if not solid[ry, rx]:
                    out[ry, rx] = (150, 180, 210)
                    if rx + 1 < self.aw:
                        out[ry, rx + 1] = (120, 150, 185)
                    if ry + 1 < self.ah:
                        out[ry + 1, rx] = (120, 150, 185)
        for particle in self.ecology.debris:
            px = int(particle.x)
            py = int(particle.y)
            if 0 <= px < self.aw and 0 <= py < self.ah:
                material = MATERIALS.get(particle.material)
                if material is not None:
                    out[py, px] = material.albedo
        return out

    @staticmethod
    def _draw_disc(rgb: np.ndarray, cx: int, cy: int, radius: int, colour: tuple[int, int, int]) -> None:
        """在 RGB 图像上画实心圆（太阳/月亮用）。"""
        height, width = rgb.shape[:2]
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius:
                    if 0 <= x < width and 0 <= y < height:
                        rgb[y, x] = colour

    def optics_scene(self) -> Scene:
        """把窗口内的天然物体翻译成光学层的几何形状（局部屏幕像素）。

        与 PixelScene 相同的映射：树 → 树干 Box + 树冠 Disc；岩石/浆果丛 →
        Disc；池塘 → 水面 Box；地面 → 土壤 Box。结果缓存，直到窗口平移或
        原子被修改才重建。
        """
        if self._optics_cache is not None:
            return self._optics_cache
        objects: list[Box | Disc] = []
        for obj in self.objects:
            if obj.kind == "tree":
                x_c = (obj.x0 + obj.x1) / 2.0 * ATOM
                trunk_w = ATOM * 3
                trunk_top = obj.top * ATOM
                trunk_bottom = obj.bottom * ATOM
                objects.append(
                    Box(x_c - trunk_w / 2, trunk_top, x_c + trunk_w / 2, trunk_bottom,
                        albedo=(0.43, 0.29, 0.19))
                )
                objects.append(
                    Disc(x_c, trunk_top + ATOM * 3, (obj.x1 - obj.x0) * ATOM * 0.45,
                         albedo=(0.31, 0.54, 0.25))
                )
            elif obj.kind in ("rock", "bush"):
                x_c = (obj.x0 + obj.x1) / 2.0 * ATOM
                y_c = (obj.top + obj.bottom) / 2.0 * ATOM
                radius = max(ATOM, (obj.x1 - obj.x0) * ATOM * 0.5)
                colour = tuple(channel / 255.0 for channel in obj.albedo)
                objects.append(Disc(x_c, y_c, radius, albedo=colour))  # type: ignore[arg-type]
        # 池塘水面
        left, right = self.pond
        objects.append(
            Box(left * ATOM, (self.pond_surface + 0.2) * ATOM, right * ATOM, (self.pond_surface + 3) * ATOM,
                albedo=(0.19, 0.38, 0.60))
        )
        # 地面条带
        objects.append(
            Box(0, (self.surface + 1) * ATOM, self.aw * ATOM, self.ah * ATOM, albedo=(0.34, 0.25, 0.17))
        )
        self._optics_cache = Scene(objects=tuple(objects))
        return self._optics_cache

    def material_composition_text(self) -> str:
        """窗口材料构成摘要（前 8 种，按原子数排序），UI 展示用。

        注意：统计的是**窗口**而不是整个世界（世界横向无界，无法全量统计）。
        """
        counts: dict[str, int] = {}
        for material in MATERIAL_LIST:
            index = MATERIAL_INDEX[material.key]
            total = int((self.grid == index).sum())
            if total:
                counts[material.key] = total
        parts = [
            f"{material_name_zh(key)} {count}"
            for key, count in sorted(counts.items(), key=lambda item: -item[1])
        ]
        return "，".join(parts[:8])

    def material_name_zh(self, key: str) -> str:
        """按 key 取材料中文名（实例方法，便于 UI 层统一调用）。"""
        return material_name_zh(key)
