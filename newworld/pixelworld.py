"""像素原子世界：地形、天然物体、光照与阴影。

每个可见像素方块就是一个原子（5×5 屏幕像素）。世界的数据结构是
**材料编号网格**（uint8，0 = 空气）加上一组天然 :class:`PixelObject`
（岩石、树、浆果丛、草、花、池塘）。物体不是美术资源，而是由世界自身
的地质/生物规则生成：

* 重的石头沉到轻的土壤之下；土壤支撑植物；
* 水汇集在低洼处（池塘）；植物需要土壤并向着太阳生长；
* 树/草/浆果丛由木头/树叶/浆果原子构成；浆果可食用。

本模块还提供：碰撞查询（脚/手对原子）、摘浆果、找水、给眼睛用的
光学场景（optics_scene），以及带昼夜光照/阴影的原子级 RGB 渲染。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from . import materials
from .daynight import DayCycle
from .materials import MATERIAL_BY_INDEX, MATERIAL_INDEX, MATERIAL_LIST, MATERIALS, Material
from .optics import Box, Disc, Scene

ATOM = materials.ATOM_SIZE


@dataclass
class PixelObject:
    """一个由像素原子构成的天然物体（岩石/树/浆果丛等）。"""

    kind: str
    name_zh: str
    x0: int
    x1: int
    top: int
    bottom: int
    albedo: tuple[int, int, int]
    atom_count: int
    thermal_mass_j_k: float
    berries: list[tuple[int, int]] = field(default_factory=list)
    berry_slots: list[tuple[int, int]] = field(default_factory=list)
    #: 是否阻挡行走（石头/树干阻挡；浆果丛可穿过）
    blocking: bool = False

    @property
    def width(self) -> int:
        """物体宽度（原子列数）。"""
        return self.x1 - self.x0


class PixelScene:
    """材料编号网格 + 天然物体 + 物理/光学查询。

    这是"世界的一半"：纯数据与规则，不包含角色和文明。坐标约定：
    网格索引 ``(y, x)`` 单位是**原子**；对外接口多用**屏幕像素**，
    内部按 ``ATOM`` 换算。
    """

    def __init__(self, config, seed: int = 0) -> None:
        """初始化网格并生成世界。"""
        self.config = config
        self.rng = random.Random(seed)
        # 网格尺寸（原子数）：屏幕分辨率 / ATOM_SIZE，至少 16×16
        self.aw = max(16, config.width // ATOM)
        self.ah = max(16, config.height // ATOM)
        # 地面行（原子行号）：ground_y 屏幕像素 → 原子行
        self.ground_row = int(round(config.ground_y / ATOM))
        self.surface = self.ground_row - 1
        self.grid = np.zeros((self.ah, self.aw), dtype=np.uint8)  # 0 = 空气
        self.objects: list[PixelObject] = []
        self.berries: list[tuple[int, int]] = []
        self.terrain_bottom = self.ah - 1
        self.generate()
        # 生成后建立四个派生缓存（网格改完必须调 refresh_caches 重建）
        self.solid = self.grid > 0
        self.blocking = self._blocking_mask()
        self.top_row = self._top_rows()
        self.albedo_rgb = self._albedo_rgb()
        self.shadow_ground = np.zeros(self.aw, dtype=bool)
        self._refresh_lighting_cache = None
        from .ecology import Ecology

        self.ecology = Ecology(self)

    # ------------------------------------------------------------------
    # 画格子的底层工具（生成期使用）
    # ------------------------------------------------------------------
    def _set(self, x: int, y: int, material: str) -> None:
        """把单个原子设为某材料（越界忽略）。"""
        if 0 <= x < self.aw and 0 <= y < self.ah:
            self.grid[y, x] = MATERIAL_INDEX[material]

    def _fill(self, x0: int, x1: int, y0: int, y1: int, material: str) -> None:
        """矩形填充（自动裁剪到网格内）。"""
        x0 = max(0, x0)
        x1 = min(self.aw - 1, x1)
        y0 = max(0, y0)
        y1 = min(self.ah - 1, y1)
        if x1 < x0 or y1 < y0:
            return
        self.grid[y0:y1 + 1, x0:x1 + 1] = MATERIAL_INDEX[material]

    def _disc(self, cx: float, cy: float, radius: float, material: str) -> int:
        """填充一个实心圆盘（树冠/岩石用），返回填充的原子数。"""
        count = 0
        radius_sq = radius * radius
        for y in range(int(cy - radius), int(cy + radius) + 1):
            for x in range(int(cx - radius), int(cx + radius) + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius_sq:
                    if 0 <= x < self.aw and 0 <= y < self.ah:
                        self.grid[y, x] = MATERIAL_INDEX[material]
                        count += 1
        return count

    # ------------------------------------------------------------------
    # 世界生成（一次性，由 seed 决定，可复现）
    # ------------------------------------------------------------------
    def generate(self) -> None:
        """按地质/生物规则生成整个场景。

        顺序：基岩-土壤-草皮 → 池塘 → 岩石 → 树 → 浆果丛 → 草簇/花，
        最后 :meth:`_ensure_food_supply` 兜底保证食物网存在。
        """
        rng = self.rng
        aw, ah = self.aw, self.ah
        surface = self.surface
        # -- 地质分层：深处石头，其上土壤，表层草皮 --------------------
        self._fill(0, aw - 1, surface + 3, ah - 1, "stone")
        self._fill(0, aw - 1, surface + 1, surface + 3, "soil")
        self._fill(0, aw - 1, surface, surface, "grass")

        # -- 池塘（水填满一个浅洼地） ----------------------------------
        pond_left = rng.randint(aw // 8, aw // 3)
        pond_width = rng.randint(18, 30)
        pond_right = min(aw - 2, pond_left + pond_width)
        for x in range(pond_left, pond_right + 1):
            # 正弦洼地：中间深、两边浅
            depth = 2 + int(1.4 * math.sin(math.pi * (x - pond_left) / max(1, pond_width)))
            self._set(x, surface, "water")
            for y in range(surface + 1, surface + 1 + depth):
                self._set(x, y, "water")
            self._set(x, surface + 1 + depth, "soil")
        self.pond = (pond_left, pond_right)
        self.pond_surface = surface

        # -- 岩石落在土壤上（避开池塘） --------------------------------
        for _ in range(rng.randint(5, 8)):
            cx = rng.randint(6, aw - 7)
            if pond_left - 4 <= cx <= pond_right + 4:
                continue
            radius = rng.uniform(2.0, 4.2)
            count = self._disc(cx, surface - radius * 0.55, radius, "stone")
            if count:
                self._add_object(
                    "rock", "岩石", int(cx - radius), int(cx + radius),
                    int(surface - 2 * radius), surface, (126, 126, 130), count,
                    2700.0, 790.0, blocking=True,
                )

        # -- 树（木头树干 + 树叶树冠，向上生长） -----------------------
        reserved = set(range(pond_left - 6, pond_right + 7))
        tree_positions = []
        for _ in range(rng.randint(3, 5)):
            for _attempt in range(30):
                cx = rng.randint(16, aw - 17)
                if cx in reserved or any(abs(cx - other) < 30 for other in tree_positions):
                    continue
                tree_positions.append(cx)
                trunk_h = rng.randint(22, 34)
                trunk_w = rng.choice((3, 3, 4))
                self._fill(cx - trunk_w // 2, cx + trunk_w // 2, surface - trunk_h, surface, "wood")
                canopy_r = rng.uniform(8.0, 12.0)
                canopy_count = self._disc(cx, surface - trunk_h - 2, canopy_r, "leaf")
                trunk_count = (trunk_w + 1) * (trunk_h + 1)
                self._add_object(
                    "tree", "树",
                    int(cx - trunk_w // 2),
                    int(cx + trunk_w // 2 + canopy_r * 2),
                    int(surface - trunk_h - 2 - canopy_r),
                    surface,
                    (78, 128, 62), trunk_count + canopy_count, 600.0, 1700.0,
                    blocking=True,
                )
                break

        # -- 浆果丛（树叶 + 可食用浆果） -------------------------------
        # 注意：下面的随机摆放在小世界上可能 30 次全部失败（实测 480x300
        # 与 800x500 生成 **0** 个浆果丛 = 完全没有食物，任何机制都救不
        # 回全员饿死的结局）。``_ensure_food_supply`` 在其后兜底，保证
        # 任何世界尺寸下食物网都存在。
        for _ in range(rng.randint(3, 5)):
            for _attempt in range(30):
                cx = rng.randint(12, aw - 13)
                if cx in reserved or any(abs(cx - other) < 22 for other in tree_positions):
                    continue
                reserved.add(cx)
                self._place_bush(cx, rng.uniform(3.0, 5.0), rng)
                break

        # -- 草簇与"花" ------------------------------------------------
        for _ in range(rng.randint(30, 48)):
            x = rng.randint(2, aw - 3)
            if self.grid[surface, x] != MATERIAL_INDEX["grass"]:
                continue
            self._fill(x, x + rng.choice((1, 1, 2)), surface - 1, surface - 1, "grass")
        for _ in range(rng.randint(10, 18)):
            # "花" = 草茎 + 顶端一颗浆果色原子
            x = rng.randint(3, aw - 4)
            if self.grid[surface, x] != MATERIAL_INDEX["grass"]:
                continue
            self._set(x, surface - 1, "grass")
            self._set(x, surface - 2, "grass")
            self._set(x, surface - 3, "berry")

        self._ensure_food_supply(reserved, tree_positions, rng)

    # ------------------------------------------------------------------
    def _place_bush(self, cx: int, radius: float, rng) -> int:
        """在原子列 ``cx`` 放一个浆果丛；返回新增浆果数。"""
        surface = self.surface
        leaf_count = self._disc(cx, surface - radius * 0.8, radius, "leaf")
        berry_positions: list[tuple[int, int]] = []
        for _b in range(rng.randint(2, 5)):
            bx = int(cx + rng.uniform(-radius * 0.7, radius * 0.7))
            by = int(surface - radius * 0.8 + rng.uniform(-radius * 0.5, radius * 0.5))
            if not (0 <= bx < self.aw and 0 <= by < self.ah):
                continue
            # 浆果只能长在树叶原子上（被采走后原位变回 leaf，可再结果）
            if self.grid[by, bx] == MATERIAL_INDEX["leaf"]:
                self._set(bx, by, "berry")
                berry_positions.append((bx, by))
        self.berries.extend(berry_positions)
        self._add_object(
            "bush", "浆果丛", int(cx - radius), int(cx + radius),
            int(surface - 2 * radius), surface, (74, 138, 66), leaf_count,
            260.0, 1600.0, blocking=False, berries=berry_positions,
        )
        # berry_slots 记住"哪里长过浆果"，生态再生时在原位补
        self.objects[-1].berry_slots = list(berry_positions)
        return len(berry_positions)

    def _ensure_food_supply(self, reserved, tree_positions, rng) -> None:
        """兜底保证最少数量的可食用植物（食物网必须存在）。

        最少数量按**面积**（而非宽度）缩放，让实验用的小世界也能活、
        全屏世界又不至于变成浆果农场：``max(2, round(原子面积 / 40000))``。
        """
        wanted = max(2, round(self.aw * self.ah / 40000.0))
        placed = sum(1 for obj in self.objects if obj.kind == "bush")
        if placed >= wanted:
            return
        for _ in range((wanted - placed) * 40):
            if placed >= wanted:
                break
            cx = rng.randint(12, self.aw - 13)
            if cx in reserved:
                continue
            if any(abs(cx - other) < 12 for other in tree_positions):
                continue
            if any(abs(cx - int((obj.x0 + obj.x1) / 2)) < 18
                   for obj in self.objects if obj.kind == "bush"):
                continue
            reserved.add(cx)
            if self._place_bush(cx, rng.uniform(3.0, 5.0), rng) > 0:
                placed += 1

    def _add_object(
        self,
        kind: str,
        name_zh: str,
        x0: int,
        x1: int,
        top: int,
        bottom: int,
        albedo: tuple[int, int, int],
        atom_count: int,
        density: float,
        specific_heat: float,
        *,
        blocking: bool = False,
        berries: list[tuple[int, int]] | None = None,
    ) -> None:
        """登记一个天然物体并计算热质量（体积×密度×比热）。"""
        volume = max(0, atom_count) * (0.05 ** 3)  # 每个原子 = 5 cm 的立方体
        mass = volume * density
        self.objects.append(
            PixelObject(
                kind=kind,
                name_zh=name_zh,
                x0=max(0, x0),
                x1=min(self.aw - 1, x1),
                top=max(0, top),
                bottom=min(self.ah - 1, bottom),
                albedo=albedo,
                atom_count=atom_count,
                thermal_mass_j_k=mass * specific_heat,
                berries=list(berries or ()),
                blocking=blocking,
            )
        )

    # ------------------------------------------------------------------
    # 原子级修改（生态模块用：生长、破坏、水）
    # ------------------------------------------------------------------
    def set_atom(self, x: int, y: int, material: str) -> None:
        """把原子 (x, y) 改成某材料（越界忽略；不刷新缓存）。"""
        if 0 <= x < self.aw and 0 <= y < self.ah:
            self.grid[y, x] = MATERIAL_INDEX[material]

    def clear_atom(self, x: int, y: int) -> None:
        """清空原子 (x, y)（变成空气；不刷新缓存）。"""
        if 0 <= x < self.aw and 0 <= y < self.ah:
            self.grid[y, x] = 0

    def atom_material(self, x: int, y: int) -> Material | None:
        """查询单个原子的材料（越界/空气返回 None）。"""
        if 0 <= x < self.aw and 0 <= y < self.ah:
            return MATERIAL_BY_INDEX.get(int(self.grid[y, x]))
        return None

    def refresh_caches(self) -> None:
        """重建派生缓存（solid/blocking/top_row/albedo）。原子级修改后必须调用。"""
        self.solid = self.grid > 0
        self.blocking = self._blocking_mask()
        self.top_row = self._top_rows()
        self.albedo_rgb = self._albedo_rgb()

    def add_berry(self, x: int, y: int) -> bool:
        """在树叶原子上结一颗浆果（生态再生用），成功返回 True。"""
        if not (0 <= x < self.aw and 0 <= y < self.ah):
            return False
        if self.grid[y, x] != MATERIAL_INDEX["leaf"]:
            return False
        self.grid[y, x] = MATERIAL_INDEX["berry"]
        if (x, y) not in self.berries:
            self.berries.append((x, y))
        return True

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def _blocking_mask(self) -> np.ndarray:
        """硬阻挡掩码：只有石头和木头挡身体（土壤/草/树叶可踩）。"""
        mask = np.zeros_like(self.solid)
        for key in ("stone", "wood"):
            mask |= self.grid == MATERIAL_INDEX[key]
        return mask

    def _top_rows(self) -> np.ndarray:
        """每列最上方实心原子的行号（整列为空时 = ah）。用于落点/阴影计算。"""
        solid = self.grid > 0
        rows = np.full(self.aw, self.ah, dtype=np.int32)
        for x in range(self.aw):
            column = np.nonzero(solid[:, x])[0]
            if column.size:
                rows[x] = int(column[0])
        return rows

    def _albedo_rgb(self) -> np.ndarray:
        """把材料编号网格翻译成 RGB 图像（ah × aw × 3，uint8）。"""
        table = np.zeros((len(MATERIAL_LIST), 3), dtype=np.uint8)
        for index, material in MATERIAL_BY_INDEX.items():
            table[index] = material.albedo
        return table[self.grid]

    # -- 碰撞 ------------------------------------------------------
    def solid_at(self, atom_x: int, atom_y: int) -> bool:
        """原子 (x, y) 是否实心（非空气）。"""
        if 0 <= atom_x < self.aw and 0 <= atom_y < self.ah:
            return bool(self.grid[atom_y, atom_x])
        return False

    def material_at(self, atom_x: int, atom_y: int) -> Material | None:
        """查原子 (x, y) 的材料（与 atom_material 相同语义）。"""
        if 0 <= atom_x < self.aw and 0 <= atom_y < self.ah:
            return MATERIAL_BY_INDEX.get(int(self.grid[atom_y, atom_x]))
        return None

    def is_blocked(self, x_screen: float, body_width_screen: float, feet_y_screen: float, body_height_screen: float) -> bool:
        """一个身体大小的盒子是否与石头/木头重叠（硬障碍检测）。

        参数全部是屏幕像素；内部换算成原子格再查 blocking 掩码。
        """
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
        """找半径内最近的一颗浆果（原子坐标），没有返回 None。"""
        best = None
        best_d = radius_atoms * radius_atoms
        for bx, by in self.berries:
            distance = (bx - atom_x) ** 2 + (by - atom_y) ** 2
            if distance <= best_d:
                best_d = distance
                best = (bx, by)
        return best

    def pick_berry(self, berry: tuple[int, int]) -> bool:
        """摘走一颗浆果：原子变回树叶，等待生态再生。"""
        if berry not in self.berries:
            return False
        x, y = berry
        self.grid[y, x] = MATERIAL_INDEX["leaf"]
        self.berries.remove(berry)
        for obj in self.objects:
            if berry in obj.berries:
                obj.berries.remove(berry)
        self.solid = self.grid > 0
        self.blocking = self._blocking_mask()
        self.albedo_rgb = self._albedo_rgb()
        return True

    def water_near(self, atom_x: int, atom_y: int, radius_atoms: float = 3.0) -> bool:
        """半径内是否有水原子（喝水前的检测）。"""
        x0 = max(0, atom_x - int(radius_atoms))
        x1 = min(self.aw - 1, atom_x + int(radius_atoms))
        y0 = max(0, atom_y - int(radius_atoms))
        y1 = min(self.ah - 1, atom_y + int(radius_atoms))
        return bool((self.grid[y0:y1 + 1, x0:x1 + 1] == MATERIAL_INDEX["water"]).any())

    def surface_screen_y(self, x_screen: float) -> float:
        """屏幕 x 处地表的屏幕 y 坐标（站立/落点计算）。"""
        x = int(x_screen / ATOM)
        x = max(0, min(self.aw - 1, x))
        return (self.top_row[x] + 1) * ATOM

    def refresh_lighting(self, day: DayCycle, hour: float, elevation_deg: float) -> None:
        """计算地面阴影掩码（基于高度图的光线投射：高列向背光方向投影）。"""
        if elevation_deg <= 1.0:
            self.shadow_ground[:] = False
            return
        shadow = np.zeros(self.aw, dtype=bool)
        # 太阳在屏幕哪半边，阴影就投向另一边
        sun_dir = -1 if day.sun_horizontal_frac(hour) >= 0.5 else 1  # 阴影背向太阳
        tan_elev = math.tan(math.radians(min(80.0, elevation_deg)))
        for x in range(self.aw):
            top = int(self.top_row[x])
            height = self.surface - top
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
        """踩踏破坏原子（角色落地的效果），返回破坏的原子数。"""
        return self.ecology.trample(self, x_screen, y_screen, force)

    def render_atoms(self, day: DayCycle, hour: float) -> np.ndarray:
        """渲染世界为原子 RGB 图像 (H, W, 3)。

        分四步：基础反照率 → 昼夜光照（环境光 + 太阳光，阴影再打折）
        → 湿度变暗 + 雨云压暗天空 → 叠加太阳/月亮/星星/雨点/碎屑。
        """
        elevation = day.sun_elevation_deg(hour)
        daylight = day.daylight(hour)
        ambient = day.ambient_light(hour)
        sky = np.array(day.sky_colour(hour), dtype=np.float64)
        self.refresh_lighting(day, hour, elevation)

        rgb = self.albedo_rgb.astype(np.float64) / 255.0
        solid = self.solid
        shadowed = np.zeros((self.ah, self.aw), dtype=bool)
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
        # 空气/透明像素填天空色
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
            # 夜晚：月亮 + 星星
            self._draw_disc(out, cx, cy, 3, (208, 214, 226))
            self._draw_disc(out, cx - 1, cy - 1, 2, (170, 178, 196))
            rng = np.random.default_rng(7)
            for _ in range(14):
                sx = int(rng.integers(0, self.aw))
                sy = int(rng.integers(0, max(4, int(self.ah * 0.35))))
                if not solid[sy, sx]:
                    out[sy, sx] = (200, 205, 225)
        if rain > 0.05:
            # 雨：一条条短斜线
            rng = np.random.default_rng(11)
            for _ in range(int(rain * 120)):
                rx = int(rng.integers(0, self.aw - 2))
                ry = int(rng.integers(0, self.ah - 2))
                if not solid[ry, rx]:
                    out[ry, rx] = (150, 180, 210)
                    out[ry, rx + 1] = (120, 150, 185)
                    out[ry + 1, rx] = (120, 150, 185)
        for particle in self.ecology.debris:
            # 生态碎屑（被踩飞的原子）直接画上
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

    # ------------------------------------------------------------------
    # 光学（眼睛看到的东西）
    # ------------------------------------------------------------------
    def optics_scene(self) -> Scene:
        """把天然物体翻译成光学层的几何形状（Box/Disc，屏幕像素坐标）。

        树 → 树干 Box + 树冠 Disc；岩石/浆果丛 → Disc；池塘 → 水面 Box；
        地面 → 土壤 Box。结果缓存（物体静止，无需每帧重建）。
        """
        if getattr(self, "_optics_cache", None) is not None:
            return self._optics_cache
        objects = []
        for obj in self.objects:
            if obj.kind == "tree":
                # 树干 Box + 树冠 Disc，单位屏幕像素
                x_c = (obj.x0 + obj.x1) / 2.0 * ATOM
                trunk_w = ATOM * 3
                trunk_top = obj.top * ATOM
                trunk_bottom = obj.bottom * ATOM
                objects.append(Box(x_c - trunk_w / 2, trunk_top, x_c + trunk_w / 2, trunk_bottom, albedo=(0.43, 0.29, 0.19)))
                objects.append(Disc(x_c, trunk_top + ATOM * 3, (obj.x1 - obj.x0) * ATOM * 0.45, albedo=(0.31, 0.54, 0.25)))
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
        """世界材料构成摘要（前 8 种，按原子数排序），UI 展示用。"""
        counts: dict[str, int] = {}
        for material in MATERIAL_LIST:
            index = MATERIAL_INDEX[material.key]
            total = int((self.grid == index).sum())
            if total:
                counts[material.key] = total
        parts = [f"{material_name_zh(key)} {count}" for key, count in sorted(counts.items(), key=lambda item: -item[1])]
        return "，".join(parts[:8])


def material_name_zh(key: str) -> str:
    """按 key 取材料中文名（本地转发，避免循环依赖）。"""
    from .materials import MATERIALS

    return MATERIALS[key].name_zh
