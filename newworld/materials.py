"""像素原子与材料定义（世界的"元素周期表"）。

在 newworld 中**一个原子（atom）= ``ATOM_SIZE`` 个屏幕像素**（可见的像素
方块），"世界的原子就是像素"由本模块落实。每种材料用一组物理属性描述：
元素组成（原子比例）、密度、比热、导热率、反照率（颜色）、硬度与相态。
场景中的物体和角色都由这些原子构成，因此它们的质量、热容、外观和行为
都源自同一份材料定义。

元素组成是**为仿真做的近似**，尽量贴近真实物质的体成分：水 = H2O、
石头 = SiO2、草/木头 = 纤维素、空气 = N2/O2/Ar、肉体 = 水 + 有机物、
骨骼 = 磷酸钙 + 胶原蛋白。
"""

from __future__ import annotations

from dataclasses import dataclass

#: 每个世界原子占多少屏幕像素（可见的像素艺术方块大小）
ATOM_SIZE = 5


@dataclass(frozen=True)
class Material:
    """一种由像素原子构成的物质（不可变数据类）。"""

    key: str
    name_zh: str
    #: 元素组成（原子比例），如水 = (("H", 2/3), ("O", 1/3))
    elements: tuple[tuple[str, float], ...]
    density_kg_m3: float
    specific_heat: float  # 比热 J/(kg K)
    thermal_conductivity: float  # 导热率 W/(m K)
    albedo: tuple[int, int, int]
    hardness: float  # 硬度 0..10（石头硬、草软）
    phase: str  # 相态："solid" | "liquid" | "gas"
    opacity: float = 1.0
    edible: bool = False
    drinkable: bool = False

    @property
    def formula(self) -> str:
        """把元素组成格式化成化学式风格的字符串（如 "H0.67 O0.33"）。"""
        parts = []
        for symbol, fraction in self.elements:
            parts.append(f"{symbol}{fraction:.2f}")
        return " ".join(parts)

    @property
    def elements_text(self) -> str:
        """把元素组成格式化成 "H 67%  O 33%" 的展示文本。"""
        return "  ".join(
            f"{symbol} {fraction * 100:.0f}%" for symbol, fraction in self.elements
        )

    @property
    def is_solid(self) -> bool:
        """是否固态。"""
        return self.phase == "solid"

    @property
    def is_fluid(self) -> bool:
        """是否流体（液态或气态）。"""
        return self.phase in ("liquid", "gas")

    @property
    def colour(self) -> tuple[int, int, int]:
        """反照率即渲染颜色（RGB）。"""
        return self.albedo


#: 全部材料注册表：key -> Material。世界的"物质清单"只有这一份。
MATERIALS: dict[str, Material] = {
    "air": Material(  # 空气：全透明（opacity=0），只作为传声/传热介质
        "air", "空气",
        (("N", 0.78), ("O", 0.21), ("Ar", 0.01)),
        1.225, 1005.0, 0.026, (205, 228, 242), 0.0, "gas", opacity=0.0,
    ),
    "water": Material(  # 水：唯一 drinkable 的材料（池塘/取水目标都指向它）
        "water", "水",
        (("H", 2 / 3), ("O", 1 / 3)),
        1000.0, 4186.0, 0.60, (48, 96, 152), 0.0, "liquid",
        drinkable=True,
    ),
    "soil": Material(  # 土壤：参与土壤水循环（湿度影响草/树生长）
        "soil", "土壤",
        (("Si", 0.30), ("O", 0.55), ("C", 0.12), ("H", 0.03)),
        1300.0, 840.0, 0.30, (108, 78, 52), 2.0, "solid",
    ),
    "stone": Material(  # 石头：高硬度，是石器（石斧等）的材料来源
        "stone", "石头",
        (("Si", 1 / 3), ("O", 2 / 3)),
        2700.0, 790.0, 2.50, (126, 126, 130), 8.0, "solid",
    ),
    "grass": Material(
        "grass", "草",
        (("C", 0.286), ("H", 0.476), ("O", 0.238)),
        700.0, 1300.0, 0.10, (86, 148, 68), 1.0, "solid",
    ),
    "wood": Material(  # 木头：建造/篝火/维护设施的主要材料
        "wood", "木头",
        (("C", 0.35), ("H", 0.40), ("O", 0.25)),
        600.0, 1700.0, 0.16, (110, 74, 48), 5.0, "solid",
    ),
    "leaf": Material(
        "leaf", "树叶",
        (("C", 0.30), ("H", 0.44), ("O", 0.26)),
        260.0, 1600.0, 0.08, (66, 138, 62), 0.5, "solid",
    ),
    "berry": Material(  # 浆果：唯一 edible 的天然食物（采集目标的核心资源）
        "berry", "浆果",
        (("C", 0.20), ("H", 0.48), ("O", 0.32)),
        850.0, 3000.0, 0.20, (172, 48, 66), 0.4, "solid", edible=True,
    ),
    "flesh": Material(  # 肉体：角色身体的主要成分（体温/热平衡按它算）
        "flesh", "肉体",
        (("H", 0.50), ("O", 0.30), ("C", 0.17), ("N", 0.03)),
        1000.0, 3500.0, 0.45, (226, 126, 81), 0.8, "solid",
    ),
    "bone": Material(  # 骨骼：角色骨骼，渲染层用来画身体骨架
        "bone", "骨骼",
        (("Ca", 0.15), ("P", 0.10), ("O", 0.45), ("H", 0.25), ("C", 0.05)),
        1900.0, 1300.0, 0.30, (222, 210, 186), 7.0, "solid",
    ),
}

#: 全代码通用的短名（M["water"] 这种写法）
M = MATERIALS

#: 材料编号 <-> 材料 的双向查找表（0 号 = air/空格子）。
#: 网格存储用整数编号，显示/查询时再换回 Material。
MATERIAL_LIST: tuple[Material, ...] = tuple(MATERIALS.values())
MATERIAL_INDEX: dict[str, int] = {material.key: index for index, material in enumerate(MATERIAL_LIST)}
MATERIAL_BY_INDEX: dict[int, Material] = {index: material for index, material in enumerate(MATERIAL_LIST)}


def material(key: str) -> Material:
    """按 key 取材料（如 material("water")）。"""
    return MATERIALS[key]


def material_name_zh(key: str) -> str:
    """按 key 取材料中文名（UI 展示用）。"""
    return MATERIALS[key].name_zh


