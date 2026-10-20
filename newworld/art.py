"""newworld 居民的像素艺术精灵。

精灵从用户的蓝图 PDF（``ref/blueprint.pdf``）重建。蓝图内嵌一张
480×480 的 PNG：一只站在绿幕背景上的像素画生物。画面分布在一个
13.25 px 间距、起点 (0, 3.3125) 的网格上；每个网格元取其主色、
移除绿色背景、裁剪到精灵包围盒。

得到的原生精灵是 23 × 29 艺术像素：

* ``O`` = 橙色身体 ``#E27E51``
* ``B`` = 蓝色思维气泡 ``#6699D0``
* ``K`` = 黑色眼睛 / 思考点 ``#000000``
* `` `` = 透明

``tools/extract_character.py`` 可以从 PDF 重新推导这份网格并校验
两者一致（``python tools/extract_character.py --check``）。
"""

from __future__ import annotations

from PIL import Image

# fmt: off
SPRITE_ROWS: tuple[str, ...] = (
    "             BBBBBBB   ",
    "            BBBBBBBBB  ",
    "            BBBBBBBBB  ",
    "            BBKBBBBBB  ",
    "            BBBBBBBBB  ",
    "            BBBBBBBBB  ",
    "             BBBBBBB   ",
    "               BBB     ",
    "               BB      ",
    "               B       ",
    "                       ",
    "                OOOO   ",
    "        OOOOOOOOOOOO   ",
    "    OOOOOOOOOOOOOOOO   ",
    "    OOOOOOOOOOOKKKOOOO ",
    "    OOKKKOOOOOOOOOOO OO",
    "    OOOOOOOOOOOOOOOO OO",
    "    OOOOOOOOOOOOOOOOOOO",
    "OOOOOOOOOOOOOOOOOOOOOOO",
    "OOOOOOOOOOOOOOOOOOOOOO ",
    "OOOOOOOOOOOOOOOOOOOO   ",
    "OOOOOOOOOOOOOOOOOOOO   ",
    "    OOOOOOOOOOOOOOOO   ",
    "    OOOOOOOOOOOOOOOO   ",
    "    OOOOOOOO  OO  OO   ",
    "    OO  OO    OO  OO   ",
    "    OO  OO    OO  OO   ",
    "    OO  OO    OO  OO   ",
    "    OO  OO    OO  OO   ",
)
# fmt: on

#: 精灵的宽 / 高（艺术像素数）
SPRITE_W: int = len(SPRITE_ROWS[0])
SPRITE_H: int = len(SPRITE_ROWS)

#: 蓝图用的调色板。Alpha 0 表示"透明"。
PALETTE: dict[str, tuple[int, int, int, int]] = {
    " ": (0, 0, 0, 0),
    "O": (226, 126, 81, 255),   # 身体橙
    "B": (102, 153, 208, 255),  # 思维气泡蓝
    "K": (0, 0, 0, 255),        # 眼睛 / 思考点
}

#: 人类可读的颜色名（调试工具用）
PALETTE_NAMES: dict[str, str] = {
    " ": "transparent",
    "O": "orange",
    "B": "blue",
    "K": "black",
}


def validate() -> None:
    """内嵌精灵数据格式不对时抛 ``ValueError``（import 时自检）。"""
    if not SPRITE_ROWS:
        raise ValueError("sprite has no rows")
    for index, row in enumerate(SPRITE_ROWS):
        if len(row) != SPRITE_W:
            raise ValueError(
                f"row {index} has {len(row)} cells, expected {SPRITE_W}"
            )
        unknown = set(row) - set(PALETTE)
        if unknown:
            raise ValueError(f"row {index} has unknown palette keys: {unknown}")


validate()


def sprite_image(cell: int = 1, facing: int = 1, variant: str = "male") -> Image.Image:
    """把精灵输出为 RGBA :class:`PIL.Image.Image`。

    ``cell`` 把每个艺术像素放大到 ``cell`` 个屏幕像素（最近邻，
    保持像素画观感）。``facing`` = -1 水平镜像精灵。
    """
    if cell < 1:
        raise ValueError(f"cell must be >= 1, got {cell}")
    palette = dict(PALETTE)
    if variant == "female":
        palette["O"] = (214, 116, 142, 255)  # 玫瑰色身体
        palette["B"] = (196, 154, 212, 255)  # 淡紫色思维气泡
    image = Image.new("RGBA", (SPRITE_W, SPRITE_H), (0, 0, 0, 0))
    pixels = image.load()
    assert pixels is not None
    for y, row in enumerate(SPRITE_ROWS):
        for x, key in enumerate(row):
            colour = palette[key]
            if colour[3]:
                pixels[x, y] = colour
    if cell != 1:
        image = image.resize((SPRITE_W * cell, SPRITE_H * cell), Image.NEAREST)
    if facing < 0:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def sprite_image_native(variant: str = "male") -> Image.Image:
    """原生 23×29 艺术像素分辨率的精灵。"""
    return sprite_image(1, 1, variant=variant)
