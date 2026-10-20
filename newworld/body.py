"""居民的像素身体：**蓝图原版生物**，带关节分解。

蓝图精灵（23 × 29 艺术像素）被原样保留：橙色身体、黑色眼睛、
蓝色思维气泡和四条小短腿。为了既保住原版外观、又有解剖结构和
可交互点，精灵被拆成几部分：

* 思维气泡 + 头/躯干（第 0..23 行，含眼睛和小手臂），
* 左腿对与右腿对（第 24..28 行），行走时摆动、跳跃时收起。

渲染器返回眼睛、嘴、手指和脚的原子位置；行为层用手指/脚去
触摸/吃/喝，用头/嘴去看/说话。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

from . import art
from .materials import ATOM_SIZE


@dataclass
class Pose:
    """身体的动画状态。"""

    state: str = "idle"  # idle, walk, jump, eat, drink, sleep
    phase: float = 0.0
    facing: int = 1


@dataclass
class BodyFrame:
    """一帧渲染好的身体 + 解剖交互点（精灵像素坐标）。"""

    image: Image.Image
    anchor: tuple[int, int]  # 脚中心（图像坐标）
    hands: tuple[tuple[int, int], ...]
    feet: tuple[tuple[int, int], ...]
    head_center: tuple[int, int]
    mouth: tuple[int, int]
    facing: int


#: 原生精灵尺寸（原子数）
BODY_W: int = art.SPRITE_W
BODY_H: int = art.SPRITE_H


class BodyRenderer:
    """渲染原版生物：腿部带关节，支持各种姿态。"""

    def __init__(self, variant: str = "male") -> None:
        """加载蓝图精灵；female 变体会加几像素粉色装饰。"""
        self.variant = variant
        self.base = art.sprite_image_native(variant)
        if variant == "female":
            pink = (238, 150, 176, 255)
            for x, y in ((13, 11), (14, 10), (15, 11), (16, 11)):
                self.base.putpixel((x, y), pink)
        self._head_body = self.base.copy()
        # 清空腿部行；腿单独画，才能动
        for y in range(24, art.SPRITE_H):
            for x in range(art.SPRITE_W):
                self._head_body.putpixel((x, y), (0, 0, 0, 0))
        self._left_legs = self.base.crop((4, 24, 12, 29))  # 左腿对 (8x5)
        self._right_legs = self.base.crop((14, 24, 20, 29))  # 右腿对 (6x5)

    # ------------------------------------------------------------------
    def render(self, pose: Pose) -> BodyFrame:
        """按姿态渲染一帧身体，返回带交互点（手/脚/头/嘴）的 BodyFrame。"""
        if pose.state == "sleep":
            # 睡觉：整个身体横躺 90 度
            lying = self.base.transpose(Image.ROTATE_90)
            anchor = (lying.width // 2, lying.height - 1)
            return BodyFrame(
                image=lying,
                anchor=anchor,
                hands=(),
                feet=(),
                head_center=(anchor[0] - 4, anchor[1] - 4),
                mouth=(anchor[0] - 4, anchor[1] - 4),
                facing=pose.facing,
            )

        image = self._head_body.copy()
        bob = 0
        left_dy = 0
        right_dy = 0
        if pose.state == "walk":
            # 行走：双腿按相位交替摆动，身体随节奏起伏
            swing = math.sin(pose.phase)
            bob = -1 if abs(swing) > 0.6 else 0
            left_dy = -1 if swing > 0 else 0
            right_dy = -1 if swing < 0 else 0
        elif pose.state == "jump":
            # 跳跃：双腿收起
            bob = -1
            left_dy = -2
            right_dy = -2
        elif pose.state in ("eat", "drink"):
            bob = 0

        if bob:
            shifted = Image.new("RGBA", image.size, (0, 0, 0, 0))
            shifted.paste(image, (0, bob), image)
            image = shifted
        image.paste(self._left_legs, (4, 24 + left_dy), self._left_legs)
        image.paste(self._right_legs, (14, 24 + right_dy), self._right_legs)

        # 解剖交互点（精灵像素坐标）
        anchor = (11, 28)
        hands = ((0, 19), (22, 16))  # 左指尖, 右指尖
        feet = ((6, 28 + left_dy), (17, 28 + right_dy))
        head_center = (11, 14 + bob)
        mouth = (11, 18 + bob)

        if pose.facing < 0:
            # 朝左：整帧水平镜像，交互点坐标也要镜像
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mirror_x = art.SPRITE_W - 1
            hands = tuple((mirror_x - x, y) for x, y in hands)
            feet = tuple((mirror_x - x, y) for x, y in feet)
            head_center = (mirror_x - head_center[0], head_center[1])
            mouth = (mirror_x - mouth[0], mouth[1])
            anchor = (mirror_x - anchor[0], anchor[1])

        return BodyFrame(
            image=image,
            anchor=anchor,
            hands=tuple(hands),
            feet=feet,
            head_center=head_center,
            mouth=mouth,
            facing=pose.facing,
        )
