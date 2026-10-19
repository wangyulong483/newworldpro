"""The inhabitant's pixel body: the **original blueprint creature**, articulated.

The blueprint sprite (23 x 29 art pixels) is kept exactly: orange body, black
eyes, blue thought bubble and four little legs.  To keep the original look while
still having anatomy and interaction, the sprite is decomposed into parts:

* thought bubble + head/body (rows 0..23, with the eyes and the little arms),
* a left leg pair and a right leg pair (rows 24..28), animated for walking and
  tucked in for jumping.

The renderer returns the atom positions of the eyes, mouth, hand tips and feet;
the behaviour layer uses the hands and feet for touching/eating/drinking, and
the head/mouth for seeing and speaking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

from . import art
from .materials import ATOM_SIZE


@dataclass
class Pose:
    """Animation state of the body."""

    state: str = "idle"  # idle, walk, jump, eat, drink, sleep
    phase: float = 0.0
    facing: int = 1


@dataclass
class BodyFrame:
    """A rendered body plus anatomical interaction points (sprite pixels)."""

    image: Image.Image
    anchor: tuple[int, int]  # feet centre in image coordinates
    hands: tuple[tuple[int, int], ...]
    feet: tuple[tuple[int, int], ...]
    head_center: tuple[int, int]
    mouth: tuple[int, int]
    facing: int


#: native sprite dimensions (atoms)
BODY_W: int = art.SPRITE_W
BODY_H: int = art.SPRITE_H


class BodyRenderer:
    """Render the original creature, with articulated legs and poses."""

    def __init__(self, variant: str = "male") -> None:
        self.variant = variant
        self.base = art.sprite_image_native(variant)
        if variant == "female":
            pink = (238, 150, 176, 255)
            for x, y in ((13, 11), (14, 10), (15, 11), (16, 11)):
                self.base.putpixel((x, y), pink)
        self._head_body = self.base.copy()
        # clear the leg rows; legs are drawn separately so they can move
        for y in range(24, art.SPRITE_H):
            for x in range(art.SPRITE_W):
                self._head_body.putpixel((x, y), (0, 0, 0, 0))
        self._left_legs = self.base.crop((4, 24, 12, 29))  # left pair (8x5)
        self._right_legs = self.base.crop((14, 24, 20, 29))  # right pair (6x5)

    # ------------------------------------------------------------------
    def render(self, pose: Pose) -> BodyFrame:
        if pose.state == "sleep":
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
            swing = math.sin(pose.phase)
            bob = -1 if abs(swing) > 0.6 else 0
            left_dy = -1 if swing > 0 else 0
            right_dy = -1 if swing < 0 else 0
        elif pose.state == "jump":
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

        anchor = (11, 28)
        hands = ((0, 19), (22, 16))  # left arm tip, right arm tip
        feet = ((6, 28 + left_dy), (17, 28 + right_dy))
        head_center = (11, 14 + bob)
        mouth = (11, 18 + bob)

        if pose.facing < 0:
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
            feet=tuple(feet),
            head_center=head_center,
            mouth=mouth,
            facing=pose.facing,
        )
