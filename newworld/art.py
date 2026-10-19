"""Pixel-art sprite for the newworld inhabitant.

The sprite is reconstructed from the user's blueprint PDF
(``ref/blueprint.pdf``).  The blueprint embeds a 480x480 PNG of a pixel-art
creature standing on a green-screen background.  The artwork lives on a grid
with a pitch of 13.25 px starting at (0, 3.3125); each grid cell was reduced to
its dominant colour, the background (green) was removed and the result was
trimmed to the sprite bounding box.

The resulting native sprite is 23 x 29 art pixels:

* ``O`` = orange body  ``#E27E51``
* ``B`` = blue thought bubble  ``#6699D0``
* ``K`` = black eyes / thought dot  ``#000000``
* `` `` = transparent

``tools/extract_character.py`` can re-derive this grid from the PDF and verify
that the two agree (``python tools/extract_character.py --check``).
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

#: width / height of the sprite measured in art pixels
SPRITE_W: int = len(SPRITE_ROWS[0])
SPRITE_H: int = len(SPRITE_ROWS)

#: palette used by the blueprint.  Alpha 0 means "transparent".
PALETTE: dict[str, tuple[int, int, int, int]] = {
    " ": (0, 0, 0, 0),
    "O": (226, 126, 81, 255),   # body orange
    "B": (102, 153, 208, 255),  # thought bubble blue
    "K": (0, 0, 0, 255),        # eyes / thought dot
}

#: human readable colour names (used by the debug tooling)
PALETTE_NAMES: dict[str, str] = {
    " ": "transparent",
    "O": "orange",
    "B": "blue",
    "K": "black",
}


def validate() -> None:
    """Raise ``ValueError`` when the embedded sprite data is malformed."""
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
    """Return the sprite as an RGBA :class:`PIL.Image.Image`.

    ``cell`` scales every art pixel to ``cell`` screen pixels (nearest
    neighbour, so the pixel-art look is preserved).  ``facing`` = -1 mirrors the
    sprite horizontally.
    """
    if cell < 1:
        raise ValueError(f"cell must be >= 1, got {cell}")
    palette = dict(PALETTE)
    if variant == "female":
        palette["O"] = (214, 116, 142, 255)  # rose body
        palette["B"] = (196, 154, 212, 255)  # lilac thought bubble
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
    """Return the sprite at its native 23x29 art-pixel resolution."""
    return sprite_image(1, 1, variant=variant)
