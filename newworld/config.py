"""Configuration for the newworld simulation.

A world is a rectangle measured in *screen pixels*.  Its size defaults to the
physical resolution of the primary display (``2560x1600`` on the machine this
world was built for), so the world fills the screen edge to edge.

**The world is pure white.**  There is no painted sky, no sun, no clouds and no
visible floor: "above the inhabitant is sky" is a *perceptual* fact for the
inhabitant, not a rendered feature.  The inhabitant walks along a fixed
baseline near the bottom of the screen and can only move left/right; its
activity range is kept as large as possible (the sprite is deliberately small,
so almost the whole screen width is walkable).

Coordinate system: x grows to the right, y grows **downwards**, so
``y = 0`` is the top of the screen and ``y = height`` is the bottom.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from typing import Any

from . import art

WORLD_NAME = "newworld"
WORLD_VERSION = "0.9.0"

RGB = tuple[int, int, int]


# ---------------------------------------------------------------------------
# DPI / screen helpers
# ---------------------------------------------------------------------------
def enable_dpi_awareness() -> None:
    """Make this process DPI aware (Windows only, no-op elsewhere).

    Without this, Windows silently scales the window bitmap on high-DPI
    displays and the pixel-art sprite becomes blurry.  With per-monitor DPI
    awareness the world is laid out in *physical* pixels and the sprite is
    drawn 1:1.
    """
    if sys.platform != "win32":
        return
    try:  # Windows 8.1+
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
        return
    except Exception:  # pragma: no cover - depends on Windows version
        pass
    try:  # Windows Vista / 7 fallback
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # pragma: no cover
        pass


def detect_screen_size() -> tuple[int, int]:
    """Return the primary display size in physical pixels."""
    enable_dpi_awareness()
    if sys.platform == "win32":
        try:
            user32 = ctypes.windll.user32
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
            if width > 0 and height > 0:
                return width, height
        except Exception:  # pragma: no cover
            pass
    try:  # cross platform fallback
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        size = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
        root.destroy()
        if size[0] > 0 and size[1] > 0:
            return size
    except Exception:  # pragma: no cover
        pass
    return 1920, 1080


def parse_size(text: str) -> tuple[int, int]:
    """Parse ``"2560x1600"`` style strings (also accepts ``x`` or ``,``)."""
    lowered = text.lower().replace("x", " ").replace(",", " ").replace("*", " ")
    parts = [part for part in lowered.split() if part]
    if len(parts) != 2:
        raise ValueError(f"cannot parse size {text!r}, expected e.g. 2560x1600")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"size must be positive, got {text!r}")
    return width, height


# ---------------------------------------------------------------------------
# World configuration
# ---------------------------------------------------------------------------
@dataclass
class WorldConfig:
    """Everything that defines the newworld geometry, look and physics.

    All ``*_ratio`` values are relative to the world height (or width, where
    noted) so the world looks proportionally the same at any display
    resolution.
    """

    # -- geometry (pixels) -------------------------------------------------
    width: int = 2560
    height: int = 1600

    # -- proportions -------------------------------------------------------
    #: distance between the inhabitant's feet and the bottom screen edge
    ground_margin_ratio: float = 0.02
    #: sprite height relative to the screen height.  Small on purpose: the
    #: smaller the inhabitant, the larger its walkable range.
    character_size_ratio: float = 0.10
    #: top walking speed, screen heights per second
    speed_ratio: float = 0.30
    accel_ratio: float = 3.0     # acceleration, multiples of speed / s
    friction_ratio: float = 3.5  # deceleration, multiples of speed / s

    # -- simulation --------------------------------------------------------
    fixed_dt: float = 1.0 / 60.0  # default physics step (seconds)

    # -- look --------------------------------------------------------------
    #: the whole world is this colour.  Nothing else is painted.
    background: RGB = (255, 255, 255)

    # ------------------------------------------------------------------
    # derived values, filled in by __post_init__
    # ------------------------------------------------------------------
    ground_y: int = 0
    cell: int = 0
    sprite_w: int = 0
    sprite_h: int = 0
    speed: float = 0.0
    accel: float = 0.0
    friction: float = 0.0

    def __post_init__(self) -> None:
        if self.width < 320 or self.height < 240:
            raise ValueError(
                f"world is too small: {self.width}x{self.height} "
                "(minimum 320x240)"
            )
        if not 0.0 < self.ground_margin_ratio < 0.4:
            raise ValueError("ground_margin_ratio must be between 0 and 0.4")
        if not 0.02 <= self.character_size_ratio <= 0.6:
            raise ValueError("character_size_ratio must be between 0.02 and 0.6")

        self.ground_y = round(self.height * (1.0 - self.ground_margin_ratio))
        self.ground_y = min(self.ground_y, self.height - 1)

        # small inhabitant -> almost the entire screen width is walkable
        cell = round(self.height * self.character_size_ratio / art.SPRITE_H)
        max_cell = min(
            self.width // art.SPRITE_W,
            max(1, int(self.height * 0.6) // art.SPRITE_H),
        )
        self.cell = max(4, min(cell, max(1, max_cell)))
        self.sprite_w = art.SPRITE_W * self.cell
        self.sprite_h = art.SPRITE_H * self.cell
        if self.sprite_h > self.ground_y:
            raise ValueError(
                "world is too short for the inhabitant "
                f"({self.sprite_h}px sprite vs {self.ground_y}px baseline)"
            )

        self.speed = self.height * self.speed_ratio
        self.accel = self.speed * self.accel_ratio
        self.friction = self.speed * self.friction_ratio

    # -- convenience -------------------------------------------------------
    @property
    def half_sprite(self) -> float:
        return self.sprite_w / 2.0

    @property
    def min_x(self) -> float:
        """Left-most legal centre x of an inhabitant (sprite stays on screen)."""
        return self.half_sprite

    @property
    def max_x(self) -> float:
        """Right-most legal centre x of an inhabitant."""
        return self.width - self.half_sprite

    @property
    def travel_span(self) -> float:
        """Horizontal distance the inhabitant may travel."""
        return max(0.0, self.max_x - self.min_x)

    @property
    def travel_ratio(self) -> float:
        """Walkable fraction of the screen width (0 .. 1)."""
        if self.width <= 0:
            return 0.0
        return self.travel_span / self.width

    def describe(self) -> str:
        """Multi-line human readable summary (used by ``python -m newworld --info``)."""
        from .materials import ATOM_SIZE

        atoms_w = self.width // ATOM_SIZE
        atoms_h = self.height // ATOM_SIZE
        return "\n".join(
            [
                f"world         : {WORLD_NAME} v{WORLD_VERSION}",
                f"screen size   : {self.width} x {self.height} px (physical)",
                f"pixel atoms   : {atoms_w} x {atoms_h} atoms "
                f"(1 atom = {ATOM_SIZE}x{ATOM_SIZE} screen px)",
                "scene         : pixel-atom terrain (stone/soil/grass), pond, rocks, "
                "trees, bushes, flowers; day-night sky",
                f"ground line   : y = {self.ground_y} (above it is sky)",
                "inhabitant    : pixel body (head/torso/arms/legs) ~90 x 175 px",
                f"walk range    : x = {self.min_x:.0f} .. {self.max_x:.0f} "
                f"(travel span {self.travel_span:.0f} px, "
                f"{self.travel_ratio * 100:.1f}% of screen width)",
                f"motion        : speed {self.speed:.0f} px/s, "
                f"accel {self.accel:.0f} px/s^2, friction {self.friction:.0f} px/s^2",
            ]
        )

    def as_dict(self) -> dict[str, Any]:
        keys = (
            "width", "height", "ground_y", "cell", "sprite_w", "sprite_h",
            "speed", "accel", "friction", "fixed_dt",
        )
        return {key: getattr(self, key) for key in keys}


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------
def make_config(
    width: int | None = None,
    height: int | None = None,
    *,
    cell: int | None = None,
    **overrides: Any,
) -> WorldConfig:
    """Build a :class:`WorldConfig`, defaulting to the current screen size."""
    if width is None or height is None:
        env_size = os.environ.get("NEWWORLD_SIZE")
        if env_size:
            detected_w, detected_h = parse_size(env_size)
        else:
            detected_w, detected_h = detect_screen_size()
        width = detected_w if width is None else width
        height = detected_h if height is None else height

    config = WorldConfig(width=int(width), height=int(height), **overrides)
    if cell is not None:
        if cell < 1:
            raise ValueError(f"cell must be >= 1, got {cell}")
        cell = int(cell)
        if cell * art.SPRITE_W > config.width:
            raise ValueError(
                f"cell={cell} makes the sprite wider than the world "
                f"({cell * art.SPRITE_W} > {config.width})"
            )
        if cell * art.SPRITE_H > config.ground_y:
            raise ValueError(
                f"cell={cell} makes the sprite taller than the walkable area "
                f"({cell * art.SPRITE_H} > {config.ground_y})"
            )
        config.cell = cell
        config.sprite_w = art.SPRITE_W * config.cell
        config.sprite_h = art.SPRITE_H * config.cell
    return config


# ---------------------------------------------------------------------------
# Civilization / evolution configuration (P1–P4)
# ---------------------------------------------------------------------------
@dataclass
class CivilizationConfig:
    """Knobs for the civilization layer, including the ablation switches.

    The **hard constraint** of the task is ``max_agents_created``: the world may
    never *create* more than this many agents over its whole life, no matter how
    many of them have died.  Everything else here is a mechanism switch used by
    the ablation experiments A/B/C/D (see :data:`ABLATIONS`).

    Time scales are compressed for observability (a ``lifespan_s`` of 900 world
    seconds is ~1.5 virtual days, not a demographic calibration).
    """

    # -- hard constraint ---------------------------------------------------
    max_agents_created: int = 8
    lifespan_s: float = 900.0
    maturity_s: float = 70.0
    pregnancy_s: float = 24.0

    # -- mechanism switches (ablation) ------------------------------------
    enable_planning: bool = True        # P1: goal manager + planner
    enable_knowledge: bool = True       # P2: knowledge store / tech graph
    enable_inheritance: bool = True     # P3: memory, archiving, teaching
    enable_social: bool = True          # P4: cooperation, facilities, rules
    enable_disturbance: bool = False    # P4: droughts / storms / scarcity
    expanded_observation: bool = True   # P3: 24 -> 40 neural inputs
    enable_perception: bool = True      # experiments may disable (95% of cost)

    # -- experiment budget ------------------------------------------------
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_agents_created < 1:
            raise ValueError("max_agents_created must be >= 1")
        if not 0.0 < self.maturity_s < self.lifespan_s:
            raise ValueError("maturity_s must be < lifespan_s")
        if self.pregnancy_s <= 0.0:
            raise ValueError("pregnancy_s must be > 0")

    def as_dict(self) -> dict[str, Any]:
        keys = (
            "max_agents_created", "lifespan_s", "maturity_s", "pregnancy_s",
            "enable_planning", "enable_knowledge", "enable_inheritance",
            "enable_social", "enable_disturbance", "expanded_observation",
            "enable_perception", "seed",
        )
        return {key: getattr(self, key) for key in keys}


#: Ablation groups required by the task book (§7):
#: A = original mechanisms, B = +planning, C = +persistent knowledge/inheritance,
#: D = full system (+social mechanisms + environment disturbance).
#:
#: Note: the *hard* 8-agent creation cap is **not** an ablation switch — it is a
#: constraint that holds in every group, including A.
ABLATIONS: dict[str, dict[str, bool]] = {
    "A": {
        "enable_planning": False,
        "enable_knowledge": False,
        "enable_inheritance": False,
        "enable_social": False,
        "enable_disturbance": False,
        "expanded_observation": False,
    },
    "B": {
        "enable_planning": True,
        "enable_knowledge": False,
        "enable_inheritance": False,
        "enable_social": False,
        "enable_disturbance": False,
        "expanded_observation": False,
    },
    "C": {
        "enable_planning": True,
        "enable_knowledge": True,
        "enable_inheritance": True,
        "enable_social": False,
        "enable_disturbance": False,
        "expanded_observation": True,
    },
    "D": {
        "enable_planning": True,
        "enable_knowledge": True,
        "enable_inheritance": True,
        "enable_social": True,
        "enable_disturbance": True,
        "expanded_observation": True,
    },
}


def make_civilization_config(
    group: str | None = None,
    *,
    seed: int = 0,
    **overrides: Any,
) -> CivilizationConfig:
    """Build a :class:`CivilizationConfig`, optionally from an ablation group."""
    values: dict[str, Any] = {"seed": seed}
    if group is not None:
        try:
            values.update(ABLATIONS[group.upper()])
        except KeyError:
            raise ValueError(
                f"unknown ablation group {group!r}, available: {', '.join(sorted(ABLATIONS))}"
            ) from None
    values.update(overrides)
    return CivilizationConfig(**values)
