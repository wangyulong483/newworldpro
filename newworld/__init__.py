"""newworld — a screen-sized pixel-art world.

The world is a rectangle the size of the physical screen and it is painted
**pure white**: no sky, sun, clouds or floor are drawn — "above the inhabitant
is sky" is a perceptual fact for the inhabitant, not a visual element.  The
inhabitant (the creature from the user's blueprint) is deliberately small so
that almost the whole screen width is walkable; it can only move left/right and
is clamped to the screen bounds.

The package is split so the simulation can be reused for RL research:

* :mod:`newworld.world` — pure simulation (:class:`~newworld.world.World`)
* :mod:`newworld.character` — inhabitants and their physics
* :mod:`newworld.env` — Gymnasium-style environment
  (:class:`~newworld.env.NewWorldEnv`) with pluggable rewards
* :mod:`newworld.render` — headless PIL renderer used by previews/recordings
* :mod:`newworld.app_tk` — interactive fullscreen frontend
* :mod:`newworld.preview` — documentation / preview images

Typical use::

    python -m newworld                     # enter the world
    python -m newworld --headless          # run without a window
    python -m newworld --preview           # render PNG previews
"""

from .config import (
    WORLD_NAME,
    WORLD_VERSION,
    WorldConfig,
    detect_screen_size,
    make_config,
    parse_size,
)

__version__ = WORLD_VERSION

from .character import Character  # noqa: E402
from .world import World  # noqa: E402
from .env import NewWorldEnv  # noqa: E402
from . import physics  # noqa: E402
from .daynight import DayCycle  # noqa: E402
from .physiology import Physiology  # noqa: E402
from .ecology import Ecology  # noqa: E402
from .neural import NeuralPolicy  # noqa: E402
from .pixelworld import PixelScene  # noqa: E402
from .acoustics import SoundField  # noqa: E402
from .environment import Air, Sun  # noqa: E402
from .optics import Box, Disc, Scene  # noqa: E402
from .perception import PerceptionState, VisualSystem  # noqa: E402
from .autonomy import FreeWill, Mind  # noqa: E402

__all__ = [
    "WORLD_NAME",
    "WORLD_VERSION",
    "__version__",
    "WorldConfig",
    "Character",
    "World",
    "NewWorldEnv",
    "Box",
    "Disc",
    "Scene",
    "physics",
    "DayCycle",
    "Physiology",
    "PixelScene",
    "Ecology",
    "NeuralPolicy",
    "SoundField",
    "Air",
    "Sun",
    "VisualSystem",
    "PerceptionState",
    "FreeWill",
    "Mind",
    "detect_screen_size",
    "make_config",
    "parse_size",
]
