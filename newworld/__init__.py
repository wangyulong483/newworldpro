"""newworld —— 一个屏幕大小的像素世界（微型文明仿真）。

世界是一个物理屏幕大小的矩形，被涂成**纯白**：没有画天空、太阳、
云或地板——"居民头顶是天空"是居民的**感知**事实，不是视觉元素。
居民（用户蓝图上的生物）故意画得很小，因此几乎整个屏幕宽度都可
以走；他只能左右移动，并被夹在屏幕边界内。

包的拆分让仿真可以被 RL 研究复用：

* :mod:`newworld.world` —— 纯仿真（:class:`~newworld.world.World`）
* :mod:`newworld.character` —— 居民与其物理
* :mod:`newworld.env` —— Gymnasium 风格环境
  （:class:`~newworld.env.NewWorldEnv`），奖励可插拔
* :mod:`newworld.render` —— 预览/录像用的 headless PIL 渲染器
* :mod:`newworld.app_tk` —— 交互式全屏前端
* :mod:`newworld.preview` —— 文档 / 预览图

典型用法::

    python -m newworld                     # 进入世界
    python -m newworld --headless          # 无窗口运行
    python -m newworld --preview           # 渲染 PNG 预览
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
