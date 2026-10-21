"""newworld —— 俯视地图 + 群伙（swarm）研究平台。

当前世界是一张俯视地图（地形 + 食物网格）上的个体群伙：群伙按世界
规则觅食、繁衍、死亡，社会机制层（交换/教学/继承）叠加在群伙之上。
世界是抽象模拟，没有"上帝之手"改写地图的入口。

包的拆分（只暴露当前平台的三个基础模块）：

* :mod:`newworld.map_engine` —— 地图与食物网格（世界规则唯一权威）
* :mod:`newworld.swarm` —— 群伙个体与人口动态
* :mod:`newworld.social_mech` —— P3 社会机制层（交换/教学/继承）
* :mod:`newworld.map_render` / :mod:`newworld.viewport` —— 渲染与视口
* :mod:`newworld.lab_server` —— 浏览器实验平台（P2）
* :mod:`newworld.app_map` / :mod:`newworld.cli` —— 桌面 GUI 与命令行

典型用法::

    python -m newworld                          # 桌面 GUI
    python -m newworld.lab_server --port 8600   # 浏览器实验平台
"""

from __future__ import annotations

from .map_engine import MapConfig, MapEngine
from .social_mech import SocialConfig, SocialLayer
from .swarm import Swarm, SwarmConfig

#: 版本号（当前平台 1.0）
__version__ = "1.0.0"

__all__ = [
    "__version__",
    "MapConfig",
    "MapEngine",
    "Swarm",
    "SwarmConfig",
    "SocialConfig",
    "SocialLayer",
]
