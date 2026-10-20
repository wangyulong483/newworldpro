"""newworld 仿真的配置（世界几何 + 文明机制开关）。

世界是一个以**屏幕像素**为单位计量的矩形。默认尺寸取主显示器的
物理分辨率（本项目开发的机器是 2560×1600），让世界铺满整屏。

**世界是纯白的。** 没有画上去的天空、太阳、云或地板：对居民来说
"头顶上方是天空"是一个**感知**事实，不是渲染出来的特性。居民沿屏幕
底部附近的一条固定基线行走，只能左右移动；活动范围被尽量做大
（精灵故意画得很小，因此几乎整个屏幕宽度都可以走）。

坐标系：x 向右增长，y **向下**增长，``y = 0`` 是屏幕顶端，
``y = height`` 是屏幕底端。
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
# DPI / 屏幕辅助
# ---------------------------------------------------------------------------
def enable_dpi_awareness() -> None:
    """让进程感知 DPI（仅 Windows，其他平台空操作）。

    不开启的话，Windows 会在高 DPI 显示器上悄悄缩放窗口位图，
    像素艺术的精灵就会变模糊。开启"每显示器 DPI 感知"后，世界按
    *物理*像素布局，精灵 1:1 绘制。
    """
    if sys.platform != "win32":
        return
    try:  # Windows 8.1+
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # 每显示器感知
        return
    except Exception:  # pragma: no cover - 依 Windows 版本而定
        pass
    try:  # Windows Vista / 7 的后备方案
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # pragma: no cover
        pass


def detect_screen_size() -> tuple[int, int]:
    """返回主显示器的物理像素尺寸。"""
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
    try:  # 跨平台后备：问 tkinter
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
    """解析 ``"2560x1600"`` 风格的字符串（也接受 ``x``、``,``、``*`` 作分隔）。"""
    lowered = text.lower().replace("x", " ").replace(",", " ").replace("*", " ")
    parts = [part for part in lowered.split() if part]
    if len(parts) != 2:
        raise ValueError(f"cannot parse size {text!r}, expected e.g. 2560x1600")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"size must be positive, got {text!r}")
    return width, height


# ---------------------------------------------------------------------------
# 世界配置
# ---------------------------------------------------------------------------
@dataclass
class WorldConfig:
    """定义 newworld 几何、外观与物理的全部参数。

    所有 ``*_ratio`` 值都是相对世界高度（或宽度，见注释）的比例，
    因此任意分辨率下世界的观感比例保持一致。
    """

    # -- 几何（像素） ----------------------------------------------------
    width: int = 2560
    height: int = 1600

    # -- 比例 -------------------------------------------------------------
    #: 居民脚底到屏幕底边的距离（相对高度）
    ground_margin_ratio: float = 0.02
    #: 精灵高度相对屏幕高度的比例。故意小：居民越小，可走范围越大。
    character_size_ratio: float = 0.10
    #: 最大行走速度，单位"屏高/秒"
    speed_ratio: float = 0.30
    accel_ratio: float = 3.0     # 加速度 = 速度的多少倍 / s
    friction_ratio: float = 3.5  # 减速度 = 速度的多少倍 / s

    # -- 仿真 --------------------------------------------------------------
    fixed_dt: float = 1.0 / 60.0  # 默认物理步长（秒）

    # -- 无限世界（V2） ----------------------------------------------------
    #: True：横向无界，屏幕只是局部窗口（InfiniteScene + 自动跟随平移）；
    #: False：保持原有有限屏幕世界（418+ 个既有测试的默认行为）
    infinite: bool = False
    #: 窗口原子宽度；0 表示自动（= 2 × width // ATOM_SIZE）
    window_atoms: int = 0

    # -- 外观 --------------------------------------------------------------
    #: 整个世界的背景色。除此之外不画任何东西。
    background: RGB = (255, 255, 255)

    # ------------------------------------------------------------------
    # 派生值，由 __post_init__ 填写
    # ------------------------------------------------------------------
    ground_y: int = 0
    cell: int = 0
    sprite_w: int = 0
    sprite_h: int = 0
    speed: float = 0.0
    accel: float = 0.0
    friction: float = 0.0

    def __post_init__(self) -> None:
        """校验参数并计算所有派生值（地面线、精灵尺寸、速度等）。"""
        if self.width < 320 or self.height < 240:
            raise ValueError(
                f"world is too small: {self.width}x{self.height} "
                "(minimum 320x240)"
            )
        if not 0.0 < self.ground_margin_ratio < 0.4:
            raise ValueError("ground_margin_ratio must be between 0 and 0.4")
        if not 0.02 <= self.character_size_ratio <= 0.6:
            raise ValueError("character_size_ratio must be between 0.02 and 0.6")

        # 地面基线的 y 坐标
        self.ground_y = round(self.height * (1.0 - self.ground_margin_ratio))
        self.ground_y = min(self.ground_y, self.height - 1)

        # 小居民 → 几乎整屏宽度可走
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
        if self.window_atoms < 0:
            raise ValueError("window_atoms must be >= 0 (0 = auto)")
        if self.infinite and self.window_atoms != 0 and self.window_atoms < 16:
            raise ValueError("window_atoms must be >= 16 when infinite")

    # -- 便捷属性 -----------------------------------------------------------
    @property
    def half_sprite(self) -> float:
        """精灵宽度的一半（用于把中心 x 夹在屏幕内）。"""
        return self.sprite_w / 2.0

    @property
    def min_x(self) -> float:
        """居民中心 x 的最小合法值（精灵不出屏）。"""
        return self.half_sprite

    @property
    def max_x(self) -> float:
        """居民中心 x 的最大合法值。"""
        return self.width - self.half_sprite

    @property
    def travel_span(self) -> float:
        """居民可以走过的水平距离。"""
        return max(0.0, self.max_x - self.min_x)

    @property
    def travel_ratio(self) -> float:
        """屏幕宽度中可走部分的比例（0..1）。"""
        if self.width <= 0:
            return 0.0
        return self.travel_span / self.width

    def describe(self) -> str:
        """多行的人类可读摘要（``python -m newworld --info`` 用）。"""
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
        """序列化为普通字典（telemetry / 存档用）。"""
        keys = (
            "width", "height", "ground_y", "cell", "sprite_w", "sprite_h",
            "speed", "accel", "friction", "fixed_dt",
        )
        return {key: getattr(self, key) for key in keys}


# ---------------------------------------------------------------------------
# 构造器
# ---------------------------------------------------------------------------
def make_config(
    width: int | None = None,
    height: int | None = None,
    *,
    cell: int | None = None,
    **overrides: Any,
) -> WorldConfig:
    """构造 :class:`WorldConfig`，默认取当前屏幕尺寸（可用 NEWWORLD_SIZE 覆盖）。"""
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
        # 显式指定精灵缩放倍数：必须放得下世界
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
# 文明 / 演化配置（P1–P4）
# ---------------------------------------------------------------------------
@dataclass
class CivilizationConfig:
    """文明层的旋钮，包括消融实验的机制开关。

    任务的**硬约束**是 ``max_agents_created``：世界在其整个生命周期内
    *创建*的个体总数绝不能超过该值，无论其中多少已死亡。其余都是
    消融实验 A/B/C/D 用的机制开关（见 :data:`ABLATIONS`）。

    时间尺度为可观测性而压缩（``lifespan_s`` = 900 世界秒 ≈ 1.5 个
    虚拟日，不是人口学标定）。
    """

    # -- 硬约束 ------------------------------------------------------------
    max_agents_created: int = 8
    lifespan_s: float = 900.0
    maturity_s: float = 70.0
    pregnancy_s: float = 24.0

    # -- 机制开关（消融用） -----------------------------------------------
    enable_planning: bool = True        # P1：目标管理器 + 规划器
    enable_knowledge: bool = True       # P2：知识库 / 技术图
    enable_inheritance: bool = True     # P3：记忆、归档、教学传承
    enable_social: bool = True          # P4：合作、设施、规则
    enable_disturbance: bool = False    # P4：干旱 / 风暴 / 资源枯竭
    expanded_observation: bool = True   # P3：神经观测 24 -> 40 维
    enable_perception: bool = True      # 实验可关闭（感知占约 95% 开销）

    # -- 学习型决策层（本轮升级） -----------------------------------------
    #: "heuristic" = 规则优先级（旧行为）；"neural" = 训练出的高层目标
    #: 策略采样；"random" = 均匀随机目标（消融对照组）
    goal_policy_mode: str = "heuristic"
    #: 底层控制器："full" = 行为克隆网络，"random" = 随机游走，
    #: "off" = 原地站立（决策链消融）
    low_level_mode: str = "full"
    #: 有限组合式创新原型（见 newworld/innovation.py）
    enable_innovation: bool = False

    # -- V2：感知进入观测与低层 RL ----------------------------------------
    #: 真实视网膜特征（16 维）是否进入神经观测（FEATURE_DIM 40 -> 56）
    enable_visual_features: bool = True
    #: 每隔多少帧刷新一次完整感知/视网膜特征（感知占约 95% 开销）
    perception_stride: int = 4
    #: 低层动作控制器新增 "rl"：加载 RL checkpoint 并用随机策略采样
    low_level_mode: str = "full"

    # -- 实验预算 ---------------------------------------------------------
    seed: int = 0

    def __post_init__(self) -> None:
        """校验硬约束与枚举取值。"""
        if self.max_agents_created < 1:
            raise ValueError("max_agents_created must be >= 1")
        if not 0.0 < self.maturity_s < self.lifespan_s:
            raise ValueError("maturity_s must be < lifespan_s")
        if self.pregnancy_s <= 0.0:
            raise ValueError("pregnancy_s must be > 0")
        if self.perception_stride < 1:
            raise ValueError("perception_stride must be >= 1")
        if self.goal_policy_mode not in ("heuristic", "neural", "random"):
            raise ValueError(
                f"unknown goal_policy_mode {self.goal_policy_mode!r}; "
                "expected 'heuristic', 'neural' or 'random'"
            )
        if self.low_level_mode not in ("full", "random", "off", "rl"):
            raise ValueError(
                f"unknown low_level_mode {self.low_level_mode!r}; "
                "expected 'full', 'random', 'off' or 'rl'"
            )

    def as_dict(self) -> dict[str, Any]:
        """序列化为普通字典（telemetry / 存档用）。"""
        keys = (
            "max_agents_created", "lifespan_s", "maturity_s", "pregnancy_s",
            "enable_planning", "enable_knowledge", "enable_inheritance",
            "enable_social", "enable_disturbance", "expanded_observation",
            "enable_perception", "goal_policy_mode", "low_level_mode",
            "enable_innovation", "enable_visual_features", "perception_stride",
            "seed",
        )
        return {key: getattr(self, key) for key in keys}


#: 任务书（§7）要求的消融分组：
#: A = 原始机制；B = +规划；C = +持久知识/传承；D = 全系统
#: （+社会机制 +环境扰动）。
#:
#: 注意：**8 个个体创建上限**不是消融开关——它是每一组（包括 A）都
#: 必须满足的约束。
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
    """构造 :class:`CivilizationConfig`，可从消融组（A/B/C/D）一键生成。"""
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
