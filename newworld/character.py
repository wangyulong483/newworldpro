"""newworld 的居民（角色本体）。

居民是用户蓝图上的生物：橙色像素身体 + 蓝色思维气泡。它生活在屏幕
底部附近，能左右行走、**在重力下跳跃**（力学）、有**体温**（热学）、
会**说话**（声学）。运动被裁剪在屏幕内，永远出不了世界。

动作是整数（给 RL / 环境接口用）::

    -1  向左走
     0  原地站（摩擦让生物减速停下）
    +1  向右走

``update(..., jump=True)`` 会在生物站在地面时额外触发一次弹道跳跃。
自由生物（:mod:`newworld.autonomy`）两者都会用到。

这个类**故意**不包含渲染和 numpy：便于复制、pickle、被演化实验
修改、以及低成本的单元测试。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import physics
from .config import WorldConfig
from .materials import ATOM_SIZE
from .physiology import Physiology
from .technology import Inventory

LEFT = -1
STAY = 0
RIGHT = 1
ACTIONS = (LEFT, STAY, RIGHT)


def clamp(value: float, low: float, high: float) -> float:
    """把 value 夹在 [low, high] 内。"""
    return low if value < low else high if value > high else value


@dataclass
class Character:
    """世界中的一个居民（可独立 pickle/复制的纯数据 + 力学类）。"""

    x: float
    facing: int = 1
    vx: float = 0.0
    phase: float = 0.0
    agent_id: int = 0
    # 记账（研究日志有用）
    distance_walked: float = 0.0
    steps: int = 0
    # -- 力学 ---------------------------------------------------------
    #: 脚的世界像素位置（``attach`` 之前为 None）；y 向下增长
    y: float | None = None
    vy: float = 0.0
    on_ground: bool = True
    mass_kg: float = physics.BODY_MASS_KG
    jump_count: int = 0
    # -- 热学 -----------------------------------------------------------
    body_temperature_k: float = physics.HUMAN_BODY_TEMPERATURE_K
    heat_balance: object | None = None
    # -- 声学 -----------------------------------------------------------
    utterance: str | None = None
    utterance_until: float = 0.0
    speech_count: int = 0
    # -- 内在生命 ---------------------------------------------------
    behavior_state: str = "explore"
    physiology: Physiology = field(default_factory=Physiology)
    # -- 社会 --------------------------------------------------------
    name: str = ""
    sex: str = "male"          # "male" | "female"
    age: float = 0.0           # 寿命秒数
    generation: int = 0
    alive: bool = True
    pregnant: bool = False
    pregnancy_timer: float = 0.0
    partner_id: int | None = None
    parents: tuple[int, ...] = ()
    body_scale: float = 1.0
    inventory: Inventory = field(default_factory=Inventory)
    social_context: tuple[float, ...] = ()
    # -- 文明层（P3）：身份、历史、知识、计划 ------------------------
    #: 直接父代（有则是 ``parents[0]``）；显式保留以便族谱查询
    parent_id: int | None = None
    #: 创建与死亡的世界时刻（活着时 died_at 为 None）
    created_at: float = 0.0
    died_at: float | None = None
    death_cause: str = ""
    #: 该个体已学会的知识 id 列表（KnowledgeStore 的 id）
    known_knowledge_ids: list[str] = field(default_factory=list)
    #: 技能 -> 熟练度 (0..1)，程序性习得
    skills: dict[str, float] = field(default_factory=dict)
    #: 该个体保留的情景记忆 id 列表（MemoryStore）
    memory_ids: list[int] = field(default_factory=list)
    #: 当前目标 / 计划的句柄（GoalManager 的 id）
    current_goal_id: int | None = None
    current_plan_id: int | None = None
    #: 对公共设施 / 合作的贡献（类别 -> 数量）
    contributions: dict[str, float] = field(default_factory=dict)
    #: 16 维真实视网膜特征缓存（V2.1；由 Population 按 perception_stride 更新）
    visual_features: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """归一化数值字段并初始化几何缓存（attach 后才有意义）。"""
        self.x = float(self.x)
        self.facing = 1 if self.facing >= 0 else -1
        self.vx = float(self.vx)
        self.phase = float(self.phase)
        # 几何信息：由世界在创建 agent 时挂接
        self._sprite_w: int = 0
        self._sprite_h: int = 0
        self._ground: float = 0.0
        # 屏幕边界（左/右），由 :meth:`attach` 刷新
        self._min_x: float = -1.0
        self._max_x: float = -1.0

    # ------------------------------------------------------------------
    # 挂接
    # ------------------------------------------------------------------
    def attach(self, config: WorldConfig) -> None:
        """从配置挂接精灵几何与屏幕边界（y 未初始化时落到地面线）。"""
        self._sprite_w = config.sprite_w
        self._sprite_h = config.sprite_h
        self._ground = float(config.ground_y)
        self._min_x = config.min_x
        self._max_x = config.max_x
        if self.y is None:
            self.y = self._ground

    # ------------------------------------------------------------------
    # 仿真
    # ------------------------------------------------------------------
    def update(
        self,
        dt: float,
        action: int,
        config: WorldConfig,
        *,
        jump: bool = False,
    ) -> None:
        """按 ``dt`` 秒推进居民（带重力的力学积分）。

        水平：加速度 + 摩擦 + 平方空气阻力，速度夹在 ±speed；
        垂直：重力 + 可选的弹道跳跃，落地即停。
        """
        if dt <= 0.0:
            return
        if self.y is None:
            self.attach(config)
        action = int(action)
        action = RIGHT if action > 0 else LEFT if action < 0 else STAY

        # -- 水平运动 ---------------------------------------------------
        if action != STAY:
            self.vx += action * config.accel * dt
            self.vx = clamp(self.vx, -config.speed, config.speed)
            self.facing = action
            # 行走相位（决定走路弹跳动画）
            self.phase += dt * math.tau * 1.75 * (abs(self.vx) / config.speed)
        elif self.vx != 0.0:
            # 摩擦减速
            drop = config.friction * dt
            if abs(self.vx) <= drop:
                self.vx = 0.0
            else:
                self.vx -= math.copysign(drop, self.vx)

        # 空气阻力（平方阻力）：a = -0.5 rho Cd A v|v| / m
        drag = physics.air_drag_force(physics.px_to_m(self.vx)) / max(1e-6, self.mass_kg)
        drag_px = drag * physics.PIXELS_PER_METER
        if abs(self.vx) > 1e-9:
            self.vx -= math.copysign(min(abs(self.vx), abs(drag_px) * dt), self.vx)
        else:
            self.vx = 0.0

        previous_x = self.x
        self.x += self.vx * dt
        # 夹在屏幕内
        if self.x < config.min_x:
            self.x = config.min_x
            self.vx = 0.0
        elif self.x > config.max_x:
            self.x = config.max_x
            self.vx = 0.0

        # -- 垂直力学：重力 + 弹道跳跃 ---------------------------------
        if jump and self.on_ground:
            self.vy = -physics.JUMP_SPEED_PX
            self.on_ground = False
            self.jump_count += 1
        self.vy += physics.GRAVITY_PX * dt
        self.y += self.vy * dt
        if self.y >= self._ground:
            # 落回地面
            self.y = self._ground
            self.vy = 0.0
            self.on_ground = True
        elif self.y < self._sprite_h:
            # 不允许跳出屏幕顶端
            self.y = self._sprite_h
            self.vy = max(self.vy, 0.0)

        self.distance_walked += abs(self.x - previous_x)
        self.steps += 1

    def jump(self, config: WorldConfig) -> bool:
        """站在地面时起跳；返回是否真的跳了。"""
        if not self.on_ground:
            return False
        self.vy = -physics.JUMP_SPEED_PX
        self.on_ground = False
        self.jump_count += 1
        return True

    # ------------------------------------------------------------------
    # 文明钩子
    # ------------------------------------------------------------------
    def can_innovate(self) -> bool:
        """进行组合技术创新的最低条件。

        有限组合式创新原型（:mod:`newworld.innovation`）会问个体这个
        问题。世界规则是：手上有工具，**或者**至少有两单位原料可摆弄。
        这*不是*跳过材料或知识的许可：那些由创新规则与技术模拟器
        另行检查，原型的成本也会真实扣除。
        """
        return self.inventory.tool is not None or self.inventory.load() >= 2

    # ------------------------------------------------------------------
    # 几何 / 派生状态
    # ------------------------------------------------------------------
    @property
    def width(self) -> int:
        """精灵宽度（像素）。"""
        return self._sprite_w

    @property
    def height(self) -> int:
        """精灵高度（像素）。"""
        return self._sprite_h

    @property
    def at_left_wall(self) -> bool:
        """是否顶到左边界。"""
        return self.x <= self._min_x + 0.5

    @property
    def at_right_wall(self) -> bool:
        """是否顶到右边界。"""
        return self.x >= self._max_x - 0.5

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """精灵的世界像素包围盒 (left, top, right, bottom)。"""
        feet = self._ground if self.y is None else self.y
        left = self.x - self._sprite_w / 2.0
        top = feet - self._sprite_h
        return left, top, left + self._sprite_w, top + self._sprite_h

    @property
    def body_temperature_c(self) -> float:
        """体温（摄氏度）。"""
        return self.body_temperature_k - 273.15

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def observation(self, config: WorldConfig) -> tuple[float, ...]:
        """RL / 日志用的归一化特征向量（8 维）。

        ``(x_norm, vx_norm, facing, dist_left_norm, dist_right_norm,
        height_norm, vy_norm, on_ground)``

        * ``x_norm`` / ``dist_left_norm`` / ``dist_right_norm``：位置与
          到左右边界的距离（按可走跨度归一化）
        * ``height_norm``：跳离地面的高度，按满跳的弹道峰值归一化
        * ``vy_norm``：垂直速度，按跳跃初速度归一化
        * ``on_ground``：站在地面为 1.0，否则 0.0
        """
        span = config.travel_span
        if span <= 0:
            x_norm = dist_left = dist_right = 0.0
        else:
            x_norm = (self.x - config.min_x) / span
            dist_left = (self.x - config.min_x) / span
            dist_right = (config.max_x - self.x) / span
        feet = self._ground if self.y is None else self.y
        peak_px = max(1.0, physics.jump_height_m() * physics.PIXELS_PER_METER)
        height_norm = clamp((self._ground - feet) / peak_px, 0.0, 1.0)
        return (
            clamp(x_norm, 0.0, 1.0),
            clamp(self.vx / config.speed, -1.0, 1.0),
            float(self.facing),
            dist_left,
            dist_right,
            height_norm,
            clamp(self.vy / physics.JUMP_SPEED_PX, -1.0, 1.0),
            1.0 if self.on_ground else 0.0,
        )

    def bob_offset(self, config: WorldConfig) -> int:
        """行走时的垂直弹跳偏移（屏幕像素，负 = 向上）。"""
        if abs(self.vx) < 1.0 or not self.on_ground:
            return 0
        amount = min(1.0, abs(self.vx) / config.speed)
        amplitude = config.cell * 0.4
        return -int(round(abs(math.sin(self.phase)) * amplitude * amount))

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        """序列化为可 JSON 化的字典（telemetry / 存档用）。"""
        return {
            "agent_id": self.agent_id,
            "x": round(self.x, 3),
            "y": round(self.y if self.y is not None else 0.0, 3),
            "vx": round(self.vx, 3),
            "vy": round(self.vy, 3),
            "facing": self.facing,
            "on_ground": self.on_ground,
            "body_temperature_c": round(self.body_temperature_c, 3),
            "utterance": self.utterance,
            "behavior_state": self.behavior_state,
            "name": self.name,
            "sex": self.sex,
            "age": round(self.age, 2),
            "alive": self.alive,
            "pregnant": self.pregnant,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "created_at": round(self.created_at, 3),
            "died_at": None if self.died_at is None else round(self.died_at, 3),
            "death_cause": self.death_cause,
            "current_goal_id": self.current_goal_id,
            "current_plan_id": self.current_plan_id,
            "known_knowledge": list(self.known_knowledge_ids),
            "skills": {key: round(value, 3) for key, value in self.skills.items()},
            "wood": self.inventory.wood,
            "stone": self.inventory.stone,
            "fiber": getattr(self.inventory, "fiber", 0),
            "cord": getattr(self.inventory, "cord", 0),
            "tool": self.inventory.tool,
            "energy": round(self.physiology.energy, 2),
            "hydration": round(self.physiology.hydration, 2),
            "sleep_pressure": round(self.physiology.sleep_pressure, 2),
            "heart_rate": round(self.physiology.heart_rate, 2),
            "sleeping": self.physiology.sleeping,
            "distance_walked": round(self.distance_walked, 3),
            "steps": self.steps,
        }


# ---------------------------------------------------------------------------
# 身体几何辅助
# ---------------------------------------------------------------------------
def eye_position(character: Character, config: WorldConfig) -> tuple[float, float]:
    """眼睛的世界坐标（像素身体的头部中心）。"""
    feet = config.ground_y if character.y is None else character.y
    return character.x, feet - 14.0 * ATOM_SIZE


def mouth_position(character: Character, config: WorldConfig) -> tuple[float, float]:
    """嘴的世界坐标（头部朝向前方的一侧）。"""
    feet = config.ground_y if character.y is None else character.y
    return (
        character.x + character.facing * 2.0 * ATOM_SIZE,
        feet - 10.0 * ATOM_SIZE,
    )
