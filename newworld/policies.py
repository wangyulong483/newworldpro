"""测试 / RL 实验用的小型脚本策略。

**交互应用不使用这些策略。** 居民是自由的，在
:mod:`newworld.autonomy`（``FreeWill``）里自己选择行动；这些确定性
策略只服务于单元测试、环境演练，以及未来 RL 训练脚本的简单基线。

策略都是可调用对象 ``policy(world) -> action``。
"""

from __future__ import annotations

import random

from .character import LEFT, RIGHT, STAY, Character
from .world import World


class RandomPolicy:
    """均匀随机动作。"""

    def __init__(self, seed: int | None = None) -> None:
        """初始化策略。"""
        self.rng = random.Random(seed)

    def reset(self, world: World | None = None) -> None:
        """重置策略内部状态。"""
        return None

    def __call__(self, world: World) -> int:
        """取一步动作。"""
        return self.rng.choice((LEFT, STAY, RIGHT))


class WanderPolicy:
    """遵守世界边界的平滑随机游走。

    居民朝一个方向走一段随机时间、偶尔停顿、撞到边界就折返。
    这样能覆盖完整的左右运动范围，正是演示录像里展示的效果。
    """

    def __init__(
        self,
        seed: int | None = None,
        min_hold: float = 0.35,
        max_hold: float = 1.4,
        stand_probability: float = 0.12,
    ) -> None:
        """初始化策略。"""
        self.rng = random.Random(seed)
        self.min_hold = min_hold
        self.max_hold = max_hold
        self.stand_probability = stand_probability
        self.reset()

    def reset(self, world: World | None = None) -> None:
        """重置内部状态（方向、停顿与保持计数）。"""
        self.direction = self.rng.choice((LEFT, RIGHT))
        self.stand_ticks = 0
        self.remaining_ticks = 0

    def __call__(self, world: World) -> int:
        """取一步动作。"""
        character: Character = world.agent
        # 在屏幕边缘反弹（"隐形墙"行为）
        if character.at_left_wall and self.direction == LEFT:
            self.direction = RIGHT
        elif character.at_right_wall and self.direction == RIGHT:
            self.direction = LEFT

        if self.remaining_ticks <= 0:
            if self.rng.random() < self.stand_probability:
                self.stand_ticks = self.rng.randint(10, 35)
            hold_seconds = self.rng.uniform(self.min_hold, self.max_hold)
            self.remaining_ticks = max(1, int(hold_seconds / world.config.fixed_dt))
            self.direction = self.rng.choice((LEFT, RIGHT))

        if self.stand_ticks > 0:
            self.stand_ticks -= 1
            self.remaining_ticks -= 1
            return STAY
        self.remaining_ticks -= 1
        return self.direction


class ConstantPolicy:
    """永远保持一个动作（边界测试用）。"""

    def __init__(self, action: int) -> None:
        """初始化策略。"""
        self.action = int(action)

    def reset(self, world: World | None = None) -> None:
        """重置策略内部状态。"""
        return None

    def __call__(self, world: World) -> int:
        """取一步动作。"""
        return self.action


POLICIES = {
    "wander": WanderPolicy,
    "random": RandomPolicy,
    "left": lambda seed=None: ConstantPolicy(LEFT),
    "right": lambda seed=None: ConstantPolicy(RIGHT),
    "stand": lambda seed=None: ConstantPolicy(STAY),
}


def make_policy(name: str, seed: int | None = None):
    """按名字构造策略（``wander``、``random``、``left``、``right``、``stand``）。"""
    try:
        factory = POLICIES[name]
    except KeyError:
        raise ValueError(
            f"unknown policy {name!r}, available: {', '.join(sorted(POLICIES))}"
        ) from None
    return factory(seed)
