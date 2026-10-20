"""由生物自身生理驱动的自由行为。

这里没有任何外部控制：生物根据内在稳态驱动力（能量、水分、睡眠
压力）、太阳钟和周围感知**自己**决定做什么。得到的状态机是：

``explore`` → ``forage`` → ``eat`` / ``drink`` / ``sleep`` / ``rest``

* 饿 + 有浆果 → 走向浆果丛，伸手（或跳起来）够到浆果，吃掉；
* 渴 + 有池塘 → 走到水边，喝水；
* 夜里困了 → 躺下睡觉直到黎明；
* 其他情况 → 闲逛、东张西望、蹦跶和自言自语。

交互用身体的手位置对像素原子场景做检测，因此手必须真的够到
浆果/水才算数。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .materials import ATOM_SIZE

#: 手相对脚的高度与前伸距离（屏幕像素的近似值）
HAND_HEIGHT_PX = 40.0
HAND_REACH_PX = 40.0


@dataclass
class Decision:
    """一个自发动作（行为层的目标层都以此为输出单位）。"""

    move: int = 0
    jump: bool = False
    speak: bool = False
    word: str | None = None


class InnerLife:
    """稳态自由行为的状态机（也作为社会教师/规划器的回退层）。"""

    def __init__(self, seed: int | None = 0) -> None:
        """初始化状态机与各随机计时器。"""
        self.rng = random.Random(seed)
        self.walk_dir = self.rng.choice((-1, 1))
        # 各类自定时器：换向、蹦跶、说话、吃/喝动作时长
        self.walk_timer = self.rng.uniform(0.6, 2.4)
        self.jump_timer = self.rng.uniform(2.0, 5.0)
        self.speak_timer = self.rng.uniform(3.0, 7.0)
        self.eat_timer = 0.0
        self.drink_timer = 0.0
        self.target_x = None
        self.state = "explore"

    # ------------------------------------------------------------------
    def decide(self, world, dt: float) -> Decision:
        """看一眼世界，返回这一帧的自由决策。

        优先级：睡觉/醒来 → 喝水 → 找浆果吃 → 吃/喝动作收尾 →
        休息 → 闲逛探索。
        """
        character = world.agent
        physiology = character.physiology
        scene = world.pixel_scene
        hour = world.day.hour(world.t)
        daylight = world.day.daylight(hour)

        # -- 睡眠 / 醒来周期 -------------------------------------------
        if physiology.sleeping:
            character.behavior_state = "sleep"
            # 天亮且睡够了（或已过早晨）就醒
            if daylight > 0.02 and (physiology.sleep_pressure < 45.0 or hour > 6.8):
                physiology.wake()
            else:
                return Decision(0, False, False)
        elif (daylight < 0.06 and physiology.sleep_pressure > 55.0) or hour >= 22.0 or hour < 5.0:
            if physiology.sleep_pressure > 35.0:
                physiology.sleep()
                character.behavior_state = "sleep"
                return Decision(0, False, False)

        feet_y = character.y if character.y is not None else world.config.ground_y
        hand_x = character.x + character.facing * HAND_REACH_PX
        hand_y = feet_y - HAND_HEIGHT_PX

        # -- 环境触发的自发动作（蹦跶/说话） ---------------------------
        self.jump_timer -= dt
        ambient_jump = False
        if self.jump_timer <= 0.0:
            self.jump_timer = self.rng.uniform(2.5, 6.0)
            ambient_jump = character.on_ground
        self.speak_timer -= dt
        ambient_speak = False
        if self.speak_timer <= 0.0:
            self.speak_timer = self.rng.uniform(3.0, 8.0)
            ambient_speak = True

        # -- 喝水：手能碰到水就喝 --------------------------------------
        if physiology.hydration < 45.0 and self._near_water(scene, hand_x, hand_y):
            physiology.drink()
            character.behavior_state = "drink"
            self.drink_timer = 0.8
            return Decision(0, False, self.rng.random() < 0.5)

        # -- 吃 / 觅食 ------------------------------------------------
        berry = self._nearest_berry(scene, character.x, feet_y)
        if berry is not None and physiology.energy < 92.0:
            berry_x = berry[0] * ATOM_SIZE
            berry_y = berry[1] * ATOM_SIZE
            horizontal = abs(berry_x - character.x)
            # 低处浆果伸手就够得到；高处的得跳
            low = feet_y - 115.0 <= berry_y <= feet_y + 5.0
            reachable = horizontal < 42.0 and low
            high = berry_y < feet_y - 115.0
            if reachable:
                scene.pick_berry(berry)
                physiology.eat()
                character.behavior_state = "eat"
                self.eat_timer = 0.9
                return Decision(0, False, ambient_speak or self.rng.random() < 0.6)
            if horizontal > 18.0:
                # 还没走到：朝浆果方向移动
                move = 1 if berry_x > character.x else -1
                self.walk_dir = move
                character.behavior_state = "forage"
                return Decision(move, ambient_jump, False)
            if high and character.on_ground:
                # 够不着，跳一下试试
                return Decision(0, True, False)
            character.behavior_state = "forage"
            return Decision(0, ambient_jump, ambient_speak)

        # -- 收尾动作（吃/喝的短暂停留） -------------------------------
        if self.eat_timer > 0.0:
            self.eat_timer -= dt
            character.behavior_state = "eat"
            return Decision(0, False, False)
        if self.drink_timer > 0.0:
            self.drink_timer -= dt
            character.behavior_state = "drink"
            return Decision(0, False, False)
        if physiology.energy < 15.0:
            # 没力气觅食了：休息
            character.behavior_state = "rest"
            return Decision(0, False, self.rng.random() < 0.1)

        # -- 闲逛探索 --------------------------------------------------
        character.behavior_state = "explore"
        self.walk_timer -= dt
        if self.walk_timer <= 0.0:
            self.walk_timer = self.rng.uniform(0.7, 2.6)
            self.walk_dir = self.rng.choice((-1, 1))
            if self.rng.random() < 0.18:
                self.walk_dir = 0  # 偶尔停下歇会
        return Decision(self.walk_dir, ambient_jump, ambient_speak)

    # ------------------------------------------------------------------
    def _nearest_berry(self, scene, x: float, feet_y: float):
        """找头顶高度以下、水平距离最近的浆果（原子坐标）。"""
        best = None
        best_distance = math.inf
        for berry in scene.berries:
            bx = berry[0] * ATOM_SIZE
            by = berry[1] * ATOM_SIZE
            if by < feet_y - 190.0:  # 高出头顶太多，够不着
                continue
            distance = abs(bx - x)
            if distance < best_distance:
                best_distance = distance
                best = berry
        return best

    def _near_water(self, scene, hand_x: float, hand_y: float) -> bool:
        """手的位置附近是否有水（换算成原子坐标后查半径 4）。"""
        atom_x = int(hand_x / ATOM_SIZE)
        atom_y = int(hand_y / ATOM_SIZE)
        return scene.water_near(atom_x, atom_y, radius_atoms=4.0)
