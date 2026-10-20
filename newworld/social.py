"""社会/技术技能：采集、制作工具、建设社区。

这些是 agent 能执行的**技能**。写成可学习的教师策略
（``SocialLife``），用来给神经网络的行为收集训练数据：神经网络
决定何时探索/觅食等，社会层则把社区需求翻译成动作。
"""

from __future__ import annotations

from dataclasses import dataclass

from .behavior import Decision


class SocialLife:
    """手写的工具使用与社区建设教师（决策链第 3 层）。"""

    def __init__(self, seed: int = 0) -> None:
        """初始化教师。"""
        self.seed = seed

    def decide(self, world, character, dt: float) -> Decision | None:
        """社会技能教师：返回一个动作或 ``None``（交回下层控制）。

        优先级：生存门控 → 孕妇休息 → 求偶 → 建社区 → 采集缺的材料。
        """
        tech = getattr(world, "technology", None)
        if tech is None:
            return None
        community = tech.community
        inventory = character.inventory

        # -- 生存门控（P0 缺陷修复） ----------------------------------
        # 原教师会在社区建成之后给每个成年个体返回求偶/站立决策，
        # 从不返回 None，把神经网络策略整个压哑了，种群全员饿死
        # （实测：P0 基线 t=537s 时 8/8 死亡）。生存永远优先于社会/
        # 技术工作：饥饿、口渴或力竭时把控制权交回规划器/神经网络。
        physiology = character.physiology
        if physiology.energy < 35.0 or physiology.hydration < 30.0:
            return None
        if physiology.energy < 45.0 and inventory.wood < 2:
            return None

        # 孕妇待在社区附近休息
        if character.pregnant:
            character.behavior_state = "rest"
            if abs(character.x - community.x_screen) > 80.0:
                return Decision(1 if community.x_screen > character.x else -1, False, False, None)
            return Decision(0, False, False, None)

        # 社区已建成：成年人求偶（配对 / 繁殖）
        if community.built:
            if character.age < 70.0:
                return None
            mates = [
                other
                for other in world.agents
                if other.alive
                and other is not character
                and other.age >= 70.0
                and (other.sex != character.sex or character.pregnant)
            ]
            if mates:
                mate = min(mates, key=lambda other: abs(other.x - character.x))
                distance = mate.x - character.x
                if abs(distance) > 55.0:
                    # 还没走近：向对象移动
                    direction = 1 if distance > 0 else -1
                    character.facing = direction
                    character.behavior_state = "social"
                    return Decision(direction, False, False, None)
                character.facing = 1 if distance >= 0 else -1
                character.behavior_state = "social"
                return Decision(0, False, False, None)
            return None

        # 材料够了：搬去社区建造
        if inventory.wood >= 2 and inventory.stone >= 1:
            if tech.near_community(character):
                if inventory.tool is None:
                    tech.craft(character)  # 先做工具
                    character.behavior_state = "craft"
                    return Decision(0, False, False, None)
                tech.contribute(character)  # 捐材料建社区
                character.behavior_state = "build"
                return Decision(0, False, False, None)
            character.behavior_state = "forage"
            return Decision(1 if community.x_screen > character.x else -1, False, False, None)

        # 否则：去自然界采集缺的材料
        wanted = "wood" if inventory.wood < 2 else "stone"
        target = tech.nearest_resource(character, wanted)
        if target is None:
            return None
        target_x, _ = target
        character.behavior_state = "gather"
        if abs(target_x - character.x) > 14.0:
            # 走向资源
            direction = 1 if target_x > character.x else -1
            character.facing = direction
            return Decision(direction, False, False, None)
        character.facing = 1 if target_x >= character.x else -1
        tech.harvest(character, wanted)
        return Decision(0, False, False, None)
