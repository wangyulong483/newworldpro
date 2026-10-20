"""自由行为 + 心智：把行为、感知与物理世界耦合在一起。

生物是**自由的**：没有键盘控制、没有奖励、没有外部强加的目标。
它的行动来自 :class:`newworld.behavior.InnerLife`——由能量、水分、
睡眠压力与太阳钟驱动的稳态状态机。``Mind`` 把这个行为与感知
（全感官）和物理世界连成一体。
"""

from __future__ import annotations

from .behavior import Decision, InnerLife
from .neural import NeuralInnerLife, NeuralPolicy
from .perception import Perceiver, PerceptionState
from .world import World


class FreeWill(InnerLife):
    """自由行为引擎的向后兼容别名。"""

    def act(self, world: World, dt: float) -> int:
        """只返回水平动作（旧接口的辅助方法）。"""
        return self.decide(world, dt).move


class Mind:
    """完整的居民：自由行为 + 感知（全部感官）。"""

    def __init__(self, world: World, seed: int | None = 0, policy: NeuralPolicy | None = None) -> None:
        """初始化心智（可选神经策略）。"""
        self.world = world
        # 有神经网络策略就用神经内在生命，否则用规则式自由行为
        self.neural = policy is not None
        self.will = NeuralInnerLife(policy, seed) if policy is not None else FreeWill(seed)
        self.perceiver = Perceiver(seed)
        self.perception: PerceptionState | None = None
        self.action = 0
        self.last_speech = None

    # ------------------------------------------------------------------
    def step(self, dt: float) -> int:
        """活一步：决策 → 移动/跳跃/交互 → 可能说话 → 感知。"""
        if self.neural:
            decision = self.will.decide(self.world, dt, perception=self.perception)  # type: ignore[call-arg]
        else:
            decision = self.will.decide(self.world, dt)
        self.action = decision.move
        self.world.step(actions=decision, dt=dt)
        if decision.speak:
            self.last_speech = self.world.speak(self.world.agent, decision.word)
        self.perception = self.perceiver.perceive(self.world, dt)
        return self.action

    # ------------------------------------------------------------------
    def summary_lines(self) -> list[str]:
        """感知摘要（UI 调试面板展示用）。"""
        return self.perceiver.summary_lines()
