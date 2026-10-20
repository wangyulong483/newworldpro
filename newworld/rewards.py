"""低层动作策略的奖励函数（V2：主路径第一次拥有真实奖励信号）。

设计目标
========
1. **只奖励真实世界变化**：新验证的技术、新建设施、净资源增量、新掌握的知识、
   真实的合作资源转移；不奖励"活着"本身、不奖励能量数值、不读 UI。
2. **反刷分**：
   - 原地站立 / 粮仓旁挂机 / 重复无效动作：没有任何 Δ，只剩时间成本（≤0）；
   - 能量恢复不算奖励（否则蹲在篝火旁即可无限刷分）；
   - 资源奖励按**净增量**并且有单步上限（库存满后不再增长）；
   - 死亡给固定负奖励；
   - 单步奖励 clamp 到 [-3, 3]，避免单次事件主导。
3. **可审计**：每一项都来自世界状态的前后差值，`RewardBreakdown` 可序列化，
   实验报告与测试可以逐项核对。

公式（权重见 :class:`RewardWeights`）::

    r = 1.0 * Δ已验证技术
      + 0.5 * Δ活跃公共设施
      + 0.2 * Δ净资源单位           （wood+stone+fiber+cord 的净增量，≥0）
      + 0.2 * Δ个体已掌握知识条数
      + 0.1 * Δ合作资源转移量
      - 0.001 * dt                  （时间成本，抑制原地挂机）
      - 2.0   * 死亡

    单步 clamp 到 [-3.0, 3.0]。

注意：该奖励用于**低层动作策略**（mode/move/jump/speak）的 REINFORCE 训练；
它不等于文明评估指标（技术数/存活时间等），两者分开记录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardWeights:
    """奖励权重（全部可配置，便于消融与敏感性分析）。"""

    tech_verified: float = 1.0
    facility_active: float = 0.5
    resource_net: float = 0.2
    knowledge_learned: float = 0.2
    cooperation: float = 0.1
    time_cost_per_s: float = 0.001
    death: float = -2.0
    clip_low: float = -3.0
    clip_high: float = 3.0


@dataclass
class RewardBreakdown:
    """一次奖励计算的逐项分解（可序列化、可测试）。"""

    d_tech: float = 0.0
    d_facility: float = 0.0
    d_resource: float = 0.0
    d_knowledge: float = 0.0
    d_cooperation: float = 0.0
    time_cost: float = 0.0
    death_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "d_tech": round(self.d_tech, 6),
            "d_facility": round(self.d_facility, 6),
            "d_resource": round(self.d_resource, 6),
            "d_knowledge": round(self.d_knowledge, 6),
            "d_cooperation": round(self.d_cooperation, 6),
            "time_cost": round(self.time_cost, 6),
            "death_penalty": round(self.death_penalty, 6),
            "total": round(self.total, 6),
        }


def _world_snapshot(world, agent) -> dict[str, float]:
    """记录真实世界状态：技术/设施/个体资源/个体知识/合作贡献/存活。"""
    knowledge = getattr(world, "knowledge", None)
    try:
        tech = float(len(knowledge.verified_ids())) if knowledge is not None else 0.0
    except Exception:
        tech = 0.0
    civilization = getattr(world, "civilization", None)
    try:
        facilities = list(getattr(civilization, "facilities", ()) or ())
        facility = float(len([f for f in facilities if getattr(f, "active", False)]))
    except Exception:
        facility = 0.0
    inventory = getattr(agent, "inventory", None)
    try:
        resource = float(inventory.load()) if inventory is not None else 0.0
    except Exception:
        resource = 0.0
    try:
        learned = float(len(getattr(agent, "known_knowledge_ids", ()) or ()))
    except Exception:
        learned = 0.0
    try:
        cooperation = float(
            (getattr(agent, "contributions", {}) or {}).get("share_food", 0.0)
        )
    except Exception:
        cooperation = 0.0
    return {
        "tech": tech,
        "facility": facility,
        "resource": resource,
        "learned": learned,
        "cooperation": cooperation,
    }


class LowLevelReward:
    """按真实世界前后差值计算低层动作策略的单步奖励。"""

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.weights = weights or RewardWeights()
        self._previous: dict[int, dict[str, float]] = {}
        #: 逐项审计日志（有上限，便于事后核对）
        self.history: list[dict[str, Any]] = []
        self.history_limit = 2048
        self.last_breakdown: RewardBreakdown | None = None

    # ------------------------------------------------------------------
    def observe(self, world, agent, dt: float = 0.0) -> RewardBreakdown:
        """记录一次观察；返回相对上一次观察的奖励分解。

        ``dt`` 为该步的世界时间（用于时间成本）；第一次观察只建立基线，
        返回 0（没有可比的前值）。
        """
        agent_id = int(getattr(agent, "agent_id", -1))
        current = _world_snapshot(world, agent)
        previous = self._previous.get(agent_id)
        self._previous[agent_id] = current
        w = self.weights
        if previous is None:
            breakdown = RewardBreakdown()
            self.last_breakdown = breakdown
            return breakdown
        d_resource = max(0.0, current["resource"] - previous["resource"])
        breakdown = RewardBreakdown(
            d_tech=w.tech_verified * max(0.0, current["tech"] - previous["tech"]),
            d_facility=w.facility_active * max(0.0, current["facility"] - previous["facility"]),
            d_resource=w.resource_net * d_resource,
            d_knowledge=w.knowledge_learned * max(0.0, current["learned"] - previous["learned"]),
            d_cooperation=w.cooperation * max(0.0, current["cooperation"] - previous["cooperation"]),
            time_cost=-w.time_cost_per_s * max(0.0, float(dt)),
        )
        total = (
            breakdown.d_tech
            + breakdown.d_facility
            + breakdown.d_resource
            + breakdown.d_knowledge
            + breakdown.d_cooperation
            + breakdown.time_cost
        )
        breakdown.total = max(w.clip_low, min(w.clip_high, total))
        self.last_breakdown = breakdown
        if len(self.history) < self.history_limit:
            self.history.append({"agent_id": agent_id, **breakdown.to_dict()})
        return breakdown

    # ------------------------------------------------------------------
    def death_bonus(self) -> RewardBreakdown:
        """个体死亡时的固定负奖励。"""
        w = self.weights
        breakdown = RewardBreakdown(death_penalty=w.death, total=w.death)
        self._previous.pop(-1, None)
        return breakdown

    def forget(self, agent_id: int) -> None:
        self._previous.pop(int(agent_id), None)

    def summary(self) -> dict[str, float]:
        """奖励分量均值（供实验报告做"奖励是否被刷"的审计）。"""
        if not self.history:
            return {}
        keys = ("d_tech", "d_facility", "d_resource", "d_knowledge",
                "d_cooperation", "time_cost", "total")
        count = float(len(self.history))
        return {key: sum(float(item[key]) for item in self.history) / count for key in keys}
