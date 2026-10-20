"""社会：全体 agent 上的图神经网络 + 人口学过程。

**社会网络**读入每个居民的社会/生理状态，在亲密度图上聚合消息
（``affinity @ features``），为每个 agent 产出一个社会上下文向量。
这些向量写回各 agent，供其学习到的行为使用（并显示在调试面板）：
网络就是当前社会关系总和的具象化。

``Society`` 对象还处理衰老、死亡、配对、怀孕与分娩。每个孩子都是
一个真实的新 agent，继承父母的身体变体；神经网络策略由种群管理器
（population.py）创建。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import materials
from .materials import ATOM_SIZE

#: 每个个体的社会特征维数 / 输出社会上下文维数
SOCIAL_FEATURES = 11
SOCIAL_DIM = 6
LIFESPAN_S = 900.0          # 约 1.5 个虚拟日
MATURITY_S = 70.0
PREGNANCY_S = 24.0
MAX_POPULATION = 8


@dataclass
class SocietyEvent:
    """一条社会事件（出生/怀孕/死亡），UI 与日志展示用。"""

    kind: str
    text: str


class SocietyNetwork:
    """图网络：个体特征 + 亲密度消息 → 社会上下文向量。"""

    def __init__(self, seed: int = 0) -> None:
        """随机初始化两层权重（固定种子可复现；不参与训练）。"""
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.35, (SOCIAL_FEATURES * 2, 12))
        self.b1 = np.zeros(12)
        self.w2 = rng.normal(0.0, 0.35, (12, SOCIAL_DIM))
        self.b2 = np.zeros(SOCIAL_DIM)

    def forward(self, features: np.ndarray, affinity: np.ndarray) -> np.ndarray:
        """前向传播：自环加权聚合邻居特征，再过两层 tanh。"""
        if len(features) == 0:
            return np.zeros((0, SOCIAL_DIM))
        # 亲密度矩阵 + 自环（更看重自己），行归一化后做消息传递
        weights = affinity + np.eye(len(features)) * 1.5
        weights = weights / np.maximum(1e-6, weights.sum(axis=1, keepdims=True))
        messages = weights @ features
        combined = np.concatenate([features, messages], axis=1)
        hidden = np.tanh(combined @ self.w1 + self.b1)
        return np.tanh(hidden @ self.w2 + self.b2)

    def save(self, path: str | Path) -> Path:
        """把权重保存为 .npz。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SocietyNetwork":
        """从 .npz 载入权重。"""
        data = np.load(Path(path))
        network = cls()
        network.w1 = data["w1"]
        network.b1 = data["b1"]
        network.w2 = data["w2"]
        network.b2 = data["b2"]
        return network


class Society:
    """人口学过程 + 社会网络（由 world 每帧调用 update）。"""

    def __init__(self, seed: int = 0, *, on_death=None, config=None) -> None:
        """初始化社会网络与人口学参数。"""
        self.network = SocietyNetwork(seed)
        self.affinity: dict[tuple[int, int], float] = {}
        self.events: list[SocietyEvent] = []
        self.births = 0
        self.deaths = 0
        self.generation = 0
        self.last_context = np.zeros((0, SOCIAL_DIM))
        #: 回调 (agent, *, t, cause)，在移除 agent **之前**调用，
        #: 让文明层有机会归档库存/知识/记忆
        self.on_death = on_death
        #: 由 Population 设置：带预算检查的创建路径（registry 强制）
        self.population = None
        self.config = config
        #: 人口学常数可被文明配置覆盖
        self.lifespan_s = float(getattr(config, "lifespan_s", LIFESPAN_S))
        self.maturity_s = float(getattr(config, "maturity_s", MATURITY_S))
        self.pregnancy_s = float(getattr(config, "pregnancy_s", PREGNANCY_S))

    # ------------------------------------------------------------------
    def _can_create(self, world) -> bool:
        """是否可以创建新 agent（历史累计预算，而不是存活数）。

        原代码检查的是 ``len(world.agents) >= MAX_POPULATION``，即
        *存活* agent 数。死亡会把 agent 从列表移除、从而"腾出名额"，
        导致累计创建数突破 8（experiments/probe_cap_path.py 复现过）。
        挂了种群管理器时一律问它的注册表。
        """
        if self.population is not None:
            return bool(self.population.registry.can_create())
        return len(world.agents) < MAX_POPULATION

    # ------------------------------------------------------------------
    def _pair(self, a: int, b: int) -> tuple[int, int]:
        """把两个 id 归一化为一个无序对（亲密度字典的键）。"""
        return (a, b) if a < b else (b, a)

    def relate(self, a: int, b: int, amount: float = 0.4) -> None:
        """直接设置两人的亲密度。"""
        key = self._pair(a, b)
        self.affinity[key] = float(min(1.0, max(0.0, amount)))

    def affinity_between(self, a: int, b: int) -> float:
        """查询两人亲密度（默认 0.3）。"""
        return self.affinity.get(self._pair(a, b), 0.3)

    # ------------------------------------------------------------------
    def update(self, world, dt: float) -> None:
        """一帧社会更新：社会网络前向 → 靠近建立关系 → 衰老/死亡 →
        配对怀孕 → 妊娠分娩。"""
        agents = [agent for agent in world.agents if agent.alive]
        if not agents:
            return
        ids = [agent.agent_id for agent in agents]
        count = len(agents)
        # 组装亲密度矩阵（默认 0.3）
        affinity = np.full((count, count), 0.3, dtype=np.float64)
        for i in range(count):
            for j in range(i + 1, count):
                value = self.affinity_between(ids[i], ids[j])
                affinity[i, j] = affinity[j, i] = value
        features = np.asarray([self._features(agent, world, ids) for agent in agents], dtype=np.float64)
        context = self.network.forward(features, affinity)
        self.last_context = context
        # 社会上下文写回每个 agent（供学习行为/调试面板用）
        for agent, vector in zip(agents, context):
            agent.social_context = tuple(float(value) for value in vector)

        # 靠近会建立关系（亲密度随相处时间增长）
        for i in range(count):
            for j in range(i + 1, count):
                a, b = agents[i], agents[j]
                distance = abs(a.x - b.x)
                if distance < 160.0:
                    key = self._pair(a.agent_id, b.agent_id)
                    current = self.affinity.get(key, 0.3)
                    self.affinity[key] = min(1.0, current + 0.03 * dt)

        # 衰老与死亡（历史记录必须留存：先归档再移除）
        for agent in list(agents):
            agent.age += dt
            if agent.age >= self.lifespan_s or agent.physiology.energy <= 0.5:
                cause = "old_age" if agent.age >= self.lifespan_s else "starvation"
                if self.on_death is not None:
                    self.on_death(agent, t=world.t, cause=cause)
                agent.alive = False
                agent.died_at = world.t
                agent.death_cause = cause
                world.remove_agent(agent.agent_id)
                self.deaths += 1
                self.events.append(
                    SocietyEvent("death", f"{agent.agent_id}号死亡（{agent.age:.0f}s/{cause}）")
                )
                continue

        # 配对 / 怀孕（成年、靠近、亲密度与能量都达标）
        adults = [agent for agent in world.agents if agent.alive and agent.age >= self.maturity_s]
        for female in [agent for agent in adults if agent.sex == "female" and not agent.pregnant]:
            for male in [agent for agent in adults if agent.sex == "male"]:
                if abs(female.x - male.x) > 120.0:
                    continue
                if self.affinity_between(female.agent_id, male.agent_id) < 0.55:
                    continue
                if female.physiology.energy < 50.0 or male.physiology.energy < 40.0:
                    continue
                if not self._can_create(world):
                    break
                female.pregnant = True
                female.pregnancy_timer = self.pregnancy_s
                female.partner_id = male.agent_id
                male.partner_id = female.agent_id
                self.events.append(SocietyEvent("pregnant", f"{female.agent_id}号怀孕"))
                break

        # 妊娠 → 分娩
        for mother in [agent for agent in world.agents if agent.alive and agent.pregnant]:
            mother.pregnancy_timer -= dt
            if mother.pregnancy_timer > 0:
                continue
            mother.pregnant = False
            if not self._can_create(world):
                continue
            father_id = mother.partner_id
            # 性别交替出生，保持人口性别平衡
            child_sex = "female" if len(world.agents) % 2 == 0 else "male"
            child = None
            if self.population is not None:
                # 预算检查路径：登记孩子，超预算返回 None
                child = self.population.create_agent(
                    x=mother.x + (30.0 if mother.facing > 0 else -30.0),
                    facing=mother.facing,
                    sex=child_sex,
                    age=0.0,
                    generation=mother.generation + 1,
                    body_scale=0.65,
                    parents=tuple(x for x in (mother.agent_id, father_id) if x is not None),
                    parent_id=mother.agent_id,
                    cause="birth",
                )
            else:
                # 旧独立路径（没有 Population 的单元测试用）
                child = world.spawn(
                    x=mother.x + (30.0 if mother.facing > 0 else -30.0),
                    facing=mother.facing,
                    sex=child_sex,
                    age=0.0,
                    generation=mother.generation + 1,
                    body_scale=0.65,
                    parents=tuple(x for x in (mother.agent_id, father_id) if x is not None),
                    name=f"ender-{len(world.agents) + 1}",
                )
            if child is None:
                continue
            self.generation = max(self.generation, child.generation)
            # 母子/父子建立天然亲密度
            self.relate(mother.agent_id, child.agent_id, 0.9)
            if father_id is not None:
                self.relate(father_id, child.agent_id, 0.7)
            self.births += 1
            if self.population is not None:
                self.population.note_birth(child, mother=mother, father_id=father_id)
            self.events.append(SocietyEvent("birth", f"{child.agent_id}号出生（第{child.generation}代，{child_sex}）"))

    # ------------------------------------------------------------------
    def _features(self, agent, world, ids: list[int]) -> list[float]:
        """组装一个个体进入社会网络的 11 维特征（大致归一化到 0..1）。"""
        same = [self.affinity_between(agent.agent_id, other) for other in ids if other != agent.agent_id]
        community = getattr(getattr(world, "community", None), "built", False)
        progress = 0.0
        if getattr(world, "community", None) is not None:
            progress = min(1.0, (world.community.wood + world.community.stone) / 6.0)
        return [
            1.0 if agent.sex == "female" else 0.0,
            agent.age / max(1.0, self.lifespan_s),
            agent.physiology.energy / 100.0,
            agent.physiology.hydration / 100.0,
            agent.inventory.wood / 6.0,
            agent.inventory.stone / 4.0,
            progress,
            1.0 if community else 0.0,
            float(np.mean(same)) if same else 0.0,
            agent.generation / 4.0,
            max(0.4, agent.body_scale),
        ]

    # ------------------------------------------------------------------
    def summary_zh(self) -> str:
        """社会状态摘要（中文，UI 展示用）。"""
        alive = self
        return (
            f"社会网络 出生{self.births} 死亡{self.deaths} 代数{self.generation} "
            f"关系数{len(self.affinity)}"
        )

    def relation_lines(self, world) -> list[str]:
        """前 3 条异性关系（UI 调试面板展示用）。"""
        lines = []
        for i, a in enumerate(world.agents):
            for b in world.agents[i + 1:]:
                if b.sex == a.sex:
                    continue
                value = self.affinity_between(a.agent_id, b.agent_id)
                lines.append(f"{a.agent_id}↔{b.agent_id} 亲密 {value:.2f}")
        return lines[:3]
