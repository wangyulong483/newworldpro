"""Society: a graph neural network over all agents + demography.

The **society network** takes every inhabitant's social/physiological state,
aggregates messages over the affinity graph (``affinity @ features``) and
produces a social context vector per agent.  Those vectors are written back to
the agents and used by their learned behaviour (and shown in the debug panel):
the network is literally the sum of the current social relations.

The ``Society`` object also handles aging, death, pair bonding, pregnancy and
birth.  Every child is a real new agent with an inherited body variant; the
population manager creates its neural policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import materials
from .materials import ATOM_SIZE

SOCIAL_FEATURES = 11
SOCIAL_DIM = 6
LIFESPAN_S = 900.0          # about 1.5 virtual days
MATURITY_S = 70.0
PREGNANCY_S = 24.0
MAX_POPULATION = 8


@dataclass
class SocietyEvent:
    kind: str
    text: str


class SocietyNetwork:
    """Graph network: agent features + affinity messages -> social context."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.35, (SOCIAL_FEATURES * 2, 12))
        self.b1 = np.zeros(12)
        self.w2 = rng.normal(0.0, 0.35, (12, SOCIAL_DIM))
        self.b2 = np.zeros(SOCIAL_DIM)

    def forward(self, features: np.ndarray, affinity: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.zeros((0, SOCIAL_DIM))
        weights = affinity + np.eye(len(features)) * 1.5
        weights = weights / np.maximum(1e-6, weights.sum(axis=1, keepdims=True))
        messages = weights @ features
        combined = np.concatenate([features, messages], axis=1)
        hidden = np.tanh(combined @ self.w1 + self.b1)
        return np.tanh(hidden @ self.w2 + self.b2)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SocietyNetwork":
        data = np.load(Path(path))
        network = cls()
        network.w1 = data["w1"]
        network.b1 = data["b1"]
        network.w2 = data["w2"]
        network.b2 = data["b2"]
        return network


class Society:
    def __init__(self, seed: int = 0) -> None:
        self.network = SocietyNetwork(seed)
        self.affinity: dict[tuple[int, int], float] = {}
        self.events: list[SocietyEvent] = []
        self.births = 0
        self.deaths = 0
        self.generation = 0
        self.last_context = np.zeros((0, SOCIAL_DIM))

    # ------------------------------------------------------------------
    def _pair(self, a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def relate(self, a: int, b: int, amount: float = 0.4) -> None:
        key = self._pair(a, b)
        self.affinity[key] = float(min(1.0, max(0.0, amount)))

    def affinity_between(self, a: int, b: int) -> float:
        return self.affinity.get(self._pair(a, b), 0.3)

    # ------------------------------------------------------------------
    def update(self, world, dt: float) -> None:
        agents = [agent for agent in world.agents if agent.alive]
        if not agents:
            return
        ids = [agent.agent_id for agent in agents]
        count = len(agents)
        affinity = np.full((count, count), 0.3, dtype=np.float64)
        for i in range(count):
            for j in range(i + 1, count):
                value = self.affinity_between(ids[i], ids[j])
                affinity[i, j] = affinity[j, i] = value
        features = np.asarray([self._features(agent, world, ids) for agent in agents], dtype=np.float64)
        context = self.network.forward(features, affinity)
        self.last_context = context
        for agent, vector in zip(agents, context):
            agent.social_context = tuple(float(value) for value in vector)

        # proximity builds relationships
        for i in range(count):
            for j in range(i + 1, count):
                a, b = agents[i], agents[j]
                distance = abs(a.x - b.x)
                if distance < 160.0:
                    key = self._pair(a.agent_id, b.agent_id)
                    current = self.affinity.get(key, 0.3)
                    self.affinity[key] = min(1.0, current + 0.03 * dt)

        # aging and death
        for agent in list(agents):
            agent.age += dt
            if agent.age >= LIFESPAN_S or agent.physiology.energy <= 0.5:
                agent.alive = False
                world.remove_agent(agent.agent_id)
                self.deaths += 1
                self.events.append(SocietyEvent("death", f"{agent.agent_id}号死亡（{agent.age:.0f}s）"))
                continue

        # pair bonding / pregnancy
        adults = [agent for agent in world.agents if agent.alive and agent.age >= MATURITY_S]
        for female in [agent for agent in adults if agent.sex == "female" and not agent.pregnant]:
            for male in [agent for agent in adults if agent.sex == "male"]:
                if abs(female.x - male.x) > 120.0:
                    continue
                if self.affinity_between(female.agent_id, male.agent_id) < 0.55:
                    continue
                if female.physiology.energy < 50.0 or male.physiology.energy < 40.0:
                    continue
                if len(world.agents) >= MAX_POPULATION:
                    break
                female.pregnant = True
                female.pregnancy_timer = PREGNANCY_S
                female.partner_id = male.agent_id
                male.partner_id = female.agent_id
                self.events.append(SocietyEvent("pregnant", f"{female.agent_id}号怀孕"))
                break

        # gestation -> birth
        for mother in [agent for agent in world.agents if agent.alive and agent.pregnant]:
            mother.pregnancy_timer -= dt
            if mother.pregnancy_timer > 0:
                continue
            mother.pregnant = False
            if len(world.agents) >= MAX_POPULATION:
                continue
            father_id = mother.partner_id
            child_sex = "female" if len(world.agents) % 2 == 0 else "male"
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
            self.generation = max(self.generation, child.generation)
            self.relate(mother.agent_id, child.agent_id, 0.9)
            if father_id is not None:
                self.relate(father_id, child.agent_id, 0.7)
            self.births += 1
            self.events.append(SocietyEvent("birth", f"{child.agent_id}号出生（第{child.generation}代，{child_sex}）"))

    # ------------------------------------------------------------------
    def _features(self, agent, world, ids: list[int]) -> list[float]:
        same = [self.affinity_between(agent.agent_id, other) for other in ids if other != agent.agent_id]
        community = getattr(getattr(world, "community", None), "built", False)
        progress = 0.0
        if getattr(world, "community", None) is not None:
            progress = min(1.0, (world.community.wood + world.community.stone) / 6.0)
        return [
            1.0 if agent.sex == "female" else 0.0,
            agent.age / LIFESPAN_S,
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
        alive = self
        return (
            f"社会网络 出生{self.births} 死亡{self.deaths} 代数{self.generation} "
            f"关系数{len(self.affinity)}"
        )

    def relation_lines(self, world) -> list[str]:
        lines = []
        for i, a in enumerate(world.agents):
            for b in world.agents[i + 1:]:
                if b.sex == a.sex:
                    continue
                value = self.affinity_between(a.agent_id, b.agent_id)
                lines.append(f"{a.agent_id}↔{b.agent_id} 亲密 {value:.2f}")
        return lines[:3]
