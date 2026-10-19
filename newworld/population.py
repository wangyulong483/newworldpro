"""A small multi-agent population: per-agent brains, society and demography.

Each inhabitant has its **own** neural policy (plus a shared social skill
teacher).  Every step the population:

1. asks every agent brain for a decision (social teacher overrides when the
   community needs materials),
2. steps the physical world once with all actions,
3. lets the experienced agents perceive, speak and update the society network,
4. handles births/deaths and creates/destroys the corresponding brains.

Checkpoints are stored per agent under ``assets/agents/``; F2 resets every
agent's network from scratch, automatic training continues each agent's own
weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .neural import NeuralInnerLife, NeuralPolicy
from .perception import Perceiver
from .society import Society, SocietyNetwork
from .social import SocialLife
from .training import DEFAULT_POLICY, fine_tune, pretrain

AGENT_DIR = Path(__file__).resolve().parent.parent / "assets" / "agents"
SOCIETY_PATH = Path(__file__).resolve().parent.parent / "assets" / "society.npz"


class AgentMind:
    def __init__(self, character, policy: NeuralPolicy, seed: int) -> None:
        self.character = character
        self.policy = policy
        self.will = NeuralInnerLife(policy, seed)
        self.social = SocialLife(seed)
        self.perceiver = Perceiver(seed)
        self.perception = None
        self.last_decision = None

    def decide(self, world, dt: float):
        decision = self.social.decide(world, self.character, dt)
        if decision is None:
            decision = self.will.decide(
                world,
                dt,
                perception=self.perception,
                character=self.character,
            )
        self.last_decision = decision
        return decision


class Population:
    def __init__(self, world, seed: int = 0) -> None:
        self.world = world
        self.seed = seed
        self.society = Society(seed)
        if SOCIETY_PATH.exists():
            try:
                self.society.network = SocietyNetwork.load(SOCIETY_PATH)
            except Exception:
                SOCIETY_PATH.parent.mkdir(parents=True, exist_ok=True)
                self.society.network.save(SOCIETY_PATH)
        else:
            self.society.network.save(SOCIETY_PATH)
        self.minds: dict[int, AgentMind] = {}
        self._ensure_initial_agents()
        self.sync()

    # ------------------------------------------------------------------
    def _ensure_initial_agents(self) -> None:
        if len(self.world.agents) == 0:
            y = self.world.config.ground_y
            self.world.spawn(sex="male", age=150.0, x=self.world.config.width * 0.42, name="ender")
            self.world.spawn(sex="female", age=130.0, x=self.world.config.width * 0.52, name="ender-2")
        elif len(self.world.agents) == 1:
            male = self.world.agents[0]
            male.sex = "male"
            male.age = 150.0
            male.name = "ender"
            self.world.spawn(sex="female", age=130.0, x=male.x + 120.0, name="ender-2")
        for agent in self.world.agents:
            if agent.age <= 0.0:
                agent.age = 150.0
            agent.inventory.wood = 0
            agent.inventory.stone = 0

    # ------------------------------------------------------------------
    def policy_path(self, agent_id: int) -> Path:
        return AGENT_DIR / f"agent_{agent_id}.npz"

    def _load_policy(self, agent) -> NeuralPolicy:
        path = self.policy_path(agent.agent_id)
        if path.exists():
            try:
                return NeuralPolicy.load(path)
            except ValueError:
                pass
        if DEFAULT_POLICY.exists():
            try:
                base = NeuralPolicy.load(DEFAULT_POLICY)
                return NeuralPolicy(weights=base.flat_weights().copy(), seed=self.seed + agent.agent_id)
            except ValueError:
                pass
        return NeuralPolicy(seed=self.seed + agent.agent_id)

    def _inherit_policy(self, agent) -> NeuralPolicy:
        parents = [self.minds[pid] for pid in agent.parents if pid in self.minds]
        if parents:
            father = parents[0].policy.flat_weights()
            mother = parents[-1].policy.flat_weights()
            rng = np.random.default_rng(self.seed + agent.agent_id)
            mask = rng.random(father.shape) < 0.5
            child = np.where(mask, father, mother)
            child = child + rng.normal(0.0, 0.03, child.shape)
            return NeuralPolicy(weights=child, seed=self.seed + agent.agent_id)
        return NeuralPolicy(seed=self.seed + agent.agent_id)

    # ------------------------------------------------------------------
    def sync(self) -> None:
        """Create brains for newborns and remove brains of the dead."""
        alive_ids = {agent.agent_id for agent in self.world.agents if agent.alive}
        for agent_id in list(self.minds):
            if agent_id not in alive_ids:
                del self.minds[agent_id]
        for agent in self.world.agents:
            if not agent.alive or agent.agent_id in self.minds:
                continue
            if agent.parents:
                policy = self._inherit_policy(agent)
            else:
                policy = self._load_policy(agent)
            self.minds[agent.agent_id] = AgentMind(agent, policy, self.seed + agent.agent_id)

    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        agents = [agent for agent in self.world.agents if agent.alive]
        decisions = []
        active = []
        for agent in agents:
            mind = self.minds.get(agent.agent_id)
            if mind is None:
                continue
            active.append((agent, mind))
            decisions.append(mind.decide(self.world, dt))
        if not decisions:
            return
        self.world.step(actions=decisions, dt=dt)
        for agent, mind in active:
            decision = mind.last_decision
            if decision is not None and decision.speak:
                self.world.speak(agent, decision.word)
            mind.perception = mind.perceiver.perceive(self.world, dt, character=agent)
        self.society.update(self.world, dt)
        self.sync()

    # ------------------------------------------------------------------
    def reset_policies(self, seed: int | None = None) -> None:
        """F2: new random neural network for every agent (from zero)."""
        seed = self.seed if seed is None else seed
        self.society.network = SocietyNetwork(seed)
        self.society.network.save(SOCIETY_PATH)
        for agent in self.world.agents:
            policy = NeuralPolicy(seed=seed + agent.agent_id)
            self.policy_path(agent.agent_id).parent.mkdir(parents=True, exist_ok=True)
            policy.save(self.policy_path(agent.agent_id))
            mind = self.minds.get(agent.agent_id)
            if mind is not None:
                mind.policy = policy
                mind.will.policy = policy

    def train_round(self, seed: int = 0, reset: bool = False) -> dict[str, float]:
        """Train every agent's own network (from scratch if ``reset``)."""
        if reset:
            self.reset_policies(seed)
        total = 0.0
        accuracy = 0.0
        for agent in self.world.agents:
            if not agent.alive:
                continue
            path = self.policy_path(agent.agent_id)
            if reset:
                policy, metrics = pretrain(path, seed=seed + agent.agent_id, steps_per_hour=160, epochs=10)
            else:
                policy, metrics = fine_tune(path, seed=seed + agent.agent_id, steps_per_hour=120, epochs=4)
            mind = self.minds.get(agent.agent_id)
            if mind is not None:
                mind.policy = policy
                mind.will.policy = policy
            total += 1.0
            accuracy += metrics["mode_accuracy"]
        self.society.network.save(SOCIETY_PATH)
        return {"agents": total, "mode_accuracy": accuracy / max(1.0, total)}

    # ------------------------------------------------------------------
    def summary_zh(self) -> str:
        agents = [agent for agent in self.world.agents if agent.alive]
        names = " ".join(
            f"{agent.name or agent.agent_id}号({'♀' if agent.sex == 'female' else '♂'},{agent.generation}代)"
            for agent in agents
        )
        return (
            f"人口 {len(agents)} · {self.society.summary_zh()} · "
            f"{self.world.technology.summary_zh()} · 角色：{names}"
        )
