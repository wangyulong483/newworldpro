"""A small multi-agent population: per-agent brains, society and demography.

This is the integration layer of the civilization redesign.  Every step:

1. every living agent decides (``GoalManager`` if planning is enabled → hand
   written social teacher → its own neural policy),
2. the physical world steps once with all actions,
3. agents speak / perceive (perception is optional: it costs ~95% of the frame,
   and the neural feature extractor ignores it, so experiments disable it),
4. :class:`~newworld.society.Society` ages agents, handles pair bonding, birth
   and death, **archiving the dead into the historical registry** instead of
   throwing them away,
5. the civilization layer decays facilities, tracks collapse/extinction and
   registers new structures,
6. knowledge is transmitted between neighbours (social learning), memory records
   episodes, facility effects apply, and telemetry samples the world.

The 8-agent **historical** creation budget is enforced by
:meth:`~newworld.world.World.create_agent`; :meth:`Population.create_agent` is
the only sanctioned birth path.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .behavior import Decision
from .goals import GoalManager
from .neural import NeuralInnerLife, NeuralPolicy
from .perception import Perceiver
from .society import Society, SocietyNetwork
from .social import SocialLife
from .training import DEFAULT_POLICY, fine_tune, pretrain

AGENT_DIR = Path(__file__).resolve().parent.parent / "assets" / "agents"
SOCIETY_PATH = Path(__file__).resolve().parent.parent / "assets" / "society.npz"

#: how often (world seconds) a behaviour change is written into episodic memory
EPISODE_INTERVAL_S = 0.5
#: distance within which living agents can teach each other verified knowledge
TEACHING_RANGE_PX = 170.0
#: how often (world seconds) the limited combinatorial innovation lab is tried
INNOVATION_INTERVAL_S = 5.0

#: reward weights of the learned high-level goal policy (real world deltas)
REWARD_ALIVE_PER_S = 0.02
REWARD_TECH = 5.0
REWARD_FACILITY = 3.0
REWARD_RESOURCE = 0.5
REWARD_KNOWLEDGE = 2.0
REWARD_COOPERATION = 1.0
REWARD_GOAL_FAILED = -2.5
REWARD_DEATH = -8.0
REWARD_INFEASIBLE_GOAL = -0.5


class AgentMind:
    def __init__(self, character, policy: NeuralPolicy, seed: int, memory=None) -> None:
        self.character = character
        self.policy = policy
        self.will = NeuralInnerLife(policy, seed)
        self.social = SocialLife(seed)
        self.perceiver = Perceiver(seed)
        self.perception = None
        self.last_decision = None
        self.memory = memory
        #: which module produced the last decision ("teacher" | "neural")
        self.last_source = "neural"

    def decide(self, world, dt: float):
        decision = self.social.decide(world, self.character, dt)
        if decision is None:
            decision = self.will.decide(
                world,
                dt,
                perception=self.perception,
                character=self.character,
            )
            self.last_source = "neural"
        else:
            self.last_source = "teacher"
        self.last_decision = decision
        return decision


class Population:
    def __init__(self, world, seed: int = 0, *, telemetry=None) -> None:
        self.world = world
        self.seed = seed
        self.config = getattr(world, "civ_config", None)
        # shared civilization services live on the world (single source of truth)
        self.registry = world.registry
        self.memory = world.memory
        self.knowledge = world.knowledge
        self.civilization = world.civilization
        self.society = Society(seed, on_death=self._on_death, config=self.config)
        if SOCIETY_PATH.exists():
            try:
                self.society.network = SocietyNetwork.load(SOCIETY_PATH)
            except Exception:
                SOCIETY_PATH.parent.mkdir(parents=True, exist_ok=True)
                self.society.network.save(SOCIETY_PATH)
        else:
            SOCIETY_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.society.network.save(SOCIETY_PATH)
        self.society.population = self
        self.goals = (
            GoalManager(
                seed=seed,
                knowledge=self.knowledge,
                memory=self.memory,
                civilization=self.civilization,
            )
            if self._flag("enable_planning", True)
            else None
        )
        self.telemetry = telemetry
        self.minds: dict[int, AgentMind] = {}
        self._last_episode_at: dict[int, float] = {}
        self._last_state: dict[int, str] = {}
        # -- learned high-level decision layer (this upgrade) ----------------
        #: GoalPolicy (newworld/highlevel.py) or a compatible object; None means
        #: the heuristic goal priorities are used (previous behaviour)
        self.goal_policy = None
        #: (features, action_index, reward) tuples for the policy-gradient trainer
        self.policy_transitions: list[tuple[np.ndarray, int, float]] = []
        #: per-agent pending policy episode: features/action/start snapshot
        self._policy_pending: dict[int, dict[str, object]] = {}
        #: which module made each per-frame decision (audit / anti-misattribution)
        self.decision_counts: Counter[str] = Counter()
        # -- limited combinatorial innovation --------------------------------
        self._innovation_lab = None
        self._last_innovation_t = -1e9
        # expose the goal manager on the world so neural.features() can read it
        world.goal_manager = self.goals
        self._ensure_initial_agents()
        self.sync()

    # ------------------------------------------------------------------
    def _flag(self, name: str, default: bool) -> bool:
        if self.config is None:
            return default
        if hasattr(self.config, name):
            return bool(getattr(self.config, name))
        if isinstance(self.config, dict):
            return bool(self.config.get(name, default))
        return default

    # ------------------------------------------------------------------
    def _ensure_initial_agents(self) -> None:
        world = self.world
        if len(world.agents) == 0:
            y = world.config.ground_y
            first = world.create_agent(
                sex="male", age=150.0, x=world.config.width * 0.42, name="ender", cause="initial"
            )
            second = world.create_agent(
                sex="female", age=130.0, x=world.config.width * 0.52, name="ender-2",
                cause="initial",
            )
            for agent in (first, second):
                if agent is not None:
                    agent.attach(world.config)
                    agent.y = y
        elif len(world.agents) == 1:
            male = world.agents[0]
            male.sex = "male"
            male.age = 150.0
            male.name = male.name or "ender"
            world.adopt_agent(male)
            world.create_agent(
                sex="female",
                age=130.0,
                x=male.x + 120.0,
                name="ender-2",
                cause="initial",
            )
        for agent in world.agents:
            if agent.age <= 0.0:
                agent.age = 150.0
            # start from a clean material economy; knowledge/memory persist
            agent.inventory.wood = 0
            agent.inventory.stone = 0
            agent.created_at = agent.created_at or 0.0
            world.adopt_agent(agent)

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
        """Create brains for newborns; drop the brains of the dead.

        Dropping the *brain object* is intentional (a dead agent must never act
        again) — everything that has civilizational value (inventory, known
        knowledge, skills, memory entries, identity) was archived by
        :meth:`_on_death` into the historical registry before this point.
        """
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
            self.minds[agent.agent_id] = AgentMind(
                agent, policy, self.seed + agent.agent_id, memory=self.memory
            )

    # ------------------------------------------------------------------
    # creation / death (the only sanctioned paths)
    # ------------------------------------------------------------------
    def create_agent(self, *, parent_id: int | None = None, cause: str = "birth", **kwargs):
        """Create a new agent if the historical budget allows it.

        Returns the new :class:`~newworld.character.Character` or ``None`` when
        the 8-agent total has already been reached.  Inheritance (memory,
        knowledge) is handled by :meth:`note_birth`/``Society``.
        """
        child = self.world.create_agent(parent_id=parent_id, cause=cause, **kwargs)
        if child is None:
            return None
        self.sync()
        return child

    def note_birth(self, child, *, mother=None, father_id: int | None = None) -> None:
        """P3: transfer procedural memory and knowledge to a newborn."""
        if not self._flag("enable_inheritance", True):
            return
        parents = [pid for pid in ((mother.agent_id if mother else None), father_id) if pid is not None]
        for parent_id in parents:
            try:
                self.memory.transfer(parent_id, child.agent_id)
            except Exception:
                pass
        inherited: list[str] = []
        for parent_id in parents:
            parent = self.world.agent_by_id(parent_id)
            if parent is None:
                record = self.registry.record(parent_id)
                source = list(record.known_knowledge_ids) if record is not None else []
            else:
                source = list(parent.known_knowledge_ids)
            for knowledge_id in source:
                knowledge = self.knowledge.get(knowledge_id)
                if knowledge is not None and knowledge.usable() and knowledge_id not in inherited:
                    inherited.append(knowledge_id)
        if inherited:
            child.known_knowledge_ids = sorted(set(child.known_knowledge_ids) | set(inherited))
            self._emit("inherit", agent_id=child.agent_id, knowledge=list(inherited))

    def _on_death(self, agent, *, t: float, cause: str) -> None:
        """Archive the dead agent *before* it leaves ``world.agents``.

        The historical record is always kept (task constraint: a dead agent's
        history and knowledge record must not be deleted).  Only the *transfer*
        of memory/knowledge to others is a mechanism switch.
        """
        # close a pending policy episode with the death penalty (honest signal)
        pending = self._policy_pending.pop(agent.agent_id, None)
        if pending is not None:
            reward = self._goal_reward(dict(pending["snapshot"]), failed=False, died=True)  # type: ignore[arg-type]
            self.policy_transitions.append(
                (np.asarray(pending["features"], dtype=np.float64), int(pending["action"]), reward)
            )
        final_state = {
            "x": round(float(agent.x), 2),
            "energy": round(float(agent.physiology.energy), 2),
            "hydration": round(float(agent.physiology.hydration), 2),
            "behavior_state": agent.behavior_state,
            "age": round(float(agent.age), 2),
            # exact archive field names understood by AgentRegistry.mark_dead;
            # numeric counters only (the tool name goes to final_state)
            "inventory": {
                "wood": int(agent.inventory.wood),
                "stone": int(agent.inventory.stone),
                "fiber": int(agent.inventory.fiber),
                "cord": int(agent.inventory.cord),
                "basket": int(agent.inventory.basket),
            },
            "known_knowledge_ids": list(agent.known_knowledge_ids),
            "skills": dict(agent.skills),
        }
        try:
            self.memory.archive(agent.agent_id)
        except Exception:
            pass
        self.registry.mark_dead(agent.agent_id, t=t, cause=cause, final_state=final_state)
        # keep the lifetime contribution record even after death (the
        # civilization-level cooperation index must not reset when the last
        # individual dies)
        record = self.registry.record(agent.agent_id)
        if record is not None:
            record.contributions = {str(key): float(value) for key, value in agent.contributions.items()}
        self._emit("death", agent_id=agent.agent_id, cause=cause, t=round(t, 2))

    # ------------------------------------------------------------------
    # per-step decision
    # ------------------------------------------------------------------
    def _infrastructure_directive(self, agent) -> Decision | None:
        """Use public facilities when the individual is running low.

        Berries only regrow in daylight, so a group that never settles starves at
        night however good its technology is.  This reflex makes agents walk to an
        active food/warmth structure (granary / campfire / hut) and rest there —
        the mechanism by which built infrastructure actually changes survival, and
        the reason a village forms around the granary instead of staying scattered.
        Only structures that already exist (i.e. were built through validated
        technology) can attract anyone, so it cannot fake progress.
        """
        facilities = getattr(self.civilization, "facilities", None)
        if not facilities:
            return None
        physiology = agent.physiology
        if physiology.energy >= 55.0 or physiology.hydration < 25.0:
            return None
        best = None
        best_distance = 1e9
        for facility in facilities:
            if not getattr(facility, "active", False):
                continue
            if getattr(facility, "kind", "") not in ("granary", "campfire", "hut"):
                continue
            distance = abs(float(facility.x_screen) - agent.x)
            if distance < best_distance:
                best_distance = distance
                best = facility
        if best is None:
            return None
        if best_distance > 40.0:
            direction = 1 if best.x_screen > agent.x else -1
            agent.behavior_state = "forage"
            return Decision(direction, False, False, None)
        agent.behavior_state = "rest"
        return Decision(0, False, False, None)

    def _low_level_decision(self, agent, mind: AgentMind, dt: float):
        """Primitive action from the configured low-level controller.

        ``full`` = behaviour-cloning network, ``random`` = random walk,
        ``off`` = stand still (decision-chain ablations).  Applies to *all*
        delegated frames as well, otherwise the ablation would be bypassed by the
        planner's explore delegation.
        """
        mode = str(getattr(self.config, "low_level_mode", "full") if self.config else "full")
        if mode == "off":
            self.decision_counts["lowlevel_off"] += 1
            mind.last_decision = Decision(0, False, False, None)
            return mind.last_decision
        if mode == "random":
            self.decision_counts["lowlevel_random"] += 1
            rng = self.world.rng
            direction = rng.choice((-1, 0, 1))
            mind.last_decision = Decision(direction, rng.random() < 0.03, False, None)
            return mind.last_decision
        decision = mind.will.decide(
            self.world, dt, perception=mind.perception, character=agent
        )
        self.decision_counts["neural"] += 1
        mind.last_decision = decision
        return decision

    def _decide(self, agent, mind: AgentMind, dt: float):
        directive = self._infrastructure_directive(agent)
        if directive is not None:
            self.decision_counts["infrastructure"] += 1
            mind.last_decision = directive
            return directive
        if self.goal_policy is not None:
            self._maybe_request_policy_goal(agent)
        if self.goals is not None:
            decision = self.goals.update(self.world, agent, dt)
            if decision is not None:
                self.decision_counts["goal"] += 1
                mind.last_decision = decision
                return decision
            if self.goals.consume_delegation(agent.agent_id):
                # the semantic layer chose "explore/idle" and delegated the
                # primitive action: go straight to the low-level controller.
                # Do NOT let the mate-seeking teacher hijack these frames — that
                # produced a stand-still loop (measured repeat_run_max 9476).
                return self._low_level_decision(agent, mind, dt)
        mode = str(getattr(self.config, "low_level_mode", "full") if self.config else "full")
        if mode != "full":
            return self._low_level_decision(agent, mind, dt)
        decision = mind.decide(self.world, dt)
        self.decision_counts[str(mind.last_source)] += 1
        return decision

    # ------------------------------------------------------------------
    # learned high-level goal policy integration (real REINFORCE loop)
    # ------------------------------------------------------------------
    def attach_goal_policy(self, policy) -> None:
        """Attach a learned goal-selection policy (idempotent, replaces old one).

        ``policy`` only needs ``sample(features) -> action_index``; the trainer in
        ``newworld/highlevel.py`` calls this on the env it is given, so the very
        policy under training is the one driving the simulation (no silent
        fallback to a different policy).
        """
        self.goal_policy = policy
        self._policy_pending.clear()

    def detach_goal_policy(self) -> None:
        self.goal_policy = None
        self._policy_pending.clear()

    def drain_policy_transitions(self) -> list[tuple[np.ndarray, int, float]]:
        """Pop all finished (features, action_index, reward) transitions."""
        out = self.policy_transitions
        self.policy_transitions = []
        return out

    def _snapshot_metrics(self) -> dict[str, float]:
        alive = [agent for agent in self.world.agents if agent.alive]
        return {
            "t": float(self.world.t),
            "tech": float(len(self.knowledge.verified_ids())),
            "fac": float(
                len([f for f in self.civilization.facilities if getattr(f, "active", False)])
            ),
            "res": float(sum(agent.inventory.load() for agent in alive)),
            "known": float(sum(len(agent.known_knowledge_ids) for agent in alive)),
            "coop": float(sum(agent.contributions.get("share_food", 0.0) for agent in alive)),
        }

    def _goal_reward(self, start: dict[str, float], *, failed: bool, died: bool) -> float:
        """Reward of one policy episode, computed from **real world changes**.

        Formula (documented in docs/EXPERIMENT_REPORT.md): survival time,
        newly verified technologies, active facilities, resource units, known
        knowledge and cooperation transfers are rewarded; goal failure and death
        are penalised.  It uses only world state deltas measured between the
        moment the goal was requested and its end — never BC labels.
        """
        now = self._snapshot_metrics()
        reward = REWARD_ALIVE_PER_S * max(0.0, now["t"] - start["t"])
        reward += REWARD_TECH * (now["tech"] - start["tech"])
        reward += REWARD_FACILITY * (now["fac"] - start["fac"])
        reward += REWARD_RESOURCE * (now["res"] - start["res"])
        reward += REWARD_KNOWLEDGE * (now["known"] - start["known"])
        reward += REWARD_COOPERATION * (now["coop"] - start["coop"])
        if failed:
            reward += REWARD_GOAL_FAILED
        if died:
            reward += REWARD_DEATH
        return float(max(-25.0, min(35.0, reward)))

    def _maybe_request_policy_goal(self, agent) -> None:
        """Ask the learned policy for a goal when the agent has none active."""
        policy = self.goal_policy
        if policy is None or self.goals is None:
            return
        agent_id = agent.agent_id
        if agent_id in self._policy_pending:
            return
        if self.goals.has_active_goal(agent_id) or self.goals.active_plan(agent_id) is not None:
            return
        try:
            from .highlevel import GOAL_ACTIONS, extract_features

            features = np.asarray(extract_features(self.world, agent, self), dtype=np.float64)
            index = int(policy.sample(features))
            action = GOAL_ACTIONS[index] if 0 <= index < len(GOAL_ACTIONS) else "explore"
        except Exception:
            self.decision_counts["policy_error"] += 1
            return
        snapshot = self._snapshot_metrics()
        goal = self.goals.request_goal(self.world, agent, action)
        if goal is None:
            # infeasible action: small negative reward, no episode (honest signal)
            self.policy_transitions.append((features, index, REWARD_INFEASIBLE_GOAL))
            self.decision_counts["policy_infeasible"] += 1
            return
        self._policy_pending[agent_id] = {
            "features": features,
            "action": index,
            "snapshot": snapshot,
            "goal_type": action,
        }
        self.decision_counts["policy_goal"] += 1

    def _settle_policy_episodes(self) -> None:
        """Close policy episodes whose goal ended (completed/failed/dead)."""
        if not self._policy_pending:
            return
        for agent_id in list(self._policy_pending):
            pending = self._policy_pending[agent_id]
            agent = self.world.agent_by_id(agent_id)
            alive = agent is not None and agent.alive
            goal = self.goals.current_goal(agent_id) if self.goals is not None else None
            if alive and goal is not None and goal.status == "active":
                continue
            failed = bool(goal is not None and goal.status in ("failed", "abandoned"))
            reward = self._goal_reward(
                dict(pending["snapshot"]),  # type: ignore[arg-type]
                failed=failed,
                died=not alive,
            )
            self.policy_transitions.append(
                (np.asarray(pending["features"], dtype=np.float64), int(pending["action"]), reward)
            )
            del self._policy_pending[agent_id]

    def decision_share(self) -> dict[str, float]:
        """Fraction of per-frame decisions made by each module (audit metric)."""
        total = float(sum(self.decision_counts.values()))
        if total <= 0.0:
            return {}
        return {key: round(value / total, 4) for key, value in self.decision_counts.items()}

    def step(self, dt: float) -> None:
        agents = [agent for agent in self.world.agents if agent.alive]
        if not agents:
            self._advance_empty(dt)
            self.civilization.update(self.world, dt)
            self._enforce_extinction_rules()
            self._emit_step(dt)
            return
        self.sync()
        decisions = []
        active = []
        for agent in agents:
            mind = self.minds.get(agent.agent_id)
            if mind is None:
                continue
            active.append((agent, mind))
            decisions.append(self._decide(agent, mind, dt))
        if not decisions:
            self._advance_empty(dt)
            self.civilization.update(self.world, dt)
            self._enforce_extinction_rules()
            self._emit_step(dt)
            return
        self.world.step(actions=decisions, dt=dt)
        perceive = self._flag("enable_perception", True)
        for agent, mind in active:
            decision = mind.last_decision
            if decision is not None and decision.speak:
                self.world.speak(agent, decision.word)
            if perceive:
                mind.perception = mind.perceiver.perceive(self.world, dt, character=agent)
        self.society.update(self.world, dt)
        self.sync()
        self.civilization.update(self.world, dt)
        self._consume_facility_queue()
        for agent in [a for a in self.world.agents if a.alive]:
            self.world.technology.apply_facility_effects(
                agent, self.civilization.facilities, dt
            )
            self._maybe_record_episode(agent)
        if self._flag("enable_inheritance", True):
            self._social_learning(dt)
        self._cooperation(dt)
        self._maybe_innovate(dt)
        self._settle_policy_episodes()
        self._enforce_extinction_rules()
        self._emit_step(dt)

    # ------------------------------------------------------------------
    # limited combinatorial innovation integration
    # ------------------------------------------------------------------
    def _maybe_innovate(self, dt: float) -> None:
        """Try one bounded combinational invention per interval (real materials).

        Only runs when ``enable_innovation`` is on; proposals come from
        :class:`newworld.innovation.InnovationLab`, which validates components
        against the agent's **verified** knowledge and registers effects through
        ``Technology.register_custom_effect``.  A successful invention changes
        real numbers (harvest yield / carry capacity / regeneration).
        """
        if not self._flag("enable_innovation", False):
            return
        if self._innovation_lab is None:
            try:
                from .innovation import InnovationLab

                self._innovation_lab = InnovationLab(seed=self.seed)
            except Exception:
                return
        if self.world.t - self._last_innovation_t < INNOVATION_INTERVAL_S:
            return
        self._last_innovation_t = self.world.t
        lab = self._innovation_lab
        # lifecycle guard: World.reset() rebuilds Technology (custom effects are
        # lost) while the lab keeps its inventions — re-register them so records
        # and real world effects never diverge.
        try:
            for tool_name, effects in lab.custom_effects().items():
                if tool_name not in self.world.technology.custom_effects:
                    self.world.technology.register_custom_effect(tool_name, dict(effects))
        except Exception:
            pass
        for agent in [a for a in self.world.agents if a.alive]:
            try:
                # only attempt combinations the individual can actually afford:
                # an unaffordable proposal is a precondition miss, not an experiment
                proposals = lab.proposals(agent, self.knowledge, affordable_only=True)
            except Exception:
                continue
            if not proposals:
                continue
            try:
                ok, reason, innovation = lab.attempt(agent, proposals[0], self.world)
            except Exception:
                continue
            if ok and innovation is not None:
                # capacity is a property of the tool the agent now carries: read it
                # from the technology effect table so tools registered elsewhere work too
                bonus = int(round(float(
                    self.world.technology.custom_effect(
                        agent.inventory.tool, "capacity_bonus"
                    )
                )))
                if bonus > 0:
                    agent.inventory.capacity_bonus = max(agent.inventory.capacity_bonus, bonus)
                try:
                    self.memory.record(
                        agent.agent_id,
                        self.world.t,
                        "procedural",
                        f"innovate:{innovation.innovation_id}",
                        f"ok:{innovation.name_zh}",
                        True,
                        detail=str(innovation.effects),
                        salience=0.8,
                    )
                except Exception:
                    pass
                self._emit(
                    "innovation",
                    agent_id=agent.agent_id,
                    innovation=innovation.innovation_id,
                    name=innovation.name_zh,
                    effects=dict(innovation.effects),
                )
            elif not ok and reason not in (
                "missing_material",
                "no_effect_backend",
                "already_invented",
                "max_innovations",
                "component_not_verified",
                "component_not_known",
                "unknown_attribute",
                "incompatible_tags",
            ):
                self._emit("innovation_failed", agent_id=agent.agent_id, reason=str(reason))

    def innovations(self) -> list[dict[str, object]]:
        lab = self._innovation_lab
        if lab is None:
            return []
        try:
            return [item.to_dict() for item in lab.innovations()]
        except Exception:
            return []

    def _cooperation(self, dt: float) -> None:
        """P4: food sharing between neighbours (a real, measurable social rule).

        When a well-fed individual stands next to a starving one, it gives part of
        its energy away (up to 10 points).  This is not decoration: it keeps
        foragers alive through a bad night, and it is counted in
        ``cooperation_index`` / ``cooperation_benefit`` because the transfer is a
        genuine resource flow that no individual could have produced alone.
        Gated by ``enable_social`` so ablation group C (no social mechanisms)
        cannot use it.
        """
        if not self._flag("enable_social", True):
            return
        alive = [agent for agent in self.world.agents if agent.alive]
        if len(alive) < 2:
            return
        chance = min(1.0, 0.5 * dt)
        for donor in alive:
            if donor.physiology.energy < 60.0:
                continue
            for receiver in alive:
                if receiver is donor:
                    continue
                if abs(donor.x - receiver.x) > 60.0:
                    continue
                if receiver.physiology.energy > 60.0:
                    continue
                if self.world.rng.random() > chance:
                    continue
                transfer = min(6.0, donor.physiology.energy - 50.0)
                if transfer <= 0.0:
                    continue
                donor.physiology.energy -= transfer
                receiver.physiology.energy = min(100.0, receiver.physiology.energy + transfer)
                donor.contributions["share_food"] = donor.contributions.get("share_food", 0.0) + transfer
                self.memory.record(
                    donor.agent_id,
                    self.world.t,
                    "social",
                    "share_food",
                    f"ok:energy-{transfer:.1f}",
                    True,
                    detail=f"to:{receiver.agent_id}",
                    salience=0.7,
                )
                self._emit(
                    "coop_output",
                    agent_id=donor.agent_id,
                    receiver_id=receiver.agent_id,
                    amount=round(transfer, 2),
                    kind="share_food",
                )
                return

    def _enforce_extinction_rules(self) -> None:
        """A closed world with zero living agents is extinct, full stop.

        ``CivilizationState`` follows the frozen contract literally (it requires
        the creation budget to be exhausted as well), but there is no spontaneous
        creation path in this world: with no living adults no birth can happen, so
        a zero population is an absorbing state regardless of the remaining
        budget.  The integration layer makes that explicit and records *why* the
        run ended (starvation vs old age) so the report can explain it.
        """
        civilization = self.civilization
        if self.registry.alive_count() > 0:
            return
        causes = [record.death_cause for record in self.registry.records() if not record.alive]
        cause = "unknown"
        if causes:
            starvation = causes.count("starvation")
            cause = "starvation" if starvation >= causes.count("old_age") else "old_age"
        if not getattr(civilization, "extinct", False):
            civilization.extinct = True
            civilization.extinct_at = self.world.t
            civilization.collapse_state = "collapsed"
            self._emit("extinction", cause=cause, t=round(self.world.t, 2))
        # the civilization layer may have flagged extinction first (budget
        # exhausted) with a placeholder cause: replace it with the real reason.
        if hasattr(civilization, "extinction_cause"):
            current = str(getattr(civilization, "extinction_cause", "") or "")
            if current in ("", "unknown"):
                civilization.extinction_cause = cause

    def _advance_empty(self, dt: float) -> None:
        self.world.t += dt
        self.world.steps += 1
        self.world._update_environment()

    # ------------------------------------------------------------------
    def _consume_facility_queue(self) -> None:
        pending = getattr(self.world.technology, "pending_facilities", None)
        if not pending:
            return
        for kind, x_screen, agent_id in list(pending):
            facility = self.civilization.build_facility(
                kind, x_screen, agent_id, self.world.t, inventory=None
            )
            if facility is not None:
                self._emit(
                    "build", facility=facility.facility_id, kind=kind, agent_id=agent_id
                )
                builder = self.world.agent_by_id(agent_id)
                if builder is not None:
                    builder.contributions["built"] = builder.contributions.get("built", 0.0) + 1.0
        pending.clear()

    def _maybe_record_episode(self, agent) -> None:
        state = agent.behavior_state
        now = self.world.t
        last = self._last_episode_at.get(agent.agent_id, -1e9)
        if state == self._last_state.get(agent.agent_id) and now - last < EPISODE_INTERVAL_S * 4:
            return
        if now - last < EPISODE_INTERVAL_S:
            return
        self._last_episode_at[agent.agent_id] = now
        self._last_state[agent.agent_id] = state
        success = state not in ("explore", "rest", "social")
        try:
            self.memory.record(
                agent.agent_id,
                now,
                "episode",
                f"state:{state}",
                f"state:{state}",
                success,
                detail="",
                salience=0.6 if success else 0.3,
            )
        except Exception:
            pass

    def _social_learning(self, dt: float) -> None:
        """Verified knowledge spreads between neighbours (teaching / imitation)."""
        alive = [agent for agent in self.world.agents if agent.alive]
        if len(alive) < 2:
            return
        chance = min(1.0, 0.35 * dt)
        for i, teacher in enumerate(alive):
            teacher_ids = list(teacher.known_knowledge_ids)
            if not teacher_ids:
                continue
            for student in alive[i + 1:]:
                if abs(teacher.x - student.x) > TEACHING_RANGE_PX:
                    continue
                new_ids = [kid for kid in teacher_ids if kid not in student.known_knowledge_ids]
                if not new_ids:
                    continue
                if self.world.rng.random() > chance:
                    continue
                student.known_knowledge_ids = sorted(
                    set(student.known_knowledge_ids) | {new_ids[0]}
                )
                self._emit(
                    "teach",
                    teacher_id=teacher.agent_id,
                    student_id=student.agent_id,
                    knowledge=new_ids[0],
                )

    # ------------------------------------------------------------------
    def _emit(self, event_kind: str, **fields) -> None:
        if self.telemetry is not None:
            try:
                self.telemetry.event(event_kind, **fields)
            except Exception:
                pass
        recorder = getattr(self.civilization, "record_event", None)
        if callable(recorder):
            try:
                recorder(event_kind, **fields)
            except Exception:
                pass

    def _emit_step(self, dt: float) -> None:
        if self.telemetry is not None:
            try:
                self.telemetry.step(self.world, self, dt)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def reset_policies(self, seed: int | None = None) -> None:
        """F2: new random neural network for every living agent (from zero)."""
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
        created = self.registry.total_created
        limit = getattr(self.config, "max_agents_created", 8)
        return (
            f"人口 {len(agents)} · 累计创建 {created}/{limit} · 死亡 {self.registry.total_deaths} · "
            f"{self.society.summary_zh()} · {self.world.technology.summary_zh()} · "
            f"{self.knowledge.summary_zh()} · 角色：{names}"
        )

    # ------------------------------------------------------------------
    def archive_summary(self) -> dict[str, object]:
        """Compact view of the historical registry (dead included)."""
        return {
            "total_created": self.registry.total_created,
            "total_deaths": self.registry.total_deaths,
            "alive": self.registry.alive_count(),
            "records": [
                {
                    "agent_id": record.agent_id,
                    "parent_id": record.parent_id,
                    "born_at": round(record.born_at, 2),
                    "died_at": None if record.died_at is None else round(record.died_at, 2),
                    "cause": record.death_cause,
                    "knowledge": list(record.known_knowledge_ids),
                }
                for record in self.registry.records()
            ],
        }
