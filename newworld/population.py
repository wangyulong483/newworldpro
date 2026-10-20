"""小型多种群：每个 agent 的脑、社会与人口学（文明重设计的集成层）。

每一步发生的事：

1. 每个存活 agent 决策（启用规划时 ``GoalManager`` 优先 → 手写
   社会教师 → 自己的神经网络策略）；
2. 物理世界带着全部动作推进一帧；
3. agent 说话 / 感知（感知可选：约占帧开销的 95%，且神经特征抽取
   不用它，因此实验会关掉）；
4. :class:`~newworld.society.Society` 让 agent 衰老、配对、生育、
   死亡，并把死者**归档进历史注册表**而不是直接丢弃；
5. 文明层让设施损耗、跟踪崩溃/灭绝、登记新建筑；
6. 知识在邻居之间传播（社会学习），记忆记录情节，设施效果生效，
   telemetry 采样世界。

8 个个体的**历史**创建预算由 :meth:`~newworld.world.World.create_agent`
强制执行；:meth:`Population.create_agent` 是唯一被认可的出生路径。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .behavior import Decision
from .goals import GoalManager
from .neural import FEATURE_DIM, HIDDEN1, V2_FEATURE_DIM, NeuralInnerLife, NeuralPolicy
from .rewards import LowLevelReward
from .perception import Perceiver
from .society import Society, SocietyNetwork
from .social import SocialLife
from .training import DEFAULT_POLICY, fine_tune, pretrain

AGENT_DIR = Path(__file__).resolve().parent.parent / "assets" / "agents"
SOCIETY_PATH = Path(__file__).resolve().parent.parent / "assets" / "society.npz"

#: 行为变化写入情景记忆的间隔（世界秒）
EPISODE_INTERVAL_S = 0.5
#: 存活 agent 之间可以互相传授已验证知识的距离
TEACHING_RANGE_PX = 170.0
#: 有限组合式创新实验室的尝试间隔（世界秒）
INNOVATION_INTERVAL_S = 5.0

#: 学习型高层目标策略的奖励权重（全部来自真实世界变化量）
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
    """一个 agent 的"大脑"：社会教师 + 神经策略 + 感知器的容器。"""

    def __init__(self, character, policy: NeuralPolicy, seed: int, memory=None) -> None:
        """初始化种群（脑/社会/目标/指标齐全）。"""
        self.character = character
        self.policy = policy
        self.will = NeuralInnerLife(policy, seed)
        self.social = SocialLife(seed)
        self.perceiver = Perceiver(seed)
        self.perception = None
        self.last_decision = None
        self.memory = memory
        #: 上一帧的决策来自哪个模块（"teacher" | "neural"）
        self.last_source = "neural"

    def decide(self, world, dt: float):
        """教师优先（生存门控过不了会返回 None），否则用神经网络。"""
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
    """种群管理器：决策链、出生/死亡、设施、合作、教学与指标的集成层。"""

    def __init__(
        self, world, seed: int = 0, *, telemetry=None, persist_society: bool = False
    ) -> None:
        """初始化种群（脑/社会/目标/指标齐全）。

        ``persist_society=False``（默认）时社交网络只在内存中演化，不读/写
        ``assets/society.npz``——否则不同实验 run 会互相污染、破坏可复现性。
        GUI 若想跨启动保留社交网络，可显式传 True。
        """
        self.world = world
        self.persist_society = bool(persist_society)
        self.seed = seed
        self.config = getattr(world, "civ_config", None)
        # 共享的文明服务都放在 world 上（单一事实来源）
        self.registry = world.registry
        self.memory = world.memory
        self.knowledge = world.knowledge
        self.civilization = world.civilization
        self.society = Society(seed, on_death=self._on_death, config=self.config)
        if self.persist_society:
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
        # P1 规划层：目标管理器（消融组 A/B 关闭）
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
        # -- 学习型高层决策层（本轮升级） ----------------------------
        #: GoalPolicy（newworld/highlevel.py）或兼容对象；None 表示
        #: 使用启发式目标优先级（旧行为）
        self.goal_policy = None
        #: (features, action_index, reward) 元组，供策略梯度训练器取用
        self.policy_transitions: list[tuple[np.ndarray, int, float]] = []
        #: 每个 agent 的待结算策略回合：特征/动作/起始快照
        self._policy_pending: dict[int, dict[str, object]] = {}
        #: 每帧决策来自哪个模块的计数（审计 / 防误归因）
        self.decision_counts: Counter[str] = Counter()
        # -- 有限组合式创新 ------------------------------------------
        self._innovation_lab = None
        self._last_innovation_t = -1e9
        # -- V2：感知缓存 / 低层 RL / 真实奖励 --------------------------
        #: LowLevelPolicy（newworld/lowlevel_rl.py）或兼容对象
        self.lowlevel_policy = None
        #: (agent_id, features, action, reward) 低层 RL 转移
        self.lowlevel_transitions: list[tuple[int, np.ndarray, object, float]] = []
        #: 真实世界变化奖励（见 newworld/rewards.py）
        self.lowlevel_reward = LowLevelReward()
        #: 本帧刚采样、等待结算奖励的低层动作
        self._pending_lowlevel: dict[int, dict[str, object]] = {}
        #: 尚未结算的累积奖励（跨非 RL 帧）
        self._pending_lowlevel_reward: dict[int, float] = {}
        #: V2.1：真实视网膜特征提取器（懒加载）与刷新计数
        self._visual_extractor = None
        self._visual_tick = 0
        #: 最近若干帧的视觉特征样本（供实验报告审计"视觉是否真的变化"）
        self._visual_samples: list[np.ndarray] = []
        # 把目标管理器挂到 world 上，neural.features() 能读计划状态
        world.goal_manager = self.goals
        self._ensure_initial_agents()
        self.sync()

    # ------------------------------------------------------------------
    def _flag(self, name: str, default: bool) -> bool:
        """读文明配置的机制开关（支持对象与 dict 两种形态）。"""
        if self.config is None:
            return default
        if hasattr(self.config, name):
            return bool(getattr(self.config, name))
        if isinstance(self.config, dict):
            return bool(self.config.get(name, default))
        return default

    # ------------------------------------------------------------------
    def _ensure_initial_agents(self) -> None:
        """保证世界有开局的一对成年居民（遵守创建预算）。"""
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
            # 已有一个（世界构造时自动创建的）：补登记 + 配一个伴侣
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
            # 从干净的材料经济开局；知识/记忆持久保留
            agent.inventory.wood = 0
            agent.inventory.stone = 0
            agent.created_at = agent.created_at or 0.0
            world.adopt_agent(agent)

    # ------------------------------------------------------------------
    def policy_path(self, agent_id: int) -> Path:
        """该 agent 的神经网络 checkpoint 路径。"""
        return AGENT_DIR / f"agent_{agent_id}.npz"

    def _load_policy(self, agent) -> NeuralPolicy:
        """载入该 agent 的策略；没有就用默认/全新网络。"""
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
        """孩子从父母的网络权重"交叉 + 微变异"继承策略。"""
        parents = [self.minds[pid] for pid in agent.parents if pid in self.minds]
        if parents:
            father = parents[0].policy.flat_weights()
            mother = parents[-1].policy.flat_weights()
            rng = np.random.default_rng(self.seed + agent.agent_id)
            mask = rng.random(father.shape) < 0.5
            child = np.where(mask, father, mother)
            noise = rng.normal(0.0, 0.03, child.shape)
            # 视觉输入列（w1 的尾 VISUAL_DIM 行）不做随机变异：否则未训练的
            # 随机投影会被误读成"视觉能力"。它们只随 BC/RL 学习信号更新。
            noise[V2_FEATURE_DIM * HIDDEN1 : FEATURE_DIM * HIDDEN1] = 0.0
            child = child + noise
            return NeuralPolicy(weights=child, seed=self.seed + agent.agent_id)
        return NeuralPolicy(seed=self.seed + agent.agent_id)

    # ------------------------------------------------------------------
    def sync(self) -> None:
        """给新生儿建脑；删掉死者的脑。

        丢弃*脑对象*是刻意的（死去的 agent 绝不能再行动）——一切有
        文明价值的东西（库存、已知知识、技能、记忆条目、身份）在
        此之前已经由 :meth:`_on_death` 归档进历史注册表。
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
    # 创建 / 死亡（唯一被认可的路径）
    # ------------------------------------------------------------------
    def create_agent(self, *, parent_id: int | None = None, cause: str = "birth", **kwargs):
        """历史预算允许时创建新 agent。

        返回新的 :class:`~newworld.character.Character`；8 个累计名额
        已用完时返回 ``None``。继承（记忆、知识）由 :meth:`note_birth`
        与 ``Society`` 处理。
        """
        child = self.world.create_agent(parent_id=parent_id, cause=cause, **kwargs)
        if child is None:
            return None
        self.sync()
        return child

    def note_birth(self, child, *, mother=None, father_id: int | None = None) -> None:
        """P3：向新生儿转移程序记忆与已验证知识。"""
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
            # 父母可能已死：从注册表档案里取知识，否则从活人身上取
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
        """在死者离开 ``world.agents`` *之前*归档。

        历史记录永远保留（任务约束：死者的历史与知识记录不可删除）。
        只有记忆/知识向他人的*转移*是机制开关。
        """
        # 用死亡惩罚结算待定的策略回合（诚实的信号）
        pending = self._policy_pending.pop(agent.agent_id, None)
        if pending is not None:
            reward = self._goal_reward(dict(pending["snapshot"]), failed=False, died=True)  # type: ignore[arg-type]
            self.policy_transitions.append(
                (np.asarray(pending["features"], dtype=np.float64), int(pending["action"]), reward)
            )
        # 低层 RL：死亡时用固定负奖励结算待定动作（诚实信号）
        pending_lowlevel = self._pending_lowlevel.pop(agent.agent_id, None)
        pending_reward = self._pending_lowlevel_reward.pop(agent.agent_id, 0.0)
        if pending_lowlevel is not None:
            self.lowlevel_transitions.append(
                (
                    agent.agent_id,
                    np.asarray(pending_lowlevel["features"], dtype=np.float64),
                    pending_lowlevel["action"],
                    float(pending_reward + self.lowlevel_reward.death_bonus().total),
                )
            )
        self.lowlevel_reward.forget(agent.agent_id)
        final_state = {
            "x": round(float(agent.x), 2),
            "energy": round(float(agent.physiology.energy), 2),
            "hydration": round(float(agent.physiology.hydration), 2),
            "behavior_state": agent.behavior_state,
            "age": round(float(agent.age), 2),
            # AgentRegistry.mark_dead 认识的确切字段名；
            # 只放数值计数（工具名进 final_state）
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
        # 终身贡献记录在死后也保留（最后一个体死去时，
        # 文明级的合作指数不能清零）
        record = self.registry.record(agent.agent_id)
        if record is not None:
            record.contributions = {str(key): float(value) for key, value in agent.contributions.items()}
        self._emit("death", agent_id=agent.agent_id, cause=cause, t=round(t, 2))

    # ------------------------------------------------------------------
    # 每帧决策
    # ------------------------------------------------------------------
    def _infrastructure_directive(self, agent) -> Decision | None:
        """个体状态低迷时使用公共设施（基础设施反射）。

        浆果只在白天再生，因此从不定居的群体无论技术多好，夜里都
        会挨饿。这个反射让 agent 走向一个活跃的食物/取暖建筑
        （粮仓/篝火/小屋）并在那里休息——这是建成的基建真正改变
        生存的机制，也是村庄围绕粮仓形成而不是继续散居的原因。
        只有已经存在的建筑（即通过已验证技术建出来的）能吸引人，
        因此它伪造不了进度。
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
        """按配置的底层控制器产出一个原语动作。

        ``full`` = 行为克隆网络；``random`` = 随机游走；``off`` = 原地
        站立（决策链消融）。对*所有*被委托的帧同样生效，否则规划器
        的探索委托会绕过消融。
        """
        mode = str(getattr(self.config, "low_level_mode", "full") if self.config else "full")
        if mode == "rl" and self.lowlevel_policy is not None:
            return self._rl_low_level_decision(agent, mind, dt)
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

    # ------------------------------------------------------------------
    # V2：低层动作策略（REINFORCE）与真实奖励集成
    # ------------------------------------------------------------------
    def attach_lowlevel_policy(self, policy) -> None:
        """挂载低层动作策略（幂等）；训练器拿到的策略就是驱动仿真的那个。"""
        self.lowlevel_policy = policy
        self._pending_lowlevel.clear()

    def detach_lowlevel_policy(self) -> None:
        """卸载低层动作策略（回退到 BC / 其它 low_level_mode）。"""
        self.lowlevel_policy = None
        self._pending_lowlevel.clear()

    def drain_lowlevel_transitions(self) -> list[tuple[int, np.ndarray, object, float]]:
        """取走全部低层 RL 转移：``(agent_id, features, action, reward)``。"""
        out = self.lowlevel_transitions
        self.lowlevel_transitions = []
        return out

    def _rl_low_level_decision(self, agent, mind: AgentMind, dt: float):
        """用 LowLevelPolicy 采样原语动作，并登记待结算的 RL 转移。"""
        from .neural import features as neural_features

        feats = np.asarray(neural_features(self.world, character=agent), dtype=np.float64)
        sample = self.lowlevel_policy.sample(feats)
        try:
            from .lowlevel_rl import MOVE_VALUES
        except Exception:  # pragma: no cover - 防御：模块缺失时与 NeuralPolicy.act 一致
            MOVE_VALUES = (1, 0, -1)
        move_index = int(getattr(sample, "move", 1))
        move = MOVE_VALUES[move_index] if 0 <= move_index < len(MOVE_VALUES) else 0
        decision = Decision(
            int(move),
            bool(getattr(sample, "jump", False)),
            bool(getattr(sample, "speak", False)),
            None,
        )
        self._pending_lowlevel[agent.agent_id] = {"features": feats, "action": sample}
        self.decision_counts["lowlevel_rl"] += 1
        mind.last_decision = decision
        return decision

    def _finalize_lowlevel_rewards(self, dt: float) -> None:
        """用真实世界变化结算奖励；把奖励归属到最近一次低层动作。"""
        if str(getattr(self.config, "low_level_mode", "full") if self.config else "full") != "rl":
            return
        for agent in [a for a in self.world.agents if a.alive]:
            breakdown = self.lowlevel_reward.observe(self.world, agent, dt)
            pending = self._pending_lowlevel.pop(agent.agent_id, None)
            if pending is not None:
                reward = self._pending_lowlevel_reward.pop(agent.agent_id, 0.0) + breakdown.total
                self.lowlevel_transitions.append(
                    (
                        agent.agent_id,
                        np.asarray(pending["features"], dtype=np.float64),
                        pending["action"],
                        float(reward),
                    )
                )
            else:
                self._pending_lowlevel_reward[agent.agent_id] = (
                    self._pending_lowlevel_reward.get(agent.agent_id, 0.0) + breakdown.total
                )

    def lowlevel_summary(self) -> dict[str, object]:
        """低层 RL 的可审计摘要（实验报告用）。"""
        count = len(self.lowlevel_transitions)
        mean_reward = (
            sum(float(item[3]) for item in self.lowlevel_transitions) / count if count else 0.0
        )
        return {
            "mode": str(getattr(self.config, "low_level_mode", "full") if self.config else "full"),
            "transitions": float(count),
            "mean_reward": round(mean_reward, 4),
            "reward_components": self.lowlevel_reward.summary(),
        }

    # ------------------------------------------------------------------
    # V2.1：真实视网膜特征刷新（感知真正进入观测）
    # ------------------------------------------------------------------
    def _ensure_visual_extractor(self):
        if self._visual_extractor is not None:
            return self._visual_extractor
        try:
            from .visual_features import VisualFeatureExtractor

            self._visual_extractor = VisualFeatureExtractor(seed=self.seed)
        except Exception:
            self._visual_extractor = False
            return None
        return self._visual_extractor

    def visual_summary(self) -> dict[str, object]:
        """视觉特征的真实使用摘要：维度/各维标准差/有变化维数。"""
        count = len(self._visual_samples)
        if count == 0:
            return {"samples": 0, "dims": 0, "active_dims": 0, "std": []}
        matrix = np.vstack(self._visual_samples)
        std = matrix.std(axis=0)
        return {
            "samples": count,
            "dims": int(matrix.shape[1]),
            "active_dims": int((std > 1e-4).sum()),
            "std": [round(float(value), 5) for value in std],
            "mean": [round(float(value), 5) for value in matrix.mean(axis=0)],
        }

    def _refresh_visual_features(self, active, dt: float) -> None:
        """每 perception_stride 帧刷新一次 16 维视网膜特征（写入 agent）。"""
        if not self._flag("enable_visual_features", True):
            return
        extractor = self._ensure_visual_extractor()
        if extractor is None:
            return
        for agent, _mind in active:
            try:
                state = extractor.observe(self.world, agent, dt)
                agent.visual_features = tuple(float(value) for value in state.vector)
                if len(self._visual_samples) < 512:
                    self._visual_samples.append(np.asarray(state.vector, dtype=np.float64))
            except Exception:
                continue

    def _decide(self, agent, mind: AgentMind, dt: float):
        """一个 agent 一帧的完整决策链（优先级从高到低）。

        基础设施反射 → 学习型高层策略要目标 → 目标规划器 →（规划
        委托时）底层控制器 → 消融模式 → 社会教师/神经网络。
        """
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
                # 语义层选了"探索/空闲"并把原语动作委托下来：
                # 直接到底层控制器。不要让求偶教师劫持这些帧——
                # 那会产生原地不动的死循环（实测 repeat_run_max 9476）。
                return self._low_level_decision(agent, mind, dt)
        mode = str(getattr(self.config, "low_level_mode", "full") if self.config else "full")
        if mode != "full":
            return self._low_level_decision(agent, mind, dt)
        decision = mind.decide(self.world, dt)
        self.decision_counts[str(mind.last_source)] += 1
        return decision

    # ------------------------------------------------------------------
    # 学习型高层目标策略集成（真实 REINFORCE 回路）
    # ------------------------------------------------------------------
    def attach_goal_policy(self, policy) -> None:
        """挂载学习到的目标选择策略（幂等，替换旧策略）。

        ``policy`` 只需要 ``sample(features) -> action_index``；
        ``newworld/highlevel.py`` 的训练器在它拿到的 env 上调用本方法，
        因此被训练的策略正是驱动仿真的那个（不会悄悄回退到别的
        策略）。
        """
        self.goal_policy = policy
        self._policy_pending.clear()

    def detach_goal_policy(self) -> None:
        """卸载学习型策略，回到启发式目标优先级。"""
        self.goal_policy = None
        self._policy_pending.clear()

    def drain_policy_transitions(self) -> list[tuple[np.ndarray, int, float]]:
        """弹出全部已结算的 (features, action_index, reward) 转移。"""
        out = self.policy_transitions
        self.policy_transitions = []
        return out

    def _snapshot_metrics(self) -> dict[str, float]:
        """策略奖励用的世界指标快照（技术/设施/资源/知识/合作）。"""
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
        """一个策略回合的奖励，全部由**真实世界变化**计算。

        公式（见 docs/EXPERIMENT_REPORT.md）：存活时间、新验证技术、
        活跃设施、资源量、已知知识与合作转移都给正奖励；目标失败
        与死亡给负奖励。只用目标发起时刻与结束时刻之间的世界状态
        差值——绝不用 BC 标签。
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
        """agent 没有活跃目标时，向学习型策略请求一个目标。"""
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
            # 不可行动作：小额负奖励、不建回合（诚实的信号）
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
        """结算目标已结束（完成/失败/死亡）的策略回合。"""
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
        """各模块做出的每帧决策占比（审计指标）。"""
        total = float(sum(self.decision_counts.values()))
        if total <= 0.0:
            return {}
        return {key: round(value / total, 4) for key, value in self.decision_counts.items()}

    def step(self, dt: float) -> None:
        """推进一整帧：决策 → 世界 → 感知 → 社会 → 文明 → 设施 →
        学习/合作/创新 → 策略结算 → 灭绝规则 → telemetry。"""
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
        stride = max(
            1, int(getattr(self.config, "perception_stride", 1) if self.config else 1)
        )
        self._visual_tick += 1
        refresh = (self._visual_tick % stride == 0) or (self._visual_tick == 1)
        for agent, mind in active:
            decision = mind.last_decision
            if decision is not None and decision.speak:
                self.world.speak(agent, decision.word)
            if perceive and refresh:
                mind.perception = mind.perceiver.perceive(
                    self.world, dt * stride, character=agent
                )
        if refresh:
            self._refresh_visual_features(active, dt * stride)
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
        self._finalize_lowlevel_rewards(dt)
        self._enforce_extinction_rules()
        self._emit_step(dt)

    # ------------------------------------------------------------------
    # 有限组合式创新集成
    # ------------------------------------------------------------------
    def _maybe_innovate(self, dt: float) -> None:
        """每个间隔尝试一次有界的组合式发明（真实材料成本）。

        只在 ``enable_innovation`` 开启时运行；提案来自
        :class:`newworld.innovation.InnovationLab`，它会用 agent 的
        **已验证**知识校验组件，并通过 ``Technology.register_custom_effect``
        登记效果。成功的发明会改变真实数值（采集量/负重/恢复）。
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
        # 生命周期保护：World.reset() 会重建 Technology（自定义效果
        # 丢失），而实验室保留着发明——重新登记，保证记录与真实
        # 世界效果永不脱节。
        try:
            for tool_name, effects in lab.custom_effects().items():
                if tool_name not in self.world.technology.custom_effects:
                    self.world.technology.register_custom_effect(tool_name, dict(effects))
        except Exception:
            pass
        for agent in [a for a in self.world.agents if a.alive]:
            try:
                # 只尝试个体真的付得起的组合：付不起是前置缺失，
                # 不是实验
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
                # 负重是工具的属性：从技术效果表读，别处登记的工具
                # 也能生效
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
                # 这些是前置缺失/重复发明等"正常拒绝"，不算新闻
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
        """已发明列表（字典形式），无实验室时为空。"""
        lab = self._innovation_lab
        if lab is None:
            return []
        try:
            return [item.to_dict() for item in lab.innovations()]
        except Exception:
            return []

    def _cooperation(self, dt: float) -> None:
        """P4：邻居间分享食物（真实、可度量的社会规则）。

        吃饱的个体站在挨饿个体旁边时，会让出部分能量（最多 10 点）。
        这不是装饰：它让采集者熬过糟糕的夜晚，并计入
        ``cooperation_index`` / ``cooperation_benefit``——因为转移是
        任何单个个体都无法独立产出的真实资源流。由 ``enable_social``
        门控，消融组 C（无社会机制）用不了。
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
        """存活为 0 的封闭世界就是灭绝，没有例外。

        ``CivilizationState`` 按冻结契约的字面意思执行（它还要求创建
        预算也耗尽），但这个世界没有自发创建路径：没有活着的成年人
        就没有出生，因此零人口无论剩多少预算都是吸收态。集成层把
        这一点挑明，并记录运行*为什么*结束（饿死 vs 老死），报告才
        能解释。
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
        # 文明层可能先一步（预算耗尽）用占位原因标记了灭绝：
        # 换成真实原因。
        if hasattr(civilization, "extinction_cause"):
            current = str(getattr(civilization, "extinction_cause", "") or "")
            if current in ("", "unknown"):
                civilization.extinction_cause = cause

    def _advance_empty(self, dt: float) -> None:
        """无存活个体时只推进时钟与环境。"""
        self.world.t += dt
        self.world.steps += 1
        self.world._update_environment()

    # ------------------------------------------------------------------
    def _consume_facility_queue(self) -> None:
        """消费 Technology 排队的设施：登记为文明层的公共建筑。"""
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
        """按间隔把行为状态变化写入情景记忆（episode 条目）。"""
        state = agent.behavior_state
        now = self.world.t
        last = self._last_episode_at.get(agent.agent_id, -1e9)
        if state == self._last_state.get(agent.agent_id) and now - last < EPISODE_INTERVAL_S * 4:
            return
        if now - last < EPISODE_INTERVAL_S:
            return
        self._last_episode_at[agent.agent_id] = now
        self._last_state[agent.agent_id] = state
        # 探索/休息/社交算"没做成事"，其余状态算成功经历
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
        """已验证知识在邻居间传播（教学 / 模仿）。"""
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
                # 每次只教一条（真实的教学节奏）
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
        """向 telemetry 与文明层各发一条事件（吞掉所有异常）。"""
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
        """让 telemetry 采样这一帧。"""
        if self.telemetry is not None:
            try:
                self.telemetry.step(self.world, self, dt)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def reset_policies(self, seed: int | None = None) -> None:
        """F2：给每个存活 agent 换一个从零开始的全新随机网络。"""
        seed = self.seed if seed is None else seed
        self.society.network = SocietyNetwork(seed)
        if self.persist_society:
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
        """训练每个 agent 自己的网络（``reset`` 时从零开始）。"""
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
        if self.persist_society:
            self.society.network.save(SOCIETY_PATH)
        return {"agents": total, "mode_accuracy": accuracy / max(1.0, total)}

    # ------------------------------------------------------------------
    def summary_zh(self) -> str:
        """人口摘要（中文，UI 调试层展示用）。"""
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
        """历史注册表的紧凑视图（含死者）。"""
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
