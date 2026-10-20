"""世界：几何、居民、物理（力学/声学/热学）。

``World`` 是纯仿真核心。它对 tkinter、PIL、渲染一无所知，因此
导入便宜、易测试、也能安全地跑在 headless 训练循环里。

除了居民，它还携带物理环境：

* **重力** —— 每个居民以 ``g = 9.80665 m/s^2`` 下落
* **太阳** —— 一个提供光与辐射热的黑体光源
* **空气** —— 20 C 的介质，传声并施加阻力
* **声场** —— 压力波（说话，以及未来的环境声）

渲染在 :mod:`newworld.render`；强化学习封装在 :mod:`newworld.env`。
"""

from __future__ import annotations

import random

from . import physics
from .acoustics import SpeechEvent, SoundField, synthesize_speech
from .character import Character, mouth_position
from .civilization import AgentRegistry, CivilizationState, PopulationLimitError
from .config import CivilizationConfig, WorldConfig
from .daynight import DayCycle
from .environment import Air, Sun, step_body_temperature
from .knowledge import KnowledgeStore
from .memory import MemoryStore
from .optics import Scene, Shape
from .physiology import Physiology
from .materials import ATOM_SIZE
from .pixelworld import PixelScene
from .technology import Technology


class NoAgentsError(RuntimeError):
    """代码访问 ``world.agent`` 但没有存活居民时抛出。

    原始实现会在属性访问时悄悄生成一个替补居民，这会让文明在灭绝后
    "复活"自己、绕过 8 个创建预算。现在创建必须显式进行
    （:meth:`World.create_agent`）。
    """


class World:
    """一个屏幕大小、由 :class:`Character` 居住的物理世界。

    Parameters
    ----------
    config:
        几何 / 物理 / 调色配置。缺省为按当前屏幕分辨率生成的
        :class:`WorldConfig`。
    seed:
        内部 :class:`random.Random` 的种子，用于摆放居民与编造话语。
    civ_config:
        文明层开关与**硬性**创建预算
        （:class:`~newworld.config.CivilizationConfig`）。
        缺省为全部机制开启、``max_agents_created = 8``。
    """

    def __init__(
        self,
        config: WorldConfig | None = None,
        seed: int | None = None,
        *,
        civ_config: CivilizationConfig | None = None,
    ) -> None:
        """组装世界各服务层并创建第一个居民。"""
        self.config = config if config is not None else WorldConfig()
        self.rng = random.Random(seed)
        self.seed_value = seed
        self.civ_config = civ_config if civ_config is not None else CivilizationConfig(seed=seed or 0)
        self.agents: list[Character] = []
        #: 手动添加的可见 2-D 物体（天然物体在 pixel_scene 里）
        self.objects: list[Shape] = []
        #: V2 无限世界：窗口左边缘的全局像素偏移（有限世界恒为 0）
        self.window_origin_px = 0.0
        #: InfiniteScene 不可用时的降级原因（空字符串 = 正常）
        self.infinite_fallback_reason = ""
        #: 像素原子世界：地形、岩石、树、浆果丛、池塘（无限模式为窗口化场景）
        self.pixel_scene = self._make_scene(self.seed_value or 0)
        #: 社区/工具层（采集、制作、建造）
        self.technology = Technology(self.config, self.pixel_scene, seed=self.seed_value or 0)
        #: 历史注册表：创建过的每个 agent（死亡的会被归档）
        self.registry = AgentRegistry(self.civ_config.max_agents_created)
        #: 个体情景/程序记忆（死后保留）
        self.memory = MemoryStore()
        #: 持久的、可验证的知识（配方、状态、验证历史）
        self.knowledge = KnowledgeStore(
            seed=self.seed_value or 0,
            bootstrap=bool(self.civ_config.enable_knowledge),
        )
        #: 文明级状态：设施、契约、崩溃、灭绝
        self.civilization = CivilizationState(
            self.registry,
            knowledge=self.knowledge,
            memory=self.memory,
            config=self.civ_config,
        )
        #: 神经观测是否扩到 40 维输入（P3）
        self.expanded_observation = bool(self.civ_config.expanded_observation)
        #: V2.1：真实视网膜特征是否进入观测（16 维，FEATURE_DIM 40 -> 56）
        self.enable_visual_features = bool(
            getattr(self.civ_config, "enable_visual_features", True)
        )
        #: 由 Population 设置：目标管理器，features() 读计划状态用
        self.goal_manager = None
        #: 虚拟时钟与太阳几何
        self.day = DayCycle()
        #: 物理环境
        self.sun = Sun()
        self.air = Air()
        self.sound = SoundField()
        self.hour: float = self.day.start_hour
        self.daylight: float = self.day.daylight(self.hour)
        self.t: float = 0.0
        self.steps: int = 0
        self._next_id = 0
        #: 被拒绝的创建尝试（预算用尽），供 telemetry 用
        self.creation_denied: list[dict[str, object]] = []
        #: 有预算检查地创建第一个居民
        self.create_agent()

    # ------------------------------------------------------------------
    # 场景 / 光场
    # ------------------------------------------------------------------
    def scene(self) -> Scene:
        """居民眼睛看到的可见世界（像素物体 + 额外附加的物体）。"""
        base = self.pixel_scene.optics_scene()
        return Scene(objects=tuple(base.objects) + tuple(self.objects))

    def _update_environment(self) -> None:
        """推进太阳钟与气温。"""
        self.hour = self.day.hour(self.t)
        self.daylight = self.day.daylight(self.hour)
        self.sun.elevation_deg = self.day.sun_elevation_deg(self.hour)
        # 简单的昼夜气温：夜晚 10 C，正午 24 C
        self.air.temperature_k = 283.15 + 14.0 * self.daylight

    def add_object(self, obj: Shape) -> Shape:
        """向世界添加一个可见物体（预留给未来的实验）。"""
        self.objects.append(obj)
        return obj

    def clear_objects(self) -> None:
        """清空手动添加的可见物体。"""
        self.objects.clear()

    # ------------------------------------------------------------------
    # 居民
    # ------------------------------------------------------------------
    @property
    def agent(self) -> Character:
        """第一个居民。

        没有存活居民时抛出 :class:`NoAgentsError`：世界从不隐式生成
        居民，否则灭绝的文明能"复活"自己、绕过创建预算。
        """
        if not self.agents:
            raise NoAgentsError(
                "no inhabitants: creation must go through World.create_agent()"
            )
        return self.agents[0]

    @property
    def n_agents(self) -> int:
        """当前居民总数（含濒死者）。"""
        return len(self.agents)

    @property
    def alive_agents(self) -> list[Character]:
        """存活居民列表。"""
        return [character for character in self.agents if character.alive]

    def spawn(
        self,
        x: float | None = None,
        facing: int = 1,
        agent_id: int | None = None,
        *,
        name: str = "",
        sex: str = "male",
        age: float = 0.0,
        generation: int = 0,
        body_scale: float = 1.0,
        parents: tuple[int, ...] = (),
    ) -> Character:
        """底层*几何*生成（测试、预览、工具用）。

        **这不算一次创建事件**：它绕过注册表与 8 个预算。仿真内部
        一律改用 :meth:`create_agent`。
        """
        cfg = self.config
        if x is None:
            x = self.rng.uniform(cfg.min_x, cfg.max_x)
        character = Character(
            x=float(x),
            facing=facing,
            agent_id=self._next_id if agent_id is None else agent_id,
            name=name,
            sex=sex,
            age=age,
            generation=generation,
            body_scale=body_scale,
            parents=parents,
            parent_id=(parents[0] if parents else None),
        )
        character.attach(cfg)
        self._next_id = max(self._next_id, character.agent_id) + 1
        self.agents.append(character)
        return character

    def create_agent(
        self,
        x: float | None = None,
        facing: int = 1,
        *,
        name: str = "",
        sex: str = "male",
        age: float = 0.0,
        generation: int = 0,
        body_scale: float = 1.0,
        parents: tuple[int, ...] = (),
        parent_id: int | None = None,
        cause: str = "birth",
    ) -> Character | None:
        """仿真权威的创建路径：遵守**累计**预算。

        历史创建预算（``civ_config.max_agents_created``）用尽时返回
        ``None``；拒绝会被记录到 :attr:`civilization`，telemetry 可以
        上报。这是仿真过程中添加居民的*唯一*合法方式。
        """
        if not self.registry.can_create():
            self.creation_denied.append(
                {"t": round(self.t, 3), "cause": cause, "total_created": self.registry.total_created}
            )
            return None
        character = self.spawn(
            x=x,
            facing=facing,
            name=name,
            sex=sex,
            age=age,
            generation=generation,
            body_scale=body_scale,
            parents=parents,
        )
        if parent_id is None and parents:
            parent_id = parents[0]
        character.created_at = self.t
        character.parent_id = parent_id
        self.registry.register(character, t=self.t, parent_id=parent_id)
        return character

    def adopt_agent(self, character: Character, *, t: float | None = None) -> None:
        """补登记一个已存在的（几何方式生成的）居民。

        :class:`~newworld.population.Population` 接管一个在注册表出现
        之前建好的世界时使用（测试、预览场景）。
        """
        if self.registry.record(character.agent_id) is not None:
            return
        character.created_at = self.t if t is None else t
        self.registry.register(character, t=character.created_at, parent_id=character.parent_id)

    def remove_agent(self, agent_id: int) -> None:
        """把居民从*存活*列表移除（死亡/离开）。

        历史记录**不会**被删除：请先调用
        :meth:`~newworld.civilization.AgentRegistry.mark_dead`
        （或用 :meth:`Population.report_death`），agent 才能留在档案里。
        """
        self.agents = [character for character in self.agents if character.agent_id != agent_id]

    def agent_by_id(self, agent_id: int) -> Character | None:
        """按 id 查存活居民，没有返回 None。"""
        for character in self.agents:
            if character.agent_id == agent_id:
                return character
        return None

    # ------------------------------------------------------------------
    # V2：无限世界（窗口化场景 + 自动跟随平移）
    # ------------------------------------------------------------------
    def _make_scene(self, seed: int):
        """按 ``config.infinite`` 构建有限 PixelScene 或无限窗口场景。"""
        if not bool(getattr(self.config, "infinite", False)):
            return PixelScene(self.config, seed)
        try:
            from .infinite_scene import InfiniteScene

            window = int(getattr(self.config, "window_atoms", 0) or 0)
            if window <= 0:
                # 默认窗口 = 屏幕宽度（1:1 渲染，不横向压缩）；实验可显式放大
                window = max(16, int(getattr(self.config, "width", 640)) // ATOM_SIZE)
            return InfiniteScene(self.config, seed, window_atoms=window)
        except Exception as exc:  # 防御：模块缺失/构造失败时退回有限世界
            self.infinite_fallback_reason = f"{type(exc).__name__}: {exc}"
            return PixelScene(self.config, seed)

    def _shift_world_x(self, shift_px: float) -> None:
        """把全部动态 x 坐标平移，使局部坐标窗口保持一致。"""
        for character in self.agents:
            character.x = float(character.x) + shift_px
        community = getattr(getattr(self, "technology", None), "community", None)
        if community is not None:
            community.x_screen = float(community.x_screen) + shift_px
        for facility in list(getattr(self.civilization, "facilities", ()) or ()):
            facility.x_screen = float(facility.x_screen) + shift_px

    def _maybe_recenter(self) -> None:
        """无限世界：人口接近窗口边缘时平移窗口（世界无界，屏幕只是窗口）。"""
        scene = self.pixel_scene
        if not bool(getattr(self.config, "infinite", False)):
            return
        if not hasattr(scene, "recenter"):
            return
        alive = [agent for agent in self.agents if agent.alive]
        if not alive:
            return
        cfg = self.config
        margin = cfg.width * 0.25
        positions = [float(agent.x) for agent in alive]
        if min(positions) > margin and max(positions) < cfg.width - margin:
            return
        center = sum(positions) / len(positions)
        step_px = cfg.width * 0.25
        direction = 1.0 if center > cfg.width / 2.0 else -1.0
        delta_px = step_px * direction
        delta_atoms = int(round(delta_px / ATOM_SIZE))
        if delta_atoms == 0:
            return
        try:
            scene.recenter(delta_atoms)
        except Exception as exc:  # 不因单次平移失败而中断仿真
            self.infinite_fallback_reason = f"recenter failed: {type(exc).__name__}: {exc}"
            return
        self._shift_world_x(-delta_px)
        self.window_origin_px = float(
            getattr(scene, "window_start_px", self.window_origin_px + delta_px)
        )

    def reset(
        self,
        seed: int | None = None,
        positions: list[float] | None = None,
    ) -> list[tuple[float, ...]]:
        """把世界重置到 ``t = 0`` 并返回初始观测。

        ``positions`` 可选地固定每个居民的 x 位置（长度必须与居民数
        一致）。缺省时每个居民被随机放在一个合法位置。

        .. note::
           这是**开发/测试辅助**（重摆位置、重播种地形），不是文明
           重置：历史注册表、知识与记忆有意保留。全新实验应当新建
           一个 :class:`World`。
        """
        if seed is not None:
            self.seed_value = seed
            self.rng.seed(seed)
        if positions is not None and len(positions) != len(self.agents):
            raise ValueError(
                f"positions length {len(positions)} != agent count {len(self.agents)}"
            )
        self.t = 0.0
        self.steps = 0
        self.sound = SoundField()
        if seed is not None:
            self.pixel_scene = self._make_scene(seed)
            self.window_origin_px = float(
                getattr(self.pixel_scene, "window_start_px", 0.0)
            )
            self.technology = Technology(self.config, self.pixel_scene, seed=seed)
        self.technology.pending_facilities.clear()
        self._update_environment()
        for index, character in enumerate(self.agents):
            character.attach(self.config)  # config 可能被调整过尺寸
            character.x = (
                float(positions[index])
                if positions is not None
                else self.rng.uniform(self.config.min_x, self.config.max_x)
            )
            character.x = min(max(character.x, self.config.min_x), self.config.max_x)
            character.y = self.config.ground_y
            character.vx = 0.0
            character.vy = 0.0
            character.on_ground = True
            character.phase = 0.0
            character.distance_walked = 0.0
            character.steps = 0
            character.body_temperature_k = physics.HUMAN_BODY_TEMPERATURE_K
            character.utterance = None
            character.utterance_until = 0.0
            character.jump_count = 0
            character.speech_count = 0
            character.behavior_state = "explore"
            character.physiology = Physiology()
        return self.observations()

    # ------------------------------------------------------------------
    # 仿真
    # ------------------------------------------------------------------
    @staticmethod
    def _split_action(action) -> tuple[int, bool]:
        """把 int 或 Decision 拆成 (move, jump)。"""
        move = getattr(action, "move", action)
        jump = bool(getattr(action, "jump", False))
        return int(move), jump

    def step(
        self,
        dt: float | None = None,
        actions: int | list | None = None,
    ) -> list[tuple[float, ...]]:
        """把世界推进 ``dt`` 秒。

        ``actions`` 可以是单个动作（作用于所有居民），也可以是每个
        居民一个动作。动作要么是整数 ``-1/0/+1``，要么是带
        ``.move``/``.jump`` 属性的对象
        （:class:`newworld.autonomy.Decision`）。返回所有居民的归一化
        观测。
        """
        dt = self.config.fixed_dt if dt is None else float(dt)
        count = len(self.agents)
        if count == 0:
            self.t += dt
            self.steps += 1
            self.sound.advance(dt)
            return []
        if actions is None:
            action_list = [0] * count
        elif isinstance(actions, int):
            action_list = [actions] * count
        elif not isinstance(actions, (list, tuple)):
            action_list = [actions] * count
        else:
            action_list = list(actions)
            if len(action_list) == 1 and count > 1:
                action_list = action_list * count
            elif len(action_list) != count:
                raise ValueError(
                    f"expected {count} actions, got {len(action_list)}"
                )
        self.t += dt
        self._update_environment()
        for character, action in zip(self.agents, action_list):
            move, jump = self._split_action(action)
            character.update(dt, move, self.config, jump=jump)
            # 逐居民推进物理热平衡与生理
            step_body_temperature(
                character,
                self.air,
                self.sun,
                dt,
                activity=1.2 if move != 0 else 1.0,
            )
            character.physiology.update(
                dt,
                hour=self.hour,
                daylight=self.daylight,
                moving=move != 0,
                jumped=jump,
                body_temperature_c=character.body_temperature_c,
            )
        # 生态：土壤水循环、生长与原子级破坏
        self.pixel_scene.update_ecology(dt, self.day, self.hour)
        if abs(self.agent.vx) > 40.0 and self.rng.random() < dt * 0.45:
            # 跑得快时偶尔踩坏地面
            feet = self.agent.y if self.agent.y is not None else self.config.ground_y
            self.pixel_scene.trample(self.agent.x, feet, force=1.4)
        self.steps += 1
        self.sound.advance(dt)
        self._maybe_recenter()
        return self.observations()

    # ------------------------------------------------------------------
    # 声学
    # ------------------------------------------------------------------
    def speak(self, character: Character, text: str | None = None) -> SpeechEvent:
        """让居民说点什么（默认：随机编一个"词"）。"""
        from .acoustics import random_word

        phrase = text if text is not None else random_word(self.rng)
        origin_x, origin_y = mouth_position(character, self.config)
        event = synthesize_speech(
            origin_x,
            origin_y,
            self.t,
            phrase,
            self.rng,
            speaker_id=character.agent_id,
        )
        for wave in event.waves:
            self.sound.add(wave)
        character.utterance = phrase
        character.utterance_until = self.t + event.duration_s + 0.3
        character.speech_count += 1
        return event

    def sound_pressure_at(self, x: float, y: float) -> float:
        """世界某点当前的瞬时声压（Pa）。"""
        return self.sound.pressure_at(x, y, self.t)

    def sound_rms_at(self, x: float, y: float) -> float:
        """世界某点当前的 RMS 声压（Pa，听觉/分贝用）。"""
        return self.sound.rms_pressure_at(x, y, self.t)

    # ------------------------------------------------------------------
    # 环境快照
    # ------------------------------------------------------------------
    def environment_state(self) -> dict[str, float]:
        """环境物理量快照（重力/时刻/光照/气温/声速等）。"""
        return {
            "gravity_m_s2": physics.STANDARD_GRAVITY,
            "hour": self.hour,
            "daylight": self.daylight,
            "sun_elevation_deg": self.sun.elevation_deg,
            "solar_irradiance_w_m2": self.sun.horizontal_irradiance,
            "sun_temperature_k": self.sun.temperature_k,
            "visible_irradiance_w_m2": self.sun.visible_irradiance,
            "air_temperature_c": self.air.temperature_c,
            "air_density_kg_m3": self.air.density_kg_m3,
            "speed_of_sound_m_s": self.air.speed_of_sound_m_s,
        }

    def observations(self) -> list[tuple[float, ...]]:
        """所有居民的归一化观测（8 维）。"""
        return [character.observation(self.config) for character in self.agents]

    # ------------------------------------------------------------------
    # 自省
    # ------------------------------------------------------------------
    def state(self) -> dict[str, object]:
        """可序列化的世界快照（JSONL 研究日志用）。"""
        return {
            "t": round(self.t, 6),
            "steps": self.steps,
            "seed": self.seed_value,
            "world": self.config.as_dict(),
            "civ_config": self.civ_config.as_dict(),
            "environment": self.environment_state(),
            "agents": [
                {
                    **character.to_dict(),
                    "at_left_wall": character.at_left_wall,
                    "at_right_wall": character.at_right_wall,
                }
                for character in self.agents
            ],
            "population": {
                "alive": len(self.alive_agents),
                "total_agents_created": self.registry.total_created,
                "max_agents_created": self.civ_config.max_agents_created,
                "total_deaths": self.registry.total_deaths,
                "extinct": self.civilization.extinct,
                "collapse_state": self.civilization.collapse_state,
            },
            "technology": self.technology.summary_zh(),
            "knowledge": {
                "verified": len(self.knowledge.verified_ids()),
                "total": len(self.knowledge.all()),
                "graph_depth": self.knowledge.depth(),
            },
            "facilities": [facility.to_dict() for facility in self.civilization.facilities],
            "infinite": bool(getattr(self.config, "infinite", False)),
            "window_origin_px": round(float(self.window_origin_px), 2),
            "infinite_fallback": self.infinite_fallback_reason,
        }

    # ------------------------------------------------------------------
    # 序列化（世界状态存/取档）
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """完整可序列化的世界状态（存/取档测试用）。

        先把存活居民的库存/知识/技能快照进注册表，否则*活人*的数据
        不进历史记录，存取档一个来回就会丢失。
        """
        for character in self.agents:
            record = self.registry.record(character.agent_id)
            if record is None:
                continue
            record.known_knowledge_ids = tuple(character.known_knowledge_ids)
            record.skills = {str(key): float(value) for key, value in character.skills.items()}
            record.inventory = {
                "wood": int(character.inventory.wood),
                "stone": int(character.inventory.stone),
                "fiber": int(character.inventory.fiber),
                "cord": int(character.inventory.cord),
                "basket": int(character.inventory.basket),
            }
        return {
            "state": self.state(),
            "registry": self.registry.to_dict(),
            "knowledge": self.knowledge.to_dict(),
            "memory": self.memory.to_dict(),
            "civilization": self.civilization.to_dict(),
        }

    def save(self, path: str | Path) -> Path:
        """把世界状态写入 JSON 文件。"""
        import json
        from pathlib import Path as _Path

        path = _Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path, config: WorldConfig | None = None) -> "World":
        """从 :meth:`save` 的输出重建世界（注册表 + 知识 + 文明）。"""
        import json
        from pathlib import Path as _Path

        data = json.loads(_Path(path).read_text(encoding="utf-8"))
        civ_data = data["state"].get("civ_config", {})
        civ_config = CivilizationConfig(**{
            key: civ_data[key]
            for key in civ_data
            if key in CivilizationConfig.__dataclass_fields__
        })
        world = cls(config, seed=int(data["state"].get("seed") or 0), civ_config=civ_config)
        world.registry = AgentRegistry.from_dict(data["registry"])
        world.knowledge = KnowledgeStore.from_dict(data["knowledge"])
        world.memory = MemoryStore.from_dict(data["memory"])
        world.civilization = CivilizationState.from_dict(
            data["civilization"],
            knowledge=world.knowledge,
            memory=world.memory,
        )
        # 存活居民：只重建注册表里仍标记为存活的那些
        world.agents.clear()
        for record in world.registry.records():
            if not record.alive:
                continue
            agent = world.spawn(
                x=0.0,
                name=record.name,
                sex=record.sex,
                age=0.0,
                generation=record.generation,
                agent_id=record.agent_id,
            )
            agent.created_at = record.born_at
            agent.parent_id = record.parent_id
            agent.inventory.wood = int(record.inventory.get("wood", 0))
            agent.inventory.stone = int(record.inventory.get("stone", 0))
            agent.inventory.fiber = int(record.inventory.get("fiber", 0))
            agent.inventory.cord = int(record.inventory.get("cord", 0))
            agent.inventory.tool = record.inventory.get("tool")  # type: ignore[assignment]
            agent.known_knowledge_ids = list(record.known_knowledge_ids)
            agent.skills = dict(record.skills)
        state = data["state"]
        world.t = float(state.get("t") or 0.0)
        world.steps = int(state.get("steps") or 0)
        return world

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        """调试表示。"""
        return (
            f"<World {self.config.width}x{self.config.height} "
            f"agents={len(self.agents)} t={self.t:.2f}>"
        )
