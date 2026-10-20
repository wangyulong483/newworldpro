"""Headless 文明运行器：不开窗口做实验。

``SocietyRunner`` 按 :class:`RunSpec` 建好世界 + 种群，跑固定步数的
仿真，施加 seed 驱动的**环境扰动**（P4），用
:class:`~newworld.telemetry.Telemetry` 收集指标，返回可 JSON 化的
结果。这是 ``experiments/`` 用的入口。

设计要点
--------
* 运行器从不背着 agent 偷偷改世界数据：扰动走
  :class:`~newworld.pixelworld.PixelScene` / ecology 的公开 API，
  与天气或觅食 agent 一模一样。
* 扰动时间表是**seed 化的，不是剧本**：每个
  ``enable_disturbance=True`` 的组都会经历一次干旱、一次风暴、
  一次资源枯竭，各自发生在开局几分钟内 seed 随机的时刻，保证
  三种扰动都真的出现，而发展路线仍未被锁死。（实测：只随机安排
  一种扰动时 C、D 两组完全相同——单个迟到冲击改变不了已定的
  路线。）
* 感知约占一帧 95% 的开销，而神经特征抽取根本不读它
  （``neural.features`` 从不使用 ``perception``），因此实验通过
  ``spec.perception = False`` 默认关闭——行为结果完全一致，
  墙钟时间约快 30 倍。

CLI::

    python -m newworld.simulation --group D --seed 0 --steps 20000 --dt 0.05
    python -m newworld.simulation --group A --seed 1 --steps 5000 --save run.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    ABLATIONS,
    CivilizationConfig,
    WorldConfig,
    make_civilization_config,
    make_config,
)
from .population import Population
from .telemetry import Telemetry
from .world import World

#: seed 驱动的环境扰动种类（P4）
DISTURBANCE_KINDS: tuple[str, ...] = ("drought", "storm", "scarcity")


@dataclass
class RunSpec:
    """一次实验的全部定义（设计上可复现）。"""

    group: str = "D"
    seed: int = 0
    steps: int = 20000
    dt: float = 0.05
    width: int = 800
    height: int = 500
    #: 墙钟/行为开关——属于实验级配置，不属于消融
    perception: bool = False
    disturbance: bool | None = None          # None → 跟随消融组
    telemetry_window_s: float = 60.0
    log_dir: str | Path | None = None        # 事件与摘要写到这里
    save_path: str | Path | None = None      # 结束时保存完整世界状态
    #: 第一次灭绝即停止（默认：记录崩溃）
    stop_on_extinction: bool = True
    #: 行为克隆训练轮数预算（可选）
    train_rounds: int = 0
    #: 高层目标策略："heuristic" | "neural" | "random"
    goal_policy_mode: str = "heuristic"
    #: goal_policy_mode="neural" 的 checkpoint（None 则新建）
    goal_policy_path: str | Path | None = None
    #: 底层控制器消融："full" | "random" | "off"
    low_level_mode: str = "full"
    #: 有限组合式创新原型（None → 跟随消融组）
    enable_innovation: bool | None = None
    #: V2.1：真实视网膜特征是否进入观测（None → 配置默认 True）
    visual_features: bool | None = None
    #: 低层 RL checkpoint（low_level_mode="rl" 时加载；None → assets/lowlevel_rl.npz）
    lowlevel_policy_path: str | Path | None = None
    #: V2 无限世界：横向无界 + 窗口跟随（屏幕只是局部窗口）
    infinite: bool = False
    window_atoms: int = 0
    #: 累计创建上限覆盖（None → 消融组默认 8；对照实验可传 32/64）
    max_agents_created: int | None = None
    #: 感知/视觉特征刷新间隔帧数（None → 配置默认 4；大预算实验可加大摊薄成本）
    perception_stride: int | None = None
    #: 是否把社交网络读写 assets/society.npz（默认 False：实验 run 互相独立）
    persist_society: bool = False

    def as_dict(self) -> dict[str, Any]:
        """序列化为普通字典（进结果 JSON 用）。"""
        data = dict(self.__dict__)
        data["log_dir"] = None if self.log_dir is None else str(self.log_dir)
        data["save_path"] = None if self.save_path is None else str(self.save_path)
        data["goal_policy_path"] = (
            None if self.goal_policy_path is None else str(self.goal_policy_path)
        )
        data["lowlevel_policy_path"] = (
            None if self.lowlevel_policy_path is None else str(self.lowlevel_policy_path)
        )
        return data


class _RandomGoalPolicy:
    """均匀随机的高层策略，用作决策链消融的对照。"""

    def __init__(self, seed: int = 0) -> None:
        """初始化。"""
        self.rng = random.Random(seed)

    def sample(self, features, rng=None) -> int:  # noqa: ANN001 - 鸭子类型
        """采样一个动作索引。"""
        from .highlevel import GOAL_ACTIONS

        return self.rng.randrange(len(GOAL_ACTIONS))


@dataclass
class EnvironmentDisturbance:
    """seed 化的环境冲击（干旱 / 风暴 / 资源枯竭）。

    每个事件有持续时间，作用在*真实*世界状态上（降雨目标、植物/
    森林破坏、土壤湿度），因此它改变的是可行的发展路线，而不是
    一个 UI 标志。
    """

    seed: int = 0
    enabled: bool = True
    #: 首个事件时间窗（世界秒）——早到足以影响走向
    first_min_s: float = 60.0
    first_max_s: float = 150.0
    #: 相邻事件的最小/最大间隔
    interval_min_s: float = 110.0
    interval_max_s: float = 210.0
    duration_s: float = 90.0
    rng: random.Random = field(init=False, repr=False)
    schedule: list[tuple[float, str]] = field(init=False, default_factory=list)
    active_kind: str = ""
    active_until: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """生成三种扰动的乱序时间表（seed 决定，可复现）。"""
        self.rng = random.Random(self.seed * 7919 + 13)
        if self.enabled:
            kinds = list(DISTURBANCE_KINDS)
            self.rng.shuffle(kinds)
            when = self.rng.uniform(self.first_min_s, self.first_max_s)
            for kind in kinds:
                self.schedule.append((when, kind))
                when += self.rng.uniform(self.interval_min_s, self.interval_max_s)

    # ------------------------------------------------------------------
    def apply(self, world, telemetry: Telemetry | None = None) -> None:
        """推进扰动时间表；每个仿真步调用一次。"""
        if not self.enabled:
            return
        now = world.t
        ecology = world.pixel_scene.ecology
        if self.active_kind and now >= self.active_until:
            self.active_kind = ""
        while self.schedule and now >= self.schedule[0][0] and not self.active_kind:
            _, kind = self.schedule.pop(0)
            self.active_kind = kind
            self.active_until = now + self.duration_s
            record = {"t": round(now, 2), "kind": kind, "duration_s": self.duration_s}
            self.history.append(record)
            if telemetry is not None:
                try:
                    telemetry.event("disturbance", **record)
                except Exception:
                    pass
            # 事件开始时的一次性冲击：必须能改变发展路线，
            # 而不只是天气显示
            if kind == "scarcity":
                self._remove_resources(world)
            elif kind == "storm":
                self._destroy_plant(world)
            elif kind == "drought":
                self._dry_out(world)
        if self.active_kind == "drought":
            # 不下雨 + 快速蒸发：池塘缩小、生长停止
            ecology.rain_target = 0.0
            ecology.rain = max(0.0, ecology.rain - 0.05)
            ecology.rain_timer = max(ecology.rain_timer, self.duration_s)
        elif self.active_kind == "storm":
            ecology.rain_target = 0.95
            self._storm_damage(world)

    # ------------------------------------------------------------------
    def _remove_resources(self, world) -> None:
        """资源枯竭：整棵树 + 整块岩石被移除。

        只删散落原子没有可测影响（第一批消融已验证：C、D 两组相同），
        所以这里删的是规划器真正会导航过去的物体——真实的供给冲击。
        """
        scene = world.pixel_scene
        from . import materials

        for target_kind, material_key in (("tree", "wood"), ("rock", "stone")):
            candidates = [obj for obj in scene.objects if obj.kind == target_kind]
            if not candidates:
                continue
            victim = self.rng.choice(candidates)
            material = materials.MATERIAL_INDEX[material_key]
            leaf = materials.MATERIAL_INDEX["leaf"]
            for y in range(max(0, victim.top), min(scene.ah, victim.bottom + 1)):
                for x in range(max(0, victim.x0), min(scene.aw, victim.x1 + 1)):
                    if int(scene.grid[y, x]) in (material, leaf):
                        scene.clear_atom(x, y)
            if victim in scene.objects:
                scene.objects.remove(victim)
        scene.refresh_caches()

    def _destroy_plant(self, world) -> None:
        """风暴：掀翻一整丛浆果丛（食物供给冲击）。"""
        scene = world.pixel_scene
        bushes = [obj for obj in scene.objects if obj.kind == "bush"]
        if not bushes:
            return
        victim = self.rng.choice(bushes)
        for slot in list(getattr(victim, "berry_slots", ())):
            if slot in scene.berries:
                scene.berries.remove(slot)
        from . import materials

        leaf = materials.MATERIAL_INDEX["leaf"]
        berry = materials.MATERIAL_INDEX["berry"]
        for y in range(max(0, victim.top), min(scene.ah, victim.bottom + 1)):
            for x in range(max(0, victim.x0), min(scene.aw, victim.x1 + 1)):
                if int(scene.grid[y, x]) in (leaf, berry):
                    scene.clear_atom(x, y)
        if victim in scene.objects:
            scene.objects.remove(victim)
        scene.refresh_caches()

    def _dry_out(self, world) -> None:
        """干旱：土壤湿度骤降，再生停滞一段时间。"""
        ecology = world.pixel_scene.ecology
        ecology.moisture = (ecology.moisture * 0.35).astype(ecology.moisture.dtype)

    def _storm_damage(self, world) -> None:
        """风暴：在几个随机列上撕掉软植物原子（持续伤害）。"""
        scene = world.pixel_scene
        for _ in range(2):
            x = self.rng.uniform(0.0, float(world.config.width))
            y = float(world.config.ground_y)
            scene.trample(x, y, force=3.0)


class SocietyRunner:
    """Headless 地跑一个文明实验。"""

    def __init__(self, spec: RunSpec | None = None) -> None:
        """初始化。"""
        self.spec = spec if spec is not None else RunSpec()
        civ_config = make_civilization_config(
            self.spec.group,
            seed=self.spec.seed,
            enable_perception=bool(self.spec.perception),
        )
        if self.spec.disturbance is not None:
            civ_config.enable_disturbance = bool(self.spec.disturbance)
        civ_config.goal_policy_mode = str(self.spec.goal_policy_mode)
        civ_config.low_level_mode = str(self.spec.low_level_mode)
        if self.spec.enable_innovation is not None:
            civ_config.enable_innovation = bool(self.spec.enable_innovation)
        if self.spec.visual_features is not None:
            civ_config.enable_visual_features = bool(self.spec.visual_features)
        if self.spec.max_agents_created is not None:
            civ_config.max_agents_created = int(self.spec.max_agents_created)
        if self.spec.perception_stride is not None:
            civ_config.perception_stride = max(1, int(self.spec.perception_stride))
        self.civ_config: CivilizationConfig = civ_config
        self.world_config: WorldConfig = make_config(
            self.spec.width,
            self.spec.height,
            infinite=bool(self.spec.infinite),
            window_atoms=int(self.spec.window_atoms or 0),
        )
        self.world = World(self.world_config, seed=self.spec.seed, civ_config=self.civ_config)
        self.world.reset(seed=self.spec.seed)
        self.events_path = None
        if self.spec.log_dir is not None:
            log_dir = Path(self.spec.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.events_path = log_dir / f"events_{self.spec.group}_seed{self.spec.seed}.jsonl"
        self.telemetry = Telemetry(
            window_s=self.spec.telemetry_window_s,
            log_path=self.events_path,
            seed=self.spec.seed,
            group=self.spec.group,
        )
        self.population = Population(
            self.world,
            self.spec.seed,
            telemetry=self.telemetry,
            persist_society=bool(self.spec.persist_society),
        )
        self._attach_goal_policy()
        self._attach_lowlevel_policy()
        self.disturbance = EnvironmentDisturbance(
            seed=self.spec.seed,
            enabled=bool(self.civ_config.enable_disturbance),
        )
        self.result: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    def _attach_goal_policy(self) -> None:
        """按配置挂载学习型 / 随机高层策略。"""
        mode = str(self.spec.goal_policy_mode or "heuristic")
        if mode == "heuristic":
            return
        if mode == "random":
            self.population.attach_goal_policy(_RandomGoalPolicy(self.spec.seed))
            return
        try:
            from .highlevel import GoalPolicy

            if self.spec.goal_policy_path:
                policy = GoalPolicy.load(self.spec.goal_policy_path)
            else:
                policy = GoalPolicy(seed=self.spec.seed)
            self.population.attach_goal_policy(policy)
        except Exception as exc:  # pragma: no cover - 防御性
            raise RuntimeError(f"cannot attach neural goal policy: {exc}") from exc

    # ------------------------------------------------------------------
    def _attach_lowlevel_policy(self) -> None:
        """low_level_mode="rl" 时加载训练好的低层策略 checkpoint。"""
        if str(self.spec.low_level_mode) != "rl":
            return
        path = self.spec.lowlevel_policy_path
        if path is None:
            default = Path(__file__).resolve().parent.parent / "assets" / "lowlevel_rl.npz"
            path = default if default.exists() else None
        if path is None:
            # 没有 checkpoint：保持 BC 回退，但如实记录（不伪造 RL）
            self.population.attach_lowlevel_policy(None)
            return
        try:
            from .lowlevel_rl import LowLevelPolicy

            policy = LowLevelPolicy.load(str(path), seed=self.spec.seed)
            self.population.attach_lowlevel_policy(policy)
        except Exception:
            self.population.attach_lowlevel_policy(None)

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        """跑到步数上限或灭绝，返回完整结果字典。"""
        spec = self.spec
        started = time.perf_counter()
        steps_run = 0
        for step in range(spec.steps):
            self.disturbance.apply(self.world, self.telemetry)
            self.population.step(spec.dt)
            steps_run = step + 1
            if self.world.civilization.extinct and spec.stop_on_extinction:
                break
        for _ in range(spec.train_rounds):
            self.population.train_round(seed=spec.seed)
        summary = self.telemetry.summary()
        elapsed = time.perf_counter() - started
        self.result = {
            "spec": spec.as_dict(),
            "civ_config": self.civ_config.as_dict(),
            "world_config": self.world_config.as_dict(),
            "steps_run": steps_run,
            "world_time_s": round(self.world.t, 3),
            "wall_s": round(elapsed, 3),
            "steps_per_s": round(steps_run / max(elapsed, 1e-9), 1),
            "metrics": summary,
            "civilization": self.world.civilization.metrics(self.world),
            "goals": self.population.goals.stats() if self.population.goals else {},
            "knowledge": {
                "verified": self.world.knowledge.verified_ids(),
                "technologies": {
                    knowledge.knowledge_id: {
                        "status": knowledge.status,
                        "confidence": round(knowledge.confidence, 3),
                        "discovered_by": knowledge.discovered_by,
                        "discovered_at": round(knowledge.discovered_at, 2),
                        "successful_uses": knowledge.successful_uses,
                        "failed_uses": knowledge.failed_uses,
                    }
                    for knowledge in self.world.knowledge.all()
                },
                "graph_depth": self.world.knowledge.depth(),
                "graph_edges": len(self.world.knowledge.verified_edges()),
                "innovation_efficiency": round(self.world.knowledge.innovation_efficiency(), 4),
                "blocked_attempts": getattr(self.world.knowledge, "blocked_attempts", 0),
            },
            "archive": self.population.archive_summary(),
            "decisions": self.population.decision_share(),
            "decision_counts": dict(self.population.decision_counts),
            "lowlevel": self.population.lowlevel_summary(),
            "visual_features": bool(
                getattr(self.civ_config, "enable_visual_features", True)
            ),
            "visual": self.population.visual_summary(),
            "perception_stride": int(getattr(self.civ_config, "perception_stride", 1)),
            "infinite": bool(getattr(self.world.config, "infinite", False)),
            "window_origin_px": float(
                getattr(self.world.pixel_scene, "window_start_px", 0.0)
            ),
            "innovations": self.population.innovations(),
            "goal_policy_mode": str(self.spec.goal_policy_mode),
            "disturbances": list(self.disturbance.history),
            "telemetry_windows": getattr(self.telemetry, "windows", []),
            "events_count": len(getattr(self.telemetry, "events", ()) or ()),
            "extra_events": {"creation_denied": list(self.world.creation_denied)},
        }
        self.telemetry.close()
        self._write_summary()
        return self.result

    # ------------------------------------------------------------------
    def _write_summary(self) -> None:
        """把结果 JSON 写入 log_dir（配置了才写）。"""
        if self.result is None or self.spec.log_dir is None:
            return
        log_dir = Path(self.spec.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"summary_{self.spec.group}_seed{self.spec.seed}.json"
        path.write_text(
            json.dumps(self.result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def save(self, path: str | Path) -> Path:
        """保存完整世界状态（注册表、知识、记忆、文明）。"""
        return self.world.save(path)

    @classmethod
    def load(cls, path: str | Path) -> World:
        """载入 :meth:`save` 保存的世界状态。"""
        return World.load(path)

    # ------------------------------------------------------------------
    def summary_lines(self) -> list[str]:
        """结果的多行文本摘要（控制台打印用）。"""
        if self.result is None:
            return ["(not run yet)"]
        metrics = self.result["metrics"]
        civilization = self.result["civilization"]
        return [
            f"group={self.spec.group} seed={self.spec.seed} steps={self.result['steps_run']} "
            f"t={self.result['world_time_s']}s wall={self.result['wall_s']}s",
            f"total_agents_created={metrics.get('total_agents_created')} "
            f"alive={civilization.get('alive_count')} deaths={civilization.get('total_deaths')}",
            f"extinct={civilization.get('extinct')} at={civilization.get('extinction_time')} "
            f"cause={civilization.get('extinction_cause')}",
            f"tech_verified={metrics.get('tech_verified')} depth={metrics.get('tech_graph_depth')} "
            f"retention={metrics.get('knowledge_retention')}",
            f"goal_completion_rate={metrics.get('goal_completion_rate')} "
            f"repeat_run_max={metrics.get('repeat_run_max')} "
            f"no_progress={metrics.get('no_progress_repeat_ratio')}",
            f"collapse_state={civilization.get('collapse_state_code')} "
            f"cooperation_index={civilization.get('cooperation_index')}",
            f"disturbances={self.result.get('disturbances')}",
        ]


# ---------------------------------------------------------------------------
# 学习型高层目标策略的环境（REINFORCE 训练/评估）
# ---------------------------------------------------------------------------
class GoalPolicyEnv:
    """``newworld/highlevel.py`` 的 ``GoalPolicyTrainer`` 用的最小环境。

    它跑与实验完全相同的世界规则（同一个 ``Population``、同一套
    知识/文明服务），只是没有 UI、也没有环境扰动时间表（训练要
    平稳的任务）。

    训练器在构造之后通过 ``population.attach_goal_policy(policy)``
    挂上自己的策略，因此被优化的策略正是驱动仿真的那个。
    """

    def __init__(
        self,
        seed: int,
        *,
        group: str = "D",
        steps: int = 4000,
        dt: float = 0.05,
        width: int = 640,
        height: int = 400,
    ) -> None:
        """初始化。"""
        spec = RunSpec(
            group=group,
            seed=seed,
            steps=steps,
            dt=dt,
            width=width,
            height=height,
            perception=False,
            disturbance=False,
            goal_policy_mode="heuristic",  # 训练器自己挂策略
        )
        self.spec = spec
        self.runner = SocietyRunner(spec)
        self.population = self.runner.population
        self.world = self.runner.world
        self._steps = 0
        self._max_steps = int(steps)

    # ------------------------------------------------------------------
    def step(self) -> None:
        """推进一帧仿真。"""
        self.population.step(self.spec.dt)
        self._steps += 1

    def ended(self) -> bool:
        """是否已经灭绝/无人存活。"""
        if self.world.civilization.extinct:
            return True
        return not any(agent.alive for agent in self.world.agents)

    def result(self) -> dict[str, float]:
        """训练奖励/评估用的结果指标快照。"""
        world = self.world
        return {
            "world_time_s": round(float(world.t), 3),
            "steps_run": float(self._steps),
            "alive_count": float(len([a for a in world.agents if a.alive])),
            "total_agents_created": float(world.registry.total_created),
            "total_deaths": float(world.registry.total_deaths),
            "tech_verified": float(len(world.knowledge.verified_ids())),
            "facilities_active": float(
                len([f for f in world.civilization.facilities if getattr(f, "active", False)])
            ),
            "innovations": float(len(self.population.innovations())),
            "cooperation_index": float(getattr(world.civilization, "cooperation_index", 0.0)),
            "extinction_time": float(
                -1.0 if not world.civilization.extinct else (world.civilization.extinct_at or -1.0)
            ),
            "goal_decisions": float(self.population.decision_counts.get("policy_goal", 0)),
        }

    # ------------------------------------------------------------------
    def attach_goal_policy(self, policy) -> None:  # pragma: no cover - 便捷转发
        """把训练中的策略挂到种群上。"""
        self.population.attach_goal_policy(policy)


def make_goal_policy_env(
    seed: int,
    *,
    group: str = "D",
    steps: int = 4000,
    dt: float = 0.05,
    width: int = 640,
    height: int = 400,
    **kwargs,
) -> GoalPolicyEnv:
    """为高层目标策略构造训练/评估环境。"""
    del kwargs  # 容忍训练器将来传入的关键字
    return GoalPolicyEnv(
        seed,
        group=group,
        steps=steps,
        dt=dt,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# 低层动作策略的环境（V2：REINFORCE 训练/评估）
# ---------------------------------------------------------------------------
class LowLevelPolicyEnv:
    """``newworld/lowlevel_rl.py`` 的 ``LowLevelTrainer`` 用的最小环境。

    与 :class:`GoalPolicyEnv` 相同的世界规则；``low_level_mode="rl"``
    由训练器通过 ``population.attach_lowlevel_policy(policy)`` 挂上策略。
    训练/评估 seed 由调用方保证不重叠。
    """

    def __init__(
        self,
        seed: int,
        *,
        group: str = "C",
        steps: int = 4000,
        dt: float = 0.05,
        width: int = 640,
        height: int = 400,
        infinite: bool = False,
        window_atoms: int = 0,
        max_agents_created: int | None = None,
        visual_features: bool = True,
    ) -> None:
        """初始化。"""
        spec = RunSpec(
            group=group,
            seed=seed,
            steps=steps,
            dt=dt,
            width=width,
            height=height,
            perception=False,
            disturbance=False,
            low_level_mode="rl",       # 训练器自己挂策略
            visual_features=visual_features,
            infinite=infinite,
            window_atoms=window_atoms,
            max_agents_created=max_agents_created,
        )
        self.spec = spec
        self.runner = SocietyRunner(spec)
        self.population = self.runner.population
        self.world = self.runner.world
        self._steps = 0
        self._max_steps = int(steps)

    def step(self) -> None:
        """推进一帧仿真。"""
        self.population.step(self.spec.dt)
        self._steps += 1

    def ended(self) -> bool:
        """是否已经灭绝/无人存活。"""
        if self.world.civilization.extinct:
            return True
        return not any(agent.alive for agent in self.world.agents)

    def result(self) -> dict[str, float]:
        """训练/评估用的结果指标快照。"""
        world = self.world
        return {
            "world_time_s": round(float(world.t), 3),
            "steps_run": float(self._steps),
            "alive_count": float(len([a for a in world.agents if a.alive])),
            "total_agents_created": float(world.registry.total_created),
            "total_deaths": float(world.registry.total_deaths),
            "tech_verified": float(len(world.knowledge.verified_ids())),
            "facilities_active": float(
                len([f for f in world.civilization.facilities if getattr(f, "active", False)])
            ),
            "innovations": float(len(self.population.innovations())),
            "cooperation_index": float(getattr(world.civilization, "cooperation_index", 0.0)),
            "extinction_time": float(
                -1.0 if not world.civilization.extinct else (world.civilization.extinct_at or -1.0)
            ),
            "lowlevel_decisions": float(
                self.population.decision_counts.get("lowlevel_rl", 0)
            ),
        }

    def attach_lowlevel_policy(self, policy) -> None:  # pragma: no cover
        """便捷转发（训练器通常直接调 population 上的方法）。"""
        self.population.attach_lowlevel_policy(policy)


def make_lowlevel_rl_env(
    seed: int,
    *,
    group: str = "C",
    steps: int = 4000,
    dt: float = 0.05,
    width: int = 640,
    height: int = 400,
    infinite: bool = False,
    window_atoms: int = 0,
    max_agents_created: int | None = None,
    visual_features: bool = True,
    **kwargs,
) -> LowLevelPolicyEnv:
    """为低层动作策略构造训练/评估环境（真实奖励、真实世界规则）。"""
    del kwargs  # 容忍训练器将来传入的关键字
    return LowLevelPolicyEnv(
        seed,
        group=group,
        steps=steps,
        dt=dt,
        width=width,
        height=height,
        infinite=infinite,
        window_atoms=window_atoms,
        max_agents_created=max_agents_created,
        visual_features=visual_features,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """命令行入口：解析参数 → 跑实验 → 打印摘要。"""
    parser = argparse.ArgumentParser(description="headless civilization experiment runner")
    parser.add_argument("--group", default="D", choices=sorted(ABLATIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--perception", action="store_true", help="enable visual perception (slow)")
    parser.add_argument("--disturbance", choices=("on", "off"), default=None)
    parser.add_argument("--policy-mode", choices=("heuristic", "neural", "random"), default="heuristic",
                        help="high-level goal policy: rules / trained network / uniform random")
    parser.add_argument("--policy-path", default=None, help="checkpoint for --policy-mode neural")
    parser.add_argument("--low-level-mode", choices=("full", "random", "off", "rl"),
                        default="full", help="low-level controller (bc / random / off / rl)")
    parser.add_argument("--innovation", action="store_true", help="enable combinatorial innovation")
    parser.add_argument("--visual", dest="visual", action="store_true", default=None,
                        help="enable real retina features in the observation (default: on)")
    parser.add_argument("--no-visual", dest="visual", action="store_false",
                        help="disable retina features (ablation)")
    parser.add_argument("--infinite", action="store_true", help="unbounded scrolling world")
    parser.add_argument("--window-atoms", type=int, default=0,
                        help="infinite world window width in atoms (0 = auto)")
    parser.add_argument("--cap", type=int, default=None,
                        help="override cumulative agent creation cap (comparison experiments)")
    parser.add_argument("--lowlevel-policy", default=None,
                        help="checkpoint for --low-level-mode rl")
    parser.add_argument("--train-rounds", type=int, default=0)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--json", default=None, help="write the full result JSON here")
    parser.add_argument("--keep-going", action="store_true", help="do not stop at extinction")
    args = parser.parse_args(argv)

    spec = RunSpec(
        group=args.group,
        seed=args.seed,
        steps=args.steps,
        dt=args.dt,
        width=args.width,
        height=args.height,
        perception=args.perception,
        disturbance=None if args.disturbance is None else args.disturbance == "on",
        train_rounds=args.train_rounds,
        goal_policy_mode=args.policy_mode,
        goal_policy_path=args.policy_path,
        low_level_mode=args.low_level_mode,
        lowlevel_policy_path=args.lowlevel_policy,
        visual_features=args.visual,
        infinite=bool(args.infinite),
        window_atoms=int(args.window_atoms),
        max_agents_created=args.cap,
        enable_innovation=True if args.innovation else None,
        log_dir=args.log_dir,
        save_path=args.save,
        stop_on_extinction=not args.keep_going,
    )
    runner = SocietyRunner(spec)
    result = runner.run()
    if args.save:
        runner.save(args.save)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    print("\n".join(runner.summary_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
