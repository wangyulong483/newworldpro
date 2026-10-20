"""千人群伙（V3 世界盒子的"人"）：numpy 结构化数组 + 向量化更新。

设计目标
========
单个居民不再是一个对象（旧世界的 Character + 神经网络 + 感知栈），
而是**结构化数组里的一行**。千人 = 6×1000 个数字，每帧整批更新：

* 饥饿增长（统一的代谢速率）
* 进食（从所在 cell 的 food 网格里扣，就地取食）
* 移动（食物浓度梯度 + 随机扩散 + 避水；饥饿驱动远离贫瘠区）
* 繁衍（成年 + 饱足 + 所在 cell 食物充足 + 承载力有余）
* 死亡（饥饿归零 / 老龄 / 站在水或火里）

承载力是**涌现的**而非硬编码：人口增长直到食物再生率与消耗率平衡。
这便是世界盒子的精髓——智能不在个体里，在规则的统计后果里。

数组语义（SoA，面向向量化）
---------------------------
``x, y`` 为 float cell 坐标（渲染与邻域计算时取整）；``hunger``
0..1（1 = 饱，0 = 饿死）；``age`` 世界秒；``tribe`` 是二期的部落
编号（一期恒为 0，为染色列预留）。``alive`` 掩码 + ``active`` 前缀
数量（死亡后槽位回收复用，避免反复分配）。

确定性
------
同 seed 下，初始化位置与每帧的随机数都来自固定序列的 Generator；
同一 (seed, dt 序列) 得到同一轨迹（测试断言用）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .map_engine import (
    DEEP_WATER,
    SHALLOW_WATER,
    FOREST,
    MapEngine,
)


@dataclass
class SwarmConfig:
    """群伙参数（人口动态的全部旋钮）。"""

    initial_population: int = 220
    #: 硬上限（数组容量；承载力之外的保险丝）
    capacity_hard: int = 4000
    #: 代谢：饥饿每秒下降（1.0 → 0 用时约 100 秒）
    metabolism_per_s: float = 0.010
    #: 一次进食从 cell 扣除的食物量（并转化为等量饥饿）
    bite_size: float = 0.35
    #: 成年（可繁衍）年龄（世界秒）
    mature_age_s: float = 12.0
    #: 繁衍间隔（世界秒）
    breed_cooldown_s: float = 14.0
    #: 繁衍条件之一：所在 cell 食物高于此值（就地取食后的残余即可支撑）
    breed_food_threshold: float = 0.12
    #: 繁衍条件之二：饱足高于此值
    breed_hunger_threshold: float = 0.55
    #: 繁衍概率上限（每秒）：即使条件全满足也不是必然繁殖（密度自限）
    breed_probability_per_s: float = 0.10
    #: 老龄死亡：超过该年龄后每秒的危险率（寿命期望 ≈ old_age + 1/hazard）
    old_age_s: float = 90.0
    old_age_hazard_per_s: float = 0.05
    #: 移动速度（cell/秒）
    move_speed: float = 3.0
    #: 移动随机性（0 = 纯梯度，1 = 纯随机游走）
    wander: float = 0.55
    #: 低于该饱足度才开始觅食移动（高于则原地休息省食物）
    forage_hunger: float = 0.95


#: 结构化数组的 dtype（顺序冻结：渲染层按名字取列）
SWARM_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("hunger", np.float32),
        ("age", np.float32),
        ("breed_cd", np.float32),
        ("tribe", np.int16),
        ("alive", np.bool_),
    ]
)


class Swarm:
    """俯视地图上的千人群伙。"""

    def __init__(
        self,
        engine: MapEngine,
        config: SwarmConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.engine = engine
        self.config = config or SwarmConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        #: 结构化数组（固定容量，alive 掩码标记有效行）
        capacity = max(64, int(self.config.capacity_hard))
        self.data = np.zeros(capacity, dtype=SWARM_DTYPE)
        self.count = 0  #: 有效行数（含死亡但未回收？否：count = 活着的行数上限）
        # 统计（telemetry / 状态栏）
        self.births = 0
        self.deaths = 0
        self.starvation_deaths = 0
        self.old_age_deaths = 0
        self.drowned = 0
        self.burned = 0
        #: P1.6：逐个体死亡年龄（精确 mean_life 分子）
        self.death_ages: list[float] = []
        # -- P1：策略注入 + 奖励/转移缓冲（见 swarm_policy.py） -----------
        #: None = V3 原规则行为；注入 SwarmPolicy 后移动决策由策略接管
        self.policy = None
        #: (obs, action, reward) 转移缓冲（policy 非 None 时记录）
        self.transitions: list[tuple[np.ndarray, int, float]] = []
        #: 累计奖励（评估指标）
        self.total_reward = 0.0
        self.eat_events = 0
        #: 本帧待结算的 (row, obs, action)——奖励在世界规则执行后回填
        self._pending: list[tuple[int, np.ndarray, int]] = []
        #: 每行上一次决策 (row -> (obs, action))：死亡帧补记终止转移用
        self._last_decision: dict[int, tuple[np.ndarray, int]] = {}
        self._spawn_initial()

    # ------------------------------------------------------------------
    # P1：策略接口
    # ------------------------------------------------------------------
    def attach_policy(self, policy) -> None:
        """注入移动策略（幂等）；记录转移供训练器 drain。"""
        from .swarm_policy import SwarmPolicy  # noqa: F401 - 类型提示
        if not callable(getattr(policy, "act", None)):
            raise TypeError("policy 必须提供 act(obs_batch) -> actions")
        self.policy = policy
        self.transitions.clear()
        self._pending.clear()

    def detach_policy(self) -> None:
        """卸载策略，回到 V3 原规则行为。"""
        self.policy = None
        self.transitions.clear()
        self._pending.clear()

    def drain_swarm_transitions(self) -> list[tuple[np.ndarray, int, float]]:
        """取走全部转移（训练器调用；顺序=个体在数组中的连续轨迹）。"""
        out = self.transitions
        self.transitions = []
        return out

    def _observations(self, view, rows: np.ndarray) -> np.ndarray:
        """为给定行组装 (len(rows), 9) 观测（P1 冻结定义）。"""
        engine = self.engine
        from .swarm_policy import OBS_DIM

        n = rows.shape[0]
        obs = np.zeros((n, OBS_DIM), dtype=np.float64)
        xs = np.clip(view["x"][rows].astype(np.int32), 0, engine.width - 1)
        ys = np.clip(view["y"][rows].astype(np.int32), 0, engine.height - 1)
        obs[:, 0] = np.clip(view["hunger"][rows], 0.0, 1.0)
        obs[:, 1] = np.minimum(view["age"][rows] / 100.0, 1.0)
        obs[:, 2] = engine.food[ys, xs]
        # 四方向 3 格食物（不可走=-1）
        for k, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
            nx = np.clip(xs + dx * 3, 0, engine.width - 1)
            ny = np.clip(ys + dy * 3, 0, engine.height - 1)
            food_there = engine.food[ny, nx].astype(np.float64)
            walkable = np.isin(engine.terrain[ny, nx], (3, 4, 5, 6, 7))
            obs[:, 3 + k] = np.where(walkable, food_there, -1.0)
        # 邻水警示：四邻有水域
        near_water = np.zeros(n, dtype=bool)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = np.clip(xs + dx, 0, engine.width - 1)
            ny = np.clip(ys + dy, 0, engine.height - 1)
            near_water |= np.isin(engine.terrain[ny, nx], (1, 2))
        obs[:, 7] = near_water.astype(np.float64)
        obs[:, 8] = 1.0  # bias
        return obs

    def _move_by_policy(self, dt: float) -> None:
        """P1：由注入策略决定移动；登记转移，事后回填奖励。

        决策覆盖**全部存活个体**（吃饱"懒得动"由策略学出来的原地动作
        表达，而非规则跳过）——保证每个个体每帧恰好一条转移，
        奖励结算与 eat_events 口径一致。世界规则仍约束执行：目标格
        不可走则动作被拒（原地不动）。
        """
        from .swarm_policy import ACTION_DELTAS

        cfg = self.config
        engine = self.engine
        view = self.data[: self.count]
        alive = view["alive"]
        rows = np.nonzero(alive)[0]
        if rows.size == 0:
            return
        obs = self._observations(view, rows)
        actions = np.asarray(self.policy.act(obs), dtype=np.int64)
        if actions.shape != (rows.size,):
            raise ValueError(f"policy.act 返回形状 {actions.shape} != ({rows.size},)")
        step = cfg.move_speed * dt
        moved_x = view["x"][rows].copy()
        moved_y = view["y"][rows].copy()
        for action_id in range(len(ACTION_DELTAS)):
            dx, dy = ACTION_DELTAS[action_id]
            mask = actions == action_id
            if dx == 0 and dy == 0:
                continue
            moved_x[mask] += dx * step
            moved_y[mask] += dy * step
        # 目标格不可走就别去（世界规则约束策略动作）
        nx_i = np.clip(moved_x.astype(np.int32), 0, engine.width - 1)
        ny_i = np.clip(moved_y.astype(np.int32), 0, engine.height - 1)
        walkable = np.isin(engine.terrain[ny_i, nx_i], (3, 4, 5, 6, 7))
        view["x"][rows] = np.where(walkable, np.clip(moved_x, 0, engine.width - 1), view["x"][rows])
        view["y"][rows] = np.where(walkable, np.clip(moved_y, 0, engine.height - 1), view["y"][rows])
        # 记录本帧转移（奖励待进食/死亡结算后回填）+ 缓存决策（死亡补记用）
        for i, row in enumerate(rows.tolist()):
            self._pending.append((row, obs[i], int(actions[i])))
            self._last_decision[int(row)] = (obs[i], int(actions[i]))

    def _move_by_rules(self, dt, cfg, view, alive, xi, yi) -> None:
        """V3 原规则移动（梯度+wander），保持逐比特原行为。"""
        engine = self.engine
        move_active = alive & (view["hunger"] < cfg.forage_hunger)  # 吃饱了懒得动
        n_active = int(move_active.sum())
        if n_active:
            rows = np.nonzero(move_active)[0]
            rand = self.rng.random((n_active, 4))  # 每个方向的选择概率
            best_dx = np.zeros(n_active, dtype=np.float32)
            best_dy = np.zeros(n_active, dtype=np.float32)
            cx = view["x"][rows]
            cy = view["y"][rows]
            # 四个探测方向（右左上下）
            for k, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
                nx = np.clip((cx + dx * 3).astype(np.int32), 0, engine.width - 1)
                ny = np.clip((cy + dy * 3).astype(np.int32), 0, engine.height - 1)
                score = engine.food[ny, nx] * (0.5 + rand[:, k])
                walkable = np.isin(engine.terrain[ny, nx],
                                   (3, 4, 5, 6, 7))  # WALKABLE 的字面展开（避免 import 环）
                score = np.where(walkable, score, -1.0)
                if k == 0:
                    best_score = score.copy()
                    best_dx = np.where(True, dx, 0).astype(np.float32) * np.ones(n_active, dtype=np.float32)
                    best_dy = np.zeros(n_active, dtype=np.float32)
                else:
                    better = score > best_score
                    best_score = np.where(better, score, best_score)
                    best_dx = np.where(better, dx, best_dx)
                    best_dy = np.where(better, dy, best_dy)
            # 混入随机游走（wander 越大约束越松）
            jitter_x = self.rng.normal(0.0, cfg.wander, n_active)
            jitter_y = self.rng.normal(0.0, cfg.wander, n_active)
            step = cfg.move_speed * dt
            new_x = view["x"][rows] + (best_dx + jitter_x * 0.5) * step
            new_y = view["y"][rows] + (best_dy + jitter_y * 0.5) * step
            # 目标格不可走就别去（原地不动）
            nx_i = np.clip(new_x.astype(np.int32), 0, engine.width - 1)
            ny_i = np.clip(new_y.astype(np.int32), 0, engine.height - 1)
            walkable = np.isin(engine.terrain[ny_i, nx_i], (3, 4, 5, 6, 7))
            new_x = np.where(walkable, new_x, view["x"][rows])
            new_y = np.where(walkable, new_y, view["y"][rows])
            view["x"][rows] = np.clip(new_x, 0, engine.width - 1)
            view["y"][rows] = np.clip(new_y, 0, engine.height - 1)

    def _settle_transitions(self, dt: float, view, dying_rows: np.ndarray,
                            ate_rows: set[int]) -> None:
        """P1.5：结算本帧转移奖励并写缓冲。

        奖励：进食按**实际吃到的食物量**（``self._eat_amount``，消除了
        P1 的常数 +1 带来的占位运气歧义——站桩只能吃到格子存量，
        持续进食必须学会移动）；每步时间成本。死亡惩罚在死亡判定处
        已按"上一动作"补记。
        """
        from .swarm_policy import REWARD_TIME_PER_S

        if not self._pending:
            return
        eat_amounts = self._eat_amount or {}
        for (row, obs, action) in self._pending:
            reward = REWARD_TIME_PER_S * dt
            reward += eat_amounts.get(row, 0.0)
            self.transitions.append((obs, action, float(reward)))
            self.total_reward += float(reward)
        # eat_events 口径 = 决策个体中吃到食物的（与转移对齐）
        self.eat_events += len(ate_rows & {p[0] for p in self._pending}) if self._pending else 0
        self._pending.clear()

    # ------------------------------------------------------------------
    def _spawn_initial(self) -> None:
        """把初始人口撒在最肥沃的可行走 cell 上。"""
        engine = self.engine
        cfg = self.config
        walkable = engine.walkable_mask()
        fertility = np.where(walkable, engine.food, 0.0)
        # 只要有点食物的地方都行，按食物加权撒点
        ys, xs = np.nonzero(fertility > 0.10)
        if ys.size == 0:
            # 全图贫瘠：退化为任何可行走格
            ys, xs = np.nonzero(walkable)
        if ys.size == 0:
            return
        weights = fertility[ys, xs] + 0.01
        picks = self.rng.choice(ys.size, size=min(cfg.initial_population, ys.size),
                                replace=False, p=weights / weights.sum())
        for index in picks:
            y, x = ys[index], xs[index]
            self._add(float(x), float(y), hunger=0.7)

    # ------------------------------------------------------------------
    def _add(self, x: float, y: float, *, hunger: float, tribe: int = 0) -> int:
        """在第一个空槽放一个新个体（找不到空槽返回 -1）。"""
        if self.count >= self.data.shape[0]:
            return -1
        index = self.count
        self.count += 1
        self.data[index] = (x, y, hunger, 0.0, 0.0, tribe, True)
        return index

    def _compact(self) -> None:
        """把存活行压实到数组前段（死亡槽位复用）。"""
        if self.count == 0:
            return
        alive_mask = self.data["alive"][: self.count]
        alive_count = int(alive_mask.sum())
        if alive_count == self.count:
            return
        self.data[:alive_count] = self.data[: self.count][alive_mask]
        self.count = alive_count

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @property
    def population(self) -> int:
        """当前存活个体数。"""
        if self.count == 0:
            return 0
        return int(self.data["alive"][: self.count].sum())

    def positions_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """存活个体的 (xs, ys) 坐标（渲染散点用）。"""
        if self.count == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty
        view = self.data[: self.count]
        alive = view["alive"]
        return view["x"][alive], view["y"][alive]

    def mean_hunger(self) -> float:
        """平均饱足度（0 = 濒死，1 = 吃饱）。"""
        if self.count == 0:
            return 0.0
        view = self.data[: self.count]
        alive = view["alive"]
        if not alive.any():
            return 0.0
        return float(view["hunger"][alive].mean())

    def carrying_capacity_estimate(self) -> int:
        """粗略承载力估计：总食物再生流量 / 个体消耗速率。

        用于 UI 展示与实验分析（不是硬约束——真实人口由动态决定）。
        """
        regrow_total = (
            self.engine.food_capacity() * self.engine.config.food_regrow_per_s
        )
        consumption = self.config.metabolism_per_s + self.config.bite_size * 0.02
        if consumption <= 0:
            return 0
        return int(regrow_total / consumption)

    # ------------------------------------------------------------------
    # 每帧更新（世界规则）
    # ------------------------------------------------------------------
    def update(self, dt: float) -> dict[str, float]:
        """推进所有个体一个物理步；返回本步的人口变化统计。

        顺序：饥饿 → 死亡判定 → 进食 → 移动 → 繁衍 → 压实。
        """
        if self.count == 0:
            return {"births": 0.0, "deaths": 0.0, "population": 0.0}
        cfg = self.config
        engine = self.engine
        view = self.data[: self.count]
        alive = view["alive"]

        # -- 饥饿与年龄增长 ------------------------------------------
        view["age"] += dt
        view["hunger"] -= cfg.metabolism_per_s * dt

        # -- 死亡判定（饥饿 / 老龄 / 水 / 火） --------------------------
        starve = alive & (view["hunger"] <= 0.0)
        old = alive & (view["age"] > cfg.old_age_s)
        # 老龄危险率：指数分布的危险函数，期望剩余寿命 1/hazard
        old &= self.rng.random(self.count) < min(1.0, cfg.old_age_hazard_per_s * dt)
        xi = np.clip(view["x"].astype(np.int32), 0, engine.width - 1)
        yi = np.clip(view["y"].astype(np.int32), 0, engine.height - 1)
        on_water = alive & np.isin(engine.terrain[yi, xi], (DEEP_WATER, SHALLOW_WATER))
        on_fire = alive & engine.burning[yi, xi]
        dying = starve | old | on_water | on_fire
        if dying.any():
            # P1：死亡帧补终止转移——奖励挂在"导致死亡的上一动作"上
            if self.policy is not None:
                from .swarm_policy import REWARD_DEATH
                for row in np.nonzero(dying)[0].tolist():
                    last = self._last_decision.get(int(row))
                    if last is not None:
                        obs_last, action_last = last
                        self.transitions.append((obs_last, action_last, float(REWARD_DEATH)))
                        self.total_reward += float(REWARD_DEATH)
            # P1.6：逐个体记录死亡年龄（mean_life 指标的精确分子，
            # 替换实验脚本里的种群级近似——那在人口打满容量时失真）
            for row in np.nonzero(dying)[0].tolist():
                self.death_ages.append(float(view["age"][row]))
            view["alive"] = view["alive"] & ~dying
            self.starvation_deaths += int(starve.sum())
            self.old_age_deaths += int(old.sum())
            self.drowned += int(on_water.sum())
            self.burned += int(on_fire.sum())
            total_new_deaths = int(dying.sum())
            self.deaths += total_new_deaths
        else:
            total_new_deaths = 0

        # 重新取活着的行
        alive = view["alive"]
        if not alive.any():
            self._compact()
            return {
                "births": 0.0,
                "deaths": float(total_new_deaths),
                "population": 0.0,
            }

        # -- 进食：从所在 cell 扣 bite_size（就地取食） ---------------
        # P1.5：门槛从"半口"降为"有残食就吃"（消除占位运气歧义——
        # 原 0.175 门槛让个体在残食格上挨饿，随机游走反而占优）。
        xs_i = xi[alive]
        ys_i = yi[alive]
        available = engine.food[ys_i, xs_i]
        eat_mask = available > 0.01
        #: 本帧各行实际吃到的量（P1.5 奖励按量结算）；
        #: 保留 None 以便 V3 规则路径零开销
        self._eat_amount: dict[int, float] | None = None
        if eat_mask.any():
            rows = np.nonzero(alive)[0][eat_mask]
            eat_amount = np.minimum(cfg.bite_size, engine.food[ys_i[eat_mask], xs_i[eat_mask]])
            engine.food[ys_i[eat_mask], xs_i[eat_mask]] -= eat_amount
            view["hunger"][rows] = np.minimum(
                1.0, view["hunger"][rows] + eat_amount
            )
            if self.policy is not None:
                self._eat_amount = {
                    int(row): float(amount)
                    for row, amount in zip(rows.tolist(), eat_amount.tolist())
                }

        # -- 移动：食物梯度 + 扩散噪声 + 避水 ---------------------------
        # P1：决策段已抽成可注入策略（swarm_policy.py）。
        # policy=None 时保持 V3 原规则行为（RuleSwarmPolicy 的等价实现）。
        if self.policy is not None:
            self._move_by_policy(dt)
        else:
            self._move_by_rules(dt, cfg, view, alive, xi, yi)

        # -- P1：结算奖励（进食按实际吃到量 / 死亡 / 时间成本） -------
        if self.policy is not None:
            ate_rows = set(np.nonzero(alive)[0][eat_mask].tolist()) if eat_mask.any() else set()
            self._settle_transitions(dt, view, dying_rows=np.nonzero(dying)[0] if dying.any() else np.array([], dtype=int), ate_rows=ate_rows)

        # -- 繁衍 -----------------------------------------------------
        # 冷却对所有存活个体流逝（不只是吃到东西的）
        view["breed_cd"] -= dt
        can_breed = (
            alive
            & (view["age"] >= cfg.mature_age_s)
            & (view["breed_cd"] <= 0.0)
            & (view["hunger"] >= cfg.breed_hunger_threshold)
        )
        # P1.5：删除 rich_cell（脚下食物 >= 阈值）条件——残食进食规则下
        # 个体会吃空脚下导致"自我抑制"（饱足 1.0 却永不繁衍）。
        # 营养条件已由 breed_hunger_threshold 表达，无需重复。
        # 繁衍概率：每秒 breed_probability_per_s（密度自限，避免瞬时爆炸）
        can_breed &= self.rng.random(self.count) < min(1.0, cfg.breed_probability_per_s * dt)
        # 承载力：软约束（超载时全体暂停繁衍）
        cap_estimate = max(32, self.carrying_capacity_estimate())
        if self.population >= cap_estimate:
            can_breed &= False
        rows = np.nonzero(can_breed)[0]
        births = 0
        if rows.size:
            for row in rows.tolist():
                if self.count >= self.data.shape[0]:
                    break
                x = float(view["x"][row]); y = float(view["y"][row])
                if self._add(x, y, hunger=0.45, tribe=int(view["tribe"][row])) >= 0:
                    view["breed_cd"][row] = cfg.breed_cooldown_s
                    births += 1
            self.births += births

        self._compact()
        return {
            "births": float(births),
            "deaths": float(total_new_deaths),
            "population": float(self.population),
        }

    # ------------------------------------------------------------------
    def summary_zh(self) -> str:
        """单行中文摘要（状态栏）。"""
        return (
            f"人口 {self.population}（出生{self.births}/死亡{self.deaths}"
            f"：饿{self.starvation_deaths} 老{self.old_age_deaths}"
            f" 溺{self.drowned} 焚{self.burned}）"
            f" 均饱足 {self.mean_hunger():.2f}"
        )
