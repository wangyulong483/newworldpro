"""P6-B 生成式记忆与反思层（去 LLM 版 Generative Agents 机制）。

定位
====
Generative Agents（Park et al., 2023）的三件核心机制，本模块用纯数值规则
复现，不调用任何 LLM：

1. 记忆流（memory stream）：每个个体一条固定容量的事件流
   MemoryRecord（时间、类型、标签、重要度、特征向量）。
2. 检索（retrieval）：score = w_rec·2^(-(now-t)/halflife)
   + w_imp·importance + w_rel·cosine(feature, query)——新近 × 重要 × 相关，
   完全可微分数值，无自然语言 embedding。
3. 反思（reflection）：每 reflection_every_s 秒把"自上次反思以来"的事件
   聚合成更高层记忆（主觅食方向 / 风险占比），写入 kind=reflection、
   importance>=0.7 的新记录。
4. 条件化行为：检索结果被压成 6 维观测
   [remembered_food_dx, remembered_food_dy, recent_success_rate,
   recent_danger_rate, reflection_food_bias, reflection_risk_bias]，由策略
   网络据此决定动作——即"计划/行动条件于检索到的记忆"。注意：特征只来自
   已存储事件（+ 作为检索 query 的当前感知），不存在任何"正确答案"硬编码。

设计取舍
========
* 记忆容量固定（MemorySettings.capacity，默认 24）：满了按"保留分数最低者"
  淘汰（分数=当前权重下、以新记录时刻为 now 的检索分，relevance 项为 0；
  同分淘汰更旧的）。这样记忆流不会无界增长。
* 每帧工作量上界 O(N · capacity · FEATURE_DIM)（N=存活个体数），另加反射
  触发的 O(E · capacity)；不存在跨帧累积的隐藏状态。
* 事件采样有节流（避免刷屏）：进食逐次记录；濒死只在"进入 <0.2"时记录
  一次；休息每 REST_EVENT_INTERVAL_S 秒最多一条。这些节流规则只影响记忆流
  的采样率，不影响检索/反思逻辑。
* 确定性：无随机采样（反思相位由 seed 决定），同 seed 同轨迹 → 同记忆。

与核心栈的接口
==============
* build_memory_layer(swarm, cfg, seed=0) 构造 MemoryLayer；
* attach_to_swarm() 向 swarm.register_obs_provider 注册 6 维观测；
* update(dt) 由 Simulation.step 在 swarm.update 之后调用；
* stats / count_of(row) / summary_zh() 供实验/实验室后端读取。

行号重排（compaction）
======================
Swarm 用死亡行压实（_compact）复用槽位。记忆层按 row 维护状态，必须在
swarm.compact_generation 变化时用 swarm.compact_keep（old row ids，按新顺序
排列）做一次重排；本帧事件（进食行号是压实前旧行号）也据此映射到新行号。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .settings import MemorySettings

#: 记忆特征维度：食物方向 dx/dy、食物强度、危险标记、紧迫度
FEATURE_DIM = 5
#: 记忆层追加到基础观测（9 维）后的观测维度
OBS_DIM = 6
#: 基础观测里的四方向（东/西/南/北，与 swarm._base_observations 一致）
DIRECTION_UNITS = np.asarray(
    ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)), dtype=np.float64
)
DIRECTION_TAGS = ("E", "W", "S", "N")

KIND_ATE = "ate"
KIND_NEAR_DEATH = "near_death"
KIND_REST = "rest"
KIND_REFLECTION = "reflection"
EVENT_KINDS = (KIND_ATE, KIND_NEAR_DEATH, KIND_REST, KIND_REFLECTION)

#: 特征列语义（诊断/文档用）
FEATURE_NAMES = ("food_dx", "food_dy", "food_strength", "danger", "urgency")

#: 濒死阈值（与任务口径一致：hunger < 0.2）
NEAR_DEATH_HUNGER = 0.2
#: 休息事件最少间隔（世界秒）：只用于节流，避免刷屏
REST_EVENT_INTERVAL_S = 5.0
#: 休息事件的饱足门槛
REST_HUNGER_THRESHOLD = 0.6
#: 小于一口 5% 的"啃残渣"不入记忆流（否则满饱个体逐帧微吃会刷爆容量，
#: 让真正有意义的事件被瞬间挤出；见报告 limitation）
MIN_ATE_AMOUNT_FRAC = 0.05
#: 反射重要性下界（任务要求 >= 0.7）
REFLECTION_MIN_IMPORTANCE = 0.7

_ZERO_OBS = np.zeros(OBS_DIM, dtype=np.float64)
_ZERO_FEATURE = np.zeros(FEATURE_DIM, dtype=np.float64)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _unit_xy(dx: float, dy: float) -> tuple[float, float]:
    norm = math.hypot(float(dx), float(dy))
    if norm <= 1e-12:
        return (0.0, 0.0)
    return (float(dx) / norm, float(dy) / norm)


def direction_bucket(dx: float, dy: float) -> str:
    """把 (dx, dy) 离散成 E/W/S/N/here（与基础观测四方向同序）。"""
    dx, dy = float(dx), float(dy)
    if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
        return "here"
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "S" if dy > 0 else "N"


def parse_retrieval_weights(weights: Any) -> tuple[float, float, float]:
    """把权重统一成 (w_recency, w_importance, w_relevance)。

    接受 None（全 1）、dict（键 recency/importance/relevance，缺省 1）、或
    长度 3 的序列。
    """
    if weights is None:
        return (1.0, 1.0, 1.0)
    if isinstance(weights, dict):
        return (
            float(weights.get("recency", 1.0)),
            float(weights.get("importance", 1.0)),
            float(weights.get("relevance", 1.0)),
        )
    values = list(weights)
    if len(values) != 3:
        raise ValueError(f"weights 需要长度 3，得到 {len(values)}")
    return (float(values[0]), float(values[1]), float(values[2]))


def cosine_similarity(features: np.ndarray, query: np.ndarray) -> np.ndarray:
    """逐行余弦相似度；任一范数为 0 时该行取 0（避免 0/0）。"""
    matrix = np.asarray(features, dtype=np.float64)
    vector = np.asarray(query, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != vector.size:
        raise ValueError(f"特征形状 {matrix.shape} 与 query {vector.shape} 不匹配")
    out = np.zeros(matrix.shape[0], dtype=np.float64)
    if matrix.shape[0] == 0 or vector.size == 0:
        return out
    query_norm = float(np.linalg.norm(vector))
    if query_norm <= 1e-12:
        return out
    row_norm = np.linalg.norm(matrix, axis=1)
    valid = row_norm > 1e-12
    if valid.any():
        out[valid] = (matrix[valid] @ vector) / (row_norm[valid] * query_norm)
    return np.clip(out, -1.0, 1.0)


def retrieval_scores(
    times: np.ndarray,
    importances: np.ndarray,
    features: np.ndarray,
    now_t: float,
    query: Sequence[float] | np.ndarray | None,
    weights: Any = (1.0, 1.0, 1.0),
    halflife: float = 30.0,
) -> np.ndarray:
    """Generative Agents 式检索分数（本模块唯一实现，供流/层/测试复用）。

    w_rec·2^(-(now-t)/halflife) + w_imp·importance + w_rel·cosine(feature, q)

    * 未来时间（now < t）按 age=0 处理（不许新近性 > 1）；
    * query=None 或零范数 → relevance 项为 0；
    * halflife 必须 > 0。
    """
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    importances = np.asarray(importances, dtype=np.float64).reshape(-1)
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != times.size:
        raise ValueError(f"features 形状 {matrix.shape} 与 times {times.shape} 不匹配")
    halflife = float(halflife)
    if halflife <= 0.0 or not math.isfinite(halflife):
        raise ValueError("halflife 必须 > 0")
    w_rec, w_imp, w_rel = parse_retrieval_weights(weights)
    age = np.maximum(0.0, float(now_t) - times)
    recency = np.power(2.0, -age / halflife)
    if query is None:
        relevance = np.zeros(times.size, dtype=np.float64)
    else:
        relevance = cosine_similarity(matrix, query)
    return w_rec * recency + w_imp * importances + w_rel * relevance


# ---------------------------------------------------------------------------
# 记忆记录 + 事件流（不依赖 swarm，可单元测试）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MemoryRecord:
    """一条记忆事件（不可变，便于跨层共享与测试断言）。"""

    t: float
    kind: str
    tag: str
    importance: float
    feature: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", float(self.t))
        if not math.isfinite(self.t):
            raise ValueError("MemoryRecord.t 必须有限")
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"未知记忆类型 {self.kind!r}；可用 {EVENT_KINDS}")
        if not isinstance(self.tag, str) or not self.tag:
            raise ValueError("MemoryRecord.tag 必须是非空字符串")
        importance = float(self.importance)
        if not math.isfinite(importance) or not 0.0 <= importance <= 1.0:
            raise ValueError(f"importance 必须在 [0,1]，得到 {importance!r}")
        object.__setattr__(self, "importance", importance)
        feature = np.asarray(self.feature, dtype=np.float64).reshape(-1)
        if feature.size != FEATURE_DIM:
            raise ValueError(f"feature 维度 {feature.size} != {FEATURE_DIM}")
        if not np.all(np.isfinite(feature)):
            raise ValueError("feature 含非有限值")
        object.__setattr__(self, "feature", feature)


class MemoryStream:
    """固定容量记忆流（Generative Agents memory stream 的数值版）。

    存储为插入有序的 list[MemoryRecord]（容量很小，cap≈24），检索时一次性
    向量化。容量满后先淘汰保留分数最低的旧记录再追加：

    * 保留分数 = 以新记录时刻为 now 的检索分（relevance 项为 0，因为淘汰时
      没有 query），权重/半衰期用流自身的默认配置；
    * 同分淘汰时间更早的记录（np.lexsort((times, scores))）。

    这样频繁被检索到（新近/重要）的记忆活得更久，纯"最旧淘汰"会让重要事件
    过早消失。
    """

    def __init__(
        self,
        capacity: int = 24,
        *,
        weights: Any = (1.0, 1.0, 1.0),
        halflife: float = 30.0,
    ) -> None:
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError("MemoryStream.capacity >= 1")
        halflife = float(halflife)
        if halflife <= 0.0:
            raise ValueError("MemoryStream.halflife > 0")
        self.capacity = capacity
        self.weights = parse_retrieval_weights(weights)
        self.halflife = halflife
        self._records: list[MemoryRecord] = []
        self._times = np.zeros(0, dtype=np.float64)
        self._importances = np.zeros(0, dtype=np.float64)
        self._features = np.zeros((0, FEATURE_DIM), dtype=np.float64)
        self._dirty = False

    # -- 基本访问 -------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> list[MemoryRecord]:
        """当前记忆记录的副本（浅拷贝列表，记录本身不可变）。"""
        return list(self._records)

    def _ensure_arrays(self) -> None:
        if not self._dirty:
            return
        self._times = np.asarray([r.t for r in self._records], dtype=np.float64)
        self._importances = np.asarray(
            [r.importance for r in self._records], dtype=np.float64
        )
        self._features = (
            np.vstack([r.feature for r in self._records])
            if self._records
            else np.zeros((0, FEATURE_DIM), dtype=np.float64)
        )
        self._dirty = False

    # -- 写入 -----------------------------------------------------------
    def add(self, record: MemoryRecord) -> MemoryRecord | None:
        """追加记忆；容量已满时淘汰保留分数最低者并返回它（否则 None）。"""
        if not isinstance(record, MemoryRecord):
            raise TypeError(f"需要 MemoryRecord，得到 {type(record).__name__}")
        evicted: MemoryRecord | None = None
        if len(self._records) >= self.capacity:
            self._ensure_arrays()
            scores = retrieval_scores(
                self._times,
                self._importances,
                self._features,
                record.t,
                None,
                self.weights,
                self.halflife,
            )
            order = np.lexsort((self._times, scores))
            index = int(order[0])
            evicted = self._records.pop(index)
            self._dirty = True
        self._records.append(record)
        self._dirty = True
        return evicted

    # -- 检索 -----------------------------------------------------------
    def scores(
        self,
        query: Sequence[float] | np.ndarray | None,
        now_t: float,
        weights: Any = None,
        halflife: float | None = None,
    ) -> np.ndarray:
        """当前全部记录的检索分数（与 retrieve 同口径）。"""
        self._ensure_arrays()
        return retrieval_scores(
            self._times,
            self._importances,
            self._features,
            now_t,
            query,
            self.weights if weights is None else weights,
            self.halflife if halflife is None else halflife,
        )

    def retrieve(
        self,
        query: Sequence[float] | np.ndarray | None,
        now_t: float,
        weights: Any = None,
        halflife: float | None = None,
        top_k: int = 3,
    ) -> list[tuple[float, MemoryRecord]]:
        """返回 top_k 条 (score, record)，按分数降序。

        确定性排序：分数降序 → importance 降序 → t 降序（新者优先）→ 插入序。
        分数相同时不会因 numpy 非稳定排序而抖动。
        """
        top_k = int(top_k)
        if top_k < 1:
            raise ValueError("top_k >= 1")
        scores = self.scores(query, now_t, weights, halflife)
        if scores.size == 0:
            return []
        count = scores.size
        order = np.lexsort(
            (np.arange(count), -self._times, -self._importances, -scores)
        )
        picked = order[: min(top_k, count)]
        return [(float(scores[i]), self._records[int(i)]) for i in picked]


# ---------------------------------------------------------------------------
# 反思聚合（纯函数，单元测试友好）
# ---------------------------------------------------------------------------
def aggregate_reflection(
    records: Sequence[MemoryRecord],
) -> tuple[str, np.ndarray, float]:
    """把一批事件聚合成一条更高层反思记忆。

    规则（确定性，无 LLM）：
    * 主觅食方向 = 所有 ate 事件方向向量的重要度加权和（归一化）；
    * 成功率 success = ate 数 / 事件数，风险率 danger = near_death 数 / 事件数；
    * 紧迫度 = 事件 urgency 均值；
    * 反思重要性 = clip(0.7 + 0.3·confidence, 0.7, 1.0)，
      confidence = max(|方向合力|/max(1,ate数), success)，保证 >= 0.7。

    返回 (tag, feature, importance)。
    """
    events = [record for record in records if record.kind != KIND_REFLECTION]
    if not events:
        return ("reflect:empty", _ZERO_FEATURE.copy(), REFLECTION_MIN_IMPORTANCE)
    direction = np.zeros(2, dtype=np.float64)
    ate_count = 0
    danger_count = 0
    urgency_total = 0.0
    for record in events:
        feature = record.feature
        urgency_total += float(feature[4])
        if record.kind == KIND_ATE:
            ate_count += 1
            dx, dy = _unit_xy(float(feature[0]), float(feature[1]))
            if dx != 0.0 or dy != 0.0:
                direction += np.asarray((dx, dy)) * record.importance
        elif record.kind == KIND_NEAR_DEATH:
            danger_count += 1
    total = float(len(events))
    success = ate_count / total
    danger = danger_count / total
    urgency = urgency_total / total
    norm = float(np.linalg.norm(direction))
    if norm > 1e-12:
        unit = direction / norm
        bucket = direction_bucket(float(unit[0]), float(unit[1]))
        tag = f"reflect:{bucket}"
    else:
        unit = np.zeros(2, dtype=np.float64)
        tag = "reflect:danger" if danger >= 0.5 else "reflect:neutral"
    confidence = min(1.0, max(norm / max(1.0, float(ate_count)), success))
    importance = min(1.0, max(REFLECTION_MIN_IMPORTANCE, 0.7 + 0.3 * confidence))
    feature = np.asarray(
        (float(unit[0]), float(unit[1]), success, danger, urgency),
        dtype=np.float64,
    )
    return (tag, feature, importance)


# ---------------------------------------------------------------------------
# 记忆层：per-row 记忆流 + 检索特征 + 周期反思
# ---------------------------------------------------------------------------
class MemoryLayer:
    """群伙级记忆/反思机制层（逐个体一条 MemoryStream）。"""

    def __init__(self, swarm: Any, cfg: MemorySettings, *, seed: int = 0) -> None:
        self.swarm = swarm
        self.cfg = cfg
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._weights = (
            float(cfg.weight_recency),
            float(cfg.weight_importance),
            float(cfg.weight_relevance),
        )
        self._obs_dim = 0
        self._streams: dict[int, MemoryStream] = {}
        self._features: dict[int, np.ndarray] = {}
        self._last_reflection: dict[int, np.ndarray] = {}
        self._prev_xy: dict[int, tuple[float, float]] = {}
        self._heading: dict[int, tuple[float, float]] = {}
        self._prev_hunger: dict[int, float] = {}
        self._prev_near_death: dict[int, bool] = {}
        self._last_rest_t: dict[int, float] = {}
        self._last_reflect_t: dict[int, float] = {}
        self._time = 0.0
        self._dt = 0.05
        self._generation = int(getattr(swarm, "compact_generation", 0))
        self._frame_old_to_new: dict[int, int] | None = None
        # 统计计数器（累计）
        self.updates = 0
        self.events_recorded = 0
        self.retrievals = 0
        self.reflections = 0  # 反思轮次（一次 tick 至少产生一条洞察）
        self.insights = 0  # 反思记忆（洞察）条数
        self.reindexed = 0
        # 反思相位由 seed 决定（0.5~1.0 个周期后首次反思），保证可复现
        interval = float(cfg.reflection_every_s)
        self._next_reflection_t = interval * (0.5 + 0.5 * float(self._rng.random()))

    # -- 接口 -----------------------------------------------------------
    def attach_to_swarm(self) -> int:
        """注册 6 维记忆观测 provider，返回 swarm 总观测维度。"""
        self._obs_dim = int(
            self.swarm.register_obs_provider(self._provide_observations, OBS_DIM)
        )
        return self._obs_dim

    @property
    def memory_obs_dim(self) -> int:
        return OBS_DIM

    def count_of(self, row: int) -> int:
        """某行当前记忆条数（未知行返回 0）。"""
        stream = self._streams.get(int(row))
        return len(stream) if stream is not None else 0

    # -- 观测 provider（每次策略前向都会调用）--------------------------
    def _provide_observations(self, view: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """返回缓存的 6 维记忆特征；新行（出生等）现场补算。"""
        rows_array = np.asarray(rows, dtype=np.int64).reshape(-1)
        count = rows_array.size
        if count == 0:
            return np.zeros((0, OBS_DIM), dtype=np.float64)
        cache = self._features
        missing = [int(row) for row in rows_array.tolist() if int(row) not in cache]
        if missing:
            self._refresh_features(view, np.asarray(missing, dtype=np.int64))
        out = np.empty((count, OBS_DIM), dtype=np.float64)
        for index, row in enumerate(rows_array.tolist()):
            out[index] = cache.get(int(row), _ZERO_OBS)
        if not np.all(np.isfinite(out)):
            raise ValueError("记忆观测含非有限值")
        return out

    # -- 每帧更新 -------------------------------------------------------
    def update(self, dt: float) -> None:
        """推进一帧：重排 -> 记录事件 -> 反思 -> 检索 -> 更新基线。

        复杂度上界 O(N·capacity·FEATURE_DIM)（N=存活行数）。
        """
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("dt 必须是非负有限值")
        self.updates += 1
        self._dt = dt
        self._time += dt
        swarm = self.swarm
        self._frame_old_to_new = self._reindex_if_needed()
        count = int(getattr(swarm, "count", 0))
        if count <= 0:
            return
        view = swarm.data[:count]
        alive_rows = np.nonzero(view["alive"])[0].astype(np.int64)
        alive_set = set(int(row) for row in alive_rows.tolist())
        self._drop_dead(alive_set)
        for row in alive_rows.tolist():
            row = int(row)
            if row not in self._streams:
                self._streams[row] = self._new_stream()
        self._record_events(view, alive_rows)
        self._maybe_reflect(alive_rows)
        base_obs = swarm._base_observations(view, alive_rows)
        self._refresh_features(view, alive_rows, base_obs)
        self._update_baselines(view, alive_rows)

    # -- 内部：行号重排 --------------------------------------------------
    def _reindex_if_needed(self) -> dict[int, int] | None:
        generation = int(getattr(self.swarm, "compact_generation", 0))
        if generation == self._generation:
            return None
        keep = np.asarray(
            getattr(self.swarm, "compact_keep", np.arange(0)), dtype=np.int64
        ).reshape(-1)
        old_to_new = {int(old): new for new, old in enumerate(keep.tolist())}
        self._streams = {
            int(new): self._streams[int(old)]
            for new, old in enumerate(keep.tolist())
            if int(old) in self._streams
        }
        self._features = {
            int(new): self._features[int(old)]
            for new, old in enumerate(keep.tolist())
            if int(old) in self._features
        }
        self._last_reflection = {
            int(new): self._last_reflection[int(old)]
            for new, old in enumerate(keep.tolist())
            if int(old) in self._last_reflection
        }
        for attribute in (
            self._prev_xy,
            self._heading,
            self._prev_hunger,
            self._prev_near_death,
            self._last_rest_t,
            self._last_reflect_t,
        ):
            remapped = {
                int(old_to_new[int(old)]): value
                for old, value in attribute.items()
                if int(old) in old_to_new
            }
            attribute.clear()
            attribute.update(remapped)
        self._generation = generation
        self.reindexed += 1
        return old_to_new

    def _new_stream(self) -> MemoryStream:
        return MemoryStream(
            self.cfg.capacity,
            weights=self._weights,
            halflife=float(self.cfg.recency_halflife_s),
        )

    def _drop_dead(self, alive_set: set[int]) -> None:
        if not self._streams:
            return
        for row in list(self._streams):
            if row not in alive_set:
                del self._streams[row]
                self._features.pop(row, None)
                self._last_reflection.pop(row, None)
                self._prev_xy.pop(row, None)
                self._heading.pop(row, None)
                self._prev_hunger.pop(row, None)
                self._prev_near_death.pop(row, None)
                self._last_rest_t.pop(row, None)
                self._last_reflect_t.pop(row, None)

    # -- 内部：事件记录 --------------------------------------------------
    def _record_events(self, view: np.ndarray, alive_rows: np.ndarray) -> None:
        if alive_rows.size == 0:
            return
        swarm = self.swarm
        cfg = getattr(swarm, "config", None)
        metabolism = float(getattr(cfg, "metabolism_per_s", 0.01))
        bite_size = max(1e-9, float(getattr(cfg, "bite_size", 0.35)))
        eat_amounts = self._eat_amounts_by_new_row()
        now = self._time
        hunger = view["hunger"]
        for row in alive_rows.tolist():
            row = int(row)
            hunger_now = float(hunger[row])
            hunger_before = self._prev_hunger.get(row)
            prev_xy = self._prev_xy.get(row)
            pos = (float(view["x"][row]), float(view["y"][row]))
            # -- 进食：饥饿本帧上升 = 吃到东西（规则/策略两条路径通用）
            ate = hunger_before is not None and hunger_now > hunger_before + 1e-6
            if ate:
                amount = eat_amounts.get(row)
                if amount is None:
                    amount = max(
                        0.0,
                        min(
                            bite_size,
                            hunger_now - hunger_before + metabolism * self._dt,
                        ),
                    )
                if amount / bite_size >= MIN_ATE_AMOUNT_FRAC:
                    self._record_ate(
                        row,
                        amount,
                        bite_size,
                        prev_xy,
                        pos,
                        hunger_before,
                        now,
                    )
            # -- 濒死：只在进入 < 0.2 时记录（避免逐帧刷屏）
            near_death = hunger_now < NEAR_DEATH_HUNGER
            was_near_death = self._prev_near_death.get(row, False)
            if near_death and not was_near_death:
                urgency = min(
                    1.0,
                    max(0.0, (NEAR_DEATH_HUNGER - hunger_now) / NEAR_DEATH_HUNGER),
                )
                importance = min(1.0, 0.7 + 0.3 * urgency)
                feature = np.asarray(
                    (0.0, 0.0, 0.0, 1.0, urgency), dtype=np.float64
                )
                self._append(
                    row,
                    MemoryRecord(now, KIND_NEAR_DEATH, "danger", importance, feature),
                )
            self._prev_near_death[row] = near_death
            # -- 休息：饱足且本帧无进食，按 REST_EVENT_INTERVAL_S 节流
            if (
                not ate
                and not near_death
                and hunger_now >= REST_HUNGER_THRESHOLD
                and now - self._last_rest_t.get(row, -1e9) >= REST_EVENT_INTERVAL_S
            ):
                deficit = min(1.0, max(0.0, 1.0 - hunger_now))
                importance = min(1.0, max(0.0, 0.05 + 0.15 * hunger_now))
                feature = np.asarray(
                    (0.0, 0.0, 0.0, 0.0, deficit), dtype=np.float64
                )
                self._append(
                    row, MemoryRecord(now, KIND_REST, "rest", importance, feature)
                )
                self._last_rest_t[row] = now

    def _eat_amounts_by_new_row(self) -> dict[int, float]:
        """本帧实际进食量（映射到压实后的新行号；无策略路径时为空）。"""
        raw = getattr(self.swarm, "_eat_amount", None)
        if not raw:
            return {}
        mapping = self._frame_old_to_new
        if mapping is None:
            return {int(row): float(amount) for row, amount in raw.items()}
        out: dict[int, float] = {}
        for old, amount in raw.items():
            new = mapping.get(int(old))
            if new is not None:
                out[int(new)] = float(amount)
        return out

    def _record_ate(
        self,
        row: int,
        amount: float,
        bite_size: float,
        prev_xy: tuple[float, float] | None,
        pos: tuple[float, float],
        hunger_before: float,
        now: float,
    ) -> None:
        amount_frac = min(1.0, max(0.0, float(amount) / bite_size))
        if prev_xy is None:
            dx = dy = 0.0
        else:
            dx = pos[0] - prev_xy[0]
            dy = pos[1] - prev_xy[1]
        ux, uy = _unit_xy(dx, dy)
        # 原地吃（本帧位移≈0）时用最近一次有意义的移动朝向：记忆编码的是
        # "我朝哪个方向走才吃到东西"，而不是零向量。
        if ux == 0.0 and uy == 0.0:
            ux, uy = self._heading.get(row, (0.0, 0.0))
            if ux != 0.0 or uy != 0.0:
                dx, dy = ux, uy
        deficit = min(1.0, max(0.0, 1.0 - hunger_before))
        importance = min(1.0, max(0.0, 0.3 + 0.5 * amount_frac + 0.2 * deficit))
        feature = np.asarray((ux, uy, amount_frac, 0.0, deficit), dtype=np.float64)
        tag = "food:" + direction_bucket(dx, dy)
        self._append(row, MemoryRecord(now, KIND_ATE, tag, importance, feature))

    def _append(self, row: int, record: MemoryRecord) -> None:
        stream = self._streams.get(row)
        if stream is None:
            stream = self._new_stream()
            self._streams[row] = stream
        stream.add(record)
        self.events_recorded += 1

    # -- 内部：反思 ------------------------------------------------------
    def _maybe_reflect(self, alive_rows: np.ndarray) -> None:
        interval = float(self.cfg.reflection_every_s)
        if interval <= 0.0 or alive_rows.size == 0:
            return
        if self._time + 1e-9 < self._next_reflection_t:
            return
        minimum = int(self.cfg.reflection_min_events)
        created = 0
        for row in alive_rows.tolist():
            row = int(row)
            stream = self._streams.get(row)
            if stream is None:
                continue
            since = self._last_reflect_t.get(row, -math.inf)
            recent = [record for record in stream.records() if record.t > since]
            events = [record for record in recent if record.kind != KIND_REFLECTION]
            if len(events) < minimum:
                continue
            tag, feature, importance = aggregate_reflection(events)
            stream.add(
                MemoryRecord(self._time, KIND_REFLECTION, tag, importance, feature)
            )
            self._last_reflection[row] = feature.copy()
            self._last_reflect_t[row] = self._time
            self.events_recorded += 1
            created += 1
        if created:
            self.reflections += 1
            self.insights += created
        # 固定推进到下个 tick（即使本次无人够格也推进，避免相位漂移）
        while self._next_reflection_t <= self._time + 1e-9:
            self._next_reflection_t += interval

    # -- 内部：检索 -> 6 维特征 -----------------------------------------
    def _refresh_features(
        self,
        view: np.ndarray,
        rows: np.ndarray,
        base_obs: np.ndarray | None = None,
    ) -> None:
        if rows.size == 0:
            return
        if base_obs is None:
            base_obs = self.swarm._base_observations(view, rows)
        queries = self._queries_from_observations(base_obs)
        cfg = self.cfg
        for index, row in enumerate(rows.tolist()):
            row = int(row)
            stream = self._streams.get(row)
            if stream is None or len(stream) == 0:
                self._features[row] = _ZERO_OBS.copy()
                continue
            retrieved = stream.retrieve(
                queries[index],
                self._time,
                weights=self._weights,
                halflife=float(cfg.recency_halflife_s),
                top_k=int(cfg.top_k),
            )
            if retrieved:
                self.retrievals += 1
            self._features[row] = self._features_from_retrieved(row, retrieved)

    @staticmethod
    def _queries_from_observations(base_obs: np.ndarray) -> np.ndarray:
        """检索 query = 当前四方向食物（one-hot 方向 + 强度）+ 饥饿亏空。"""
        obs = np.asarray(base_obs, dtype=np.float64)
        count = obs.shape[0]
        queries = np.zeros((count, FEATURE_DIM), dtype=np.float64)
        if count == 0:
            return queries
        food = np.clip(obs[:, 3:7], 0.0, None)  # 不可走=-1 → 0
        best = np.argmax(food, axis=1)
        strength = np.clip(food.max(axis=1), 0.0, 1.0)
        sensed = strength > 1e-3
        units = DIRECTION_UNITS[best]
        queries[:, 0] = np.where(sensed, units[:, 0], 0.0)
        queries[:, 1] = np.where(sensed, units[:, 1], 0.0)
        queries[:, 2] = strength
        queries[:, 4] = np.clip(1.0 - obs[:, 0], 0.0, 1.0)  # 饥饿亏空
        return queries

    def _features_from_retrieved(
        self, row: int, retrieved: list[tuple[float, MemoryRecord]]
    ) -> np.ndarray:
        """把 top_k 检索结果压成 6 维观测（全部来自已存储记忆）。"""
        feature = np.zeros(OBS_DIM, dtype=np.float64)
        reflection = self._last_reflection.get(row)
        if retrieved:
            total = float(len(retrieved))
            success = sum(1 for _s, record in retrieved if record.kind == KIND_ATE)
            danger = sum(
                1 for _s, record in retrieved if record.kind == KIND_NEAR_DEATH
            )
            feature[2] = success / total
            feature[3] = danger / total
            best_direction = None
            for _score, record in retrieved:  # 已按分数降序
                if record.kind not in (KIND_ATE, KIND_REFLECTION):
                    continue
                if float(record.feature[2]) <= 0.05:
                    continue
                ux, uy = _unit_xy(float(record.feature[0]), float(record.feature[1]))
                if ux != 0.0 or uy != 0.0:
                    best_direction = (ux, uy)
                    break
            if best_direction is not None:
                feature[0], feature[1] = best_direction
            elif reflection is not None:
                ux, uy = _unit_xy(float(reflection[0]), float(reflection[1]))
                feature[0], feature[1] = ux, uy
        elif reflection is not None:
            ux, uy = _unit_xy(float(reflection[0]), float(reflection[1]))
            feature[0], feature[1] = ux, uy
        if reflection is not None:
            feature[4] = min(1.0, max(0.0, float(reflection[2])))
            feature[5] = min(1.0, max(0.0, float(reflection[3])))
        return feature

    def _update_baselines(self, view: np.ndarray, alive_rows: np.ndarray) -> None:
        for row in alive_rows.tolist():
            row = int(row)
            self._prev_hunger[row] = float(view["hunger"][row])
            position = (float(view["x"][row]), float(view["y"][row]))
            previous = self._prev_xy.get(row)
            self._prev_xy[row] = position
            if previous is not None:
                ux, uy = _unit_xy(
                    position[0] - previous[0], position[1] - previous[1]
                )
                if ux != 0.0 or uy != 0.0:
                    self._heading[row] = (ux, uy)

    # -- 统计/摘要 -------------------------------------------------------
    @property
    def stats(self) -> dict[str, Any]:
        streams = list(self._streams.values())
        total = sum(len(stream) for stream in streams)
        agents = len(streams)
        payload: dict[str, Any] = {
            "memory_obs_dim": OBS_DIM,
            "updates": self.updates,
            "events_recorded": self.events_recorded,
            "retrievals": self.retrievals,
            "reflections": self.reflections,
            "insights": self.insights,
            "mean_memories": round(total / agents, 4) if agents else 0.0,
            "total_memories": total,
            "agents_with_memory": agents,
            "reindexed": self.reindexed,
        }
        payload.update(self._feature_activity())
        return payload

    def _feature_activity(self) -> dict[str, float]:
        """特征活跃度诊断（记忆观测到底有没有信号）。"""
        if not self._features:
            return {
                "obs_food_dir_abs_mean": 0.0,
                "obs_success_mean": 0.0,
                "obs_danger_mean": 0.0,
                "obs_reflect_food_mean": 0.0,
                "obs_reflect_risk_mean": 0.0,
                "obs_food_dir_nonzero_frac": 0.0,
            }
        matrix = np.vstack(list(self._features.values()))
        return {
            "obs_food_dir_abs_mean": float(np.mean(np.abs(matrix[:, 0:2]))),
            "obs_success_mean": float(np.mean(matrix[:, 2])),
            "obs_danger_mean": float(np.mean(matrix[:, 3])),
            "obs_reflect_food_mean": float(np.mean(matrix[:, 4])),
            "obs_reflect_risk_mean": float(np.mean(matrix[:, 5])),
            "obs_food_dir_nonzero_frac": float(
                np.mean(np.hypot(matrix[:, 0], matrix[:, 1]) > 1e-6)
            ),
        }

    def summary_zh(self) -> str:
        stream_count = len(self._streams)
        mean = (
            sum(len(stream) for stream in self._streams.values()) / stream_count
            if stream_count
            else 0.0
        )
        return (
            f"记忆 {stream_count} 个体/平均 {mean:.1f} 条"
            f" 事件 {self.events_recorded} 检索 {self.retrievals}"
            f" 反思 {self.reflections} 次/{self.insights} 条洞察"
        )


# ---------------------------------------------------------------------------
# 空记忆对照层（方法学控制：obs_dim=15 但记忆观测恒 0）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NullMemorySettings(MemorySettings):
    """空记忆配置（仅用于 off_padded 对照臂）。

    enabled=True 让 experiment._build_memory 走正常构造路径，但
    zero_observation=True 使 build_memory_layer 返回 NullMemoryLayer：
    不记录事件、不检索、不反思，只注册 6 维恒零观测。用途是把
    "离线对照臂 obs_dim=9 vs 记忆臂 obs_dim=15" 的初始化尺度 confound
    控制住（W1 按 1/sqrt(obs_dim) 缩放），从而把"记忆信息"与"输入维度
    本身"分开。
    """

    zero_observation: bool = True


class NullMemoryLayer:
    """空记忆层：6 维观测恒 0，不做任何记忆/反思（off_padded 对照）。"""

    memory_obs_dim = OBS_DIM

    def __init__(self, swarm: Any, cfg: MemorySettings, *, seed: int = 0) -> None:
        self.swarm = swarm
        self.cfg = cfg
        self.seed = int(seed)
        self.updates = 0
        self._obs_dim = 0

    def attach_to_swarm(self) -> int:
        self._obs_dim = int(
            self.swarm.register_obs_provider(self._provide_observations, OBS_DIM)
        )
        return self._obs_dim

    @staticmethod
    def _provide_observations(view: np.ndarray, rows: np.ndarray) -> np.ndarray:
        count = int(np.asarray(rows).reshape(-1).size)
        return np.zeros((count, OBS_DIM), dtype=np.float64)

    def update(self, dt: float) -> None:
        self.updates += 1

    def count_of(self, row: int) -> int:
        return 0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "memory_obs_dim": OBS_DIM,
            "updates": self.updates,
            "events_recorded": 0,
            "retrievals": 0,
            "reflections": 0,
            "insights": 0,
            "mean_memories": 0.0,
            "total_memories": 0,
            "agents_with_memory": 0,
            "reindexed": 0,
            "obs_food_dir_abs_mean": 0.0,
            "obs_success_mean": 0.0,
            "obs_danger_mean": 0.0,
            "obs_reflect_food_mean": 0.0,
            "obs_reflect_risk_mean": 0.0,
            "obs_food_dir_nonzero_frac": 0.0,
        }

    def summary_zh(self) -> str:
        return "空记忆对照层（观测恒 0，不记录/不检索/不反思）"


# ---------------------------------------------------------------------------
# 工厂（experiment.build_simulation 通过它构造）
# ---------------------------------------------------------------------------
def build_memory_layer(
    swarm: Any, cfg: MemorySettings, seed: int = 0
) -> MemoryLayer | NullMemoryLayer:
    """按配置构造记忆层（不自动注册观测；由调用方 attach_to_swarm）。

    cfg 带 zero_observation=True（NullMemorySettings）时返回空记忆对照层，
    观测维数仍是 6，用于排除 obs_dim/初始化尺度 confound。
    """
    if not isinstance(cfg, MemorySettings):
        raise TypeError(f"cfg 需要 MemorySettings，得到 {type(cfg).__name__}")
    if getattr(cfg, "zero_observation", False):
        return NullMemoryLayer(swarm, cfg, seed=int(seed))
    return MemoryLayer(swarm, cfg, seed=int(seed))


__all__ = [
    "FEATURE_DIM",
    "OBS_DIM",
    "FEATURE_NAMES",
    "DIRECTION_TAGS",
    "KIND_ATE",
    "KIND_NEAR_DEATH",
    "KIND_REST",
    "KIND_REFLECTION",
    "MemoryRecord",
    "MemoryStream",
    "MemoryLayer",
    "NullMemorySettings",
    "NullMemoryLayer",
    "build_memory_layer",
    "aggregate_reflection",
    "retrieval_scores",
    "cosine_similarity",
    "direction_bucket",
    "parse_retrieval_weights",
]
