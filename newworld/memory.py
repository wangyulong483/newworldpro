"""个体记忆（P3）：情节 / 程序 / 社会记忆、8 维摘要特征、代际传承与死亡档案。

契约来源：``docs/civilization_design.md`` §6（冻结接口）。本模块只依赖
标准库 + numpy，**不** import ``population`` / ``world`` / ``character`` /
``civilization`` / ``knowledge``（避免循环依赖）；对世界的所有交互都由集成层
（``population.py``）以参数形式注入，本模块只认识 ``agent_id`` 与记忆内容。

容量与逐出策略（确定公式，保证可复现）
--------------------------------------
每个 agent 最多保存 ``capacity_per_agent`` 条记忆（默认 256，最小按 1 处理）。
写入新条目后若超出容量，逐出 **1 条**，被逐出的是下面这个键最小的条目::

    victim = min(entries, key=lambda e: (e.salience, e.t, e.memory_id))

即：**先逐出显著度最低的条目；显著度相同则逐出最旧（``t`` 最小）的；仍相同
则逐出最早写入（``memory_id`` 最小）的**。全过程无随机数，同一输入序列得到
同一逐出结果。逐出只影响"保留率"，不会让 ``archive()`` 丢失历史（档案是一次性
快照，见下）。

``feature_vector`` 8 维（§6 契约：形状 ``(8,)``、``float32``、每维 ``[-1, 1]``）
-----------------------------------------------------------------------------
维度顺序与公式（``now`` = 本 store 观测到的最大时间戳，记录/学习/传承都会更新）：

0. 近期成功率：最近 ``RECENT_WINDOW=32`` 条记忆中 ``success=True`` 的比例
1. 近期失败率：``1 - 近期成功率``（近期无条目时两者都取 0.0）
2. 重复动作比例：最近条目中"与前一条 action 相同"的比例（不足 2 条 → 0.0）
3. 动作种类数 / 8：``min(1, len({action}) / 8)``（最近 ``RECENT_WINDOW`` 条）
4. 是否掌握技能：该 agent 有程序记忆（``Procedure``）→ 1.0，否则 0.0
5. 最佳技能成功率：所有 ``Procedure.success_rate()`` 的最大值（无技能 → 0.0）
6. 目标成功率：``action`` 以 ``"goal:"`` 开头的条目成功率（无 → 0.0）
7. 距上次有效进展归一化：``max(0, 1 - (now - last_success_t) / 60s)``；
   没有成功记录 → 0.0

所有维度的取值都在 ``[0, 1]`` 内（因此天然满足 ``[-1, 1]``），且只由已存内容
决定：同一输入序列必然得到同一向量；**无任何数据 → 全 0 向量**。

``archive()`` 语义（§3 / §6）
------------------------------
死亡时返回一份**可 JSON 序列化**的档案（条目、程序技能、保留率、8 维向量、
失败统计），**不删除** store 中的任何条目：调用前后 ``len(entries(agent_id))``
必须相等。档案由集成层写进 ``AgentRecord.memory_archive``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 记忆摘要特征维度（§6 冻结为 8）
MEMORY_FEATURE_DIM = 8
#: 每个 agent 的记忆容量上限（超过后按容量策略逐出）
CAPACITY_PER_AGENT = 256
#: 合法记忆种类（契约 §6：episode | procedural | social）
MEMORY_KINDS = ("episode", "procedural", "social")
#: ``feature_vector`` 里"近期"窗口的条目数
RECENT_WINDOW = 32
#: 第 7 维"距上次有效进展"的归一化时间尺度（世界秒）
PROGRESS_HORIZON_S = 60.0
#: 目标类记忆的 action 前缀（第 6 维的统计依据）
GOAL_ACTION_PREFIX = "goal:"


@dataclass
class MemoryEntry:
    """一条个体记忆（情节 / 程序 / 社会）。

    字段与 ``docs/civilization_design.md`` §6 完全一致；``action`` 与
    ``outcome`` 都是机器可读短字符串（如 ``"gather:wood"`` / ``"ok:wood+1"``），
    便于集成层统计与追责，不做自然语言解析。
    """

    memory_id: int
    agent_id: int
    t: float
    kind: str
    action: str
    outcome: str
    success: bool
    detail: str = ""
    salience: float = 0.5

    def to_dict(self) -> dict[str, object]:
        """转为可 JSON 序列化的 dict。"""
        return {
            "memory_id": int(self.memory_id),
            "agent_id": int(self.agent_id),
            "t": float(self.t),
            "kind": str(self.kind),
            "action": str(self.action),
            "outcome": str(self.outcome),
            "success": bool(self.success),
            "detail": str(self.detail),
            "salience": float(self.salience),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """从 :meth:`to_dict` 的结果还原（缺失字段用默认值，容忍旧存档）。"""
        if not isinstance(data, dict):
            raise TypeError("MemoryEntry.from_dict 需要 dict")
        return cls(
            memory_id=int(data.get("memory_id", 0)),
            agent_id=int(data.get("agent_id", 0)),
            t=float(data.get("t", 0.0)),
            kind=str(data.get("kind", "episode")),
            action=str(data.get("action", "")),
            outcome=str(data.get("outcome", "")),
            success=bool(data.get("success", False)),
            detail=str(data.get("detail", "")),
            salience=float(data.get("salience", 0.5)),
        )


@dataclass
class Procedure:
    """程序性记忆：一个技能（``skill``）及其步骤序列与成败计数。"""

    skill: str
    steps: tuple[str, ...] = ()
    successes: int = 0
    failures: int = 0
    last_used_at: float = 0.0

    def success_rate(self) -> float:
        """成功率 ``successes / (successes + failures)``；无记录返回 0.0。"""
        total = int(self.successes) + int(self.failures)
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, float(self.successes) / float(total)))

    def to_dict(self) -> dict[str, object]:
        """转为可 JSON 序列化的 dict。"""
        return {
            "skill": str(self.skill),
            "steps": [str(step) for step in self.steps],
            "successes": int(self.successes),
            "failures": int(self.failures),
            "last_used_at": float(self.last_used_at),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Procedure":
        """从 :meth:`to_dict` 的结果还原。"""
        if not isinstance(data, dict):
            raise TypeError("Procedure.from_dict 需要 dict")
        steps = data.get("steps", ())
        return cls(
            skill=str(data.get("skill", "")),
            steps=tuple(str(step) for step in steps),
            successes=int(data.get("successes", 0)),
            failures=int(data.get("failures", 0)),
            last_used_at=float(data.get("last_used_at", 0.0)),
        )


class MemoryStore:
    """所有 agent 的个体记忆仓库（§6 冻结接口）。

    一个 store 同时保存全部 agent 的记忆，按 ``agent_id`` 分区；
    ``capacity_per_agent`` 是**每个** agent 的容量上限。
    """

    def __init__(self, capacity_per_agent: int = CAPACITY_PER_AGENT) -> None:
        self.capacity_per_agent = max(1, int(capacity_per_agent))
        #: agent_id -> 记忆条目列表（写入顺序，即 t 升序）
        self._entries: dict[int, list[MemoryEntry]] = {}
        #: agent_id -> {skill: Procedure}
        self._procedures: dict[int, dict[str, Procedure]] = {}
        #: agent_id -> 历史累计写入条数（含被逐出，用于 retention）
        self._ever_recorded: dict[int, int] = {}
        #: agent_id -> 历史累计被逐出条数
        self._evicted: dict[int, int] = {}
        self._next_memory_id = 0
        #: 观测到的最大世界时间戳，供 feature_vector 第 7 维使用
        self._now = 0.0

    # ------------------------------------------------------------------
    # 写入 / 查询
    # ------------------------------------------------------------------
    def record(
        self,
        agent_id: int,
        t: float,
        kind: str,
        action: str,
        outcome: str,
        success: bool,
        detail: str = "",
        salience: float = 0.5,
    ) -> MemoryEntry:
        """写入一条记忆并返回它（超过容量时按容量策略逐出最旧最低显著度条目）。"""
        agent_id = int(agent_id)
        try:
            salience_value = float(salience)
        except (TypeError, ValueError):
            salience_value = 0.5
        salience_value = max(0.0, min(1.0, salience_value))
        entry = MemoryEntry(
            memory_id=self._next_memory_id,
            agent_id=agent_id,
            t=float(t),
            kind=str(kind),
            action=str(action),
            outcome=str(outcome),
            success=bool(success),
            detail=str(detail),
            salience=salience_value,
        )
        self._next_memory_id += 1
        return self._append_entry(entry)

    def entries(
        self,
        agent_id: int,
        *,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """按写入顺序（``t`` 升序）返回该 agent 的记忆条目。

        ``kind`` 为 ``None`` 时返回全部种类；``limit`` 为 ``None`` 时返回全部，
        否则只返回**最近** ``limit`` 条（仍按写入顺序升序排列）。
        """
        items = list(self._entries.get(int(agent_id), ()))
        if kind is not None:
            items = [entry for entry in items if entry.kind == kind]
        if limit is not None:
            limit = int(limit)
            if limit <= 0:
                return []
            items = items[-limit:]
        return items

    def recent_failures(self, agent_id: int, k: int = 5) -> list[MemoryEntry]:
        """最近 ``k`` 条失败的记忆（按写入顺序升序，最新的在最后）。"""
        k = int(k)
        if k <= 0:
            return []
        failures = [entry for entry in self._entries.get(int(agent_id), ()) if not entry.success]
        return failures[-k:]

    def failure_counts(self, agent_id: int, window: int = 32) -> dict[str, int]:
        """最近 ``window`` 条记忆里按 ``action`` 统计失败次数。"""
        window = int(window)
        if window <= 0:
            return {}
        recent = self._entries.get(int(agent_id), ())[-window:]
        counts: dict[str, int] = {}
        for entry in recent:
            if entry.success:
                continue
            counts[entry.action] = counts.get(entry.action, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # 8 维摘要特征（§6 / §7）
    # ------------------------------------------------------------------
    def feature_vector(self, agent_id: int) -> np.ndarray:
        """返回 ``(8,)``、``float32``、每维 ``[-1, 1]`` 的确定性记忆摘要。"""
        agent_id = int(agent_id)
        vector = np.zeros(MEMORY_FEATURE_DIM, dtype=np.float32)
        entries = self._entries.get(agent_id, ())
        procedures = self._procedures.get(agent_id, {})
        if not entries and not procedures:
            # 无数据（空输入）→ 全 0，契约要求
            return vector

        recent = list(entries[-RECENT_WINDOW:])
        if recent:
            successes = sum(1 for entry in recent if entry.success)
            recent_success = float(successes) / float(len(recent))
            recent_failure = 1.0 - recent_success
            repeated = sum(
                1
                for previous, current in zip(recent, recent[1:])
                if previous.action == current.action
            )
            repeat_ratio = float(repeated) / float(len(recent) - 1) if len(recent) > 1 else 0.0
            diversity = min(1.0, len({entry.action for entry in recent}) / 8.0)
        else:
            recent_success = 0.0
            recent_failure = 0.0
            repeat_ratio = 0.0
            diversity = 0.0

        if procedures:
            has_skill = 1.0
            best_skill = max(procedure.success_rate() for procedure in procedures.values())
        else:
            has_skill = 0.0
            best_skill = 0.0

        goal_entries = [entry for entry in entries if entry.action.startswith(GOAL_ACTION_PREFIX)]
        if goal_entries:
            goal_success = float(sum(1 for entry in goal_entries if entry.success)) / float(
                len(goal_entries)
            )
        else:
            goal_success = 0.0

        successful_times = [entry.t for entry in entries if entry.success]
        if successful_times:
            last_success = max(successful_times)
            recency = max(0.0, 1.0 - max(0.0, self._now - last_success) / PROGRESS_HORIZON_S)
        else:
            recency = 0.0

        vector = np.asarray(
            [
                recent_success,
                recent_failure,
                repeat_ratio,
                diversity,
                has_skill,
                best_skill,
                goal_success,
                recency,
            ],
            dtype=np.float32,
        )
        np.clip(vector, -1.0, 1.0, out=vector)
        return vector

    # ------------------------------------------------------------------
    # 程序性记忆 / 传承
    # ------------------------------------------------------------------
    def learn_procedure(self, agent_id: int, skill: str, steps: tuple[str, ...]) -> Procedure:
        """登记（或更新）一个程序技能；已存在时覆盖 ``steps`` 但保留成败计数。"""
        agent_id = int(agent_id)
        skill = str(skill)
        bucket = self._procedures.setdefault(agent_id, {})
        procedure = bucket.get(skill)
        if procedure is None:
            procedure = Procedure(skill=skill, steps=tuple(str(step) for step in steps))
            bucket[skill] = procedure
        else:
            procedure.steps = tuple(str(step) for step in steps)
        return procedure

    def procedures(self, agent_id: int) -> list[Procedure]:
        """该 agent 的全部程序技能（按技能名升序，保证可复现）。"""
        bucket = self._procedures.get(int(agent_id), {})
        return [bucket[key] for key in sorted(bucket)]

    def transfer(
        self,
        source_id: int,
        target_id: int,
        *,
        kinds: tuple[str, ...] = ("procedural",),
        limit: int = 64,
    ) -> int:
        """代际 / 师徒传承：把 ``source_id`` 指定 kinds 的记忆复制给 ``target_id``。

        只复制 ``kinds`` 中列出的记忆种类（默认只复制程序性记忆）；``limit``
        限制每类最多复制的条数（取最近的条目）。复制的条目获得新的
        ``memory_id``、``agent_id=target_id``，其余字段（t/action/outcome/
        success/detail/salience）原样保留；目标容量策略同样生效。

        当 ``"procedural"`` 在 ``kinds`` 中时，同时把源的程序技能（``Procedure``）
        合并给目标：目标缺失的整套复制；目标已有的取 ``successes`` / ``failures``
        / ``last_used_at`` 的较大值并保留目标自己的 steps。

        返回**复制的记忆条目数**（程序技能合并数不计入返回值）。
        """
        source_id = int(source_id)
        target_id = int(target_id)
        if source_id == target_id:
            return 0
        kinds = tuple(str(kind) for kind in kinds)
        limit = int(limit)
        if limit <= 0:
            return 0

        selected = [entry for entry in self._entries.get(source_id, ()) if entry.kind in kinds]
        selected = selected[-limit:]
        copied = 0
        for entry in selected:
            clone = MemoryEntry(
                memory_id=self._next_memory_id,
                agent_id=target_id,
                t=entry.t,
                kind=entry.kind,
                action=entry.action,
                outcome=entry.outcome,
                success=entry.success,
                detail=entry.detail,
                salience=entry.salience,
            )
            self._next_memory_id += 1
            self._append_entry(clone)
            copied += 1

        if "procedural" in kinds:
            target_bucket = self._procedures.setdefault(target_id, {})
            for skill, source_procedure in self._procedures.get(source_id, {}).items():
                existing = target_bucket.get(skill)
                if existing is None:
                    target_bucket[skill] = Procedure(
                        skill=source_procedure.skill,
                        steps=tuple(source_procedure.steps),
                        successes=int(source_procedure.successes),
                        failures=int(source_procedure.failures),
                        last_used_at=float(source_procedure.last_used_at),
                    )
                else:
                    existing.successes = max(existing.successes, int(source_procedure.successes))
                    existing.failures = max(existing.failures, int(source_procedure.failures))
                    existing.last_used_at = max(
                        existing.last_used_at, float(source_procedure.last_used_at)
                    )
        return copied

    # ------------------------------------------------------------------
    # 死亡档案 / 保留率
    # ------------------------------------------------------------------
    def archive(self, agent_id: int) -> dict[str, object]:
        """返回死亡档案（可 JSON 序列化），**不删除**任何条目。"""
        agent_id = int(agent_id)
        entries = self._entries.get(agent_id, [])
        procedures = self._procedures.get(agent_id, {})
        return {
            "agent_id": agent_id,
            "archived_at": float(self._now),
            "entry_count": len(entries),
            "kind_counts": self._kind_counts(agent_id),
            "entries": [entry.to_dict() for entry in entries],
            "procedures": {
                skill: procedure.to_dict() for skill, procedure in sorted(procedures.items())
            },
            "retention": self.retention(agent_id),
            "feature_vector": [float(value) for value in self.feature_vector(agent_id)],
            "failure_counts": self.failure_counts(agent_id),
        }

    def retention(self, agent_id: int) -> float:
        """记忆保留率：``当前保留条目数 / 历史累计写入条目数``；无记录 → 0.0。"""
        agent_id = int(agent_id)
        ever = int(self._ever_recorded.get(agent_id, 0))
        if ever <= 0:
            return 0.0
        kept = len(self._entries.get(agent_id, ()))
        return max(0.0, min(1.0, float(kept) / float(ever)))

    def agent_count(self) -> int:
        """有记忆或有程序技能的 agent 数量。"""
        return len(set(self._entries) | set(self._procedures))

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """导出全部状态（JSON 可序列化）。"""
        return {
            "capacity_per_agent": int(self.capacity_per_agent),
            "next_memory_id": int(self._next_memory_id),
            "now": float(self._now),
            "entries": {
                str(agent_id): [entry.to_dict() for entry in entries]
                for agent_id, entries in sorted(self._entries.items())
            },
            "procedures": {
                str(agent_id): {
                    skill: procedure.to_dict()
                    for skill, procedure in sorted(procedures.items())
                }
                for agent_id, procedures in sorted(self._procedures.items())
            },
            "ever_recorded": {str(key): int(value) for key, value in sorted(self._ever_recorded.items())},
            "evicted": {str(key): int(value) for key, value in sorted(self._evicted.items())},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryStore":
        """从 :meth:`to_dict` 的结果还原。"""
        if not isinstance(data, dict):
            raise TypeError("MemoryStore.from_dict 需要 dict")
        store = cls(capacity_per_agent=int(data.get("capacity_per_agent", CAPACITY_PER_AGENT)))
        store._now = float(data.get("now", 0.0))
        raw_entries = data.get("entries", {}) or {}
        max_id = -1
        for raw_agent_id, items in raw_entries.items():
            agent_id = int(raw_agent_id)
            bucket = store._entries.setdefault(agent_id, [])
            for item in items:
                entry = MemoryEntry.from_dict(item)
                entry.agent_id = agent_id
                bucket.append(entry)
                max_id = max(max_id, int(entry.memory_id))
        raw_procedures = data.get("procedures", {}) or {}
        for raw_agent_id, items in raw_procedures.items():
            agent_id = int(raw_agent_id)
            bucket = store._procedures.setdefault(agent_id, {})
            for raw_skill, item in items.items():
                procedure = Procedure.from_dict(item)
                bucket[str(raw_skill)] = procedure
        store._ever_recorded = {
            int(key): int(value) for key, value in (data.get("ever_recorded", {}) or {}).items()
        }
        store._evicted = {int(key): int(value) for key, value in (data.get("evicted", {}) or {}).items()}
        store._next_memory_id = max(
            int(data.get("next_memory_id", 0)), max_id + 1
        )
        return store

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _append_entry(self, entry: MemoryEntry) -> MemoryEntry:
        """追加条目、更新计数与时间戳，并执行容量逐出。"""
        agent_id = int(entry.agent_id)
        bucket = self._entries.setdefault(agent_id, [])
        bucket.append(entry)
        self._ever_recorded[agent_id] = self._ever_recorded.get(agent_id, 0) + 1
        self._now = max(self._now, float(entry.t))
        self._evict(agent_id)
        return entry

    def _evict(self, agent_id: int) -> None:
        """按 ``(salience, t, memory_id)`` 升序逐出，直到不超容量。"""
        bucket = self._entries.get(agent_id)
        if not bucket:
            return
        while len(bucket) > self.capacity_per_agent:
            victim_index = min(
                range(len(bucket)),
                key=lambda index: (
                    float(bucket[index].salience),
                    float(bucket[index].t),
                    int(bucket[index].memory_id),
                ),
            )
            bucket.pop(victim_index)
            self._evicted[agent_id] = self._evicted.get(agent_id, 0) + 1

    def _kind_counts(self, agent_id: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self._entries.get(int(agent_id), ()):
            counts[entry.kind] = counts.get(entry.kind, 0) + 1
        return counts
