"""P2 实验平台后端：FastAPI + WebSocket 的仿真服务。

架构（docs/RESEARCH_ROADMAP.md §P2 接口契约）
================================================
``GET  /api/snapshot``   只读世界快照（地形摘要 + 人口统计 + 个体位图）
``WS   /api/stream``     增量帧流（个体位置/饱足 + 事件 + 指标，~15Hz）
``GET  /api/agent/{id}``  个体详情（状态/最近动作/累计奖励）
``POST /api/command``    实验干预（pause/resume/speed/seed/reset/catastrophe）
``GET  /``               前端页面（static/index.html）

铁律（与 ROADMAP 一致）：
1. 仿真线程是**唯一的状态权威**，API 层只读快照/发命令；
2. 一切干预走 /command 并记录（时间/类型/参数）到 interventions 列表，
   人工操作造成的成果不可归功于智能体学习；
3. 渲染层（前端）不写世界状态，点击个体只查询。

世界配置：P1.7 稀缺环境（food_regrow=0.0005）——那里有真实的
觅食选择压力，是学习实验的正确舞台。

运行::

    python -m newworld.lab_server              # http://127.0.0.1:8600
    python -m newworld.lab_server --port 9000 --seed 42
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .map_engine import MapConfig, MapEngine
from .swarm import Swarm, SwarmConfig
from .swarm_policy import RuleSwarmPolicy
from .catastrophes import Catastrophes

DT = 0.05
#: 稀缺世界（P1.7 的结论环境）
SCARCE_REGROW = 0.0005
DEFAULT_PORT = 8600
STREAM_HZ = 15.0

#: 个体详情保留的最近动作数
AGENT_HISTORY = 64


class LabSession:
    """一份后台运行的仿真世界（线程安全：一把大锁足够 ~15Hz 的读写）。

    仿真在 daemon 线程里以固定 dt 步进（速度档 = 每 tick 步数）；
    指标按秒级采样保留时间序列（前端画曲线）。
    """

    def __init__(self, seed: int = 0) -> None:
        self.lock = threading.Lock()
        self.seed = int(seed)
        #: 干预记录（时间/类型/参数——实验审计用）
        self.interventions: list[dict[str, Any]] = []
        #: 个体动作历史（"_dirty" 键 = 最近转移窗口；详情面板用）
        self.agent_history: dict[Any, list[tuple[float, int, float]]] = {}
        #: 指标时间序列（每秒一条）
        self.series: list[dict[str, float]] = []
        self._last_series_t = -1.0
        self._stop = False
        self._build(seed)
        self.running = False
        self.speed_index = 2  # SPEEDS[2] = 4 步/tick
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------
    def _build(self, seed: int) -> None:
        """（重）建世界：稀缺地图 + 群伙 + 规则策略 + 灾害。"""
        self.seed = int(seed)
        map_config = dataclasses.replace(
            MapConfig(seed=self.seed, width=256, height=160),
            food_regrow_per_s=SCARCE_REGROW,
        )
        self.engine = MapEngine(map_config)
        self.swarm = Swarm(
            self.engine,
            SwarmConfig(initial_population=120, capacity_hard=3000),
            seed=self.seed,
        )
        # P2 默认规则策略（学习策略挂载接口在 /command 里：attach-trained）
        self.policy = RuleSwarmPolicy(seed=self.seed)
        self.swarm.attach_policy(self.policy)
        self.cats = Catastrophes(self.engine, self.swarm)
        self.agent_history.clear()
        self.series = []
        self._last_series_t = -1.0

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        """仿真主循环（daemon 线程）。"""
        speeds = (1, 2, 4, 8, 16)
        last = time.perf_counter()
        period = 1.0 / STREAM_HZ
        while not self._stop:
            now = time.perf_counter()
            if now - last < period:
                time.sleep(max(0.0, period - (now - last)))
                continue
            last = now
            if not self.running:
                continue
            with self.lock:
                steps = speeds[self.speed_index]
                for _ in range(steps):
                    self.engine.update(DT, 0.8)
                    self.swarm.update(DT)
                self._archive_recent_actions()
                self._sample_series()

    def _archive_recent_actions(self) -> None:
        """把本 tick 的转移按个体行号归档（详情面板用，只留窗口）。"""
        # transitions 是 (obs, action, reward) 的平铺流——我们无法从平铺流
        # 恢复行号，因此在 Swarm 里 drain 到本地缓冲前先复制末尾窗口；
        # 更简单的做法：让 Swarm 在 attach 状态下不 drain（我们定期取末尾）。
        # 这里采用"读后截断"：取全部 transitions 的末尾，按帧边界近似归档。
        # P2 一期：个体详情展示观测与状态为主，动作历史记全局统计。
        transitions = self.swarm.transitions
        window = transitions[-32:]
        if window and not self.agent_history.get("_dirty"):
            self.agent_history["_dirty"] = [(round(self.engine.t, 2), int(a), float(r))
                                            for (_o, a, r) in window]
        # 防止缓冲无限增长（不 drain：训练器不在循环里）
        if len(transitions) > 200_000:
            del transitions[:100_000]

    def _sample_series(self) -> None:
        """秒级指标采样（人口/食物/出生/死亡/饱足）。"""
        t = self.engine.t
        if t - self._last_series_t < 1.0:
            return
        self._last_series_t = t
        self.series.append({
            "t": round(t, 1),
            "population": float(self.swarm.population),
            "food": round(self.engine.food_capacity(), 0),
            "births": float(self.swarm.births),
            "deaths": float(self.swarm.deaths),
            "satiation": round(self.swarm.mean_hunger(), 3),
        })

    # ------------------------------------------------------------------
    # 命令（唯一写路径）
    # ------------------------------------------------------------------
    def command(self, kind: str, **params: Any) -> dict[str, Any]:
        """执行实验干预（全部记录在案）。"""
        record = {
            "t": round(self.engine.t, 2),
            "kind": kind,
            "params": params,
        }
        with self.lock:
            if kind == "pause":
                self.running = False
            elif kind == "resume":
                self.running = True
            elif kind == "speed":
                index = int(params.get("index", self.speed_index))
                self.speed_index = max(0, min(4, index))
            elif kind == "step":
                for _ in range(max(1, int(params.get("steps", 1)))):
                    self.engine.update(DT, 0.8)
                    self.swarm.update(DT)
                self._sample_series()
            elif kind == "reset":
                self._build(int(params.get("seed", self.seed + 1)))
                self.running = True
            elif kind == "catastrophe":
                name = str(params.get("name", ""))
                x = int(params.get("x", 0)); y = int(params.get("y", 0))
                radius = int(params.get("radius", 12))
                if name == "meteor":
                    self.cats.meteor(x, y, radius)
                elif name == "lightning":
                    self.cats.lightning(x, y)
                elif name == "flood":
                    self.cats.flood(x, y, radius)
                elif name == "heal":
                    self.cats.heal(x, y, radius)
                else:
                    return {"ok": False, "error": f"unknown catastrophe {name}"}
            else:
                return {"ok": False, "error": f"unknown command {kind}"}
            self.interventions.append(record)
            if len(self.interventions) > 500:
                del self.interventions[:250]
            return {"ok": True, "record": record}

    # ------------------------------------------------------------------
    # 只读快照
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """完整快照（前端初次加载/重置用）。"""
        with self.lock:
            xs, ys = self.swarm.positions_xy()
            view = self.swarm.data[: self.swarm.count]
            hungers = view["hunger"][view["alive"]] if self.swarm.count else np.zeros(0)
            return {
                "t": round(self.engine.t, 2),
                "running": self.running,
                "speed_index": self.speed_index,
                "seed": self.seed,
                "map": {"width": self.engine.width, "height": self.engine.height},
                "terrain": self.engine.terrain.tobytes().hex(),  # 压缩地形（uint8 网格 hex）
                "population": self.swarm.population,
                "births": self.swarm.births,
                "deaths": self.swarm.deaths,
                "satiation": round(self.swarm.mean_hunger(), 3),
                "food_total": round(self.engine.food_capacity(), 0),
                "interventions": list(self.interventions[-20:]),
                "series": list(self.series[-120:]),
            }

    def frame(self) -> dict[str, Any]:
        """增量帧（WS 流推送：个体位置/饱足 + 事件摘要 + 最新指标）。"""
        with self.lock:
            xs, ys = self.swarm.positions_xy()
            view = self.swarm.data[: self.swarm.count]
            alive = view["alive"]
            hungers = view["hunger"][alive].tolist() if self.swarm.count else []
            # 编码为三个并行数组（JSON 更小更快）
            payload = {
                "t": round(self.engine.t, 2),
                "running": self.running,
                "x": [round(float(v), 1) for v in xs.tolist()],
                "y": [round(float(v), 1) for v in ys.tolist()],
                "h": [round(float(v), 2) for v in hungers],
                "pop": self.swarm.population,
                "food": round(self.engine.food_capacity(), 0),
                "burn": int(self.engine.burning.sum()),
                "sat": round(self.swarm.mean_hunger(), 3),
            }
            return payload

    def agent_detail(self, index: int) -> dict[str, Any]:
        """个体详情（详情面板；index = 快照里 x/y 数组的下标）。"""
        with self.lock:
            view = self.swarm.data[: self.swarm.count]
            alive = np.nonzero(view["alive"])[0]
            if not (0 <= index < alive.size):
                return {"ok": False, "error": "index out of range"}
            row = int(alive[index])
            entry = view[row]
            obs = self.swarm._observations(view, np.asarray([row]))[0]
            recent = self.agent_history.get("_dirty", [])[-AGENT_HISTORY:]
            return {
                "ok": True,
                "row": row,
                "x": round(float(entry["x"]), 1),
                "y": round(float(entry["y"]), 1),
                "hunger": round(float(entry["hunger"]), 3),
                "age": round(float(entry["age"]), 1),
                "breed_cd": round(float(entry["breed_cd"]), 1),
                "alive": True,
                "observation": [round(float(v), 3) for v in obs.tolist()],
                "recent_actions": recent,
            }

    def close(self) -> None:
        self._stop = True


def create_app(seed: int = 0) -> FastAPI:
    """构建实验平台 app（lab_session 挂在 app.state 上）。"""
    from contextlib import asynccontextmanager

    session = LabSession(seed=seed)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        session.close()

    app = FastAPI(title="newworld 实验平台", lifespan=lifespan)
    app.state.session = session
    static_dir = Path(__file__).resolve().parent.parent / "lab_static"

    @app.get("/api/snapshot")
    def snapshot() -> JSONResponse:
        return JSONResponse(session.snapshot())

    @app.post("/api/command")
    def command(body: dict[str, Any]) -> JSONResponse:
        kind = str(body.pop("kind", ""))
        result = session.command(kind, **body)
        return JSONResponse(result)

    @app.get("/api/agent/{index}")
    def agent(index: int) -> JSONResponse:
        return JSONResponse(session.agent_detail(index))

    @app.websocket("/api/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                await ws.send_json(session.frame())
                await asyncio_sleep(1.0 / STREAM_HZ)
        except WebSocketDisconnect:
            pass

    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


async def asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：python -m newworld.lab_server [--port 8600] [--seed 0]。"""
    parser = argparse.ArgumentParser(description="newworld 实验平台服务器")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import uvicorn

    app = create_app(seed=args.seed)
    print(f"实验平台: http://127.0.0.1:{args.port}  (seed={args.seed}, 稀缺世界)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
