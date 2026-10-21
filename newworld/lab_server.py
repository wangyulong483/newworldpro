"""P2 实验平台后端：FastAPI + WebSocket 的仿真服务（P5 配置化）。

架构（docs/RESEARCH_ROADMAP.md §P2 接口契约）
================================================
GET  /api/snapshot        只读世界快照（地形摘要 + 人口统计 + 机制统计）
WS   /api/stream          增量帧流（个体位置/饱足 + 指标，~15Hz）
GET  /api/agent/{index}   个体详情（状态/最近动作/观测）
POST /api/command         实验控制（pause/resume/speed/step/reset）
GET  /                    前端页面（lab_static/index.html）

铁律：
1. 仿真线程是唯一的状态权威，API 层只读快照/发命令；
2. 一切干预走 /command 并记录到 interventions（人工操作不可归功于学习）；
3. 渲染层不写世界状态，点击个体只查询。

世界由 :func:`newworld.experiment.build_simulation` 按 ExperimentSettings
构造——与无窗口实验跑的是同一套代码路径（P5 的单一代码路径要求）。
默认配置是 P1.7 的稀缺环境（food_regrow=0.0005），那里存在真实的
觅食选择压力。

运行::

    python -m newworld.lab_server
    python -m newworld --lab --port 8600 --seed 42
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
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .settings import ExperimentSettings, WorldSettings

DT = 0.05
#: 稀缺世界（P1.7 的结论环境）
SCARCE_REGROW = 0.0005
DEFAULT_PORT = 8600
STREAM_HZ = 15.0
#: 个体详情保留的最近动作数
AGENT_HISTORY = 64


def default_lab_settings(seed: int = 0) -> ExperimentSettings:
    """实验平台默认配置（P1.7 稀缺世界 + P3 社会机制）。"""
    return ExperimentSettings(
        name=f"lab_seed{int(seed)}",
        protocol="lab",
        world=WorldSettings(width=256, height=160, seed=int(seed),
                            food_regrow_per_s=SCARCE_REGROW),
        policy=dataclasses.replace(
            ExperimentSettings().policy, kind="rule", seed=int(seed)
        ),
    )


class LabSession:
    """一份后台运行的仿真世界（线程安全：一把大锁足够 ~15Hz 的读写）。

    仿真在 daemon 线程里以固定 dt 步进（速度档 = 每 tick 步数）；
    指标按秒级采样保留时间序列（前端画曲线）。
    """

    def __init__(self, seed: int = 0, settings: ExperimentSettings | None = None) -> None:
        self.lock = threading.Lock()
        self.seed = int(seed)
        self.settings = settings or default_lab_settings(seed)
        #: 干预记录（时间/类型/参数——实验审计用）
        self.interventions: list[dict[str, Any]] = []
        #: 个体动作历史（"_dirty" 键 = 最近转移窗口；详情面板用）
        self.agent_history: dict[Any, list[tuple[float, int, float]]] = {}
        #: 指标时间序列（每秒一条）
        self.series: list[dict[str, float]] = []
        self._last_series_t = -1.0
        self._stop = False
        self._build(self.seed)
        self.running = False
        self.speed_index = 2  # SPEEDS[2] = 4 步/tick
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------
    def _build(self, seed: int) -> None:
        """（重）建世界：配置 -> 仿真组合（与无窗口实验同一路径）。"""
        from .experiment import build_simulation

        self.seed = int(seed)
        self.settings = dataclasses.replace(
            self.settings,
            name=f"lab_seed{self.seed}",
            world=dataclasses.replace(self.settings.world, seed=self.seed),
            policy=dataclasses.replace(self.settings.policy, seed=self.seed),
        )
        self.sim = build_simulation(self.settings, seed=self.seed)
        self.engine = self.sim.engine
        self.swarm = self.sim.swarm
        self.policy = self.sim.policy
        self.social = self.sim.social
        self.comm = self.sim.comm
        self.memory = self.sim.memory
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
                    self.sim.step()
                self._archive_recent_actions()
                self._sample_series()

    def _archive_recent_actions(self) -> None:
        """把最近转移按窗口归档（详情面板用）。"""
        transitions = self.swarm.transitions
        window = transitions[-32:]
        if window:
            self.agent_history["_dirty"] = [
                (round(self.engine.t, 2), int(action), float(reward))
                for (_obs, action, reward) in window
            ]
        # 防止缓冲无限增长（训练器不在循环里）
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
        record = {"t": round(self.engine.t, 2), "kind": kind, "params": params}
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
                    self.sim.step()
                self._sample_series()
            elif kind == "reset":
                self._build(int(params.get("seed", self.seed + 1)))
                self.running = True
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
            payload = {
                "t": round(self.engine.t, 2),
                "running": self.running,
                "speed_index": self.speed_index,
                "seed": self.seed,
                "map": {"width": self.engine.width, "height": self.engine.height},
                "terrain": self.engine.terrain.tobytes().hex(),
                "population": self.swarm.population,
                "births": self.swarm.births,
                "deaths": self.swarm.deaths,
                "satiation": round(self.swarm.mean_hunger(), 3),
                "food_total": round(self.engine.food_capacity(), 0),
                "interventions": list(self.interventions[-20:]),
                "series": list(self.series[-120:]),
                "config_hash": self.settings.hash(),
            }
            if self.social is not None:
                payload["social"] = {key: _plain(value) for key, value in self.social.stats.items()}
            if self.comm is not None:
                payload["comm"] = {key: _plain(value) for key, value in dict(self.comm.stats).items()}
            if self.memory is not None:
                payload["memory"] = {key: _plain(value) for key, value in dict(self.memory.stats).items()}
            return payload

    def frame(self) -> dict[str, Any]:
        """增量帧（WS 流推送：个体位置/饱足 + 指标）。"""
        with self.lock:
            xs, ys = self.swarm.positions_xy()
            view = self.swarm.data[: self.swarm.count]
            alive = view["alive"]
            hungers = view["hunger"][alive].tolist() if self.swarm.count else []
            payload = {
                "t": round(self.engine.t, 2),
                "running": self.running,
                "x": [round(float(v), 1) for v in xs.tolist()],
                "y": [round(float(v), 1) for v in ys.tolist()],
                "h": [round(float(v), 2) for v in hungers],
                "pop": self.swarm.population,
                "food": round(self.engine.food_capacity(), 0),
                "sat": round(self.swarm.mean_hunger(), 3),
            }
            if self.comm is not None:
                payload["comm"] = {key: _plain(value) for key, value in dict(self.comm.stats).items()}
            if self.memory is not None:
                payload["memory"] = {key: _plain(value) for key, value in dict(self.memory.stats).items()}
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
            payload: dict[str, Any] = {
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
            if self.social is not None:
                from .social_mech import SKILLS

                skill_id = int(self.social.skills[row])
                payload["skill"] = SKILLS[skill_id]["name"] if skill_id else None
            if self.comm is not None:
                payload["signal"] = _plain(self.comm.symbol_of(row))
            if self.memory is not None:
                payload["memory_n"] = _plain(self.memory.count_of(row))
            return payload

    def close(self) -> None:
        self._stop = True


def _plain(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def create_app(seed: int = 0, settings: ExperimentSettings | None = None) -> FastAPI:
    """构建实验平台 app（lab_session 挂在 app.state 上）。"""
    from contextlib import asynccontextmanager

    session = LabSession(seed=seed, settings=settings)

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
    from .settings import apply_overrides, load_config

    parser = argparse.ArgumentParser(description="newworld 实验平台服务器")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", metavar="YAML", help="用一份实验配置启动（可选）")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    settings = None
    if args.config:
        settings = apply_overrides(load_config(args.config), list(args.overrides))
        settings = dataclasses.replace(
            settings,
            world=dataclasses.replace(settings.world, seed=int(args.seed)),
            policy=dataclasses.replace(settings.policy, seed=int(args.seed)),
        )
    import uvicorn

    app = create_app(seed=args.seed, settings=settings)
    print(f"实验平台: http://127.0.0.1:{args.port}  (seed={args.seed}, 稀缺世界默认)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
