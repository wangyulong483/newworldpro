"""P2 实验平台服务端单元测试（不依赖真实网络，用 TestClient）。

验证**API 契约与铁律**（ROADMAP §P2）：
快照只读、命令全记录（人工干预审计）、个体详情、静态页面挂载、
仿真线程的启停与确定性。
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from newworld.lab_server import DT, SCARCE_REGROW, create_app  # noqa: E402


def make_client(seed: int = 0) -> TestClient:
    """构造一个带头测试客户端（app 生命周期内仿真线程在跑）。"""
    return TestClient(create_app(seed=seed))


# ---------------------------------------------------------------------------
class SnapshotApiTest(unittest.TestCase):
    """GET /api/snapshot 只读契约。"""

    def test_snapshot_shape(self) -> None:
        """快照包含契约规定的全部字段。"""
        with make_client(seed=0) as client:
            response = client.get("/api/snapshot")
            self.assertEqual(response.status_code, 200)
            snap = response.json()
            for key in ("t", "running", "speed_index", "seed", "map",
                        "terrain", "population", "interventions", "series"):
                self.assertIn(key, snap)
            self.assertEqual(snap["map"], {"width": 256, "height": 160})
            # 地形 hex 是 uint8 网格的双倍长度
            self.assertEqual(len(snap["terrain"]), 256 * 160 * 2)

    def test_snapshot_is_read_only(self) -> None:
        """连续快照不推进世界（GET 无副作用）。"""
        with make_client(seed=0) as client:
            a = client.get("/api/snapshot").json()
            b = client.get("/api/snapshot").json()
            # 仿真线程可能推了一点时间，但人口/干预不变
            self.assertEqual(a["population"], b["population"])
            self.assertEqual(a["interventions"], b["interventions"])

    def test_scarce_environment_config(self) -> None:
        """服务端世界是稀缺配置（P1.7 结论环境）。"""
        from newworld.map_engine import MapEngine
        from newworld.lab_server import create_app
        with make_client(seed=0) as client:
            app = client.app
            self.assertAlmostEqual(
                app.state.session.engine.config.food_regrow_per_s,
                SCARCE_REGROW, places=9,
            )


# ---------------------------------------------------------------------------
class CommandApiTest(unittest.TestCase):
    """POST /api/command 干预与审计铁律。"""

    def test_unknown_command_rejected(self) -> None:
        """未知命令返回 ok=False 且不入审计。"""
        with make_client(seed=0) as client:
            response = client.post("/api/command", json={"kind": "explode"})
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertFalse(body["ok"])
            snap = client.get("/api/snapshot").json()
            self.assertEqual(snap["interventions"], [])

    def test_commands_recorded_for_audit(self) -> None:
        """全部合法命令入干预审计（时间/类型/参数）。"""
        with make_client(seed=0) as client:
            for payload in (
                {"kind": "resume"},
                {"kind": "speed", "index": 3},
                {"kind": "step", "steps": 10},
            ):
                result = client.post("/api/command", json=payload).json()
                self.assertTrue(result["ok"], payload)
                self.assertIn("t", result["record"])
                self.assertIn("kind", result["record"])
            snap = client.get("/api/snapshot").json()
            kinds = [i["kind"] for i in snap["interventions"]]
            self.assertEqual(kinds, ["resume", "speed", "step"])
            # 参数也被记录（审计可复现）
            self.assertEqual(snap["interventions"][2]["params"]["steps"], 10)

    def test_pause_resume_roundtrip(self) -> None:
        """pause 后帧流不再推进；resume 恢复。"""
        with make_client(seed=0) as client:
            client.post("/api/command", json={"kind": "resume"})
            client.post("/api/command", json={"kind": "speed", "index": 4})
            import time
            time.sleep(0.3)
            t_running = client.get("/api/snapshot").json()["t"]
            client.post("/api/command", json={"kind": "pause"})
            time.sleep(0.3)
            t_paused = client.get("/api/snapshot").json()["t"]
            self.assertGreater(t_running, 0.0)
            # 暂停后时间最多前进一点点（在途 tick 收尾），不应持续增长
            self.assertLess(t_paused - t_running, 0.5 + t_running * 0.1)

    def test_step_command_advances(self) -> None:
        """step 命令推进世界（暂停态也可单步）。"""
        with make_client(seed=0) as client:
            client.post("/api/command", json={"kind": "pause"})
            t0 = client.get("/api/snapshot").json()["t"]
            client.post("/api/command", json={"kind": "step", "steps": 100})
            t1 = client.get("/api/snapshot").json()["t"]
            self.assertAlmostEqual(t1 - t0, 100 * DT, places=6)

    def test_reset_changes_seed(self) -> None:
        """reset 重建世界（seed 递增或指定）。"""
        with make_client(seed=0) as client:
            client.post("/api/command", json={"kind": "reset", "seed": 42})
            snap = client.get("/api/snapshot").json()
            self.assertEqual(snap["seed"], 42)

    def test_social_mechanisms_live(self) -> None:
        """P4：社会机制在线（快照含 social 统计字段）。"""
        with make_client(seed=0) as client:
            client.post("/api/command", json={"kind": "resume"})
            client.post("/api/command", json={"kind": "step", "steps": 400})
            snap = client.get("/api/snapshot").json()
            self.assertIn("social", snap)
            for key in ("barter_events", "teach_events", "skilled_population"):
                self.assertIn(key, snap["social"])


# ---------------------------------------------------------------------------
class AgentApiTest(unittest.TestCase):
    """GET /api/agent/{index} 个体详情。"""

    def test_agent_detail_shape(self) -> None:
        """详情返回状态与 9 维观测。"""
        with make_client(seed=0) as client:
            response = client.get("/api/agent/0")
            self.assertEqual(response.status_code, 200)
            detail = response.json()
            self.assertTrue(detail["ok"])
            self.assertEqual(len(detail["observation"]), 9)
            for key in ("x", "y", "hunger", "age"):
                self.assertIn(key, detail)

    def test_agent_out_of_range(self) -> None:
        """越界 index 返回 ok=False（不抛 500）。"""
        with make_client(seed=0) as client:
            detail = client.get("/api/agent/99999").json()
            self.assertFalse(detail["ok"])


# ---------------------------------------------------------------------------
class StaticFrontendTest(unittest.TestCase):
    """前端页面挂载。"""

    def test_index_served(self) -> None:
        """GET / 返回 HTML（含 PixiJS 引用与标题）。"""
        with make_client(seed=0) as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            html = response.text
            self.assertIn("newworld 实验平台", html)
            self.assertIn("pixi", html.lower())


# ---------------------------------------------------------------------------
class DeterminismTest(unittest.TestCase):
    """同 seed 世界一致性。"""

    def test_same_seed_same_map(self) -> None:
        """两个同 seed 的 app 地形一致（复现性贯穿到 API 层）。"""
        with make_client(seed=7) as client:
            terrain = client.get("/api/snapshot").json()["terrain"]
        with make_client(seed=7) as client:
            terrain2 = client.get("/api/snapshot").json()["terrain"]
        self.assertEqual(terrain, terrain2)


if __name__ == "__main__":
    unittest.main()
