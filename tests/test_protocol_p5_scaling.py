"""P5 标度律协议冒烟测试：注册/配置加载/两段扫描/拟合/报告产物。

小尺寸（64x48、人口 [10,20]、episodes [1]、2 个评估 seed、5s），
确保测试在 1~2 秒内跑完；真实 256x160 扫描见 configs/p5_scaling.yaml。
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from newworld.experiment import run_experiment
from newworld.registry import PROTOCOLS
from newworld.settings import (
    ExperimentSettings,
    RunSettings,
    SocialSettings,
    SwarmSettings,
    WorldSettings,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
(PROJECT_ROOT / ".tmp").mkdir(exist_ok=True)


def tiny_settings(output_dir: Path) -> ExperimentSettings:
    """极小配置：两个 sweep 都只保留最小点数，5s 世界时间。"""
    return ExperimentSettings(
        name="p5_smoke",
        protocol="scaling_sweep",
        world=WorldSettings(
            width=64,
            height=48,
            seed=3,
            ocean_margin=4,
            noise_scale=24,
            food_regrow_per_s=0.0005,
        ),
        swarm=SwarmSettings(initial_population=20, capacity_hard=200),
        social=SocialSettings(enabled=False),
        run=RunSettings(
            seconds=5.0,
            dt=0.05,
            seeds=(0,),
            eval_seconds=5.0,
            output_dir=str(output_dir),
        ),
        protocol_params={
            "population_values": [10, 20],
            "population_seconds": 5.0,
            "population_seed": 0,
            "episode_values": [1],
            "episode_seconds": 5.0,
            "train_seeds": [1000],
            "eval_seeds": [0, 1],
            "eval_seconds": 5.0,
            "model_name": "p5_smoke_c1",
            "zoo_tags": ["p5", "smoke"],
            "save_model": True,
            "zoo_root": str(Path(output_dir) / "zoo"),
        },
    )


class ProtocolRegistrationTest(unittest.TestCase):
    def test_scaling_sweep_registered(self) -> None:
        import newworld.protocols  # noqa: F401  (触发注册)

        self.assertIn("scaling_sweep", PROTOCOLS)
        self.assertTrue(PROTOCOLS.notes().get("scaling_sweep"), "协议缺少一行说明")

    def test_real_config_loads(self) -> None:
        settings = load_config(PROJECT_ROOT / "configs" / "p5_scaling.yaml")
        self.assertEqual(settings.protocol, "scaling_sweep")
        self.assertEqual(settings.name, "p5_scaling")
        params = settings.protocol_params
        self.assertEqual(params["population_values"], [60, 120, 240, 480, 960])
        self.assertEqual(params["episode_values"], [1, 2, 4, 8, 16])
        self.assertEqual(params["eval_seeds"], [0, 1, 2])
        self.assertGreaterEqual(
            settings.swarm.capacity_hard, max(params["population_values"])
        )
        self.assertFalse(settings.social.enabled)
        self.assertFalse(settings.comm.enabled)
        self.assertFalse(settings.memory.enabled)


class ScalingSweepSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / ".tmp"))
        cls.tmp = Path(cls._tmp.name)
        cls.summary = run_experiment(tiny_settings(cls.tmp))
        cls.artifact_dir = Path(cls.summary["_artifact_dir"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_summary_shape(self) -> None:
        summary = self.summary
        self.assertEqual(summary["protocol"], "scaling_sweep")
        self.assertEqual(summary["name"], "p5_smoke")
        self.assertIn("verdict", summary)
        self.assertGreater(float(summary["wall_s"]), 0.0)
        self.assertEqual([sweep["name"] for sweep in summary["sweeps"]],
                         ["population_throughput", "learning_compute"])

    def test_both_sweeps_have_finite_fits(self) -> None:
        for sweep in self.summary["sweeps"]:
            self.assertIn("points", sweep)
            self.assertGreaterEqual(len(sweep["points"]), 1)
            fit = sweep["fit"]
            for key in ("a", "b", "r2", "r2_log", "r2_linear", "n", "valid"):
                self.assertIn(key, fit)
            self.assertTrue(math.isfinite(float(fit["b"])), f"{sweep['name']}: b 非有限")
            self.assertTrue(math.isfinite(float(fit["r2"])), f"{sweep['name']}: r2 非有限")
            self.assertTrue(math.isfinite(float(fit["a"])), f"{sweep['name']}: a 非有限")
            self.assertGreaterEqual(int(fit["n"]), 1)
            # 各分点数值必须也是有限数（summary.json 可序列化 + 可比较）
            for point in sweep["points"]:
                for value in point.values():
                    if isinstance(value, float):
                        self.assertTrue(math.isfinite(value))

    def test_population_points_and_throughput(self) -> None:
        sweep = self.summary["sweeps"][0]
        populations = [point["initial_population"] for point in sweep["points"]]
        self.assertEqual(populations, [10, 20])
        for point in sweep["points"]:
            self.assertGreater(point["throughput_steps_per_s"], 0.0)
            self.assertEqual(point["steps"], 100)  # 5s / 0.05
            self.assertGreater(point["mean_population"], 0.0)

    def test_compute_point_metrics(self) -> None:
        sweep = self.summary["sweeps"][1]
        self.assertEqual(len(sweep["points"]), 1)
        point = sweep["points"][0]
        self.assertEqual(point["episodes"], 1)
        self.assertGreater(point["compute_agent_seconds"], 0.0)
        self.assertEqual(int(point["paired"]["n"]), 2)
        self.assertGreaterEqual(point["mean_eat_events"], 0.0)
        self.assertIn("mean_eat_events", sweep["baseline"])
        self.assertIn("paired_trend", sweep)
        model = sweep["model"]
        self.assertTrue(model.get("saved"), model)
        self.assertTrue(model.get("verified"), model)
        self.assertTrue(Path(model["path"]).exists())

    def test_artifacts_written(self) -> None:
        self.assertTrue((self.artifact_dir / "summary.json").exists())
        self.assertTrue((self.artifact_dir / "config.yaml").exists())
        report = self.artifact_dir / "report.md"
        self.assertTrue(report.exists())
        text = report.read_text(encoding="utf-8")
        self.assertIn("Sweep A", text)
        self.assertIn("Sweep B", text)
        self.assertIn("不说明", text)
        payload = json.loads((self.artifact_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["protocol"], "scaling_sweep")
        self.assertEqual(len(payload["sweeps"]), 2)


if __name__ == "__main__":
    unittest.main()
