"""P1/P3/P4 协议冒烟测试：注册表 / YAML 配置 / run_from_file 端到端。

只做冒烟（tiny 地图、1 回合、2 seed、2 世界秒、存档关闭），断言：
* 三个协议已注册，三份 YAML 能通过严格 schema 载入并正确继承 base.yaml；
* 每个协议都能走完 run_from_file 的完整代码路径（含产物写盘）；
* summary 具备协议承诺的字段（per-arm / paired / validity / verdict 等）。

真实实验（非 tiny）由手工命令跑：python -m newworld --config configs/...
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from newworld import protocols as _protocols  # noqa: F401 - 触发协议注册
from newworld.experiment import run_from_file
from newworld.registry import PROTOCOLS
from newworld.settings import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
(PROJECT_ROOT / ".tmp").mkdir(exist_ok=True)

#: 所有协议共用的 tiny 化覆盖（地图/人口/时长/存档）
_TINY_OVERRIDES = [
    "world.width=64",
    "world.height=48",
    "world.ocean_margin=4",
    "world.noise_scale=24",
    "swarm.initial_population=20",
    "swarm.capacity_hard=200",
    "run.seconds=2.0",
    "run.eval_seconds=2.0",
    "run.save_models=false",
]

#: 每个协议：配置文件、注册名、专属 tiny 覆盖、summary 必需字段
_CASES: dict[str, dict] = {
    "p1_scarce": {
        "config": "configs/p1_scarce.yaml",
        "protocol": "learning_compare",
        "overrides": _TINY_OVERRIDES
        + [
            "protocol_params.train_episodes=1",
            "protocol_params.train_seeds=1",
            "protocol_params.episode_seconds=2.0",
            "protocol_params.eval_seeds=2",
            "protocol_params.eval_seconds=2.0",
            "protocol_params.densities=[{name: normal, population: 20}]",
        ],
        "required": [
            "name",
            "protocol",
            "per_arm",
            "paired",
            "secondary_paired",
            "gate",
            "honest_reading",
            "verdict",
            "train_log",
            "zoo_model",
            "caveats",
        ],
    },
    "p3_social": {
        "config": "configs/p3_social.yaml",
        "protocol": "social_ablation",
        "overrides": _TINY_OVERRIDES
        + [
            "protocol_params.seeds=2",
            "protocol_params.seed_start=700",
            "protocol_params.eval_seconds=2.0",
        ],
        "required": [
            "name",
            "protocol",
            "configs",
            "per_config",
            "paired_vs_control",
            "validity",
            "hypotheses",
            "honest_reading",
            "verdict",
            "caveats",
        ],
    },
    "p4_evolution": {
        "config": "configs/p4_evolution.yaml",
        "protocol": "evolution_compare",
        "overrides": _TINY_OVERRIDES
        + [
            "protocol_params.eval_seeds=2",
            "protocol_params.eval_seed_start=1100",
            "protocol_params.train_seeds=1",
            "protocol_params.episodes=1",
            "protocol_params.episode_seconds=2.0",
            "protocol_params.generations=1",
            "protocol_params.gen_seconds=2.0",
            "protocol_params.eval_seconds=2.0",
        ],
        "required": [
            "name",
            "protocol",
            "per_arm",
            "paired",
            "evolution_arms",
            "train_log",
            "honest_reading",
            "verdict",
            "zoo_model",
            "caveats",
        ],
    },
}


class ProtocolRegistrationTest(unittest.TestCase):
    def test_three_protocols_registered(self) -> None:
        for case in _CASES.values():
            self.assertIn(case["protocol"], PROTOCOLS)


class ConfigLoadTest(unittest.TestCase):
    def test_configs_load_with_base_inheritance(self) -> None:
        for name, case in _CASES.items():
            with self.subTest(config=name):
                settings = load_config(PROJECT_ROOT / case["config"])
                self.assertEqual(settings.protocol, case["protocol"])
                # base.yaml 的 256x160 世界应当被继承
                self.assertEqual(settings.world.width, 256)
                self.assertEqual(settings.world.height, 160)
                # P1.7 稀缺再生速率（三份配置都覆盖为 0.0005）
                self.assertAlmostEqual(settings.world.food_regrow_per_s, 0.0005)
                self.assertTrue(settings.protocol_params)


class RunFromFileTest(unittest.TestCase):
    def test_each_protocol_runs_and_writes_artifacts(self) -> None:
        for name, case in _CASES.items():
            with self.subTest(config=name):
                config_path = PROJECT_ROOT / case["config"]
                with tempfile.TemporaryDirectory(
                    dir=str(PROJECT_ROOT / ".tmp")
                ) as tmp:
                    out = Path(tmp) / "out"
                    summary = run_from_file(
                        config_path,
                        overrides=list(case["overrides"]),
                        directory=out,
                    )
                    for key in case["required"]:
                        self.assertIn(key, summary, f"{name} 缺少 summary[{key!r}]")
                    self.assertEqual(summary["protocol"], case["protocol"])
                    self.assertIn("_artifact_dir", summary)
                    # 产物：summary.json + config.yaml 都已写盘
                    summary_path = out / "summary.json"
                    self.assertTrue(summary_path.exists())
                    self.assertTrue((out / "config.yaml").exists())
                    payload = json.loads(summary_path.read_text(encoding="utf-8"))
                    self.assertEqual(payload["protocol"], case["protocol"])
                    self.assertIn("config_hash", payload)
                    for key in case["required"]:
                        self.assertIn(key, payload)
                    # tiny 冒烟关闭了存档：不应写 zoo
                    if "zoo_model" in payload:
                        self.assertIsNone(payload["zoo_model"])

    def test_p1_protocol_structure(self) -> None:
        case = _CASES["p1_scarce"]
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / ".tmp")) as tmp:
            summary = run_from_file(
                PROJECT_ROOT / case["config"],
                overrides=list(case["overrides"]),
                directory=Path(tmp) / "out",
            )
        self.assertEqual(summary["arms"], ["random", "rule", "untrained", "trained"])
        self.assertIn("normal", summary["per_arm"])
        self.assertIn("trained", summary["per_arm"]["normal"])
        for metric in ("eat_events", "eat_rate", "mean_life"):
            self.assertIn(metric, summary["paired"]["normal"]["trained"])
        self.assertIn("passed", summary["gate"])
        self.assertTrue(summary["caveats"])

    def test_p3_protocol_structure(self) -> None:
        case = _CASES["p3_social"]
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / ".tmp")) as tmp:
            summary = run_from_file(
                PROJECT_ROOT / case["config"],
                overrides=list(case["overrides"]),
                directory=Path(tmp) / "out",
            )
        self.assertEqual(
            list(summary["configs"]),
            ["off_all", "barter_only", "teach_only", "inherit_only", "all_on"],
        )
        for mechanism in ("barter", "teaching", "inheritance"):
            self.assertIn(mechanism, summary["validity"])
            self.assertIn("active", summary["validity"][mechanism])
        # off_all 的机制计数器必须为 0（机制关闭 = 不产生事件）
        off_all = summary["per_config"]["off_all"]
        self.assertEqual(off_all["barter_events"]["mean"], 0.0)
        self.assertEqual(off_all["teach_events"]["mean"], 0.0)
        self.assertEqual(off_all["inherit_events"]["mean"], 0.0)
        self.assertIn("population", summary["paired_vs_control"]["barter_only"])

    def test_p4_protocol_structure(self) -> None:
        case = _CASES["p4_evolution"]
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / ".tmp")) as tmp:
            summary = run_from_file(
                PROJECT_ROOT / case["config"],
                overrides=list(case["overrides"]),
                directory=Path(tmp) / "out",
            )
        self.assertEqual(summary["arms"], ["rule", "evolution", "learning"])
        for arm in ("rule", "evolution", "learning"):
            self.assertIn(arm, summary["per_arm"])
            self.assertIn("eat_events", summary["per_arm"][arm])
        for arm in ("evolution", "learning"):
            self.assertIn("eat_rate", summary["paired"][arm])
        self.assertEqual(len(summary["evolution_arms"]), 2)
        self.assertIn("theta_final", summary["evolution_arms"][0])
        self.assertIn("learning_train_eval_seeds_disjoint", summary["experiment"])


if __name__ == "__main__":
    unittest.main()
