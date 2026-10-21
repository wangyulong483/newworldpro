#!/usr/bin/env python
"""统一实验入口（P5）：一份 YAML = 一次可复现实验。

用法::

    python experiments/run.py --config configs/p1_scarce.yaml
    python experiments/run.py --config configs/p5_scaling.yaml --set run.seeds=[0,1] --out experiments/results/my_run
    python experiments/run.py --list                 # 列出注册的策略/协议
    python experiments/run.py --config ... --json    # 只打印 summary JSON

产物：<out 或 run.output_dir/name>/{config.yaml,summary.json}；
协议自己还可以写 report.md / 模型到 assets/zoo。

旧版 12 个 run_*.py 脚本已删除：所有实验现在走 registry + YAML 的
同一条代码路径（run_from_file -> PROTOCOLS[name]）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from newworld.experiment import run_from_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="newworld 配置化实验入口")
    parser.add_argument("--config", metavar="YAML", help="实验配置文件")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="覆盖配置字段（可重复）")
    parser.add_argument("--out", metavar="DIR", help="产物目录")
    parser.add_argument("--list", action="store_true", help="列出已注册策略/协议")
    parser.add_argument("--json", action="store_true", help="打印完整 summary JSON")
    return parser


def list_components() -> int:
    from newworld.experiment import _import_protocols
    from newworld.registry import POLICIES, PROTOCOLS

    _import_protocols()
    print("策略:")
    for name in POLICIES.names():
        print(f"  {name:16s} {POLICIES.notes().get(name, '')}")
    print("协议:")
    for name in PROTOCOLS.names():
        print(f"  {name:24s} {PROTOCOLS.notes().get(name, '')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        return list_components()
    if not args.config:
        build_parser().print_help()
        return 2
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = PROJECT_ROOT / config_path
        config_path = candidate if candidate.exists() else config_path
    summary = run_from_file(config_path, overrides=list(args.overrides), directory=args.out)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    print(f"实验 {summary.get('name', config_path.stem)} 完成（协议 {summary.get('protocol', '?')}）")
    if "verdict" in summary:
        print(f"结论: {summary['verdict']}")
    print(f"产物: {summary.get('_artifact_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
