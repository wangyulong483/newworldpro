"""newworld 命令行入口（P5：GUI / 实验平台 / 配置化实验三合一）。

用法::

    python -m newworld                          # 俯视地图 GUI
    python -m newworld --seed 7 --size 256x160  # 指定世界
    python -m newworld --lab --port 8600        # 浏览器实验平台
    python -m newworld --config configs/p1_scarce.yaml          # 跑配置化实验
    python -m newworld --config configs/p5_scaling.yaml --set run.seeds=[0,1] --out experiments/results/run1
    python -m newworld --info                   # 版本/依赖/已注册组件
    python -m newworld --list-components        # 策略与协议注册表

所有实验都通过 YAML 配置 + 注册表选择组件；本文件只做参数分发，
不再包含任何世界/策略实现。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .settings import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newworld",
        description="newworld — 俯视地图 + 千人群伙的开放式人工社会研究平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "GUI 快捷键：空格 开始/暂停，+/- 调速，Esc 退出。\n"
            "实验输出：experiments/results/<name>/{config.yaml,summary.json}"
        ),
    )
    parser.add_argument("--version", action="version", version=f"newworld {__version__}")

    mode = parser.add_argument_group("运行模式")
    mode.add_argument("--lab", action="store_true", help="启动浏览器实验平台（FastAPI）")
    mode.add_argument("--config", metavar="YAML", help="按配置跑一次无窗口实验")
    mode.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                      help="覆盖配置字段（可重复），例如 --set world.width=128")
    mode.add_argument("--out", metavar="DIR", help="实验产物目录（默认按 run.output_dir/name）")
    mode.add_argument("--info", action="store_true", help="打印版本/依赖/组件注册表后退出")
    mode.add_argument("--list-components", action="store_true", help="列出已注册策略与协议")

    gui = parser.add_argument_group("GUI / 实验平台")
    gui.add_argument("--seed", type=int, default=0, help="世界随机种子（默认 0）")
    gui.add_argument("--size", metavar="WxH", help="地图尺寸，例如 256x160（默认 512x320）")
    gui.add_argument("--cell-px", type=int, default=3, help="每个 cell 的屏幕像素（默认 3）")
    gui.add_argument("--port", type=int, default=8600, help="实验平台端口（默认 8600）")
    gui.add_argument("--screenshot", metavar="PATH", help="GUI 启动后截图并退出（验证用）")
    gui.add_argument("--autoclose-ms", type=int, metavar="MS", help="GUI 自动关闭毫秒数（验证用）")
    parser.add_argument("--traceback", action="store_true", help="出错时打印完整堆栈")
    return parser


# ---------------------------------------------------------------------------
def _parse_size(text: str | None, default: tuple[int, int] = (512, 320)) -> tuple[int, int]:
    if not text:
        return default
    lowered = text.lower().replace("x", " ").replace(",", " ").replace("*", " ")
    parts = [part for part in lowered.split() if part]
    if len(parts) != 2:
        raise SystemExit(f"--size 需要 WxH 形式，得到 {text!r}")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise SystemExit(f"--size 解析失败: {exc}") from exc
    if width < 32 or height < 32:
        raise SystemExit("--size 至少 32x32")
    return width, height


def run_info() -> int:
    import numpy
    import PIL

    print(f"newworld {__version__}  ({PROJECT_ROOT})")
    print(f"依赖: numpy {numpy.__version__}, pillow {PIL.__version__}")
    try:
        import yaml

        print(f"      pyyaml {yaml.__version__}")
    except Exception:  # pragma: no cover
        print("      pyyaml 未安装（配置化实验不可用）")
    try:
        import fastapi

        print(f"      fastapi {fastapi.__version__}")
    except Exception:  # pragma: no cover
        print("      fastapi 未安装（实验平台不可用）")
    print()
    print(_components_text())
    print()
    print("配置文件见 configs/；实验产出见 experiments/results/<name>/。")
    return 0


def _components_text() -> str:
    from .experiment import _import_protocols
    from .registry import POLICIES, PROTOCOLS

    _import_protocols()
    lines = ["策略（POLICIES）:"]
    for name in POLICIES.names():
        lines.append(f"  - {name:16s} {POLICIES.notes().get(name, '')}")
    lines.append("实验协议（PROTOCOLS）:")
    for name in PROTOCOLS.names():
        lines.append(f"  - {name:24s} {PROTOCOLS.notes().get(name, '')}")
    return "\n".join(lines)


def run_gui(args: argparse.Namespace) -> int:
    from .app_map import run_map_app

    width, height = _parse_size(args.size)
    return run_map_app(
        seed=int(args.seed),
        width=int(width),
        height=int(height),
        cell_px=int(args.cell_px),
        screenshot=args.screenshot,
        autoclose_ms=args.autoclose_ms,
    )


def run_lab(args: argparse.Namespace) -> int:
    from .lab_server import main as lab_main

    return lab_main(["--port", str(int(args.port)), "--seed", str(int(args.seed))])


def run_config(args: argparse.Namespace) -> int:
    from .experiment import run_from_file

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path) if not config_path.exists() else config_path
    summary = run_from_file(
        config_path,
        overrides=list(args.overrides),
        directory=args.out,
    )
    print(f"协议 {summary.get('protocol', '?')} 完成：{summary.get('name', config_path.stem)}")
    for key in ("verdict", "conclusion", "notes"):
        if key in summary:
            print(f"  {key}: {summary[key]}")
    print(f"产物: {summary.get('_artifact_dir', args.out or 'experiments/results/...')}")
    return 0


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.info:
            return run_info()
        if args.list_components:
            print(_components_text())
            return 0
        if args.config:
            return run_config(args)
        if args.lab:
            return run_lab(args)
        return run_gui(args)
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130
    except Exception as exc:  # 保持 CLI 友好
        if args.traceback:
            raise
        print(f"错误：{exc}")
        print("（使用 --traceback 查看完整堆栈）")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
