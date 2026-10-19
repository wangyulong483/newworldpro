"""Generate the documentation preview images.

Usage::

    python tools/make_preview.py
    python tools/make_preview.py --outdir assets --size 2560x1600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from newworld.config import make_config, parse_size  # noqa: E402
from newworld.preview import generate_previews  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="渲染 newworld 预览图")
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "assets"))
    parser.add_argument("--size", help="世界尺寸，默认使用当前屏幕（例如 2560x1600）")
    args = parser.parse_args(argv)

    config = make_config(*parse_size(args.size)) if args.size else make_config()
    print(config.describe())
    paths = generate_previews(config, Path(args.outdir))
    for name, path in paths.items():
        print(f"  {name:24s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
