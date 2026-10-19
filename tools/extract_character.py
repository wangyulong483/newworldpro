"""Re-derive the inhabitant sprite from the blueprint PDF and verify art.py.

The blueprint (``ref/blueprint.pdf``) embeds a 480x480 PNG of the creature on a
green screen.  This tool:

1. extracts that PNG from the PDF (requires ``pymupdf``),
2. removes the green background,
3. reconstructs the 36x36 pixel-art grid (pitch 13.25 px, y offset 3.3125 px),
4. trims it to the sprite (23x29 art pixels),
5. compares the result with ``newworld/art.py`` and writes the asset PNGs.

Usage::

    python tools/extract_character.py            # verify + write assets
    python tools/extract_character.py --check    # verify only (no writes)
    python tools/extract_character.py --pdf other.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from newworld import art  # noqa: E402

GREEN = (0, 255, 0)
PALETTE = {" ": GREEN, "O": (226, 126, 81), "B": (102, 153, 208), "K": (0, 0, 0)}
PALETTE_NAMES = list(PALETTE)

PITCH = 13.25
OFFSET_X = 0.0
OFFSET_Y = 3.3125
GRID_COLS = 37
GRID_ROWS = 36


def extract_png_from_pdf(pdf_path: Path) -> Image.Image:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - tooling dependency
        raise SystemExit(
            "需要 pymupdf 才能从 PDF 提取蓝图：pip install pymupdf"
        ) from exc

    document = fitz.open(pdf_path)
    page = document[0]
    images = page.get_images(full=True)
    if not images:
        raise SystemExit(f"{pdf_path} 中没有嵌入图片")
    # the character is the largest embedded raster
    xref = max(images, key=lambda entry: entry[2] * entry[3])[0]
    payload = document.extract_image(xref)
    from io import BytesIO

    return Image.open(BytesIO(payload["image"])).convert("RGB")


def reconstruct_grid(image: Image.Image) -> tuple[list[str], tuple[int, int, int, int]]:
    """Snap the raster to the pixel-art grid and trim to the sprite."""
    pixels = np.asarray(image.convert("RGB")).astype(np.int32)
    palette = np.array([PALETTE[key] for key in PALETTE_NAMES], dtype=np.int32)
    codes = ((pixels[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(-1).argmin(-1)

    full_grid: list[str] = []
    for row in range(GRID_ROWS):
        y0 = int(round(OFFSET_Y + row * PITCH))
        y1 = int(round(OFFSET_Y + (row + 1) * PITCH))
        line = []
        for col in range(GRID_COLS):
            x0 = int(round(OFFSET_X + col * PITCH))
            x1 = int(round(OFFSET_X + (col + 1) * PITCH))
            cell = codes[max(0, y0):y1, max(0, x0):x1]
            values, counts = np.unique(cell, return_counts=True)
            line.append(PALETTE_NAMES[int(values[counts.argmax()])])
        full_grid.append("".join(line))

    occupied_rows = [index for index, line in enumerate(full_grid) if line.strip()]
    occupied_cols = [
        col for col in range(GRID_COLS) if any(full_grid[row][col] != " " for row in range(GRID_ROWS))
    ]
    row_min, row_max = min(occupied_rows), max(occupied_rows)
    col_min, col_max = min(occupied_cols), max(occupied_cols)
    trimmed = [line[col_min : col_max + 1] for line in full_grid[row_min : row_max + 1]]
    return trimmed, (row_min, row_max, col_min, col_max)


def compare_with_art(grid: list[str]) -> bool:
    expected = list(art.SPRITE_ROWS)
    if grid == expected:
        print(f"  art.py 校验通过：{len(grid)} x {len(grid[0])} 像素网格完全一致")
        return True
    print("  art.py 校验失败，差异如下：")
    for index, (actual, reference) in enumerate(zip(grid, expected)):
        if actual != reference:
            print(f"   row {index:02d} actual={actual!r}")
            print(f"          ref   ={reference!r}")
    if len(grid) != len(expected):
        print(f"  行数不同：actual {len(grid)} vs ref {len(expected)}")
    return False


def write_assets(grid: list[str], outdir: Path) -> None:
    native = Image.new("RGBA", (len(grid[0]), len(grid)), (0, 0, 0, 0))
    pixels = native.load()
    assert pixels is not None
    for y, row in enumerate(grid):
        for x, key in enumerate(row):
            colour = art.PALETTE[key]
            if colour[3]:
                pixels[x, y] = colour
    outdir.mkdir(parents=True, exist_ok=True)
    native.save(outdir / "character.png")
    native.resize((native.width * 12, native.height * 12), Image.NEAREST).save(
        outdir / "character_x12.png"
    )
    print(f"  已写入 {outdir / 'character.png'} 与 {outdir / 'character_x12.png'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从蓝图 PDF 重建 newworld 角色精灵")
    parser.add_argument("--pdf", default=str(PROJECT_ROOT / "ref" / "blueprint.pdf"))
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "assets"))
    parser.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = parser.parse_args(argv)

    pdf_path = Path(args.pdf)
    print(f"读取蓝图: {pdf_path}")
    image = extract_png_from_pdf(pdf_path)
    print(f"  提取到角色图: {image.width} x {image.height}")
    grid, bbox = reconstruct_grid(image)
    print(f"  网格裁剪范围 rows={bbox[0]}..{bbox[1]} cols={bbox[2]}..{bbox[3]} -> {len(grid)} x {len(grid[0])}")

    ok = compare_with_art(grid)
    if not ok and not args.check:
        print("  注意：将按 art.py 的网格写入资源（art.py 是运行时唯一事实来源）")
    if not args.check:
        write_assets(list(art.SPRITE_ROWS), Path(args.outdir))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
