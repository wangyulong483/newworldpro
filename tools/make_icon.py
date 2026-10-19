"""Generate the application / desktop-shortcut icon from the character blueprint.

The blueprint (``ref/blueprint.pdf``) is a white A4 page containing the pixel
creature plus diagonal "试用水印" watermarks.  A PDF cannot be used directly as a
Windows shortcut icon, so this tool:

1. **extracts the embedded sprite** from the PDF (the page contains the clean
   480×480 pixel-art artwork together with its soft mask; the watermark is a
   separate embedded image and is ignored);
2. crops the sprite to its opaque bounding box;
3. composites the pixel-art creature on the app's dark rounded background
   (matches ``app_tk.py``'s theme), so it reads well on light and dark desktops;
4. writes a multi-size ``.ico`` (16–256 px) plus a PNG preview.

A page-render fallback is kept for robustness if a future PDF has no usable
embedded artwork.

Usage::

    cd newworldpro
    python tools/make_icon.py
    python tools/make_icon.py --pdf ref/blueprint.pdf --out assets/newworld.ico

Requires: pillow (project runtime dependency) and PyMuPDF (only for this asset
tool, not a runtime dependency; see requirements.txt).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = ROOT / "ref" / "blueprint.pdf"
DEFAULT_OUT = ROOT / "assets" / "newworld.ico"
DEFAULT_PREVIEW = ROOT / "assets" / "newworld_icon_256.png"

#: icon canvas size and palette (matches the dark UI theme in app_tk.py)
ICON_SIZE = 256
BG = (18, 22, 28, 255)          # #12161c
BORDER = (42, 51, 63, 255)      # #2a333f
GLOW = (226, 126, 81)           # #e27e51 (creature orange)
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _require_fitz():
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - tool-only dependency
        raise SystemExit(
            "PyMuPDF is required only for this asset tool "
            "(python -m pip install pymupdf)"
        ) from exc
    return fitz


def _chroma_key_green(image: Image.Image) -> Image.Image:
    """Make the sprite's pure-green backdrop transparent (fallback path)."""
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if g > 180 and r < 120 and b < 120:
                pixels[x, y] = (r, g, b, 0)
    return rgba


def _opaque_bbox(image: Image.Image, threshold: int = 16) -> tuple[int, int, int, int]:
    alpha = image.getchannel("A")
    box = alpha.point(lambda value: 255 if value > threshold else 0).getbbox()
    if box is None:
        raise SystemExit("artwork is fully transparent")
    return box


def _saturated_score(image: Image.Image) -> int:
    """Count strongly coloured opaque pixels (sprite has orange + blue)."""
    rgb = image.convert("RGB")
    alpha = image.getchannel("A")
    pixels = rgb.load()
    mask = alpha.load()
    width, height = rgb.size
    score = 0
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            if mask[x, y] < 32:
                continue
            r, g, b = pixels[x, y]
            high = max(r, g, b)
            low = min(r, g, b)
            if high > 0 and (high - low) / high >= 0.22:
                score += 1
    return score


def extract_artwork(pdf_path: Path) -> Image.Image:
    """Return the cleanest embedded sprite as RGBA with transparency."""
    fitz = _require_fitz()
    document = fitz.open(str(pdf_path))
    best: Image.Image | None = None
    best_score = 0
    for page in document:
        for info in page.get_images(full=True):
            xref = int(info[0])
            smask = int(info[1]) if len(info) > 1 and info[1] else 0
            try:
                pixmap = fitz.Pixmap(document, xref)
                if pixmap.n - pixmap.alpha >= 4:  # CMYK -> RGB
                    pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
                if smask:
                    mask = fitz.Pixmap(document, smask)
                    pixmap = fitz.Pixmap(pixmap, mask)
            except Exception:
                continue
            mode = "RGBA" if pixmap.alpha else "RGB"
            image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
            if image.mode != "RGBA":
                image = _chroma_key_green(image)
            score = _saturated_score(image)
            if score > best_score:
                best, best_score = image, score
    document.close()
    if best is None or best_score <= 0:
        return _render_page_fallback(pdf_path)
    return best.crop(_opaque_bbox(best))


def _render_page_fallback(pdf_path: Path) -> Image.Image:
    """Render the page and crop saturated artwork (watermark-tolerant fallback)."""
    fitz = _require_fitz()
    document = fitz.open(str(pdf_path))
    page = document[0]
    pixmap = page.get_pixmap(matrix=fitz.Matrix(6, 6), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    document.close()
    rgb = image.load()
    width, height = image.size
    min_x, min_y, max_x, max_y = width, height, -1, -1
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            r, g, b = rgb[x, y]
            high = max(r, g, b)
            low = min(r, g, b)
            if high == 0:
                continue
            if (high - low) / high >= 0.22 and not (r > 245 and g > 245 and b > 245):
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    if max_x < 0:
        raise SystemExit("no artwork found in the PDF")
    margin = 6
    return image.crop(
        (
            max(0, min_x - margin),
            max(0, min_y - margin),
            min(width, max_x + margin),
            min(height, max_y + margin),
        )
    ).convert("RGBA")


def build_icon(pdf_path: Path, out_path: Path, preview_path: Path) -> tuple[Path, Path]:
    artwork = extract_artwork(pdf_path)

    # fit the creature into the icon's inner area (pixel art: nearest-neighbour)
    inner = int(ICON_SIZE * 0.76)
    scale = min(inner / artwork.width, inner / artwork.height)
    target = (max(1, round(artwork.width * scale)), max(1, round(artwork.height * scale)))
    sprite = artwork.resize(target, Image.NEAREST)

    canvas = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (4, 4, ICON_SIZE - 4, ICON_SIZE - 4), radius=40, fill=BG, outline=BORDER, width=3
    )
    # soft warm glow behind the creature
    glow = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    gdraw.ellipse(
        (
            ICON_SIZE / 2 - inner * 0.55,
            ICON_SIZE / 2 - inner * 0.45,
            ICON_SIZE / 2 + inner * 0.55,
            ICON_SIZE / 2 + inner * 0.62,
        ),
        fill=(*GLOW, 70),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(18))
    canvas.alpha_composite(glow)

    position = (
        (ICON_SIZE - sprite.width) // 2,
        int((ICON_SIZE - sprite.height) * 0.54),
    )
    canvas.alpha_composite(sprite, position)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (4, 4, ICON_SIZE - 4, ICON_SIZE - 4), radius=40, outline=BORDER, width=3
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="ICO", sizes=[(size, size) for size in ICO_SIZES])
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_path, format="PNG")
    return out_path, preview_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the newworld icon from the blueprint PDF")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--preview", default=str(DEFAULT_PREVIEW))
    args = parser.parse_args(argv)
    out, preview = build_icon(Path(args.pdf), Path(args.out), Path(args.preview))
    print(f"icon    -> {out}  ({out.stat().st_size} bytes)")
    print(f"preview -> {preview}  ({preview.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
