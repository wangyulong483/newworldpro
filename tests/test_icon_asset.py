"""桌面图标资产测试（由 ref/blueprint.pdf 生成，见 tools/make_icon.py）。

这些是真实入库的交付资产：Windows 桌面快捷方式指向 ``assets/newworld.ico``，
所以图标必须存在、包含常见尺寸，并且有可预览的 PNG。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ICON = ROOT / "assets" / "newworld.ico"
PREVIEW = ROOT / "assets" / "newworld_icon_256.png"


class TestIconAssets(unittest.TestCase):
    def test_icon_exists_with_windows_sizes(self) -> None:
        self.assertTrue(ICON.exists(), "assets/newworld.ico 缺失（运行 python tools/make_icon.py）")
        icon = Image.open(ICON)
        self.assertEqual(icon.format, "ICO")
        sizes = {size for size in icon.info.get("sizes", [])}
        for expected in ((16, 16), (32, 32), (48, 48), (256, 256)):
            self.assertIn(expected, sizes)

    def test_preview_is_square_rgba(self) -> None:
        self.assertTrue(PREVIEW.exists(), "assets/newworld_icon_256.png 缺失")
        preview = Image.open(PREVIEW)
        self.assertEqual(preview.size, (256, 256))
        self.assertEqual(preview.mode, "RGBA")

    def test_icon_is_not_blank(self) -> None:
        icon = Image.open(ICON)
        icon.size = (256, 256)
        rgba = icon.convert("RGBA")
        opaque = sum(1 for pixel in rgba.getdata() if pixel[3] > 0)
        self.assertGreater(opaque, 0.5 * 256 * 256, "图标应当大部分不透明（深色圆角底）")

    def test_shortcut_tool_exists(self) -> None:
        self.assertTrue((ROOT / "tools" / "update_shortcut.ps1").exists())
        self.assertTrue((ROOT / "tools" / "make_icon.py").exists())


if __name__ == "__main__":
    unittest.main()
