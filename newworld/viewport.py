"""视口：世界↔屏幕坐标变换 + 缩放 / 平移（纯逻辑层，无 tkinter）。

本模块只保留地图 GUI 需要的 :class:`Viewport`：坐标变换、滚轮缩放、
拖动平移与边界约束。它不依赖 tkinter，因此可以在无显示环境下被测试
直接导入。

约定
----
* 世界坐标 ``(wx, wy)`` 以像素为单位，``y`` 向下增长；
* 屏幕坐标 ``(sx, sy)`` 是**世界画布控件内部**的像素（不是整个窗口）；
* 变换为 ``screen = (world - offset) * scale``，
  即 ``offset`` 是"世界左上角在视图中的偏移（世界像素）"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

#: 缩放档位限制
MIN_SCALE = 0.25
MAX_SCALE = 4.0


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _num(value, default: float = 0.0) -> float:
    """把任意输入安全地转成有限 float（异常/NaN/inf 回退默认值）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number


def _clamp(value: float, low: float, high: float) -> float:
    """把 value 夹在 [low, high]。"""
    if low > high:
        low, high = high, low
    return low if value < low else high if value > high else value


def _int(value, default: int = 0) -> int:
    """安全转 int，失败给 default。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


# ---------------------------------------------------------------------------
# 视口
# ---------------------------------------------------------------------------
@dataclass
class Viewport:
    """世界↔屏幕坐标变换 + 缩放/平移（所有变换方法都返回新实例）。

    ``offset_x / offset_y`` 是世界坐标在视图中的左上角偏移（世界像素）：
    ``screen = (world - offset) * scale``。
    """

    world_width: int
    world_height: int
    view_width: int
    view_height: int
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    # -- 基本变换 ------------------------------------------------------
    def _safe_scale(self) -> float:
        """受保护的世界→屏幕缩放系数。"""
        return max(MIN_SCALE * 1e-3, _num(self.scale, 1.0))

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        """屏幕坐标转世界坐标。"""
        scale = self._safe_scale()
        return (_num(sx) / scale + _num(self.offset_x), _num(sy) / scale + _num(self.offset_y))

    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        """世界坐标转屏幕坐标。"""
        scale = self._safe_scale()
        return ((_num(wx) - _num(self.offset_x)) * scale, (_num(wy) - _num(self.offset_y)) * scale)

    # -- 缩放 / 平移 ---------------------------------------------------
    def zoom_at(self, sx: float, sy: float, factor: float) -> "Viewport":
        """以屏幕点 ``(sx, sy)`` 为锚点缩放：光标下的世界点保持不动。"""
        factor = _num(factor, 1.0)
        if factor <= 0.0:
            factor = 1.0
        scale = _clamp(_num(self.scale, 1.0) * factor, MIN_SCALE, MAX_SCALE)
        anchor = self.screen_to_world(sx, sy)
        zoomed = replace(self, scale=scale)
        zoomed.offset_x = anchor[0] - _num(sx) / scale
        zoomed.offset_y = anchor[1] - _num(sy) / scale
        return zoomed.clamp()

    def pan(self, dx_screen: float, dy_screen: float) -> "Viewport":
        """按屏幕像素拖动画面：内容跟随鼠标移动 ``(dx_screen, dy_screen)``。"""
        scale = self._safe_scale()
        return replace(
            self,
            offset_x=_num(self.offset_x) - _num(dx_screen) / scale,
            offset_y=_num(self.offset_y) - _num(dy_screen) / scale,
        )

    # -- 约束 ----------------------------------------------------------
    def clamp(self) -> "Viewport":
        """世界不被拖出视图；``scale`` 限制在 ``[0.25, 4.0]``。

        世界在当前缩放下比视图小的时候居中显示（允许出现留白），
        否则把空闲边限制为 0（世界边缘不会跑进视图内部）。
        """
        scale = _clamp(_num(self.scale, 1.0), MIN_SCALE, MAX_SCALE)
        view_w = max(1.0, _num(self.view_width, 1.0))
        view_h = max(1.0, _num(self.view_height, 1.0))
        world_w = max(1.0, _num(self.world_width, 1.0))
        world_h = max(1.0, _num(self.world_height, 1.0))

        visible_w = view_w / scale
        visible_h = view_h / scale
        if world_w <= visible_w:
            offset_x = (world_w - visible_w) / 2.0
        else:
            offset_x = _clamp(_num(self.offset_x), 0.0, world_w - visible_w)
        if world_h <= visible_h:
            offset_y = (world_h - visible_h) / 2.0
        else:
            offset_y = _clamp(_num(self.offset_y), 0.0, world_h - visible_h)
        return Viewport(
            world_width=_int(self.world_width, 1),
            world_height=_int(self.world_height, 1),
            view_width=_int(self.view_width, 1),
            view_height=_int(self.view_height, 1),
            scale=scale,
            offset_x=offset_x,
            offset_y=offset_y,
        )

    def resize(self, view_width: int, view_height: int) -> "Viewport":
        """改变视图尺寸：保持世界中心仍位于视图中心（随后 clamp）。"""
        new_w = max(1, _int(view_width, _int(self.view_width, 1)))
        new_h = max(1, _int(view_height, _int(self.view_height, 1)))
        scale = _clamp(_num(self.scale, 1.0), MIN_SCALE, MAX_SCALE)
        center = self.screen_to_world(_num(self.view_width) / 2.0, _num(self.view_height) / 2.0)
        resized = Viewport(
            world_width=_int(self.world_width, 1),
            world_height=_int(self.world_height, 1),
            view_width=new_w,
            view_height=new_h,
            scale=scale,
            offset_x=center[0] - new_w / 2.0 / scale,
            offset_y=center[1] - new_h / 2.0 / scale,
        )
        return resized.clamp()

    def visible_world_rect(self) -> tuple[float, float, float, float]:
        """当前可见的世界矩形 ``(x0, y0, x1, y1)``（世界像素）。"""
        scale = self._safe_scale()
        x0 = _num(self.offset_x)
        y0 = _num(self.offset_y)
        return (x0, y0, x0 + _num(self.view_width) / scale, y0 + _num(self.view_height) / scale)
