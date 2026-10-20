"""UI 纯逻辑层：视口、命中测试、选择、控制状态机、布局与面板数据。

本模块**禁止**使用 tkinter（契约 §1.1）：它只做坐标变换、状态机与
“从真实世界对象读数”的整理工作，因此可以在无显示环境下被测试直接导入。

约定
----
* 世界坐标 ``(wx, wy)`` 以像素为单位，``y`` 向下增长；
* 屏幕坐标 ``(sx, sy)`` 是**世界画布控件内部**的像素（不是整个窗口）；
* ``Viewport`` 的变换为 ``screen = (world - offset) * scale``，
  即 ``offset`` 是“世界左上角在视图中的偏移（世界像素）”。
* 所有面板数据都从真实对象读取；读不到的字段统一显示 :data:`NO_DATA`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

#: 缺失数据统一占位（契约 §1.1 要求）
NO_DATA = "暂无数据"

#: 缩放档位限制（契约 §1.1）
MIN_SCALE = 0.25
MAX_SCALE = 4.0

#: 速度档位（契约 §1.1）
SPEEDS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


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


def _get(obj, name: str, default=None):
    """容错的属性读取（对象 / dict 都支持，异常一律回退默认值）。"""
    if obj is None:
        return default
    try:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)
    except Exception:
        return default


def _call(obj, name: str, *args, **kwargs):
    """容错的方法调用；不存在或抛异常时返回 ``None``。"""
    method = _get(obj, name, None)
    if not callable(method):
        return None
    try:
        return method(*args, **kwargs)
    except Exception:
        return None


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


# ---------------------------------------------------------------------------
# 命中测试
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentBox:
    """一个个体在**世界坐标**里的可点击包围盒。

    ``(x, y)`` 是包围盒**中心**的世界坐标，``sprite_w / sprite_h`` 是世界像素尺寸。
    """

    agent_id: int
    x: float
    y: float
    sprite_w: float
    sprite_h: float
    alive: bool


@dataclass(frozen=True)
class AgentHit:
    """命中结果：``method`` 为 ``"bbox"``（点进包围盒）或 ``"nearest"``（最近邻）。"""

    agent_id: int
    distance: float
    method: str


class HitTester:
    """屏幕坐标 → 个体命中（契约 §1.1）。

    规则：先找屏幕点落在精灵包围盒内的**存活**个体（多个取最近）；
    没有则取中心距 ``<= max_distance_px``（屏幕像素）的最近存活个体；再没有
    返回 ``None``。死亡个体永远不参与命中。
    """

    def __init__(self, viewport: Viewport, *, max_distance_px: float = 60.0) -> None:
        """初始化。"""
        self.viewport = viewport
        self.max_distance_px = max(0.0, _num(max_distance_px, 60.0))

    def _screen_center(self, box: AgentBox) -> tuple[float, float]:
        """视口中心的屏幕坐标。"""
        return self.viewport.world_to_screen(_num(box.x), _num(box.y))

    def hit(self, agents: list[AgentBox], sx: float, sy: float) -> AgentHit | None:
        """点击命中测试。"""
        point_x = _num(sx, float("nan"))
        point_y = _num(sy, float("nan"))
        if not (math.isfinite(point_x) and math.isfinite(point_y)):
            return None
        scale = self.viewport._safe_scale()  # noqa: SLF001 - 同模块协作
        boxes = list(agents or ())
        best_box: AgentHit | None = None
        best_distance = float("inf")
        for box in boxes:
            if not _get(box, "alive", False):
                continue
            cx, cy = self._screen_center(box)
            half_w = max(0.5, _num(_get(box, "sprite_w", 0.0)) * scale / 2.0)
            half_h = max(0.5, _num(_get(box, "sprite_h", 0.0)) * scale / 2.0)
            if abs(point_x - cx) <= half_w and abs(point_y - cy) <= half_h:
                distance = math.hypot(point_x - cx, point_y - cy)
                if distance < best_distance:
                    best_distance = distance
                    best_box = AgentHit(_int(_get(box, "agent_id", -1), -1), distance, "bbox")
        if best_box is not None:
            return best_box
        best_distance = float("inf")
        best_hit: AgentHit | None = None
        for box in boxes:
            if not _get(box, "alive", False):
                continue
            cx, cy = self._screen_center(box)
            distance = math.hypot(point_x - cx, point_y - cy)
            if distance <= self.max_distance_px and distance < best_distance:
                best_distance = distance
                best_hit = AgentHit(_int(_get(box, "agent_id", -1), -1), distance, "nearest")
        return best_hit


# ---------------------------------------------------------------------------
# 选择模型
# ---------------------------------------------------------------------------
class SelectionModel:
    """当前选中的个体；只允许存活 id，死亡后由 :meth:`sync` 清理。"""

    def __init__(self) -> None:
        """初始化。"""
        self.selected_id: int | None = None
        self._alive: set[int] | None = None

    # -- 查询 ----------------------------------------------------------
    @property
    def has_selection(self) -> bool:
        """当前是否有选中个体。"""
        return self.selected_id is not None

    def alive_ids(self) -> set[int] | None:
        """最近一次 :meth:`sync` 传入的存活集合（未同步过为 ``None``）。"""
        return None if self._alive is None else set(self._alive)

    # -- 修改 ----------------------------------------------------------
    def select(self, agent_id: int, *, alive_ids: set[int] | None = None) -> bool:
        """选中 ``agent_id``；只接受存活 id，返回选择是否发生变化。"""
        target = _int(agent_id, -1)
        if target < 0:
            return False
        alive = set(alive_ids) if alive_ids is not None else self._alive
        if alive is not None and target not in alive:
            return False
        if self.selected_id == target:
            return False
        self.selected_id = target
        return True

    def toggle(self, agent_id: int, *, alive_ids: set[int] | None = None) -> bool:
        """点击切换：再点一次同一个体则取消选中。"""
        target = _int(agent_id, -1)
        if self.selected_id is not None and self.selected_id == target:
            return self.clear()
        return self.select(target, alive_ids=alive_ids)

    def clear(self) -> bool:
        """取消选中，返回状态是否发生变化。"""
        if self.selected_id is None:
            return False
        self.selected_id = None
        return True

    def sync(self, alive_ids: set[int]) -> bool:
        """同步存活集合；选中者已死亡/不存在则清空选择。返回是否被清理。"""
        alive = {_int(agent_id, -1) for agent_id in (alive_ids or ())}
        alive.discard(-1)
        self._alive = alive
        if self.selected_id is not None and self.selected_id not in alive:
            self.selected_id = None
            return True
        return False


# ---------------------------------------------------------------------------
# 控制状态机
# ---------------------------------------------------------------------------
class ControlState:
    """播放/暂停 + 速度档位 + 单步（纯状态机，不碰世界）。"""

    SPEEDS = SPEEDS

    def __init__(self, *, playing: bool = True, speed: float = 1.0) -> None:
        """初始化。"""
        self.playing = bool(playing)
        self.speed = self._snap(speed) or 1.0
        self._pending_steps = 0

    # -- 内部 ----------------------------------------------------------
    @classmethod
    def _snap(cls, speed: float) -> float | None:
        """同步选中状态到当前世界。"""
        value = _num(speed, float("nan"))
        if not math.isfinite(value):
            return None
        for candidate in cls.SPEEDS:
            if abs(value - candidate) <= 1e-6:
                return candidate
        return None

    @property
    def paused(self) -> bool:
        """是否暂停。"""
        return not self.playing

    @property
    def pending_steps(self) -> int:
        """尚未执行的单步请求数。"""
        return self._pending_steps

    @property
    def speed_index(self) -> int:
        """当前速度档位。"""
        for index, candidate in enumerate(self.SPEEDS):
            if abs(self.speed - candidate) <= 1e-6:
                return index
        return self.SPEEDS.index(1.0)

    # -- 播放/暂停 -----------------------------------------------------
    def pause(self) -> bool:
        """暂停；返回状态是否变化。"""
        if not self.playing:
            return False
        self.playing = False
        return True

    def resume(self) -> bool:
        """继续；返回状态是否变化。"""
        if self.playing:
            return False
        self.playing = True
        return True

    def toggle(self) -> bool:
        """切换播放/暂停；返回切换后的 ``playing``。"""
        if self.playing:
            self.pause()
        else:
            self.resume()
        return self.playing

    # -- 速度 ----------------------------------------------------------
    def set_speed(self, speed: float) -> bool:
        """设置速度档位；不是合法档位时返回 ``False`` 且不改变状态。"""
        snapped = self._snap(speed)
        if snapped is None:
            return False
        self.speed = snapped
        return True

    def nudge_speed(self, direction: int) -> float:
        """``-1`` 慢一档 / ``+1`` 快一档（越界保持不动），返回新的速度。"""
        step = 1 if _num(direction) > 0 else -1 if _num(direction) < 0 else 0
        index = _clamp(self.speed_index + step, 0, len(self.SPEEDS) - 1)
        self.speed = self.SPEEDS[int(index)]
        return self.speed

    # -- 单步 ----------------------------------------------------------
    def request_step(self) -> bool:
        """请求一次单步；只有暂停时接受（返回是否被接受）。"""
        if self.playing:
            return False
        self._pending_steps += 1
        return True

    def consume_step(self) -> bool:
        """取走一次待执行单步；没有待执行单步返回 ``False``。"""
        if self._pending_steps <= 0:
            return False
        self._pending_steps -= 1
        return True

    # -- 时间 ----------------------------------------------------------
    def scaled_dt(self, fixed_dt: float) -> float:
        """实际推进的世界秒：运行时为 ``speed * fixed_dt``，暂停时为 0。"""
        if not self.playing:
            return 0.0
        return max(0.0, _num(fixed_dt)) * self.speed

    def available_actions(self) -> dict[str, bool]:
        """当前可用动作（供按钮状态使用）。"""
        return {
            "start": not self.playing,
            "pause": self.playing,
            "resume": not self.playing,
            "step": not self.playing,
            "slower": self.speed_index > 0,
            "faster": self.speed_index < len(self.SPEEDS) - 1,
            "reset": True,
            "toggle_panel": True,
        }


# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------
class Layout:
    """窗口区域划分：顶部状态栏 / 世界画布 / 右侧详情 / 底部控制+日志。

    ``rects()`` 的值为 ``(x, y, width, height)``（窗口像素）。
    """

    #: 顶部状态栏高度范围
    TOP_MIN = 24
    TOP_MAX = 76
    #: 底部控制+日志高度范围
    BOTTOM_MIN = 80
    BOTTOM_MAX = 280
    #: 右侧详情面板宽度范围
    PANEL_MIN = 120
    PANEL_MAX = 460
    #: 世界画布至少保留的像素
    WORLD_MIN_W = 80
    WORLD_MIN_H = 40

    def __init__(self, width: int, height: int, *, panel_visible: bool = True) -> None:
        """初始化。"""
        self.width = max(1, _int(width, 1))
        self.height = max(1, _int(height, 1))
        self.panel_visible = bool(panel_visible)
        (self.top_h, self.bottom_h, self.panel_w) = self._measure()

    # -- 内部 ----------------------------------------------------------
    def _measure(self) -> tuple[int, int, int]:
        """测量当前文本行所需尺寸。"""
        width, height = self.width, self.height
        top_h = int(_clamp(round(height * 0.06), self.TOP_MIN, self.TOP_MAX))
        bottom_h = int(_clamp(round(height * 0.26), self.BOTTOM_MIN, self.BOTTOM_MAX))
        minimum_world = min(self.WORLD_MIN_H, max(1, height - 2))
        if top_h + bottom_h > height - minimum_world:
            spare = max(0, height - minimum_world)
            total = max(1, top_h + bottom_h)
            top_h = max(1, int(top_h * spare / total))
            bottom_h = max(1, int(bottom_h * spare / total))
        panel_w = 0
        if self.panel_visible:
            panel_w = int(_clamp(round(width * 0.28), self.PANEL_MIN, self.PANEL_MAX))
            max_panel = width - min(self.WORLD_MIN_W, max(1, width - 2))
            panel_w = max(0, min(panel_w, max_panel))
        return top_h, bottom_h, panel_w

    # -- 查询 ----------------------------------------------------------
    def rects(self) -> dict[str, tuple[int, int, int, int]]:
        """四个区域的矩形；面板隐藏时 ``panel`` 为零宽矩形。"""
        top = (0, 0, self.width, self.top_h)
        bottom_y = self.height - self.bottom_h
        bottom = (0, bottom_y, self.width, self.bottom_h)
        world_h = max(1, bottom_y - self.top_h)
        world_w = max(1, self.width - self.panel_w)
        world = (0, self.top_h, world_w, world_h)
        panel = (self.width - self.panel_w, self.top_h, self.panel_w, world_h)
        return {"top": top, "world": world, "panel": panel, "bottom": bottom}

    def region_at(self, x: float, y: float) -> str:
        """窗口坐标落在哪个区域：``top`` / ``world`` / ``panel`` / ``bottom``。

        越界坐标会被夹到窗口内（不抛错）。
        """
        px = _clamp(_num(x), 0.0, float(self.width - 1))
        py = _clamp(_num(y), 0.0, float(self.height - 1))
        if py < self.top_h:
            return "top"
        if py >= self.height - self.bottom_h:
            return "bottom"
        if self.panel_w > 0 and px >= self.width - self.panel_w:
            return "panel"
        return "world"

    def with_panel(self, visible: bool) -> "Layout":
        """带侧栏的布局矩形。"""
        return Layout(self.width, self.height, panel_visible=visible)

    def resize(self, width: int, height: int) -> "Layout":
        """窗口尺寸变化时重排。"""
        return Layout(width, height, panel_visible=self.panel_visible)


# ---------------------------------------------------------------------------
# 详情面板数据
# ---------------------------------------------------------------------------
@dataclass
class DetailRow:
    """详情面板的一行：字段名 + 值（值一律是字符串）。"""

    label: str
    value: str


@dataclass
class AgentDetail:
    """个体详情：标题 + 若干真实数据行。"""

    agent_id: int | None
    title: str
    rows: list[DetailRow] = field(default_factory=list)
    selected: bool = False


def _fmt(value, suffix: str = "", digits: int = 1) -> str:
    """格式化数字（保留一位小数）。"""
    number = _num(value, float("nan"))
    if not math.isfinite(number):
        return NO_DATA
    return f"{number:.{digits}f}{suffix}"


def _text(value) -> str:
    """安全转文本。"""
    if value is None:
        return NO_DATA
    text = str(value).strip()
    return text if text else NO_DATA


def _fmt_time(seconds) -> str:
    """把世界秒格式化为 mm:ss。"""
    value = _num(seconds, float("nan"))
    if not math.isfinite(value):
        return NO_DATA
    if value >= 120.0:
        return f"{value / 60.0:.1f} 分钟"
    return f"{value:.1f} 秒"


def _knowledge_text(world, agent) -> str:
    """已掌握知识 / 技术（读取 ``world.knowledge.get(id).status``）。"""
    ids = list(_get(agent, "known_knowledge_ids", ()) or ())
    if not ids:
        return NO_DATA
    knowledge = _get(world, "knowledge", None)
    parts: list[str] = []
    for knowledge_id in ids[:8]:
        item = _call(knowledge, "get", knowledge_id) if knowledge is not None else None
        if item is None:
            parts.append(f"{knowledge_id}(未登记)")
            continue
        name = _text(_get(item, "name_zh", knowledge_id))
        status = _text(_get(item, "status", None))
        parts.append(f"{name}({status})")
    suffix = f" 等 {len(ids)} 项" if len(ids) > 8 else ""
    return "、".join(parts) + suffix


def _inventory_text(agent) -> str:
    """库存一行的中文文本。"""
    inventory = _get(agent, "inventory", None)
    if inventory is None:
        return NO_DATA
    capacity = _call(inventory, "capacity", None)
    capacity_value = _num(capacity, float("nan"))
    pieces = [
        f"木{_int(_get(inventory, 'wood', 0))}",
        f"石{_int(_get(inventory, 'stone', 0))}",
        f"纤维{_int(_get(inventory, 'fiber', 0))}",
        f"绳{_int(_get(inventory, 'cord', 0))}",
        f"篮{_int(_get(inventory, 'basket', 0))}",
    ]
    text = " ".join(pieces)
    if math.isfinite(capacity_value):
        text += f"（容量 {capacity_value:.0f}）"
    return text


def _skill_text(agent) -> str:
    """技能一行的中文文本。"""
    skills = _get(agent, "skills", None)
    if not skills:
        return NO_DATA
    try:
        items = sorted(dict(skills).items(), key=lambda pair: _num(pair[1]), reverse=True)
    except Exception:
        return NO_DATA
    return " · ".join(f"{_text(name)} {_num(value):.2f}" for name, value in items[:5])


def _memory_text(world, agent_id: int) -> str:
    """记忆一行的中文文本。"""
    memory = _get(world, "memory", None)
    entries = _call(memory, "entries", agent_id, limit=5) if memory is not None else None
    if not entries:
        return NO_DATA
    parts = []
    for entry in entries[-4:]:
        moment = _num(_get(entry, "t", None), float("nan"))
        stamp = f"{moment:.1f}s " if math.isfinite(moment) else ""
        parts.append(f"{stamp}{_text(_get(entry, 'action', None))}→{_text(_get(entry, 'outcome', None))}")
    return "；".join(parts)


def _relation_text(world, population, agent_id: int) -> str:
    """关系一行的中文文本。"""
    society = _get(population, "society", None)
    if society is None:
        return NO_DATA
    others = [
        other.agent_id
        for other in _get(world, "agents", ()) or ()
        if _get(other, "alive", False) and _int(_get(other, "agent_id", -1)) != agent_id
    ]
    pairs = []
    for other_id in others[:8]:
        affinity = _call(society, "affinity_between", agent_id, other_id)
        if affinity is None:
            continue
        pairs.append((_num(affinity), other_id))
    if not pairs:
        return NO_DATA
    pairs.sort(reverse=True)
    return " · ".join(f"{other_id}号 {value:.2f}" for value, other_id in pairs[:3])


def _goal_rows(world, population, agent_id: int) -> list[DetailRow]:
    """当前目标 / 计划 / 进度、最近目标结果（都尽力读取，缺失显示暂无数据）。"""
    manager = _get(population, "goals", None)
    goals = _call(manager, "goals_of", agent_id) if manager is not None else None
    goals = list(goals or ())
    active = [goal for goal in goals if _get(goal, "status", "") == "active"]
    agent = _call(world, "agent_by_id", agent_id)
    current_goal_id = _get(agent, "current_goal_id", None)
    current = None
    if current_goal_id is not None:
        for goal in goals:
            if _int(_get(goal, "goal_id", -1)) == _int(current_goal_id, -2):
                current = goal
                break
    if current is None and active:
        current = active[-1]

    rows: list[DetailRow] = []
    if current is None:
        rows.append(DetailRow("当前目标", NO_DATA))
        rows.append(DetailRow("目标进度", NO_DATA))
    else:
        goal_type = _text(_get(current, "goal_type", None))
        source = _get(current, "source", None)
        rows.append(DetailRow("当前目标", f"{goal_type}（来源 {_text(source)}）"))
        progress = _num(_get(current, "progress", 0.0))
        steps_spent = _int(_get(current, "steps_spent", 0))
        rows.append(
            DetailRow(
                "目标进度",
                f"{progress * 100.0:.0f}% · 状态 {_text(_get(current, 'status', None))}"
                f" · 已用 {steps_spent} 步",
            )
        )

    plan = _call(manager, "active_plan", agent_id) if manager is not None else None
    if plan is None:
        rows.append(DetailRow("当前计划", NO_DATA))
    else:
        steps = list(_get(plan, "steps", ()) or ())
        current_step = _call(plan, "current_step", None)
        verb = _text(_get(current_step, "verb", None)) if current_step is not None else NO_DATA
        rows.append(
            DetailRow(
                "当前计划",
                f"{len(steps)} 步 · 正在执行 {verb} · 版本 {_int(_get(plan, 'version', 0))}",
            )
        )

    finished = [goal for goal in goals if _get(goal, "status", "active") != "active"]
    if not finished:
        rows.append(DetailRow("最近目标结果", NO_DATA))
    else:
        last = finished[-1]
        status = _text(_get(last, "status", None))
        goal_type = _text(_get(last, "goal_type", None))
        reason = _get(last, "reason", None) or _get(last, "rationale", None)
        if not reason:
            params = _get(last, "params", None)
            if isinstance(params, dict):
                reason = params.get("reason")
        if not reason and plan is not None:
            reason = _get(plan, "failure_reason", None)
        reason_text = f"（原因：{_text(reason)}）" if reason else ""
        rows.append(DetailRow("最近目标结果", f"{goal_type} {status}{reason_text}"))
    return rows


def build_agent_detail(world, population, agent_id: int | None) -> AgentDetail:
    """从**真实对象**构建个体详情；没有的数据一律显示 :data:`NO_DATA`。"""
    if agent_id is None:
        return AgentDetail(
            agent_id=None,
            title="未选中个体",
            rows=[
                DetailRow("提示", "在画布上点击一个个体查看详情"),
                DetailRow("操作", "左键选中/切换，再次点击取消；滚轮缩放，右键拖动平移"),
            ],
            selected=False,
        )
    agent_id = _int(agent_id, -1)
    if agent_id < 0:
        return AgentDetail(agent_id=None, title="未选中个体", rows=[DetailRow("提示", "无效的个体 id")])

    agent = _call(world, "agent_by_id", agent_id)
    registry = _get(world, "registry", None)
    record = _call(registry, "record", agent_id) if registry is not None else None
    if agent is None and record is None:
        return AgentDetail(
            agent_id=agent_id,
            title=f"{agent_id} 号 · {NO_DATA}",
            rows=[DetailRow("状态", f"找不到 {agent_id} 号个体")],
        )

    alive = bool(_get(agent, "alive", False)) if agent is not None else bool(_get(record, "alive", False))
    name = _text(_get(agent, "name", None) if agent is not None else _get(record, "name", None))
    if name == NO_DATA:
        name = f"{agent_id} 号"
    title = f"{name} · {agent_id} 号 · {'存活' if alive else '已死亡'}"

    age = _get(agent, "age", None) if agent is not None else (_get(record, "final_state", {}) or {}).get("age")
    rows: list[DetailRow] = [
        DetailRow("ID", str(agent_id)),
        DetailRow("存活", "是" if alive else "否"),
        DetailRow("年龄", _fmt_time(age)),
        DetailRow(
            "代数",
            _text(_get(agent, "generation", None) if agent is not None else _get(record, "generation", None)),
        ),
        DetailRow("父母", _parents_text(agent, record)),
    ]
    if not alive:
        died_at = _get(record, "died_at", None)
        rows.append(DetailRow("死亡时间", _fmt_time(died_at)))
        rows.append(DetailRow("死亡原因", _text(_get(record, "death_cause", None))))
        final_state = _get(record, "final_state", {}) or {}
        rows.append(DetailRow("最后行为", _text(_get(final_state, "behavior_state", None))))
        return AgentDetail(agent_id=agent_id, title=title, rows=rows)

    # 生理（能量/水分/睡眠/健康）：真实字段，缺失显示暂无数据
    physiology = _get(agent, "physiology", None)
    rows.extend(
        [
            DetailRow("行为状态", _text(_get(agent, "behavior_state", None))),
            DetailRow("位置 x", _fmt(_get(agent, "x", None))),
            DetailRow("能量", _fmt(_get(physiology, "energy", None), digits=1)),
            DetailRow("水分", _fmt(_get(physiology, "hydration", None), digits=1)),
            DetailRow("睡眠压力", _fmt(_get(physiology, "sleep_pressure", None), digits=1)),
            DetailRow("健康", _fmt(_get(physiology, "health", None), digits=1)),  # 世界模型没有该字段 → 暂无数据
            DetailRow("心/呼吸", _heart_text(physiology)),
            DetailRow("库存", _inventory_text(agent)),
            DetailRow("工具", _text(_get(_get(agent, "inventory", None), "tool", None))),
        ]
    )
    rows.extend(_goal_rows(world, population, agent_id))
    rows.extend(
        [
            DetailRow("已掌握知识", _knowledge_text(world, agent)),
            DetailRow("技能", _skill_text(agent)),
            DetailRow("记忆摘要", _memory_text(world, agent_id)),
            DetailRow("社会关系", _relation_text(world, population, agent_id)),
        ]
    )
    return AgentDetail(agent_id=agent_id, title=title, rows=rows, selected=False)


def _parents_text(agent, record) -> str:
    """父母信息文本。"""
    ids = []
    if agent is not None:
        ids = [_int(item, -1) for item in (_get(agent, "parents", ()) or ())]
        ids = [item for item in ids if item >= 0]
        if not ids:
            parent_id = _get(agent, "parent_id", None)
            ids = [int(parent_id)] if parent_id is not None else []
    elif record is not None:
        parent_id = _get(record, "parent_id", None)
        ids = [int(parent_id)] if parent_id is not None else []
    if not ids:
        return NO_DATA
    return "、".join(f"{value} 号" for value in ids)


def _heart_text(physiology) -> str:
    """心跳率文本。"""
    heart = _num(_get(physiology, "heart_rate", None), float("nan"))
    breathing = _num(_get(physiology, "breathing_rate", None), float("nan"))
    if not (math.isfinite(heart) or math.isfinite(breathing)):
        return NO_DATA
    parts = []
    if math.isfinite(heart):
        parts.append(f"心率 {heart:.0f}")
    if math.isfinite(breathing):
        parts.append(f"呼吸 {breathing:.1f}")
    return " · ".join(parts)


def build_stats(world, population) -> list[DetailRow]:
    """人口/知识/文明统计行（来自 ``world.registry / knowledge / civilization`` 等）。"""
    rows: list[DetailRow] = []
    registry = _get(world, "registry", None)
    knowledge = _get(world, "knowledge", None)
    civilization = _get(world, "civilization", None)

    t = _num(_get(world, "t", None), float("nan"))
    day = _get(world, "day", None)
    hour = _num(_get(world, "hour", None), float("nan"))
    if math.isfinite(t):
        day_index = _call(day, "day_index", t)
        day_text = f"第 {_int(day_index) + 1} 天" if day_index is not None else NO_DATA
        when = f"{day_text} {int(hour):02d}:{int((hour % 1.0) * 60):02d}" if math.isfinite(hour) else day_text
        rows.append(DetailRow("世界时间", f"{t:.1f} 秒（{when}）"))
    else:
        rows.append(DetailRow("世界时间", NO_DATA))

    alive = _call(registry, "alive_count") if registry is not None else None
    created = _get(registry, "total_created", None)
    deaths = _get(registry, "total_deaths", None)
    limit = _get(registry, "max_agents", None)
    steps = _get(world, "steps", None)
    rows.append(DetailRow("存活人口", _text(alive)))
    rows.append(DetailRow("累计创建", f"{_int(created)} / {_int(limit)}" if limit is not None else _text(created)))
    rows.append(DetailRow("死亡", _text(deaths)))
    remaining = _call(registry, "remaining") if registry is not None else None
    rows.append(DetailRow("剩余可创建", _text(remaining)))
    rows.append(DetailRow("模拟步数", _text(steps)))

    verified = _call(knowledge, "verified_ids") if knowledge is not None else None
    if verified is None:
        rows.append(DetailRow("已验证技术", NO_DATA))
    else:
        names = []
        for knowledge_id in list(verified)[:6]:
            item = _call(knowledge, "get", knowledge_id)
            names.append(_text(_get(item, "name_zh", knowledge_id)))
        rows.append(
            DetailRow("已验证技术", f"{len(verified)} 项" + (f"：{'、'.join(names)}" if names else ""))
        )
    summary = _call(knowledge, "summary_zh") if knowledge is not None else None
    rows.append(DetailRow("知识库", _text(summary)))

    facilities = list(_get(civilization, "facilities", ()) or ())
    active = sum(1 for facility in facilities if _get(facility, "active", False))
    rows.append(DetailRow("公共设施", f"{active} 座启用 / {len(facilities)} 座" if facilities else "0 座"))
    rows.append(DetailRow("合作指数", _fmt(_get(civilization, "cooperation_index", None), digits=2)))
    state = _get(civilization, "collapse_state", None)
    extinct = _get(civilization, "extinct", None)
    state_text = _text(state)
    if extinct:
        state_text += f"（已灭绝 @ {_fmt(_get(civilization, 'extinct_at', None))}）"
    rows.append(DetailRow("文明状态", state_text))
    metrics = _call(civilization, "metrics", world) if civilization is not None else None
    if isinstance(metrics, dict):
        rows.append(DetailRow("平均能量", _fmt(metrics.get("avg_energy"), digits=1)))
        rows.append(DetailRow("人口变化率", _fmt(metrics.get("population_change_ratio"), digits=3)))
    else:
        rows.append(DetailRow("平均能量", NO_DATA))

    manager = _get(population, "goals", None)
    goal_stats = _call(manager, "stats") if manager is not None else None
    if isinstance(goal_stats, dict) and goal_stats:
        rows.append(
            DetailRow(
                "目标统计",
                f"生成 {_int(goal_stats.get('goals_created'))} · "
                f"完成 {_int(goal_stats.get('goals_completed'))} · "
                f"失败 {_int(goal_stats.get('goals_failed'))} · "
                f"放弃 {_int(goal_stats.get('goals_abandoned'))}",
            )
        )
    else:
        rows.append(DetailRow("目标统计", NO_DATA))

    memory = _get(world, "memory", None)
    total_entries = 0
    counted = False
    for agent in (_get(world, "agents", ()) or ()):
        entries = _call(memory, "entries", _int(_get(agent, "agent_id", -1))) if memory is not None else None
        if entries is not None:
            total_entries += len(entries)
            counted = True
    rows.append(DetailRow("记忆条目", str(total_entries) if counted else NO_DATA))
    return rows


def build_event_log(world, population, *, limit: int = 12) -> list[str]:
    """最近的**真实**事件（社会事件 / 记忆 / 死亡 / 技术验证 / 建造）。

    数据不足时退回到由 registry 计数组成的“世界运行中”一行，保证非空。
    """
    limit = max(0, _int(limit, 12))
    if limit == 0:
        return []
    events: list[tuple[float, str]] = []
    now = _num(_get(world, "t", None), 0.0)

    society = _get(population, "society", None)
    society_events = list(_get(society, "events", ()) or ())
    for event in society_events[-4:]:
        text = _text(_get(event, "text", None))
        kind = _text(_get(event, "kind", None))
        events.append((now, f"{now:.1f}s · 社会 {kind}：{text}"))

    world_memory = _get(world, "memory", None)
    for agent in (_get(world, "agents", ()) or ()):
        if not _get(agent, "alive", False):
            continue
        agent_id = _int(_get(agent, "agent_id", -1))
        entries = _call(world_memory, "entries", agent_id, limit=4) if world_memory is not None else None
        for entry in entries or ():
            moment = _num(_get(entry, "t", None), now)
            ok = "成功" if _get(entry, "success", False) else "失败"
            events.append(
                (
                    moment,
                    f"{moment:.1f}s · {agent_id}号 {_text(_get(entry, 'action', None))}"
                    f" → {_text(_get(entry, 'outcome', None))}（{ok}）",
                )
            )

    registry = _get(world, "registry", None)
    for record in (_call(registry, "records") if registry is not None else None) or ():
        if _get(record, "alive", True):
            continue
        died_at = _num(_get(record, "died_at", None), float("nan"))
        if not math.isfinite(died_at):
            continue
        events.append(
            (
                died_at,
                f"{died_at:.1f}s · {_int(_get(record, 'agent_id', -1))}号 死亡"
                f"（{_text(_get(record, 'death_cause', None))}）",
            )
        )

    knowledge = _get(world, "knowledge", None)
    verified = _call(knowledge, "verified_ids") if knowledge is not None else None
    for knowledge_id in verified or ():
        item = _call(knowledge, "get", knowledge_id)
        discovered_at = _num(_get(item, "discovered_at", None), float("nan"))
        if not math.isfinite(discovered_at):
            continue
        events.append(
            (
                discovered_at,
                f"{discovered_at:.1f}s · 技术验证：{_text(_get(item, 'name_zh', knowledge_id))}",
            )
        )

    civilization = _get(world, "civilization", None)
    for facility in _get(civilization, "facilities", ()) or ():
        built_at = _num(_get(facility, "built_at", None), float("nan"))
        if not math.isfinite(built_at):
            continue
        events.append(
            (
                built_at,
                f"{built_at:.1f}s · 建造 {_text(_get(facility, 'name_zh', facility))}"
                f"（{_int(_get(facility, 'built_by', -1))}号）",
            )
        )

    events.sort(key=lambda item: item[0])
    lines = [text for _, text in events[-limit:]]
    if lines:
        return lines

    alive = _call(registry, "alive_count") if registry is not None else None
    fallback = (
        f"{now:.1f}s · 世界运行中 · 存活 {_text(alive)} · "
        f"累计创建 {_text(_get(registry, 'total_created', None))} · "
        f"死亡 {_text(_get(registry, 'total_deaths', None))}"
    )
    return [fallback if world is not None else NO_DATA]


__all__ = [
    "NO_DATA",
    "MIN_SCALE",
    "MAX_SCALE",
    "SPEEDS",
    "Viewport",
    "AgentBox",
    "AgentHit",
    "HitTester",
    "SelectionModel",
    "ControlState",
    "Layout",
    "DetailRow",
    "AgentDetail",
    "build_agent_detail",
    "build_stats",
    "build_event_log",
]
