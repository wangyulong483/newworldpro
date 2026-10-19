"""鼠标主导的观察窗口（newworld UI）。

布局
----
::

    ┌──────────────────────────────────────────────────────────────┐
    │ 顶部状态栏：时间 / 运行状态 / 累计创建 / 存活 / 死亡 / 剩余名额 │
    ├───────────────────────────────────────────┬──────────────────┤
    │                                           │ 个体列表（可点击）│
    │           中央世界画布（可缩放/平移）      │ 选中个体详情      │
    │                                           │ 世界统计（可折叠）│
    ├───────────────────────────────────────────┴──────────────────┤
    │ 底部控制栏：开始/暂停/继续/单步/重置 · 速度 · 统计/面板 · 训练 │
    │ 事件日志（真实世界事件 + 操作记录）                            │
    └──────────────────────────────────────────────────────────────┘

交互原则（契约 docs/upgrade_contracts_v1.md §1.2）
------------------------------------------------
* **鼠标是唯一必需入口**：左键点选/切换/取消选中个体，滚轮缩放（光标锚点），
  右键或中键拖动平移，按钮完成所有仿真控制；键盘（空格/F1/F2/Q）只是辅助。
* 选中的个体用画布上的**高亮环**标出：只改显示，不暂停仿真、不改行为。
* UI 点击与世界点击隔离：布局由 :class:`~newworld.ui_state.Layout` 划分，画布只
  覆盖 ``world`` 区域，且处理函数仍用 ``Layout.region_at`` 复核。
* 面板数据全部来自真实世界对象（``world.registry / knowledge / civilization /
  memory``、``population.goals / society``），缺失显示“暂无数据”。
* 详情/状态/日志按 0.25 s 刷新，物体坐标每帧更新，**不重建 widget**。
* ``--ui-test PATH``：程序化注入点击/按钮事件做 13 项交互断言，写 JSON 并返回
  ``0``/``1``；无需人工操作。
"""

from __future__ import annotations

import ctypes
import json
import math
import threading
import time
from pathlib import Path

from .config import WorldConfig, enable_dpi_awareness
from .population import Population
from .render import build_shadow, character_layer, render_world, speech_bubble
from .ui_state import (
    NO_DATA,
    AgentBox,
    ControlState,
    HitTester,
    Layout,
    SelectionModel,
    Viewport,
    build_agent_detail,
    build_event_log,
    build_stats,
)
from .world import World

try:  # tkinter / Pillow 是可选的运行期依赖（测试环境可能没有）
    import tkinter as tk
    from tkinter import messagebox
except ImportError as exc:  # pragma: no cover - 依赖缺失时的友好提示
    tk = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    _TK_ERROR: Exception | None = exc
else:
    _TK_ERROR = None

try:
    from PIL import Image, ImageGrab, ImageTk
except ImportError as exc:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageGrab = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]
    _PIL_ERROR: Exception | None = exc
else:
    _PIL_ERROR = None

#: tkinter + Pillow 是否可用（测试用来决定 skip 与否）
TK_AVAILABLE = tk is not None and ImageTk is not None

#: 界面配色
BG = "#0b0d10"
BAR_BG = "#151a21"
PANEL_BG = "#12161c"
TEXT_BG = "#0e1218"
FG = "#e6eaf0"
MUTED = "#8b95a3"
ACCENT = "#e27e51"
HIGHLIGHT = "#ffd166"
BTN_BG = "#1e2530"
BTN_ACTIVE = "#2c3543"

FONT = "Microsoft YaHei UI"

HELP = (
    "newworld 已启动（鼠标主导交互）。\n"
    "  鼠标：左键点选/切换/取消选中 · 滚轮缩放 · 右键或中键拖动平移\n"
    "  底部按钮：开始/暂停/继续/单步/重置/速度/统计/面板/训练\n"
    "  键盘辅助：空格 暂停 · F1 统计 · F2 训练 · Q / Esc 关闭"
)


def _disable_ime(hwnd: int) -> None:
    """关掉输入法，避免中文输入法挡住画布（Windows）。"""
    try:
        ctypes.windll.imm32.ImmAssociateContext(hwnd, 0)
    except Exception:  # pragma: no cover - 非 Windows / 无权限
        pass


class NewWorldApp:
    """观察窗口的完整状态与交互逻辑。

    所有 UI 处理函数都写成方法（而不是 ``run_app`` 里的闭包），这样
    ``--ui-test`` 可以在同一套代码路径上注入事件并断言。
    """

    #: 主循环间隔（毫秒）
    TICK_MS = 16
    #: 面板/状态栏/日志刷新间隔（秒）——不每帧重建 UI
    PANEL_INTERVAL = 0.25
    #: 世界底图重绘间隔（秒）
    WORLD_REFRESH_S = 0.5
    #: 截图时机（秒，从启动算起）
    SCREENSHOT_DELAY = 0.9

    def __init__(
        self,
        config: WorldConfig,
        *,
        fullscreen: bool = True,
        debug: bool = False,
        screenshot: str | Path | None = None,
        autoclose_ms: int | None = None,
        seed: int = 0,
        policy=None,
        policy_path=None,
        ui_test_path: str | Path | None = None,
    ) -> None:
        if not TK_AVAILABLE:  # pragma: no cover - run_app 已提前检查
            raise RuntimeError(f"tkinter/Pillow 不可用：{_TK_ERROR or _PIL_ERROR}")
        self.config = config
        self.fullscreen = bool(fullscreen)
        self.debug = bool(debug)
        self.seed = int(seed)
        self.policy = policy
        self.policy_path = policy_path
        self.screenshot = Path(screenshot) if screenshot else None
        self.autoclose_ms = autoclose_ms
        self.ui_test_path = Path(ui_test_path) if ui_test_path else None

        # -- 世界与人口 -------------------------------------------------
        self._reset_count = 0
        self.world, self.population = self._create_world(self.seed)

        # -- 纯逻辑状态 -------------------------------------------------
        self.controls = ControlState(playing=True, speed=1.0)
        self.selection = SelectionModel()
        self.layout = Layout(config.width, config.height, panel_visible=True)
        self.viewport = self._fit_viewport(self.layout.rects()["world"])
        self.panel_visible = True
        self.stats_visible = True

        # -- 画布资源缓存 -----------------------------------------------
        self._visuals: dict[int, dict[str, object]] = {}
        self._boxes: dict[int, AgentBox] = {}
        self._world_image = None
        self._world_image_dirty = True
        self._bg_photo = None
        self._shadow_photo = None
        self._shadow_scale = -1.0
        self._bubble_cache: dict[tuple[str, float], object] = {}
        self._empty_photo = None
        self._needs_refresh_background = True

        # -- 列表 / 刷新计时 -------------------------------------------
        self._list_ids: list[int] = []
        self._last_list_ids: tuple[int, ...] | None = None
        self._last_detail_text: str | None = None
        self._last_stats_text: str | None = None
        self._last_log_text: str | None = None
        self._last_status_text: str | None = None
        self._last_hint_text: str | None = None
        self._frame_key: tuple | None = None
        self._last_panel = -1e9
        self._last_bg = -1e9
        self._last_tick = time.perf_counter()
        self._started = self._last_tick
        self._window_size: tuple[int, int] | None = None

        # -- 运行状态 ---------------------------------------------------
        self._closed = False
        self._drag: tuple[int, int] | None = None
        self._screenshot_taken = False
        self._autoclose_at = (
            self._started + autoclose_ms / 1000.0 if autoclose_ms else None
        )
        self._tick_scheduled = False
        self._fps = 0.0
        self._frames = 0
        self._frames_window = 0
        self._fps_window_start = self._started
        self._training = False
        self._thread_messages: list[str] = []
        self._thread_lock = threading.Lock()
        self._log_lines: list[str] = []
        self._log_dirty = True

        # -- 窗口与控件 -------------------------------------------------
        self.root = tk.Tk()
        self.root.title("newworld · 鼠标主导观察窗口")
        self.root.configure(bg=BG)
        self.root.minsize(720, 480)
        if self.screenshot is not None:
            self.root.attributes("-topmost", True)
        if self.fullscreen:
            self.root.attributes("-fullscreen", True)
        else:
            self.root.geometry(f"{config.width}x{config.height}+40+40")
        self._build_ui()
        self._bind_events()
        self.root.update_idletasks()
        self._on_configure()
        self._fit_viewport_to_world()
        self._refresh_background(force=True)
        self._refresh_frame(force=True)
        self._refresh_panels()
        self._update_controls()

    # ==================================================================
    # 世界创建 / 重置
    # ==================================================================
    def _create_world(self, seed: int) -> tuple[World, Population]:
        """构造一个全新的世界与人口（重新开始用）。"""
        world = World(self.config, seed=seed)
        world.reset(seed=seed, positions=[self.config.width / 2.0])
        population = Population(world, seed)
        return world, population

    def _fit_viewport(self, world_rect: tuple[int, int, int, int]) -> Viewport:
        """初始视口：整张世界缩放到画布内（只缩小不放大）。"""
        view_w = max(1, int(world_rect[2]))
        view_h = max(1, int(world_rect[3]))
        scale = min(1.0, view_w / max(1, self.config.width), view_h / max(1, self.config.height))
        return Viewport(self.config.width, self.config.height, view_w, view_h, scale=scale).clamp()

    def _fit_viewport_to_world(self) -> None:
        self.viewport = self._fit_viewport(self.layout.rects()["world"])

    # ==================================================================
    # 窗口与控件
    # ==================================================================
    def _build_ui(self) -> None:
        root = self.root
        self._empty_photo = tk.PhotoImage(master=root, width=1, height=1)

        # -- 顶部状态栏 -------------------------------------------------
        self.status_frame = tk.Frame(root, bg=BAR_BG)
        self.status_var = tk.StringVar(value="newworld 启动中…")
        self.hint_var = tk.StringVar(value="")
        self.status_label = tk.Label(
            self.status_frame,
            textvariable=self.status_var,
            bg=BAR_BG,
            fg=FG,
            anchor="w",
            padx=10,
            font=(FONT, 10),
        )
        self.hint_label = tk.Label(
            self.status_frame,
            textvariable=self.hint_var,
            bg=BAR_BG,
            fg=MUTED,
            anchor="e",
            padx=10,
            font=(FONT, 9),
        )
        self.status_label.pack(side="left", fill="both", expand=True)
        self.hint_label.pack(side="right")

        # -- 中央画布 ---------------------------------------------------
        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0, bd=0)
        self._bg_item = self.canvas.create_image(0, 0, anchor="nw", image=self._empty_photo)
        self._ring_item = self.canvas.create_oval(
            0, 0, 0, 0, outline=HIGHLIGHT, width=3, state="hidden"
        )

        # -- 右侧详情面板 -----------------------------------------------
        self.panel_frame = tk.Frame(root, bg=PANEL_BG)
        self._panel_title = tk.Label(
            self.panel_frame,
            text="个体列表",
            bg=PANEL_BG,
            fg=ACCENT,
            anchor="w",
            padx=8,
            pady=4,
            font=(FONT, 10, "bold"),
        )
        list_frame = tk.Frame(self.panel_frame, bg=PANEL_BG)
        self.listbox = tk.Listbox(
            list_frame,
            bg=TEXT_BG,
            fg=FG,
            selectbackground=ACCENT,
            selectforeground="#10151b",
            activestyle="none",
            exportselection=False,
            height=5,
            bd=0,
            highlightthickness=0,
            font=(FONT, 10),
        )
        list_scroll = tk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=list_scroll.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        list_scroll.pack(side="right", fill="y")
        self._detail_title = tk.Label(
            self.panel_frame,
            text="选中个体详情",
            bg=PANEL_BG,
            fg=ACCENT,
            anchor="w",
            padx=8,
            pady=4,
            font=(FONT, 10, "bold"),
        )
        detail_frame = tk.Frame(self.panel_frame, bg=PANEL_BG)
        self.detail_text = tk.Text(
            detail_frame,
            bg=TEXT_BG,
            fg=FG,
            wrap="word",
            bd=0,
            highlightthickness=0,
            padx=8,
            pady=6,
            height=8,
            font=(FONT, 9),
            state="disabled",
            cursor="arrow",
        )
        detail_scroll = tk.Scrollbar(detail_frame, orient="vertical", command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_scroll.set)
        self.detail_text.pack(side="left", fill="both", expand=True)
        detail_scroll.pack(side="right", fill="y")
        self.detail_text.tag_configure("title", foreground=HIGHLIGHT, font=(FONT, 10, "bold"))
        self.detail_text.tag_configure("label", foreground=MUTED)
        self.detail_text.tag_configure("value", foreground=FG)
        self.detail_text.tag_configure("nodata", foreground="#6d7683")
        self._stats_title = tk.Label(
            self.panel_frame,
            text="世界统计",
            bg=PANEL_BG,
            fg=ACCENT,
            anchor="w",
            padx=8,
            pady=4,
            font=(FONT, 10, "bold"),
        )
        self._stats_frame = tk.Frame(self.panel_frame, bg=PANEL_BG)
        self.stats_text = tk.Text(
            self._stats_frame,
            bg=TEXT_BG,
            fg=FG,
            wrap="word",
            bd=0,
            highlightthickness=0,
            padx=8,
            pady=6,
            height=8,
            font=(FONT, 9),
            state="disabled",
            cursor="arrow",
        )
        stats_scroll = tk.Scrollbar(self._stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        self.stats_text.pack(side="left", fill="both", expand=True)
        stats_scroll.pack(side="right", fill="y")

        # 用 grid 而不是 pack：窗口变小时详情/统计按权重收缩，统计面板不会被挤没
        self._list_frame = list_frame
        self._detail_frame = detail_frame
        self.panel_frame.columnconfigure(0, weight=1)
        self.panel_frame.rowconfigure(3, weight=3)
        self.panel_frame.rowconfigure(5, weight=2)
        self._panel_title.grid(row=0, column=0, sticky="ew")
        list_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 4))
        self._detail_title.grid(row=2, column=0, sticky="ew")
        detail_frame.grid(row=3, column=0, sticky="nsew", padx=6, pady=(0, 4))
        self._stats_title.grid(row=4, column=0, sticky="ew")
        self._stats_frame.grid(row=5, column=0, sticky="nsew", padx=6, pady=(0, 6))

        # -- 底部控制栏 + 日志 ------------------------------------------
        self.bottom_frame = tk.Frame(root, bg=BAR_BG)
        controls = tk.Frame(self.bottom_frame, bg=BAR_BG)
        self.button_start = self._make_button(controls, "开始", self._on_start)
        self.button_pause = self._make_button(controls, "暂停", self._on_pause)
        self.button_resume = self._make_button(controls, "继续", self._on_resume)
        self.button_step = self._make_button(controls, "单步", self._on_step)
        self.button_reset = self._make_button(controls, "重置", self._on_reset)
        self.button_slower = self._make_button(controls, "速度 -", lambda: self._on_speed(-1))
        self.button_faster = self._make_button(controls, "速度 +", lambda: self._on_speed(+1))
        self.button_stats = self._make_button(controls, "统计", self._on_toggle_stats)
        self.button_panel = self._make_button(controls, "面板", self._on_toggle_panel)
        self.button_train = self._make_button(controls, "训练", self._on_train)
        for button in (
            self.button_start,
            self.button_pause,
            self.button_resume,
            self.button_step,
            self.button_reset,
            self.button_slower,
            self.button_faster,
            self.button_stats,
            self.button_panel,
            self.button_train,
        ):
            button.pack(side="left", padx=(6, 0), pady=4)
        self.speed_var = tk.StringVar(value="速度 1.00×")
        self.speed_label = tk.Label(
            controls,
            textvariable=self.speed_var,
            bg=BAR_BG,
            fg=HIGHLIGHT,
            padx=10,
            font=(FONT, 10),
        )
        self.speed_label.pack(side="left", padx=(8, 0))
        controls.pack(fill="x")
        self._controls_frame = controls

        log_frame = tk.Frame(self.bottom_frame, bg=BAR_BG)
        self.log_text = tk.Text(
            log_frame,
            bg=TEXT_BG,
            fg=MUTED,
            wrap="none",
            bd=0,
            highlightthickness=0,
            padx=8,
            pady=4,
            height=6,
            font=(FONT, 9),
            state="disabled",
            cursor="arrow",
        )
        log_scroll = tk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")
        log_frame.pack(fill="both", expand=True, pady=(0, 2))

    def _make_button(self, parent, text: str, command) -> "tk.Button":
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=BTN_BG,
            fg=FG,
            activebackground=BTN_ACTIVE,
            activeforeground=FG,
            disabledforeground="#5a6472",
            bd=0,
            highlightthickness=0,
            padx=10,
            pady=4,
            relief="flat",
            cursor="hand2",
            font=(FONT, 10),
            takefocus=0,
        )

    def _bind_events(self) -> None:
        canvas = self.canvas
        canvas.bind("<Button-1>", self._on_canvas_click)
        canvas.bind("<MouseWheel>", self._on_wheel)
        canvas.bind("<Button-4>", lambda event: self._zoom_at_screen(event.x, event.y, 1.1))
        canvas.bind("<Button-5>", lambda event: self._zoom_at_screen(event.x, event.y, 1 / 1.1))
        for button in (2, 3):
            canvas.bind(f"<ButtonPress-{button}>", self._on_pan_start)
            canvas.bind(f"<ButtonRelease-{button}>", self._on_pan_end)
        canvas.bind("<Motion>", self._on_pan_move)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)
        self.root.bind("<Configure>", self._on_configure)
        self.root.bind("<space>", self._on_space)
        self.root.bind("<F1>", self._on_toggle_stats)
        self.root.bind("<F2>", self._on_train)
        for sequence in ("<KeyPress-q>", "<KeyPress-Q>", "<Escape>"):
            self.root.bind(sequence, lambda _event=None: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ==================================================================
    # 布局
    # ==================================================================
    def _place(self, widget, rect: tuple[int, int, int, int]) -> None:
        widget.place(x=rect[0], y=rect[1], width=max(1, rect[2]), height=max(1, rect[3]))

    def _apply_layout(self) -> None:
        rects = self.layout.rects()
        self._place(self.status_frame, rects["top"])
        self._place(self.canvas, rects["world"])
        self._place(self.bottom_frame, rects["bottom"])
        if self.panel_visible:
            self._place(self.panel_frame, rects["panel"])
        else:
            self.panel_frame.place_forget()

    def _on_configure(self, event=None) -> None:
        """窗口尺寸变化：按新尺寸重建 Layout（契约 §1.2）。"""
        if event is not None and getattr(event, "widget", None) is not self.root:
            return
        width = int(self.root.winfo_width())
        height = int(self.root.winfo_height())
        if width <= 1 or height <= 1:
            return
        if (width, height) == self._window_size:
            return
        self._window_size = (width, height)
        self.layout = Layout(width, height, panel_visible=self.panel_visible)
        self._apply_layout()
        rect = self.layout.rects()["world"]
        self.viewport = self.viewport.resize(rect[2], rect[3])
        self._world_image_dirty = True
        self._needs_refresh_background = True
        self._frame_key = None

    # ==================================================================
    # 世界渲染
    # ==================================================================
    def _alive_agents(self) -> list:
        return [agent for agent in self.world.agents if getattr(agent, "alive", False)]

    def _alive_ids(self) -> set[int]:
        return {int(agent.agent_id) for agent in self._alive_agents()}

    def _refresh_background(self, force: bool = False) -> None:
        """把世界底图按当前视口裁剪/缩放后贴到画布（缩放、平移、地形更新时）。"""
        scale = self.viewport.scale
        x0, y0, x1, y1 = self.viewport.visible_world_rect()
        world_w = max(1, self.config.width)
        world_h = max(1, self.config.height)
        left = int(max(0, min(world_w - 1, math.floor(x0))))
        top = int(max(0, min(world_h - 1, math.floor(y0))))
        right = int(min(world_w, max(1, math.ceil(x1))))
        bottom = int(min(world_h, max(1, math.ceil(y1))))
        if right - left < 1 or bottom - top < 1:
            left, top, right, bottom = 0, 0, world_w, world_h
        if self._world_image is None or self._world_image_dirty:
            self._world_image = render_world(self.world)
            self._world_image_dirty = False
            self._shadow_photo = None
        crop = self._world_image.crop((left, top, right, bottom))
        if abs(scale - 1.0) > 1e-9:
            crop = crop.resize(
                (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                Image.NEAREST,
            )
        self._bg_photo = ImageTk.PhotoImage(crop, master=self.root)
        self.canvas.itemconfigure(self._bg_item, image=self._bg_photo)
        screen_x, screen_y = self.viewport.world_to_screen(left, top)
        self.canvas.coords(self._bg_item, round(screen_x), round(screen_y))
        self.canvas.tag_lower(self._bg_item)

        if self._shadow_photo is None or abs(self._shadow_scale - scale) > 1e-9:
            shadow = build_shadow(self.world)
            if abs(scale - 1.0) > 1e-9:
                shadow = shadow.resize(
                    (max(1, round(shadow.width * scale)), max(1, round(shadow.height * scale))),
                    Image.NEAREST,
                )
            self._shadow_photo = ImageTk.PhotoImage(shadow, master=self.root)
            self._shadow_scale = scale
        self._frame_key = None
        self._last_bg = time.perf_counter()

    def _sync_visuals(self) -> None:
        """按存活集合创建/删除画布上的个体图形（不重建 Canvas 控件）。"""
        alive = {int(agent.agent_id): agent for agent in self._alive_agents()}
        for agent_id in list(self._visuals):
            if agent_id in alive:
                continue
            visual = self._visuals.pop(agent_id)
            for item in visual["items"]:  # type: ignore[index]
                self.canvas.delete(item)
            self._boxes.pop(agent_id, None)
        for agent_id in alive:
            if agent_id in self._visuals:
                continue
            shadow_item = self.canvas.create_image(0, 0, anchor="nw", image=self._empty_photo)
            char_item = self.canvas.create_image(0, 0, anchor="nw", image=self._empty_photo)
            bubble_item = self.canvas.create_image(
                0, 0, anchor="nw", image=self._empty_photo, state="hidden"
            )
            # small marker above the head: this individual is executing a plan
            task_item = self.canvas.create_oval(
                0, 0, 0, 0, fill=ACCENT, outline="", state="hidden"
            )
            self._visuals[agent_id] = {
                "shadow_item": shadow_item,
                "char_item": char_item,
                "bubble_item": bubble_item,
                "task_item": task_item,
                "char_photo": self._empty_photo,
                "bubble_photo": self._empty_photo,
                "bubble_text": None,
                "items": (shadow_item, char_item, bubble_item, task_item),
            }

    def _refresh_agent(self, agent) -> None:
        visual = self._visuals.get(int(agent.agent_id))
        if visual is None:
            return
        scale = self.viewport.scale
        feet_y = agent.y if getattr(agent, "y", None) is not None else self.config.ground_y
        layer, anchor = character_layer(self.world, agent)
        layer_w, layer_h = layer.size
        anchor_x, anchor_y = float(anchor[0]), float(anchor[1])
        # 命中盒：精灵包围盒中心（世界坐标）——与绘制结果一致
        self._boxes[int(agent.agent_id)] = AgentBox(
            agent_id=int(agent.agent_id),
            x=float(agent.x) - anchor_x + layer_w / 2.0,
            y=float(feet_y) - anchor_y + layer_h / 2.0,
            sprite_w=float(layer_w),
            sprite_h=float(layer_h),
            alive=True,
        )
        if abs(scale - 1.0) > 1e-9:
            layer = layer.resize(
                (max(1, round(layer_w * scale)), max(1, round(layer_h * scale))), Image.NEAREST
            )
            anchor_x *= scale
            anchor_y *= scale
        photo = ImageTk.PhotoImage(layer, master=self.root)
        visual["char_photo"] = photo
        self.canvas.itemconfigure(visual["char_item"], image=photo)
        screen_x, screen_y = self.viewport.world_to_screen(float(agent.x), float(feet_y))
        self.canvas.coords(
            visual["char_item"], round(screen_x - anchor_x), round(screen_y - anchor_y)
        )
        # -- 任务标记：正在执行计划（有活跃 plan）的个体头顶显示小圆点 ------
        task_item = visual.get("task_item")
        if task_item is not None:
            active_plan = getattr(getattr(self.population, "goals", None), "active_plan", None)
            has_task = False
            if callable(active_plan):
                try:
                    has_task = active_plan(int(agent.agent_id)) is not None
                except Exception:
                    has_task = False
            if has_task:
                radius = max(3.0, min(7.0, 4.0 * scale + 2.0))
                dot_y = screen_y - anchor_y - radius - 4.0
                self.canvas.coords(
                    task_item,
                    screen_x - radius,
                    dot_y - radius,
                    screen_x + radius,
                    dot_y + radius,
                )
                self.canvas.itemconfigure(task_item, state="normal")
                self.canvas.tag_raise(task_item)
            else:
                self.canvas.itemconfigure(task_item, state="hidden")
        shadow_photo = self._shadow_photo
        if shadow_photo is not None:
            self.canvas.itemconfigure(visual["shadow_item"], image=shadow_photo)
            self.canvas.coords(
                visual["shadow_item"],
                round(screen_x - shadow_photo.width() / 2),
                round(screen_y - shadow_photo.height() / 2),
            )
        # 说话气泡（仅在有效期内显示；按文本+缩放缓存图片）
        text = (
            agent.utterance
            if agent.utterance is not None and agent.utterance_until >= self.world.t - 1e-9
            else None
        )
        bubble_item = visual["bubble_item"]
        if not text:
            self.canvas.itemconfigure(bubble_item, state="hidden")
            visual["bubble_text"] = None
            return
        key = (str(text), round(scale, 3))
        if key not in self._bubble_cache:
            bubble = speech_bubble(str(text))
            if abs(scale - 1.0) > 1e-9:
                bubble = bubble.resize(
                    (max(1, round(bubble.width * scale)), max(1, round(bubble.height * scale))),
                    Image.NEAREST,
                )
            self._bubble_cache[key] = ImageTk.PhotoImage(bubble, master=self.root)
            if len(self._bubble_cache) > 32:
                self._bubble_cache.pop(next(iter(self._bubble_cache)))
        photo = self._bubble_cache[key]
        visual["bubble_photo"] = photo
        visual["bubble_text"] = text
        self.canvas.itemconfigure(bubble_item, image=photo, state="normal")
        self.canvas.coords(
            bubble_item,
            round(screen_x - photo.width() / 2),
            round(screen_y - anchor_y - photo.height() - 8),
        )

    def _refresh_ring(self) -> None:
        selected_id = self.selection.selected_id
        box = self._boxes.get(int(selected_id)) if selected_id is not None else None
        if box is None:
            self.canvas.itemconfigure(self._ring_item, state="hidden")
            return
        cx, cy = self.viewport.world_to_screen(box.x, box.y)
        half_w = box.sprite_w * self.viewport.scale / 2.0 + 8.0
        half_h = box.sprite_h * self.viewport.scale / 2.0 + 8.0
        self.canvas.coords(self._ring_item, cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        self.canvas.itemconfigure(self._ring_item, state="normal")
        self.canvas.tag_raise(self._ring_item)

    def _refresh_frame(self, force: bool = False) -> None:
        """每帧只更新画布几何/图片；状态未变时直接跳过（不重建 UI）。"""
        key = (
            round(float(self.world.t), 6),
            round(self.viewport.scale, 6),
            round(self.viewport.offset_x, 6),
            round(self.viewport.offset_y, 6),
            self.selection.selected_id,
            len(self.world.agents),
        )
        if not force and key == self._frame_key:
            return
        self._frame_key = key
        self._sync_visuals()
        for agent in self._alive_agents():
            self._refresh_agent(agent)
        self._refresh_ring()

    # ==================================================================
    # 仿真推进
    # ==================================================================
    def _step_once(self) -> None:
        """按当前世界配置的 ``fixed_dt`` 推进恰好一步。"""
        self.population.step(float(self.world.config.fixed_dt))
        self._world_image_dirty = True
        self._frame_key = None

    def _consume_pending_steps(self) -> int:
        steps = 0
        while self.controls.consume_step():
            self._step_once()
            steps += 1
        if steps:
            self._log(f"单步推进 {steps} 步（{self.world.config.fixed_dt * steps:.4f} 秒）")
        return steps

    def _advance(self, wall_dt: float) -> None:
        if self.controls.playing:
            world_dt = self.controls.scaled_dt(wall_dt)
            if world_dt > 0.0:
                self.population.step(world_dt)
                self._world_image_dirty = True
        else:
            self._consume_pending_steps()

    # ==================================================================
    # 鼠标：世界点击 / 缩放 / 平移
    # ==================================================================
    def _on_canvas_click(self, event) -> None:
        rect = self.layout.rects()["world"]
        self._handle_click_at_window(rect[0] + int(event.x), rect[1] + int(event.y))

    def _handle_click_at_window(self, window_x: float, window_y: float) -> bool:
        """窗口坐标的世界点击；落在 UI 区域时直接忽略（点击隔离）。"""
        if self.layout.region_at(window_x, window_y) != "world":
            return False
        rect = self.layout.rects()["world"]
        self._handle_world_click(window_x - rect[0], window_y - rect[1])
        return True

    def _handle_world_click(self, screen_x: float, screen_y: float) -> None:
        """画布局部坐标：命中就选中/切换，点空地取消选中。"""
        tester = HitTester(self.viewport, max_distance_px=60.0)
        hit = tester.hit(list(self._boxes.values()), screen_x, screen_y)
        alive = self._alive_ids()
        if hit is None:
            changed = self.selection.clear()
        else:
            changed = self.selection.toggle(hit.agent_id, alive_ids=alive)
        if changed:
            self._on_selection_changed()

    def _on_wheel(self, event) -> None:
        delta = int(getattr(event, "delta", 0) or 0)
        if delta == 0:
            return
        factor = 1.1 ** (delta / 120.0)
        self._zoom_at_screen(int(event.x), int(event.y), factor)

    def _zoom_at_screen(self, screen_x: float, screen_y: float, factor: float) -> None:
        before = self.viewport.scale
        self.viewport = self.viewport.zoom_at(screen_x, screen_y, factor)
        if abs(self.viewport.scale - before) < 1e-9:
            return
        self._refresh_background()
        self._refresh_frame(force=True)

    def _on_pan_start(self, event) -> None:
        self._drag = (int(event.x), int(event.y))

    def _on_pan_end(self, _event) -> None:
        self._drag = None

    def _on_pan_move(self, event) -> None:
        if self._drag is None:
            return
        last_x, last_y = self._drag
        dx = int(event.x) - last_x
        dy = int(event.y) - last_y
        self._drag = (int(event.x), int(event.y))
        if dx == 0 and dy == 0:
            return
        self.viewport = self.viewport.pan(dx, dy).clamp()
        self._refresh_background()
        self._refresh_frame(force=True)

    # ==================================================================
    # 选择同步
    # ==================================================================
    def _on_list_select(self, _event=None) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        if 0 <= index < len(self._list_ids):
            if self.selection.select(self._list_ids[index], alive_ids=self._alive_ids()):
                self._on_selection_changed()

    def _on_selection_changed(self) -> None:
        self._sync_list_selection()
        self._refresh_detail()
        self._frame_key = None

    def _apply_alive_ids(self, alive_ids: set[int]) -> None:
        """同步存活集合：选中者死亡自动清理 + 列表随存活集合重建。"""
        cleaned = self.selection.sync(alive_ids)
        ordered = tuple(sorted(int(agent_id) for agent_id in alive_ids))
        if ordered != self._last_list_ids:
            self._last_list_ids = ordered
            self._refresh_agent_list(alive_ids)
        if cleaned:
            self._on_selection_changed()

    def _refresh_agent_list(self, alive_ids: set[int] | None = None) -> None:
        wanted = self._alive_ids() if alive_ids is None else {int(item) for item in alive_ids}
        alive = sorted(
            (agent for agent in self._alive_agents() if int(agent.agent_id) in wanted),
            key=lambda agent: int(agent.agent_id),
        )
        self._list_ids = [int(agent.agent_id) for agent in alive]
        self.listbox.delete(0, "end")
        for agent in alive:
            name = getattr(agent, "name", "") or f"{agent.agent_id} 号"
            state = getattr(agent, "behavior_state", "") or "?"
            self.listbox.insert(
                "end",
                f"{agent.agent_id:>2}号 {name} · {state} · 能量{agent.physiology.energy:.0f}",
            )
        self._sync_list_selection()

    def _sync_list_selection(self) -> None:
        self.listbox.selection_clear(0, "end")
        for index in range(len(self._list_ids)):
            self.listbox.itemconfig(index, foreground=FG)
        selected = self.selection.selected_id
        if selected is None or selected not in self._list_ids:
            return
        index = self._list_ids.index(selected)
        self.listbox.selection_set(index)
        self.listbox.itemconfig(index, foreground=HIGHLIGHT)
        self.listbox.see(index)

    # ==================================================================
    # 面板刷新
    # ==================================================================
    @staticmethod
    def _set_text(widget, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _fill_rows(self, widget, title: str, rows) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        if title:
            widget.insert("end", title + "\n", "title")
        for row in rows:
            widget.insert("end", f"{row.label}：", "label")
            widget.insert("end", row.value + "\n", "nodata" if row.value == NO_DATA else "value")
        widget.configure(state="disabled")

    def _refresh_detail(self) -> None:
        detail = build_agent_detail(self.world, self.population, self.selection.selected_id)
        detail.selected = self.selection.selected_id is not None
        text = detail.title + "".join(f"{row.label}：{row.value}\n" for row in detail.rows)
        if text == self._last_detail_text:
            return
        self._last_detail_text = text
        self._fill_rows(self.detail_text, detail.title, detail.rows)

    def _refresh_stats(self) -> None:
        if not self.stats_visible:
            return
        rows = build_stats(self.world, self.population)
        text = "".join(f"{row.label}：{row.value}\n" for row in rows)
        if text == self._last_stats_text:
            return
        self._last_stats_text = text
        self._fill_rows(self.stats_text, "", rows)

    def _refresh_log(self) -> None:
        events = build_event_log(self.world, self.population, limit=10)
        lines = ["# 世界事件"] + [f"· {line}" for line in events]
        if self._log_lines:
            lines.append("# 操作")
            lines.extend(f"· {line}" for line in self._log_lines[-6:])
        text = "\n".join(lines)
        if text == self._last_log_text:
            return
        self._last_log_text = text
        self._set_text(self.log_text, text)
        self.log_text.see("end")

    def _update_status(self) -> None:
        world = self.world
        registry = world.registry
        alive = registry.alive_count()
        created = registry.total_created
        deaths = registry.total_deaths
        limit = registry.max_agents
        remaining = registry.remaining()
        day = world.day.day_index(world.t) + 1
        hour = float(world.hour)
        run_state = "运行中" if self.controls.playing else "已暂停"
        text = (
            f"第 {day} 天 {int(hour):02d}:{int((hour % 1.0) * 60):02d}"
            f" · 世界时间 {world.t:6.1f}s · {run_state} · 速度 {self.controls.speed:.2f}×"
            f" · 存活 {alive} · 累计创建 {created}/{limit} · 死亡 {deaths}"
            f" · 剩余可创建 {remaining}"
        )
        if remaining <= 0:
            text += "  ⚠ 已达累计创建上限，无法继续创建"
        if self.debug:
            text += " · [调试]"
        if text != self._last_status_text:
            self._last_status_text = text
            self.status_var.set(text)

    def _update_hint(self) -> None:
        text = f"缩放 {self.viewport.scale:.2f}× · fps {self._fps:4.1f} · 左键选中"
        if text != self._last_hint_text:
            self._last_hint_text = text
            self.hint_var.set(text)

    def _update_controls(self) -> None:
        actions = self.controls.available_actions()
        mapping = {
            self.button_start: actions["start"],
            self.button_pause: actions["pause"],
            self.button_resume: actions["resume"],
            self.button_step: actions["step"],
            self.button_slower: actions["slower"],
            self.button_faster: actions["faster"],
        }
        for button, enabled in mapping.items():
            button.configure(state="normal" if enabled else "disabled")
        self.speed_var.set(f"速度 {self.controls.speed:.2f}×")
        self.button_stats.configure(text="统计 ✓" if self.stats_visible else "统计")
        self.button_panel.configure(text="面板 ✓" if self.panel_visible else "面板")

    def _refresh_panels(self) -> None:
        self._apply_alive_ids(self._alive_ids())
        self._refresh_detail()
        self._refresh_stats()
        self._refresh_log()
        self._update_status()
        self._update_hint()
        self._update_controls()

    # ==================================================================
    # 按钮 / 键盘处理
    # ==================================================================
    def _log(self, text: str) -> None:
        self._log_lines.append(f"{float(self.world.t):7.1f}s · {text}")
        del self._log_lines[:-200]
        self._log_dirty = True
        self._last_log_text = None

    def _drain_thread_messages(self) -> None:
        if not self._thread_messages:
            return
        with self._thread_lock:
            messages = list(self._thread_messages)
            self._thread_messages.clear()
        for message in messages:
            self._log(message)

    def _on_start(self) -> None:
        if not self.controls.resume():
            self._log("仿真已在运行")
        else:
            self._log("开始运行")
        self._update_controls()

    def _on_pause(self) -> None:
        if self.controls.pause():
            self._log("已暂停（画面与详情仍可查看）")
        self._update_controls()

    def _on_resume(self) -> None:
        if self.controls.resume():
            self._log("继续运行")
        self._update_controls()

    def _on_space(self, _event=None) -> None:
        if self.controls.toggle():
            self._log("继续运行")
        else:
            self._log("已暂停")
        self._update_controls()

    def _on_step(self) -> None:
        if not self.controls.request_step():
            self._log("单步只在暂停时可用")
            return
        self._consume_pending_steps()
        self._refresh_frame(force=True)
        self._refresh_panels()
        self._update_controls()

    def _on_speed(self, direction: int) -> None:
        speed = self.controls.nudge_speed(direction)
        self._log(f"速度：{speed:.2f}×")
        self._update_controls()

    def _on_toggle_stats(self, _event=None) -> None:
        self.stats_visible = not self.stats_visible
        if self.stats_visible:
            self.panel_frame.rowconfigure(3, weight=3)
            self.panel_frame.rowconfigure(5, weight=2)
            self._stats_title.grid()
            self._stats_frame.grid()
            self._last_stats_text = None
            self._refresh_stats()
            self._log("统计面板：显示")
        else:
            self._stats_title.grid_remove()
            self._stats_frame.grid_remove()
            self.panel_frame.rowconfigure(3, weight=1)
            self.panel_frame.rowconfigure(5, weight=0)
            self._log("统计面板：隐藏")
        self._update_controls()

    def _on_toggle_panel(self, _event=None) -> None:
        self.panel_visible = not self.panel_visible
        width, height = self._window_size or (self.root.winfo_width(), self.root.winfo_height())
        self.layout = Layout(int(width), int(height), panel_visible=self.panel_visible)
        self._apply_layout()
        rect = self.layout.rects()["world"]
        self.viewport = self.viewport.resize(rect[2], rect[3])
        self._world_image_dirty = True
        self._refresh_background()
        self._refresh_frame(force=True)
        self._log("详情面板：" + ("显示" if self.panel_visible else "折叠"))
        self._update_controls()

    def _on_train(self, _event=None) -> None:
        """F2 / 训练按钮：所有个体从零训练自己的神经网络（后台线程）。"""
        if self._training:
            self._log("训练已在进行中，请稍候")
            return
        self._training = True
        self._log("已请求从零训练所有个体的神经网络（后台线程）…")

        def worker() -> None:
            try:
                metrics = self.population.train_round(self.seed + self._reset_count, reset=True)
                message = (
                    f"训练完成：{int(metrics['agents'])} 个个体，"
                    f"模式准确率 {metrics['mode_accuracy']:.2f}"
                )
            except Exception as exc:  # pragma: no cover - 后台反馈
                message = f"训练失败：{exc}"
            finally:
                self._training = False
            with self._thread_lock:
                self._thread_messages.append(message)

        threading.Thread(target=worker, daemon=True).start()

    def _ask_reset(self) -> bool:
        """重置确认对话框（契约 §1.2 要求 messagebox.askyesno）。"""
        if messagebox is None:  # pragma: no cover
            return False
        try:
            return bool(
                messagebox.askyesno(
                    "确认重置",
                    "重置会创建一个全新的世界与个体，当前进度不会保留。\n确定要重置吗？",
                    parent=self.root,
                )
            )
        except Exception:  # pragma: no cover - 无显示/被禁用时保守拒绝
            return False

    def _on_reset(self) -> None:
        self._do_reset(ask=True)

    def _do_reset(self, *, ask: bool = True) -> bool:
        """重置世界：确认后重建世界/人口，清空选择与列表并恢复运行。"""
        if ask and not self._ask_reset():
            self._log("已取消重置")
            return False
        self._reset_count += 1
        seed = self.seed + self._reset_count * 101
        self.world, self.population = self._create_world(seed)
        self.selection.clear()
        for agent_id in list(self._visuals):
            visual = self._visuals.pop(agent_id)
            for item in visual["items"]:  # type: ignore[index]
                self.canvas.delete(item)
        self._boxes.clear()
        speed = self.controls.speed
        self.controls = ControlState(playing=True, speed=speed)
        self._list_ids = []
        self._last_list_ids = None
        self._last_detail_text = None
        self._last_stats_text = None
        self._last_log_text = None
        self._last_status_text = None
        self._last_hint_text = None
        self._frame_key = None
        self._world_image = None
        self._world_image_dirty = True
        self._shadow_photo = None
        self._shadow_scale = -1.0
        self._bubble_cache.clear()
        rect = self.layout.rects()["world"]
        self.viewport = self._fit_viewport(rect)
        self._refresh_background(force=True)
        self._refresh_frame(force=True)
        self._refresh_panels()
        self._log(f"已重置世界（种子 {seed}）")
        return True

    # ==================================================================
    # 主循环
    # ==================================================================
    def _start_tick(self) -> None:
        if self._tick_scheduled:
            return
        self._tick_scheduled = True
        self._last_tick = time.perf_counter()
        self.root.after(0, self._tick)

    def _tick(self) -> None:
        if self._closed:
            return
        now = time.perf_counter()
        dt = min(max(now - self._last_tick, 0.0), 0.05)
        self._last_tick = now
        try:
            self._drain_thread_messages()
            self._advance(dt)
            self._refresh_frame()
            if self._needs_refresh_background or now - self._last_bg >= self.WORLD_REFRESH_S:
                self._needs_refresh_background = False
                self._refresh_background()
            if now - self._last_panel >= self.PANEL_INTERVAL:
                self._last_panel = now
                self._refresh_panels()
        except Exception as exc:  # pragma: no cover - 保证窗口不因单帧异常退出
            print(f"UI 帧异常：{exc}")
        # fps 统计（0.5 s 窗口）
        self._frames += 1
        self._frames_window += 1
        if now - self._fps_window_start >= 0.5:
            self._fps = self._frames_window / max(1e-6, now - self._fps_window_start)
            self._frames_window = 0
            self._fps_window_start = now
        self._maybe_screenshot(now)
        if self._autoclose_at is not None and now >= self._autoclose_at:
            self.close()
            return
        self.root.after(self.TICK_MS, self._tick)

    def _maybe_screenshot(self, now: float) -> None:
        if self.screenshot is None or self._screenshot_taken:
            return
        if now - self._started < self.SCREENSHOT_DELAY:
            return
        self._screenshot_taken = True
        try:
            self._grab_screenshot()
        finally:
            self.close()

    def _grab_screenshot(self) -> None:
        path = Path(self.screenshot)  # type: ignore[arg-type]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            x = int(self.root.winfo_rootx())
            y = int(self.root.winfo_rooty())
            width = max(1, int(self.root.winfo_width()))
            height = max(1, int(self.root.winfo_height()))
            image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
            image.save(path)
            print(f"截图已保存：{path}（{width}x{height} 窗口）")
        except Exception as exc:  # pragma: no cover - 抓屏依赖桌面环境
            print(f"截图失败：{exc}")

    def announce(self) -> None:
        print(HELP)
        print(
            f"窗口 {self.root.winfo_width()}x{self.root.winfo_height()}，"
            f"世界 {self.config.width}x{self.config.height}，"
            f"区域 {self.layout.rects()}"
        )
        _disable_ime(int(self.root.winfo_id()))
        self.root.lift()
        self.root.focus_force()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.root.destroy()
        except Exception:  # pragma: no cover
            pass

    def run(self) -> int:
        self._start_tick()
        self.root.after(250, self.announce)
        started = time.perf_counter()
        try:
            self.root.mainloop()
        finally:
            elapsed = max(1e-6, time.perf_counter() - started)
            print(
                f"离开 newworld：{self._frames} 帧 / {elapsed:.1f}s，"
                f"平均 {self._frames / elapsed:.1f} fps；{self.population.summary_zh()}"
            )
        return 0

    # ==================================================================
    # --ui-test：程序化自检（契约 §1.2 / §1.3）
    # ==================================================================
    UI_TEST_CHECKS = (
        "layout_regions_and_click_isolation",
        "viewport_transform_and_zoom_anchor",
        "world_click_selects_agent",
        "world_click_switches_selection",
        "world_click_toggles_off",
        "world_click_empty_clears_selection",
        "wheel_zoom_and_drag_pan_hit_consistency",
        "pause_freezes_time_but_panel_visible",
        "single_step_advances_exactly_one_frame",
        "list_click_syncs_selection",
        "dead_agent_clears_selection",
        "detail_and_stats_use_real_data",
        "reset_clears_selection_and_list",
    )

    def run_ui_test(self, path: str | Path | None = None) -> int:
        """注入点击/按钮事件并断言；写 JSON 到 ``path``，返回退出码 0/1。"""
        path = Path(path) if path is not None else self.ui_test_path
        started = time.perf_counter()
        checks: list[dict[str, object]] = []
        self._start_tick()
        self.root.update()
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline and self.root.winfo_width() <= 1:
            self.root.update()
            time.sleep(0.01)
        # 让主循环跑几帧，确保画布几何/命中盒已就绪
        for _ in range(4):
            self.root.update()
            time.sleep(0.03)

        runners = {
            "layout_regions_and_click_isolation": self._ui_layout,
            "viewport_transform_and_zoom_anchor": self._ui_viewport,
            "world_click_selects_agent": self._ui_click_select,
            "world_click_switches_selection": self._ui_click_switch,
            "world_click_toggles_off": self._ui_click_toggle,
            "world_click_empty_clears_selection": self._ui_click_empty,
            "wheel_zoom_and_drag_pan_hit_consistency": self._ui_zoom_pan,
            "pause_freezes_time_but_panel_visible": self._ui_pause,
            "single_step_advances_exactly_one_frame": self._ui_single_step,
            "list_click_syncs_selection": self._ui_list_select,
            "dead_agent_clears_selection": self._ui_dead_cleanup,
            "detail_and_stats_use_real_data": self._ui_detail_stats,
            "reset_clears_selection_and_list": self._ui_reset,
        }
        payload: dict[str, object] = {"ok": False}
        try:
            for name in self.UI_TEST_CHECKS:
                runner = runners[name]
                try:
                    detail = runner()
                    checks.append({"name": name, "ok": True, "detail": str(detail or "")})
                    print(f"[ui-test] PASS {name} — {detail or ''}")
                except Exception as exc:  # 断言失败/异常都记为失败并继续
                    message = f"{type(exc).__name__}: {exc}"
                    checks.append({"name": name, "ok": False, "detail": message})
                    print(f"[ui-test] FAIL {name} — {message}")
                self.root.update()
        finally:
            passed = sum(1 for check in checks if check["ok"])
            failed = len(checks) - passed
            payload = {
                "ok": failed == 0,
                "total": len(checks),
                "passed": passed,
                "failed": failed,
                "checks": checks,
                "world": {
                    "size": f"{self.config.width}x{self.config.height}",
                    "alive": int(self.world.registry.alive_count()),
                    "total_created": int(self.world.registry.total_created),
                    "total_deaths": int(self.world.registry.total_deaths),
                    "t": round(float(self.world.t), 6),
                    "selection": self.selection.selected_id,
                    "playing": self.controls.playing,
                    "speed": self.controls.speed,
                },
                "elapsed_s": round(time.perf_counter() - started, 3),
                "python": "%d.%d.%d" % __import__("sys").version_info[:3],
                "tk": str(getattr(tk, "TkVersion", "?")),
                "ui_test_path": str(path) if path is not None else None,
            }
            if path is not None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                except Exception as exc:  # pragma: no cover
                    print(f"[ui-test] 写 JSON 失败：{exc}")
                    payload["ok"] = False
            print(
                f"[ui-test] 结果：{passed}/{len(checks)} 项通过"
                + ("" if failed == 0 else f"，{failed} 项失败")
                + (f"，JSON -> {path}" if path is not None else "")
            )
            self.close()
        return 0 if bool(payload.get("ok")) else 1

    # -- 自检辅助 ------------------------------------------------------
    def _press(self, button) -> None:
        button.invoke()
        self.root.update()

    def _clear_selection(self) -> None:
        """测试开始前清空选择（走真实的选择变更路径）。"""
        if self.selection.clear():
            self._on_selection_changed()

    def _inject_click(self, screen_x: float, screen_y: float) -> None:
        self.canvas.event_generate("<Button-1>", x=int(round(screen_x)), y=int(round(screen_y)))
        self.root.update()

    def _agent_box(self, agent):
        for _ in range(30):
            box = self._boxes.get(int(agent.agent_id))
            if box is not None:
                return box
            self.root.update()
            time.sleep(0.01)
        raise AssertionError(f"{agent.agent_id} 号个体没有命中盒（画布未渲染？）")

    def _click_agent(self, agent) -> None:
        box = self._agent_box(agent)
        screen_x, screen_y = self.viewport.world_to_screen(box.x, box.y)
        self._inject_click(screen_x, screen_y)

    def _first_two_alive(self):
        agents = sorted(self._alive_agents(), key=lambda agent: int(agent.agent_id))
        if len(agents) < 2:
            raise AssertionError(f"存活个体不足 2 个（实际 {len(agents)}），无法测试切换")
        return agents[0], agents[1]

    # -- 13 项断言 -----------------------------------------------------
    def _ui_layout(self) -> str:
        layout = self.layout
        rects = layout.rects()
        assert set(rects) == {"top", "world", "panel", "bottom"}
        assert self.panel_visible and rects["panel"][2] > 0
        assert layout.region_at(5, 5) == "top"
        assert layout.region_at(layout.width // 2, layout.height // 2) == "world"
        assert layout.region_at(layout.width - 5, layout.height // 2) == "panel"
        assert layout.region_at(layout.width // 2, layout.height - 5) == "bottom"
        # UI 点击隔离：把「UI 区域」的窗口坐标喂给世界点击处理函数，必须被忽略
        agent = self._alive_agents()[0]
        self._click_agent(agent)
        self.assert_selected(agent.agent_id)
        ui_points = (
            (5, 5),
            (layout.width - 5, layout.height // 2),
            (layout.width // 2, layout.height - 5),
        )
        for point in ui_points:
            assert self._handle_click_at_window(*point) is False, f"{point} 被判成了世界点击"
        self.assert_selected(agent.agent_id)
        return f"top/world/panel/bottom 区域正确，UI 点击被隔离（选中 {agent.agent_id} 号保持不变）"

    def _ui_viewport(self) -> str:
        rect = self.layout.rects()["world"]
        viewport = self.viewport
        assert viewport.world_width == self.config.width
        world_point = (123.5, 456.25)
        screen = viewport.world_to_screen(*world_point)
        back = viewport.screen_to_world(*screen)
        assert abs(back[0] - world_point[0]) < 1e-6 and abs(back[1] - world_point[1]) < 1e-6
        anchor = (rect[2] / 2.0, rect[3] / 2.0)
        before = viewport.screen_to_world(*anchor)
        zoomed = viewport.zoom_at(anchor[0], anchor[1], 1.25)
        after = zoomed.screen_to_world(*anchor)
        assert abs(after[0] - before[0]) < 1e-6 and abs(after[1] - before[1]) < 1e-6, "缩放锚点漂移"
        assert ui_state_scale_ok(zoomed.scale)
        return f"往返误差<1e-6；缩放 {viewport.scale:.2f}→{zoomed.scale:.2f} 光标锚点不动"

    def _ui_click_select(self) -> str:
        self._clear_selection()
        agent = sorted(self._alive_agents(), key=lambda item: int(item.agent_id))[0]
        self._click_agent(agent)
        self.assert_selected(agent.agent_id)
        ring_visible = self.canvas.itemcget(self._ring_item, "state") == "normal"
        return f"点击 {agent.agent_id} 号精灵中心 → 选中，高亮环 {'显示' if ring_visible else '未显示'}"

    def _ui_click_switch(self) -> str:
        self._clear_selection()
        first, second = self._first_two_alive()
        self._click_agent(first)
        self.assert_selected(first.agent_id)
        self._click_agent(second)
        self.assert_selected(second.agent_id)
        return f"选中 {first.agent_id} 号后再点 {second.agent_id} 号 → 选择切换"

    def _ui_click_toggle(self) -> str:
        self._clear_selection()
        agent, _ = self._first_two_alive()
        self._click_agent(agent)
        self.assert_selected(agent.agent_id)
        self._click_agent(agent)  # 再点同一个体 → 取消
        assert self.selection.selected_id is None, "再次点击同一个体没有取消选中"
        return f"再次点击 {agent.agent_id} 号 → 取消选中"

    def _ui_click_empty(self) -> str:
        self._clear_selection()
        agent, _ = self._first_two_alive()
        self._click_agent(agent)
        self.assert_selected(agent.agent_id)
        rect = self.layout.rects()["world"]
        tester = HitTester(self.viewport, max_distance_px=60.0)
        boxes = list(self._boxes.values())
        empty = None
        for y in range(4, max(5, rect[3]), 12):
            for x in range(4, max(5, rect[2]), 12):
                if tester.hit(boxes, x, y) is None:
                    empty = (x, y)
                    break
            if empty is not None:
                break
        assert empty is not None, "画布上找不到远离个体的空白点"
        self._inject_click(*empty)
        assert self.selection.selected_id is None, "点击空白处没有取消选中"
        return f"点击空白处 {empty} → 取消选中"

    def _ui_zoom_pan(self) -> str:
        self._clear_selection()
        agent, _ = self._first_two_alive()
        box = self._agent_box(agent)
        screen_x, screen_y = self.viewport.world_to_screen(box.x, box.y)
        scale_before = self.viewport.scale
        # 真实滚轮事件注入（×1.1 三档）
        for _ in range(3):
            self.canvas.event_generate(
                "<MouseWheel>", x=int(screen_x), y=int(screen_y), delta=120
            )
            self.root.update()
        assert self.viewport.scale > scale_before + 1e-9, "滚轮没有放大视图"
        # 真实右键拖动事件注入
        offset_before = (self.viewport.offset_x, self.viewport.offset_y)
        self.canvas.event_generate("<ButtonPress-3>", x=int(screen_x), y=int(screen_y))
        self.root.update()
        self.canvas.event_generate(
            "<Motion>", x=int(screen_x) - 70, y=int(screen_y) - 40, state=1 << 10
        )
        self.root.update()
        self.canvas.event_generate(
            "<ButtonRelease-3>", x=int(screen_x) - 70, y=int(screen_y) - 40
        )
        self.root.update()
        assert (
            abs(self.viewport.offset_x - offset_before[0]) > 1e-6
            or abs(self.viewport.offset_y - offset_before[1]) > 1e-6
        ), "右键拖动没有产生平移"
        self.assert_selected(None)
        # 平移+缩放之后，按新的屏幕位置点击仍能命中同一世界个体
        self._click_agent(agent)
        self.assert_selected(agent.agent_id)
        return (
            f"滚轮放大 {scale_before:.2f}→{self.viewport.scale:.2f}，右键拖动平移，"
            f"缩放平移后仍点击选中 {agent.agent_id} 号"
        )

    def _ui_pause(self) -> str:
        self._press(self.button_pause)
        assert self.controls.paused, "点击暂停按钮后仍在运行"
        t0 = float(self.world.t)
        for _ in range(6):
            time.sleep(0.04)
            self.root.update()
        assert abs(float(self.world.t) - t0) < 1e-9, (
            f"暂停后世界时间仍在前进：{t0} -> {self.world.t}"
        )
        detail = build_agent_detail(self.world, self.population, self.selection.selected_id)
        assert detail.rows, "暂停时详情面板没有数据"
        stats = build_stats(self.world, self.population)
        assert stats, "暂停时统计没有数据"
        return (
            f"暂停 0.24s 世界时间保持 {t0:.6f}s；"
            f"暂停时详情 {len(detail.rows)} 行、统计 {len(stats)} 行可查看"
        )

    def _ui_single_step(self) -> str:
        assert self.controls.paused, "单步测试要求先处于暂停状态"
        fixed_dt = float(self.world.config.fixed_dt)
        t0 = float(self.world.t)
        self._press(self.button_step)
        t1 = float(self.world.t)
        assert abs((t1 - t0) - fixed_dt) < 1e-9, f"单步应推进 fixed_dt={fixed_dt}，实际 {t1 - t0}"
        self._press(self.button_step)
        t2 = float(self.world.t)
        assert abs((t2 - t1) - fixed_dt) < 1e-9, f"第二次单步推进 {t2 - t1} != {fixed_dt}"
        assert self.controls.paused, "单步不应恢复运行"
        # 运行时单步必须被拒绝
        self._press(self.button_resume)
        assert self.controls.playing
        t3 = float(self.world.t)
        self._press(self.button_step)
        assert abs(float(self.world.t) - t3) < 1e-9, "运行时按单步仍推进了世界"
        self._press(self.button_pause)
        return f"暂停下单步两次各推进 {fixed_dt:.6f}s；运行中单步被拒绝"

    def _ui_list_select(self) -> str:
        self._clear_selection()
        self._refresh_panels()
        if len(self._list_ids) < 2:
            raise AssertionError(f"个体列表不足 2 项：{self._list_ids}")
        index = 1
        target = self._list_ids[index]
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(index)
        self.listbox.event_generate("<<ListboxSelect>>")
        self.root.update()
        self.assert_selected(target)
        return f"点击个体列表第 {index + 1} 项 → 选中 {target} 号（与场景选择同步）"

    def _ui_dead_cleanup(self) -> str:
        self._clear_selection()
        agent, _ = self._first_two_alive()
        self._click_agent(agent)
        self.assert_selected(agent.agent_id)
        alive = self._alive_ids()
        self._apply_alive_ids(set())  # 主循环每 0.25s 走的真实清理路径
        assert self.selection.selected_id is None, "存活集合为空时选中者没有被清理"
        assert self.listbox.size() == 0, "存活集合为空时列表没有同步清空"
        self._apply_alive_ids(alive)
        assert self.listbox.size() == len(alive), "恢复存活集合后列表数量不对"
        return f"存活集合清空 → 选中 {agent.agent_id} 号被自动清理、列表同步；恢复 {len(alive)} 项"

    def _ui_detail_stats(self) -> str:
        agent = sorted(self._alive_agents(), key=lambda item: int(item.agent_id))[0]
        detail = build_agent_detail(self.world, self.population, agent.agent_id)
        labels = {row.label: row.value for row in detail.rows}
        assert labels, "详情没有任何行"
        assert labels["ID"] == str(agent.agent_id)
        assert labels["能量"] != NO_DATA, "真实生理数据（能量）没有被读到"
        assert labels["健康"] == NO_DATA, "世界模型没有 health 字段，必须显示“暂无数据”"
        stats = build_stats(self.world, self.population)
        stat_values = {row.label: row.value for row in stats}
        assert stat_values["存活人口"] == str(self.world.registry.alive_count())
        assert stat_values["累计创建"].startswith(str(self.world.registry.total_created))
        events = build_event_log(self.world, self.population, limit=10)
        assert events, "事件日志为空"
        return (
            f"详情 {len(detail.rows)} 行（能量={labels['能量']}，健康={labels['健康']}），"
            f"统计 {len(stats)} 行，事件日志 {len(events)} 行"
        )

    def _ui_reset(self) -> str:
        self._clear_selection()
        agent, _ = self._first_two_alive()
        self._click_agent(agent)
        self.assert_selected(agent.agent_id)
        old_world = self.world
        before_created = old_world.registry.total_created
        assert self._do_reset(ask=False) is True, "程序化重置失败"
        assert self.world is not old_world, "重置后世界对象没有重建"
        assert float(self.world.t) == 0.0, f"重置后世界时间不是 0：{self.world.t}"
        assert self.selection.selected_id is None, "重置后选择没有清空"
        alive = self._alive_agents()
        assert alive, "重置后没有存活个体"
        assert self.listbox.size() == len(alive), (
            f"重置后列表 {self.listbox.size()} 项 != 存活 {len(alive)}"
        )
        assert self.controls.playing, "重置后没有恢复运行状态"
        assert sorted(self._list_ids) == sorted(int(item.agent_id) for item in alive)
        return (
            f"重置（seed={self.seed + self._reset_count * 101}）→ 新世界 t=0、"
            f"选择清空、列表 {self.listbox.size()} 项、恢复运行（旧累计创建 {before_created}）"
        )

    # -- 断言小工具 ----------------------------------------------------
    def assert_selected(self, expected: int | None) -> None:
        actual = self.selection.selected_id
        if actual != expected:
            raise AssertionError(f"期望选中 {expected}，实际 {actual}")


def ui_state_scale_ok(scale: float) -> bool:
    """缩放必须落在 ui_state 约定的档位范围内。"""
    from .ui_state import MAX_SCALE, MIN_SCALE

    return MIN_SCALE - 1e-9 <= scale <= MAX_SCALE + 1e-9


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def run_app(
    config: WorldConfig,
    *,
    fullscreen: bool = True,
    debug: bool = False,
    screenshot: str | Path | None = None,
    autoclose_ms: int | None = None,
    seed: int = 0,
    policy=None,
    policy_path=None,
    ui_test_path: str | Path | None = None,
) -> int:
    """启动观察窗口（签名与旧版兼容，新增 ``ui_test_path``）。

    返回 0 表示正常退出；2 表示缺少 tkinter/Pillow；``--ui-test`` 模式返回
    0/1 表示自检通过/失败。
    """
    enable_dpi_awareness()
    if not TK_AVAILABLE:  # pragma: no cover - 依赖缺失
        print(f"无法启动窗口：缺少 tkinter 或 Pillow\n{_TK_ERROR or _PIL_ERROR}")
        return 2
    try:
        app = NewWorldApp(
            config,
            fullscreen=fullscreen,
            debug=debug,
            screenshot=screenshot,
            autoclose_ms=autoclose_ms,
            seed=seed,
            policy=policy,
            policy_path=policy_path,
            ui_test_path=ui_test_path,
        )
    except Exception as exc:  # pragma: no cover - 无显示环境
        print(f"无法创建窗口：{exc}")
        return 2
    if ui_test_path is not None:
        return app.run_ui_test(ui_test_path)
    return app.run()


__all__ = ["run_app", "NewWorldApp", "TK_AVAILABLE", "HELP"]
