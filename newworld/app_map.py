"""俯视地图世界的 GUI（V3 世界盒子）：画布 + 缩放平移 + 上帝之手。

复用 :mod:`newworld.ui_state.Viewport`（世界↔屏幕变换、滚轮缩放、
拖动平移）与旧 GUI 的布局思想（控制栏/右侧面板/状态栏），但世界
渲染、选择逻辑、交互全部是新的：

* 画布：``MapRenderer`` 的底图 PhotoImage + 群伙散点 Canvas 矩形；
* 上帝之手工具栏：观察 / 陨石 / 闪电 / 洪水 / 治愈，点击地图施放；
* 右侧面板：人口统计（人口/出生死亡/死因/食物/地形分布）+ 灾害史；
* 底部控制：开始/暂停/单步/速度/重置（沿用旧交互习惯）。

运行::

    python -m newworld            # 新世界（俯视地图）
    python -m newworld --classic # 旧世界（侧视像素小人）
"""

from __future__ import annotations

import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageTk

try:
    import tkinter  # noqa: F401
    TK_AVAILABLE = True
except ImportError:  # pragma: no cover
    TK_AVAILABLE = False

from .catastrophes import Catastrophes
from .map_engine import MapConfig, MapEngine, TERRAIN_INFO
from .map_render import MapRenderer
from .swarm import Swarm, SwarmConfig
from .ui_state import Viewport

#: 上帝之手的工具（标签 → 内部名）
TOOLS: tuple[tuple[str, str], ...] = (
    (" 观察 ", "observe"),
    (" ☄ 陨石 ", "meteor"),
    (" ⚡ 闪电 ", "lightning"),
    (" 🌊 洪水 ", "flood"),
    (" ✚ 治愈 ", "heal"),
)

#: 仿真速度档（每 tick 推进的物理步数）
SPEEDS: tuple[int, ...] = (1, 2, 4, 8, 16)


@dataclass
class MapAppConfig:
    """GUI 参数。"""

    width: int = 512
    height: int = 320
    cell_px: int = 3
    seed: int = 0
    window_title: str = "newworld · 像素世界盒子"
    tick_ms: int = 33  # ~30 ticks/s


class MapWorldApp:
    """俯视地图世界的主窗口。"""

    def __init__(self, master: tk.Tk, config: MapAppConfig | None = None) -> None:
        self.config = config or MapAppConfig()
        cfg = self.config
        self.master = master
        master.title(cfg.window_title)
        master.configure(bg="#101216")

        # -- 世界 ---------------------------------------------------
        self.engine = MapEngine(MapConfig(seed=cfg.seed, width=cfg.width, height=cfg.height))
        self.swarm = Swarm(self.engine, SwarmConfig(), seed=cfg.seed)
        self.cats = Catastrophes(self.engine, self.swarm)
        self.renderer = MapRenderer(self.engine, cell_px=cfg.cell_px)

        # -- 状态 ----------------------------------------------------
        self.running = False
        self.speed_index = 1
        self.selected_tool = "observe"
        self._drag_last: tuple[int, int] | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._agent_ids: list[int] = []
        self._frames = 0
        self._fps_clock = time.perf_counter()
        self.fps = 0.0

        self._build_ui()
        self._bind_events()
        self._refresh_all(force=True)

    # ==================================================================
    # UI 构建
    # ==================================================================
    def _build_ui(self) -> None:
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(1, weight=1)

        # -- 顶部：上帝之手工具栏 ------------------------------------
        toolbar = tk.Frame(self.master, bg="#1a1e26")
        toolbar.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        self.tool_buttons: dict[str, tk.Button] = {}
        for label, name in TOOLS:
            state = tk.DISABLED if name != "observe" else tk.NORMAL
            button = tk.Button(
                toolbar, text=label, relief=tk.FLAT, font=("Microsoft YaHei UI", 10),
                bg="#2a3140" if name != self.selected_tool else "#3d6ea5",
                fg="#e8ecf2", padx=8, pady=2, state=state,
                command=lambda n=name: self._select_tool(n),
            )
            button.pack(side=tk.LEFT, padx=3)
            self.tool_buttons[name] = button

        # -- 中部：地图画布 + 右侧面板 --------------------------------
        middle = tk.Frame(self.master, bg="#101216")
        middle.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)
        middle.columnconfigure(0, weight=1)
        middle.rowconfigure(0, weight=1)

        canvas_frame = tk.Frame(middle, bg="#0a0c10")
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        self.canvas = tk.Canvas(
            canvas_frame, bg="#07090c", highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        panel = tk.Frame(middle, bg="#161a22", width=270)
        panel.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        panel.pack_propagate(False)
        self._build_panel(panel)

        # -- 底部：控制栏 + 状态栏 ------------------------------------
        bottom = tk.Frame(self.master, bg="#1a1e26")
        bottom.grid(row=2, column=0, sticky="ew", padx=4, pady=2)
        self.start_button = tk.Button(bottom, text=" ▶ 开始 ", command=self._toggle_run,
                                       relief=tk.FLAT, bg="#2f5940", fg="#e8ecf2",
                                       font=("Microsoft YaHei UI", 10), padx=10, pady=3)
        self.start_button.pack(side=tk.LEFT, padx=3)
        tk.Button(bottom, text=" 单步 ", command=self._step_once, relief=tk.FLAT,
                  bg="#2a3140", fg="#e8ecf2", font=("Microsoft YaHei UI", 10),
                  padx=10, pady=3).pack(side=tk.LEFT, padx=3)
        self.speed_label = tk.Label(bottom, text="", bg="#1a1e26", fg="#9aa4b2",
                                    font=("Microsoft YaHei UI", 10))
        self.speed_label.pack(side=tk.LEFT, padx=10)
        tk.Button(bottom, text=" 重置 ", command=self._reset_world, relief=tk.FLAT,
                  bg="#5a3040", fg="#e8ecf2", font=("Microsoft YaHei UI", 10),
                  padx=10, pady=3).pack(side=tk.RIGHT, padx=3)

        self.status = tk.Label(self.master, text="", anchor="w", bg="#12151b",
                                fg="#c8d0da", font=("Microsoft YaHei UI", 9),
                                padx=8, pady=2)
        self.status.grid(row=3, column=0, sticky="ew")

    def _build_panel(self, parent: tk.Frame) -> None:
        """右侧信息面板（人口/地图/灾害/事件）。"""
        self.stat_text = tk.Text(parent, bg="#161a22", fg="#c8d0da", relief=tk.FLAT,
                                 font=("Consolas", 9), height=26, wrap=tk.NONE,
                                 state=tk.DISABLED)
        self.stat_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ==================================================================
    # 事件
    # ==================================================================
    def _bind_events(self) -> None:
        self.canvas.bind("<Configure>", lambda e: self._on_resize())
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self._drag_stop())
        # 中键或右键平移；滚轮缩放
        self.canvas.bind("<Button-2>", self._pan_start)
        self.canvas.bind("<B2-Motion>", self._pan_move)
        self.canvas.bind("<Button-3>", self._pan_start)
        self.canvas.bind("<B3-Motion>", self._pan_move)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.master.bind("<space>", lambda e: self._toggle_run())
        self.master.bind("<Escape>", lambda e: self.master.destroy())
        self.master.bind("<plus>", lambda e: self._change_speed(1))
        self.master.bind("<equal>", lambda e: self._change_speed(1))
        self.master.bind("<minus>", lambda e: self._change_speed(-1))

    def _viewport(self) -> Viewport | None:
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        world_w = self.engine.width * self.config.cell_px
        world_h = self.engine.height * self.config.cell_px
        scale = min(canvas_w / world_w, canvas_h / world_h)
        if not hasattr(self, "_vp") or self._vp is None:
            self._vp = Viewport(world_w, world_h, canvas_w, canvas_h, scale=scale)
        else:
            self._vp = Viewport(world_w, world_h, canvas_w, canvas_h,
                                scale=self._vp.scale, offset_x=self._vp.offset_x,
                                offset_y=self._vp.offset_y)
        return self._vp

    def _on_resize(self) -> None:
        self._refresh_all(force=True)

    def _on_click(self, event) -> None:
        """点击地图：按当前工具施放（观察模式只统计位置）。"""
        scale = getattr(self, "_draw_scale", 1.0)
        ox = getattr(self, "_draw_ox", 0)
        oy = getattr(self, "_draw_oy", 0)
        wx = (event.x - ox) / scale
        wy = (event.y - oy) / scale
        cx = int(wx // self.config.cell_px)
        cy = int(wy // self.config.cell_px)
        tool = self.selected_tool
        if tool == "meteor":
            self.cats.meteor(cx, cy)
        elif tool == "lightning":
            self.cats.lightning(cx, cy)
        elif tool == "flood":
            self.cats.flood(cx, cy)
        elif tool == "heal":
            self.cats.heal(cx, cy)
        self._refresh_all(force=True)

    def _on_wheel(self, event) -> None:
        """滚轮：一期不做视口缩放（整图可视），留给二期。"""
        self._change_speed(1 if event.delta > 0 else -1)

    def _on_drag(self, event) -> None:
        """左键拖动：一期不做视口平移（工具施放优先）。"""

    def _drag_stop(self) -> None:
        """结束拖动（一期无操作）。"""

    def _pan_start(self, event) -> None:
        """中/右键按下（一期无操作）。"""

    def _pan_move(self, event) -> None:
        """中/右键拖动（一期无操作）。"""

    def _on_wheel(self, event) -> None:
        vp = self._viewport()
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self._vp = vp.zoom_at(event.x, event.y, factor)
        self._refresh_all()

    # ==================================================================
    # 控制
    # ==================================================================
    def _select_tool(self, name: str) -> None:
        self.selected_tool = name
        for tool_name, button in self.tool_buttons.items():
            button.configure(
                bg="#3d6ea5" if tool_name == name else "#2a3140",
                state=tk.NORMAL if tool_name == "observe" or True else tk.DISABLED,
            )

    def _toggle_run(self) -> None:
        self.running = not self.running
        self.start_button.configure(
            text=" ⏸ 暂停 " if self.running else " ▶ 开始 ",
            bg="#5a4a2f" if self.running else "#2f5940",
        )
        if self.running:
            self._tick()

    def _step_once(self) -> None:
        self._advance(SPEEDS[self.speed_index])
        self._refresh_all()

    def _change_speed(self, delta: int) -> None:
        self.speed_index = max(0, min(len(SPEEDS) - 1, self.speed_index + delta))
        self._refresh_status()

    def _reset_world(self) -> None:
        self.engine = MapEngine(MapConfig(seed=self.config.seed,
                                          width=self.config.width, height=self.config.height))
        self.swarm = Swarm(self.engine, SwarmConfig(), seed=self.config.seed)
        self.cats = Catastrophes(self.engine, self.swarm)
        self.renderer = MapRenderer(self.engine, cell_px=self.config.cell_px)
        self._vp = None
        self._refresh_all(force=True)

    # ==================================================================
    # 主循环
    # ==================================================================
    def _tick(self) -> None:
        if not self.running:
            return
        self._advance(SPEEDS[self.speed_index])
        self._refresh_all()
        self.master.after(self.config.tick_ms, self._tick)

    def _advance(self, steps: int) -> None:
        for _ in range(steps):
            self.engine.update(0.05, 0.8)
            self.swarm.update(0.05)
        self.renderer.refresh()

    # ==================================================================
    # 渲染刷新
    # ==================================================================
    def _refresh_all(self, force: bool = False) -> None:
        self._draw_map()
        self._draw_agents()
        self._refresh_status()
        self._refresh_panel()

    def _draw_map(self) -> None:
        """把整张地图缩放到画布内并居中（一期：整图可视，世界盒子式）。"""
        display = self.renderer.display
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        if canvas_w <= 1 or canvas_h <= 1:
            # 尚未完成布局：用请求尺寸
            canvas_w = max(1, self.canvas.winfo_reqwidth())
            canvas_h = max(1, int(canvas_w * display.height / max(1, display.width)))
        scale = min(canvas_w / display.width, canvas_h / display.height)
        if scale <= 0:
            scale = 1.0
        self._draw_scale = scale
        sw = max(1, int(display.width * scale))
        sh = max(1, int(display.height * scale))
        image = display.resize((sw, sh), Image.NEAREST)
        photo = ImageTk.PhotoImage(image)
        self._photo = photo  # 防 GC
        self._draw_ox = (canvas_w - sw) // 2
        self._draw_oy = (canvas_h - sh) // 2
        self.canvas.delete("map")
        self.canvas.create_image(self._draw_ox, self._draw_oy, image=photo,
                                 anchor="nw", tags="map")
        self.canvas.tag_lower("map")

    def _draw_agents(self) -> None:
        """群伙散点（使用 _draw_map 计算的缩放与偏移）。"""
        scale = getattr(self, "_draw_scale", 1.0)
        ox = getattr(self, "_draw_ox", 0)
        oy = getattr(self, "_draw_oy", 0)
        points = self.renderer.agent_points(self.swarm)
        self.canvas.delete("agent")
        if not points:
            return
        size = max(2, int(round(scale * 2)))
        # 降采样：超过 6000 点时隔点画（视觉密度不变）
        step = 2 if len(points) > 6000 else 1
        for (x, y, colour) in points[::step]:
            sx = x * scale + ox
            sy = y * scale + oy
            hex_colour = "#%02x%02x%02x" % colour
            self.canvas.create_rectangle(
                sx, sy, sx + size, sy + size, fill=hex_colour, outline="", tags="agent"
            )

    def _refresh_status(self) -> None:
        swarm = self.swarm
        engine = self.engine
        self.status.configure(
            text=(
                f"人口 {swarm.population}  出生 {swarm.births}  死亡 {swarm.deaths}  "
                f"食物 {engine.food_capacity():.0f}  燃烧 {int(engine.burning.sum())}  "
                f"速度 ×{SPEEDS[self.speed_index]}  "
                f"{self.fps:.0f} FPS  工具：{self._tool_name()}"
            )
        )

    def _tool_name(self) -> str:
        for label, name in TOOLS:
            if name == self.selected_tool:
                return label.strip()
        return "?"

    def _refresh_panel(self) -> None:
        swarm = self.swarm
        engine = self.engine
        summary = engine.terrain_summary()
        death = (
            f"饿死 {swarm.starvation_deaths}  老死 {swarm.old_age_deaths}  "
            f"溺亡 {swarm.drowned}  烧亡 {swarm.burned}\n"
        )
        lines = [
            "── 人口 ──",
            f"当前 {swarm.population}   承载力≈{swarm.carrying_capacity_estimate()}",
            f"累计出生 {swarm.births}   累计死亡 {swarm.deaths}",
            death,
            "── 地图 ──",
        ]
        for name, count in summary.items():
            lines.append(f"{name}  {count}")
        lines.append(f"食物总量 {engine.food_capacity():.0f}")
        lines.append("")
        lines.append("── 灾害史 ──")
        for event in list(self.cats.history)[-8:]:
            names = {"meteor": "陨石", "lightning": "闪电", "flood": "洪水", "heal": "治愈"}
            killed = event.killed
            effect = f"治愈{-killed}" if killed < 0 else f"击杀{killed}"
            lines.append(
                f"[{event.t:7.1f}s] {names.get(event.kind, event.kind)}"
                f" @({event.x},{event.y}) {effect}"
            )
        if not self.cats.history:
            lines.append("（无）")
        self.stat_text.configure(state=tk.NORMAL)
        self.stat_text.delete("1.0", tk.END)
        self.stat_text.insert("1.0", "\n".join(lines))
        self.stat_text.configure(state=tk.DISABLED)

    # ==================================================================
    def run(self) -> None:
        """进入主循环（含 FPS 统计）。"""
        self._fps_loop()
        self.master.mainloop()

    def _fps_loop(self) -> None:
        now = time.perf_counter()
        self._frames += 1
        elapsed = now - self._fps_clock
        if elapsed >= 1.0:
            self.fps = self._frames / elapsed
            self._frames = 0
            self._fps_clock = now
            self._refresh_status()
        self.master.after(200, self._fps_loop)


def run_map_app(
    *,
    seed: int = 0,
    width: int = 512,
    height: int = 320,
    cell_px: int = 3,
    screenshot: str | Path | None = None,
    autoclose_ms: int | None = None,
) -> int:
    """启动俯视地图世界（CLI 入口调用）。

    ``screenshot`` / ``autoclose_ms`` 为自动化验证留的钩子。
    """
    if not TK_AVAILABLE:  # pragma: no cover
        print("无法启动窗口：缺少 tkinter")
        return 2
    root = tk.Tk()
    app = MapWorldApp(root, MapAppConfig(seed=seed, width=width, height=height, cell_px=cell_px))
    root.geometry("1280x800")

    if screenshot or autoclose_ms:
        def _auto_close() -> None:
            if screenshot:
                try:
                    import io

                    x = root.winfo_rootx() + app.canvas.winfo_x()
                    y = root.winfo_rooty() + app.canvas.winfo_y()
                    ps = app.canvas.postscript()
                    # 用 PIL 从 canvas 内容抓图更可靠：直接抓主窗
                    from PIL import ImageGrab

                    img = ImageGrab.grab((x, y, x + app.canvas.winfo_width(),
                                          y + app.canvas.winfo_height()))
                    path = Path(screenshot)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(path)
                    print(f"screenshot -> {path}")
                except Exception as exc:  # pragma: no cover
                    print(f"screenshot failed: {exc}")
            root.destroy()

        if not app.running:
            app._toggle_run()
        delay = int(autoclose_ms) if autoclose_ms else 1500
        # 先跑一小段让人口动起来再截屏
        for _ in range(200):
            app._advance(1)
        root.after(delay, _auto_close)

    app.run()
    return 0
