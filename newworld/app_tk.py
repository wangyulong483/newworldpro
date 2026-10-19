"""Observer window for the newworld society (Chinese UI).

The world now contains several inhabitants (ender and ender-2 plus children).
Each has its own neural behaviour and body; a society network aggregates their
relations.  Opening the program automatically starts continuous training of all
agents' networks in a background thread; **F2 starts from zero** (new random
weights for every agent).
"""

from __future__ import annotations

import ctypes
import threading
import time
from pathlib import Path

from .config import WorldConfig, enable_dpi_awareness
from .population import Population
from .render import (
    build_debug_panel,
    build_hint,
    build_shadow,
    character_layer,
    debug_lines_zh,
    render_world,
    speech_bubble,
)
from .world import World

HELP = (
    "newworld 已启动：ender 与 ender-2 在像素自然世界里生活（自动连续训练中）。\n"
    "  空格 暂停 · F1 中文调试层 · F2 从零训练所有 agent 的神经网络 · Q / Esc 关闭"
)


def _disable_ime(hwnd: int) -> None:
    try:
        ctypes.windll.imm32.ImmAssociateContext(hwnd, 0)
    except Exception:  # pragma: no cover
        pass


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
) -> int:
    enable_dpi_awareness()
    try:
        import tkinter as tk
        from PIL import Image, ImageGrab, ImageTk
    except ImportError as exc:  # pragma: no cover
        print(f"无法启动窗口：缺少 tkinter 或 Pillow\n{exc}")
        return 2

    cfg = config
    world = World(cfg, seed=seed)
    world.reset(seed=seed, positions=[cfg.width / 2.0])
    population = Population(world, seed)

    root = tk.Tk()
    root.title("newworld society")
    root.configure(bg="#0b0d10")
    if screenshot:
        root.attributes("-topmost", True)
    if fullscreen:
        root.attributes("-fullscreen", True)
    else:
        root.geometry(f"{cfg.width}x{cfg.height}+60+60")
    canvas = tk.Canvas(root, width=cfg.width, height=cfg.height, highlightthickness=0, bd=0, bg="#0b0d10")
    canvas.pack(fill="both", expand=True)

    background_photo = ImageTk.PhotoImage(render_world(world))
    bg_item = canvas.create_image(0, 0, anchor="nw", image=background_photo)
    empty = ImageTk.PhotoImage(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    panel_item = canvas.create_image(16, 16, anchor="nw", image=empty, state="hidden")
    hint_photo = ImageTk.PhotoImage(build_hint(cfg))
    hint_item = canvas.create_image(cfg.width - 20, 16, anchor="ne", image=hint_photo, state="normal")

    visuals: dict[int, dict] = {}
    state = {
        "paused": False,
        "debug": debug,
        "hint": True,
        "closed": False,
        "panel_photo": empty,
        "last_world": -1.0,
        "training": False,
        "train_status": "启动后自动连续训练" if screenshot else "启动中…",
        "train_mode": "",
        "reset_requested": False,
        "sessions": 0,
    }

    # ------------------------------------------------------------------
    def sync_visuals() -> None:
        alive = {agent.agent_id for agent in world.agents if agent.alive}
        for agent_id in list(visuals):
            if agent_id not in alive:
                for item in visuals[agent_id]["items"].values():
                    canvas.delete(item)
                del visuals[agent_id]
        for agent in world.agents:
            if not agent.alive or agent.agent_id in visuals:
                continue
            shadow_photo = ImageTk.PhotoImage(build_shadow(world))
            shadow_item = canvas.create_image(0, 0, anchor="nw", image=shadow_photo)
            layer, anchor = character_layer(world, agent)
            char_photo = ImageTk.PhotoImage(layer)
            char_item = canvas.create_image(0, 0, anchor="nw", image=char_photo)
            bubble_photo = ImageTk.PhotoImage(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
            bubble_item = canvas.create_image(0, 0, anchor="nw", image=bubble_photo, state="hidden")
            visuals[agent.agent_id] = {
                "shadow_photo": shadow_photo,
                "shadow_item": shadow_item,
                "char_photo": char_photo,
                "char_item": char_item,
                "anchor": anchor,
                "bubble_photo": bubble_photo,
                "bubble_item": bubble_item,
                "bubble_text": None,
                "items": {"shadow": shadow_item, "char": char_item, "bubble": bubble_item},
            }

    def refresh_background() -> None:
        nonlocal background_photo
        background_photo = ImageTk.PhotoImage(render_world(world))
        canvas.itemconfigure(bg_item, image=background_photo)

    def refresh_agent(agent) -> None:
        visual = visuals.get(agent.agent_id)
        if visual is None:
            return
        feet_y = agent.y if agent.y is not None else cfg.ground_y
        shadow_photo = visual["shadow_photo"]
        canvas.coords(
            visual["shadow_item"],
            int(agent.x - shadow_photo.width() / 2),
            int(feet_y - shadow_photo.height() / 2),
        )
        layer, anchor = character_layer(world, agent)
        visual["anchor"] = anchor
        visual["char_photo"] = ImageTk.PhotoImage(layer)
        canvas.itemconfigure(visual["char_item"], image=visual["char_photo"])
        canvas.coords(
            visual["char_item"],
            int(agent.x - anchor[0]),
            int(feet_y - anchor[1]),
        )
        text = agent.utterance if agent.utterance_until >= world.t - 1e-9 else None
        bubble = visual["bubble_item"]
        if not text:
            canvas.itemconfigure(bubble, state="hidden")
            visual["bubble_text"] = None
            return
        if text != visual["bubble_text"]:
            visual["bubble_photo"] = ImageTk.PhotoImage(speech_bubble(text))
            visual["bubble_text"] = text
            canvas.itemconfigure(bubble, image=visual["bubble_photo"])
        photo = visual["bubble_photo"]
        bx = int(agent.x - photo.width() / 2)
        by = int(feet_y - anchor[1] - photo.height() - 8)
        canvas.coords(bubble, max(0, min(cfg.width - photo.width(), bx)), max(0, by))
        canvas.itemconfigure(bubble, state="normal")

    def refresh_panel(fps: float) -> None:
        lines = debug_lines_zh(world, None, fps=fps, paused=state["paused"])
        neural_state = "训练中" if state["training"] else "已加载"
        lines.insert(2, f"神经网络：{neural_state} · {state['train_status']}")
        lines.insert(3, f"人口：{population.summary_zh()}")
        relations = "  ".join(population.society.relation_lines(world)[:2])
        if relations:
            lines.insert(4, f"关系：{relations}")
        panel = build_debug_panel(world, lines)
        photo = ImageTk.PhotoImage(panel)
        state["panel_photo"] = photo
        canvas.itemconfigure(panel_item, image=photo)
        canvas.itemconfigure(panel_item, state="normal" if state["debug"] else "hidden")

    def close() -> None:
        if state["closed"]:
            return
        state["closed"] = True
        root.destroy()

    def on_escape(_event=None) -> None:
        close()

    def on_space(_event=None) -> None:
        state["paused"] = not state["paused"]
        print("已暂停" if state["paused"] else "继续")

    def on_debug(_event=None) -> None:
        state["debug"] = not state["debug"]
        canvas.itemconfigure(panel_item, state="normal" if state["debug"] else "hidden")
        print("调试层：开" if state["debug"] else "调试层：关")

    def on_train(_event=None) -> None:
        """F2: every agent starts from a fresh random network."""
        state["reset_requested"] = True
        state["train_status"] = "已请求：从零训练所有 agent（等待当前一轮）"
        print("F2：已请求从零训练所有 agent 的神经网络")

    TRAIN_INTERVAL = 30.0

    def training_worker() -> None:
        while not state["closed"]:
            reset = state["reset_requested"]
            state["reset_requested"] = False
            state["training"] = True
            state["train_mode"] = "reset" if reset else "continue"
            state["train_status"] = (
                "从零训练中（所有 agent 新随机权重）…" if reset else "自动连续训练中…"
            )
            try:
                metrics = population.train_round(seed + state["sessions"], reset=reset)
                state["sessions"] += 1
                label = "从零训练完成" if reset else "连续训练完成"
                state["train_status"] = (
                    f"{label}·第{state['sessions']}轮·{int(metrics['agents'])}个agent "
                    f"模式准确率 {metrics['mode_accuracy']:.2f}"
                )
                print(f"神经网络：{state['train_status']}")
            except Exception as exc:  # pragma: no cover - GUI feedback
                state["train_status"] = f"训练失败：{exc}"
                print(f"神经网络：训练失败：{exc}")
            finally:
                state["training"] = False
            deadline = time.time() + TRAIN_INTERVAL
            while time.time() < deadline and not state["closed"] and not state["reset_requested"]:
                time.sleep(0.2)

    def hide_hint(_event=None) -> None:
        if state["hint"]:
            state["hint"] = False
            canvas.itemconfigure(hint_item, state="hidden")

    root.bind("<KeyPress>", hide_hint)
    root.bind("<Escape>", on_escape)
    root.bind("<KeyPress-q>", on_escape)
    root.bind("<KeyPress-Q>", on_escape)
    root.bind("<space>", on_space)
    root.bind("<F1>", on_debug)
    root.bind("<F2>", on_train)
    root.bind("<Button-1>", lambda _event: root.focus_force())
    root.protocol("WM_DELETE_WINDOW", close)

    sync_visuals()
    if screenshot is None:
        threading.Thread(target=training_worker, daemon=True).start()

    clock = time.perf_counter
    start = clock()
    last = start
    fps = 0.0
    frames = 0
    frames_since = 0
    window_start = start
    last_panel = 0.0
    screenshot_taken = False
    autoclose_at = start + autoclose_ms / 1000.0 if autoclose_ms else None

    def grab_screenshot() -> None:
        try:
            path = Path(screenshot)  # type: ignore[arg-type]
            path.parent.mkdir(parents=True, exist_ok=True)
            x0, y0 = canvas.winfo_rootx(), canvas.winfo_rooty()
            width = canvas.winfo_width() or cfg.width
            height = canvas.winfo_height() or cfg.height
            ImageGrab.grab(bbox=(x0, y0, x0 + width, y0 + height)).save(path)
            print(f"截图已保存：{path}")
        except Exception as exc:  # pragma: no cover
            print(f"截图失败：{exc}")

    def tick() -> None:
        nonlocal last, fps, frames, frames_since, window_start, last_panel, screenshot_taken
        if state["closed"]:
            return
        now = clock()
        dt = min(max(now - last, 0.0), 0.05)
        last = now
        if not state["paused"]:
            population.step(dt)
            sync_visuals()
            for agent in world.agents:
                if agent.alive:
                    refresh_agent(agent)
            if now - state["last_world"] > 0.8:
                state["last_world"] = now
                refresh_background()
        frames += 1
        frames_since += 1
        if now - window_start >= 0.5:
            fps = frames_since / (now - window_start)
            frames_since = 0
            window_start = now
        if now - start > 7.0 and state["hint"]:
            hide_hint()
        if state["debug"] and now - last_panel > 0.3:
            last_panel = now
            refresh_panel(fps)
        if screenshot is not None and not screenshot_taken and now - start > 0.9:
            screenshot_taken = True
            grab_screenshot()
            close()
            return
        if autoclose_at is not None and now >= autoclose_at:
            close()
            return
        root.after(16, tick)

    root.after(0, tick)
    root.focus_force()

    def announce() -> None:
        print(HELP)
        print(f"窗口 {canvas.winfo_width()}x{canvas.winfo_height()}，世界 {cfg.width}x{cfg.height}")
        _disable_ime(int(root.winfo_id()))
        root.lift()
        root.focus_force()

    root.after(250, announce)
    try:
        root.mainloop()
    finally:
        elapsed = max(1e-6, clock() - start)
        print(f"离开 newworld：{frames} 帧 / {elapsed:.1f}s，平均 {frames / elapsed:.1f} fps；{population.summary_zh()}")
    return 0
