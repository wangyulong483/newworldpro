"""Documentation preview images: day/night world, creature, Chinese debug."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .autonomy import Mind
from .population import Population
from .config import WorldConfig
from .render import (
    _cjk_font,
    _text_size,
    debug_lines_zh,
    render_frame,
    save_png,
)
from .world import World

DEFAULT_OUTDIR = Path(__file__).resolve().parent.parent / "assets"


def _at_hour(world: World, hour: float) -> None:
    world.t = ((hour - world.day.start_hour) % 24.0) / 24.0 * world.day.day_length_s
    world._update_environment()


def _downscale(frame: Image.Image, width: int) -> Image.Image:
    return frame.resize((width, round(width * frame.height / frame.width)), Image.LANCZOS)


def _caption_panel(frame: Image.Image, caption: str, width: int = 900) -> Image.Image:
    panel = _downscale(frame, width)
    strip = max(38, round(width * 0.05))
    canvas = Image.new("RGB", (width, panel.height + strip), (14, 16, 20))
    canvas.paste(panel, (0, strip))
    draw = ImageDraw.Draw(canvas)
    font = _cjk_font(max(18, round(strip * 0.48)))
    text_width, text_height = _text_size(caption, font)
    draw.text((18, (strip - text_height) // 2 - 1), caption, font=font, fill=(235, 238, 242))
    return canvas


def generate_previews(config: WorldConfig, outdir: str | Path | None = None) -> dict[str, Path]:
    out = Path(outdir) if outdir is not None else DEFAULT_OUTDIR
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    world = World(config, seed=0)
    world.reset(seed=0, positions=[(config.min_x + config.max_x) / 2.0])
    population = Population(world, seed=0)

    # -- daytime world (ender + ender-2) --------------------------------
    _at_hour(world, 12.0)
    for _ in range(240):
        population.step(1.0 / 60.0)
    clean = render_frame(world)
    paths["world_preview"] = save_png(clean, out / "world_preview.png")
    paths["world_preview_1280"] = save_png(
        _downscale(clean, min(1280, config.width)), out / "world_preview_1280.png"
    )

    # -- night world (both sleep) --------------------------------------
    for agent in world.agents:
        agent.physiology.sleep()
        agent.behavior_state = "sleep"
    _at_hour(world, 1.0)
    night = render_frame(world)
    paths["world_night_preview"] = save_png(
        _downscale(night, min(1280, config.width)), out / "world_night_preview.png"
    )

    # -- debug / action shot -------------------------------------------
    for agent in world.agents:
        agent.physiology.wake()
        agent.behavior_state = "explore"
    _at_hour(world, 12.0)
    for _ in range(2200):
        population.step(1.0 / 60.0)
        active = [
            agent
            for agent in world.agents
            if agent.utterance is not None and agent.utterance_until >= world.t
        ]
        if active or any(not agent.on_ground for agent in world.agents):
            if active:
                break
    speech = next(
        (
            agent.utterance
            for agent in world.agents
            if agent.utterance and agent.utterance_until >= world.t
        ),
        None,
    )
    debug_lines = list(debug_lines_zh(world, None, fps=59.6))
    debug_lines.insert(2, f"神经网络：自动训练中 · 人口 {len(world.agents)}")
    debug_lines.insert(3, f"社会：{population.society.summary_zh()}")
    debug_lines.insert(4, f"社区：{world.technology.summary_zh()}")
    debug_frame = render_frame(
        world,
        debug=True,
        debug_lines=tuple(debug_lines),
        speech=speech,
    )
    paths["world_debug_preview"] = save_png(debug_frame, out / "world_debug_preview.png")

    # -- day/night strip -------------------------------------------------
    panels = []
    for hour, caption in ((6.6, "黎明 06:36"), (12.0, "正午 12:00"), (20.5, "夜晚 20:30")):
        for agent in world.agents:
            agent.physiology.wake()
            agent.behavior_state = "explore"
        _at_hour(world, hour)
        frame = render_frame(world)
        panels.append(_caption_panel(frame, caption))
    gap = 12
    strip = Image.new(
        "RGB",
        (panels[0].width, sum(panel.height for panel in panels) + gap * (len(panels) - 1)),
        (8, 10, 14),
    )
    y = 0
    for panel in panels:
        strip.paste(panel, (0, y))
        y += panel.height + gap
    paths["world_daynight_preview"] = save_png(strip, out / "world_daynight_preview.png")
    return paths
