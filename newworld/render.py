"""渲染：像素原子世界、分部位生物、中文调试覆盖层。

渲染管线：原子网格 → RGB 图像（pixelworld.render_atoms）→ 最近邻
放大到屏幕 → 叠加社区结构 → 角色（分部位的 BodyRenderer）→ 阴影
→ 语言气泡 → 调试面板/提示条。全部用 PIL，不依赖 tkinter。
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .body import ATOM_SIZE, BODY_W, BODY_H, BodyRenderer, Pose
from .config import WORLD_NAME, WORLD_VERSION, WorldConfig
from .world import World


def _cjk_font(size: int) -> ImageFont.ImageFont:
    """加载一个支持中文的系统字体（微软雅黑 → 宋体 → 默认）。"""
    for path in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _text_size(text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """测量文本的像素尺寸。"""
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


# ---------------------------------------------------------------------------
# 世界 + 生物
# ---------------------------------------------------------------------------
def render_world(world: World) -> Image.Image:
    """世界图像（地形、物体、天空、阴影、社区结构）。"""
    atoms = world.pixel_scene.render_atoms(world.day, world.hour)
    image = Image.fromarray(atoms, mode="RGB")
    image = image.resize((world.config.width, world.config.height), Image.NEAREST)
    # 旧版社区的简易像素画（材料堆/小屋/篝火）
    community = getattr(getattr(world, "technology", None), "community", None)
    if community is not None:
        draw = ImageDraw.Draw(image)
        cx = int(community.x_screen)
        ground = int(world.config.ground_y)
        # 材料堆随社区积累的资源变大
        if community.wood or community.stone:
            draw.rectangle((cx - 80, ground - 14, cx - 30, ground - 4), fill=(120, 78, 44))
        if community.structures >= 1:
            # 简易像素小屋
            draw.rectangle((cx - 34, ground - 70, cx + 34, ground - 6), fill=(122, 82, 50))
            draw.polygon(((cx - 44, ground - 68), (cx, ground - 105), (cx + 44, ground - 68)), fill=(156, 102, 58))
            draw.rectangle((cx - 10, ground - 34, cx + 8, ground - 6), fill=(54, 40, 30))
        if community.campfire:
            draw.rectangle((cx + 80, ground - 18, cx + 92, ground - 6), fill=(110, 68, 36))
            draw.ellipse((cx + 78, ground - 28, cx + 96, ground - 8), fill=(238, 150, 52))
            draw.ellipse((cx + 82, ground - 28, cx + 92, ground - 16), fill=(255, 220, 120))
    return image


def pose_for(world: World, character=None) -> Pose:
    """按角色当前状态选择身体姿态（睡/跳/吃/喝/走/站）。"""
    character = character if character is not None else world.agent
    facing = character.facing
    if character.physiology.sleeping:
        return Pose("sleep", 0.0, facing)
    if not character.on_ground:
        return Pose("jump", 0.0, facing)
    if character.behavior_state == "eat":
        return Pose("eat", character.phase, facing)
    if character.behavior_state == "drink":
        return Pose("drink", character.phase, facing)
    if abs(character.vx) > 5.0:
        return Pose("walk", character.phase, facing)
    return Pose("idle", 0.0, facing)


def character_layer(world: World, character=None) -> tuple[Image.Image, tuple[int, int]]:
    """渲染一个居民（艺术像素放大）并返回其在屏幕空间的锚点。"""
    character = character if character is not None else world.agent
    body = BodyRenderer(variant=getattr(character, "sex", "male")).render(pose_for(world, character))
    # 孩子体型小（body_scale 0.65），按比例缩小
    scale = ATOM_SIZE * max(0.4, getattr(character, "body_scale", 1.0))
    image = body.image.resize(
        (max(1, int(body.image.width * scale)), max(1, int(body.image.height * scale))),
        Image.NEAREST,
    )
    return image, (int(body.anchor[0] * scale), int(body.anchor[1] * scale))


def build_shadow(world: World) -> Image.Image:
    """跟随生物的柔性接触阴影（大小随日光变化）。"""
    cfg = world.config
    daylight = world.daylight
    width = int(max(40, BODY_W * ATOM_SIZE * (1.1 + 0.6 * daylight)))
    height = max(10, int(cfg.cell * 2.2))
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    alpha = int(30 + 90 * daylight)
    draw.ellipse((0, 0, width - 1, height - 1), fill=(20, 16, 12, alpha))
    return image


def speech_bubble(text: str) -> Image.Image:
    """画一个带尾巴的圆角语言气泡（内容居中）。"""
    font = _cjk_font(28)
    padding_x, padding_y = 16, 8
    text_width, text_height = _text_size(text, font)
    width = text_width + padding_x * 2
    height = text_height + padding_y * 2 + 12
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    body_h = height - 12
    draw.rounded_rectangle(
        (0, 0, width - 1, body_h), radius=14, fill=(250, 250, 252, 235), outline=(110, 110, 120, 220), width=2
    )
    draw.polygon(((width * 0.4, body_h - 2), (width * 0.5, height - 1), (width * 0.6, body_h - 2)),
                 fill=(250, 250, 252, 235))
    draw.text(((width - text_width) / 2, (body_h - text_height) / 2), text, font=font, fill=(40, 44, 50, 255))
    return image


def render_frame(
    world: World,
    *,
    debug: bool = False,
    debug_lines: tuple[str, ...] = (),
    speech: str | None = None,
    fps: float = 0.0,
    paused: bool = False,
    background: Image.Image | None = None,
) -> Image.Image:
    """合成一整屏画面（headless 录像与预览用）。

    顺序：背景（世界或调用方给的图）→ 每个存活角色的阴影/身体/
    语言气泡 → 可选的中文调试面板。
    """
    frame = render_world(world) if background is None else background.copy()
    shadow = build_shadow(world)
    for character in world.agents:
        if not getattr(character, "alive", True):
            continue
        feet_x = character.x
        feet_y = character.y if character.y is not None else world.config.ground_y
        frame.paste(shadow, (int(feet_x - shadow.width / 2), int(feet_y - shadow.height / 2)), shadow)
        layer, anchor = character_layer(world, character)
        frame.paste(layer, (int(feet_x - anchor[0]), int(feet_y - anchor[1])), layer)
        utterance = (
            character.utterance
            if character.utterance is not None and character.utterance_until >= world.t
            else None
        )
        if utterance or (character is world.agent and speech):
            text = utterance or speech
            bubble = speech_bubble(text)
            bx = int(feet_x - bubble.width / 2)
            by = int(feet_y - anchor[1] - bubble.height - 8)
            bx = max(0, min(frame.width - bubble.width, bx))
            by = max(0, by)
            frame.paste(bubble, (bx, by), bubble)

    if debug:
        lines = list(debug_lines) or debug_lines_zh(world, None, fps=fps, paused=paused)
        panel = build_debug_panel(world, lines)
        margin = max(14, world.config.width // 90)
        frame.paste(panel, (margin, margin), panel)
    return frame


# ---------------------------------------------------------------------------
# 调试覆盖层（中文）
# ---------------------------------------------------------------------------
def debug_lines_zh(world: World, mind, *, fps: float = 0.0, paused: bool = False) -> list[str]:
    """组装调试面板的中文文本行（时间/生理/世界/生态/文明等）。"""
    character = world.agent
    physiology = character.physiology
    scene = world.pixel_scene
    hour = world.hour
    day = world.day
    hh = int(hour)
    mm = int((hour - hh) * 60)
    phase = day.phase_name_zh(hour)
    elevation = day.sun_elevation_deg(hour)
    sun_state = "白昼" if elevation > 0 else "夜晚"
    lines = [
        f"{WORLD_NAME} v{WORLD_VERSION}   第{day.day_index(world.t) + 1}天  {hh:02d}:{mm:02d}  {phase}（{sun_state}）",
        f"太阳高度 {elevation:+5.1f}°   白昼强度 {world.daylight:4.2f}   环境温度 {world.air.temperature_c:4.1f}°C   fps {fps:4.1f}"
        + ("   [已暂停]" if paused else ""),
        f"角色：{character.behavior_state:6s}  位置 x={character.x:6.0f}  朝向 {'右' if character.facing > 0 else '左'}"
        f"  跳跃 {character.jump_count} 次  说话 {character.speech_count} 次",
        f"生理：{physiology.summary_zh()}",
        f"体温 {character.body_temperature_c:4.1f}°C   太阳辐照 {world.sun.horizontal_irradiance:5.0f}W/m²   声压 {mind.perception.sound_spl_db if mind and mind.perception else 0.0:5.1f}dB",
        f"世界：像素原子 {scene.aw}×{scene.ah}（1原子={ATOM_SIZE}px）  物体 {len(scene.objects)} 个  浆果 {len(scene.berries)} 颗",
        f"生态：{scene.ecology.summary_zh()}",
        f"材料：{scene.material_composition_text()}",
        "按键：空格 暂停  ·  F1 开关调试  ·  Q/Esc 关闭",
    ]
    # -- 文明层（只有世界携带时才显示） --------------------------------
    registry = getattr(world, "registry", None)
    if registry is not None:
        limit = getattr(getattr(world, "civ_config", None), "max_agents_created", 8)
        lines.insert(
            3,
            f"人口：存活 {registry.alive_count()} / 累计创建 {registry.total_created}/"
            f"{limit}  死亡 {registry.total_deaths}",
        )
        knowledge = getattr(world, "knowledge", None)
        civilization = getattr(world, "civilization", None)
        parts = []
        if knowledge is not None:
            parts.append(knowledge.summary_zh())
        if civilization is not None:
            parts.append(civilization.summary_zh())
        if parts:
            lines.insert(4, "文明：" + "  ·  ".join(parts))
    return lines


def build_debug_panel(world: World, lines: list[str]) -> Image.Image:
    """把调试文本行画成半透明深色圆角面板。"""
    config = world.config
    font = _cjk_font(max(18, round(config.height * 0.016)))
    line_height = _text_size("Ag", font)[1] + 8
    padding = 18
    text_width = max(_text_size(line, font)[0] for line in lines)
    width = text_width + padding * 2 + 8
    height = line_height * len(lines) + padding * 2
    panel = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=12, fill=(12, 15, 20, 220))
    draw.rectangle((0, 0, 6, height - 1), fill=(226, 126, 81, 255))
    y = padding
    for index, line in enumerate(lines):
        # 第 0 行标题白亮、第 1 行信息淡蓝、其余浅灰
        colour = (244, 246, 250, 255) if index == 0 else (200, 208, 218, 255)
        if index == 1:
            colour = (150, 190, 220, 255)
        draw.text((padding + 8, y), line, font=font, fill=colour)
        y += line_height
    return panel


# ---------------------------------------------------------------------------
# 提示条
# ---------------------------------------------------------------------------
def build_hint(config: WorldConfig) -> Image.Image:
    """底部操作提示条（中文）。"""
    text = "像素世界 · 他在自由生活（走 / 跳 / 说话 / 吃 / 睡）    空格 暂停    F1 调试    F2 从零训练    Q / Esc 关闭"
    font = _cjk_font(max(18, round(config.height * 0.018)))
    padding_x, padding_y = 22, 12
    text_width, text_height = _text_size(text, font)
    width = text_width + padding_x * 2
    height = text_height + padding_y * 2
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=14, fill=(12, 15, 20, 200))
    draw.text(((width - text_width) / 2, (height - text_height) / 2 - 1), text, font=font, fill=(236, 240, 245, 255))
    return image


# ---------------------------------------------------------------------------
# IO 辅助
# ---------------------------------------------------------------------------
def save_png(image: Image.Image, path: str | Path) -> Path:
    """把图像保存为 PNG（自动建目录）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
