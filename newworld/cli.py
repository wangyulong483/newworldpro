"""Command line interface for newworld.

Examples
--------
Launch the interactive world (fullscreen, screen-sized)::

    python -m newworld

Windowed development mode::

    python -m newworld --windowed --size 1280x800

Headless demonstration run with a JSONL log and recorded frames::

    python -m newworld --headless --steps 900 --policy wander --log run.jsonl --frames frames/

Generate the documentation images::

    python -m newworld --preview
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .config import (
    WORLD_NAME,
    WorldConfig,
    detect_screen_size,
    make_config,
    parse_size,
)
from .policies import make_policy
from .world import World

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WINDOWED_SIZE = "1280x800"


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newworld",
        description=(
            "newworld — 一块屏幕大小的纯白世界：没有天空/云朵/太阳的图形，"
            "只住着一个很小的像素小人，他用左右方向键在屏幕里活动。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "交互按键：←/→ 或 A/D 移动，空格暂停，R 随机重置，F1 调试，Q / Esc 退出。"
        ),
    )
    parser.add_argument("--version", action="version", version=f"{WORLD_NAME} {__version__}")

    mode = parser.add_argument_group("运行模式")
    mode.add_argument("--info", action="store_true", help="打印屏幕识别与世界参数后退出")
    mode.add_argument("--preview", nargs="?", const=str(PROJECT_ROOT / "assets"), metavar="DIR",
                      help="生成预览图（默认输出到 assets/）")
    mode.add_argument("--headless", action="store_true", help="无窗口运行仿真（用于记录/测试）")

    geometry = parser.add_argument_group("世界参数")
    geometry.add_argument("--size", metavar="WxH", help="覆盖世界尺寸，例如 2560x1600（会切换到窗口模式）")
    geometry.add_argument("--cell", type=int, help="角色像素块大小（屏幕像素），越小活动范围越大")
    geometry.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    geometry.add_argument("--windowed", action="store_true", help="以窗口方式运行（默认全屏）")
    geometry.add_argument("--debug", action="store_true", help="启动时显示调试信息层")
    geometry.add_argument("--neural", nargs="?", const="assets/neural_policy.npz", metavar="PATH",
                          help="用神经网络行为策略（可指定权重文件，默认 assets/neural_policy.npz）")
    geometry.add_argument("--traceback", action="store_true", help="出错时打印完整堆栈")
    geometry.add_argument("--screenshot", metavar="PATH", help="启动约 1 秒后截图并退出（用于验证渲染）")
    geometry.add_argument("--autoclose-ms", type=int, metavar="MS", help="自动关闭窗口（开发用）")
    geometry.add_argument(
        "--ui-test",
        metavar="PATH",
        dest="ui_test",
        help="无人工鼠标自检：程序化点击场景/面板并断言交互，结果写 JSON 后退出",
    )

    headless = parser.add_argument_group("无窗口运行（--headless）")
    headless.add_argument("--steps", type=int, default=600, help="仿真步数（默认 600）")
    headless.add_argument("--policy", default="free",
                          help="free=他自主活动（默认）；wander/random/left/right/stand 仅供测试")
    headless.add_argument("--log", metavar="PATH", help="把逐步状态（含感知）写入 JSONL 日志")
    headless.add_argument("--frames", metavar="DIR", help="把渲染帧保存为 PNG")
    headless.add_argument("--frame-every", type=int, default=60, metavar="N",
                          help="每 N 步保存一帧（默认 60）")
    return parser


def _build_config(
    args: argparse.Namespace,
    width: int | None = None,
    height: int | None = None,
) -> WorldConfig:
    if width is not None and height is not None:
        return make_config(width, height, cell=args.cell)
    if args.size:
        width, height = parse_size(args.size)
        return make_config(width, height, cell=args.cell)
    return make_config(cell=args.cell)


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def run_info() -> int:
    detected = detect_screen_size()
    print(f"{WORLD_NAME} v{__version__}")
    print(f"检测到的屏幕（物理像素）: {detected[0]} x {detected[1]}")
    print()
    config = make_config()
    print(config.describe())
    print()
    from .eye import EyeParameters
    from .perception import GazeParameters
    from .retina import RetinaParameters

    eye = EyeParameters()
    retina = RetinaParameters()
    gaze = GazeParameters()
    print(
        f"视觉参数: 单眼 FOV {eye.field_of_view_deg:.0f}deg，"
        f"瞳孔 {eye.pupil_min_mm:.0f}-{eye.pupil_max_mm:.0f} mm，"
        f"焦距 {eye.focal_length_mm:.0f} mm，视网膜 {retina.n_cones} cones + "
        f"{retina.n_rods} rods（按人眼比例缩放）"
    )
    print(
        f"眼动参数: 扫视 ~{gaze.saccades_per_second:.1f}/s，注视 "
        f"{gaze.fixation_s[0]*1000:.0f}-{gaze.fixation_s[1]*1000:.0f} ms，"
        f"眨眼 ~{gaze.blink_per_minute:.0f}/min"
    )
    from . import physics
    from .environment import Air, Sun

    sun = Sun()
    air = Air()
    print(
        f"物理参数: 重力 {physics.STANDARD_GRAVITY:.5f} m/s2（1 px = 1 cm），"
        f"跳跃初速 {physics.JUMP_SPEED:.1f} m/s（峰高 {physics.jump_height_m():.2f} m）"
    )
    print(
        f"太阳: 高度角 {sun.elevation_deg:.0f}deg，地面辐照 "
        f"{sun.horizontal_irradiance:.0f} W/m2（可见光 {sun.visible_irradiance:.0f} W/m2，"
        f"色温 {sun.temperature_k:.0f} K，峰值 {sun.peak_wavelength_nm:.0f} nm）"
    )
    print(
        f"声学: 声速 {air.speed_of_sound_m_s:.1f} m/s，听觉 "
        f"{physics.HEARING_MIN_HZ:.0f}-{physics.HEARING_MAX_HZ:.0f} Hz，"
        f"参考声压 {physics.REFERENCE_SOUND_PRESSURE_PA*1e6:.0f} uPa"
    )
    print(
        f"热学: 气温 {air.temperature_c:.0f} C，体温设定 {physics.HUMAN_BODY_TEMPERATURE_K-273.15:.0f} C，"
        f"Stefan-Boltzmann {physics.STEFAN_BOLTZMANN:.3e} W/m2K4"
    )
    from .daynight import DayCycle
    from .materials import ATOM_SIZE, MATERIALS

    day = DayCycle()
    print(
        f"昼夜: 一天 {day.day_length_s:.0f} s，日出 06:00，正午太阳高度 {day.max_elevation_deg:.0f}deg，"
        f"日落 18:00；启动时刻 {day.start_hour:.0f}:00"
    )
    print(f"像素原子: 1 原子 = {ATOM_SIZE}x{ATOM_SIZE} 屏幕像素；材料 {len(MATERIALS)} 种（水/土壤/石头/草/木/叶/浆果/肉体/骨骼/空气）")
    print()
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"蓝图 PDF  : {PROJECT_ROOT / 'ref' / 'blueprint.pdf'}")
    try:
        import numpy
        import PIL

        print(f"依赖: pillow {PIL.__version__}, numpy {numpy.__version__}")
    except Exception:  # pragma: no cover
        print("依赖: 未检测到 pillow/numpy")
    try:
        import tkinter

        print(f"tkinter: 可用（Tk {tkinter.TkVersion}）")
    except Exception:  # pragma: no cover
        print("tkinter: 不可用（无法打开交互窗口）")
    return 0


def run_preview(args: argparse.Namespace) -> int:
    from .preview import generate_previews

    config = _build_config(args)
    outdir = Path(args.preview) if args.preview else PROJECT_ROOT / "assets"
    print(f"正在渲染预览图 -> {outdir} ...")
    paths = generate_previews(config, outdir)
    for name, path in paths.items():
        print(f"  {name:24s} {path}")
    return 0


def _load_neural_policy(args: argparse.Namespace):
    """Load the optional neural behaviour policy (random weights if missing)."""
    from .neural import NeuralPolicy

    spec = getattr(args, "neural", None)
    if not spec:
        return None
    path = Path(spec)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists():
        try:
            policy = NeuralPolicy.load(path)
            print(f"神经网络行为策略：{path}")
            return policy
        except ValueError as exc:
            print(f"策略文件不兼容（{exc}），先使用随机网络；按 F2 训练并保存新策略")
            return NeuralPolicy(seed=getattr(args, "seed", 0))
    print(f"未找到策略文件 {path}，使用随机初始化的神经网络")
    return NeuralPolicy(seed=getattr(args, "seed", 0))


def run_headless(args: argparse.Namespace) -> int:
    from .autonomy import Mind
    from .render import render_frame

    if args.steps <= 0:
        print("--steps 必须大于 0")
        return 2
    config = _build_config(args)
    world = World(config, seed=args.seed)
    world.reset(seed=args.seed)

    free = args.policy == "free"
    neural_policy = _load_neural_policy(args)
    mind = Mind(world, seed=args.seed, policy=neural_policy) if free else None
    policy = None if free else make_policy(args.policy, seed=args.seed)
    fixed_dt = config.fixed_dt

    log_handle = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")

    frames_dir = Path(args.frames) if args.frames else None
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)

    x_min = x_max = world.agent.x
    left_hits = right_hits = 0
    saved_frames = 0
    started = time.perf_counter()
    try:
        for step in range(args.steps):
            if free:
                action = mind.step(fixed_dt)
            else:
                action = int(policy(world))
                world.step(actions=action)
            agent = world.agent
            x_min = min(x_min, agent.x)
            x_max = max(x_max, agent.x)
            left_hits += int(agent.at_left_wall)
            right_hits += int(agent.at_right_wall)

            if log_handle is not None:
                record = {
                    "step": step,
                    "t": round(world.t, 6),
                    "action": action,
                    "x": round(agent.x, 3),
                    "vx": round(agent.vx, 3),
                    "facing": agent.facing,
                    "at_left_wall": agent.at_left_wall,
                    "at_right_wall": agent.at_right_wall,
                    "obs": [round(value, 6) for value in agent.observation(config)],
                    "policy": "neural" if neural_policy is not None else "homeostatic",
                }
                perception = mind.perception if mind is not None else None
                record.update(
                    {
                        "y": round(agent.y if agent.y is not None else 0.0, 3),
                        "vy": round(agent.vy, 3),
                        "on_ground": agent.on_ground,
                        "body_temperature_c": round(agent.body_temperature_c, 3),
                        "air_temperature_c": round(world.air.temperature_c, 3),
                        "solar_irradiance_w_m2": round(world.sun.horizontal_irradiance, 2),
                    }
                )
                if perception is not None:
                    record.update(
                        {
                            "gaze_deg": round(perception.gaze_world_deg, 3),
                            "eye_mode": perception.mode,
                            "blink": perception.blink,
                            "pupil_mm": round(perception.pupil_mm, 3),
                            "accommodation_dpt": round(perception.accommodation_dpt, 4),
                            "blur_sigma_deg": round(perception.blur_sigma_deg, 4),
                            "mean_luminance": round(perception.mean_luminance, 5),
                            "foveal_luminance": round(perception.foveal_luminance, 5),
                            "retina_feature_dim": int(perception.feature_vector().size),
                            "sound_spl_db": round(perception.sound_spl_db, 3),
                            "sound_pressure_pa": round(perception.sound_pressure_pa, 6),
                            "hearing": perception.hearing,
                            "utterance": perception.utterance,
                        }
                    )
                log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if frames_dir is not None and step % max(1, args.frame_every) == 0:
                speech = (
                    agent.utterance
                    if agent.utterance and agent.utterance_until >= world.t
                    else None
                )
                frame = render_frame(world, speech=speech)
                frame.save(frames_dir / f"frame_{step:05d}.png")
                saved_frames += 1
    finally:
        if log_handle is not None:
            log_handle.close()

    elapsed = time.perf_counter() - started
    agent = world.agent
    print(f"无窗口仿真完成：{args.steps} 步 / {world.t:.1f}s 世界时间")
    if free:
        print(f"  模式：他自主活动（free），种子 {args.seed}，世界 {config.width}x{config.height}")
    else:
        print(f"  模式：测试策略 {args.policy}，种子 {args.seed}，世界 {config.width}x{config.height}")
    print(f"  角色 x 范围 [{x_min:.1f}, {x_max:.1f}] px（合法范围 "
          f"[{config.min_x:.1f}, {config.max_x:.1f}]）")
    print(f"  累计移动 {agent.distance_walked:.0f} px，触屏边次数 左 {left_hits} / 右 {right_hits}")
    print(
        f"  物理：跳跃 {agent.jump_count} 次，说话 {agent.speech_count} 次；"
        f"体温 {agent.body_temperature_c:.1f} C / 气温 {world.air.temperature_c:.1f} C；"
        f"太阳辐照 {world.sun.horizontal_irradiance:.0f} W/m2"
    )
    if mind is not None and mind.perceiver.visual is not None:
        gaze = mind.perceiver.visual.gaze.state
        print(
            f"  感知：视网膜 {mind.perceiver.visual.retina_parameters.n_cones} cones + "
            f"{mind.perceiver.visual.retina_parameters.n_rods} rods；扫视 {gaze.saccade_count} 次，"
            f"眨眼 {gaze.blink_count} 次，最后视线 {mind.perceiver.visual.last_perception.gaze_world_deg:+.1f} deg"
        )
        if mind.perception is not None:
            print(
                f"  视觉输入：平均亮度 {mind.perception.mean_luminance:.3f}，"
                f"中央凹亮度 {mind.perception.foveal_luminance:.3f}（纯白世界 → 均匀）"
            )
            print(
                f"  听觉：{mind.perception.sound_spl_db:.1f} dB SPL，"
                f"听到={'是' if mind.perception.hearing else '否'}"
            )
    print(f"  计算耗时 {elapsed:.2f}s（{args.steps / max(elapsed, 1e-9):.0f} steps/s）")
    if args.log:
        print(f"  JSONL 日志: {args.log}")
    if frames_dir is not None:
        print(f"  渲染帧 {saved_frames} 张 -> {frames_dir}")
    return 0


def run_interactive(args: argparse.Namespace) -> int:
    from .app_tk import run_app

    if args.size or args.windowed:
        if args.size:
            width, height = parse_size(args.size)
        else:
            width, height = parse_size(DEFAULT_WINDOWED_SIZE)
        config = _build_config(args, width, height)
        fullscreen = False
    else:
        config = _build_config(args)
        fullscreen = True
    neural_policy = _load_neural_policy(args)
    policy_path = None
    if getattr(args, "neural", None):
        spec = Path(args.neural)
        policy_path = spec if spec.is_absolute() else PROJECT_ROOT / spec
    return run_app(
        config,
        fullscreen=fullscreen,
        debug=args.debug,
        screenshot=args.screenshot,
        autoclose_ms=args.autoclose_ms,
        seed=args.seed,
        policy=neural_policy,
        policy_path=policy_path,
        ui_test_path=getattr(args, "ui_test", None),
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.info:
            return run_info()
        if args.preview is not None:
            return run_preview(args)
        if args.headless:
            return run_headless(args)
        return run_interactive(args)
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130
    except Exception as exc:  # keep the CLI friendly
        if args.traceback:
            raise
        print(f"错误：{exc}")
        print("（使用 --traceback 查看完整堆栈）")
        return 1
