"""Run the inhabitant for a while and inspect his perception.

    python examples/vision_rollout.py

He acts on his own (no keyboard, no reward) and looks around with human-like
saccades.  The script prints the perceptual state and saves one debug frame
showing his position, his eye and his gaze direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newworld import make_config  # noqa: E402
from newworld.autonomy import Mind  # noqa: E402
from newworld.render import character_layer, debug_lines_zh, render_frame  # noqa: E402
from newworld.world import World  # noqa: E402


def main() -> int:
    config = make_config()
    world = World(config, seed=0)
    world.reset(seed=0)
    mind = Mind(world, seed=7)

    steps = 900
    for _ in range(steps):
        mind.step(config.fixed_dt)

    agent = world.agent
    perception = mind.perception
    print(f"world      : {config.width}x{config.height} pure white")
    print(
        f"position   : x = {agent.x:.1f} px, y = {agent.y:.1f} px, "
        f"vx = {agent.vx:+.1f}, vy = {agent.vy:+.1f} px/s, on_ground = {agent.on_ground}"
    )
    print(
        f"physics    : jumps = {agent.jump_count}, speech = {agent.speech_count}, "
        f"body = {agent.body_temperature_c:.1f} C, air = {world.air.temperature_c:.1f} C, "
        f"sun = {world.sun.horizontal_irradiance:.0f} W/m2"
    )
    if perception is not None:
        print(f"eye        : ({perception.eye_x:.1f}, {perception.eye_y:.1f}) px")
        print(f"gaze       : {perception.gaze_world_deg:+.1f} deg, mode = {perception.mode}")
        print(f"optics     : pupil {perception.pupil_mm:.1f} mm, "
              f"focus {perception.accommodation_dpt:.2f} D, blur {perception.blur_sigma_deg:.2f} deg")
        print(f"retina     : mean luminance {perception.mean_luminance:.3f}, "
              f"fovea {perception.foveal_luminance:.3f}, "
              f"feature dim {perception.feature_vector().size}")
        print(f"audio      : {perception.sound_spl_db:.1f} dB SPL, heard = {perception.hearing}")
        print(f"eye moves  : saccades {mind.perceiver.visual.gaze.state.saccade_count}, "
              f"blinks {mind.perceiver.visual.gaze.state.blink_count}")
        print("note       : the pixel world has trees, rocks, bushes and a pond; he sees them")

    lines = debug_lines_zh(world, mind, fps=0.0)
    frame = render_frame(
        world,
        debug=True,
        debug_lines=tuple(lines),
        speech=world.agent.utterance if world.agent.utterance_until >= world.t else None,
    )
    frame = frame.resize((1280, round(1280 * frame.height / frame.width)))
    out = ROOT / "examples" / "vision_frame.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.save(out)
    print(f"debug frame saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
