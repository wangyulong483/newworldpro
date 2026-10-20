"""最小 RL 风格 rollout 示例。

在项目根目录运行::

    python examples/hello_env.py

演示 Gymnasium 风格 API、可插拔奖励函数与渲染帧保存。这是在
newworld 上开展策略 / 神经网络实验的起点。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newworld.env import ExplorationReward, NewWorldEnv  # noqa: E402


def main() -> int:
    env = NewWorldEnv(reward=ExplorationReward(bins=16), seed=0, max_steps=600)
    observation, info = env.reset(seed=0)
    print(f"observation space : {env.observation_dim} floats")
    print(f"action space      : {env.n_actions} actions (0=left, 1=stand, 2=right)")
    print(f"initial obs       : {observation}")

    total_reward = 0.0
    steps = 0
    while True:
        action = env.action_space_sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if steps % 100 == 0:
            print(
                f"step {steps:4d} | x={info['x']:7.1f}px | vx={info['vx']:+7.1f}px/s "
                f"| reward={reward:+.2f} | total={total_reward:+.2f}"
            )
        if terminated or truncated:
            break

    print(f"episode finished: {steps} steps, total reward {total_reward:+.2f}")
    frame = env.render_image(debug=True)
    frame = frame.resize((1280, round(1280 * frame.height / frame.width)))
    out = ROOT / "examples" / "hello_env_frame.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.save(out)
    print(f"frame saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
