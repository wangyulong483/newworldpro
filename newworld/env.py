"""A Gymnasium-style reinforcement-learning interface for newworld.

The world was built as a substrate for RL / neural-network research on social
behaviour and evolution.  This module wraps the pure simulation in the
``reset``/``step`` convention used by Gymnasium (``gymnasium`` is *not* a hard
dependency):

.. code-block:: python

    from newworld.env import NewWorldEnv

    env = NewWorldEnv(seed=0)
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        action = env.action_space_sample()   # or your policy
        obs, reward, terminated, truncated, info = env.step(action)

    # Wrapping for Gymnasium is one line:
    # class GymWorld(gym.Env):
    #     def __init__(self):
    #         self.env = NewWorldEnv()
    #         self.action_space = gym.spaces.Discrete(3)
    #         self.observation_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)
    #     def reset(self, *, seed=None, options=None):
    #         obs, info = self.env.reset(seed=seed)
    #         return np.asarray(obs, dtype=np.float32), info
    #     def step(self, action):
    #         obs, r, term, trunc, info = self.env.step(int(action))
    #         return np.asarray(obs, dtype=np.float32), r, term, trunc, info

Observation (8 floats, see :meth:`Character.observation`; mechanics now include
gravity and jumping, so the vertical state is part of the observation):

=========== =========================================================
``[0]``     normalised x position (0 = left wall, 1 = right wall)
``[1]``     normalised signed horizontal velocity (-1 .. 1)
``[2]``     facing (-1 or +1)
``[3]``     normalised distance to the left wall
``[4]``     normalised distance to the right wall
``[5]``     jump height above the ground (0 .. 1 of a full jump)
``[6]``     normalised vertical velocity (-1 .. 1)
``[7]``     1.0 when standing on the ground, 0.0 in the air
=========== =========================================================

Action (discrete, 3 values):

=========== =========================================================
``0``       walk left
``1``       stand still
``2``       walk right
=========== =========================================================
"""

from __future__ import annotations

import random
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from PIL import Image

from .config import WorldConfig
from .world import World

#: number of discrete actions exposed to an RL policy
N_ACTIONS = 3
#: dimension of the observation vector
OBSERVATION_DIM = 8


# ---------------------------------------------------------------------------
# reward functions
# ---------------------------------------------------------------------------
class RewardFunction(Protocol):
    """Hook for plugging a scalar reward into the world.

    Implementations may keep per-episode state; :meth:`reset` is called by the
    environment at the start of every episode.
    """

    def reset(self, world: World) -> None: ...  # noqa: D401

    def __call__(self, world: World, observation: Sequence[float]) -> float: ...


class ZeroReward:
    """No reward at all: pure simulation (useful for demonstrations)."""

    def reset(self, world: World) -> None:
        return None

    def __call__(self, world: World, observation: Sequence[float]) -> float:
        return 0.0


class AliveReward:
    """Small constant reward per step, the classic "stay alive" baseline."""

    def __init__(self, scale: float = 0.01) -> None:
        self.scale = float(scale)

    def reset(self, world: World) -> None:
        return None

    def __call__(self, world: World, observation: Sequence[float]) -> float:
        return self.scale


class ExplorationReward:
    """Reward for visiting never-seen horizontal bins (an intrinsic reward)."""

    def __init__(self, bins: int = 12, new_bin_reward: float = 1.0, step_cost: float = 0.0) -> None:
        if bins < 1:
            raise ValueError("bins must be >= 1")
        self.bins = bins
        self.new_bin_reward = float(new_bin_reward)
        self.step_cost = float(step_cost)
        self.visited: set[int] = set()

    def reset(self, world: World) -> None:
        self.visited.clear()

    def __call__(self, world: World, observation: Sequence[float]) -> float:
        span = world.config.travel_span
        if span <= 0:
            return -self.step_cost
        agent = world.agent
        bin_index = min(self.bins - 1, int((agent.x - world.config.min_x) / span * self.bins))
        reward = -self.step_cost
        if bin_index not in self.visited:
            self.visited.add(bin_index)
            reward += self.new_bin_reward
        return reward


class CenterReward:
    """Reward for approaching / staying at a target x position."""

    def __init__(self, target_norm: float = 0.5, scale: float = 1.0, tolerance: float = 0.02) -> None:
        self.target_norm = float(target_norm)
        self.scale = float(scale)
        self.tolerance = float(tolerance)

    def reset(self, world: World) -> None:
        return None

    def __call__(self, world: World, observation: Sequence[float]) -> float:
        span = world.config.travel_span
        if span <= 0:
            return 0.0
        target_x = world.config.min_x + span * self.target_norm
        error_norm = abs(world.agent.x - target_x) / span
        return -self.scale * error_norm


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
class NewWorldEnv:
    """Gymnasium-style environment around :class:`~newworld.world.World`."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: WorldConfig | None = None,
        *,
        reward: RewardFunction | Callable[[World, Sequence[float]], float] | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.world = World(config, seed=seed)
        self.reward_fn: Any = reward if reward is not None else ZeroReward()
        self.max_steps = max_steps
        self._episode_steps = 0
        self._last_observation: tuple[float, ...] = self.world.agent.observation(self.world.config)
        self._last_action = 1
        self._closed = False

    # -- spaces (plain attributes, no gymnasium dependency) -------------
    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def observation_dim(self) -> int:
        return OBSERVATION_DIM

    def action_space_sample(self) -> int:
        return random.randrange(N_ACTIONS)

    # -- core API -------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        positions = None
        if options:
            raw_positions = options.get("positions")
            if raw_positions is not None:
                positions = [float(value) for value in raw_positions]
        observations = self.world.reset(seed=seed, positions=positions)
        reset_hook = getattr(self.reward_fn, "reset", None)
        if callable(reset_hook):
            reset_hook(self.world)
        self._episode_steps = 0
        self._last_observation = observations[0]
        self._last_action = 1
        return self._obs(observations[0]), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("environment is closed")
        self._last_action = int(action)
        observations = self.world.step(actions=self._last_action)
        observation = observations[0]
        self._last_observation = observation
        reward = float(self.reward_fn(self.world, observation))
        self._episode_steps += 1
        terminated = False
        truncated = self.max_steps is not None and self._episode_steps >= self.max_steps
        return self._obs(observation), reward, terminated, truncated, self._info()

    # -- helpers --------------------------------------------------------
    def _obs(self, observation: Sequence[float]) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        agent = self.world.agent
        return {
            "t": self.world.t,
            "step": self._episode_steps,
            "x": agent.x,
            "vx": agent.vx,
            "facing": agent.facing,
            "at_left_wall": agent.at_left_wall,
            "at_right_wall": agent.at_right_wall,
            "action": self._last_action,
        }

    def render(self) -> np.ndarray:
        """Return the current frame as an ``(height, width, 3)`` uint8 array."""
        from .render import render_frame

        frame = render_frame(self.world)
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)

    def render_image(self, *, debug: bool = False, fps: float = 0.0) -> Image.Image:
        from .render import render_frame

        return render_frame(self.world, debug=debug)

    def close(self) -> None:
        self._closed = True
