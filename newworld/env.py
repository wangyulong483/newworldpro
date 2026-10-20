"""newworld 的 Gymnasium 风格强化学习接口。

这个世界本来就是为社会行为与演化研究的 RL / 神经网络研究搭建的
基底。本模块把纯仿真包成 Gymnasium 使用的 ``reset``/``step``
约定（``gymnasium`` *不是*硬依赖）：

.. code-block:: python

    from newworld.env import NewWorldEnv

    env = NewWorldEnv(seed=0)
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        action = env.action_space_sample()   # 或你自己的策略
        obs, reward, terminated, truncated, info = env.step(action)

    # 包装成 Gymnasium 只要一行：
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

观测（8 个浮点，见 :meth:`Character.observation`；力学现在包含
重力与跳跃，因此垂直状态也是观测的一部分）：

=========== =========================================================
``[0]``     归一化 x 位置（0 = 左墙，1 = 右墙）
``[1]``     归一化带符号水平速度（-1 .. 1）
``[2]``     朝向（-1 或 +1）
``[3]``     到左墙的归一化距离
``[4]``     到右墙的归一化距离
``[5]``     跳离地面的高度（0 .. 1，按满跳归一化）
``[6]``     归一化垂直速度（-1 .. 1）
``[7]``     站在地面为 1.0，空中为 0.0
=========== =========================================================

动作（离散，3 个值）：

=========== =========================================================
``0``       向左走
``1``       原地站
``2``       向右走
=========== =========================================================
"""

from __future__ import annotations

import random
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from PIL import Image

from .config import WorldConfig
from .world import World

#: 暴露给 RL 策略的离散动作数
N_ACTIONS = 3
#: 观测向量维数
OBSERVATION_DIM = 8


# ---------------------------------------------------------------------------
# 奖励函数
# ---------------------------------------------------------------------------
class RewardFunction(Protocol):
    """向世界插入标量奖励的钩子。

    实现可以持有每个回合的状态；每回合开始时环境会调用
    :meth:`reset`。
    """

    def reset(self, world: World) -> None:
        """回合开始钩子。"""
        ...  # noqa: D401

    def __call__(self, world: World, observation: Sequence[float]) -> float:
        """计算本步奖励。"""
        ...


class ZeroReward:
    """完全没有奖励：纯仿真（演示用）。"""

    def reset(self, world: World) -> None:
        """重置回合状态。"""
        return None

    def __call__(self, world: World, observation: Sequence[float]) -> float:
        """取一步动作。"""
        return 0.0


class AliveReward:
    """每步小额常数奖励，经典"活着"基线。"""

    def __init__(self, scale: float = 0.01) -> None:
        """初始化。"""
        self.scale = float(scale)

    def reset(self, world: World) -> None:
        """重置回合状态。"""
        return None

    def __call__(self, world: World, observation: Sequence[float]) -> float:
        """取一步动作。"""
        return self.scale


class ExplorationReward:
    """到达从未见过的水平分箱时给奖励（内在奖励）。"""

    def __init__(self, bins: int = 12, new_bin_reward: float = 1.0, step_cost: float = 0.0) -> None:
        """初始化。"""
        if bins < 1:
            raise ValueError("bins must be >= 1")
        self.bins = bins
        self.new_bin_reward = float(new_bin_reward)
        self.step_cost = float(step_cost)
        self.visited: set[int] = set()

    def reset(self, world: World) -> None:
        """清空已访问分箱。"""
        self.visited.clear()

    def __call__(self, world: World, observation: Sequence[float]) -> float:
        """取一步动作。"""
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
    """接近 / 停留在目标 x 位置的奖励。"""

    def __init__(self, target_norm: float = 0.5, scale: float = 1.0, tolerance: float = 0.02) -> None:
        """初始化。"""
        self.target_norm = float(target_norm)
        self.scale = float(scale)
        self.tolerance = float(tolerance)

    def reset(self, world: World) -> None:
        """重置回合状态。"""
        return None

    def __call__(self, world: World, observation: Sequence[float]) -> float:
        """取一步动作。"""
        span = world.config.travel_span
        if span <= 0:
            return 0.0
        target_x = world.config.min_x + span * self.target_norm
        error_norm = abs(world.agent.x - target_x) / span
        return -self.scale * error_norm


# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------
class NewWorldEnv:
    """包着 :class:`~newworld.world.World` 的 Gymnasium 风格环境。"""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: WorldConfig | None = None,
        *,
        reward: RewardFunction | Callable[[World, Sequence[float]], float] | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        """初始化。"""
        self.world = World(config, seed=seed)
        self.reward_fn: Any = reward if reward is not None else ZeroReward()
        self.max_steps = max_steps
        self._episode_steps = 0
        self._last_observation: tuple[float, ...] = self.world.agent.observation(self.world.config)
        self._last_action = 1
        self._closed = False

    # -- 空间（普通属性，不依赖 gymnasium） ----------------------------
    @property
    def n_actions(self) -> int:
        """离散动作数。"""
        return N_ACTIONS

    @property
    def observation_dim(self) -> int:
        """观测维数。"""
        return OBSERVATION_DIM

    def action_space_sample(self) -> int:
        """均匀随机采样一个动作。"""
        return random.randrange(N_ACTIONS)

    # -- 核心 API -------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """重置世界（可用 options["positions"] 指定位置），返回 (obs, info)。"""
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
        """执行一个动作，返回 (obs, reward, terminated, truncated, info)。"""
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

    # -- 辅助 ------------------------------------------------------------
    def _obs(self, observation: Sequence[float]) -> np.ndarray:
        """把观测元组转成 float32 数组。"""
        return np.asarray(observation, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        """附带的调试信息字典。"""
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
        """当前帧的 ``(height, width, 3)`` uint8 数组。"""
        from .render import render_frame

        frame = render_frame(self.world)
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)

    def render_image(self, *, debug: bool = False, fps: float = 0.0) -> Image.Image:
        """当前帧的 PIL 图像（可带调试面板）。"""
        from .render import render_frame

        return render_frame(self.world, debug=debug)

    def close(self) -> None:
        """关闭环境（之后 step 会报错）。"""
        self._closed = True
