"""带学习型原语言的神经网络行为策略。

居民可以由一个更大的 MLP 驱动：

* 输入：40 维感官/生理特征（v0.9 扩展；前 24 维与旧版一致）；
* 隐藏层：64 tanh 单元 → 32 tanh 单元；
* 输出：5 个行为模式、3 个移动方向、跳跃/说话开关，以及一个从
  16 音节词表中取的 **3 音节词**。

默认 checkpoint 是 ``assets/neural_policy.npz``。启动时载入且
**从不自动重训**。只有观察者按 **F2** 才训练；训练从已载入的权重
继续并覆盖 checkpoint，因此策略跨会话持续演化。

网络行为靠模仿手写的稳态行为（走/觅食/吃/喝/睡）来引导；语言
头为每个内在状态学一个专属的词，让生物拥有一小套涌现的
"原语言"，而不是固定的随机音节。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .behavior import Decision, HAND_HEIGHT_PX, HAND_REACH_PX, InnerLife
from .materials import ATOM_SIZE

# ---------------------------------------------------------------------------
# 网络结构
# ---------------------------------------------------------------------------
#: 文明层加入之前的感官/生理特征维数
LEGACY_FEATURE_DIM = 24
#: 文明层追加的上下文（P3）：8 记忆摘要 + 4 目标/计划 + 4 文明上下文
CONTEXT_DIM = 16
#: v2（文明版）输入宽度：24 感知/生理 + 16 文明上下文
V2_FEATURE_DIM = LEGACY_FEATURE_DIM + CONTEXT_DIM
#: V2.1（感知打通）追加的真实视网膜特征维数
VISUAL_DIM = 16
#: 完整输入宽度。旧 24 维与 v2 40 维 checkpoint 都通过**零初始化**新输入列迁移，
#: 因此重训之前行为逐比特一致（视觉特征全 0 时输出不变）。
FEATURE_DIM = V2_FEATURE_DIM + VISUAL_DIM
HIDDEN1 = 64
HIDDEN2 = 32
MODES: tuple[str, ...] = ("explore", "forage", "drink", "sleep", "rest")
MOVE_DIM = 3
AUX_DIM = 2
LANG_TOKENS = 3
VOCAB: tuple[str, ...] = (
    "ba", "da", "gu", "ki", "mi", "no", "po", "su",
    "ta", "vi", "zo", "lu", "ne", "ra", "fi", "mo",
)
LANG_DIM = LANG_TOKENS * len(VOCAB)
OUTPUT_DIM = len(MODES) + MOVE_DIM + AUX_DIM + LANG_DIM

#: 旧 24 输入网络的扁平参数量（文明层之前的 v0.8.0）
LEGACY_FLAT_SIZE = (
    LEGACY_FEATURE_DIM * HIDDEN1 + HIDDEN1
    + HIDDEN1 * HIDDEN2 + HIDDEN2
    + HIDDEN2 * OUTPUT_DIM + OUTPUT_DIM
)
#: v2（40 输入，文明版）网络的扁平参数量
V2_FLAT_SIZE = (
    V2_FEATURE_DIM * HIDDEN1 + HIDDEN1
    + HIDDEN1 * HIDDEN2 + HIDDEN2
    + HIDDEN2 * OUTPUT_DIM + OUTPUT_DIM
)
#: 当前（56 输入）网络的扁平参数量
CURRENT_FLAT_SIZE = (
    FEATURE_DIM * HIDDEN1 + HIDDEN1
    + HIDDEN1 * HIDDEN2 + HIDDEN2
    + HIDDEN2 * OUTPUT_DIM + OUTPUT_DIM
)

#: 每个内在状态一个专属的词（训练时的语言目标）
MODE_WORDS: dict[str, tuple[str, ...]] = {
    "explore": ("ba", "ne", "po"),
    "forage": ("mi", "ta", "ki"),
    "drink": ("su", "vi", "da"),
    "sleep": ("no", "vi", "lu"),
    "rest": ("ta", "ne", "mo"),
}


@dataclass
class PolicyOutput:
    """一次前向传播解析出的高层输出。"""

    mode: str
    move: int
    jump: bool
    speak: bool
    word: str


class NeuralPolicy:
    """双隐藏层 MLP 策略（纯 numpy，无框架依赖）。"""

    def __init__(self, weights: np.ndarray | None = None, seed: int = 0) -> None:
        """初始化（随机权重或从扁平数组还原）。"""
        rng = np.random.default_rng(seed)
        if weights is None:
            # 三层权重的尺度递减（0.5 → 0.4 → 0.3）利于初始行为温和
            self.w1 = rng.normal(0.0, 0.5, (FEATURE_DIM, HIDDEN1))
            self.b1 = np.zeros(HIDDEN1)
            self.w2 = rng.normal(0.0, 0.4, (HIDDEN1, HIDDEN2))
            self.b2 = np.zeros(HIDDEN2)
            self.w3 = rng.normal(0.0, 0.3, (HIDDEN2, OUTPUT_DIM))
            self.b3 = np.zeros(OUTPUT_DIM)
        else:
            self.set_flat_weights(weights)

    # ------------------------------------------------------------------
    def flat_weights(self) -> np.ndarray:
        """把全部权重拍平成一维数组（遗传/存档用）。"""
        return np.concatenate(
            [
                self.w1.reshape(-1),
                self.b1.reshape(-1),
                self.w2.reshape(-1),
                self.b2.reshape(-1),
                self.w3.reshape(-1),
                self.b3.reshape(-1),
            ]
        )

    def set_flat_weights(self, flat: np.ndarray) -> None:
        """从一维数组按顺序还原各层权重。"""
        flat = np.asarray(flat, dtype=np.float64)
        index = 0
        self.w1 = flat[index : index + FEATURE_DIM * HIDDEN1].reshape(FEATURE_DIM, HIDDEN1)
        index += FEATURE_DIM * HIDDEN1
        self.b1 = flat[index : index + HIDDEN1]
        index += HIDDEN1
        self.w2 = flat[index : index + HIDDEN1 * HIDDEN2].reshape(HIDDEN1, HIDDEN2)
        index += HIDDEN1 * HIDDEN2
        self.b2 = flat[index : index + HIDDEN2]
        index += HIDDEN2
        self.w3 = flat[index : index + HIDDEN2 * OUTPUT_DIM].reshape(HIDDEN2, OUTPUT_DIM)
        index += HIDDEN2 * OUTPUT_DIM
        self.b3 = flat[index : index + OUTPUT_DIM]

    # ------------------------------------------------------------------
    def forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """前向传播：返回 (logits, h1, h2)（隐藏层供训练用）。"""
        h1 = np.tanh(features @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        logits = h2 @ self.w3 + self.b3
        return logits, h1, h2

    def act(self, features: np.ndarray) -> PolicyOutput:
        """贪心解码一帧：模式/移动取 argmax，跳跃说话看正负，语言取每音节 argmax。"""
        logits, _, _ = self.forward(features)
        mode_index = int(np.argmax(logits[: len(MODES)]))
        move_index = int(np.argmax(logits[len(MODES) : len(MODES) + MOVE_DIM]))
        move = (1, 0, -1)[move_index]
        aux_start = len(MODES) + MOVE_DIM
        jump = bool(logits[aux_start] > 0.0)
        speak = bool(logits[aux_start + 1] > 0.0)
        lang_start = aux_start + AUX_DIM
        tokens = []
        for token in range(LANG_TOKENS):
            block = logits[lang_start + token * len(VOCAB) : lang_start + (token + 1) * len(VOCAB)]
            tokens.append(VOCAB[int(np.argmax(block))])
        return PolicyOutput(MODES[mode_index], move, jump, speak, "-".join(tokens))

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """把权重与结构元数据保存为 .npz checkpoint。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=self.flat_weights(),
            feature_dim=FEATURE_DIM,
            hidden1=HIDDEN1,
            hidden2=HIDDEN2,
            output_dim=OUTPUT_DIM,
            vocab=np.asarray(VOCAB),
            version=np.asarray(4),
        )
        return path

    @classmethod
    def from_legacy_weights(cls, flat: np.ndarray) -> "NeuralPolicy":
        """把 24 输入的旧 checkpoint 零初始化迁移到 40 输入。

        新增输入列的权重全为 0，因此额外特征对第一隐藏层的贡献恰为
        ``0``，迁移后的网络输出与原网络**逐比特一致**。继续训练再
        逐渐学习（或忽略）新的上下文。
        """
        flat = np.asarray(flat, dtype=np.float64).reshape(-1)
        if flat.size != LEGACY_FLAT_SIZE:
            raise ValueError(
                f"not a legacy checkpoint: {flat.size} values, expected {LEGACY_FLAT_SIZE}"
            )
        policy = cls(seed=0)
        index = 0
        legacy_w1 = flat[index : index + LEGACY_FEATURE_DIM * HIDDEN1].reshape(LEGACY_FEATURE_DIM, HIDDEN1)
        index += LEGACY_FEATURE_DIM * HIDDEN1
        policy.w1 = np.zeros((FEATURE_DIM, HIDDEN1))
        policy.w1[:LEGACY_FEATURE_DIM] = legacy_w1
        policy.b1 = flat[index : index + HIDDEN1].copy()
        index += HIDDEN1
        policy.w2 = flat[index : index + HIDDEN1 * HIDDEN2].reshape(HIDDEN1, HIDDEN2).copy()
        index += HIDDEN1 * HIDDEN2
        policy.b2 = flat[index : index + HIDDEN2].copy()
        index += HIDDEN2
        policy.w3 = flat[index : index + HIDDEN2 * OUTPUT_DIM].reshape(HIDDEN2, OUTPUT_DIM).copy()
        index += HIDDEN2 * OUTPUT_DIM
        policy.b3 = flat[index : index + OUTPUT_DIM].copy()
        return policy

    @classmethod
    def from_v2_weights(cls, flat: np.ndarray) -> "NeuralPolicy":
        """把 v2（40 输入，文明版）checkpoint 零初始化迁移到 56 输入。

        新增的 16 维视觉特征权重全为 0，因此视觉输入为 0 时迁移后的网络
        输出与原网络**逐比特一致**；从 BC 数据继续训练后才逐渐学习视觉。
        """
        flat = np.asarray(flat, dtype=np.float64).reshape(-1)
        if flat.size != V2_FLAT_SIZE:
            raise ValueError(
                f"not a v2 checkpoint: {flat.size} values, expected {V2_FLAT_SIZE}"
            )
        policy = cls(seed=0)
        index = 0
        v2_w1 = flat[index : index + V2_FEATURE_DIM * HIDDEN1].reshape(V2_FEATURE_DIM, HIDDEN1)
        index += V2_FEATURE_DIM * HIDDEN1
        policy.w1 = np.zeros((FEATURE_DIM, HIDDEN1))
        policy.w1[:V2_FEATURE_DIM] = v2_w1
        policy.b1 = flat[index : index + HIDDEN1].copy()
        index += HIDDEN1
        policy.w2 = flat[index : index + HIDDEN1 * HIDDEN2].reshape(HIDDEN1, HIDDEN2).copy()
        index += HIDDEN1 * HIDDEN2
        policy.b2 = flat[index : index + HIDDEN2].copy()
        index += HIDDEN2
        policy.w3 = flat[index : index + HIDDEN2 * OUTPUT_DIM].reshape(HIDDEN2, OUTPUT_DIM).copy()
        index += HIDDEN2 * OUTPUT_DIM
        policy.b3 = flat[index : index + OUTPUT_DIM].copy()
        return policy

    @classmethod
    def load(cls, path: str | Path) -> "NeuralPolicy":
        """载入 checkpoint；自动迁移 24 维（v0.8）与 40 维（v2）旧格式。"""
        data = np.load(Path(path), allow_pickle=False)
        if "hidden1" not in data or int(data["hidden1"]) != HIDDEN1:
            raise ValueError("incompatible neural policy checkpoint (old network size)")
        weights = np.asarray(data["weights"], dtype=np.float64).reshape(-1)
        if weights.size == LEGACY_FLAT_SIZE:
            # v0.8.0 checkpoint（24 输入）→ 零初始化迁移
            return cls.from_legacy_weights(weights)
        if weights.size == V2_FLAT_SIZE:
            # v2 checkpoint（40 输入，文明版）→ 零初始化视觉列
            return cls.from_v2_weights(weights)
        if weights.size != CURRENT_FLAT_SIZE:
            raise ValueError(
                f"incompatible neural policy checkpoint ({weights.size} weights, "
                f"expected {CURRENT_FLAT_SIZE})"
            )
        return cls(weights=weights)


# ---------------------------------------------------------------------------
# 特征抽取
# ---------------------------------------------------------------------------
def _context_features(world, character) -> list[float]:
    """16 个附加输入：记忆摘要 (8) + 目标/计划 (4) + 文明 (4)。

    ``world.expanded_observation`` 关闭（消融组 A/B）*或*文明层缺失时
    全为 0，从而保持旧行为逐比特一致
    （见 :meth:`NeuralPolicy.from_legacy_weights`）。
    """
    context = [0.0] * CONTEXT_DIM
    if not bool(getattr(world, "expanded_observation", False)):
        return context
    agent_id = int(getattr(character, "agent_id", -1))

    memory = getattr(world, "memory", None)
    if memory is not None:
        try:
            vector = np.asarray(memory.feature_vector(agent_id), dtype=np.float64).reshape(-1)
            context[: min(8, vector.size)] = np.clip(vector[:8], -1.0, 1.0).tolist()
        except Exception:
            pass

    manager = getattr(world, "goal_manager", None)
    if manager is not None:
        try:
            goals = manager.goals_of(agent_id)
            plan = manager.active_plan(agent_id)
            context[8] = min(1.0, len([g for g in goals if g.status == "active"]) / 3.0)
            if plan is not None and plan.steps:
                context[9] = min(1.0, plan.current / max(1, len(plan.steps)))
            context[10] = 1.0 if plan is not None else 0.0
            context[11] = min(1.0, len(goals) / 20.0)
        except Exception:
            pass

    knowledge = getattr(world, "knowledge", None)
    if knowledge is not None:
        try:
            context[12] = min(1.0, len(knowledge.verified_ids()) / 6.0)
        except Exception:
            pass
    civilization = getattr(world, "civilization", None)
    if civilization is not None:
        try:
            facilities = list(getattr(civilization, "facilities", ()) or ())
            context[13] = min(1.0, len([f for f in facilities if getattr(f, "active", False)]) / 4.0)
            context[14] = min(1.0, len([a for a in world.agents if a.alive]) / 8.0)
            context[15] = max(0.0, min(1.0, float(getattr(civilization, "cooperation_index", 0.0))))
        except Exception:
            pass
    return context


def _visual_features(world, character) -> list[float]:
    """16 个真实视网膜特征（V2.1 感知打通）。

    由 ``Population`` / ``AgentMind`` 每隔 ``perception_stride`` 帧用
    :class:`newworld.visual_features.VisualFeatureExtractor` 计算一次，写入
    ``character.visual_features``；本函数只做读取、裁剪与降级，不做光学计算。
    世界未启用视觉特征、或个体尚无缓存时全为 0，从而保持旧 checkpoint
    行为逐比特一致（见 :meth:`NeuralPolicy.from_v2_weights`）。
    """
    visual = [0.0] * VISUAL_DIM
    if not bool(getattr(world, "enable_visual_features", False)):
        return visual
    raw = getattr(character, "visual_features", None)
    if raw is None:
        return visual
    try:
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        count = min(VISUAL_DIM, values.size)
        values = np.nan_to_num(values[:count], nan=0.0, posinf=1.0, neginf=-1.0)
        visual[:count] = np.clip(values, -1.0, 1.0).tolist()
    except Exception:
        pass
    return visual


def features(world, perception=None, character=None) -> np.ndarray:
    """神经策略的 56 维输入向量。

    前 24 个是 v0.8.0 原有的感官/生理特征；中间 16 个是文明上下文
    （记忆、目标、知识、社会）；最后 16 个是真实视网膜特征（V2.1）。
    世界未启用扩展观测/视觉特征时，相应段全为 0。
    """
    character = character if character is not None else world.agent
    scene = world.pixel_scene
    physiology = character.physiology
    cfg = world.config
    hour = world.hour
    feet_y = character.y if character.y is not None else cfg.ground_y

    # 最近浆果的相对方向/高度/存在标志
    berry_dx = 0.0
    berry_dy = 0.0
    berry_flag = 0.0
    if scene.berries:
        nearest = min(scene.berries, key=lambda item: abs(item[0] * ATOM_SIZE - character.x))
        berry_dx = max(-1.0, min(1.0, (nearest[0] * ATOM_SIZE - character.x) / 600.0))
        berry_dy = max(-1.0, min(1.0, (nearest[1] * ATOM_SIZE - feet_y) / 250.0))
        berry_flag = 1.0
    # 池塘方向与"附近有水"标志
    left, right = scene.pond
    pond_dx = max(-1.0, min(1.0, ((left + right) / 2.0 * ATOM_SIZE - character.x) / 800.0))
    pond_flag = (
        1.0
        if scene.water_near(
            int(character.x / ATOM_SIZE), int((feet_y - 40) / ATOM_SIZE), 6.0
        )
        else 0.0
    )
    moisture = float(scene.ecology.moisture[scene.surface].mean())
    vector = [
        character.x / cfg.width,
        character.vx / max(1.0, cfg.speed),
        float(character.facing),
        1.0 if character.on_ground else 0.0,
        max(-1.0, min(1.0, character.vy / 500.0)),
        physiology.energy / 100.0,
        physiology.hydration / 100.0,
        physiology.sleep_pressure / 100.0,
        physiology.alertness / 100.0,
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        world.daylight,
        berry_dx,
        berry_dy,
        berry_flag,
        pond_dx,
        pond_flag,
        scene.ecology.rain,
        moisture,
        (character.body_temperature_c - 32.0) / 10.0,
        min(1.0, character.jump_count / 10.0),
        min(1.0, character.speech_count / 20.0),
        hour / 24.0,
        scene.ecology.water_level,
    ]
    vector.extend(_context_features(world, character))
    vector.extend(_visual_features(world, character))
    return np.asarray(vector, dtype=np.float64)


# ---------------------------------------------------------------------------
# 神经行为
# ---------------------------------------------------------------------------
class NeuralInnerLife(InnerLife):
    """高层选择与话语都来自网络的行为引擎（决策链最底层）。"""

    def __init__(self, policy: NeuralPolicy, seed: int = 0) -> None:
        """初始化（随机权重或从扁平数组还原）。"""
        super().__init__(seed)
        self.policy = policy
        self.last_output: PolicyOutput | None = None
        # 说话冷却/自言自语计时/自发蹦跳计时
        self.speech_cooldown = 0.0
        self.talk_timer = 1.5
        self.jump_timer = 3.0

    def decide(self, world, dt: float, perception=None, character=None) -> Decision:
        """用网络输出驱动一帧行为（睡眠/喝水/觅食优先处理）。"""
        character = character if character is not None else world.agent
        physiology = character.physiology
        scene = world.pixel_scene
        hour = world.hour
        daylight = world.daylight
        output = self.policy.act(features(world, perception, character))
        self.last_output = output
        self.speech_cooldown -= dt
        self.talk_timer -= dt
        self.jump_timer -= dt
        speak = output.speak and self.speech_cooldown <= 0.0
        if self.talk_timer <= 0.0 and not physiology.sleeping:
            # 定时自言自语，让语言头保持活跃
            self.talk_timer = self.rng.uniform(2.5, 6.0)
            speak = True
        if speak:
            self.speech_cooldown = self.rng.uniform(2.0, 5.0)
        spontaneous_jump = self.jump_timer <= 0.0 and character.on_ground and not physiology.sleeping
        if spontaneous_jump:
            self.jump_timer = self.rng.uniform(4.0, 9.0)

        # -- 睡眠/醒来（与规则版 InnerLife 相同的守卫） -----------------
        if physiology.sleeping:
            character.behavior_state = "sleep"
            if daylight > 0.02 and (physiology.sleep_pressure < 45.0 or hour > 6.8):
                physiology.wake()
            else:
                return Decision(0, False, False, None)
        elif output.mode == "sleep" and (physiology.sleep_pressure > 30.0 or daylight < 0.08):
            physiology.sleep()
            character.behavior_state = "sleep"
            return Decision(0, False, False, None)

        feet_y = character.y if character.y is not None else world.config.ground_y
        hand_x = character.x + character.facing * HAND_REACH_PX

        if output.mode == "drink":
            # 喝水：先走到池塘，手碰到水就喝
            left, right = scene.pond
            target = (left + right) / 2.0 * ATOM_SIZE
            if abs(target - character.x) > 30.0:
                character.behavior_state = "forage"
                return Decision(1 if target > character.x else -1, False, False, None)
            atom_x = int(hand_x / ATOM_SIZE)
            atom_y = int((feet_y - HAND_HEIGHT_PX) / ATOM_SIZE)
            if scene.water_near(atom_x, atom_y, 4.0):
                physiology.drink()
                character.behavior_state = "drink"
                return Decision(0, False, speak, output.word if speak else None)
            character.behavior_state = "drink"
            return Decision(0, False, False, None)

        if output.mode in ("forage", "explore"):
            # 觅食：探索模式也响应低能量
            berry = self._nearest_berry(scene, character.x, feet_y)
            if berry is not None and (output.mode == "forage" or physiology.energy < 55.0):
                berry_x = berry[0] * ATOM_SIZE
                berry_y = berry[1] * ATOM_SIZE
                low = feet_y - 115.0 <= berry_y <= feet_y + 5.0
                reachable = abs(berry_x - character.x) < 42.0 and low
                if reachable:
                    scene.pick_berry(berry)
                    physiology.eat()
                    character.behavior_state = "eat"
                    return Decision(0, False, speak, output.word if speak else None)
                if abs(berry_x - character.x) > 18.0:
                    character.behavior_state = "forage"
                    return Decision(1 if berry_x > character.x else -1, False, False, None)
                if berry_y < feet_y - 115.0 and character.on_ground:
                    return Decision(0, True, False, None)

        if output.mode == "rest":
            character.behavior_state = "rest"
            return Decision(0, False, speak, output.word if speak else None)
        character.behavior_state = "explore"
        jump = (output.jump or spontaneous_jump) and character.on_ground
        move = output.move
        # 环境夹紧：永远不要顶着屏幕边缘走
        if character.at_left_wall and move < 0:
            move = 1
        elif character.at_right_wall and move > 0:
            move = -1
        return Decision(move, jump, speak, output.word if speak else None)
