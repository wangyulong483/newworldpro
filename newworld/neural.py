"""Neural-network behaviour policy with a learned proto-language.

The inhabitant can be driven by a larger MLP:

* input: 24 sensory/physiology features;
* hidden: 64 tanh units -> 32 tanh units;
* output: 5 behaviour modes, 3 movement directions, jump/speak, and a
  **3-syllable word** drawn from a 16-syllable vocabulary.

The default checkpoint is ``assets/neural_policy.npz``.  It is loaded at start-up
and **never retrained automatically**.  Training only happens when the observer
presses **F2**; it continues from the loaded weights and overwrites the
checkpoint, so the policy evolves continuously across sessions.

The network's behaviour is bootstrapped by imitating the hand-written
homeostatic behaviour (walk / forage / eat / drink / sleep); the language head
learns a distinct word for each internal state, giving the creature a small
emergent vocabulary ("proto-language") rather than fixed random syllables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .behavior import Decision, HAND_HEIGHT_PX, HAND_REACH_PX, InnerLife
from .materials import ATOM_SIZE

# ---------------------------------------------------------------------------
# architecture
# ---------------------------------------------------------------------------
FEATURE_DIM = 24
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

#: a distinct word for each internal state (the language target during training)
MODE_WORDS: dict[str, tuple[str, ...]] = {
    "explore": ("ba", "ne", "po"),
    "forage": ("mi", "ta", "ki"),
    "drink": ("su", "vi", "da"),
    "sleep": ("no", "vi", "lu"),
    "rest": ("ta", "ne", "mo"),
}


@dataclass
class PolicyOutput:
    mode: str
    move: int
    jump: bool
    speak: bool
    word: str


class NeuralPolicy:
    """Two-hidden-layer MLP policy (numpy)."""

    def __init__(self, weights: np.ndarray | None = None, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        if weights is None:
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
        h1 = np.tanh(features @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        logits = h2 @ self.w3 + self.b3
        return logits, h1, h2

    def act(self, features: np.ndarray) -> PolicyOutput:
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
            version=np.asarray(2),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NeuralPolicy":
        data = np.load(Path(path), allow_pickle=False)
        if "hidden1" not in data or int(data["hidden1"]) != HIDDEN1:
            raise ValueError("incompatible neural policy checkpoint (old network size)")
        return cls(weights=data["weights"])


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------
def features(world, perception=None, character=None) -> np.ndarray:
    character = character if character is not None else world.agent
    scene = world.pixel_scene
    physiology = character.physiology
    cfg = world.config
    hour = world.hour
    feet_y = character.y if character.y is not None else cfg.ground_y

    berry_dx = 0.0
    berry_dy = 0.0
    berry_flag = 0.0
    if scene.berries:
        nearest = min(scene.berries, key=lambda item: abs(item[0] * ATOM_SIZE - character.x))
        berry_dx = max(-1.0, min(1.0, (nearest[0] * ATOM_SIZE - character.x) / 600.0))
        berry_dy = max(-1.0, min(1.0, (nearest[1] * ATOM_SIZE - feet_y) / 250.0))
        berry_flag = 1.0
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
    return np.asarray(vector, dtype=np.float64)


# ---------------------------------------------------------------------------
# neural behaviour
# ---------------------------------------------------------------------------
class NeuralInnerLife(InnerLife):
    """Behaviour whose high-level choices and words come from the network."""

    def __init__(self, policy: NeuralPolicy, seed: int = 0) -> None:
        super().__init__(seed)
        self.policy = policy
        self.last_output: PolicyOutput | None = None
        self.speech_cooldown = 0.0
        self.talk_timer = 1.5
        self.jump_timer = 3.0

    def decide(self, world, dt: float, perception=None, character=None) -> Decision:
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
            self.talk_timer = self.rng.uniform(2.5, 6.0)
            speak = True
        if speak:
            self.speech_cooldown = self.rng.uniform(2.0, 5.0)
        spontaneous_jump = self.jump_timer <= 0.0 and character.on_ground and not physiology.sleeping
        if spontaneous_jump:
            self.jump_timer = self.rng.uniform(4.0, 9.0)

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
        # environmental clamp: never keep pushing into the screen edge
        if character.at_left_wall and move < 0:
            move = 1
        elif character.at_right_wall and move > 0:
            move = -1
        return Decision(move, jump, speak, output.word if speak else None)
