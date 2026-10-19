"""Perception for the newworld inhabitant.

This module chains the physical pieces into a human-like sensory system:

``scene`` (light field) -> eye optics (pupil / accommodation / defocus /
diffraction) -> retina (cone/rod mosaic, foveal acuity, L/M/S colour, light
adaptation) -> retinal ganglion cells (centre-surround contrast).

On top of that it implements the *active* part of human vision: the eye is
never still.  Human observers make 3-4 saccades per second, each fixation lasts
about 200-300 ms and each saccade 5-100 ms; during a saccade the image sweeps
across the retina at hundreds of degrees per second and perception is strongly
suppressed ("saccadic suppression").  There are also microsaccades and blinks
(https://www.cs.cmu.edu/~xingyu/saccade.html).  All of that is modelled here,
because it determines what the inhabitant can actually see.

The other senses are assembled by :class:`Perceiver`:

* **vision** — the optical/retinal pipeline above;
* **audition** — sound pressure waves in the air
  (:mod:`newworld.acoustics`), arriving at the ear after ``r / c`` seconds and
  converted to dB SPL (threshold 20 uPa);
* **thermoception / interoception** — body temperature from the heat balance
  (:mod:`newworld.environment`) and the air temperature.

The visual field is one-dimensional because the world is a 2-D side view; the
retina is a 1-D foveated mosaic over azimuth.  The inhabitant's *own* world is
currently a uniform white environment, so the optical image is uniform and the
ganglion cells (contrast detectors) report ~zero contrast — the honest result of
looking at an empty white world.  As soon as objects are added to the world the
same pipeline produces real images.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

import numpy as np

from . import physics
from .character import eye_position
from .eye import (
    EyeParameters,
    EyeState,
    accommodation_for_distance,
    angular_gaussian_blur,
    blur_sigma_deg,
    defocus_diopters,
    pupil_diameter_mm,
)
from .optics import Scene
from .retina import Retina, RetinaObservation, RetinaParameters


# ---------------------------------------------------------------------------
# gaze / eye movements
# ---------------------------------------------------------------------------
@dataclass
class GazeParameters:
    """Human eye-movement statistics (see module docstring)."""

    saccades_per_second: float = 3.5
    fixation_s: tuple[float, float] = (0.20, 0.30)
    saccade_deg: tuple[float, float] = (1.0, 12.0)
    max_rotation_deg: float = 50.0
    microsaccade_per_second: float = 1.0
    microsaccade_deg: tuple[float, float] = (0.17, 0.5)
    blink_per_minute: float = 17.0
    blink_s: tuple[float, float] = (0.10, 0.40)
    drift_deg_per_s: float = 0.10


@dataclass
class GazeState:
    offset_deg: float = 0.0
    target_deg: float = 0.0
    start_deg: float = 0.0
    mode: str = "fixation"  # "fixation" | "saccade"
    timer_s: float = 0.0
    saccade_duration_s: float = 0.0
    saccade_progress: float = 0.0
    saccade_count: int = 0
    blink: bool = False
    blink_timer_s: float = 0.0
    next_blink_s: float = 3.0
    microsaccade_timer_s: float = 0.5
    blink_count: int = 0
    drift: float = 0.0


class GazeController:
    """Saccades, microsaccades, drift and blinks."""

    def __init__(self, params: GazeParameters, rng: random.Random) -> None:
        self.params = params
        self.rng = rng
        self.state = GazeState(offset_deg=rng.uniform(-5.0, 5.0))
        self.state.timer_s = rng.uniform(*params.fixation_s)
        self.state.next_blink_s = self._next_blink_time()

    def _next_blink_time(self) -> float:
        mean = 60.0 / max(1e-3, self.params.blink_per_minute)
        return self.rng.expovariate(1.0 / mean)

    def reset(self, rng: random.Random | None = None) -> None:
        if rng is not None:
            self.rng = rng
        self.state = GazeState(offset_deg=self.rng.uniform(-5.0, 5.0))
        self.state.timer_s = self.rng.uniform(*self.params.fixation_s)
        self.state.next_blink_s = self._next_blink_time()

    # ------------------------------------------------------------------
    def _start_saccade(self, amplitude_deg: float) -> None:
        state = self.state
        # human main sequence: duration grows with amplitude (~21 + 2.2*|A| ms)
        duration = (21.0 + 2.2 * abs(amplitude_deg)) / 1000.0
        state.mode = "saccade"
        state.start_deg = state.offset_deg
        target = state.offset_deg + amplitude_deg
        limit = self.params.max_rotation_deg
        state.target_deg = max(-limit, min(limit, target))
        state.saccade_duration_s = max(0.015, min(0.20, duration))
        state.saccade_progress = 0.0
        state.saccade_count += 1

    def update(self, dt: float) -> GazeState:
        params = self.params
        state = self.state

        # -- blinks -----------------------------------------------------
        if state.blink:
            state.blink_timer_s -= dt
            if state.blink_timer_s <= 0.0:
                state.blink = False
                state.next_blink_s = self._next_blink_time()
        else:
            state.next_blink_s -= dt
            if state.next_blink_s <= 0.0:
                state.blink = True
                state.blink_timer_s = self.rng.uniform(*params.blink_s)
                state.blink_count += 1

        # -- fixational drift + microsaccades ---------------------------
        state.drift += self.rng.gauss(0.0, params.drift_deg_per_s * dt)
        state.microsaccade_timer_s -= dt
        if state.microsaccade_timer_s <= 0.0:
            state.microsaccade_timer_s = self.rng.expovariate(
                max(1e-3, params.microsaccade_per_second)
            )
            state.drift += self.rng.choice((-1.0, 1.0)) * self.rng.uniform(
                *params.microsaccade_deg
            )

        # -- saccade / fixation ----------------------------------------
        if state.mode == "saccade":
            state.saccade_progress += dt / max(1e-3, state.saccade_duration_s)
            progress = min(1.0, state.saccade_progress)
            # smooth acceleration/deceleration of a ballistic saccade
            eased = progress * progress * (3.0 - 2.0 * progress)
            state.offset_deg = state.start_deg + (state.target_deg - state.start_deg) * eased
            if progress >= 1.0:
                state.mode = "fixation"
                state.timer_s = self.rng.uniform(*params.fixation_s)
        else:
            state.timer_s -= dt
            if state.timer_s <= 0.0:
                amplitude = self.rng.uniform(*params.saccade_deg)
                amplitude *= self.rng.choice((-1.0, 1.0))
                self._start_saccade(amplitude)

        limit = params.max_rotation_deg
        state.offset_deg = max(-limit, min(limit, state.offset_deg))
        return state


# ---------------------------------------------------------------------------
# perception
# ---------------------------------------------------------------------------
@dataclass
class PerceptionState:
    """Everything the inhabitant perceives in one simulation step."""

    time: float
    eye_x: float
    eye_y: float
    head_deg: float
    gaze_offset_deg: float
    gaze_world_deg: float
    mode: str
    blink: bool
    pupil_mm: float
    accommodation_dpt: float
    defocus_dpt: float
    blur_sigma_deg: float
    mean_luminance: float
    foveal_luminance: float
    saccade_velocity_deg_s: float
    retina: RetinaObservation
    # -- other senses (filled by :class:`Perceiver`) --------------------
    body_temperature_c: float = physics.HUMAN_BODY_TEMPERATURE_K - 273.15
    air_temperature_c: float = physics.AIR_TEMPERATURE_K - 273.15
    sound_pressure_pa: float = 0.0
    sound_spl_db: float = 0.0
    hearing: bool = False
    feet_y: float = 0.0
    vy: float = 0.0
    on_ground: bool = True
    utterance: str | None = None
    solar_irradiance_w_m2: float = 0.0

    def feature_vector(self) -> np.ndarray:
        return self.retina.feature_vector()

    def summary(self) -> str:
        return (
            f"gaze={self.gaze_world_deg:+06.1f}deg mode={self.mode:8s} "
            f"pupil={self.pupil_mm:4.1f}mm focus={self.accommodation_dpt:4.2f}D "
            f"blur={self.blur_sigma_deg:4.2f}deg blink={int(self.blink)} "
            f"body={self.body_temperature_c:4.1f}C sound={self.sound_spl_db:4.1f}dB"
        )


class VisualSystem:
    """Eye + retina + gaze control, producing a PerceptionState each step."""

    def __init__(
        self,
        seed: int | None = 0,
        *,
        eye_parameters: EyeParameters | None = None,
        retina_parameters: RetinaParameters | None = None,
        gaze_parameters: GazeParameters | None = None,
    ) -> None:
        self.eye_parameters = eye_parameters or EyeParameters()
        self.retina_parameters = retina_parameters or RetinaParameters()
        self.gaze_parameters = gaze_parameters or GazeParameters()
        self.eye_state = EyeState(self.eye_parameters)
        self._seed = 0 if seed is None else int(seed)
        self.retina = Retina(
            self.retina_parameters, np.random.default_rng(self._seed)
        )
        self.gaze = GazeController(self.gaze_parameters, random.Random(self._seed))
        self.time = 0.0
        self.saccade_velocity_deg_s = 0.0
        self.last_perception: PerceptionState | None = None

    # ------------------------------------------------------------------
    def perceive(
        self,
        scene: Scene,
        character,
        config,
        dt: float,
    ) -> PerceptionState:
        """Look at the scene through the eye and return the retinal signal."""
        self.time += dt
        gaze = self.gaze.update(dt)

        head_deg = 0.0 if character.facing >= 0 else 180.0
        gaze_world = head_deg + gaze.offset_deg
        eye_x, eye_y = eye_position(character, config)

        mean_lum_prev = self.eye_state.mean_luminance
        pupil = pupil_diameter_mm(mean_lum_prev, self.eye_parameters)

        panorama = scene.panorama(
            eye_x,
            eye_y,
            gaze_world,
            self.eye_parameters.field_of_view_deg,
            self.eye_parameters.panorama_samples_per_deg,
        )

        # -- accommodation is driven by whatever the fovea is looking at --
        fixation_distance = float(
            np.interp(0.0, panorama.rel_angles_deg, panorama.distance_px)
        )
        if not math.isfinite(fixation_distance):
            target_accommodation = self.eye_parameters.resting_accommodation_dpt
        else:
            target_accommodation = accommodation_for_distance(
                fixation_distance, self.eye_parameters
            )
        alpha = min(
            1.0, dt / max(1e-3, self.eye_parameters.accommodation_tau_s)
        )
        self.eye_state.accommodation_dpt += (
            target_accommodation - self.eye_state.accommodation_dpt
        ) * alpha
        self.eye_state.pupil_mm = pupil
        defocus = defocus_diopters(
            fixation_distance, self.eye_state.accommodation_dpt, self.eye_parameters
        )

        # -- optical blur: defocus + diffraction + saccadic smear --------
        sigma = blur_sigma_deg(pupil, defocus)
        if gaze.mode == "saccade":
            amplitude = abs(gaze.target_deg - gaze.start_deg)
            velocity = amplitude / max(1e-3, gaze.saccade_duration_s)
            self.saccade_velocity_deg_s = velocity
            sigma += min(5.0, 0.5 * velocity * dt)
        else:
            self.saccade_velocity_deg_s = 0.0
        blurred = angular_gaussian_blur(panorama, sigma)

        # -- retinal image ----------------------------------------------
        lid = 0.0 if gaze.blink else 1.0
        neural_gain = 0.25 if gaze.mode == "saccade" else 1.0  # saccadic suppression
        retina_observation = self.retina.sample(
            blurred, dt, lid_transmission=lid, neural_gain=neural_gain
        )

        self.eye_state.mean_luminance = retina_observation.mean_luminance
        perception = PerceptionState(
            time=self.time,
            eye_x=eye_x,
            eye_y=eye_y,
            head_deg=head_deg,
            gaze_offset_deg=gaze.offset_deg,
            gaze_world_deg=gaze_world,
            mode=gaze.mode,
            blink=gaze.blink,
            pupil_mm=pupil,
            accommodation_dpt=self.eye_state.accommodation_dpt,
            defocus_dpt=defocus,
            blur_sigma_deg=sigma,
            mean_luminance=retina_observation.mean_luminance,
            foveal_luminance=retina_observation.foveal_luminance,
            saccade_velocity_deg_s=self.saccade_velocity_deg_s,
            retina=retina_observation,
        )
        self.last_perception = perception
        return perception

    # ------------------------------------------------------------------
    def summary_lines(self) -> list[str]:
        """Compact human-readable state for the debug overlay."""
        perception = self.last_perception
        eye = self.eye_parameters
        retina = self.retina_parameters
        if perception is None:
            return [
                f"vision: FOV {eye.field_of_view_deg:.0f}deg  retina "
                f"{retina.n_cones} cones + {retina.n_rods} rods  (starting up)"
            ]
        return [
            f"vision  FOV {eye.field_of_view_deg:.0f}deg  retina {retina.n_cones} cones "
            f"+ {retina.n_rods} rods  pupil {perception.pupil_mm:.1f}mm",
            f"gaze {perception.gaze_world_deg:+7.1f}deg  mode {perception.mode:8s} "
            f"blink {'yes' if perception.blink else 'no ':3s} "
            f"blur {perception.blur_sigma_deg:4.2f}deg  focus {perception.accommodation_dpt:4.2f}D",
            f"retina  mean={perception.mean_luminance:5.3f}  fovea={perception.foveal_luminance:5.3f}  "
            f"saccades={self.gaze.state.saccade_count:4d}  blinks={self.gaze.state.blink_count:3d}",
        ]


# ---------------------------------------------------------------------------
# other senses: hearing, thermoception
# ---------------------------------------------------------------------------
@dataclass
class AuditoryObservation:
    """What arrives at the ear from the acoustic field."""

    pressure_pa: float
    rms_pa: float
    spl_db: float
    hearing: bool


class Audition:
    """Hearing through the air: pressure waves -> dB SPL."""

    def __init__(self) -> None:
        self.last_observation: AuditoryObservation | None = None

    def listen(self, world, character) -> AuditoryObservation:
        ear_x, ear_y = eye_position(character, world.config)
        pressure = world.sound.pressure_at(ear_x, ear_y, world.t)
        rms = world.sound.rms_pressure_at(ear_x, ear_y, world.t)
        level = physics.spl_db(rms)
        observation = AuditoryObservation(
            pressure_pa=pressure,
            rms_pa=rms,
            spl_db=level,
            hearing=rms > 0.0 and level >= 0.0,  # threshold of hearing (20 uPa)
        )
        self.last_observation = observation
        return observation


class Perceiver:
    """All senses together: vision + hearing + thermoception + interoception."""

    def __init__(self, seed: int | None = 0) -> None:
        self.visual = VisualSystem(seed)
        self.audition = Audition()
        self.last_perception: PerceptionState | None = None
        self._world = None
        self._agent = None

    def perceive(self, world, dt: float, character=None) -> PerceptionState:
        agent = character if character is not None else world.agent
        visual = self.visual.perceive(world.scene(), agent, world.config, dt)
        audio = self.audition.listen(world, agent)
        active_utterance = (
            agent.utterance if agent.utterance_until >= world.t - 1e-9 else None
        )
        perception = replace(
            visual,
            body_temperature_c=agent.body_temperature_c,
            air_temperature_c=world.air.temperature_c,
            sound_pressure_pa=audio.pressure_pa,
            sound_spl_db=audio.spl_db,
            hearing=audio.hearing,
            feet_y=agent.y if agent.y is not None else world.config.ground_y,
            vy=agent.vy,
            on_ground=agent.on_ground,
            utterance=active_utterance,
            solar_irradiance_w_m2=world.sun.horizontal_irradiance,
        )
        self.last_perception = perception
        self._world = world
        self._agent = agent
        return perception

    # ------------------------------------------------------------------
    def feature_vector(self) -> np.ndarray | None:
        """Retinotopic features plus the non-visual sensory scalars."""
        perception = self.last_perception
        if perception is None:
            return None
        extra = np.array(
            (
                perception.body_temperature_c / 50.0,
                perception.air_temperature_c / 50.0,
                perception.sound_spl_db / 120.0,
                perception.feet_y / 2000.0,
                perception.vy / 600.0,
                1.0 if perception.on_ground else 0.0,
            ),
            dtype=np.float32,
        )
        return np.concatenate((perception.feature_vector(), extra))

    # ------------------------------------------------------------------
    def summary_lines(self) -> list[str]:
        lines = self.visual.summary_lines()
        perception = self.last_perception
        agent = self._agent
        if perception is None or agent is None:
            return lines
        world = self._world
        lines.append(
            f"mechanics  feet y={perception.feet_y:7.1f}px  vy={perception.vy:+7.1f}px/s  "
            f"on_ground={str(perception.on_ground):5s}  jumps={agent.jump_count}"
        )
        lines.append(
            f"thermo  body={perception.body_temperature_c:5.1f}C  "
            f"air={perception.air_temperature_c:5.1f}C  sun={perception.solar_irradiance_w_m2:5.0f}W/m2"
        )
        heard = "yes" if perception.hearing else "no "
        says = f"  says '{perception.utterance}'" if perception.utterance else ""
        lines.append(
            f"audio   {perception.sound_spl_db:5.1f}dB SPL  "
            f"heard={heard}  speech={agent.speech_count:3d}{says}"
        )
        if (
            abs(perception.mean_luminance - 1.0) < 1e-3
            and float(np.abs(perception.retina.ganglion).max(initial=0.0)) < 1e-4
        ):
            lines.append("view: uniform white (the world currently contains nothing else)")
        return lines
