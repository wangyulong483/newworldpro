"""A human-like retina for the newworld inhabitant.

The blueprint of a human retina:

* ~120 million rods and ~6 million cones in the human retina; rods are
  concentrated in the periphery and are 500-1000x more sensitive to light,
  while cones are concentrated in the centre and carry colour vision
  (https://www.aao.org/eye-health/anatomy/rods,
   https://www.aao.org/eye-health/anatomy/cones);
* three cone types: ~60 % L (red), 30 % M (green), 10 % S (blue) cones
  (https://www.aao.org/eye-health/anatomy/cones);
* the fovea (centre of the macula) is where vision is sharpest; acuity falls
  off quickly towards the periphery
  (https://www.aao.org/eye-health/anatomy/fovea,
   https://www.aao.org/eye-health/anatomy/peripheral-retina);
* photoreceptors adapt to the mean light level over time, which is why the
  same object looks different after dark/light adaptation.

Because the newworld is a 2-D side-view world, the light field has one angular
dimension, so the retina here is a 1-D mosaic along eccentricity.  The full
human-like machinery (rod/cone mosaic, foveal acuity falloff, L/M/S colour
sampling, photopic/scotopic adaptation, centre-surround ganglion cells) is
implemented on that axis; if the world later becomes 2-D/3-D in the visual
sense, the same class extends naturally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .optics import Panorama

#: cone type codes
L_CONE = 0
M_CONE = 1
S_CONE = 2

#: sRGB (linear) -> LMS (Hunt-Pointer-Estevez, D65 normalised; all rows sum to 1)
_RGB_TO_LMS = np.array(
    (
        (0.38971, 0.68898, -0.07868),
        (-0.22981, 1.18340, 0.04641),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)

#: photopic luminance weights
_LUMINANCE = np.array((0.2126, 0.7152, 0.0722), dtype=np.float64)


@dataclass
class RetinaParameters:
    """Mosaic sizes and adaptation constants (scaled to something computable)."""

    #: total photoreceptors per mosaic (scaled down from ~126 million!)
    n_cones: int = 1800
    n_rods: int = 3000
    #: cone L/M/S proportions (AAO: 60/30/10)
    cone_proportions: tuple[float, float, float] = (0.60, 0.30, 0.10)
    #: central fovea (deg): no rods here, S cones suppressed in the foveola
    fovea_radius_deg: float = 1.5
    foveola_radius_deg: float = 0.5
    #: eccentricity constant E2 of the acuity falloff (deg)
    eccentricity_constant_deg: float = 2.5
    #: rods peak at ~18 deg eccentricity
    rod_peak_deg: float = 18.0
    rod_width_deg: float = 9.0
    #: photopic / scotopic adaptation time constants
    cone_adaptation_tau_s: float = 0.20
    rod_adaptation_tau_s: float = 8.0
    #: temporal filtering of the photoreceptor response
    response_tau_s: float = 0.05
    #: number of bins in the retinotopic feature vector
    n_bins: int = 32


@dataclass
class RetinaObservation:
    """What leaves the retina towards the brain (and towards RL policies)."""

    eccentricity_deg: np.ndarray       # (n_cones,)
    cone_types: np.ndarray             # (n_cones,) int, L/M/S
    cone_responses: np.ndarray         # (n_cones, 3) L/M/S response
    rod_eccentricity_deg: np.ndarray   # (n_rods,)
    rod_responses: np.ndarray          # (n_rods,)
    ganglion: np.ndarray               # (n_bins,) centre-surround output
    mean_luminance: float
    foveal_luminance: float
    adaptation: float

    def feature_vector(self) -> np.ndarray:
        """Compact, fixed-size retinotopic feature vector for learning."""
        bins = self.ganglion.size
        span = 160.0
        cone_hist = np.zeros((bins, 3), dtype=np.float32)
        if bins and self.eccentricity_deg.size:
            indices = np.clip(
                ((self.eccentricity_deg + span / 2.0) / span * (bins - 1)).astype(int),
                0,
                bins - 1,
            )
            for channel in range(3):
                np.add.at(cone_hist[:, channel], indices, self.cone_responses[:, channel])
        rod_hist = np.zeros(bins, dtype=np.float32)
        if bins and self.rod_responses.size:
            indices = np.clip(
                ((self.rod_eccentricity_deg + span / 2.0) / span * (bins - 1)).astype(int),
                0,
                bins - 1,
            )
            np.add.at(rod_hist, indices, self.rod_responses)
        scalars = np.array(
            (
                self.mean_luminance,
                self.foveal_luminance,
                self.adaptation,
            ),
            dtype=np.float32,
        )
        return np.concatenate(
            (cone_hist.ravel(), rod_hist, self.ganglion, scalars)
        ).astype(np.float32)


class Retina:
    """Photoreceptor mosaic + adaptation + retinal ganglion cells."""

    def __init__(self, params: RetinaParameters, rng: np.random.Generator) -> None:
        self.params = params
        self.rng = rng
        self.cone_gain = np.ones(3, dtype=np.float64)
        self.rod_gain = 1.0
        self._build_mosaic()

    # ------------------------------------------------------------------
    # mosaic
    # ------------------------------------------------------------------
    def _build_mosaic(self) -> None:
        params = self.params
        n_cones = params.n_cones
        fov = 160.0  # matches EyeParameters.field_of_view_deg
        self.cone_ecc = self._sample_angles(
            lambda ecc: 1.0 / (1.0 + ecc / params.eccentricity_constant_deg),
            n_cones,
            fov,
        )
        rod_density = lambda ecc: np.where(  # noqa: E731
            ecc < params.fovea_radius_deg,
            0.0,
            np.exp(-0.5 * ((ecc - params.rod_peak_deg) / params.rod_width_deg) ** 2),
        )
        self.rod_ecc = self._sample_angles(rod_density, params.n_rods, fov)

        l_prob, m_prob, _s_prob = params.cone_proportions
        total = l_prob + m_prob + params.cone_proportions[2]
        l_threshold = l_prob / total
        m_threshold = l_threshold + m_prob / total
        u = self.rng.random(n_cones)
        types = np.where(u < l_threshold, L_CONE, np.where(u < m_threshold, M_CONE, S_CONE))
        # the foveola has (almost) no S cones - a well known human property
        central = np.abs(self.cone_ecc) < params.foveola_radius_deg
        types[central & (types == S_CONE)] = M_CONE
        self.cone_types = types.astype(np.int8)

    @staticmethod
    def _sample_angles(density, n: int, fov: float) -> np.ndarray:
        grid = np.linspace(-fov / 2.0, fov / 2.0, 4001)
        weights = density(np.abs(grid)).astype(np.float64)
        weights[weights < 0] = 0.0
        if weights.sum() <= 0.0:
            return np.zeros(0, dtype=np.float64)
        cdf = np.cumsum(weights)
        cdf -= cdf[0]
        cdf /= cdf[-1]
        u = (np.arange(n) + 0.5) / n
        # stratified sampling → deterministic and smooth; tiny jitter avoids
        # perfectly regular aliasing of the mosaic
        jitter = (np.random.default_rng(12345).random(n) - 0.5) / n
        u = np.clip(u + 0.8 * jitter, 0.0, 1.0 - 1e-9)
        u.sort()
        return np.interp(u, cdf, grid)

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def sample(
        self,
        panorama: Panorama,
        dt: float,
        *,
        lid_transmission: float = 1.0,
        neural_gain: float = 1.0,
    ) -> RetinaObservation:
        """Turn an optical panorama into photoreceptor + ganglion responses."""
        params = self.params
        rel = panorama.rel_angles_deg
        radiance = panorama.radiance.astype(np.float64) * lid_transmission

        cone_rgb = np.stack(
            [np.interp(self.cone_ecc, rel, radiance[:, channel]) for channel in range(3)],
            axis=1,
        )
        lms = cone_rgb @ _RGB_TO_LMS.T
        lms = np.clip(lms, 0.0, None)

        panorama_luminance = radiance @ _LUMINANCE
        foveal = float(np.interp(0.0, rel, panorama_luminance))
        mean_lum = float(np.mean(panorama_luminance)) if panorama_luminance.size else 0.0

        # -- adaptation: gain follows the mean light level (Weber-like) --
        if dt > 0.0:
            alpha_cone = 1.0 - math.exp(-dt / max(1e-3, params.cone_adaptation_tau_s))
            target_gain = 1.0 / (lms.mean(axis=0) + 0.05)
            self.cone_gain += (target_gain - self.cone_gain) * alpha_cone

            alpha_rod = 1.0 - math.exp(-dt / max(1e-3, params.rod_adaptation_tau_s))
            target_rod = 1.0 / (mean_lum + 0.02)
            self.rod_gain += (target_rod - self.rod_gain) * alpha_rod

        # -- photoreceptor responses (Naka-Rushton soft saturation) -------
        activated = np.clip(lms * self.cone_gain, 0.0, None)
        cone_responses = activated / (1.0 + activated) * neural_gain

        rod_rgb = np.stack(
            [np.interp(self.rod_ecc, rel, radiance[:, channel]) for channel in range(3)],
            axis=1,
        )
        rod_lum = np.clip(rod_rgb @ _LUMINANCE, 0.0, None)
        rod_activated = np.clip(rod_lum * self.rod_gain, 0.0, None)
        photopic = 1.0 / (1.0 + (mean_lum / 0.05) ** 2)  # rods saturate in daylight
        rod_responses = rod_activated / (1.0 + rod_activated) * photopic * neural_gain

        cone_luminance = cone_rgb @ _LUMINANCE
        ganglion = self._ganglion(cone_luminance, neural_gain)
        return RetinaObservation(
            eccentricity_deg=self.cone_ecc.copy(),
            cone_types=self.cone_types.copy(),
            cone_responses=cone_responses.astype(np.float32),
            rod_eccentricity_deg=self.rod_ecc.copy(),
            rod_responses=rod_responses.astype(np.float32),
            ganglion=ganglion,
            mean_luminance=mean_lum,
            foveal_luminance=foveal,
            adaptation=float(np.mean(self.cone_gain)),
        )

    # ------------------------------------------------------------------
    # retinal ganglion cells
    # ------------------------------------------------------------------
    def _ganglion(self, luminance: np.ndarray, neural_gain: float) -> np.ndarray:
        """Centre-surround (difference-of-Gaussians) output on a regular grid.

        Uniform fields (like the current all-white world) give zero: retinal
        ganglion cells report contrast, not absolute light.
        """
        params = self.params
        bins = max(2, params.n_bins)
        if neural_gain <= 0.0:
            return np.zeros(bins, dtype=np.float32)
        profile = self.ganglion_profile(luminance, neural_gain=neural_gain)
        edges = np.linspace(0, profile.size, bins + 1).astype(int)
        return np.array(
            [profile[start:end].mean() for start, end in zip(edges[:-1], edges[1:])],
            dtype=np.float32,
        )

    def ganglion_profile(
        self,
        luminance: np.ndarray,
        *,
        neural_gain: float = 1.0,
        n_samples: int = 2048,
    ) -> np.ndarray:
        """Fine-grained centre-surround response (used for display / analysis)."""
        if neural_gain <= 0.0:
            return np.zeros(n_samples, dtype=np.float32)
        # resample the irregular photoreceptor mosaic onto a fine regular grid
        # so that the small centre-surround receptive fields can be resolved
        fine = np.linspace(-80.0, 80.0, n_samples)
        order = np.argsort(self.cone_ecc)
        sampled = np.interp(fine, self.cone_ecc[order], luminance[order])
        spacing = fine[1] - fine[0]
        centre = _gaussian_filter_1d(sampled, 0.10, spacing=spacing)
        surround = _gaussian_filter_1d(sampled, 0.35, spacing=spacing)
        return ((centre - surround) * neural_gain).astype(np.float32)


def _gaussian_filter_1d(values: np.ndarray, sigma: float, spacing: float) -> np.ndarray:
    sigma_samples = max(1e-3, sigma / max(1e-9, spacing))
    radius = max(1, int(round(4.0 * sigma_samples)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values.astype(np.float64), (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")
