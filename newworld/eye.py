"""Human-like eye optics for the newworld inhabitant.

The model follows the standard schematic-eye/camera-lens treatment used in
visual optics and ophthalmology (cornea + crystalline lens -> equivalent thin
lens, pupil -> aperture, retina -> image plane):

* the cornea provides most of the focusing power (~70 %) and the crystalline
  lens the rest (~30 %); the lens can change shape to accommodate near objects
  (https://www.aao.org/eye-health/anatomy/parts-of-eye);
* the iris/pupil controls how much light reaches the retina, with a diameter of
  roughly 2 mm in bright light up to 8 mm in the dark;
* the image on the retina is limited by **defocus** (the eye is focused at a
  different distance than the object) and by **diffraction** (the pupil is a
  finite aperture: Airy disk, first minimum at ``1.22 * lambda / D``);
* acuity is highest in the fovea and falls off with eccentricity (the retina
  side of that lives in :mod:`newworld.retina`).

All angular quantities are in degrees; ``pupil_mm`` is the pupil diameter.
"""

from __future__ import annotations

import math

import numpy as np

from .optics import Panorama

#: wavelength used for the diffraction limit (photopic peak, nm -> m)
WAVELENGTH_M = 550e-9

#: radius of the Airy disk (first dark ring) is 1.22 * lambda / D; the Gaussian
#: approximation to the diffraction PSF has this standard deviation
_AIRY_TO_SIGMA = 0.42


class EyeParameters:
    """Anatomical / optical parameters of one eye."""

    __slots__ = (
        "field_of_view_deg",
        "max_rotation_deg",
        "focal_length_mm",
        "axial_length_mm",
        "pupil_min_mm",
        "pupil_max_mm",
        "resting_accommodation_dpt",
        "accommodation_range_dpt",
        "accommodation_tau_s",
        "world_scale_m_per_px",
        "panorama_samples_per_deg",
    )

    def __init__(self) -> None:
        #: horizontal field of view of a human eye (~160 deg monocular)
        self.field_of_view_deg = 160.0
        #: the eye can rotate up to ~50 deg inside the head
        self.max_rotation_deg = 50.0
        #: effective focal length of the human eye (~17 mm)
        self.focal_length_mm = 17.0
        #: distance from the lens to the retina (~24 mm)
        self.axial_length_mm = 24.0
        #: pupil diameter in bright light / darkness
        self.pupil_min_mm = 2.0
        self.pupil_max_mm = 8.0
        #: tonic accommodation when there is nothing to focus on (~1 diopter)
        self.resting_accommodation_dpt = 1.0
        #: accommodation range of a young adult (diopters)
        self.accommodation_range_dpt = 8.0
        #: time constant of the accommodation reflex
        self.accommodation_tau_s = 0.35
        #: how many world pixels make one metre (used to turn hit distances
        #: into diopters).  1 px = 1 cm makes the inhabitant ~1.4 m tall.
        self.world_scale_m_per_px = 0.01
        #: angular resolution (samples per degree) of the light field
        self.panorama_samples_per_deg = 120.0


# ---------------------------------------------------------------------------
# pupil + blur
# ---------------------------------------------------------------------------
def pupil_diameter_mm(mean_luminance: float, params: EyeParameters) -> float:
    """Pupil diameter as a function of retinal illuminance (log law).

    Bright white (luminance 1.0) -> 2 mm; dark (1e-3) -> 8 mm.  The mapping is
    a smooth log interpolation between those two extremes.
    """
    luminance = min(max(float(mean_luminance), 1e-4), 1.0)
    fraction = (math.log10(luminance) + 3.0) / 3.0  # 0 (dark) .. 1 (bright)
    fraction = min(1.0, max(0.0, fraction))
    return params.pupil_max_mm - (params.pupil_max_mm - params.pupil_min_mm) * fraction


def circle_of_confusion_deg(
    pupil_mm: float, defocus_dpt: float
) -> float:
    """Angular diameter of the defocus blur disk.

    For a thin lens the angular blur is approximately
    ``pupil_diameter * defocus_in_diopters`` (radians).
    """
    return math.degrees(pupil_mm * 1e-3 * abs(defocus_dpt))


def diffraction_limit_arcmin(pupil_mm: float) -> float:
    """Angular diameter of the Airy disk's first minimum, in arcminutes."""
    theta = 1.22 * WAVELENGTH_M / (pupil_mm * 1e-3)  # radians
    return math.degrees(theta) * 60.0


def blur_sigma_deg(pupil_mm: float, defocus_dpt: float) -> float:
    """Combined (diffraction + defocus) Gaussian-equivalent blur sigma.

    The defocus PSF is a disk of angular diameter ``c`` whose variance is
    ``c^2 / 16``, i.e. a Gaussian with ``sigma = c / 4``; the diffraction PSF is
    approximated by a Gaussian of ``sigma = 0.42 * lambda / D``.  Variances add.
    """
    coc = pupil_mm * 1e-3 * abs(defocus_dpt)          # radians
    sigma_defocus = coc / 4.0
    sigma_diffraction = _AIRY_TO_SIGMA * WAVELENGTH_M / (pupil_mm * 1e-3)
    return math.degrees(math.hypot(sigma_defocus, sigma_diffraction))


def defocus_diopters(distance_px: float, accommodation_dpt: float, params: EyeParameters) -> float:
    """Defocus of an object at ``distance_px`` for the current accommodation."""
    if not math.isfinite(distance_px) or distance_px <= 0.0:
        return abs(accommodation_dpt)  # looking at infinity
    distance_m = distance_px * params.world_scale_m_per_px
    vergence_dpt = 1.0 / max(distance_m, 1e-6)
    return abs(accommodation_dpt - vergence_dpt)


def accommodation_for_distance(distance_px: float, params: EyeParameters) -> float:
    """Accommodation (diopters) needed to focus an object at ``distance_px``."""
    if not math.isfinite(distance_px) or distance_px <= 0.0:
        return 0.0
    distance_m = distance_px * params.world_scale_m_per_px
    return min(params.accommodation_range_dpt, 1.0 / max(distance_m, 1e-6))


# ---------------------------------------------------------------------------
# angular blur
# ---------------------------------------------------------------------------
def angular_gaussian_blur(panorama: Panorama, sigma_deg: float) -> Panorama:
    """Blur a panorama with a 1-D Gaussian (edge-padded)."""
    if sigma_deg <= 1e-4 or len(panorama) < 3:
        return panorama
    step = float(panorama.rel_angles_deg[1] - panorama.rel_angles_deg[0])
    sigma_samples = max(1e-3, sigma_deg / abs(step))
    half_width = max(1, int(round(4.0 * sigma_samples)))
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    kernel /= kernel.sum()

    radiance = panorama.radiance.astype(np.float64)
    padded = np.pad(radiance, ((half_width, half_width), (0, 0)), mode="edge")
    if half_width > 24:
        # large saccadic/defocus blur: FFT convolution (O(n log n))
        length = padded.shape[0] + kernel.size - 1
        fft_length = 1
        while fft_length < length:
            fft_length *= 2
        kernel_padded = np.zeros(fft_length, dtype=np.float64)
        kernel_padded[: kernel.size] = kernel
        spectrum = np.fft.rfft(padded, fft_length, axis=0) * np.fft.rfft(
            kernel_padded, fft_length
        )[:, None]
        convolved = np.fft.irfft(spectrum, fft_length, axis=0)
        filtered = convolved[2 * half_width : 2 * half_width + len(panorama)].astype(np.float32)
    else:
        filtered = np.empty_like(radiance, dtype=np.float32)
        for channel in range(radiance.shape[1]):
            filtered[:, channel] = np.convolve(
                padded[:, channel], kernel, mode="valid"
            ).astype(np.float32)
    return Panorama(panorama.rel_angles_deg, filtered, panorama.distance_px)


# ---------------------------------------------------------------------------
# eye state
# ---------------------------------------------------------------------------
class EyeState:
    """Slow state of the eye: accommodation and pupil."""

    __slots__ = ("accommodation_dpt", "pupil_mm", "mean_luminance")

    def __init__(self, params: EyeParameters) -> None:
        self.accommodation_dpt = params.resting_accommodation_dpt
        self.pupil_mm = params.pupil_min_mm
        self.mean_luminance = 1.0

    def update(
        self,
        dt: float,
        fixation_distance_px: float,
        mean_luminance: float,
        params: EyeParameters,
    ) -> float:
        """Reflex updates; returns the defocus in diopters."""
        target = accommodation_for_distance(fixation_distance_px, params)
        alpha = min(1.0, dt / max(1e-3, params.accommodation_tau_s))
        self.accommodation_dpt += (target - self.accommodation_dpt) * alpha
        self.pupil_mm = pupil_diameter_mm(mean_luminance, params)
        self.mean_luminance = float(mean_luminance)
        return defocus_diopters(fixation_distance_px, self.accommodation_dpt, params)
