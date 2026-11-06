"""Reject EEG windows contaminated by blinks, movement or muscle tension.

Blinks and head movement cause large slow swings (high peak-to-peak voltage).
Jaw clenching adds broadband high-frequency muscle (EMG) noise. Either one can
fool the decoder into a false command, so those windows are skipped.
"""

from dataclasses import dataclass

import numpy as np

from .config import ArtifactConfig


@dataclass
class ArtifactReport:
    ok: bool
    peak_to_peak_uv: float
    emg_ratio: float
    reason: str = ""


def check_window(eeg: np.ndarray, fs: float, cfg: ArtifactConfig = ArtifactConfig()) -> ArtifactReport:
    """eeg: (n_samples, n_channels) in microvolts."""
    centered = eeg - eeg.mean(axis=0)
    ptp = float(np.max(np.ptp(centered, axis=0)))

    spectrum = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    freqs = np.fft.rfftfreq(centered.shape[0], d=1.0 / fs)
    in_band = (freqs >= 1.0) & (freqs <= min(90.0, fs / 2))
    muscle = (freqs >= 30.0) & in_band
    total = spectrum[in_band].sum()
    emg_ratio = float(spectrum[muscle].sum() / total) if total > 0 else 0.0

    if ptp > cfg.max_peak_to_peak_uv:
        return ArtifactReport(False, ptp, emg_ratio, "blink/movement")
    if emg_ratio > cfg.max_emg_power_ratio:
        return ArtifactReport(False, ptp, emg_ratio, "muscle (EMG)")
    return ArtifactReport(True, ptp, emg_ratio)
