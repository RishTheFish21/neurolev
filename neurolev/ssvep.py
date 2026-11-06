"""SSVEP decoding with filter-bank canonical correlation analysis (FBCCA).

Idea: when you watch a light flickering at f Hz, your visual cortex produces
an oscillation at f, 2f, 3f... We compare a window of multi-channel EEG with
clean sine/cosine references at each candidate frequency. The reference that
correlates best (CCA finds the best mix of channels automatically) is the
light being watched. The filter bank repeats this on several frequency bands
so the weaker harmonics also contribute.

Reference: Chen, Wang, Gao, Jung & Gao (2015), "Filter bank canonical
correlation analysis for implementing a high-speed SSVEP-based brain-computer
interface", Journal of Neural Engineering 12(4).

No calibration data is needed, which is why a new user can start immediately.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import cheby1, sosfiltfilt

from .config import DecoderConfig


def reference_signals(freq: float, n_samples: int, fs: float, n_harmonics: int) -> np.ndarray:
    """Sine/cosine templates at `freq` and its harmonics. Shape (n_samples, 2*n_harmonics)."""
    t = np.arange(n_samples) / fs
    cols = []
    for h in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * h * freq * t))
        cols.append(np.cos(2 * np.pi * h * freq * t))
    return np.column_stack(cols)


def max_canonical_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Largest canonical correlation between column sets x and y.

    x: (n_samples, n_channels), y: (n_samples, n_refs).
    Uses QR + SVD, which is numerically stable and has no matrix inverses.
    """
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    qx, _ = np.linalg.qr(x)
    qy, _ = np.linalg.qr(y)
    s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(np.clip(s[0], 0.0, 1.0))


@dataclass
class Decision:
    index: int                 # which stimulus won
    frequency: float
    scores: np.ndarray         # one score per stimulus
    confidence_ratio: float    # best score / runner-up score
    confident: bool = False    # True if the ratio clears the threshold


class FBCCADecoder:
    def __init__(self, frequencies, fs: float, cfg: DecoderConfig = DecoderConfig()):
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.fs = float(fs)
        self.cfg = cfg

        nyquist = fs / 2
        top_harmonic = self.frequencies.max() * cfg.n_harmonics
        if top_harmonic >= nyquist:
            raise ValueError(
                f"{cfg.n_harmonics} harmonics of {self.frequencies.max()} Hz exceed "
                f"the Nyquist frequency ({nyquist} Hz). Use fewer harmonics."
            )
        if cfg.subband_low_edges[0] >= self.frequencies.min():
            raise ValueError("The first sub-band must start below the lowest stimulus frequency.")

        high = min(cfg.subband_high_edge, nyquist * 0.9)
        self._filters = [
            cheby1(4, 0.5, [low, high], btype="bandpass", fs=fs, output="sos")
            for low in cfg.subband_low_edges
            if low < high
        ]
        m = np.arange(1, len(self._filters) + 1)
        self._weights = m ** (-cfg.weight_a) + cfg.weight_b
        self._ref_cache = {}

    def _refs(self, n_samples: int):
        if n_samples not in self._ref_cache:
            self._ref_cache[n_samples] = [
                reference_signals(f, n_samples, self.fs, self.cfg.n_harmonics)
                for f in self.frequencies
            ]
        return self._ref_cache[n_samples]

    def scores(self, eeg: np.ndarray) -> np.ndarray:
        """eeg: (n_samples, n_channels) in microvolts. Returns one score per stimulus."""
        if eeg.ndim != 2:
            raise ValueError("eeg must be 2-D: (n_samples, n_channels)")
        refs = self._refs(eeg.shape[0])
        out = np.zeros(len(self.frequencies))
        for w, sos in zip(self._weights, self._filters):
            band = sosfiltfilt(sos, eeg, axis=0)
            for k, ref in enumerate(refs):
                out[k] += w * max_canonical_correlation(band, ref) ** 2
        return out

    def decide(self, eeg: np.ndarray) -> Decision:
        s = self.scores(eeg)
        order = np.argsort(s)[::-1]
        best, second = s[order[0]], s[order[1]]
        ratio = float(best / second) if second > 0 else float("inf")
        return Decision(
            index=int(order[0]),
            frequency=float(self.frequencies[order[0]]),
            scores=s,
            confidence_ratio=ratio,
            confident=ratio >= self.cfg.min_confidence_ratio,
        )


class AgreementGate:
    """Only lets a command through after N consecutive confident, matching decisions.

    This trades a little speed for far fewer accidental commands.
    """

    def __init__(self, required: int):
        self.required = max(1, required)
        self._last = None
        self._count = 0

    def update(self, decision: Decision):
        if not decision.confident:
            self._last, self._count = None, 0
            return None
        if decision.index == self._last:
            self._count += 1
        else:
            self._last, self._count = decision.index, 1
        if self._count >= self.required:
            self._last, self._count = None, 0   # reset so one gaze = one command
            return decision.index
        return None

    def reset(self):
        self._last, self._count = None, 0
