"""Offline sanity check of the decoder on simulated EEG at several noise levels.

Useful for tuning window length and thresholds before real recordings.
Simulated numbers are NOT real results; use run_eval_session.py for those.

  python scripts/benchmark_simulated.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurolev.config import Config  # noqa: E402
from neurolev.eeg import SimulatedSource  # noqa: E402
from neurolev.metrics import information_transfer_rate  # noqa: E402
from neurolev.ssvep import FBCCADecoder  # noqa: E402


def main(trials_per_target: int = 30):
    cfg = Config()
    freqs = cfg.stimulus.frequencies
    print(f"{'signal uV':>9} {'window s':>8} {'accuracy':>9} {'ITR bits/min':>13}")
    for snr in (1.0, 1.5, 2.0, 3.0):
        for window in (1.0, 2.0, 2.5, 3.0):
            sim = SimulatedSource(snr_uv=snr, seed=42)
            dec = FBCCADecoder(freqs, sim.fs, cfg.decoder)
            n = int(window * sim.fs)
            hits = []
            for target, f in enumerate(freqs):
                for _ in range(trials_per_target):
                    hits.append(dec.decide(sim.window(n, attended=f)).index == target)
            acc = float(np.mean(hits))
            itr = information_transfer_rate(len(freqs), acc, window + 1.0)
            print(f"{snr:9.1f} {window:8.1f} {acc:9.1%} {itr:13.1f}")


if __name__ == "__main__":
    main()
