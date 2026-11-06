import numpy as np
import pytest

from neurolev.config import Config
from neurolev.eeg import SimulatedSource
from neurolev.ssvep import AgreementGate, Decision, FBCCADecoder, max_canonical_correlation, reference_signals


def test_cca_finds_perfect_match():
    fs, n = 200.0, 400
    ref = reference_signals(12.0, n, fs, 2)
    x = np.column_stack([ref[:, 0] * 2 + ref[:, 1], ref[:, 0] - ref[:, 1]])
    assert max_canonical_correlation(x, ref) == pytest.approx(1.0, abs=1e-6)


def test_cca_low_for_unrelated_signals():
    rng = np.random.default_rng(0)
    ref = reference_signals(12.0, 500, 200.0, 2)
    x = rng.normal(size=(500, 4))
    assert max_canonical_correlation(x, ref) < 0.3


def test_decoder_identifies_each_target_on_simulated_eeg():
    cfg = Config()
    sim = SimulatedSource(snr_uv=3.0, seed=1)
    dec = FBCCADecoder(cfg.stimulus.frequencies, sim.fs, cfg.decoder)
    n = int(cfg.decoder.window_s * sim.fs)
    hits = []
    for target, f in enumerate(cfg.stimulus.frequencies):
        for _ in range(10):
            hits.append(dec.decide(sim.window(n, attended=f)).index == target)
    assert np.mean(hits) >= 0.9


def test_decoder_rejects_harmonics_above_nyquist():
    cfg = Config()
    with pytest.raises(ValueError):
        FBCCADecoder((20.0, 30.0, 40.0), fs=100.0, cfg=cfg.decoder)


def _decision(index, confident=True):
    return Decision(index=index, frequency=0.0, scores=np.zeros(5),
                    confidence_ratio=2.0 if confident else 1.0, confident=confident)


def test_agreement_gate_needs_consecutive_matches():
    gate = AgreementGate(required=2)
    assert gate.update(_decision(3)) is None
    assert gate.update(_decision(3)) == 3
    # resets after firing, so the next single decision doesn't fire
    assert gate.update(_decision(3)) is None


def test_agreement_gate_ignores_low_confidence():
    gate = AgreementGate(required=2)
    gate.update(_decision(1))
    assert gate.update(_decision(1, confident=False)) is None
    assert gate.update(_decision(1)) is None
