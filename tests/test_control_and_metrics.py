import numpy as np
import pytest

from neurolev.artifacts import check_window
from neurolev.control import DropletController
from neurolev.eeg import SimulatedSource
from neurolev.metrics import information_transfer_rate


def test_itr_matches_wolpaw_formula():
    # 5 targets, 88% accuracy, 3 s per selection -> about 31 bits/min
    assert information_transfer_rate(5, 0.88, 3.0) == pytest.approx(31.05, abs=0.1)
    assert information_transfer_rate(5, 1.0, 3.0) == pytest.approx(np.log2(5) * 20)
    assert information_transfer_rate(5, 0.2, 3.0) == 0.0


def test_direction_commands_move_droplet_a():
    c = DropletController()
    start = c.state.a.copy()
    c.apply("RIGHT")
    assert c.state.a[0] == pytest.approx(start[0] + c.cfg.step_m)


def test_movement_is_clamped_to_bounds():
    c = DropletController()
    for _ in range(100):
        c.apply("LEFT")
    assert c.state.a[0] == pytest.approx(c.cfg.bounds_m[0][0])


def test_merge_eventually_joins_droplets():
    c = DropletController()
    for _ in range(50):
        c.apply("MERGE")
        if c.state.merged:
            break
    assert c.state.merged
    assert len(c.state.targets()) == 1


def test_artifact_check_flags_blinks_and_passes_clean_data():
    sim = SimulatedSource(seed=3)
    clean = sim.window(500, attended=11.0)
    assert check_window(clean, sim.fs).ok
    blink = clean.copy()
    blink[200:260, :] += 300.0
    report = check_window(blink, sim.fs)
    assert not report.ok and report.reason == "blink/movement"
