import numpy as np

from neurolev.config import ArrayConfig
from neurolev.levitator import (DryRunTransport, Levitator, build_opposed_arrays, pressure_field,
                                quantize, trap_phases, wavenumber)

CFG = ArrayConfig()
TX = build_opposed_arrays(CFG)


def test_array_geometry():
    assert TX.positions.shape == (2 * CFG.grid ** 2, 3)
    assert TX.is_top.sum() == CFG.grid ** 2


def test_single_array_focuses_at_target():
    """Bottom array alone, no standing wave: pressure should peak at the target."""
    target = np.array([0.004, -0.003, 0.05])
    phases = trap_phases(target, TX, CFG, standing_wave=False)
    phases[TX.is_top] = 0
    bottom_only = TX.__class__(TX.positions[~TX.is_top], TX.normals[~TX.is_top], TX.is_top[~TX.is_top])
    xs = np.linspace(-0.015, 0.015, 121)
    line = np.column_stack([xs, np.full_like(xs, target[1]), np.full_like(xs, target[2])])
    p = np.abs(pressure_field(line, phases[~TX.is_top], bottom_only, CFG))
    assert abs(xs[np.argmax(p)] - target[0]) < 0.001


def test_standing_wave_puts_node_at_target():
    """With the pi offset, the target is a pressure minimum between two maxima along z."""
    target = np.array([0.0, 0.0, 0.05])
    phases = trap_phases(target, TX, CFG)
    wavelength = 2 * np.pi / wavenumber(CFG)
    zs = np.linspace(target[2] - wavelength / 2, target[2] + wavelength / 2, 201)
    line = np.column_stack([np.zeros_like(zs), np.zeros_like(zs), zs])
    p = np.abs(pressure_field(line, phases, TX, CFG))
    assert abs(zs[np.argmin(p)] - target[2]) < 0.0005
    assert p.min() < 0.2 * p.max()


def test_quantize_range():
    steps = quantize(np.linspace(0, 2 * np.pi, 50), 10)
    assert steps.min() >= 0 and steps.max() <= 9


def test_levitator_sends_one_frame_per_update():
    t = DryRunTransport(verbose=False)
    lev = Levitator(t, CFG)
    out = lev.set_traps([[0, 0, 0.05], [0.01, 0, 0.05]])
    assert t.frames_sent == 1
    assert len(out) == 2 * CFG.grid ** 2
