"""Phase solver and driver interface for two opposed 40 kHz ultrasonic arrays.

How a trap is made: each transducer is delayed (phase-shifted) so that its
wave arrives at the target point in step with every other transducer's wave.
Doing this for both the bottom array (firing up) and the top array (firing
down), with the top array shifted by an extra half cycle (pi), creates a
standing wave whose low-pressure node sits exactly at the target. A small
droplet or foam bead gets pushed into that node and stays there.

Move the target a little, recompute the phases, and the droplet follows.
"""

import struct
from dataclasses import dataclass

import numpy as np

from .config import ArrayConfig


@dataclass
class Transducers:
    positions: np.ndarray   # (n, 3) metres
    normals: np.ndarray     # (n, 3) unit vectors pointing where each one fires
    is_top: np.ndarray      # (n,) bool


def build_opposed_arrays(cfg: ArrayConfig = ArrayConfig()) -> Transducers:
    """Two flat grids facing each other: bottom at z=0, top at z=separation."""
    idx = (np.arange(cfg.grid) - (cfg.grid - 1) / 2) * cfg.pitch_m
    xx, yy = np.meshgrid(idx, idx, indexing="ij")
    flat = np.column_stack([xx.ravel(), yy.ravel()])
    n = flat.shape[0]

    bottom = np.column_stack([flat, np.zeros(n)])
    top = np.column_stack([flat, np.full(n, cfg.separation_m)])
    positions = np.vstack([bottom, top])
    normals = np.vstack([np.tile([0, 0, 1.0], (n, 1)), np.tile([0, 0, -1.0], (n, 1))])
    is_top = np.concatenate([np.zeros(n, bool), np.ones(n, bool)])
    return Transducers(positions, normals, is_top)


def wavenumber(cfg: ArrayConfig) -> float:
    return 2 * np.pi * cfg.frequency_hz / cfg.speed_of_sound


def trap_phases(targets, tx: Transducers, cfg: ArrayConfig = ArrayConfig(),
                standing_wave: bool = True) -> np.ndarray:
    """Emission phase (radians, 0..2pi) for every transducer.

    targets: one (x, y, z) point or a list of them. For several traps, each
    transducer's contributions are summed as complex numbers and the angle
    of the sum is used. That's a simple approach; fancier solvers (e.g.
    Gerchberg-Saxton) give stronger multi-trap fields.
    """
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    k = wavenumber(cfg)
    total = np.zeros(len(tx.positions), dtype=complex)
    for p in targets:
        d = np.linalg.norm(tx.positions - p, axis=1)
        total += np.exp(-1j * k * d)          # cancel the travel delay
    phases = np.angle(total)
    if standing_wave:
        phases = phases + np.pi * tx.is_top   # half-cycle offset -> node at target
    return np.mod(phases, 2 * np.pi)


def quantize(phases: np.ndarray, steps: int) -> np.ndarray:
    """Round phases to the driver's resolution. Returns integers 0..steps-1."""
    return (np.round(phases / (2 * np.pi) * steps).astype(int)) % steps


def pressure_field(points, phases, tx: Transducers, cfg: ArrayConfig = ArrayConfig(),
                   amplitude: float = 1.0) -> np.ndarray:
    """Complex pressure at `points` from point sources with a simple directivity.

    Good enough to check that traps land where they should; not a full
    acoustic simulation (Ultraino's own simulator is far more complete).
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    k = wavenumber(cfg)
    diff = points[:, None, :] - tx.positions[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    cos_theta = np.einsum("pnj,nj->pn", diff, tx.normals) / np.maximum(d, 1e-9)
    directivity = np.clip(cos_theta, 0, 1)
    contrib = amplitude * directivity / np.maximum(d, 1e-6) * np.exp(1j * (k * d + phases[None, :]))
    return contrib.sum(axis=1)


class DryRunTransport:
    """Prints instead of sending. Use it until the hardware is connected."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.frames_sent = 0

    def send(self, phase_steps: np.ndarray):
        self.frames_sent += 1
        if self.verbose:
            print(f"[dry-run] frame {self.frames_sent}: {len(phase_steps)} phases, "
                  f"first 8 = {phase_steps[:8].tolist()}")

    def close(self):
        pass


class SerialTransport:
    """Sends one frame of quantized phases over serial.

    Frame layout: 0xFE start byte, 2-byte little-endian count, then one byte
    per transducer. IMPORTANT: this is a placeholder protocol. Match it to the
    firmware you actually run (e.g. the Ultraino Arduino Mega sketch) before
    using it, or write the matching receiver yourself.
    """

    START = 0xFE

    def __init__(self, port: str, baud: int = 115200):
        try:
            import serial
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install pyserial first: pip install pyserial") from exc
        self._ser = serial.Serial(port, baud, timeout=0.1)

    def send(self, phase_steps: np.ndarray):
        payload = bytes(int(v) & 0xFF for v in phase_steps)
        header = bytes([self.START]) + struct.pack("<H", len(payload))
        self._ser.write(header + payload)

    def close(self):
        self._ser.close()


class Levitator:
    def __init__(self, transport, cfg: ArrayConfig = ArrayConfig()):
        self.cfg = cfg
        self.tx = build_opposed_arrays(cfg)
        self.transport = transport

    def set_traps(self, targets) -> np.ndarray:
        phases = trap_phases(targets, self.tx, self.cfg)
        steps = quantize(phases, self.cfg.phase_steps)
        self.transport.send(steps)
        return steps
