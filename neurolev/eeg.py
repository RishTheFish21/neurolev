"""EEG sources: a real headset through BrainFlow, or a simulator for testing.

Both expose the same interface:
    source.start()
    source.latest(n_samples) -> np.ndarray of shape (n_samples, n_channels), microvolts
    source.fs
    source.stop()
"""

import time

import numpy as np


class BrainFlowSource:
    """Streams from any BrainFlow-supported board (OpenBCI Ganglion, Cyton, ...).

    board: "ganglion"          OpenBCI Ganglion via the USB dongle (needs serial_port)
           "ganglion-native"   OpenBCI Ganglion over the computer's own Bluetooth
           "cyton"             OpenBCI Cyton (needs serial_port)
           "synthetic"         BrainFlow's built-in fake board (no hardware)
    channels: indices into the board's EEG channel list to keep, e.g. the ones
              you placed over O1, Oz, O2. None keeps all of them.
    """

    def __init__(self, board: str = "synthetic", serial_port: str = "", mac_address: str = "",
                 channels=None):
        try:
            from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise ImportError("Install BrainFlow first: pip install brainflow") from exc

        ids = {
            "ganglion": BoardIds.GANGLION_BOARD,
            "ganglion-native": BoardIds.GANGLION_NATIVE_BOARD,
            "cyton": BoardIds.CYTON_BOARD,
            "synthetic": BoardIds.SYNTHETIC_BOARD,
        }
        if board not in ids:
            raise ValueError(f"Unknown board '{board}'. Options: {', '.join(ids)}")

        params = BrainFlowInputParams()
        params.serial_port = serial_port
        params.mac_address = mac_address
        self._board_id = ids[board].value
        self._shim = BoardShim(self._board_id, params)
        self._eeg_rows = BoardShim.get_eeg_channels(self._board_id)
        if channels is not None:
            self._eeg_rows = [self._eeg_rows[i] for i in channels]
        self.fs = float(BoardShim.get_sampling_rate(self._board_id))

    def start(self):
        self._shim.prepare_session()
        self._shim.start_stream()
        time.sleep(1.0)  # let the buffer fill a little

    def latest(self, n_samples: int) -> np.ndarray:
        data = self._shim.get_current_board_data(n_samples)
        return data[self._eeg_rows, :].T

    def stop(self):
        try:
            self._shim.stop_stream()
        finally:
            self._shim.release_session()


class SimulatedSource:
    """Fake occipital EEG containing an SSVEP at whichever frequency is 'attended'.

    Good for testing the pipeline end to end before the headset arrives.
    Real SSVEPs are usually weaker and messier than this, so don't treat
    simulated accuracy as a real result.
    """

    def __init__(self, fs: float = 200.0, n_channels: int = 4, snr_uv: float = 2.0,
                 noise_uv: float = 8.0, seed: int = 0):
        self.fs = fs
        self.n_channels = n_channels
        self.snr_uv = snr_uv
        self.noise_uv = noise_uv
        self.attended = None          # Hz, or None for "looking at nothing"
        self._rng = np.random.default_rng(seed)
        self._gains = self._rng.uniform(0.5, 1.0, n_channels)
        self._t0 = time.monotonic()

    def start(self):
        self._t0 = time.monotonic()

    def stop(self):
        pass

    def window(self, n_samples: int, attended=None, t_start: float = 0.0) -> np.ndarray:
        """Generate a window directly (used by tests and offline benchmarks)."""
        t = t_start + np.arange(n_samples) / self.fs
        x = np.zeros((n_samples, self.n_channels))
        if attended is not None:
            phase = self._rng.uniform(0, 2 * np.pi)
            for h, amp in ((1, 1.0), (2, 0.5), (3, 0.25)):
                wave = amp * self.snr_uv * np.sin(2 * np.pi * h * attended * t + h * phase)
                x += np.outer(wave, self._gains)
        # Background: 1/f ("pink") brain noise like real EEG, an alpha rhythm,
        # a small white noise floor from the amplifier, and mains hum.
        x += self._pink_noise(n_samples) * self.noise_uv
        alpha = 3.0 * np.sin(2 * np.pi * 10.0 * t + self._rng.uniform(0, 2 * np.pi))
        x += np.outer(alpha, self._rng.uniform(0.3, 1.0, self.n_channels))
        x += self._rng.normal(0, 0.15 * self.noise_uv, (n_samples, self.n_channels))
        x += 2.0 * np.sin(2 * np.pi * 60.0 * t)[:, None]
        return x

    def _pink_noise(self, n_samples: int) -> np.ndarray:
        """Unit-variance noise whose power falls off as 1/f, per channel."""
        spectrum = (self._rng.normal(size=(n_samples // 2 + 1, self.n_channels))
                    + 1j * self._rng.normal(size=(n_samples // 2 + 1, self.n_channels)))
        f = np.fft.rfftfreq(n_samples, d=1.0 / self.fs)
        f[0] = f[1]
        spectrum /= np.sqrt(f)[:, None]
        noise = np.fft.irfft(spectrum, n=n_samples, axis=0)
        noise -= noise.mean(axis=0)
        return noise / noise.std(axis=0)

    def latest(self, n_samples: int) -> np.ndarray:
        now = time.monotonic() - self._t0
        return self.window(n_samples, self.attended, t_start=now - n_samples / self.fs)
