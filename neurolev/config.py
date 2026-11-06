"""Central settings. Change values here rather than hunting through modules."""

from dataclasses import dataclass, field


@dataclass
class StimulusConfig:
    # Flicker rate of each LED, in Hz. Avoid ~10 Hz (alpha rhythm) and keep
    # every frequency (and its first few harmonics) clear of the others.
    frequencies: tuple = (7.0, 9.0, 11.0, 13.0, 15.0)
    # What each LED means, in the same order as `frequencies`.
    commands: tuple = ("LEFT", "RIGHT", "FORWARD", "BACK", "MERGE")


@dataclass
class DecoderConfig:
    window_s: float = 2.5          # length of EEG analyzed per decision
    hop_s: float = 0.5             # how often a new decision is attempted
    n_harmonics: int = 3           # reference harmonics used by CCA
    # Filter-bank sub-bands (low edge of each band, Hz). The first band must
    # sit below the lowest stimulus frequency.
    subband_low_edges: tuple = (6.0, 14.0, 22.0, 30.0, 38.0)
    subband_high_edge: float = 80.0
    # Score weights w(m) = m^-a + b  (Chen et al., 2015)
    weight_a: float = 1.25
    weight_b: float = 0.25
    # Act only when the best score beats the runner-up by this ratio.
    min_confidence_ratio: float = 1.15
    # Act only after this many consecutive windows agree.
    required_agreement: int = 2


@dataclass
class ArtifactConfig:
    max_peak_to_peak_uv: float = 150.0   # blinks / movement
    max_emg_power_ratio: float = 0.45    # share of power above 30 Hz (jaw clench)


@dataclass
class ArrayConfig:
    frequency_hz: float = 40_000.0
    speed_of_sound: float = 343.0        # m/s at ~20 C
    grid: int = 8                        # 8 x 8 transducers per array
    pitch_m: float = 0.0105              # centre-to-centre spacing
    separation_m: float = 0.10           # distance between the two arrays
    phase_steps: int = 10                # Ultraino driver: pi/5 resolution


@dataclass
class ControlConfig:
    step_m: float = 0.002                # how far one command moves a droplet
    merge_distance_m: float = 0.004      # closer than this counts as merged
    # Allowed travel around the array centre (x, y, z) in metres.
    bounds_m: tuple = ((-0.02, 0.02), (-0.02, 0.02), (0.04, 0.06))


@dataclass
class Config:
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    array: ArrayConfig = field(default_factory=ArrayConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
