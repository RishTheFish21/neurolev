"""Live demo: EEG headset -> SSVEP decoder -> levitated droplets.

Examples
  # No hardware at all (simulated brain, printed phases):
  python scripts/run_demo.py --source simulated --dry-run

  # Real Ganglion headset, levitator not connected yet:
  python scripts/run_demo.py --source ganglion --serial-port /dev/ttyACM0 --dry-run

  # Everything connected:
  python scripts/run_demo.py --source ganglion --serial-port COM3 --levitator-port COM4
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurolev.artifacts import check_window  # noqa: E402
from neurolev.config import Config  # noqa: E402
from neurolev.control import DropletController  # noqa: E402
from neurolev.eeg import BrainFlowSource, SimulatedSource  # noqa: E402
from neurolev.levitator import DryRunTransport, Levitator, SerialTransport  # noqa: E402
from neurolev.ssvep import AgreementGate, FBCCADecoder  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="simulated",
                   choices=["simulated", "synthetic", "ganglion", "ganglion-native", "cyton"])
    p.add_argument("--serial-port", default="", help="EEG dongle port, e.g. COM3 or /dev/ttyACM0")
    p.add_argument("--channels", default=None,
                   help="Comma-separated EEG channel indices to use, e.g. 0,1,2")
    p.add_argument("--levitator-port", default="", help="Serial port of the array driver")
    p.add_argument("--dry-run", action="store_true", help="Print phases instead of sending them")
    p.add_argument("--log", default="data/session_log.csv", help="Where to log every decision")
    p.add_argument("--duration", type=float, default=120.0, help="Seconds to run")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    channels = [int(c) for c in args.channels.split(",")] if args.channels else None
    if args.source == "simulated":
        source = SimulatedSource()
        source.attended = cfg.stimulus.frequencies[0]
        print("Simulated brain is 'looking at' the first LED the whole time.")
    else:
        source = BrainFlowSource(args.source, serial_port=args.serial_port, channels=channels)

    transport = DryRunTransport(verbose=False) if args.dry_run or not args.levitator_port \
        else SerialTransport(args.levitator_port)
    levitator = Levitator(transport, cfg.array)
    controller = DropletController(cfg.control)
    decoder = FBCCADecoder(cfg.stimulus.frequencies, source.fs, cfg.decoder)
    gate = AgreementGate(cfg.decoder.required_agreement)
    n_window = int(cfg.decoder.window_s * source.fs)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    source.start()
    levitator.set_traps(controller.state.targets())

    t_start = time.monotonic()
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "artifact_ok", "artifact_reason", "decoded", "confidence",
                         "command", "ax", "ay", "az", "merged"])
        try:
            while time.monotonic() - t_start < args.duration:
                loop_start = time.monotonic()
                eeg = source.latest(n_window)
                command, decoded, conf = "", "", 0.0
                report = check_window(eeg, source.fs, cfg.artifacts)

                if report.ok:
                    decision = decoder.decide(eeg)
                    decoded, conf = cfg.stimulus.commands[decision.index], decision.confidence_ratio
                    fired = gate.update(decision)
                    if fired is not None:
                        command = cfg.stimulus.commands[fired]
                        state = controller.apply(command)
                        t0 = time.perf_counter()
                        levitator.set_traps(state.targets())
                        ms = (time.perf_counter() - t0) * 1000
                        print(f"{command:8s} -> A at {state.a.round(4)} "
                              f"(phase update {ms:.1f} ms){'  MERGED!' if state.merged else ''}")
                else:
                    gate.reset()
                    print(f"skipped window: {report.reason}")

                a = controller.state.a
                writer.writerow([round(time.monotonic() - t_start, 3), report.ok, report.reason,
                                 decoded, round(conf, 3), command, *a.round(5), controller.state.merged])
                if controller.state.merged:
                    print("Droplets merged. Ending demo.")
                    break
                time.sleep(max(0.0, cfg.decoder.hop_s - (time.monotonic() - loop_start)))
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            source.stop()
            transport.close()
    print(f"Decision log saved to {log_path}")


if __name__ == "__main__":
    main()
