"""Cued accuracy test: measures real decoder accuracy and ITR on a real person.

The script tells the user which LED to look at, records the EEG, decodes it,
and at the end reports accuracy, information transfer rate, and per-target
results. Raw EEG is saved so you can re-analyze later with different settings.

These are the numbers that belong in a write-up or resume. Run it on several
people (each with their own --subject name).

  python scripts/run_eval_session.py --source ganglion --serial-port COM3 --subject alex
  python scripts/run_eval_session.py --source simulated --subject test   # dry rehearsal
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurolev.artifacts import check_window  # noqa: E402
from neurolev.config import Config  # noqa: E402
from neurolev.eeg import BrainFlowSource, SimulatedSource  # noqa: E402
from neurolev.metrics import information_transfer_rate  # noqa: E402
from neurolev.ssvep import FBCCADecoder  # noqa: E402

GAZE_SHIFT_S = 1.0   # rest + time to move eyes to the next target


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="simulated",
                   choices=["simulated", "synthetic", "ganglion", "ganglion-native", "cyton"])
    p.add_argument("--serial-port", default="")
    p.add_argument("--channels", default=None)
    p.add_argument("--subject", required=True)
    p.add_argument("--trials-per-target", type=int, default=8)
    args = p.parse_args()

    cfg = Config()
    freqs, names = cfg.stimulus.frequencies, cfg.stimulus.commands
    channels = [int(c) for c in args.channels.split(",")] if args.channels else None
    source = SimulatedSource(seed=int(time.time())) if args.source == "simulated" else \
        BrainFlowSource(args.source, serial_port=args.serial_port, channels=channels)
    decoder = FBCCADecoder(freqs, source.fs, cfg.decoder)
    n_window = int(cfg.decoder.window_s * source.fs)

    trials = [i for i in range(len(freqs)) for _ in range(args.trials_per_target)]
    random.shuffle(trials)

    out_dir = Path("data") / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    print("SAFETY: flickering lights can trigger seizures in people with photosensitive "
          "epilepsy. Do not run this on anyone with a seizure history.\n")
    input(f"{len(trials)} trials. Press Enter when {args.subject} is ready...")

    source.start()
    records, raw = [], []
    try:
        for n, target in enumerate(trials, 1):
            print(f"[{n}/{len(trials)}] Look at the {names[target]} LED ({freqs[target]} Hz)")
            if isinstance(source, SimulatedSource):
                source.attended = freqs[target]
            time.sleep(GAZE_SHIFT_S + cfg.decoder.window_s)
            eeg = source.latest(n_window)
            report = check_window(eeg, source.fs, cfg.artifacts)
            d = decoder.decide(eeg)
            correct = d.index == target
            print(f"    decoded {names[d.index]}  {'OK' if correct else 'miss'}"
                  f"{'' if report.ok else '  (artifact: ' + report.reason + ')'}")
            records.append({"target": target, "decoded": d.index, "correct": bool(correct),
                            "confidence": d.confidence_ratio, "artifact_ok": report.ok,
                            "scores": d.scores.tolist()})
            raw.append(eeg)
    finally:
        source.stop()

    clean = [r for r in records if r["artifact_ok"]]
    acc_all = np.mean([r["correct"] for r in records])
    acc_clean = np.mean([r["correct"] for r in clean]) if clean else float("nan")
    seconds = cfg.decoder.window_s + GAZE_SHIFT_S
    itr = information_transfer_rate(len(freqs), acc_all, seconds)

    print("\n=== Results ===")
    print(f"Accuracy (all trials):      {acc_all:.1%}  ({sum(r['correct'] for r in records)}/{len(records)})")
    print(f"Accuracy (artifact-free):   {acc_clean:.1%}  ({len(clean)} trials)")
    print(f"ITR: {itr:.1f} bits/min at {seconds:.1f} s per selection")
    for i, name in enumerate(names):
        hits = [r["correct"] for r in records if r["target"] == i]
        print(f"  {name:8s} {np.mean(hits):.0%}")

    summary = {"subject": args.subject, "source": args.source, "accuracy": acc_all,
               "accuracy_artifact_free": acc_clean, "itr_bits_per_min": itr,
               "seconds_per_selection": seconds, "frequencies": list(freqs),
               "trials": records}
    (out_dir / f"{stamp}_results.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out_dir / f"{stamp}_raw.npz", eeg=np.stack(raw), fs=source.fs,
                        targets=np.array(trials))
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
