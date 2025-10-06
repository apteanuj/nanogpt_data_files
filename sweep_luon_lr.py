#!/usr/bin/env python3
import argparse
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

VAL_RE = re.compile(r"val_loss:(?P<val>[-+]?\d+(\.\d+)?([eE][-+]?\d+)?)")

def logspace(start: float, stop: float, num: int):
    # inclusive logspace like numpy
    if num < 2:
        return [start]
    log_start, log_stop = math.log10(start), math.log10(stop)
    step = (log_stop - log_start) / (num - 1)
    return [10 ** (log_start + i * step) for i in range(num)]

def main():
    ap = argparse.ArgumentParser(description="Sweep Luon LR with torchrun")
    ap.add_argument("--script", type=str, required=True,
                    help="Path to your training .py (the big DDP script you pasted).")
    ap.add_argument("--nproc_per_node", type=int, required=True,
                    help="Number of GPUs to use per run (passed to torchrun).")
    ap.add_argument("--min_lr", type=float, default=0.0002, help="Min LR (inclusive).")
    ap.add_argument("--max_lr", type=float, default=0.001, help="Max LR (inclusive).")
    ap.add_argument("--points", type=int, default=5, help="Number of LR points (log-spaced).")
    ap.add_argument("--outfile", type=str, default=None,
                    help="CSV path for results (default: logs/luon_lr_sweep_<timestamp>.csv).")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Extra args to pass after the script (if your script accepts any).")
    args = ap.parse_args()

    train_script = Path(args.script).resolve()
    if not train_script.exists():
        print(f"ERROR: script not found: {train_script}", file=sys.stderr)
        sys.exit(1)

    lrs = np.linspace(args.max_lr, args.min_lr, args.points)  # from 0.01 down to 0.001
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    out_csv = Path(args.outfile) if args.outfile else logs_dir / f"luon_lr_sweep_{timestamp}.csv"

    print(f"Will sweep {len(lrs)} LR values (log-spaced) from {args.max_lr} → {args.min_lr}")
    print(f"Summary CSV: {out_csv}")

    with open(out_csv, "w") as fcsv:
        fcsv.write("lr,best_val,last_val,log_path,return_code\n")

        for lr in lrs:
            lr_str = f"{lr:.6g}"
            print(f"\n=== LR {lr_str} ===")
            # Per-run stdout log
            run_log = logs_dir / f"sweep_luon_lr_{timestamp}_lr{lr_str.replace('.','p')}.out"

            env = os.environ.copy()
            env["LUON_LR"] = lr_str
            # (Optional) isolate Inductor cache per-run if you see cross-run pollution
            # env["TORCHINDUCTOR_CACHE_DIR"] = str((logs_dir / f"inductor_cache_lr{lr_str.replace('.','p')}").resolve())

            cmd = [
                sys.executable,  # use current python
                "-m", "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.nproc_per_node}",
                str(train_script),
            ] + args.extra

            print("CMD:", " ".join(cmd))
            print(f"LOG → {run_log}")

            with open(run_log, "w") as fout:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
                best_val = None
                last_val = None
                # stream & tee
                for line in proc.stdout:
                    sys.stdout.write(line)
                    fout.write(line)
                    m = VAL_RE.search(line)
                    if m:
                        v = float(m.group("val"))
                        last_val = v
                        best_val = v if (best_val is None or v < best_val) else best_val
                proc.wait()
                rc = proc.returncode

            fcsv.write(f"{lr_str},{'' if best_val is None else best_val},"
                       f"{'' if last_val is None else last_val},"
                       f"{run_log},{rc}\n")
            fcsv.flush()

            if rc != 0:
                print(f"WARNING: run with LR={lr_str} exited code {rc}")

    print("\nSweep complete.")
    print(f"Results saved to: {out_csv}")

if __name__ == "__main__":
    main()
