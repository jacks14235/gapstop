#!/usr/bin/env python3
"""Batch runner for multiple gapstop template-matching jobs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run `gapstop run_tm` for every STAR file that matches a pattern "
            "and record how long each run takes."
        )
    )
    parser.add_argument(
        "--pattern",
        default="data_input_fixed_*.star",
        help="Glob pattern (relative to --workdir) for STAR files to process.",
    )
    parser.add_argument(
        "--workdir",
        default=".",
        help="Directory to run the commands from (default: current directory).",
    )
    parser.add_argument(
        "--gapstop-binary",
        default="gapstop",
        help="Executable to invoke for gapstop (default: gapstop).",
    )
    parser.add_argument(
        "--n-tiles",
        type=int,
        default=8,
        help="Value to pass to `-n` when calling gapstop (default: 8).",
    )
    parser.add_argument(
        "--output",
        default="tm_run_times.txt",
        help="File to append timing information to (default: tm_run_times.txt).",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Abort remaining runs if a job exits with a non-zero code.",
    )
    return parser.parse_args()


def format_elapsed(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    if not workdir.is_dir():
        print(f"[error] Workdir {workdir} does not exist.", file=sys.stderr)
        return 2

    star_paths = sorted(workdir.glob(args.pattern))
    if not star_paths:
        print(
            f"[error] No STAR files matched pattern '{args.pattern}' in {workdir}",
            file=sys.stderr,
        )
        return 1

    output_path = (workdir / args.output).resolve()
    output_lines = []

    for star_path in star_paths:
        rel_star = star_path.relative_to(workdir)
        cmd = [
            args.gapstop_binary,
            "run_tm",
            f"./{rel_star}",
            "-n",
            str(args.n_tiles),
        ]
        print(f"[info] Starting: {' '.join(cmd)}", flush=True)
        start = time.perf_counter()
        completed = subprocess.run(cmd, cwd=workdir)
        elapsed = time.perf_counter() - start
        status = "OK" if completed.returncode == 0 else f"FAIL({completed.returncode})"
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = (
            f"{timestamp}\t{rel_star}\t{format_elapsed(elapsed)}\t{status}"
        )
        output_lines.append(line)
        print(f"[info] Finished {rel_star} in {format_elapsed(elapsed)} -> {status}")
        if completed.returncode != 0 and args.stop_on_failure:
            print("[warn] Stopping early due to failure.")
            break

    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(output_lines))
        handle.write("\n")

    print(f"[info] Wrote timings for {len(output_lines)} run(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

