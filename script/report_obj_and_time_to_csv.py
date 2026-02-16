from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

"""
Scan all subdirectories in a root directory, detect instances named:
    instance_name.mps_data_time

Then parse each directory's tuner.log to extract:
    - objective
    - total tuning time (seconds)

Outputs CSV to stdout: instance,objective,tuning_time,dir
"""

INSTANCE_RE = re.compile(r"^(?P<name>.+)\.mps_(?P<time>.+)$", re.IGNORECASE)

# Matches the ending lines of the tuner.log, e.g.:
# Objective: ...
# Total tuning time: ... seconds.
OBJ_RE = re.compile(
    r"^\s*Objective\s*:\s*([\-+]?\d+(?:\.\d+)?(?:[eE][\-+]?\d+)?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
TIME_RE = re.compile(
    r"^\s*Total\s+tuning\s+time\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds?\s*\.\s*$",
    re.IGNORECASE | re.MULTILINE,
)


# --- update regex helper to return the base instance name ---
def parse_instance_dirname(dirname: str) -> Optional[str]:
    m = INSTANCE_RE.match(dirname)
    if not m:
        return None
    return m.group("name")


def parse_tuner_log(log_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not log_path.is_file():
        return None, None

    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return None, None

    obj_m = None
    time_m = None

    # Take the last occurrence if repeated.
    for m in OBJ_RE.finditer(text):
        obj_m = m
    for m in TIME_RE.finditer(text):
        time_m = m

    objective = float(obj_m.group(1)) if obj_m else None
    tuning_time = float(time_m.group(1)) if time_m else None
    return objective, tuning_time


def iter_instance_dirs(root: Path):
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        inst = parse_instance_dirname(entry.name)
        if inst is None:
            continue
        yield inst, entry


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process instance directories and summarize results.")
    p.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Directory containing instance subdirectories (default: current directory)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output CSV file path (use '-' for stdout)",
    )
    p.add_argument(
        "--no-dir",
        action="store_true",
        help="Do not include the 'dir' column in the CSV output.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root).expanduser().resolve()

    out_path = args.output
    if out_path == "-":
        out_fh = sys.stdout
        close_out = False
    else:
        out_fh = open(out_path, "w", newline="", encoding="utf-8")
        close_out = True

    try:
        writer = csv.writer(out_fh)
        if args.no_dir:
            writer.writerow(["instance", "objective", "tuning_time"])
        else:
            writer.writerow(["instance", "objective", "tuning_time", "dir"])

        for inst, dpath in iter_instance_dirs(root):
            obj, ttime = parse_tuner_log(dpath / "tuner.log")

            row = [
                inst,  # now this is just e.g. "app1-2", "brasil"
                "failed" if obj is None else obj,
                "failed" if ttime is None else ttime,
            ]
            if not args.no_dir:
                row.append(os.fspath(dpath))

            writer.writerow(row)
    finally:
        if close_out:
            out_fh.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
