from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


SOLVER_TIME_TO_SEED_START = {
    5: 0,
    15: 10,
}


@dataclass
class BatchRunResult:
    instance_name: str
    instance_path: Path
    solver_time: int
    seed_start: int
    seed_count: int
    status: str
    return_code: int
    output_dir: Path
    summary_csv: Path
    iterated_local_search_avg_gap: Optional[float]
    paramils_avg_gap: Optional[float]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run compare_local_search_backends.py across all instances from "
            "script/run_all.sh and summarize average gap results."
        )
    )
    parser.add_argument(
        "--mpi-procs",
        type=int,
        required=True,
        help="Number of MPI processes to pass to each comparison run.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Parent directory where the batch run directory will be created.",
    )
    parser.add_argument(
        "--number-of-evaluations",
        type=int,
        required=True,
        help="Number of evaluations to enforce for each local-search run.",
    )
    parser.add_argument(
        "--instance-dir",
        help="Directory containing the .mps instances. Defaults to INSTANCE_DIR from run_all.sh.",
    )
    parser.add_argument(
        "--run-all-script",
        default="script/run_all.sh",
        help="Path to run_all.sh used to discover INSTANCE_LIST and INSTANCE_DIR.",
    )
    parser.add_argument(
        "--tuner-bin",
        default="./build/mpils",
        help="Path to the tuner executable (default: ./build/mpils).",
    )
    parser.add_argument(
        "--compare-script",
        default="script/compare_local_search_backends.py",
        help="Path to the comparison script to delegate to.",
    )
    parser.add_argument(
        "--solver-threads",
        type=int,
        default=2,
        help="Number of solver threads to pass to each delegated run (default: 2).",
    )
    parser.add_argument(
        "--tuning-objective",
        default="gap",
        help="Tuning objective to pass to each delegated run (default: gap).",
    )
    args = parser.parse_args(argv)

    if args.mpi_procs <= 0:
        parser.error("--mpi-procs must be greater than 0")
    if args.number_of_evaluations <= 0:
        parser.error("--number-of-evaluations must be greater than 0")
    if args.solver_threads <= 0:
        parser.error("--solver-threads must be greater than 0")

    return args


def parse_run_all_script(run_all_script: Path) -> tuple[Path, list[str]]:
    try:
        text = run_all_script.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Unable to read run_all.sh: {run_all_script}") from exc

    instance_dir_match = re.search(r'^INSTANCE_DIR="([^"]+)"', text, re.MULTILINE)
    if instance_dir_match is None:
        raise RuntimeError(f"Could not find INSTANCE_DIR in {run_all_script}")

    list_match = re.search(r"INSTANCE_LIST=\((.*?)\)", text, re.DOTALL)
    if list_match is None:
        raise RuntimeError(f"Could not find INSTANCE_LIST in {run_all_script}")

    instances: list[str] = []
    for raw_line in list_match.group(1).splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        quoted_values = re.findall(r'"([^"]+)"', line)
        instances.extend(quoted_values)

    if not instances:
        raise RuntimeError(f"INSTANCE_LIST is empty in {run_all_script}")

    return Path(instance_dir_match.group(1)).expanduser(), instances


def resolve_batch_output_dir(output_root: Path, mpi_procs: int, number_of_evaluations: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_root / f"batch_{timestamp}_mpi{mpi_procs}_eval{number_of_evaluations}"


def validate_runtime_inputs(
    run_all_script: Path,
    compare_script: Path,
    tuner_bin: Path,
    instance_dir: Path,
    mpi_procs: int,
) -> None:
    if not run_all_script.is_file():
        raise FileNotFoundError(f"run_all.sh not found: {run_all_script}")
    if not compare_script.is_file():
        raise FileNotFoundError(f"Comparison script not found: {compare_script}")
    if not tuner_bin.is_file():
        raise FileNotFoundError(f"Tuner binary not found: {tuner_bin}")
    if not tuner_bin.stat().st_mode & 0o111:
        raise PermissionError(f"Tuner binary is not executable: {tuner_bin}")
    if not instance_dir.is_dir():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")
    if mpi_procs > 1 and not shutil_which("mpirun"):
        raise FileNotFoundError("mpirun not found in PATH while --mpi-procs is greater than 1")


def shutil_which(program: str) -> Optional[str]:
    from shutil import which

    return which(program)


def parse_backend_objective_means(summary_csv: Path) -> tuple[Optional[float], Optional[float]]:
    if not summary_csv.is_file():
        return None, None

    ils_mean = None
    paramils_mean = None

    with summary_csv.open("r", newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("row_type") != "backend":
                continue
            name = row.get("name")
            objective_mean = row.get("objective_mean", "").strip()
            if not objective_mean:
                continue
            if name == "iterated_local_search":
                ils_mean = float(objective_mean)
            elif name == "paramils":
                paramils_mean = float(objective_mean)

    return ils_mean, paramils_mean


def optional_float(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.12g}"


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    data = list(values)
    if not data:
        return None
    return statistics.fmean(data)


def build_compare_command(
    compare_script: Path,
    tuner_bin: Path,
    instance_path: Path,
    output_dir: Path,
    solver_time: int,
    seed_start: int,
    mpi_procs: int,
    solver_threads: int,
    tuning_objective: str,
    number_of_evaluations: int,
) -> list[str]:
    return [
        "python3",
        str(compare_script),
        str(instance_path),
        "--tuner-bin",
        str(tuner_bin),
        "--output-root",
        str(output_dir),
        "--seed-start",
        str(seed_start),
        "--seed-count",
        "10",
        "--solver-time",
        str(solver_time),
        "--solver-threads",
        str(solver_threads),
        "--tuning-objective",
        tuning_objective,
        "--mpi-procs",
        str(mpi_procs),
        "--number-of-evaluations",
        str(number_of_evaluations),
    ]


def run_batch_entry(
    compare_script: Path,
    tuner_bin: Path,
    instance_path: Path,
    output_dir: Path,
    solver_time: int,
    seed_start: int,
    mpi_procs: int,
    solver_threads: int,
    tuning_objective: str,
    number_of_evaluations: int,
) -> BatchRunResult:
    output_dir.mkdir(parents=True, exist_ok=False)
    launch_log = output_dir / "batch_launch.log"
    summary_csv = output_dir / "summary.csv"
    command = build_compare_command(
        compare_script=compare_script,
        tuner_bin=tuner_bin,
        instance_path=instance_path,
        output_dir=output_dir,
        solver_time=solver_time,
        seed_start=seed_start,
        mpi_procs=mpi_procs,
        solver_threads=solver_threads,
        tuning_objective=tuning_objective,
        number_of_evaluations=number_of_evaluations,
    )

    with launch_log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    ils_mean, paramils_mean = parse_backend_objective_means(summary_csv)
    status = (
        "ok"
        if completed.returncode == 0 and ils_mean is not None and paramils_mean is not None
        else "failed"
    )

    return BatchRunResult(
        instance_name=instance_path.stem,
        instance_path=instance_path,
        solver_time=solver_time,
        seed_start=seed_start,
        seed_count=10,
        status=status,
        return_code=completed.returncode,
        output_dir=output_dir,
        summary_csv=summary_csv,
        iterated_local_search_avg_gap=ils_mean,
        paramils_avg_gap=paramils_mean,
    )


def write_batch_runs_csv(batch_dir: Path, results: list[BatchRunResult]) -> None:
    path = batch_dir / "batch_runs.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "instance",
                "instance_path",
                "solver_time",
                "seed_start",
                "seed_count",
                "status",
                "return_code",
                "iterated_local_search_avg_gap",
                "paramils_avg_gap",
                "gap_delta_paramils_minus_ils",
                "output_dir",
                "summary_csv",
            ]
        )
        for result in results:
            gap_delta = None
            if (
                result.iterated_local_search_avg_gap is not None
                and result.paramils_avg_gap is not None
            ):
                gap_delta = (
                    result.paramils_avg_gap - result.iterated_local_search_avg_gap
                )
            writer.writerow(
                [
                    result.instance_name,
                    str(result.instance_path),
                    result.solver_time,
                    result.seed_start,
                    result.seed_count,
                    result.status,
                    result.return_code,
                    optional_float(result.iterated_local_search_avg_gap),
                    optional_float(result.paramils_avg_gap),
                    optional_float(gap_delta),
                    str(result.output_dir),
                    str(result.summary_csv),
                ]
            )


def write_per_instance_solver_time_summary(batch_dir: Path, results: list[BatchRunResult]) -> None:
    path = batch_dir / "per_instance_solver_time_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "instance",
                "solver_time",
                "iterated_local_search_avg_gap",
                "paramils_avg_gap",
                "gap_delta_paramils_minus_ils",
                "status",
                "output_dir",
            ]
        )
        for result in results:
            gap_delta = None
            if (
                result.iterated_local_search_avg_gap is not None
                and result.paramils_avg_gap is not None
            ):
                gap_delta = (
                    result.paramils_avg_gap - result.iterated_local_search_avg_gap
                )
            writer.writerow(
                [
                    result.instance_name,
                    result.solver_time,
                    optional_float(result.iterated_local_search_avg_gap),
                    optional_float(result.paramils_avg_gap),
                    optional_float(gap_delta),
                    result.status,
                    str(result.output_dir),
                ]
            )


def write_per_instance_overall_summary(batch_dir: Path, results: list[BatchRunResult]) -> None:
    path = batch_dir / "per_instance_overall_summary.csv"
    grouped: dict[str, list[BatchRunResult]] = {}
    for result in results:
        grouped.setdefault(result.instance_name, []).append(result)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "instance",
                "successful_solver_time_count",
                "iterated_local_search_avg_gap",
                "paramils_avg_gap",
                "gap_delta_paramils_minus_ils",
            ]
        )
        for instance_name in sorted(grouped):
            group = grouped[instance_name]
            ils_values = [
                result.iterated_local_search_avg_gap
                for result in group
                if result.status == "ok" and result.iterated_local_search_avg_gap is not None
            ]
            paramils_values = [
                result.paramils_avg_gap
                for result in group
                if result.status == "ok" and result.paramils_avg_gap is not None
            ]
            ils_mean = mean_or_none(ils_values)
            paramils_mean = mean_or_none(paramils_values)
            gap_delta = None
            if ils_mean is not None and paramils_mean is not None:
                gap_delta = paramils_mean - ils_mean
            writer.writerow(
                [
                    instance_name,
                    sum(1 for result in group if result.status == "ok"),
                    optional_float(ils_mean),
                    optional_float(paramils_mean),
                    optional_float(gap_delta),
                ]
            )


def write_overall_summary(batch_dir: Path, results: list[BatchRunResult]) -> None:
    path = batch_dir / "overall_summary.csv"
    ils_values = [
        result.iterated_local_search_avg_gap
        for result in results
        if result.status == "ok" and result.iterated_local_search_avg_gap is not None
    ]
    paramils_values = [
        result.paramils_avg_gap
        for result in results
        if result.status == "ok" and result.paramils_avg_gap is not None
    ]
    ils_mean = mean_or_none(ils_values)
    paramils_mean = mean_or_none(paramils_values)
    gap_delta = None
    winner = ""
    if ils_mean is not None and paramils_mean is not None:
        gap_delta = paramils_mean - ils_mean
        if paramils_mean < ils_mean:
            winner = "paramils"
        elif ils_mean < paramils_mean:
            winner = "iterated_local_search"
        else:
            winner = "tie"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "successful_run_count",
                "iterated_local_search_avg_gap",
                "paramils_avg_gap",
                "gap_delta_paramils_minus_ils",
                "winner_lower_gap",
            ]
        )
        writer.writerow(
            [
                sum(1 for result in results if result.status == "ok"),
                optional_float(ils_mean),
                optional_float(paramils_mean),
                optional_float(gap_delta),
                winner,
            ]
        )


def print_terminal_recap(batch_dir: Path, results: list[BatchRunResult]) -> None:
    grouped: dict[str, list[BatchRunResult]] = {}
    for result in results:
        grouped.setdefault(result.instance_name, []).append(result)

    print(f"Batch output directory: {batch_dir}")
    print("Per-instance average gaps:")
    for instance_name in sorted(grouped):
        group = grouped[instance_name]
        ils_values = [
            result.iterated_local_search_avg_gap
            for result in group
            if result.status == "ok" and result.iterated_local_search_avg_gap is not None
        ]
        paramils_values = [
            result.paramils_avg_gap
            for result in group
            if result.status == "ok" and result.paramils_avg_gap is not None
        ]
        ils_mean = mean_or_none(ils_values)
        paramils_mean = mean_or_none(paramils_values)
        print(
            f"  {instance_name}: "
            f"iterated_local_search={optional_float(ils_mean) or 'NA'}, "
            f"paramils={optional_float(paramils_mean) or 'NA'}"
        )

    ils_values = [
        result.iterated_local_search_avg_gap
        for result in results
        if result.status == "ok" and result.iterated_local_search_avg_gap is not None
    ]
    paramils_values = [
        result.paramils_avg_gap
        for result in results
        if result.status == "ok" and result.paramils_avg_gap is not None
    ]
    ils_mean = mean_or_none(ils_values)
    paramils_mean = mean_or_none(paramils_values)
    print("Overall average gaps:")
    print(
        "  "
        f"iterated_local_search={optional_float(ils_mean) or 'NA'}, "
        f"paramils={optional_float(paramils_mean) or 'NA'}"
    )
    if ils_mean is not None and paramils_mean is not None:
        if paramils_mean < ils_mean:
            winner = "paramils"
        elif ils_mean < paramils_mean:
            winner = "iterated_local_search"
        else:
            winner = "tie"
        print(f"Overall winner by lower average gap: {winner}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    run_all_script = Path(args.run_all_script).expanduser().resolve()
    compare_script = Path(args.compare_script).expanduser().resolve()
    tuner_bin = Path(args.tuner_bin).expanduser().resolve()
    configured_instance_dir, instances = parse_run_all_script(run_all_script)
    instance_dir = (
        Path(args.instance_dir).expanduser().resolve()
        if args.instance_dir
        else configured_instance_dir.resolve()
    )
    output_root = Path(args.output_root).expanduser().resolve()

    validate_runtime_inputs(
        run_all_script=run_all_script,
        compare_script=compare_script,
        tuner_bin=tuner_bin,
        instance_dir=instance_dir,
        mpi_procs=args.mpi_procs,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    batch_dir = resolve_batch_output_dir(
        output_root=output_root,
        mpi_procs=args.mpi_procs,
        number_of_evaluations=args.number_of_evaluations,
    )
    batch_dir.mkdir(parents=True, exist_ok=False)

    results: list[BatchRunResult] = []
    had_failures = False

    for instance in instances:
        instance_path = (instance_dir / instance).resolve()
        for solver_time, seed_start in SOLVER_TIME_TO_SEED_START.items():
            run_name = f"{instance_path.stem}_time{solver_time}s"
            output_dir = batch_dir / run_name
            if not instance_path.is_file():
                print(f"Skipping missing instance file: {instance_path}")
                results.append(
                    BatchRunResult(
                        instance_name=instance_path.stem,
                        instance_path=instance_path,
                        solver_time=solver_time,
                        seed_start=seed_start,
                        seed_count=10,
                        status="missing_instance",
                        return_code=1,
                        output_dir=output_dir,
                        summary_csv=output_dir / "summary.csv",
                        iterated_local_search_avg_gap=None,
                        paramils_avg_gap=None,
                    )
                )
                had_failures = True
                continue
            print(
                f"Running {instance_path.name} with solver_time={solver_time}s, "
                f"seeds={seed_start}-{seed_start + 9}, mpi_procs={args.mpi_procs}"
            )
            result = run_batch_entry(
                compare_script=compare_script,
                tuner_bin=tuner_bin,
                instance_path=instance_path,
                output_dir=output_dir,
                solver_time=solver_time,
                seed_start=seed_start,
                mpi_procs=args.mpi_procs,
                solver_threads=args.solver_threads,
                tuning_objective=args.tuning_objective,
                number_of_evaluations=args.number_of_evaluations,
            )
            results.append(result)
            if result.status != "ok":
                had_failures = True

    write_batch_runs_csv(batch_dir, results)
    write_per_instance_solver_time_summary(batch_dir, results)
    write_per_instance_overall_summary(batch_dir, results)
    write_overall_summary(batch_dir, results)
    print_terminal_recap(batch_dir, results)

    return 1 if had_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
