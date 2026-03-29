from __future__ import annotations

import argparse
import csv
import re
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


BACKENDS = ("iterated_local_search", "paramils")

OBJ_RE = re.compile(
    r"^\s*Objective\s*:\s*([\-+]?\d+(?:\.\d+)?(?:[eE][\-+]?\d+)?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
TIME_RE = re.compile(
    r"^\s*Total\s+tuning\s+time\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds?\s*\.\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class RunResult:
    seed: int
    backend: str
    status: str
    return_code: int
    objective: Optional[float]
    tuning_time_seconds: Optional[float]
    unique_config_count: Optional[int]
    evaluation_count: Optional[int]
    run_dir: Path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exploration-only tuner comparisons for iterated_local_search and "
            "paramils across multiple seeds and summarize the results."
        )
    )
    parser.add_argument("instance", help="Path to the instance file to tune.")
    parser.add_argument(
        "--tuner-bin",
        default="./build/mpils",
        help="Path to the tuner executable (default: ./build/mpils).",
    )
    parser.add_argument(
        "--output-root",
        help=(
            "Directory where run artifacts and CSV summaries are written "
            "(default: ./comparison_runs/<instance>_<timestamp>)."
        ),
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed to use (default: 0).",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=10,
        help="Number of consecutive seeds to run (default: 10).",
    )
    parser.add_argument(
        "--solver-time",
        type=float,
        default=5.0,
        help="Per-evaluation solver cutoff time in seconds (default: 5).",
    )
    parser.add_argument(
        "--solver-threads",
        type=int,
        default=2,
        help="Number of solver threads to pass to the tuner (default: 2).",
    )
    parser.add_argument(
        "--tuning-objective",
        default="gap",
        help="Tuning objective to pass to the tuner (default: gap).",
    )
    parser.add_argument(
        "--mpi-procs",
        type=int,
        default=1,
        help="Number of MPI processes to use. Use 1 for sequential runs (default: 1).",
    )
    parser.add_argument(
        "--number-of-evaluations",
        type=int,
        default=20,
        help=(
            "Number of evaluations to perform per run. It is passed to the tuner."
        ),
    )
    args = parser.parse_args(argv)

    if args.seed_count <= 0:
        parser.error("--seed-count must be greater than 0")
    if args.solver_threads <= 0:
        parser.error("--solver-threads must be greater than 0")
    if args.solver_time <= 0:
        parser.error("--solver-time must be greater than 0")
    if args.mpi_procs <= 0:
        parser.error("--mpi-procs must be greater than 0")
    if args.number_of_evaluations <= 0:
        parser.error("--number-of-evaluations must be greater than 0")

    return args


def resolve_output_root(instance_path: Path, output_root: Optional[str]) -> Path:
    if output_root:
        return Path(output_root).expanduser().resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path.cwd() / "comparison_runs" / f"{instance_path.stem}_{timestamp}").resolve()


def find_last_match(text: str, regex: re.Pattern[str]) -> Optional[float]:
    match = None
    for candidate in regex.finditer(text):
        match = candidate
    if match is None:
        return None
    return float(match.group(1))


def parse_tuner_log(log_path: Path) -> tuple[Optional[float], Optional[float]]:
    if not log_path.is_file():
        return None, None

    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None, None

    return find_last_match(text, OBJ_RE), find_last_match(text, TIME_RE)


def count_csv_data_rows(csv_path: Path) -> Optional[int]:
    if not csv_path.is_file():
        return None

    try:
        with csv_path.open("r", newline="", encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                return 0
            count = 0
            for row in reader:
                if any(cell.strip() for cell in row):
                    count += 1
            return count
    except OSError:
        return None


def build_command(
    tuner_bin: Path,
    instance_path: Path,
    backend: str,
    seed: int,
    work_dir: Path,
    solver_time: float,
    solver_threads: int,
    tuning_objective: str,
    mpi_procs: int,
) -> list[str]:
    cmd: list[str] = []
    if mpi_procs > 1:
        cmd.extend(["mpirun", "-np", str(mpi_procs)])

    cmd.extend(
        [
            str(tuner_bin),
            str(instance_path),
            "--exploration-only",
            "--local-search-engine",
            backend,
            "--seed",
            str(seed),
            "--working-dir",
            str(work_dir),
            "--solver-time",
            str(solver_time),
            "--solver-threads",
            str(solver_threads),
            "--tuning-objective",
            tuning_objective,
        ]
    )
    return cmd


def run_single(
    tuner_bin: Path,
    instance_path: Path,
    backend: str,
    seed: int,
    output_root: Path,
    solver_time: float,
    solver_threads: int,
    tuning_objective: str,
    mpi_procs: int,
) -> RunResult:
    run_dir = output_root / backend / f"seed_{seed}"
    work_dir = run_dir / "workdir"
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    launch_log = run_dir / "launch.log"
    command = build_command(
        tuner_bin=tuner_bin,
        instance_path=instance_path,
        backend=backend,
        seed=seed,
        work_dir=work_dir,
        solver_time=solver_time,
        solver_threads=solver_threads,
        tuning_objective=tuning_objective,
        mpi_procs=mpi_procs,
    )

    with launch_log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    tuner_log = work_dir / "tuner.log"
    unique_configs_csv = work_dir / "tuner_history_unique_configs.csv"
    evaluation_log_csv = work_dir / "tuner_history_evaluation_log.csv"

    objective, tuning_time_seconds = parse_tuner_log(tuner_log)
    unique_config_count = count_csv_data_rows(unique_configs_csv)
    evaluation_count = count_csv_data_rows(evaluation_log_csv)

    metrics_ok = all(
        value is not None
        for value in (
            objective,
            tuning_time_seconds,
            unique_config_count,
            evaluation_count,
        )
    )
    status = "ok" if completed.returncode == 0 and metrics_ok else "failed"

    return RunResult(
        seed=seed,
        backend=backend,
        status=status,
        return_code=completed.returncode,
        objective=objective,
        tuning_time_seconds=tuning_time_seconds,
        unique_config_count=unique_config_count,
        evaluation_count=evaluation_count,
        run_dir=run_dir,
    )


def optional_float(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.12g}"


def optional_int(value: Optional[int]) -> str:
    return "" if value is None else str(value)


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    data = list(values)
    if not data:
        return None
    return statistics.fmean(data)


def stdev_or_none(values: Iterable[float]) -> Optional[float]:
    data = list(values)
    if len(data) < 2:
        return None
    return statistics.stdev(data)


def min_or_none(values: Iterable[float]) -> Optional[float]:
    data = list(values)
    if not data:
        return None
    return min(data)


def max_or_none(values: Iterable[float]) -> Optional[float]:
    data = list(values)
    if not data:
        return None
    return max(data)


def write_runs_csv(output_root: Path, runs: list[RunResult]) -> None:
    path = output_root / "runs.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "backend",
                "status",
                "return_code",
                "objective",
                "tuning_time_seconds",
                "unique_config_count",
                "evaluation_count",
                "run_dir",
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run.seed,
                    run.backend,
                    run.status,
                    run.return_code,
                    optional_float(run.objective),
                    optional_float(run.tuning_time_seconds),
                    optional_int(run.unique_config_count),
                    optional_int(run.evaluation_count),
                    str(run.run_dir),
                ]
            )


def write_paired_csv(output_root: Path, runs: list[RunResult]) -> None:
    path = output_root / "paired_by_seed.csv"
    runs_by_seed_backend = {(run.seed, run.backend): run for run in runs}
    seeds = sorted({run.seed for run in runs})

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "ils_objective",
                "paramils_objective",
                "objective_delta",
                "ils_tuning_time_seconds",
                "paramils_tuning_time_seconds",
                "tuning_time_delta",
                "ils_unique_config_count",
                "paramils_unique_config_count",
                "unique_config_delta",
                "ils_evaluation_count",
                "paramils_evaluation_count",
                "evaluation_count_match",
            ]
        )

        for seed in seeds:
            ils = runs_by_seed_backend.get((seed, "iterated_local_search"))
            paramils = runs_by_seed_backend.get((seed, "paramils"))

            objective_delta = None
            if ils and paramils and ils.objective is not None and paramils.objective is not None:
                objective_delta = paramils.objective - ils.objective

            tuning_time_delta = None
            if (
                ils
                and paramils
                and ils.tuning_time_seconds is not None
                and paramils.tuning_time_seconds is not None
            ):
                tuning_time_delta = paramils.tuning_time_seconds - ils.tuning_time_seconds

            unique_config_delta = None
            if (
                ils
                and paramils
                and ils.unique_config_count is not None
                and paramils.unique_config_count is not None
            ):
                unique_config_delta = paramils.unique_config_count - ils.unique_config_count

            evaluation_count_match = ""
            if (
                ils
                and paramils
                and ils.evaluation_count is not None
                and paramils.evaluation_count is not None
            ):
                evaluation_count_match = (
                    "true" if ils.evaluation_count == paramils.evaluation_count else "false"
                )

            writer.writerow(
                [
                    seed,
                    optional_float(ils.objective if ils else None),
                    optional_float(paramils.objective if paramils else None),
                    optional_float(objective_delta),
                    optional_float(ils.tuning_time_seconds if ils else None),
                    optional_float(paramils.tuning_time_seconds if paramils else None),
                    optional_float(tuning_time_delta),
                    optional_int(ils.unique_config_count if ils else None),
                    optional_int(paramils.unique_config_count if paramils else None),
                    optional_int(unique_config_delta),
                    optional_int(ils.evaluation_count if ils else None),
                    optional_int(paramils.evaluation_count if paramils else None),
                    evaluation_count_match,
                ]
            )


def count_relation(values_a: list[float], values_b: list[float], prefer_lower: bool) -> tuple[int, int, int]:
    first_wins = 0
    second_wins = 0
    ties = 0

    for left, right in zip(values_a, values_b):
        if left == right:
            ties += 1
        elif prefer_lower:
            if left < right:
                first_wins += 1
            else:
                second_wins += 1
        else:
            if left > right:
                first_wins += 1
            else:
                second_wins += 1

    return first_wins, second_wins, ties


def write_summary_csv(output_root: Path, runs: list[RunResult]) -> None:
    path = output_root / "summary.csv"

    successful_runs_by_backend = {
        backend: [run for run in runs if run.backend == backend and run.status == "ok"]
        for backend in BACKENDS
    }

    paired_successful_runs: list[tuple[RunResult, RunResult]] = []
    for seed in sorted({run.seed for run in runs}):
        ils = next(
            (run for run in runs if run.seed == seed and run.backend == "iterated_local_search"),
            None,
        )
        paramils = next(
            (run for run in runs if run.seed == seed and run.backend == "paramils"),
            None,
        )
        if ils is None or paramils is None:
            continue
        if ils.status != "ok" or paramils.status != "ok":
            continue
        paired_successful_runs.append((ils, paramils))

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row_type",
                "name",
                "run_count",
                "success_count",
                "paired_seed_count",
                "objective_mean",
                "objective_std",
                "objective_min",
                "objective_max",
                "tuning_time_mean",
                "tuning_time_std",
                "tuning_time_min",
                "tuning_time_max",
                "unique_config_mean",
                "unique_config_std",
                "unique_config_min",
                "unique_config_max",
                "evaluation_count_mean",
                "evaluation_count_std",
                "objective_delta_mean",
                "objective_delta_std",
                "tuning_time_delta_mean",
                "tuning_time_delta_std",
                "unique_config_delta_mean",
                "unique_config_delta_std",
                "paramils_lower_objective_wins",
                "ils_lower_objective_wins",
                "objective_ties",
                "paramils_faster_wins",
                "ils_faster_wins",
                "tuning_time_ties",
                "paramils_higher_unique_config_wins",
                "ils_higher_unique_config_wins",
                "unique_config_ties",
            ]
        )

        for backend in BACKENDS:
            backend_runs = successful_runs_by_backend[backend]
            objectives = [run.objective for run in backend_runs if run.objective is not None]
            tuning_times = [
                run.tuning_time_seconds
                for run in backend_runs
                if run.tuning_time_seconds is not None
            ]
            unique_configs = [
                float(run.unique_config_count)
                for run in backend_runs
                if run.unique_config_count is not None
            ]
            evaluation_counts = [
                float(run.evaluation_count)
                for run in backend_runs
                if run.evaluation_count is not None
            ]

            writer.writerow(
                [
                    "backend",
                    backend,
                    sum(1 for run in runs if run.backend == backend),
                    len(backend_runs),
                    "",
                    optional_float(mean_or_none(objectives)),
                    optional_float(stdev_or_none(objectives)),
                    optional_float(min_or_none(objectives)),
                    optional_float(max_or_none(objectives)),
                    optional_float(mean_or_none(tuning_times)),
                    optional_float(stdev_or_none(tuning_times)),
                    optional_float(min_or_none(tuning_times)),
                    optional_float(max_or_none(tuning_times)),
                    optional_float(mean_or_none(unique_configs)),
                    optional_float(stdev_or_none(unique_configs)),
                    optional_float(min_or_none(unique_configs)),
                    optional_float(max_or_none(unique_configs)),
                    optional_float(mean_or_none(evaluation_counts)),
                    optional_float(stdev_or_none(evaluation_counts)),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        objective_deltas = [paramils.objective - ils.objective for ils, paramils in paired_successful_runs]
        tuning_time_deltas = [
            paramils.tuning_time_seconds - ils.tuning_time_seconds
            for ils, paramils in paired_successful_runs
        ]
        unique_config_deltas = [
            float(paramils.unique_config_count - ils.unique_config_count)
            for ils, paramils in paired_successful_runs
        ]

        paramils_lower_objective_wins, ils_lower_objective_wins, objective_ties = count_relation(
            [paramils.objective for ils, paramils in paired_successful_runs],
            [ils.objective for ils, paramils in paired_successful_runs],
            prefer_lower=True,
        )
        paramils_faster_wins, ils_faster_wins, tuning_time_ties = count_relation(
            [paramils.tuning_time_seconds for ils, paramils in paired_successful_runs],
            [ils.tuning_time_seconds for ils, paramils in paired_successful_runs],
            prefer_lower=True,
        )
        (
            paramils_higher_unique_config_wins,
            ils_higher_unique_config_wins,
            unique_config_ties,
        ) = count_relation(
            [float(paramils.unique_config_count) for ils, paramils in paired_successful_runs],
            [float(ils.unique_config_count) for ils, paramils in paired_successful_runs],
            prefer_lower=False,
        )

        writer.writerow(
            [
                "paired",
                "paramils_vs_ils",
                "",
                "",
                len(paired_successful_runs),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                optional_float(mean_or_none(objective_deltas)),
                optional_float(stdev_or_none(objective_deltas)),
                optional_float(mean_or_none(tuning_time_deltas)),
                optional_float(stdev_or_none(tuning_time_deltas)),
                optional_float(mean_or_none(unique_config_deltas)),
                optional_float(stdev_or_none(unique_config_deltas)),
                paramils_lower_objective_wins,
                ils_lower_objective_wins,
                objective_ties,
                paramils_faster_wins,
                ils_faster_wins,
                tuning_time_ties,
                paramils_higher_unique_config_wins,
                ils_higher_unique_config_wins,
                unique_config_ties,
            ]
        )


def validate_runtime_inputs(instance_path: Path, tuner_bin: Path, mpi_procs: int) -> None:
    if not instance_path.is_file():
        raise FileNotFoundError(f"Instance file not found: {instance_path}")
    if not tuner_bin.is_file():
        raise FileNotFoundError(f"Tuner binary not found: {tuner_bin}")
    if not tuner_bin.stat().st_mode & 0o111:
        raise PermissionError(f"Tuner binary is not executable: {tuner_bin}")
    if mpi_procs > 1 and shutil.which("mpirun") is None:
        raise FileNotFoundError("mpirun not found in PATH while --mpi-procs is greater than 1")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    instance_path = Path(args.instance).expanduser().resolve()
    tuner_bin = Path(args.tuner_bin).expanduser().resolve()
    output_root = resolve_output_root(instance_path, args.output_root)

    validate_runtime_inputs(instance_path, tuner_bin, args.mpi_procs)
    output_root.mkdir(parents=True, exist_ok=True)

    runs: list[RunResult] = []

    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        for backend in BACKENDS:
            runs.append(
                run_single(
                    tuner_bin=tuner_bin,
                    instance_path=instance_path,
                    backend=backend,
                    seed=seed,
                    output_root=output_root,
                    solver_time=args.solver_time,
                    solver_threads=args.solver_threads,
                    tuning_objective=args.tuning_objective,
                    mpi_procs=args.mpi_procs,
                )
            )

    write_runs_csv(output_root, runs)
    write_paired_csv(output_root, runs)
    write_summary_csv(output_root, runs)

    return 1 if any(run.status != "ok" for run in runs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
