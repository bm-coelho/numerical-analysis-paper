#!/usr/bin/env python3
"""Ensure that all experiment artifacts exist, regenerating them when necessary."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
IMG_DIR = REPO_ROOT / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExperimentSpec:
    script: str
    artifacts: tuple[Path, ...]


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        script="run_naive_solution.py",
        artifacts=(
            IMG_DIR / "naive_dataset.png",
            IMG_DIR / "naive_degree_1.png",
            IMG_DIR / "naive_degree_3.png",
            IMG_DIR / "naive_degree_5.png",
            IMG_DIR / "naive_rmse.png",
            IMG_DIR / "naive_rmse_log.png",
            IMG_DIR / "naive_target_dataset.png",
            IMG_DIR / "naive_target_degree_1.png",
            IMG_DIR / "naive_target_degree_3.png",
            RESULTS_DIR / "naive_regression_rmse.csv",
        ),
    ),
    ExperimentSpec(
        script="run_optimized_solution.py",
        artifacts=(
            IMG_DIR / "optimized_dataset.png",
            IMG_DIR / "optimized_degree_1.png",
            IMG_DIR / "optimized_degree_3.png",
            IMG_DIR / "optimized_degree_5.png",
            IMG_DIR / "optimized_rmse.png",
            IMG_DIR / "optimized_largeP_metrics.png",
            RESULTS_DIR / "optimized_regression_rmse.csv",
            RESULTS_DIR / "optimized_largeP_metrics.csv",
        ),
    ),
    ExperimentSpec(
        script="run_regkit_usage.py",
        artifacts=(
            IMG_DIR / "usage_dataset.png",
            IMG_DIR / "usage_naive_degree_1.png",
            IMG_DIR / "usage_naive_degree_2.png",
            IMG_DIR / "usage_naive_degree_6.png",
            IMG_DIR / "usage_naive_rmse.png",
            IMG_DIR / "usage_naive_rmse_log.png",
            IMG_DIR / "usage_optimized_rmse.png",
            IMG_DIR / "usage_optimized_rmse_log.png",
            IMG_DIR / "usage_target_dataset.png",
            IMG_DIR / "usage_target_optimized_degree_1.png",
            IMG_DIR / "usage_target_optimized_degree_3.png",
            IMG_DIR / "usage_target_optimized_degree_5.png",
            IMG_DIR / "usage_target_optimized_rmse.png",
            IMG_DIR / "usage_target_optimized_rmse_log.png",
            RESULTS_DIR / "regkit_usage_rmse.csv",
        ),
    ),
    ExperimentSpec(
        script="run_benchmark_regression.py",
        artifacts=(
            IMG_DIR / "benchmark_heatmaps.png",
            IMG_DIR / "benchmark_high_load.png",
            IMG_DIR / "benchmark_high_load_bars.png",
            RESULTS_DIR / "benchmark_scaling.csv",
            RESULTS_DIR / "benchmark_high_load.csv",
        ),
    ),
    ExperimentSpec(
        script="run_wing_surface_regression.py",
        artifacts=(
            IMG_DIR / "wing_samples_points.png",
            IMG_DIR / "wing_degree_sweep.png",
            IMG_DIR / "wing_residual_diagnostics.png",
            IMG_DIR / "wing_surface_views.png",
            IMG_DIR / "wing_contour_residuals.png",
            RESULTS_DIR / "wing_degree_sweep.csv",
            RESULTS_DIR / "wing_full_fit_metrics.csv",
            RESULTS_DIR / "wing_residual_summary.json",
        ),
    ),
    ExperimentSpec(
        script="run_ill_conditioned_regression.py",
        artifacts=(
            RESULTS_DIR / "ill_conditioned_gram.csv",
        ),
    ),
)


def _run_script(script: str) -> None:
    cmd = [sys.executable, str(EXPERIMENTS_DIR / script)]
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Experiment script {script} failed with code {completed.returncode}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute artifacts even if they already exist.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check for missing artifacts without running experiments.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    missing_total = []
    for spec in EXPERIMENTS:
        missing = [path for path in spec.artifacts if args.force or not path.exists()]
        if missing:
            missing_total.extend(missing)
            if args.check:
                continue
            print(f"Missing {len(missing)} artifact(s) for {spec.script}.")
            _run_script(spec.script)

    if missing_total and args.check:
        print("Missing artifacts:")
        for path in missing_total:
            print(f" - {path.relative_to(REPO_ROOT)}")
        return 1

    if not missing_total:
        print("All experiment artifacts are present.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
