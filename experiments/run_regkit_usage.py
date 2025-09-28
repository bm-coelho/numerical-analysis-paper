#!/usr/bin/env python3
"""Replicate the consolidated usage experiments for the regkit package."""

from __future__ import annotations

import sys
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from regkit import (
    create_target_dataset,
    gen_exploration_dataset,
    plot_dataset,
    regression_naive,
    regression_optimized,
)

IMG_DIR = Path(__file__).resolve().parent.parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_current(path: Path) -> None:
    fig = plt.gcf()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_and_save(name: str, X: np.ndarray, Y: np.ndarray, **kwargs):
    plt.close("all")
    kwargs.setdefault("show", False)
    reducer = plot_dataset(X, Y, **kwargs)
    _save_current(IMG_DIR / name)
    return reducer


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _write_csv(filename: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path = RESULTS_DIR / filename
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _predict_matrix(predictor, X: np.ndarray, m: int) -> np.ndarray:
    arr = np.asarray(predictor(X))
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim == 1:
        arr = arr.reshape(-1, m)
    if arr.ndim == 2 and arr.shape[0] == m:
        arr = arr.T
    return arr


def _plot_error_curves(filename_prefix: str, degrees: list[int], errors: list[float]) -> None:
    plt.close("all")
    plt.figure()
    plt.plot(degrees, errors, marker="o")
    plt.xlabel("Polynomial degree")
    plt.ylabel("RMSE")
    plt.title(f"RMSE vs degree — {filename_prefix}")
    plt.grid(True)
    _save_current(IMG_DIR / f"{filename_prefix}_rmse.png")

    plt.figure()
    plt.plot(degrees, np.log(errors), marker="o")
    plt.xlabel("Polynomial degree")
    plt.ylabel("log(RMSE)")
    plt.title(f"log(RMSE) vs degree — {filename_prefix}")
    plt.grid(True)
    _save_current(IMG_DIR / f"{filename_prefix}_rmse_log.png")


def main() -> None:
    # --- Exploratory dataset ----------------------------------------------------------
    X_raw, Y_raw = gen_exploration_dataset(
        num_points=200,
        n=3,
        m=2,
        seed=42,
        noise=0.01,
    )
    reducer = _plot_and_save("usage_dataset.png", X_raw, Y_raw)

    X = X_raw.T
    Y = Y_raw.T

    m = Y.shape[1]
    for degree in [1, 2, 6]:
        _, _, predictor = regression_naive(X, Y, degree, quiet=True)
        Y_pred = _predict_matrix(predictor, X, m)
        _plot_and_save(
            f"usage_naive_degree_{degree}.png",
            X_raw,
            Y_pred.T,
            reducer=reducer,
        )

    max_degree_naive = 5
    degrees = list(range(1, max_degree_naive + 1))
    naive_errors = []
    naive_records: list[dict[str, object]] = []
    for degree in degrees:
        _, _, predictor = regression_naive(X, Y, degree, quiet=True)
        Y_pred = _predict_matrix(predictor, X, m)
        rmse = _rmse(Y, Y_pred)
        naive_errors.append(rmse)
        naive_records.append(
            {
                "dataset": "exploration",
                "method": "naive",
                "degree": degree,
                "rmse": rmse,
            }
        )
    _plot_error_curves("usage_naive", degrees, naive_errors)

    max_degree_optimized = 20
    opt_degrees = list(range(1, max_degree_optimized + 1))
    optimized_errors = []
    optimized_records: list[dict[str, object]] = []
    for degree in opt_degrees:
        _, _, predictor = regression_optimized(X, Y, degree, quiet=True)
        Y_pred = predictor(X)
        rmse = _rmse(Y, Y_pred)
        optimized_errors.append(rmse)
        optimized_records.append(
            {
                "dataset": "exploration",
                "method": "optimized",
                "degree": degree,
                "rmse": rmse,
            }
        )
    _plot_error_curves("usage_optimized", opt_degrees, optimized_errors)

    # --- Structured target dataset ----------------------------------------------------
    X_target, Y_target = create_target_dataset()
    reducer_target = _plot_and_save("usage_target_dataset.png", X_target, Y_target)

    X_target_train = X_target.T
    Y_target_train = Y_target.T

    for degree in [1, 3, 5]:
        _, _, predictor = regression_optimized(
            X_target_train,
            Y_target_train,
            degree,
            quiet=True,
        )
        Y_pred = predictor(X_target_train)
        _plot_and_save(
            f"usage_target_optimized_degree_{degree}.png",
            X_target,
            Y_pred.T,
            reducer=reducer_target,
        )

    max_degree_target = 10
    target_degrees = list(range(1, max_degree_target + 1))
    target_errors = []
    target_records: list[dict[str, object]] = []
    for degree in target_degrees:
        _, _, predictor = regression_optimized(
            X_target_train,
            Y_target_train,
            degree,
            quiet=True,
        )
        Y_pred = predictor(X_target_train)
        rmse = _rmse(Y_target_train, Y_pred)
        target_errors.append(rmse)
        target_records.append(
            {
                "dataset": "target",
                "method": "optimized",
                "degree": degree,
                "rmse": rmse,
            }
        )
    _plot_error_curves("usage_target_optimized", target_degrees, target_errors)

    combined_records = naive_records + optimized_records + target_records
    _write_csv(
        "regkit_usage_rmse.csv",
        ["dataset", "method", "degree", "rmse"],
        combined_records,
    )


if __name__ == "__main__":
    main()
