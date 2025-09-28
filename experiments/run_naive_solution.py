#!/usr/bin/env python3
"""Reproduce the exploratory symbolic regression experiments."""

from __future__ import annotations

import csv
import sys
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


def main() -> None:
    np.random.seed(42)

    # --- Exploratory manifold dataset -------------------------------------------------
    X_manifold, Y_manifold = gen_exploration_dataset(
        num_points=200,
        n=3,
        m=2,
        seed=42,
        noise=0.01,
    )
    reducer = _plot_and_save("naive_dataset.png", X_manifold, Y_manifold)

    X_train = X_manifold.T
    Y_train = Y_manifold.T

    degrees = [1, 3, 5]
    errors = []
    manifold_records: list[dict[str, object]] = []
    m = Y_train.shape[1]
    for degree in degrees:
        coeffs, approx, predictor = regression_naive(X_train, Y_train, degree, quiet=True)
        Y_pred = _predict_matrix(predictor, X_train, m)
        rmse = _rmse(Y_train, Y_pred)
        errors.append(rmse)
        manifold_records.append(
            {
                "dataset": "exploration",
                "degree": degree,
                "rmse": rmse,
            }
        )
        _plot_and_save(
            f"naive_degree_{degree}.png",
            X_manifold,
            Y_pred.T,
            reducer=reducer,
        )

    # Error curves up to degree 5
    all_degrees = list(range(1, 6))
    curve_errors = []
    for degree in all_degrees:
        _, _, predictor = regression_naive(X_train, Y_train, degree, quiet=True)
        Y_pred = _predict_matrix(predictor, X_train, m)
        rmse = _rmse(Y_train, Y_pred)
        curve_errors.append(rmse)
        manifold_records.append(
            {
                "dataset": "exploration",
                "degree": degree,
                "rmse": rmse,
            }
        )

    plt.close("all")
    plt.figure()
    plt.plot(all_degrees, curve_errors, marker="o")
    plt.xlabel("Polynomial degree")
    plt.ylabel("RMSE")
    plt.title("Naïve regression accuracy")
    plt.grid(True)
    _save_current(IMG_DIR / "naive_rmse.png")

    plt.figure()
    plt.plot(all_degrees, np.log(curve_errors), marker="o")
    plt.xlabel("Polynomial degree")
    plt.ylabel("log(RMSE)")
    plt.title("Log RMSE vs degree (naïve)")
    plt.grid(True)
    _save_current(IMG_DIR / "naive_rmse_log.png")

    # --- Structured target dataset ----------------------------------------------------
    X_target, Y_target = create_target_dataset()
    reducer_target = _plot_and_save("naive_target_dataset.png", X_target, Y_target)

    X_target_train = X_target.T
    Y_target_train = Y_target.T

    m_target = Y_target_train.shape[1]
    target_degrees = [1, 3]
    target_records: list[dict[str, object]] = []
    for degree in target_degrees:
        _, _, predictor = regression_naive(
            X_target_train,
            Y_target_train,
            degree,
            quiet=True,
        )
        Y_pred = _predict_matrix(predictor, X_target_train, m_target)
        rmse = _rmse(Y_target_train, Y_pred)
        target_records.append(
            {
                "dataset": "target",
                "degree": degree,
                "rmse": rmse,
            }
        )
        _plot_and_save(
            f"naive_target_degree_{degree}.png",
            X_target,
            Y_pred.T,
            reducer=reducer_target,
        )

    combined_records = manifold_records + target_records
    seen: set[tuple[str, int]] = set()
    filtered: list[dict[str, object]] = []
    for record in combined_records:
        key = (str(record["dataset"]), int(record["degree"]))
        if key in seen:
            continue
        seen.add(key)
        filtered.append(record)

    _write_csv("naive_regression_rmse.csv", ["dataset", "degree", "rmse"], filtered)


if __name__ == "__main__":
    main()
