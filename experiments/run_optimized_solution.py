#!/usr/bin/env python3
"""Reproduce the optimized regression experiments."""

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
    gen_exploration_dataset,
    plot_dataset,
    regression_largeP,
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


def main() -> None:
    # --- Optimized regression on exploratory manifold ---------------------------------
    X_raw, Y_raw = gen_exploration_dataset(
        num_points=1500,
        n=3,
        m=2,
        seed=7,
        noise=0.02,
    )
    reducer = _plot_and_save("optimized_dataset.png", X_raw, Y_raw)

    X = X_raw.T
    Y = Y_raw.T

    for degree in [1, 3, 5]:
        _, _, predict = regression_optimized(X, Y, degree=degree, quiet=True)
        Y_pred = predict(X)
        _plot_and_save(
            f"optimized_degree_{degree}.png",
            X_raw,
            Y_pred.T,
            reducer=reducer,
        )

    degrees = list(range(1, 6))
    errors = []
    rmse_records: list[dict[str, object]] = []
    for degree in degrees:
        _, _, predict = regression_optimized(X, Y, degree=degree, quiet=True)
        Y_pred = predict(X)
        rmse = _rmse(Y, Y_pred)
        errors.append(rmse)
        rmse_records.append(
            {
                "dataset": "exploration",
                "degree": degree,
                "rmse": rmse,
            }
        )

    plt.close("all")
    plt.figure()
    plt.plot(degrees, errors, marker="o")
    plt.xlabel("Polynomial degree")
    plt.ylabel("RMSE")
    plt.title("Optimized regression accuracy")
    plt.grid(True)
    _save_current(IMG_DIR / "optimized_rmse.png")

    # --- Streaming vs in-core comparison ----------------------------------------------
    X_big_raw, Y_big_raw = gen_exploration_dataset(
        num_points=40000,
        n=3,
        m=2,
        seed=123,
        noise=0.01,
    )
    X_big = X_big_raw.T
    Y_big = Y_big_raw.T

    A_full, poly_full, predict_full = regression_optimized(X_big, Y_big, degree=3, quiet=True)
    Y_full = predict_full(X_big)
    rmse_full = _rmse(Y_big, Y_full)

    A_stream, poly_stream, predict_stream = regression_largeP(
        X_big,
        Y_big,
        degree=3,
        batch_size=4096,
        ridge=1e-10,
        quiet=True,
    )
    Y_stream = predict_stream(X_big)
    rmse_stream = _rmse(Y_big, Y_stream)
    coef_diff = float(np.linalg.norm(A_full - A_stream, ord="fro"))

    _write_csv(
        "optimized_regression_rmse.csv",
        ["dataset", "degree", "rmse"],
        rmse_records,
    )

    _write_csv(
        "optimized_largeP_metrics.csv",
        ["method", "rmse", "coef_diff_fro"],
        [
            {"method": "full", "rmse": rmse_full, "coef_diff_fro": 0.0},
            {
                "method": "streaming",
                "rmse": rmse_stream,
                "coef_diff_fro": coef_diff,
            },
        ],
    )

    plt.close("all")
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    ax.axis("off")
    table_data = [
        ["Full solve", f"{rmse_full:.6f}", "—"],
        ["Streaming", f"{rmse_stream:.6f}", f"{coef_diff:.2e}"],
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=["Method", "RMSE", "‖ΔA‖₍F₎"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax.set_title("Large-P regression comparison (degree 3)")
    _save_current(IMG_DIR / "optimized_largeP_metrics.png")


if __name__ == "__main__":
    main()
