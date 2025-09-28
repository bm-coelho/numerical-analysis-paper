#!/usr/bin/env python3
"""Reproduce the benchmarking experiments for the regression implementations."""

from __future__ import annotations

import csv
import gc
import time
import tracemalloc
from itertools import product
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from regkit.datasets import gen_exploration_dataset
from regkit.naive import regression_naive
from regkit.optimized import regression_largeP, regression_optimized

IMG_DIR = Path(__file__).resolve().parent.parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

POINT_LEVELS = [200, 600]
DEGREE_LEVELS = [2, 5]
IMPLEMENTATIONS = {
    "Naïve (symbolic)": lambda X, Y, degree: regression_naive(X, Y, degree, quiet=True),
    "Optimized block solve": lambda X, Y, degree: regression_optimized(X, Y, degree, quiet=True),
    "Out-of-core largeP": lambda X, Y, degree: regression_largeP(X, Y, degree, quiet=True),
}


def _save_current(path: Path) -> None:
    fig = plt.gcf()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _make_dataset(num_points: int, *, n: int = 3, m: int = 3, seed: int = 123, noise: float = 0.02):
    X_raw, Y_raw = gen_exploration_dataset(num_points, n=n, m=m, seed=seed, noise=noise)
    return X_raw.T, Y_raw.T


def _run_once(func, X: np.ndarray, Y: np.ndarray, degree: int) -> None:
    _ = func(X, Y, degree)


def _benchmark(func, X: np.ndarray, Y: np.ndarray, degree: int, *, repeat: int = 3):
    timings: list[float] = []
    peaks: list[float] = []
    _run_once(func, X, Y, degree)
    for _ in range(repeat):
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        _run_once(func, X, Y, degree)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        timings.append(elapsed)
        peaks.append(peak / 1e6)
    return float(np.mean(timings)), float(np.std(timings)), float(np.mean(peaks))


def _write_csv(filename: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path = RESULTS_DIR / filename
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    records: list[dict[str, float | int | str]] = []
    for num_points, degree in product(POINT_LEVELS, DEGREE_LEVELS):
        X, Y = _make_dataset(num_points)
        repeats = 3 if num_points <= 500 else 2
        for name, func in IMPLEMENTATIONS.items():
            if name == "Naïve (symbolic)" and num_points > 200:
                continue
            local_repeat = 1 if name == "Naïve (symbolic)" else repeats
            mean_t, std_t, mean_mem = _benchmark(func, X, Y, degree, repeat=local_repeat)
            records.append(
                {
                    "Implementation": name,
                    "Points": num_points,
                    "Degree": degree,
                    "Repeat": local_repeat,
                    "Mean time [s]": mean_t,
                    "Std time [s]": std_t,
                    "Peak memory [MB]": mean_mem,
                }
            )

    impl_order = list(IMPLEMENTATIONS.keys())
    points = POINT_LEVELS
    degrees = DEGREE_LEVELS

    plt.style.use("seaborn-v0_8-muted")
    fig, axes = plt.subplots(len(impl_order), 2, figsize=(10, 3.5 * len(impl_order)), constrained_layout=True)

    for i, name in enumerate(impl_order):
        time_grid = np.full((len(points), len(degrees)), np.nan)
        mem_grid = np.full_like(time_grid, np.nan)
        for rec in records:
            if rec["Implementation"] != name:
                continue
            pi = points.index(rec["Points"])
            di = degrees.index(rec["Degree"])
            time_grid[pi, di] = rec["Mean time [s]"]
            mem_grid[pi, di] = rec["Peak memory [MB]"]

        for ax, grid, label in zip(
            axes[i],
            (time_grid, mem_grid),
            ("Mean time [s]", "Peak memory [MB]"),
        ):
            im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", cmap="viridis")
            ax.set_xticks(range(len(degrees)), degrees)
            ax.set_yticks(range(len(points)), points)
            ax.set_xlabel("Degree")
            ax.set_ylabel("Points")
            ax.set_title(f"{name} — {label}")
            for (y, x), value in np.ndenumerate(grid):
                if np.isnan(value):
                    text = "—"
                else:
                    text = f"{value:.2f}"
                ax.text(x, y, text, ha="center", va="center", color="white", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Runtime and memory scaling across regression implementations", fontsize=16)
    _save_current(IMG_DIR / "benchmark_heatmaps.png")

    _write_csv(
        "benchmark_scaling.csv",
        [
            "Implementation",
            "Points",
            "Degree",
            "Repeat",
            "Mean time [s]",
            "Std time [s]",
            "Peak memory [MB]",
        ],
        records,
    )

    # High load benchmark
    large_points = 1_000_000
    large_degree = 5
    high_load_impl = {
        "Optimized block solve": IMPLEMENTATIONS["Optimized block solve"],
        "Out-of-core largeP": IMPLEMENTATIONS["Out-of-core largeP"],
    }

    X_large, Y_large = _make_dataset(large_points)
    large_records: list[dict[str, float | int | str]] = []
    for name, func in high_load_impl.items():
        mean_t, std_t, mean_mem = _benchmark(func, X_large, Y_large, large_degree, repeat=1)
        large_records.append(
            {
                "Implementation": name,
                "Points": large_points,
                "Degree": large_degree,
                "Repeat": 1,
                "Mean time [s]": mean_t,
                "Std time [s]": std_t,
                "Peak memory [MB]": mean_mem,
            }
        )

    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=[
            [
                rec["Implementation"],
                f"{rec['Mean time [s]']:.3f}",
                f"{rec['Std time [s]']:.3f}",
                f"{rec['Peak memory [MB]']:.2f}",
            ]
            for rec in large_records
        ],
        colLabels=["Implementation", "Mean time [s]", "Std [s]", "Peak memory [MB]"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax.set_title(f"High-load benchmark ({large_points} points, degree {large_degree})")
    _save_current(IMG_DIR / "benchmark_high_load.png")

    _write_csv(
        "benchmark_high_load.csv",
        [
            "Implementation",
            "Points",
            "Degree",
            "Repeat",
            "Mean time [s]",
            "Std time [s]",
            "Peak memory [MB]",
        ],
        large_records,
    )

    # Large-scale summary bar charts mirroring notebook visuals
    labels = [rec["Implementation"] for rec in large_records]
    times = np.array([rec["Mean time [s]"] for rec in large_records], dtype=float)
    stds = np.array([rec["Std time [s]"] for rec in large_records], dtype=float)
    mems = np.array([rec["Peak memory [MB]"] for rec in large_records], dtype=float)

    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)

    bars_time = axes[0].bar(labels, times, yerr=stds, capsize=6, color=plt.cm.Blues(np.linspace(0.55, 0.85, len(labels))))
    axes[0].set_title("Execution time")
    axes[0].set_ylabel("Seconds")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    max_time = float(np.max(times + stds)) if np.any(times + stds) else float(np.max(times) if np.any(times) else 1.0)
    axes[0].set_ylim(0, max_time * 1.25)
    for bar, mean in zip(bars_time, times):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mean:.2f}s", ha="center", va="bottom", fontsize=10)

    bars_mem = axes[1].bar(labels, mems, color=plt.cm.Greens(np.linspace(0.55, 0.85, len(labels))))
    axes[1].set_title("Peak memory use")
    axes[1].set_ylabel("MB")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    max_mem = float(np.max(mems)) if np.any(mems) else 1.0
    axes[1].set_ylim(0, max_mem * 1.25)
    for bar, mem in zip(bars_mem, mems):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mem:.1f} MB", ha="center", va="bottom", fontsize=10)

    fig.suptitle(f"Large-scale benchmark performance ({large_points} points, degree {large_degree})", fontsize=15)
    _save_current(IMG_DIR / "benchmark_high_load_bars.png")


if __name__ == "__main__":
    main()
