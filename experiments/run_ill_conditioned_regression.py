#!/usr/bin/env python3
"""Reproduce scenarios that yield ill-conditioned Gram matrices."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from regkit import regression_largeP, regression_optimized
from regkit.optimized import build_block_system

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class Scenario:
    """Container describing a reproducible ill-conditioning setup."""

    name: str
    description: str
    degree: int
    generator: Callable[[], tuple[np.ndarray, np.ndarray]]


def _high_degree_low_points() -> tuple[np.ndarray, np.ndarray]:
    """Return samples whose Gram matrix deteriorates with high degree."""

    X = np.linspace(-1.0, 1.0, 15, dtype=float).reshape(-1, 1)
    Y = np.hstack([np.sin(np.pi * X), X**2])
    return X, Y


def _duplicate_samples() -> tuple[np.ndarray, np.ndarray]:
    """Return a dataset with coincident points causing rank deficiency."""

    X = np.zeros((10, 1), dtype=float)
    trend = np.linspace(0.0, 1.0, X.shape[0], dtype=float).reshape(-1, 1)
    Y = np.hstack([0.5 * np.ones_like(trend), trend])
    return X, Y


SCENARIOS = (
    Scenario(
        name="high_degree_low_points",
        description=(
            "15 amostras equiespaçadas em [-1, 1] ajustadas com um polinômio de grau 14."
        ),
        degree=14,
        generator=_high_degree_low_points,
    ),
    Scenario(
        name="duplicate_samples",
        description=(
            "10 amostras coincidentes com ajuste cúbico, resultando em matriz de Vandermonde degenerada."
        ),
        degree=3,
        generator=_duplicate_samples,
    ),
)


def _format_cond(condition_number: float) -> str:
    if not np.isfinite(condition_number):
        return "inf"
    return f"{condition_number:.4e}"


def main() -> None:
    rows: list[dict[str, object]] = []

    for scenario in SCENARIOS:
        X, Y = scenario.generator()
        G, _, _ = build_block_system(X, Y, degree=scenario.degree)
        cond = float(np.linalg.cond(G))
        cond_repr = _format_cond(cond)
        ill_conditioned = (not np.isfinite(cond)) or cond > 1e12

        print(f"Scenario: {scenario.name}")
        print(f"  descrição: {scenario.description}")
        print(f"  cond(G) ≈ {cond_repr}")

        # Trigger the fallbacks/noise injection in both solvers (console warns automatically)
        regression_optimized(X, Y, degree=scenario.degree, quiet=False)
        _, _, _ = regression_largeP(
            X,
            Y,
            degree=scenario.degree,
            batch_size=max(4, X.shape[0]),
            ridge=0.0,
            quiet=False,
        )

        action_optimized = "fallback_to_lstsq" if ill_conditioned else "block_solve"
        ridge_applied = 0.0 if not ill_conditioned else max(0.0, 1e-8)

        rows.append(
            {
                "scenario": scenario.name,
                "solver": "regression_optimized",
                "degree": scenario.degree,
                "num_points": X.shape[0],
                "cond": cond_repr,
                "action": action_optimized,
                "ridge_effective": 0.0,
            }
        )
        rows.append(
            {
                "scenario": scenario.name,
                "solver": "regression_largeP",
                "degree": scenario.degree,
                "num_points": X.shape[0],
                "cond": cond_repr,
                "action": "ridge_added" if ill_conditioned else "block_solve",
                "ridge_effective": ridge_applied,
            }
        )

        print("  ação (regression_optimized):", action_optimized)
        if ill_conditioned:
            print(
                "  ação (regression_largeP): ridge_added com magnitude",
                f"{ridge_applied:.1e}",
            )
        else:
            print("  ação (regression_largeP): block_solve")
        print()

    csv_path = RESULTS_DIR / "ill_conditioned_gram.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=(
                "scenario",
                "solver",
                "degree",
                "num_points",
                "cond",
                "action",
                "ridge_effective",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resultados tabulados em {csv_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
