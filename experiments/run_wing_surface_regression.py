#!/usr/bin/env python3
"""Generate the wing surface regression figures from the exploratory notebook."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- required for 3D plotting
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from regkit import regression_optimized
SPEC_DIR = REPO_ROOT / "spec"
IMG_DIR = REPO_ROOT / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8")


def _save_current(path: Path) -> None:
    fig = plt.gcf()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _prepare_grid(
    x: np.ndarray,
    y: np.ndarray,
    predict,
    scaler: StandardScaler,
    num: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_lin = np.linspace(x.min(), x.max(), num)
    y_lin = np.linspace(y.min(), y.max(), num)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    grid_scaled = scaler.transform(grid_points)
    Z_pred = predict(grid_scaled).reshape(X_grid.shape)
    return X_grid, Y_grid, Z_pred


def main() -> None:
    data_path = SPEC_DIR / "dados_asa.xls"
    raw_df = pd.read_excel(data_path)
    wing_df = raw_df.rename(
        columns={"x": "x_longitudinal", "y": "y_span", "z": "z_height"}
    )[["x_longitudinal", "y_span", "z_height"]]

    # Scatter plot of sampled points
    plt.close("all")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        wing_df["x_longitudinal"],
        wing_df["y_span"],
        wing_df["z_height"],
        s=16,
        c=wing_df["z_height"],
        cmap="viridis",
    )
    ax.set_xlabel("x (longitudinal)")
    ax.set_ylabel("y (spanwise)")
    ax.set_zlabel("z (height)")
    ax.set_title("Measured wing surface samples")
    fig.colorbar(sc, ax=ax, pad=0.1, label="Height (mm)")
    plt.tight_layout()
    _save_current(IMG_DIR / "wing_samples_points.png")

    # Prepare training/test split
    X = wing_df[["x_longitudinal", "y_span"]].to_numpy()
    y = wing_df[["z_height"]].to_numpy()
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=13
    )

    degree_range = range(2, 8)
    records: list[dict[str, float | int]] = []
    for degree in degree_range:
        _, _, predict = regression_optimized(X_train, y_train, degree=degree, quiet=True)
        y_train_hat = predict(X_train)
        y_test_hat = predict(X_test)
        records.append(
            {
                "degree": degree,
                "train_rmse": _rmse(y_train, y_train_hat),
                "test_rmse": _rmse(y_test, y_test_hat),
                "test_r2": float(r2_score(y_test, y_test_hat)),
            }
        )

    degree_df = pd.DataFrame.from_records(records)
    degree_df.to_csv(RESULTS_DIR / "wing_degree_sweep.csv", index=False)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(degree_df["degree"], degree_df["train_rmse"], marker="o", label="Train")
    ax.plot(degree_df["degree"], degree_df["test_rmse"], marker="o", label="Test")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("RMSE (mm)")
    ax.set_title("Polynomial degree sweep")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    _save_current(IMG_DIR / "wing_degree_sweep.png")

    best_row = degree_df.loc[degree_df["test_rmse"].idxmin()]
    best_degree = int(best_row["degree"])

    _, _, predict_full = regression_optimized(X_scaled, y, degree=best_degree, quiet=True)
    y_hat = predict_full(X_scaled)
    residuals = (y - y_hat).ravel()
    rmse_full = _rmse(y, y_hat)
    r2_full = float(r2_score(y, y_hat))

    metrics_path = RESULTS_DIR / "wing_full_fit_metrics.csv"
    pd.DataFrame(
        [
            {
                "degree": best_degree,
                "rmse": rmse_full,
                "r2": r2_full,
            }
        ]
    ).to_csv(metrics_path, index=False)

    residual_summary = pd.Series(residuals).describe().to_dict()
    with (RESULTS_DIR / "wing_residual_summary.json").open("w") as fh:
        json.dump(residual_summary, fh, indent=2)

    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(residuals, bins=30, color="steelblue", edgecolor="black")
    axes[0].set_title("Residual distribution")
    axes[0].set_xlabel("Prediction error (mm)")
    axes[0].set_ylabel("Frequency")
    axes[1].scatter(y, y_hat, s=20, alpha=0.7)
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], "k--", linewidth=1)
    axes[1].set_xlabel("Measured height (mm)")
    axes[1].set_ylabel("Predicted height (mm)")
    axes[1].set_title("Measured vs predicted")
    plt.tight_layout()
    _save_current(IMG_DIR / "wing_residual_diagnostics.png")

    X_grid, Y_grid, Z_pred = _prepare_grid(
        wing_df["x_longitudinal"].to_numpy(),
        wing_df["y_span"].to_numpy(),
        predict_full,
        scaler,
    )

    plt.close("all")
    fig = plt.figure(figsize=(18, 6))
    views = [(30, -60), (35, 45), (10, -90)]
    for idx, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        surf = ax.plot_surface(X_grid, Y_grid, Z_pred, cmap="viridis", linewidth=0, alpha=0.85)
        ax.scatter(
            wing_df["x_longitudinal"],
            wing_df["y_span"],
            wing_df["z_height"],
            c="k",
            s=10,
            alpha=0.6,
        )
        ax.set_xlabel("x (longitudinal)")
        ax.set_ylabel("y (spanwise)")
        ax.set_zlabel("z (height)")
        ax.set_title(f"Reconstructed surface (elev={elev}, azim={azim})")
        ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, ax=fig.axes, shrink=0.6, pad=0.05, label="Predicted height (mm)")
    plt.tight_layout()
    _save_current(IMG_DIR / "wing_surface_views.png")

    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X_grid, Y_grid, Z_pred, levels=40, cmap="viridis")
    sc_res = ax.scatter(
        wing_df["x_longitudinal"],
        wing_df["y_span"],
        c=residuals,
        cmap="coolwarm",
        edgecolor="black",
        s=35,
    )
    ax.set_xlabel("x (longitudinal)")
    ax.set_ylabel("y (spanwise)")
    ax.set_title("Surface contour and pointwise residuals")
    cbar1 = fig.colorbar(contour, ax=ax, shrink=0.75, pad=0.02)
    cbar1.set_label("Predicted height (mm)")
    cbar2 = fig.colorbar(sc_res, ax=ax, shrink=0.75, pad=0.08)
    cbar2.set_label("Residual (mm)")
    plt.tight_layout()
    _save_current(IMG_DIR / "wing_contour_residuals.png")


if __name__ == "__main__":
    main()
