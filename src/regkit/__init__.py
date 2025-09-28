"""Reusable utilities for multivariate polynomial regression."""

from .naive import regression_naive
from .optimized import (
    regression_optimized,
    regression_largeP,
    regression_largeP_iter,
)
from .datasets import gen_exploration_dataset, create_target_dataset
from .viz import plot_dataset

__all__ = [
    "regression_naive",
    "regression_optimized",
    "regression_largeP",
    "regression_largeP_iter",
    "gen_exploration_dataset",
    "create_target_dataset",
    "plot_dataset",
]
