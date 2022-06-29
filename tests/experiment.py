"""
The experiment module implements the Experiment dataclass.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Experiment:
    """Experiment class.

    Attributes
    ----------
    input_dataset: array-like of shape (n_samples, n_features)
        Training data, where `n_samples` is the number of samples
        and `n_features` is the number of features.

    input_errors: array-like of shape (n_samples, )
        Errors of a machine learning model.

    expected_top_k_slices: np.ndarray of shape (number of slices found, n_features)
        The `k` slices with the highest score.
        `None` values in slices represent unused column to define the slice.

    alpha: float, default=0.6
        Weight parameter for the importance of the average slice error.
        0 < `alpha` <= 1.

    k: int, default=1
        Maximum number of slices to return.
        Note: in case of equality between `k`-th slice score and the following ones,
        all those slices are returned, leading to more than `k` slices returned.

    max_l: int, default=4
        Maximum lattice level.
        In other words: the maximum number of predicate to define a slice.

    min_sup: int, default=10
        Minimum support threshold.
        Inspired by frequent itemset mining, it ensures statistical significance.

    verbose: bool, default=True
        Controls the verbosity.
    """

    input_dataset: np.ndarray
    input_errors: np.ndarray
    expected_top_k_slices: np.ndarray
    alpha: float = 0.95
    k: int = 2
    max_l: int = 2
    min_sup: int = 1
    verbose: bool = True
