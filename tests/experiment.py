"""
The experiment module implements the Experiment dataclass.
"""

from dataclasses import dataclass
from typing import Dict, List, Union

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
        The slices found with the highest score.
        `None` values in slices represent unused column in the slice.

    expected_top_k_slices_statistics: list of dict of length `len(expected_top_k_slices)`
        The statistics of the slices found sorted by slice's scores.
        For each slice, the following statistics are stored:
            - slice_score: the score of the slice (defined in `_score` method)
            - sum_slice_error: the sum of all the errors in the slice
            - max_slice_error: the maximum of all errors in the slice
            - slice_size: the number of elements in the slice
            - slice_average_error: the average error in the slice (sum_slice_error / slice_size)

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

    min_sup: int or float, default=10
        Minimum support threshold.
        Inspired by frequent itemset mining, it ensures statistical significance.
        If `min_sup` is a float (0 < `min_sup` < 1),
            it represents the faction of the input dataset (`X`)

    verbose: bool, default=True
        Controls the verbosity.
    """

    input_dataset: np.ndarray
    input_errors: np.ndarray
    expected_top_k_slices: np.ndarray
    expected_top_k_slices_statistics: List[Dict[str, float]]
    alpha: float = 0.95
    k: int = 2
    max_l: int = 2
    min_sup: Union[int, float] = 1
    verbose: bool = True
