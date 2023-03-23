"""
The conftest module implements two pytest fixtures:
- basic_test_data
- experiments
"""
import numpy as np
import pytest
from scipy import sparse as sp

from sliceline import slicefinder
from tests.experiment import Experiment


@pytest.fixture
def basic_test_data():
    """Implement a coherent example to test SliceLine internal functions at each step."""
    alpha = 0.95
    k = 2
    max_l = 2
    min_sup = 1
    verbose = True

    # input dataset
    # no constant column
    # integer-encoded form
    # 1-based
    # continuous integer range
    X = np.array(
        [
            [1, 1, 1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2, 1, 1],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [3, 3, 3, 1, 3, 1, 2, 1],
        ]
    ).T

    # error or label of elements
    # By default, the algorithm identifies slices targeting 0 of E.
    errors = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    slicefinder_model = slicefinder.Slicefinder(
        alpha=alpha, k=k, max_l=max_l, min_sup=min_sup, verbose=verbose
    )
    slicefinder_model.fit(X, errors)

    n_col_x_encoded = 15
    average_error = 0.5
    level = 2
    feature_offset_start = np.array([0, 2, 4, 12])
    feature_offset_end = np.array([2, 4, 12, 15])
    feature_domains = np.array([2, 2, 8, 3])
    top_k_statistics = np.array(
        [
            [0.6, 1, 1, 1],
            [0.6, 1, 1, 1],
        ]
    )
    X_encoded = sp.csr_matrix(
        [
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        ]
    )
    slices = sp.csr_matrix(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    candidates = sp.csr_matrix(
        [
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    return {
        "alpha": alpha,
        "k": k,
        "max_l": max_l,
        "min_sup": min_sup,
        "verbose": verbose,
        "X": X,
        "errors": errors,
        "slicefinder_model": slicefinder_model,
        "n_col_x_encoded": n_col_x_encoded,
        "average_error": average_error,
        "level": level,
        "feature_offset_start": feature_offset_start,
        "feature_offset_end": feature_offset_end,
        "feature_domains": feature_domains,
        "top_k_statistics": top_k_statistics,
        "X_encoded": X_encoded,
        "slices": slices,
        "candidates": candidates,
    }


@pytest.fixture
def experiments():
    """Implement several end-to-end examples with the expected outputs regarding the inputs."""
    # Experiment 1: basic case
    np.random.seed(1)
    n_small = 10
    X_1 = np.array(
        [
            [1] * n_small + [2] * (n_small // 2) + [1] * (n_small // 2),
            [1] * n_small + [1] * (n_small // 2) + [2] * (n_small // 2),
            np.random.randint(1, 4, size=2 * n_small),
        ]
    ).T
    errors_1 = np.array([1] * n_small + [0] * n_small)
    expected_top_slices_1 = np.array([[1, 1, None], [None, 1, 2], [1, None, 2]])
    experiment_1 = Experiment(X_1, errors_1, expected_top_slices_1)

    # Experiment 2: Experiment 1 + more columns and different order
    np.random.seed(2)
    X_2 = np.array(
        [
            np.random.randint(1, 4, size=2 * n_small),
            [1] * n_small + [1] * (n_small // 2) + [2] * (n_small // 2),
            np.random.randint(1, 5, size=2 * n_small),
            np.random.randint(1, 7, size=2 * n_small),
            [1] * n_small + [2] * (n_small // 2) + [1] * (n_small // 2),
            np.arange(2 * n_small) + 1,
        ]
    ).T
    errors_2 = np.array([1] * n_small + [0] * n_small)
    expected_top_slices_2 = np.array(
        [[None, 1.0, None, None, 1.0, None], [None, None, 4.0, None, 1.0, None]]
    )
    experiment_2 = Experiment(X_2, errors_2, expected_top_slices_2)

    # Experiment 3: Experiment 1 + more rows
    np.random.seed(3)
    n = 100
    X_3 = np.array(
        [
            [1] * n + [2] * (n // 2) + [1] * (n // 2),
            [1] * n + [1] * (n // 2) + [2] * (n // 2),
            np.random.randint(1, 4, size=2 * n),
        ]
    ).T
    errors_3 = np.array([1] * n + [0] * n)
    expected_top_slices_3 = np.array([[1.0, 1.0, None], [1.0, None, None], [None, 1.0, None]])
    experiment_3 = Experiment(X_3, errors_3, expected_top_slices_3)

    # Experiment 4: Experiment 3 + more columns
    np.random.seed(4)
    n = 100
    X_4 = np.array(
        [
            [1] * n + [2] * (n // 2) + [1] * (n // 2),
            [1] * n + [1] * (n // 2) + [2] * (n // 2),
            np.random.randint(1, 4, size=2 * n),
            np.random.randint(1, 5, size=2 * n),
            np.random.randint(1, 7, size=2 * n),
            np.arange(2 * n) + 1,
        ]
    ).T
    errors_4 = np.array([1] * n + [0] * n)
    expected_top_slices_4 = np.array(
        [[1.0, 1.0, None, None, None, None], [1.0, None, 3.0, None, None, None]]
    )
    experiment_4 = Experiment(X_4, errors_4, expected_top_slices_4)

    # Experiment 5: Experiment 4 w/ min_sup=50
    expected_top_slices_5 = np.array(
        [[1.0, 1.0, None, None, None, None], [1.0, None, 3.0, None, None, None]]
    )
    experiment_5 = Experiment(X_4, errors_4, expected_top_slices_5, min_sup=50)

    # Experiment 6: Experiment 4 w/ max_l=1
    expected_top_slices_6 = np.array(
        [[1.0, None, None, None, None, None], [None, 1.0, None, None, None, None]]
    )
    experiment_6 = Experiment(X_4, errors_4, expected_top_slices_6, max_l=1)

    # Experiment 7: Experiment 4 w/ alpha=0.7
    expected_top_slices_7 = np.array(
        [
            [1.0, 1.0, None, None, None, None],
            [1.0, None, None, None, None, None],
            [None, 1.0, None, None, None, None],
        ]
    )
    experiment_7 = Experiment(X_4, errors_4, expected_top_slices_7, alpha=0.7)

    # Experiment 8: Experiment 4 w/ k=3
    expected_top_slices_8 = np.array(
        [
            [1.0, 1.0, None, None, None, None],
            [1.0, None, 3.0, None, None, None],
            [1.0, None, None, None, None, None],
            [None, 1.0, None, None, None, None],
        ]
    )
    experiment_8 = Experiment(X_4, errors_4, expected_top_slices_8, k=3)

    # Experiment 9: Experiment 1 w/ float label
    np.random.seed(9)
    errors_9 = (
        np.concatenate([np.random.randint(1, 61, n_small), np.random.randint(41, 101, n_small)])
        / 100
    )
    expected_top_slices_9 = np.array([[2.0, None, None], [2.0, 1.0, None]])
    experiment_9 = Experiment(X_1, errors_9, expected_top_slices_9)

    # Experiment 10: Bigger dataset
    np.random.seed(10)
    n = 10000
    X_10 = np.array(
        [
            [1] * n + [2] * (n // 2) + [1] * (n // 2),
            [1] * n + [1] * (n // 2) + [2] * (n // 2),
            np.random.randint(1, 4, size=2 * n),
            np.random.randint(1, 5, size=2 * n),
            np.random.randint(1, 7, size=2 * n),
            np.concatenate([np.arange(1, 1 + 2 * n // 40)] * 40),
        ]
    ).T
    errors_10 = np.array([1] * n + [0] * n)
    expected_top_slices_10 = np.array(
        [
            [1.0, 1.0, None, None, None, None],
            [1.0, None, None, None, None, None],
            [None, 1.0, None, None, None, None],
        ]
    )
    experiment_10 = Experiment(X_10, errors_10, expected_top_slices_10)

    # Experiment 11: max_l=3
    X_11 = np.array(
        [
            [1] * 6 + [2] * 2 + [1] * 4,
            [1] * 8 + [2] * 2 + [1] * 2,
            [1] * 10 + [2] * 2,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ]
    ).T
    errors_11 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    expected_top_slices_11 = np.array(
        [[1.0, 1.0, 1.0, None], [None, 1.0, 1.0, None], [1, None, 1, None], [1, 1, None, None]]
    )
    experiment_11 = Experiment(X_11, errors_11, expected_top_slices_11, max_l=3)

    # Experiment 12: max_l=4
    X_12 = np.array(
        [
            [1] * 8 + [2] * 2 + [1] * 6,
            [1] * 10 + [2] * 2 + [1] * 4,
            [1] * 12 + [2] * 2 + [1] * 2,
            [1] * 14 + [2] * 2,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ]
    ).T
    errors_12 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    expected_top_slices_12 = np.array(
        [
            [1, 1, 1, 1, None],
            [None, 1, 1, 1, None],
            [1, None, 1, 1, None],
            [1, 1, None, 1, None],
            [1, 1, 1, None, None],
        ]
    )
    experiment_12 = Experiment(X_12, errors_12, expected_top_slices_12, max_l=4)

    # Experiment 13: mixed types
    X_13 = np.array(
        [
            [1, 1, 1, 1, 1, 1, 2, 2],
            ["a", "a", "a", "a", "b", "b", "a", "a"],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [3, 3, 3, 1, 3, 1, 2, 1],
        ],
        dtype=object,
    ).T
    errors_13 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    expected_top_slices_13 = np.array([[1, "a", None, None], [None, "a", None, 3]])
    experiment_13 = Experiment(X_13, errors_13, expected_top_slices_13)

    # Experiment 14: Experiment 4 w/ min_sup=10
    expected_top_slices_14 = np.array(
        [
            [1.0, 1.0, None, None, None, None],
            [1.0, None, 3.0, None, None, None],
        ]
    )
    experiment_14 = Experiment(X_4, errors_4, expected_top_slices_14, min_sup=10)

    # Experiment 15: Experiment 4 w/ alpha=0.5
    expected_top_slices_15 = np.empty((0, 6))
    experiment_15 = Experiment(X_4, errors_4, expected_top_slices_15, alpha=0.5)

    # Experiment 16: Experiment with missing parent pruning
    X_16 = np.array(
        [
            ["g", "g", "g", "g", "g", "g", "a", "a", "a", "a", "a", "a", "a", "a", "a"],
            ["b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "f", "f", "f"],
            ["h", "h", "h", "h", "c", "c", "c", "c", "h", "h", "h", "h", "h", "h", "h"],
            ["d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "e", "e", "e"],
        ],
        dtype=object,
    ).T
    errors_16 = np.array([0] * 6 + [1] * 6 + [0] * 3)
    expected_top_slices_16 = np.empty((0, 4))
    experiment_16 = Experiment(
        X_16, errors_16, expected_top_slices_16, alpha=0.01, max_l=3, min_sup=7
    )

    # Experiment 17: Experiment 4 w/ min_sup=0.1
    expected_top_slices_17 = np.array(
        [
            [1.0, 1.0, None, None, None, None],
            [1.0, None, None, None, None, None],
            [None, 1.0, None, None, None, None]
        ]
    )
    experiment_17 = Experiment(X_4, errors_4, expected_top_slices_17, min_sup=0.5)

    return {
        "experiment_1": experiment_1,
        "experiment_2": experiment_2,
        "experiment_3": experiment_3,
        "experiment_4": experiment_4,
        "experiment_5": experiment_5,
        "experiment_6": experiment_6,
        "experiment_7": experiment_7,
        "experiment_8": experiment_8,
        "experiment_9": experiment_9,
        "experiment_10": experiment_10,
        "experiment_11": experiment_11,
        "experiment_12": experiment_12,
        "experiment_13": experiment_13,
        "experiment_14": experiment_14,
        "experiment_15": experiment_15,
        "experiment_16": experiment_16,
        "experiment_17": experiment_17
    }
