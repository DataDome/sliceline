"""
The test_slicefinder module tests slicefinder class and methods.
Tests are run on different Experiments.
"""
import numpy as np
import pytest
from scipy import sparse as sp

from sliceline import slicefinder


def test_dummify(benchmark, basic_test_data):
    """Test _dummify method."""
    array = np.array([1, 3, 5, 6, 7, 8, 13, 15])
    computed = benchmark(
        basic_test_data["slicefinder_model"]._dummify,
        array,
        basic_test_data["n_col_x_encoded"],
    )

    assert np.array_equal(computed.A, basic_test_data["slices"].A)


def test_maintain_top_k(benchmark, basic_test_data):
    """Test _maintain_top_k method."""
    statistics = np.array(
        [
            [0.8666666666666666, 3, 1, 3],
            [0.42499999999999993, 3, 1, 4],
            [0.8999999999999999, 4, 1, 4],
        ]
    )
    top_k_slices = sp.csr_matrix(
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    computed_tk, computed_tkc = benchmark(
        basic_test_data["slicefinder_model"]._maintain_top_k,
        basic_test_data["candidates"],
        statistics,
        top_k_slices,
        basic_test_data["top_k_statistics"],
    )
    expected_tk = sp.csr_matrix(
        [
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    expected_tkc = np.array(
        [
            [0.8999999999999999, 4, 1, 4],
            [0.8666666666666666, 3, 1, 3],
        ]
    )

    assert np.array_equal(computed_tk.A, expected_tk.A)
    assert np.array_equal(computed_tkc, expected_tkc)


def test_score_ub(benchmark, basic_test_data):
    """Test _score_ub method."""
    slice_sizes_ub = np.array(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            3,
            4,
            1,
            1,
            1,
            1,
            3,
            4,
            1,
            1,
            1,
            1,
            6,
        ]
    )
    slice_errors_ub = np.array(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            3,
            1,
            1,
            1,
            1,
            1,
            3,
            1,
            1,
            1,
            1,
            4,
        ]
    )
    max_slice_errors_ub = np.array(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
    )

    computed = benchmark(
        basic_test_data["slicefinder_model"]._score_ub,
        slice_sizes_ub,
        slice_errors_ub,
        max_slice_errors_ub,
        basic_test_data["n_col_x_encoded"],
    )
    expected = np.array(
        [
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.7499999999999997,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.7499999999999997,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.24999999999999933,
            0.8124999999999998,
        ]
    )
    assert np.array_equal(computed, expected)


def test_analyse_top_k(benchmark, basic_test_data):
    """Test _analyse_top_k method."""
    top_k_statistics = np.array(
        [
            [0.8999999999999999, 4, 1, 4],
            [0.8666666666666666, 3, 1, 3],
        ]
    )

    computed_maxsc, computed_minsc = benchmark(
        basic_test_data["slicefinder_model"]._analyse_top_k, top_k_statistics
    )
    expected_maxsc, expected_minsc = 0.8999999999999999, 0.8666666666666666
    assert computed_maxsc == expected_maxsc
    assert computed_minsc == expected_minsc


def test_score(benchmark, basic_test_data):
    """Test _score method."""
    slice_sizes = np.array([3, 4, 4])
    slice_errors = np.array([3, 3, 4])
    n_row_x_encoded = 8

    computed = benchmark(
        basic_test_data["slicefinder_model"]._score,
        slice_sizes,
        slice_errors,
        n_row_x_encoded,
    )
    expected = np.array(
        [
            0.8666666666666666,
            0.42499999999999993,
            0.8999999999999999,
        ]
    )
    assert np.array_equal(computed, expected)


def test_eval_slice(benchmark, basic_test_data):
    """Test _eval_slice method."""
    computed = benchmark(
        basic_test_data["slicefinder_model"]._eval_slice,
        basic_test_data["X_encoded"],
        basic_test_data["errors"],
        basic_test_data["candidates"],
        basic_test_data["level"],
    )
    expected = np.array(
        [
            [0.8666666666666666, 3, 1, 3],
            [0.42499999999999993, 3, 1, 4],
            [0.8999999999999999, 4, 1, 4],
        ]
    )
    assert np.array_equal(computed, expected)


def test_create_and_score_basic_slices(benchmark, basic_test_data):
    """Test _create_and_score_basic_slices method."""
    computed_slices, computed_statistics = benchmark(
        basic_test_data["slicefinder_model"]._create_and_score_basic_slices,
        basic_test_data["X_encoded"],
        basic_test_data["n_col_x_encoded"],
        basic_test_data["errors"],
    )

    expected_r = np.array(
        [
            [0.29999999999999993, 4, 1, 6],
            [0.29999999999999993, 4, 1, 6],
            [0.5999999999999996, 1, 1, 1],
            [0.5999999999999996, 1, 1, 1],
            [0.5999999999999996, 1, 1, 1],
            [0.5999999999999996, 1, 1, 1],
            [-0.40000000000000013, 1, 1, 3],
            [0.42499999999999993, 3, 1, 4],
        ]
    )
    assert np.array_equal(computed_slices.A, basic_test_data["slices"].A)
    assert np.array_equal(computed_statistics, expected_r)


def test_get_pair_candidates(benchmark, basic_test_data):
    """Test _get_pair_candidates method."""
    statistics = np.array(
        [
            [0.29999999999999993, 4, 1, 6],
            [0.29999999999999993, 4, 1, 6],
            [0.5999999999999996, 1, 1, 1],
            [0.5999999999999996, 1, 1, 1],
            [0.5999999999999996, 1, 1, 1],
            [0.5999999999999996, 1, 1, 1],
            [-0.40000000000000013, 1, 1, 3],
            [0.42499999999999993, 3, 1, 4],
        ]
    )

    computed = benchmark(
        basic_test_data["slicefinder_model"]._get_pair_candidates,
        basic_test_data["slices"],
        statistics,
        basic_test_data["top_k_statistics"],
        basic_test_data["level"],
        basic_test_data["n_col_x_encoded"],
        basic_test_data["feature_domains"],
        basic_test_data["feature_offset_start"],
        basic_test_data["feature_offset_end"],
    )
    assert np.array_equal(computed.A, basic_test_data["candidates"].A)


def test_get_pair_candidates_with_missing_parents_pruning(
    benchmark, basic_test_data
):
    """Test _get_pair_candidates where missing parents are present in pruning."""
    slices = sp.csr_matrix(
        [
            [False, False, False, False, False, True, True, False],
            [False, False, True, False, False, False, True, False],
            [False, False, True, False, False, True, False, False],
            [True, False, False, False, False, True, False, False],
        ]
    )

    statistics = np.array(
        [
            [-0.86375, 4.0, 1.0, 8.0],
            [-0.245, 6.0, 1.0, 12.0],
            [-0.86375, 4.0, 1.0, 8.0],
            [-1.12714286, 4.0, 1.0, 7.0],
        ]
    )

    slicefinder_model_parents_pruning = slicefinder.Slicefinder(
        alpha=0.5,
        k=basic_test_data["k"],
        max_l=basic_test_data["max_l"],
        min_sup=7,
        verbose=basic_test_data["verbose"],
    )
    slicefinder_model_parents_pruning.average_error_ = 0.4

    expected = np.array(
        [[False, False, True, False, False, True, True, False]]
    )

    computed = benchmark(
        slicefinder_model_parents_pruning._get_pair_candidates,
        slices,
        statistics,
        top_k_statistics=np.array([]),
        level=3,
        n_col_x_encoded=8,
        feature_domains=np.array([2, 2, 2, 2]),
        feature_offset_start=np.array([0, 2, 4, 6]),
        feature_offset_end=np.array([2, 4, 6, 8]),
    ).A

    assert np.array_equal(computed, expected)


def test_search_slices(benchmark, basic_test_data):
    """Test _search_slices method."""
    benchmark(
        basic_test_data["slicefinder_model"]._search_slices,
        basic_test_data["X"],
        basic_test_data["errors"],
    )
    computed_top_k_slices = basic_test_data["slicefinder_model"].top_slices_
    expected_top_k_slices = np.array(
        [
            [1, 1, None, None],
            [None, 1, None, 3],
        ]
    )
    assert np.array_equal(computed_top_k_slices, expected_top_k_slices)


@pytest.mark.parametrize(
    "experiment_name",
    [
        "experiment_1",
        "experiment_2",
        "experiment_3",
        "experiment_4",
        "experiment_5",
        "experiment_6",
        "experiment_7",
        "experiment_8",
        "experiment_9",
        "experiment_10",
        "experiment_11",
        "experiment_12",
        "experiment_13",
        "experiment_14",
        "experiment_15",
        "experiment_16",
        "experiment_17",
    ],
)
def test_experiments(benchmark, experiments, experiment_name):
    """Test fit method on different experiments."""
    experiment = experiments[experiment_name]

    slicefinder_model = slicefinder.Slicefinder(
        alpha=experiment.alpha,
        k=experiment.k,
        max_l=experiment.max_l,
        min_sup=experiment.min_sup,
        verbose=experiment.verbose,
    )
    benchmark(
        slicefinder_model.fit,
        experiment.input_dataset,
        experiment.input_errors,
    )
    computed_top_k_slices = slicefinder_model.top_slices_
    assert np.array_equal(
        computed_top_k_slices, experiment.expected_top_k_slices
    )


def test_transform(benchmark, basic_test_data):
    """Test transform method."""
    computed = benchmark(
        basic_test_data["slicefinder_model"].fit_transform,
        basic_test_data["X"],
        basic_test_data["errors"],
    )
    expected = np.array(
        [[1, 1], [1, 1], [1, 1], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    )
    assert np.array_equal(computed, expected)


def test_get_slice(benchmark, basic_test_data):
    """Test get_slice method."""
    basic_test_data["slicefinder_model"].fit(
        basic_test_data["X"], basic_test_data["errors"]
    )
    computed_slice = benchmark(
        basic_test_data["slicefinder_model"].get_slice,
        basic_test_data["X"],
        0,
    )
    expected_slice = np.array(
        [[1, 1, 1, 3], [1, 1, 2, 3], [1, 1, 3, 3], [1, 1, 4, 1]]
    )
    assert np.array_equal(computed_slice, expected_slice)
