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

    assert np.array_equal(
        computed.toarray(), basic_test_data["slices"].toarray()
    )


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

    assert np.array_equal(computed_tk.toarray(), expected_tk.toarray())
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
    assert np.array_equal(
        computed_slices.toarray(), basic_test_data["slices"].toarray()
    )
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
    assert np.array_equal(
        computed.toarray(), basic_test_data["candidates"].toarray()
    )


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
    ).toarray()

    assert np.array_equal(computed, expected)


def test_search_slices(benchmark, basic_test_data):
    """Test _search_slices method."""
    benchmark(
        basic_test_data["slicefinder_model"]._search_slices,
        basic_test_data["X"],
        basic_test_data["errors"],
    )
    computed_top_k_slices = basic_test_data["slicefinder_model"].top_slices_
    computed_top_k_slices_statistics_ = basic_test_data[
        "slicefinder_model"
    ].top_slices_statistics_
    expected_top_k_slices = np.array(
        [
            [1, 1, None, None],
            [None, 1, None, 3],
        ]
    )
    expected_top_k_slices_statistics = [
        {
            "slice_score": 0.8999999999999999,
            "sum_slice_error": 4.0,
            "max_slice_error": 1.0,
            "slice_size": 4.0,
            "slice_average_error": 1.0,
        },
        {
            "slice_score": 0.8666666666666666,
            "sum_slice_error": 3.0,
            "max_slice_error": 1.0,
            "slice_size": 3.0,
            "slice_average_error": 1.0,
        },
    ]
    assert np.array_equal(computed_top_k_slices, expected_top_k_slices)
    assert (
        computed_top_k_slices_statistics_ == expected_top_k_slices_statistics
    )


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
    computed_top_k_slices_statistics = slicefinder_model.top_slices_statistics_
    assert np.array_equal(
        computed_top_k_slices, experiment.expected_top_k_slices
    )
    assert (
        computed_top_k_slices_statistics
        == experiment.expected_top_k_slices_statistics
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


def test_get_slice_with_nan(benchmark, basic_test_data):
    """Test get_slice method with NaN values in the dataset."""
    basic_test_data["slicefinder_model"].fit(
        basic_test_data["X"], basic_test_data["errors"]
    )

    dataset_nan_case = np.array(
        [
            [np.nan, 1, 1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2, 1, 1],
            [1, 2, 3, np.nan, 5, 6, 7, 8],
            [3, 3, 3, 1, 3, 1, 2, 1],
        ]
    ).T
    computed_slice_nan_case = benchmark(
        basic_test_data["slicefinder_model"].get_slice,
        dataset_nan_case,
        0,
    )
    expected_slice_nan_case = np.array(
        [[1, 1, 2, 3], [1, 1, 3, 3], [1, 1, np.nan, 1]]
    )
    assert np.array_equal(
        computed_slice_nan_case, expected_slice_nan_case, equal_nan=True
    )


class TestParameterValidation:
    """Test parameter validation in Slicefinder."""

    def test_invalid_alpha_zero(self, basic_test_data):
        """Test that alpha=0 raises ValueError."""
        model = slicefinder.Slicefinder(alpha=0)
        with pytest.raises(ValueError, match="Invalid 'alpha' parameter"):
            model.fit(basic_test_data["X"], basic_test_data["errors"])

    def test_invalid_alpha_negative(self, basic_test_data):
        """Test that negative alpha raises ValueError."""
        model = slicefinder.Slicefinder(alpha=-0.5)
        with pytest.raises(ValueError, match="Invalid 'alpha' parameter"):
            model.fit(basic_test_data["X"], basic_test_data["errors"])

    def test_invalid_alpha_greater_than_one(self, basic_test_data):
        """Test that alpha > 1 raises ValueError."""
        model = slicefinder.Slicefinder(alpha=1.5)
        with pytest.raises(ValueError, match="Invalid 'alpha' parameter"):
            model.fit(basic_test_data["X"], basic_test_data["errors"])

    def test_valid_alpha_one(self, basic_test_data):
        """Test that alpha=1 is valid."""
        model = slicefinder.Slicefinder(alpha=1.0)
        model.fit(basic_test_data["X"], basic_test_data["errors"])
        assert model.top_slices_ is not None

    def test_invalid_k_zero(self, basic_test_data):
        """Test that k=0 raises ValueError."""
        model = slicefinder.Slicefinder(k=0)
        with pytest.raises(ValueError, match="Invalid 'k' parameter"):
            model.fit(basic_test_data["X"], basic_test_data["errors"])

    def test_invalid_k_negative(self, basic_test_data):
        """Test that negative k raises ValueError."""
        model = slicefinder.Slicefinder(k=-1)
        with pytest.raises(ValueError, match="Invalid 'k' parameter"):
            model.fit(basic_test_data["X"], basic_test_data["errors"])

    def test_invalid_max_l_zero(self, basic_test_data):
        """Test that max_l=0 raises ValueError."""
        model = slicefinder.Slicefinder(max_l=0)
        with pytest.raises(ValueError, match="Invalid 'max_l' parameter"):
            model.fit(basic_test_data["X"], basic_test_data["errors"])

    def test_invalid_max_l_negative(self, basic_test_data):
        """Test that negative max_l raises ValueError."""
        model = slicefinder.Slicefinder(max_l=-1)
        with pytest.raises(ValueError, match="Invalid 'max_l' parameter"):
            model.fit(basic_test_data["X"], basic_test_data["errors"])

    def test_invalid_min_sup_negative(self, basic_test_data):
        """Test that negative min_sup raises ValueError."""
        model = slicefinder.Slicefinder(min_sup=-1)
        with pytest.raises(ValueError, match="Invalid 'min_sup' parameter"):
            model.fit(basic_test_data["X"], basic_test_data["errors"])

    def test_invalid_min_sup_float_one(self, basic_test_data):
        """Test that min_sup=1.0 (float) raises ValueError."""
        model = slicefinder.Slicefinder(min_sup=1.0)
        with pytest.raises(ValueError, match="Invalid 'min_sup' parameter"):
            model.fit(basic_test_data["X"], basic_test_data["errors"])

    def test_invalid_min_sup_float_greater_than_one(self, basic_test_data):
        """Test that min_sup > 1.0 (float) raises ValueError."""
        model = slicefinder.Slicefinder(min_sup=1.5)
        with pytest.raises(ValueError, match="Invalid 'min_sup' parameter"):
            model.fit(basic_test_data["X"], basic_test_data["errors"])

    def test_valid_min_sup_fraction(self, basic_test_data):
        """Test that min_sup as fraction (0 < min_sup < 1) is valid."""
        model = slicefinder.Slicefinder(min_sup=0.5)
        model.fit(basic_test_data["X"], basic_test_data["errors"])
        assert model.top_slices_ is not None

    def test_valid_min_sup_integer(self, basic_test_data):
        """Test that min_sup as integer is valid."""
        model = slicefinder.Slicefinder(min_sup=2)
        model.fit(basic_test_data["X"], basic_test_data["errors"])
        assert model.top_slices_ is not None


class TestMinSupMutation:
    """Test that min_sup parameter is not mutated across multiple fit calls."""

    def test_min_sup_not_mutated_with_fraction(self, basic_test_data):
        """Test that min_sup is preserved when using fractional value."""
        original_min_sup = 0.5
        model = slicefinder.Slicefinder(min_sup=original_min_sup)

        model.fit(basic_test_data["X"], basic_test_data["errors"])
        assert model.min_sup == original_min_sup

        model.fit(basic_test_data["X"], basic_test_data["errors"])
        assert model.min_sup == original_min_sup

    def test_min_sup_not_mutated_with_integer(self, basic_test_data):
        """Test that min_sup is preserved when using integer value."""
        original_min_sup = 2
        model = slicefinder.Slicefinder(min_sup=original_min_sup)

        model.fit(basic_test_data["X"], basic_test_data["errors"])
        assert model.min_sup == original_min_sup

        model.fit(basic_test_data["X"], basic_test_data["errors"])
        assert model.min_sup == original_min_sup


class TestDummifyValidation:
    """Test _dummify method validation."""

    def test_dummify_raises_on_zero_modality(self, basic_test_data):
        """Test that _dummify raises ValueError when array contains 0."""
        array_with_zero = np.array([0, 1, 2, 3])
        with pytest.raises(ValueError, match="Modality 0 is not expected"):
            basic_test_data["slicefinder_model"]._dummify(array_with_zero, 10)
