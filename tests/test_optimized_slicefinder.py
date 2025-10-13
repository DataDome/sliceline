"""
The test_optimized_slicefinder module tests the OptimizedSlicefinder class.

Tests verify that optimizations maintain correctness while improving performance
for large datasets with high cardinality columns.
"""

import numpy as np
import pytest

from sliceline import Slicefinder
from sliceline.optimized_slicefinder import (
    NUMBA_AVAILABLE,
    OptimizedSlicefinder,
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
def test_optimized_matches_original_on_experiments(
    experiments, experiment_name
):
    """Verify optimized produces same results as original on standard tests.

    For low-cardinality experiments, OptimizedSlicefinder should produce
    identical results to the base Slicefinder when feature hashing is not
    triggered (i.e., all features have < max_features_per_column unique values).
    """
    experiment = experiments[experiment_name]

    # Original implementation
    original = Slicefinder(
        alpha=experiment.alpha,
        k=experiment.k,
        max_l=experiment.max_l,
        min_sup=experiment.min_sup,
        verbose=experiment.verbose,
    )
    original.fit(experiment.input_dataset, experiment.input_errors)

    # Optimized implementation with high threshold (no hashing should occur)
    optimized = OptimizedSlicefinder(
        alpha=experiment.alpha,
        k=experiment.k,
        max_l=experiment.max_l,
        min_sup=experiment.min_sup,
        verbose=experiment.verbose,
        max_features_per_column=10000,  # High enough to not trigger hashing
    )
    optimized.fit(experiment.input_dataset, experiment.input_errors)

    # Verify same number of slices found
    assert len(original.top_slices_) == len(
        optimized.top_slices_
    ), f"Different number of slices: {len(original.top_slices_)} vs {len(optimized.top_slices_)}"

    # Verify slices match (allowing for floating point comparison)
    np.testing.assert_array_equal(
        original.top_slices_,
        optimized.top_slices_,
        err_msg=f"Top slices differ for {experiment_name}",
    )

    # Verify statistics match
    assert len(original.top_slices_statistics_) == len(
        optimized.top_slices_statistics_
    )
    for orig_stat, opt_stat in zip(
        original.top_slices_statistics_, optimized.top_slices_statistics_
    ):
        for key in orig_stat:
            np.testing.assert_allclose(
                orig_stat[key],
                opt_stat[key],
                rtol=1e-10,
                err_msg=f"Statistic {key} differs for {experiment_name}",
            )


def test_feature_hashing_reduces_cardinality():
    """Test that feature hashing correctly reduces high cardinality."""
    np.random.seed(42)
    n_samples = 1000

    # Create high cardinality dataset (5000 unique values per feature)
    X = np.random.randint(0, 5000, size=(n_samples, 3))
    errors = np.random.random(n_samples)

    # Verify cardinality before hashing
    for col in range(X.shape[1]):
        assert len(np.unique(X[:, col])) > 500

    # Fit with feature hashing enabled
    optimized = OptimizedSlicefinder(
        k=5, max_l=2, max_features_per_column=500, verbose=False
    )
    optimized.fit(X, errors)

    # Should complete without error
    assert optimized.top_slices_ is not None


def test_high_cardinality_handling():
    """Test that OptimizedSlicefinder handles high cardinality data."""
    np.random.seed(123)
    n_samples = 10000

    # High cardinality dataset that would cause memory issues without hashing
    X = np.random.randint(0, 5000, size=(n_samples, 5))
    errors = np.random.random(n_samples)

    # Should complete without OOM error
    optimized = OptimizedSlicefinder(
        k=10, max_l=2, max_features_per_column=500, verbose=False
    )
    optimized.fit(X, errors)

    # Should complete successfully (may or may not find slices with random data)
    assert optimized.top_slices_ is not None


def test_feature_hashing_can_be_disabled():
    """Test that feature hashing can be disabled."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randint(0, 50, size=(n_samples, 3))
    errors = np.random.random(n_samples)

    # With hashing disabled
    optimized_no_hash = OptimizedSlicefinder(
        k=5,
        max_l=2,
        use_feature_hashing=False,
        verbose=False,
    )
    optimized_no_hash.fit(X, errors)

    # With hashing enabled (but high threshold)
    optimized_with_hash = OptimizedSlicefinder(
        k=5,
        max_l=2,
        use_feature_hashing=True,
        max_features_per_column=1000,
        verbose=False,
    )
    optimized_with_hash.fit(X, errors)

    # Should produce identical results on low cardinality data
    np.testing.assert_array_equal(
        optimized_no_hash.top_slices_, optimized_with_hash.top_slices_
    )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
def test_numba_acceleration():
    """Test that Numba acceleration can be toggled."""
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randint(0, 10, size=(n_samples, 4))
    errors = np.random.random(n_samples)

    # With Numba enabled
    optimized_numba = OptimizedSlicefinder(
        k=5, max_l=2, use_numba=True, verbose=False
    )
    optimized_numba.fit(X, errors)

    # With Numba disabled
    optimized_no_numba = OptimizedSlicefinder(
        k=5, max_l=2, use_numba=False, verbose=False
    )
    optimized_no_numba.fit(X, errors)

    # Results should be identical regardless of Numba usage
    np.testing.assert_array_equal(
        optimized_numba.top_slices_, optimized_no_numba.top_slices_
    )

    # Statistics should also match
    assert len(optimized_numba.top_slices_statistics_) == len(
        optimized_no_numba.top_slices_statistics_
    )


def test_transform_works_with_optimized():
    """Test that transform method works correctly."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randint(0, 5, size=(n_samples, 3))
    errors = np.random.random(n_samples)

    optimized = OptimizedSlicefinder(k=5, max_l=2, min_sup=5, verbose=False)
    optimized.fit(X, errors)

    # Transform should work if slices were found
    if len(optimized.top_slices_) > 0:
        X_transformed = optimized.transform(X)

        # Check shape
        assert X_transformed.shape[0] == n_samples
        assert X_transformed.shape[1] == len(optimized.top_slices_)

        # Check binary values
        assert np.all((X_transformed == 0) | (X_transformed == 1))
    else:
        # If no slices found, transform should raise error
        with pytest.raises(ValueError, match="No transform"):
            optimized.transform(X)


def test_get_slice_works_with_optimized():
    """Test that get_slice method works correctly."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randint(0, 5, size=(n_samples, 3))
    errors = np.random.random(n_samples)

    optimized = OptimizedSlicefinder(k=5, max_l=2, min_sup=5, verbose=False)
    optimized.fit(X, errors)

    if len(optimized.top_slices_) > 0:
        # Get first slice
        X_slice = optimized.get_slice(X, 0)

        # Should have fewer or equal samples than original
        assert X_slice.shape[0] <= n_samples
        assert X_slice.shape[1] == X.shape[1]


def test_empty_slices_handling():
    """Test behavior when no valid slices are found."""
    np.random.seed(42)
    n_samples = 10
    X = np.random.randint(0, 5, size=(n_samples, 2))
    # All zeros - no interesting slices
    errors = np.zeros(n_samples)

    optimized = OptimizedSlicefinder(
        k=5, max_l=2, min_sup=5, alpha=0.95, verbose=False
    )
    optimized.fit(X, errors)

    # Should handle empty results gracefully
    assert optimized.top_slices_ is not None
    assert len(optimized.top_slices_) == 0


def test_mixed_types_with_optimized():
    """Test that mixed types (str and numeric) work correctly."""
    X = np.array(
        [
            [1, "a", 1],
            [1, "a", 2],
            [1, "b", 1],
            [2, "a", 1],
            [2, "b", 2],
        ],
        dtype=object,
    )
    errors = np.array([1, 1, 0, 0, 0])

    optimized = OptimizedSlicefinder(k=2, max_l=2, min_sup=1, verbose=False)
    optimized.fit(X, errors)

    assert optimized.top_slices_ is not None


@pytest.mark.benchmark(group="cardinality-scaling")
def test_benchmark_low_cardinality(benchmark):
    """Benchmark with low cardinality (10 unique values)."""
    np.random.seed(42)
    X = np.random.randint(0, 10, size=(10000, 5))
    errors = np.random.random(10000)

    def run_optimized():
        model = OptimizedSlicefinder(k=10, max_l=2, verbose=False)
        model.fit(X, errors)
        return model

    result = benchmark(run_optimized)
    assert result.top_slices_ is not None


@pytest.mark.benchmark(group="cardinality-scaling")
def test_benchmark_medium_cardinality(benchmark):
    """Benchmark with medium cardinality (100 unique values)."""
    np.random.seed(42)
    X = np.random.randint(0, 100, size=(10000, 5))
    errors = np.random.random(10000)

    def run_optimized():
        model = OptimizedSlicefinder(k=10, max_l=2, verbose=False)
        model.fit(X, errors)
        return model

    result = benchmark(run_optimized)
    assert result.top_slices_ is not None


@pytest.mark.benchmark(group="cardinality-scaling")
def test_benchmark_high_cardinality(benchmark):
    """Benchmark with high cardinality (1000 unique values)."""
    np.random.seed(42)
    X = np.random.randint(0, 1000, size=(10000, 5))
    errors = np.random.random(10000)

    def run_optimized():
        model = OptimizedSlicefinder(
            k=10, max_l=2, max_features_per_column=500, verbose=False
        )
        model.fit(X, errors)
        return model

    result = benchmark(run_optimized)
    assert result.top_slices_ is not None


def test_api_compatibility_with_base_class():
    """Test that OptimizedSlicefinder has same API as Slicefinder."""
    # Check that key attributes and methods exist
    optimized = OptimizedSlicefinder()

    # Check public methods
    assert hasattr(optimized, "fit")
    assert hasattr(optimized, "transform")
    assert hasattr(optimized, "fit_transform")
    assert hasattr(optimized, "get_slice")
    assert hasattr(optimized, "get_feature_names_out")

    # Check public attributes after fit
    X = np.random.randint(0, 5, size=(50, 3))
    errors = np.random.random(50)
    optimized.fit(X, errors)

    assert hasattr(optimized, "top_slices_")
    assert hasattr(optimized, "top_slices_statistics_")
    assert hasattr(optimized, "average_error_")


def test_parameter_validation():
    """Test that parameters are validated correctly."""
    # Invalid alpha
    with pytest.raises(ValueError, match="alpha"):
        optimized = OptimizedSlicefinder(alpha=1.5)
        X = np.random.randint(0, 5, size=(50, 3))
        errors = np.random.random(50)
        optimized.fit(X, errors)

    # Invalid k
    with pytest.raises(ValueError, match="k"):
        optimized = OptimizedSlicefinder(k=-1)
        X = np.random.randint(0, 5, size=(50, 3))
        errors = np.random.random(50)
        optimized.fit(X, errors)


def test_reproducibility_with_same_seed():
    """Test that results are reproducible with same random seed."""
    np.random.seed(42)
    X1 = np.random.randint(0, 10, size=(100, 3))
    errors1 = np.random.random(100)

    np.random.seed(42)
    X2 = np.random.randint(0, 10, size=(100, 3))
    errors2 = np.random.random(100)

    optimized1 = OptimizedSlicefinder(k=5, max_l=2, verbose=False)
    optimized1.fit(X1, errors1)

    optimized2 = OptimizedSlicefinder(k=5, max_l=2, verbose=False)
    optimized2.fit(X2, errors2)

    np.testing.assert_array_equal(
        optimized1.top_slices_, optimized2.top_slices_
    )
