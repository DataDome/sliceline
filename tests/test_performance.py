"""Performance regression tests for Sliceline using pytest-benchmark.

This module tests for performance regressions in the Sliceline codebase.
It differs from benchmarks/benchmarks.py in purpose:

- tests/test_performance.py: Automated regression testing integrated with CI/CD.
  These tests ensure performance does not degrade across commits. They are designed
  to be fast, repeatable, and can be compared across runs using pytest-benchmark.

- benchmarks/benchmarks.py: Manual profiling scripts for detailed performance analysis.
  These scripts produce JSON results files and detailed console output for analyzing
  performance characteristics across different cardinality levels and dataset sizes.

Run with: uv run pytest tests/test_performance.py -v --benchmark-only
"""

import numpy as np
import pytest
from scipy import sparse as sp

from sliceline import Slicefinder


def _create_correlated_errors(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """Create errors that correlate strongly with feature patterns.

    This creates a clear pattern where the first feature value of 1
    correlates with high errors, making slicefinding tests reliable.
    """
    np.random.seed(seed)
    n_samples = X.shape[0]

    # Create strongly differentiated error patterns
    mask_high_error = X[:, 0] == 1
    errors = np.zeros(n_samples)
    errors[mask_high_error] = np.random.uniform(
        0.7, 1.0, mask_high_error.sum()
    )
    errors[~mask_high_error] = np.random.uniform(
        0.0, 0.3, (~mask_high_error).sum()
    )

    return errors


@pytest.fixture
def small_dataset():
    """Small dataset for quick benchmark iterations."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    X = np.random.randint(1, 5, size=(n_samples, n_features))
    errors = _create_correlated_errors(X)
    return X, errors


@pytest.fixture
def medium_dataset():
    """Medium dataset for realistic performance testing."""
    np.random.seed(42)
    n_samples = 10000
    n_features = 20
    X = np.random.randint(1, 10, size=(n_samples, n_features))
    errors = _create_correlated_errors(X)
    return X, errors


@pytest.fixture
def large_dataset():
    """Large dataset for stress testing."""
    np.random.seed(42)
    n_samples = 50000
    n_features = 30
    X = np.random.randint(1, 15, size=(n_samples, n_features))
    errors = _create_correlated_errors(X)
    return X, errors


class TestSlicefinderPerformance:
    """Benchmark tests for Slicefinder class."""

    def test_fit_small(self, benchmark, small_dataset):
        """Benchmark fit() on small dataset (1K samples)."""
        X, errors = small_dataset
        sf = Slicefinder(k=5, max_l=3, min_sup=10, verbose=False)
        result = benchmark(sf.fit, X, errors)
        assert result.top_slices_ is not None

    def test_fit_medium(self, benchmark, medium_dataset):
        """Benchmark fit() on medium dataset (10K samples)."""
        X, errors = medium_dataset
        sf = Slicefinder(k=5, max_l=3, min_sup=100, verbose=False)
        result = benchmark(sf.fit, X, errors)
        assert result.top_slices_ is not None

    def test_fit_large(self, benchmark, large_dataset):
        """Benchmark fit() on large dataset (50K samples)."""
        X, errors = large_dataset
        sf = Slicefinder(k=5, max_l=3, min_sup=500, verbose=False)
        result = benchmark(sf.fit, X, errors)
        assert result.top_slices_ is not None

    def test_transform_small(self, benchmark, small_dataset):
        """Benchmark transform() on small dataset."""
        X, errors = small_dataset
        sf = Slicefinder(k=5, max_l=3, min_sup=10, alpha=0.9, verbose=False)
        sf.fit(X, errors)
        if len(sf.top_slices_) == 0:
            pytest.skip("No slices found for transform benchmark")
        result = benchmark(sf.transform, X)
        assert result.shape[0] == X.shape[0]

    def test_transform_medium(self, benchmark, medium_dataset):
        """Benchmark transform() on medium dataset."""
        X, errors = medium_dataset
        sf = Slicefinder(k=5, max_l=3, min_sup=100, alpha=0.9, verbose=False)
        sf.fit(X, errors)
        if len(sf.top_slices_) == 0:
            pytest.skip("No slices found for transform benchmark")
        result = benchmark(sf.transform, X)
        assert result.shape[0] == X.shape[0]


class TestInternalMethodsPerformance:
    """Benchmark tests for internal methods."""

    @pytest.fixture
    def fitted_slicefinder(self, medium_dataset):
        """Pre-fitted Slicefinder for internal method testing."""
        X, errors = medium_dataset
        sf = Slicefinder(k=5, max_l=3, min_sup=100, verbose=False)
        sf.fit(X, errors)
        return sf

    def test_dummify_performance(self, benchmark):
        """Benchmark _dummify() method for sparse matrix construction."""
        array = np.random.randint(1, 100, size=10000)
        n_col = 100

        def run_dummify():
            return Slicefinder._dummify(array, n_col)

        result = benchmark(run_dummify)
        assert result.shape == (10000, 100)
        assert result.nnz == 10000  # One entry per row

    def test_join_compatible_slices_level2(self, benchmark):
        """Benchmark _join_compatible_slices() for level 2."""
        # Create sparse slices matrix
        n_slices = 100
        n_features = 50
        slices = sp.random(
            n_slices, n_features, density=0.1, format="csr", dtype=bool
        )

        def run_join():
            return Slicefinder._join_compatible_slices(slices, level=2)

        result = benchmark(run_join)
        assert result.shape == (n_slices, n_slices)

    def test_join_compatible_slices_level3(self, benchmark):
        """Benchmark _join_compatible_slices() for level 3."""
        n_slices = 100
        n_features = 50
        slices = sp.random(
            n_slices, n_features, density=0.1, format="csr", dtype=bool
        )

        def run_join():
            return Slicefinder._join_compatible_slices(slices, level=3)

        result = benchmark(run_join)
        assert result.shape == (n_slices, n_slices)

    def test_score_performance(self, benchmark, fitted_slicefinder):
        """Benchmark _score() method."""
        n = 1000
        slice_sizes = np.random.randint(10, 1000, size=n)
        slice_errors = np.random.rand(n) * slice_sizes
        n_row = 10000

        def run_score():
            return fitted_slicefinder._score(slice_sizes, slice_errors, n_row)

        result = benchmark(run_score)
        assert result.shape == (n,)

    def test_score_ub_performance(self, benchmark, fitted_slicefinder):
        """Benchmark _score_ub() method."""
        n = 1000
        slice_sizes_ub = np.random.randint(10, 1000, size=n)
        slice_errors_ub = np.random.rand(n) * slice_sizes_ub
        max_slice_errors_ub = np.random.rand(n)
        n_col = 100

        def run_score_ub():
            return fitted_slicefinder._score_ub(
                slice_sizes_ub, slice_errors_ub, max_slice_errors_ub, n_col
            )

        result = benchmark(run_score_ub)
        assert result.shape == (n,)


class TestScalingBehavior:
    """Test how performance scales with dataset size."""

    @pytest.mark.parametrize(
        "n_samples,min_sup",
        [(1000, 10), (5000, 50), (10000, 100), (20000, 200)],
    )
    def test_scaling_with_samples(self, benchmark, n_samples, min_sup):
        """Test how fit() scales with number of samples."""
        np.random.seed(42)
        X = np.random.randint(1, 5, size=(n_samples, 10))
        errors = np.random.rand(n_samples)

        sf = Slicefinder(k=3, max_l=3, min_sup=min_sup, verbose=False)
        result = benchmark(sf.fit, X, errors)
        assert result.top_slices_ is not None

    @pytest.mark.parametrize("n_features", [5, 10, 20, 30])
    def test_scaling_with_features(self, benchmark, n_features):
        """Test how fit() scales with number of features."""
        np.random.seed(42)
        n_samples = 5000
        X = np.random.randint(1, 5, size=(n_samples, n_features))
        errors = np.random.rand(n_samples)

        sf = Slicefinder(k=3, max_l=3, min_sup=50, verbose=False)
        result = benchmark(sf.fit, X, errors)
        assert result.top_slices_ is not None

    @pytest.mark.parametrize("max_l", [2, 3, 4, 5])
    def test_scaling_with_lattice_level(self, benchmark, max_l):
        """Test how fit() scales with maximum lattice level."""
        np.random.seed(42)
        n_samples = 5000
        X = np.random.randint(1, 5, size=(n_samples, 10))
        errors = np.random.rand(n_samples)

        sf = Slicefinder(k=3, max_l=max_l, min_sup=50, verbose=False)
        result = benchmark(sf.fit, X, errors)
        assert result.top_slices_ is not None


class TestMemoryEfficiency:
    """Test memory usage of sparse operations."""

    def test_sparse_join_memory(self):
        """Verify sparse join doesn't create large dense matrices."""
        # Create large sparse matrix
        n_slices = 1000
        n_features = 200
        slices = sp.random(
            n_slices, n_features, density=0.05, format="csr", dtype=bool
        )

        # Join operation should return sparse matrix
        result = Slicefinder._join_compatible_slices(slices, level=3)

        # Verify result is sparse
        assert sp.issparse(result)
        assert isinstance(result, sp.csr_matrix)

        # Memory should be proportional to nnz, not n_slices^2
        # Dense would be ~8MB (1000*1000*8 bytes), sparse should be much smaller
        expected_max_memory = 1_000_000  # 1MB
        actual_memory = (
            result.data.nbytes + result.indices.nbytes + result.indptr.nbytes
        )
        assert actual_memory < expected_max_memory, (
            f"Sparse matrix using {actual_memory} bytes, "
            f"expected < {expected_max_memory} bytes"
        )

    def test_dummify_memory(self):
        """Verify _dummify creates efficient sparse representation."""
        array = np.random.randint(1, 1000, size=10000)
        n_col = 1000

        result = Slicefinder._dummify(array, n_col)

        # Verify result is sparse
        assert sp.issparse(result)
        assert isinstance(result, sp.csr_matrix)

        # Should have exactly one entry per row
        assert result.nnz == len(array)

        # Memory should be proportional to nnz, not n_row * n_col
        # Dense would be ~80MB (10000*1000*8 bytes), sparse should be ~240KB
        expected_max_memory = 500_000  # 500KB
        actual_memory = (
            result.data.nbytes + result.indices.nbytes + result.indptr.nbytes
        )
        assert actual_memory < expected_max_memory, (
            f"Sparse matrix using {actual_memory} bytes, "
            f"expected < {expected_max_memory} bytes"
        )


class TestNumbaConsistency:
    """Test numerical consistency between Numba and NumPy implementations."""

    def test_numba_numpy_identical_results(self):
        """Verify Numba and NumPy implementations produce identical results.

        This test ensures that the Numba-optimized code path produces
        numerically identical results to the pure NumPy fallback path.
        """
        from unittest.mock import patch

        # Create test data with clear error pattern
        np.random.seed(42)
        n_samples = 1000
        X = np.random.randint(1, 5, size=(n_samples, 10))

        # Create slice with high errors (first 500 samples, feature 0 = 1)
        X[:500, 0] = 1
        X[500:, 0] = 2

        errors = np.zeros(n_samples)
        errors[:500] = 1.0
        errors[500:] = np.random.random(500) * 0.2

        # Fit with Numba (if available)
        sf_numba = Slicefinder(k=5, max_l=3, min_sup=10, verbose=False)
        sf_numba.fit(X, errors)

        # Fit without Numba (force NumPy fallback)
        with patch("sliceline.slicefinder.NUMBA_AVAILABLE", False):
            sf_numpy = Slicefinder(k=5, max_l=3, min_sup=10, verbose=False)
            sf_numpy.fit(X, errors)

        # Verify same number of slices found
        assert len(sf_numba.top_slices_statistics_) == len(
            sf_numpy.top_slices_statistics_
        ), "Different number of slices found"

        # Verify identical statistics for each slice
        for i, (stat_numba, stat_numpy) in enumerate(
            zip(sf_numba.top_slices_statistics_, sf_numpy.top_slices_statistics_)
        ):
            # Check slice scores (must be identical within floating point precision)
            assert abs(stat_numba["slice_score"] - stat_numpy["slice_score"]) < 1e-10, (
                f"Slice {i}: score mismatch "
                f"(Numba={stat_numba['slice_score']}, "
                f"NumPy={stat_numpy['slice_score']})"
            )

            # Check slice sizes (must be exact)
            assert stat_numba["slice_size"] == stat_numpy["slice_size"], (
                f"Slice {i}: size mismatch "
                f"(Numba={stat_numba['slice_size']}, "
                f"NumPy={stat_numpy['slice_size']})"
            )

            # Check sum of errors (must be identical)
            assert (
                abs(stat_numba["sum_slice_error"] - stat_numpy["sum_slice_error"])
                < 1e-10
            ), (
                f"Slice {i}: sum_error mismatch "
                f"(Numba={stat_numba['sum_slice_error']}, "
                f"NumPy={stat_numpy['sum_slice_error']})"
            )

            # Check average error (must be identical)
            assert (
                abs(
                    stat_numba["slice_average_error"]
                    - stat_numpy["slice_average_error"]
                )
                < 1e-10
            ), (
                f"Slice {i}: avg_error mismatch "
                f"(Numba={stat_numba['slice_average_error']}, "
                f"NumPy={stat_numpy['slice_average_error']})"
            )

        # Verify transform produces identical results
        X_trans_numba = sf_numba.transform(X)
        X_trans_numpy = sf_numpy.transform(X)

        assert X_trans_numba.shape == X_trans_numpy.shape, "Transform shape mismatch"
        assert np.array_equal(
            X_trans_numba, X_trans_numpy
        ), "Transform produces different results"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
