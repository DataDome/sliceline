"""Performance benchmarks for Sliceline.

This module contains performance regression tests using pytest-benchmark.
Run with: poetry run pytest tests/test_performance.py -v
"""

import numpy as np
import pytest
from scipy import sparse as sp

from sliceline import Slicefinder


@pytest.fixture
def small_dataset():
    """Small dataset for quick benchmark iterations."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    X = np.random.randint(1, 5, size=(n_samples, n_features))
    errors = np.random.rand(n_samples)
    return X, errors


@pytest.fixture
def medium_dataset():
    """Medium dataset for realistic performance testing."""
    np.random.seed(42)
    n_samples = 10000
    n_features = 20
    X = np.random.randint(1, 10, size=(n_samples, n_features))
    errors = np.random.rand(n_samples)
    return X, errors


@pytest.fixture
def large_dataset():
    """Large dataset for stress testing."""
    np.random.seed(42)
    n_samples = 50000
    n_features = 30
    X = np.random.randint(1, 15, size=(n_samples, n_features))
    errors = np.random.rand(n_samples)
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
        sf = Slicefinder(k=5, max_l=3, min_sup=10, verbose=False)
        sf.fit(X, errors)
        result = benchmark(sf.transform, X)
        assert result.shape[0] == X.shape[0]

    def test_transform_medium(self, benchmark, medium_dataset):
        """Benchmark transform() on medium dataset."""
        X, errors = medium_dataset
        sf = Slicefinder(k=5, max_l=3, min_sup=100, verbose=False)
        sf.fit(X, errors)
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
        slices = sp.random(n_slices, n_features, density=0.1, format="csr", dtype=bool)

        def run_join():
            return Slicefinder._join_compatible_slices(slices, level=2)

        result = benchmark(run_join)
        assert result.shape == (n_slices, n_slices)

    def test_join_compatible_slices_level3(self, benchmark):
        """Benchmark _join_compatible_slices() for level 3."""
        n_slices = 100
        n_features = 50
        slices = sp.random(n_slices, n_features, density=0.1, format="csr", dtype=bool)

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
        slices = sp.random(n_slices, n_features, density=0.05, format="csr", dtype=bool)

        # Join operation should return sparse matrix
        result = Slicefinder._join_compatible_slices(slices, level=3)

        # Verify result is sparse
        assert sp.issparse(result)
        assert isinstance(result, sp.csr_matrix)

        # Memory should be proportional to nnz, not n_slices^2
        # Dense would be ~8MB (1000*1000*8 bytes), sparse should be much smaller
        expected_max_memory = 1_000_000  # 1MB
        actual_memory = result.data.nbytes + result.indices.nbytes + result.indptr.nbytes
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
        actual_memory = result.data.nbytes + result.indices.nbytes + result.indptr.nbytes
        assert actual_memory < expected_max_memory, (
            f"Sparse matrix using {actual_memory} bytes, "
            f"expected < {expected_max_memory} bytes"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
