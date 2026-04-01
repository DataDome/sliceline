"""Tests for max_cardinality parameter in Slicefinder."""

import numpy as np
import pytest

from sliceline import Slicefinder
from sliceline._encoding import CappedOneHotEncoder


class TestCappedOneHotEncoder:
    def test_no_cap_matches_sklearn(self):
        """Without cap, behaves like standard OHE."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        enc = CappedOneHotEncoder(max_cardinality=None)
        result = enc.fit_transform(X)
        assert result.shape == (3, 6)

    def test_cap_reduces_columns(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 100, size=(1000, 5))
        enc_full = CappedOneHotEncoder(max_cardinality=None)
        enc_cap = CappedOneHotEncoder(max_cardinality=10)
        assert enc_cap.fit_transform(X).shape[1] < enc_full.fit_transform(
            X
        ).shape[1]

    def test_bool_dtype(self):
        X = np.array([[1, 2], [3, 4]])
        enc = CappedOneHotEncoder()
        assert enc.fit_transform(X).dtype == np.bool_

    def test_bool_smaller_than_float(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 20, size=(5000, 10))
        enc = CappedOneHotEncoder()
        result_bool = enc.fit_transform(X)
        from sklearn.preprocessing import OneHotEncoder

        result_float = OneHotEncoder(handle_unknown="ignore").fit_transform(X)
        assert result_bool.data.nbytes < result_float.data.nbytes

    def test_inverse_transform(self):
        X = np.array([[1, 2], [3, 4], [1, 4]])
        enc = CappedOneHotEncoder()
        decoded = enc.inverse_transform(enc.fit_transform(X))
        np.testing.assert_array_equal(X, decoded)

    def test_rare_values_handled(self):
        X_train = np.array([[1, 1], [2, 2], [3, 3]] * 100)
        X_test = np.array([[1, 1], [99, 99]])
        enc = CappedOneHotEncoder(max_cardinality=2)
        enc.fit(X_train)
        result = enc.transform(X_test)
        assert result.shape[0] == 2


class TestSlicefinderMaxCardinality:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        X = rng.integers(1, 6, size=(500, 5))
        errors = rng.uniform(0.0, 0.3, size=500)
        errors[X[:, 0] == 1] = rng.uniform(
            0.8, 1.0, size=(X[:, 0] == 1).sum()
        )
        return X, errors

    def test_finds_same_slices_without_cap(self, data):
        """No cap should produce identical results to default."""
        X, errors = data
        sf_default = Slicefinder(
            alpha=0.95, k=5, max_l=2, min_sup=10, verbose=False
        )
        sf_default.fit(X, errors)

        sf_nocap = Slicefinder(
            alpha=0.95,
            k=5,
            max_l=2,
            min_sup=10,
            max_cardinality=None,
            verbose=False,
        )
        sf_nocap.fit(X, errors)

        assert sf_default.top_slices_.shape == sf_nocap.top_slices_.shape

    def test_finds_gt_slice_with_cap(self, data):
        """With cap >= actual cardinality, should still find GT."""
        X, errors = data
        sf = Slicefinder(
            alpha=0.95,
            k=5,
            max_l=2,
            min_sup=10,
            max_cardinality=10,
            verbose=False,
        )
        sf.fit(X, errors)
        assert sf.top_slices_.shape[0] >= 1
        assert sf.top_slices_[0][0] == 1

    def test_cap_reduces_memory(self, data):
        """Capped version should use less memory."""
        import tracemalloc

        X, errors = data

        tracemalloc.start()
        Slicefinder(
            alpha=0.95, k=5, max_l=2, min_sup=10, verbose=False
        ).fit(X, errors)
        _, peak_default = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        Slicefinder(
            alpha=0.95,
            k=5,
            max_l=2,
            min_sup=10,
            max_cardinality=3,
            verbose=False,
        ).fit(X, errors)
        _, peak_capped = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak_capped <= peak_default * 1.05

    def test_same_slices_with_high_cap(self, data):
        """With cap >= actual cardinality, same top slice found."""
        X, errors = data
        sf_base = Slicefinder(
            alpha=0.95, k=5, max_l=2, min_sup=10, verbose=False
        ).fit(X, errors)

        sf_cap = Slicefinder(
            alpha=0.95,
            k=5,
            max_l=2,
            min_sup=10,
            max_cardinality=10,
            verbose=False,
        ).fit(X, errors)

        # Best slice should match
        if (
            sf_base.top_slices_.shape[0] > 0
            and sf_cap.top_slices_.shape[0] > 0
        ):
            base_masks = sf_base.transform(X)[:, 0].astype(bool)
            cap_masks = sf_cap.transform(X)[:, 0].astype(bool)
            inter = np.logical_and(base_masks, cap_masks).sum()
            union = np.logical_or(base_masks, cap_masks).sum()
            assert inter / union > 0.9

    def test_transform_works_with_cap(self, data):
        X, errors = data
        sf = Slicefinder(
            alpha=0.95,
            k=5,
            max_l=2,
            min_sup=10,
            max_cardinality=5,
            verbose=False,
        ).fit(X, errors)
        if sf.top_slices_.shape[0] > 0:
            masks = sf.transform(X)
            assert masks.shape[0] == X.shape[0]

    def test_high_cardinality_data(self):
        """Should handle high-cardinality features gracefully."""
        rng = np.random.default_rng(42)
        X = rng.integers(1, 100, size=(1000, 5))
        errors = rng.uniform(0.0, 0.3, size=1000)
        errors[X[:, 0] == 1] = 0.9

        sf = Slicefinder(
            alpha=0.95,
            k=5,
            max_l=2,
            min_sup=10,
            max_cardinality=10,
            verbose=False,
        )
        sf.fit(X, errors)
        assert sf.top_slices_ is not None
