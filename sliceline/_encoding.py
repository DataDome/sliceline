"""Encoding strategies for slice finding algorithms.

Provides CappedOneHotEncoder: a memory-efficient one-hot encoder
with per-feature cardinality pruning and int8/bool dtype output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder

NDArray = npt.NDArray[Any]


class CappedOneHotEncoder:
    """OneHotEncoder with per-feature cardinality cap and compact dtype.

    Wraps sklearn's OneHotEncoder but:
    1. Prunes rare categories, keeping only the top-max_cardinality
       most frequent values per feature. All rare values are grouped
       into a shared "other" column per feature, so they can still
       form a slice predicate (e.g., "feature has a rare value").
    2. Returns sparse matrices in bool dtype (1 byte per nnz instead
       of 8 bytes for float64), reducing memory by ~8x on values.

    Parameters
    ----------
    max_cardinality : int or None, default=None
        Maximum categories per feature. None means no cap.
    """

    def __init__(self, max_cardinality: int | None = None) -> None:
        self.max_cardinality = max_cardinality
        self._encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.bool_)
        self._value_maps: list[NDArray] | None = None

    def fit(self, X: NDArray) -> CappedOneHotEncoder:
        """Fit encoder, pruning rare categories if needed."""
        if self.max_cardinality is None:
            self._encoder.fit(X)
            self.categories_ = self._encoder.categories_
            return self

        X_pruned = self._prune_rare(X, fit=True)
        self._encoder.fit(X_pruned)
        self.categories_ = self._encoder.categories_
        return self

    def transform(self, X: NDArray) -> sp.csr_matrix:
        """Transform X, mapping pruned values to 'unknown'."""
        if self._value_maps is not None:
            X_pruned = self._prune_rare(X, fit=False)
            return self._encoder.transform(X_pruned)
        return self._encoder.transform(X)

    def fit_transform(self, X: NDArray) -> sp.csr_matrix:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, encoded: sp.csr_matrix) -> NDArray:
        """Inverse transform via underlying OHE."""
        return self._encoder.inverse_transform(encoded)

    def _prune_rare(self, X: NDArray, fit: bool) -> NDArray:
        """Replace rare category values with a per-feature sentinel.

        Rare values are mapped to a sentinel value that is guaranteed
        to not collide with any real value in the column. The OHE
        learns it as a real category during fit, grouping all rare
        values into a single "other" column per feature.
        """
        X_pruned = X.copy()

        if fit:
            self._value_maps = []
            self._sentinels = []

        for f in range(X.shape[1]):
            if fit:
                unique_vals, counts = np.unique(X[:, f], return_counts=True)
                if len(unique_vals) > self.max_cardinality:
                    top_idx = np.argsort(-counts)[: self.max_cardinality]
                    self._value_maps.append(unique_vals[top_idx])
                else:
                    self._value_maps.append(unique_vals)

                # Pick a sentinel distinct from all real values
                self._sentinels.append(self._make_sentinel(unique_vals))

            kept_set = set(self._value_maps[f])
            mask = np.array([v not in kept_set for v in X_pruned[:, f]])
            X_pruned[mask, f] = self._sentinels[f]

        return X_pruned

    @staticmethod
    def _make_sentinel(unique_vals: NDArray):
        """Return a value guaranteed absent from unique_vals.

        Adapts to the column dtype:
        - numeric: min(values) - 1
        - string/object: "__OTHER__" (with suffix if collision)
        """
        if unique_vals.dtype.kind in ("i", "u", "f"):
            return int(np.min(unique_vals)) - 1

        sentinel = "__OTHER__"
        existing = set(unique_vals)
        while sentinel in existing:
            sentinel += "_"
        return sentinel
