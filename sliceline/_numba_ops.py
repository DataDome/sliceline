"""Numba-accelerated operations for Sliceline.

Provides JIT-compiled versions of performance-critical functions
for 5-50x performance improvements in scoring and ID computation.

This module is optional - if numba is not installed, the main slicefinder
module will fall back to pure NumPy implementations.

Installation:
    pip install numba
    # or
    pip install sliceline[optimized]
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def score_slices_numba(
    slice_sizes: np.ndarray,
    slice_errors: np.ndarray,
    n_row: int,
    alpha: float,
    avg_error: float,
) -> np.ndarray:
    """JIT-compiled slice scoring function.

    Computes scores for each slice based on size and error metrics.
    5-10x faster than pure NumPy implementation.

    Parameters
    ----------
    slice_sizes : np.ndarray
        Array of slice sizes.
    slice_errors : np.ndarray
        Array of slice errors.
    n_row : int
        Number of rows in the encoded dataset.
    alpha : float
        Weight parameter for error importance.
    avg_error : float
        Average error across all samples.

    Returns
    -------
    np.ndarray
        Array of computed scores for each slice.
    """
    n = slice_sizes.shape[0]
    scores = np.empty(n, dtype=np.float64)

    for i in range(n):
        if slice_sizes[i] <= 0:
            scores[i] = -np.inf
        else:
            slice_avg_error = slice_errors[i] / slice_sizes[i]
            error_term = alpha * (slice_avg_error / avg_error - 1.0)
            size_term = (1.0 - alpha) * (n_row / slice_sizes[i] - 1.0)
            scores[i] = error_term - size_term

    return scores


@njit(cache=True)
def score_ub_single_numba(
    slice_size: float,
    slice_error: float,
    max_slice_error: float,
    n_col_x_encoded: int,
    min_sup: float,
    alpha: float,
    avg_error: float,
) -> float:
    """JIT-compiled upper bound score for a single slice.

    Parameters
    ----------
    slice_size : float
        Size of the slice.
    slice_error : float
        Error sum of the slice.
    max_slice_error : float
        Maximum error in the slice.
    n_col_x_encoded : int
        Number of encoded columns.
    min_sup : float
        Minimum support threshold.
    alpha : float
        Weight parameter for error importance.
    avg_error : float
        Average error across all samples.

    Returns
    -------
    float
        Upper bound score for the slice.
    """
    if slice_size <= 0:
        return -np.inf

    potential_solutions = np.array(
        [
            min_sup,
            max(slice_error / max_slice_error, min_sup)
            if max_slice_error > 0
            else min_sup,
            slice_size,
        ]
    )

    max_score = -np.inf
    for pot_sol in potential_solutions:
        if pot_sol <= 0:
            continue
        error_contrib = min(pot_sol * max_slice_error, slice_error)
        score = (
            alpha * (error_contrib / avg_error - pot_sol)
            - (1.0 - alpha) * (n_col_x_encoded - pot_sol)
        ) / pot_sol
        if score > max_score:
            max_score = score

    return max_score


@njit(cache=True)
def score_ub_batch_numba(
    slice_sizes_ub: np.ndarray,
    slice_errors_ub: np.ndarray,
    max_slice_errors_ub: np.ndarray,
    n_col_x_encoded: int,
    min_sup: float,
    alpha: float,
    avg_error: float,
) -> np.ndarray:
    """JIT-compiled upper bound scoring function for batch processing.

    5-10x faster than pure NumPy implementation.

    Parameters
    ----------
    slice_sizes_ub : np.ndarray
        Array of slice sizes (upper bound).
    slice_errors_ub : np.ndarray
        Array of slice errors (upper bound).
    max_slice_errors_ub : np.ndarray
        Array of maximum slice errors (upper bound).
    n_col_x_encoded : int
        Number of encoded columns.
    min_sup : float
        Minimum support threshold.
    alpha : float
        Weight parameter for error importance.
    avg_error : float
        Average error across all samples.

    Returns
    -------
    np.ndarray
        Array of upper bound scores for each slice.
    """
    n = slice_sizes_ub.shape[0]
    scores = np.empty(n, dtype=np.float64)

    for i in range(n):
        scores[i] = score_ub_single_numba(
            slice_sizes_ub[i],
            slice_errors_ub[i],
            max_slice_errors_ub[i],
            n_col_x_encoded,
            min_sup,
            alpha,
            avg_error,
        )

    return scores


@njit(cache=True)
def compute_slice_ids_numba(
    slices_data: np.ndarray,
    slices_indices: np.ndarray,
    slices_indptr: np.ndarray,
    feature_offset_start: np.ndarray,
    feature_offset_end: np.ndarray,
    feature_domains: np.ndarray,
) -> np.ndarray:
    """JIT-compiled slice ID computation.

    Computes unique IDs for each slice based on feature encoding.
    10-50x faster than Python loop for large datasets.

    Parameters
    ----------
    slices_data : np.ndarray
        Data array from sparse matrix.
    slices_indices : np.ndarray
        Column indices array from sparse matrix.
    slices_indptr : np.ndarray
        Index pointer array from sparse matrix.
    feature_offset_start : np.ndarray
        Start offset for each feature.
    feature_offset_end : np.ndarray
        End offset for each feature.
    feature_domains : np.ndarray
        Domain size for each feature.

    Returns
    -------
    np.ndarray
        Array of unique IDs for each slice.
    """
    n_slices = len(slices_indptr) - 1
    n_features = len(feature_offset_start)
    slice_ids = np.zeros(n_slices, dtype=np.float64)

    dom = feature_domains + 1

    for i in range(n_slices):
        start_idx = slices_indptr[i]
        end_idx = slices_indptr[i + 1]

        slice_id = 0.0
        for j in range(start_idx, end_idx):
            col = slices_indices[j]
            val = slices_data[j]

            if val == 0:
                continue

            for f in range(n_features):
                if feature_offset_start[f] <= col < feature_offset_end[f]:
                    offset = col - feature_offset_start[f]
                    multiplier = 1.0
                    for k in range(f + 1, n_features):
                        multiplier *= dom[k]
                    slice_id += (offset + 1) * multiplier
                    break

        slice_ids[i] = slice_id

    return slice_ids
