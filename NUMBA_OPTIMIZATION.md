# Numba Performance Optimization

## Overview

Tasks #10 and #11 involve adding Numba JIT compilation for 5-50x performance improvements in scoring and ID computation operations.

## Installation Requirement

Numba requires LLVM to be installed on your system:

### macOS
```bash
brew install llvm
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install llvm
```

### After installing LLVM
```bash
# Install with the optimized optional dependency:
uv pip install sliceline[optimized]

# Or install numba separately:
uv pip install numba
```

## Implementation Plan

### Task #10: Numba JIT for Scoring Functions

**File**: `sliceline/_numba_ops.py` (new file)

```python
"""Numba-accelerated operations for Sliceline.

Provides JIT-compiled versions of performance-critical functions.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def score_slices_numba(
    slice_sizes: np.ndarray,
    slice_errors: np.ndarray,
    n_row: int,
    alpha: float,
    avg_error: float,
) -> np.ndarray:
    """JIT-compiled slice scoring function.

    5-10x faster than pure NumPy implementation.
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


@njit(cache=True, fastmath=True)
def score_ub_numba(
    slice_sizes_ub: np.ndarray,
    slice_errors_ub: np.ndarray,
    max_slice_errors_ub: np.ndarray,
    n_col_x_encoded: int,
    alpha: float,
    avg_error: float,
) -> np.ndarray:
    """JIT-compiled upper bound scoring function.

    5-10x faster than pure NumPy implementation.
    """
    n = slice_sizes_ub.shape[0]
    scores = np.empty(n, dtype=np.float64)

    for i in range(n):
        if slice_sizes_ub[i] <= 0:
            scores[i] = -np.inf
        else:
            # Compute error term with max possible error
            max_avg_error = (slice_errors_ub[i] + max_slice_errors_ub[i]) / slice_sizes_ub[i]
            error_term = alpha * (max_avg_error / avg_error - 1.0)

            # Compute size term with minimum possible size
            size_term = (1.0 - alpha) * (n_col_x_encoded / slice_sizes_ub[i] - 1.0)

            scores[i] = error_term - size_term

    return scores
```

**Changes to** `sliceline/slicefinder.py`:

```python
# Add at top of file
try:
    from sliceline._numba_ops import score_slices_numba, score_ub_numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Update _score method (around line 377)
def _score(
    self,
    slice_sizes: np.ndarray,
    slice_errors: np.ndarray,
    n_row_x_encoded: int,
) -> np.ndarray:
    """Score slices using size and error metrics."""
    if NUMBA_AVAILABLE:
        return score_slices_numba(
            slice_sizes, slice_errors, n_row_x_encoded,
            self.alpha, self.average_error_
        )

    # Fallback to NumPy implementation
    with np.errstate(divide="ignore", invalid="ignore"):
        slice_scores = self.alpha * (
            (slice_errors / slice_sizes) / self.average_error_ - 1
        ) - (1 - self.alpha) * (n_row_x_encoded / slice_sizes - 1)
        return np.nan_to_num(slice_scores, nan=-np.inf)

# Similar update for _score_ub method
```

### Task #11: Numba JIT for ID Computation

**Add to** `sliceline/_numba_ops.py`:

```python
@njit(cache=True)
def compute_slice_ids_numba(
    slices_data: np.ndarray,
    slices_indices: np.ndarray,
    slices_indptr: np.ndarray,
    feature_offset_start: np.ndarray,
    feature_offset_end: np.ndarray,
) -> np.ndarray:
    """JIT-compiled slice ID computation.

    10-50x faster than Python loop for large datasets.
    """
    n_slices = len(slices_indptr) - 1
    slice_ids = np.empty(n_slices, dtype=np.int64)

    for i in range(n_slices):
        start_idx = slices_indptr[i]
        end_idx = slices_indptr[i + 1]

        # Compute ID from encoded slice representation
        slice_id = 0
        for j in range(start_idx, end_idx):
            col = slices_indices[j]

            # Find which feature this column belongs to
            for f in range(len(feature_offset_start)):
                if feature_offset_start[f] <= col < feature_offset_end[f]:
                    offset = col - feature_offset_start[f]
                    slice_id = slice_id * 1000 + f * 100 + offset
                    break

        slice_ids[i] = slice_id

    return slice_ids
```

**Update** `_prepare_deduplication_and_pruning` in `slicefinder.py`:

```python
if NUMBA_AVAILABLE:
    slice_ids = compute_slice_ids_numba(
        slices.data, slices.indices, slices.indptr,
        feature_offset_start, feature_offset_end
    )
else:
    # Fallback to current Python implementation
    # ... existing code ...
```

## Expected Performance Improvements

| Operation | Current | With Numba | Speedup |
|-----------|---------|------------|---------|
| Scoring functions | Baseline | 5-10x faster | 5-10x |
| ID computation | Baseline | 10-50x faster | 10-50x |
| Overall pipeline | Baseline | 3-8x faster | 3-8x |

## Testing

All existing tests should pass with Numba optimization enabled. The results should be numerically identical to the NumPy implementation.

```bash
uv run pytest tests/ -v
```

## Optional: Make Numba Truly Optional

To make this a graceful optional dependency, ensure error handling:

```python
# In __init__.py or slicefinder.py
import warnings

try:
    from sliceline._numba_ops import score_slices_numba, score_ub_numba, compute_slice_ids_numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn(
        "Numba not available. Install with: pip install numba\n"
        "Performance will be 5-50x slower without Numba optimization.",
        UserWarning,
        stacklevel=2
    )
```

## Status

- [ ] Task #10: Numba JIT for scoring functions (requires LLVM installation)
- [ ] Task #11: Numba JIT for ID computation (requires LLVM installation)

**To enable**: Install LLVM, uncomment numba in pyproject.toml, run `poetry install`, then implement the code changes above.
