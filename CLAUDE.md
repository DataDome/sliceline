# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sliceline is a Python library for fast slice finding for Machine Learning model debugging. It implements the SliceLine algorithm from the paper "SliceLine: Fast, Linear-Algebra-based Slice Finding for ML Model Debugging" by Svetlana Sagadeeva and Matthias Boehm.

**Core Purpose**: Given an input dataset `X` and a model error vector `errors`, SliceLine identifies the top `k` slices (subspaces defined by predicates) where the ML model performs significantly worse.

## Development Commands

### Environment Setup
```sh
make init                    # Install dependencies via Poetry
pre-commit install --hook-type pre-push  # Install pre-commit hooks
```

### Testing
```sh
make test                    # Run unit tests with coverage (requires 80% coverage minimum)
poetry run pytest            # Run tests without coverage report
poetry run pytest tests/test_slicefinder.py::test_experiments  # Run specific test
poetry run pytest -k "experiment_1"  # Run tests matching pattern
```

### Code Quality
```sh
make lint                    # Run black, isort, and flake8
poetry run black .           # Format code (line length: 79)
poetry run isort .           # Sort imports (black profile)
poetry run flake8            # Check code style
pre-commit run --all-files   # Run pre-commit checks manually
```

### Documentation
```sh
make doc                     # Build Sphinx documentation locally
make notebook                # Start Jupyter notebook server
make execute-notebooks       # Execute all notebooks (run before releases)
```

## Architecture

### Core Algorithm (sliceline/slicefinder.py)

Standard implementation for low-to-medium cardinality datasets.



The `Slicefinder` class is a scikit-learn compatible estimator implementing the SliceLine algorithm through sparse linear algebra operations:

**Key Algorithm Steps**:
1. **One-hot encode input**: Convert categorical/numerical features to binary representation
2. **Initialize 1-slices**: Create and score basic slices (single predicates)
3. **Lattice enumeration**: Iteratively combine slices up to `max_l` levels, pruning based on size and error bounds
4. **Top-k maintenance**: Track best slices throughout enumeration

**Critical Parameters**:
- `alpha` (0 < α ≤ 1): Balance between slice size and average error
- `k`: Number of top slices to return
- `max_l`: Maximum predicates per slice (controls combinatorial explosion)
- `min_sup`: Minimum support threshold (absolute or fraction)

**Key Methods**:
- `fit(X, errors)`: Main entry point - searches for slices
- `transform(X)`: Returns binary masks indicating slice membership
- `get_slice(X, slice_index)`: Filters dataset to specific slice
- `_search_slices()`: Core algorithm implementation
- `_score()` / `_score_ub()`: Slice scoring and upper-bound pruning
- `_maintain_top_k()`: Efficiently tracks best slices

**Performance Optimizations**:
- Sparse matrix operations (scipy.sparse) throughout
- Upper-bound pruning to avoid evaluating unpromising candidates
- Missing parent detection to avoid invalid slice combinations
- Deduplication via ID-based hashing

### Optimized Implementation (sliceline/optimized_slicefinder.py)

**NEW in v0.3.0** - Performance-optimized version for large-scale datasets with high cardinality.

`OptimizedSlicefinder` extends `Slicefinder` with two key optimizations:

**1. Feature Hashing** (lines 137-172):
- Caps cardinality per feature to `max_features_per_column` (default: 1000)
- Uses deterministic hash function: `hash(str(value)) % max_features_per_column`
- Applied in `fit()` before calling parent's one-hot encoding
- Reduces memory from O(n × total_unique_values) to O(n × max_features)
- Trade-off: Hash collisions introduce approximation for high-cardinality features

**2. Numba JIT Compilation** (lines 281-357):
- Accelerates `_eval_slice()` with Numba's `@jit(nopython=True, parallel=True)`
- Replaces sparse matrix operations with optimized loops
- 3-5x speedup on slice evaluation (typically the bottleneck)
- Gracefully degrades if Numba not installed
- Parallel execution across slices using `nb.prange()`

**Key Design Decisions**:
- Inherits from `Slicefinder` to maintain API compatibility
- All optimizations optional and configurable
- Falls back to parent implementation when optimizations disabled
- Numba dependency optional (soft dependency)

**When to Use**:
- Features with >1000 unique values
- Datasets >100K rows
- When standard Slicefinder causes OOM
- Memory-constrained environments

**Performance Characteristics**:
- 2-6x faster on high-cardinality data
- 50-80% memory reduction with feature hashing
- Enables processing of datasets 10-100x larger

### Validation Module (sliceline/validation.py)

Custom validation overriding sklearn's `check_array` to **accept string/object dtype inputs** (line 554-555). This is essential because SliceLine works with categorical data that may be represented as strings. The module is derived from sklearn's validation utilities but modified specifically for this use case.

### Testing Structure (tests/)

- `test_slicefinder.py`: Comprehensive unit tests for all private and public methods
- `test_optimized_slicefinder.py`: Tests for optimized implementation
  - Validates equivalence with base implementation on low-cardinality data
  - Tests feature hashing behavior and high-cardinality handling
  - Benchmarks for performance comparison across cardinality levels
  - API compatibility tests
- `conftest.py`: Pytest fixtures for test data (17 different experiments)
- `experiment.py`: Test case definitions
- Tests use `pytest-benchmark` for performance tracking
- Parametrized tests (`experiment_1` through `experiment_17`) validate algorithm correctness on various scenarios

### Benchmarking (benchmarks/)

- `cardinality_benchmark.py`: Profiling script comparing Slicefinder vs OptimizedSlicefinder
  - Tests cardinality levels: 10, 100, 1K, 10K unique values
  - Measures time, memory, and speedup improvements
  - Outputs `benchmark_results.json` with detailed metrics
  - Run with: `python benchmarks/cardinality_benchmark.py`

## Development Guidelines

### Code Style
- Line length: 79 characters (enforced by Black)
- Import sorting: Black profile (enforced by isort)
- Docstrings: Follow numpydoc convention for all public methods
- Type hints: Used where applicable (see slicefinder.py lines 6, 91-97)

### Testing Requirements
- Unit tests must pass for all changes
- Coverage threshold: 80% minimum (configured in pyproject.toml)
- Coverage excludes: validation.py, tests/, hidden files
- Benchmarking: Available via pytest-benchmark for performance-sensitive changes

### Adding New Features
- Open a GitHub discussion before starting work
- Add docstrings following numpydoc format
- Update relevant documentation in docs/source/
- Add unit tests achieving 80%+ coverage
- Update release notes (when requested)

### scikit-learn Compatibility
The `Slicefinder` class follows scikit-learn conventions:
- Inherits from `BaseEstimator` and `TransformerMixin`
- Implements `fit()`, `transform()`, `fit_transform()` pattern
- Uses `check_is_fitted()` for state validation
- Exposes `get_feature_names_out()` for pipeline integration
- Parameters set in `__init__` without validation (validated in `fit()`)

### Working with Sparse Matrices
- All internal representations use `scipy.sparse.csr_matrix`
- Avoid `.A` shorthand for `.toarray()` - not supported in all scipy versions (see comments at lines 383-386, 405-408, 517-519)
- Use explicit `.toarray()` calls when converting to dense

## Common Pitfalls

1. **Sparse matrix compatibility**: Some scipy versions don't support `.A` attribute on certain sparse matrix types. Always use `.toarray()` explicitly.

2. **String dtype handling**: The custom validation module allows string inputs, which sklearn's standard validation rejects. Don't replace with sklearn's built-in validation.

3. **min_sup conversion**: When `min_sup` is a float (0 < min_sup < 1), it gets converted to an absolute count in `fit()` (line 160). This modifies the instance attribute.

4. **Missing parents**: The `_get_pair_candidates()` method includes logic to handle cases where some parent slices were pruned (lines 578-583). This prevents invalid combinations.

5. **Notebook execution**: Notebooks require specific execution with unlimited timeout (see Makefile line 22) due to potentially long-running experiments.

6. **Feature hashing trade-offs** (OptimizedSlicefinder): Hash collisions mean results may differ slightly from base implementation on very high-cardinality features. For exact results, set `use_feature_hashing=False` or increase `max_features_per_column`.

7. **Numba compilation overhead**: First run with `use_numba=True` includes compilation time (~1-2s). Subsequent runs benefit from caching. For single-use scripts, Numba overhead may outweigh benefits.

8. **Feature hashing determinism**: The hash function is deterministic (same input → same hash), but results depend on Python's `hash()` implementation which may vary across Python versions. For reproducibility across Python versions, consider custom hash function.
