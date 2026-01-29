# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sliceline is a Python library for fast slice finding for Machine Learning model debugging. It implements the SliceLine algorithm from the paper "SliceLine: Fast, Linear-Algebra-based Slice Finding for ML Model Debugging" by Svetlana Sagadeeva and Matthias Boehm.

**Core Purpose**: Given an input dataset `X` and a model error vector `errors`, SliceLine identifies the top `k` slices (subspaces defined by predicates) where the ML model performs significantly worse.

## Development Commands

### Environment Setup
```sh
make init                    # Install dependencies via uv
pre-commit install --hook-type pre-push  # Install pre-commit hooks
```

### Testing
```sh
make test                    # Run unit tests with coverage (requires 80% coverage minimum)
uv run pytest                # Run tests without coverage report
uv run pytest tests/test_slicefinder.py::test_experiments  # Run specific test
uv run pytest -k "experiment_1"  # Run tests matching pattern
```

### Code Quality
```sh
uv run ruff check .          # Check code style
uv run ruff format . --check # Check formatting
uv run ruff format .         # Apply formatting
```

### Documentation
```sh
make doc                     # Build Sphinx documentation locally
make notebook                # Start Jupyter notebook server
make execute-notebooks       # Execute all notebooks (run before releases)
```

### Benchmarking

The project includes two types of benchmarks:

**Standalone benchmark scripts** (in `benchmarks/`):
```sh
# Run all benchmarks (cardinality + dataset size scaling)
python benchmarks/benchmarks.py

# Results are saved to:
# - benchmarks/benchmark_results.json (cardinality benchmark)
# - benchmarks/dataset_size_results.json (dataset size benchmark)
```

**pytest-benchmark suite** (in `tests/test_performance.py`):
```sh
# Run performance regression tests with benchmarks
uv run pytest tests/test_performance.py -v --benchmark-only

# Run with full output
uv run pytest tests/test_performance.py -v
```

The standalone benchmarks are for profiling and manual performance analysis.
The pytest-benchmark suite is for regression testing to detect performance regressions.

## Architecture

### Core Algorithm (sliceline/slicefinder.py)

The `Slicefinder` class is a scikit-learn compatible estimator implementing the SliceLine algorithm through sparse linear algebra operations.

**Key Algorithm Steps**:
1. **One-hot encode input**: Convert categorical/numerical features to binary representation
2. **Initialize 1-slices**: Create and score basic slices (single predicates)
3. **Lattice enumeration**: Iteratively combine slices up to `max_l` levels, pruning based on size and error bounds
4. **Top-k maintenance**: Track best slices throughout enumeration

**Critical Parameters**:
- `alpha` (0 < alpha <= 1): Balance between slice size and average error
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

**Performance Optimizations (v0.3.0)**:
- Sparse matrix operations (scipy.sparse) throughout
- Direct CSR construction in `_dummify()` (2-3x faster than lil_matrix)
- Sparse-preserving join in `_join_compatible_slices()` (memory efficient)
- Upper-bound pruning to avoid evaluating unpromising candidates
- Missing parent detection to avoid invalid slice combinations
- Deduplication via ID-based hashing
- Deterministic ordering for reproducible results

### Validation Module (sliceline/validation.py)

Custom validation overriding sklearn's `check_array` to **accept string/object dtype inputs** (line 554-555). This is essential because SliceLine works with categorical data that may be represented as strings. The module is derived from sklearn's validation utilities but modified specifically for this use case.

### Testing Structure (tests/)

- `test_slicefinder.py`: Comprehensive unit tests for all private and public methods
- `test_performance.py`: Performance benchmark suite using pytest-benchmark
  - Dataset size scaling tests (1K to 50K samples)
  - Feature count scaling tests (5 to 30 features)
  - Lattice level scaling tests (max_l 2 to 5)
  - Memory efficiency tests for sparse operations
- `conftest.py`: Pytest fixtures for test data (17 different experiments)
- `experiment.py`: Test case definitions
- Tests use `pytest-benchmark` for performance tracking
- Parametrized tests (`experiment_1` through `experiment_17`) validate algorithm correctness on various scenarios

### Benchmarking (benchmarks/)

- `benchmarks.py`: Profiling script for performance testing
  - Cardinality benchmark: Tests cardinality levels 10, 100, 500, 1000
  - Dataset size benchmark: Tests scaling with 1K to 50K samples
  - Measures time, memory, and improvement metrics
  - Outputs `benchmark_results.json` and `dataset_size_results.json`
  - Run with: `python benchmarks/benchmarks.py`

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
- Use `.nnz` for counting non-zero elements (faster than `.sum()`)
- Direct CSR construction preferred over lil_matrix for one-hot encoding

## Common Pitfalls

1. **Sparse matrix compatibility**: Some scipy versions don't support `.A` attribute on certain sparse matrix types. Always use `.toarray()` explicitly.

2. **String dtype handling**: The custom validation module allows string inputs, which sklearn's standard validation rejects. Don't replace with sklearn's built-in validation.

3. **min_sup conversion**: When `min_sup` is a float (0 < min_sup < 1), it gets converted to an absolute count in `fit()` (line 160). This modifies the instance attribute.

4. **Missing parents**: The `_get_pair_candidates()` method includes logic to handle cases where some parent slices were pruned (lines 578-583). This prevents invalid combinations.

5. **Notebook execution**: Notebooks require specific execution with unlimited timeout (see Makefile line 22) due to potentially long-running experiments.

6. **Deterministic ordering**: Results are sorted by score first, then lexicographically by slice representation. This ensures reproducible results across runs and Python versions.

7. **Memory efficiency**: Use sparse matrices throughout. The `_join_compatible_slices()` method returns sparse format to avoid memory explosion with large numbers of slices.

## Performance Considerations

### When to Use Sliceline

Sliceline is designed for datasets where:
- You want to find subgroups where your ML model underperforms
- Features are categorical or can be binned (continuous values should be discretized)
- Dataset size is reasonable (10K-100K samples works well)

### Performance Characteristics

Based on benchmarks:
- **Small datasets (1K samples)**: < 100ms
- **Medium datasets (10K samples)**: 100ms - 1s
- **Large datasets (50K+ samples)**: 1-10s depending on cardinality

### Optimization Tips

1. **Reduce cardinality**: Bin continuous features or use feature hashing for high-cardinality columns
2. **Limit lattice depth**: Keep `max_l` small (2-3) for faster execution
3. **Increase min_sup**: Higher support threshold prunes more aggressively
4. **Use appropriate k**: Smaller `k` values enable better pruning

### Future Optimizations (See NUMBA_OPTIMIZATION.md)

Planned Numba JIT compilation for:
- Scoring functions (`_score`, `_score_ub`)
- ID computation for deduplication
- Expected speedup: 5-50x on numeric-heavy operations
