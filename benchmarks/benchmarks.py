"""
Benchmark Script for Sliceline Performance Testing with Numba Optimization.

This script profiles the performance of Slicefinder with and without Numba
JIT compilation across different dataset sizes to measure speedup ratios.

Usage:
    python benchmarks/benchmarks.py

Output:
    - benchmark_results.json: Comprehensive benchmark results with speedups
    - Console output with summary tables
"""

from __future__ import annotations

import json
import logging
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from sliceline import Slicefinder, is_numba_available  # noqa: E402

RANDOM_SEED = 42

DATASET_CONFIGS = {
    "small": {"n_samples": 1000, "n_features": 10},
    "medium": {"n_samples": 10000, "n_features": 20},
    "large": {"n_samples": 50000, "n_features": 30},
}


def generate_synthetic_data(
    n_samples: int,
    n_features: int,
    cardinality: int = 10,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for benchmarking.

    Creates data with a clear "error slice" to ensure Slicefinder finds slices.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    cardinality : int
        Number of unique values per feature.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    errors : np.ndarray
        Error vector of shape (n_samples,).
    """
    np.random.seed(seed)
    X = np.random.randint(1, cardinality + 1, size=(n_samples, n_features))

    half = n_samples // 2
    X[:half, 0] = 1
    X[half:, 0] = 2

    X[:half, 1] = 1
    X[half:, 1] = np.random.randint(1, cardinality + 1, size=n_samples - half)

    errors = np.zeros(n_samples)
    errors[:half] = 1.0
    errors[half:] = np.random.random(n_samples - half) * 0.2

    return X, errors


def benchmark_fit(
    X: np.ndarray,
    errors: np.ndarray,
    n_warmup: int = 1,
    n_runs: int = 3,
) -> dict[str, Any]:
    """Benchmark the fit method of Slicefinder.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    errors : np.ndarray
        Error vector.
    n_warmup : int
        Number of warmup runs (not timed).
    n_runs : int
        Number of timed runs.

    Returns
    -------
    dict
        Benchmark results including timing and memory.
    """
    min_sup = max(10, X.shape[0] // 100)

    for _ in range(n_warmup):
        model = Slicefinder(k=10, max_l=2, min_sup=min_sup, verbose=False)
        model.fit(X, errors)

    tracemalloc.start()
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model = Slicefinder(k=10, max_l=2, min_sup=min_sup, verbose=False)
        model.fit(X, errors)
        times.append(time.perf_counter() - start)

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "time_ms": float(np.median(times) * 1000),
        "time_std_ms": float(np.std(times) * 1000),
        "peak_memory_mb": peak_memory / 1024 / 1024,
        "n_slices_found": len(model.top_slices_),
        "method": "fit",
    }


def benchmark_transform(
    X: np.ndarray,
    errors: np.ndarray,
    n_warmup: int = 1,
    n_runs: int = 3,
) -> dict[str, Any]:
    """Benchmark the transform method of Slicefinder.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    errors : np.ndarray
        Error vector.
    n_warmup : int
        Number of warmup runs.
    n_runs : int
        Number of timed runs.

    Returns
    -------
    dict
        Benchmark results including timing and memory.
    """
    min_sup = max(10, X.shape[0] // 100)

    model = Slicefinder(k=10, max_l=2, min_sup=min_sup, verbose=False)
    model.fit(X, errors)

    if len(model.top_slices_) == 0:
        return {
            "time_ms": 0.0,
            "time_std_ms": 0.0,
            "peak_memory_mb": 0.0,
            "method": "transform",
            "note": "No slices found",
        }

    for _ in range(n_warmup):
        model.transform(X)

    tracemalloc.start()
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.transform(X)
        times.append(time.perf_counter() - start)

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "time_ms": float(np.median(times) * 1000),
        "time_std_ms": float(np.std(times) * 1000),
        "peak_memory_mb": peak_memory / 1024 / 1024,
        "method": "transform",
    }


def benchmark_score_methods(
    X: np.ndarray,
    errors: np.ndarray,
    n_warmup: int = 1,
    n_runs: int = 5,
) -> dict[str, Any]:
    """Benchmark internal _score and _score_ub methods.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    errors : np.ndarray
        Error vector.
    n_warmup : int
        Number of warmup runs.
    n_runs : int
        Number of timed runs.

    Returns
    -------
    dict
        Benchmark results for internal methods.
    """
    min_sup = max(10, X.shape[0] // 100)

    model = Slicefinder(k=10, max_l=2, min_sup=min_sup, verbose=False)
    model.fit(X, errors)

    n_samples = X.shape[0]
    slice_sizes = np.random.randint(min_sup, n_samples, size=1000).astype(
        np.float64
    )
    slice_errors = np.random.random(1000) * slice_sizes
    max_slice_errors = np.random.random(1000)

    for _ in range(n_warmup):
        model._score(slice_sizes, slice_errors, n_samples)
        model._score_ub(slice_sizes, slice_errors, max_slice_errors, n_samples)

    score_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for _ in range(100):
            model._score(slice_sizes, slice_errors, n_samples)
        score_times.append((time.perf_counter() - start) / 100)

    score_ub_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for _ in range(100):
            model._score_ub(
                slice_sizes, slice_errors, max_slice_errors, n_samples
            )
        score_ub_times.append((time.perf_counter() - start) / 100)

    return {
        "_score": {
            "time_ms": float(np.median(score_times) * 1000),
            "time_std_ms": float(np.std(score_times) * 1000),
        },
        "_score_ub": {
            "time_ms": float(np.median(score_ub_times) * 1000),
            "time_std_ms": float(np.std(score_ub_times) * 1000),
        },
    }


def run_benchmarks_without_numba() -> dict[str, Any]:
    """Run benchmarks with numba disabled via mocking.

    Returns
    -------
    dict
        Benchmark results without numba optimization.
    """
    results = {}

    with patch("sliceline.slicefinder.NUMBA_AVAILABLE", False):
        for size_name, config in DATASET_CONFIGS.items():
            logger.info("Benchmarking %s dataset without numba...", size_name)
            X, errors = generate_synthetic_data(
                config["n_samples"], config["n_features"]
            )

            fit_result = benchmark_fit(X, errors)
            transform_result = benchmark_transform(X, errors)
            score_results = benchmark_score_methods(X, errors)

            results[f"{size_name}_fit"] = fit_result
            results[f"{size_name}_transform"] = transform_result
            results[f"{size_name}_score"] = score_results["_score"]
            results[f"{size_name}_score_ub"] = score_results["_score_ub"]

            logger.info(
                "Completed %s dataset without numba (%.1fms)",
                size_name,
                fit_result["time_ms"],
            )

    return results


def run_benchmarks_with_numba() -> dict[str, Any]:
    """Run benchmarks with numba enabled.

    Returns
    -------
    dict
        Benchmark results with numba optimization.
    """
    results = {}

    for size_name, config in DATASET_CONFIGS.items():
        logger.info("Benchmarking %s dataset with numba...", size_name)
        X, errors = generate_synthetic_data(
            config["n_samples"], config["n_features"]
        )

        fit_result = benchmark_fit(X, errors)
        transform_result = benchmark_transform(X, errors)
        score_results = benchmark_score_methods(X, errors)

        results[f"{size_name}_fit"] = fit_result
        results[f"{size_name}_transform"] = transform_result
        results[f"{size_name}_score"] = score_results["_score"]
        results[f"{size_name}_score_ub"] = score_results["_score_ub"]

        logger.info(
            "Completed %s dataset with numba (%.1fms)",
            size_name,
            fit_result["time_ms"],
        )

    return results


def calculate_speedups(
    without_numba: dict[str, Any], with_numba: dict[str, Any]
) -> dict[str, Any]:
    """Calculate speedup ratios and memory reduction between results.

    Parameters
    ----------
    without_numba : dict
        Results without numba.
    with_numba : dict
        Results with numba.

    Returns
    -------
    dict
        Speedup ratios and memory reduction for each benchmark.
    """
    speedups = {}

    for key in without_numba:
        if key in with_numba:
            without_time = without_numba[key].get("time_ms", 0)
            with_time = with_numba[key].get("time_ms", 0)

            result: dict[str, Any] = {}

            if with_time > 0 and without_time > 0:
                time_speedup = without_time / with_time
                result["time_speedup"] = f"{time_speedup:.2f}x"
            else:
                result["time_speedup"] = "N/A"

            without_mem = without_numba[key].get("peak_memory_mb")
            with_mem = with_numba[key].get("peak_memory_mb")

            if without_mem is not None and with_mem is not None:
                if without_mem > 0:
                    mem_reduction_pct = (
                        (without_mem - with_mem) / without_mem * 100
                    )
                else:
                    mem_reduction_pct = 0.0

                mem_saved_mb = without_mem - with_mem
                result["memory_reduction_pct"] = round(mem_reduction_pct, 2)
                result["memory_saved_mb"] = round(mem_saved_mb, 3)

            speedups[key] = result

    return speedups


def print_results_table(
    without_numba: dict[str, Any],
    with_numba: dict[str, Any],
    speedups: dict[str, Any],
) -> None:
    """Print formatted results table with timing and memory comparisons.

    Parameters
    ----------
    without_numba : dict
        Results without numba.
    with_numba : dict
        Results with numba.
    speedups : dict
        Speedup ratios and memory reduction metrics.
    """
    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS SUMMARY - TIMING")
    print("=" * 120)

    header = (
        f"{'Benchmark':<25} {'Without Numba':<18} {'With Numba':<18} "
        f"{'Speedup':<12}"
    )
    print(header)
    print("-" * 120)

    for key in sorted(without_numba.keys()):
        without_time = without_numba[key].get("time_ms", 0)
        with_time = with_numba.get(key, {}).get("time_ms", 0)
        speedup = speedups.get(key, {}).get("time_speedup", "N/A")

        without_str = f"{without_time:.2f}ms" if without_time > 0 else "N/A"
        with_str = f"{with_time:.2f}ms" if with_time > 0 else "N/A"

        print(f"{key:<25} {without_str:<18} {with_str:<18} {speedup:<12}")

    print("=" * 120)

    has_memory_data = any(
        without_numba[key].get("peak_memory_mb") is not None
        for key in without_numba
    )

    if has_memory_data and with_numba:
        print("\n" + "=" * 120)
        print("BENCHMARK RESULTS SUMMARY - MEMORY")
        print("=" * 120)

        mem_header = (
            f"{'Benchmark':<25} {'Without Numba':<18} {'With Numba':<18} "
            f"{'Memory Saved':<25}"
        )
        print(mem_header)
        print("-" * 120)

        for key in sorted(without_numba.keys()):
            without_mem = without_numba[key].get("peak_memory_mb")
            with_mem = with_numba.get(key, {}).get("peak_memory_mb")

            if without_mem is None:
                continue

            without_str = f"{without_mem:.2f} MB"
            with_str = f"{with_mem:.2f} MB" if with_mem is not None else "N/A"

            speedup_data = speedups.get(key, {})
            mem_saved = speedup_data.get("memory_saved_mb")
            mem_pct = speedup_data.get("memory_reduction_pct")

            if mem_saved is not None and mem_pct is not None:
                if mem_saved > 0:
                    saved_str = f"-{mem_saved:.3f} MB (-{mem_pct:.1f}%)"
                elif mem_saved < 0:
                    saved_str = f"+{-mem_saved:.3f} MB (+{-mem_pct:.1f}%)"
                else:
                    saved_str = "0.000 MB (0.0%)"
            else:
                saved_str = "N/A"

            print(
                f"{key:<25} {without_str:<18} {with_str:<18} {saved_str:<25}"
            )

        print("=" * 120)


def print_summary(
    without_numba: dict[str, Any],
    with_numba: dict[str, Any],
    speedups: dict[str, Any],
) -> None:
    """Print summary section showing overall performance impact.

    Parameters
    ----------
    without_numba : dict
        Results without numba.
    with_numba : dict
        Results with numba.
    speedups : dict
        Speedup ratios and memory reduction metrics.
    """
    if not speedups:
        return

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    time_speedups = []
    for key, data in speedups.items():
        speedup_str = data.get("time_speedup", "N/A")
        if speedup_str != "N/A":
            speedup_val = float(speedup_str.replace("x", ""))
            time_speedups.append((key, speedup_val))

    if time_speedups:
        avg_speedup = sum(s for _, s in time_speedups) / len(time_speedups)
        print("\nTime Performance:")
        print(f"  Average speedup across all operations: {avg_speedup:.2f}x")

        sorted_speedups = sorted(
            time_speedups, key=lambda x: x[1], reverse=True
        )
        print("\n  Operations benefiting most from Numba:")
        for i, (op, speedup) in enumerate(sorted_speedups[:5], 1):
            print(f"    {i}. {op}: {speedup:.2f}x")

        if len(sorted_speedups) > 5:
            print("\n  Operations with least benefit:")
            for i, (op, speedup) in enumerate(sorted_speedups[-3:], 1):
                print(f"    {i}. {op}: {speedup:.2f}x")

    memory_metrics = []
    for key, data in speedups.items():
        mem_saved = data.get("memory_saved_mb")
        mem_pct = data.get("memory_reduction_pct")
        if mem_saved is not None:
            memory_metrics.append((key, mem_saved, mem_pct))

    if memory_metrics:
        total_saved = sum(m[1] for m in memory_metrics)
        avg_pct = sum(m[2] for m in memory_metrics) / len(memory_metrics)

        print("\nMemory Performance:")
        if total_saved >= 0:
            print(
                f"  Total memory saved across fit operations: "
                f"{total_saved:.3f} MB"
            )
        else:
            print(
                f"  Total memory increase across fit operations: "
                f"{-total_saved:.3f} MB"
            )

        if avg_pct >= 0:
            print(f"  Average memory reduction: {avg_pct:.1f}%")
        else:
            print(f"  Average memory increase: {-avg_pct:.1f}%")

        print("\n  Memory impact by operation:")
        sorted_memory = sorted(
            memory_metrics, key=lambda x: x[1], reverse=True
        )
        for op, saved, pct in sorted_memory:
            if saved > 0:
                print(f"    {op}: -{saved:.3f} MB (-{pct:.1f}%)")
            elif saved < 0:
                print(f"    {op}: +{-saved:.3f} MB (+{-pct:.1f}%)")
            else:
                print(f"    {op}: 0.000 MB (0.0%)")

    print("\n" + "=" * 120)


def main() -> int:
    """Run comprehensive benchmarks.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    print("=" * 80)
    print("Sliceline Numba Optimization Benchmark")
    print("=" * 80)
    print()

    numba_available = is_numba_available()
    print(f"Numba available: {numba_available}")
    print()

    if not numba_available:
        logger.warning(
            "Numba is not installed. Install with: pip install numba "
            "or: pip install sliceline[optimized]"
        )
        logger.info("Running benchmarks without numba only...")

    logger.info("Running benchmarks WITHOUT numba optimization...")
    without_numba_results = run_benchmarks_without_numba()

    if numba_available:
        logger.info("Running benchmarks WITH numba optimization...")
        with_numba_results = run_benchmarks_with_numba()

        speedups = calculate_speedups(
            without_numba_results, with_numba_results
        )
    else:
        with_numba_results = {}
        speedups = {}

    print_results_table(without_numba_results, with_numba_results, speedups)
    print_summary(without_numba_results, with_numba_results, speedups)

    results = {
        "timestamp": datetime.now().isoformat(),
        "numba_available": numba_available,
        "datasets": DATASET_CONFIGS,
        "results": {
            "without_numba": without_numba_results,
            "with_numba": with_numba_results,
            "speedups": speedups,
        },
    }

    output_file = Path(__file__).parent / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to: %s", output_file)
    print(f"\nResults saved to: {output_file}")

    return 0


def _configure_logging() -> None:
    """Configure logging for the benchmark script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    _configure_logging()
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception("Error running benchmarks: %s", e)
        sys.exit(1)
