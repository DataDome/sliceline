"""
Cardinality Benchmark Script for Sliceline Optimizations.

This script profiles the performance of Slicefinder vs OptimizedSlicefinder
across different cardinality levels to measure the impact of optimizations.

Usage:
    python benchmarks/cardinality_benchmark.py

Output:
    - benchmark_results.json: Detailed results
    - Console output with summary table
"""

import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

# Add parent directory to path to import sliceline
sys.path.insert(0, str(Path(__file__).parent.parent))

from sliceline import Slicefinder  # noqa: E402
from sliceline.optimized_slicefinder import (  # noqa: E402
    NUMBA_AVAILABLE,
    OptimizedSlicefinder,
)


def profile_implementation(SlicefinderClass, X, errors, **kwargs):
    """Profile a Slicefinder implementation.

    Parameters
    ----------
    SlicefinderClass : class
        Slicefinder or OptimizedSlicefinder class.
    X : np.ndarray
        Input data.
    errors : np.ndarray
        Error vector.
    **kwargs : dict
        Additional keyword arguments for the Slicefinder.

    Returns
    -------
    results : dict
        Dictionary containing timing and memory statistics.
    """
    tracemalloc.start()
    start = time.perf_counter()

    try:
        model = SlicefinderClass(k=10, max_l=2, verbose=False, **kwargs)
        model.fit(X, errors)

        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "time_seconds": elapsed,
            "peak_memory_mb": peak / 1024 / 1024,
            "n_slices_found": len(model.top_slices_),
            "success": True,
        }
    except MemoryError:
        tracemalloc.stop()
        return {
            "time_seconds": None,
            "peak_memory_mb": None,
            "n_slices_found": None,
            "success": False,
            "error": "MemoryError",
        }
    except Exception as e:
        tracemalloc.stop()
        return {
            "time_seconds": None,
            "peak_memory_mb": None,
            "n_slices_found": None,
            "success": False,
            "error": str(e),
        }


def run_benchmarks():
    """Run comprehensive benchmarks across cardinality levels."""
    print("=" * 70)
    print("Sliceline Cardinality Benchmark")
    print("=" * 70)
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print()

    # Define cardinality levels to test
    cardinalities = [10, 100, 1000, 10000]
    n_samples = 10000
    n_features = 5

    results = []

    for card in cardinalities:
        print(f"\nTesting cardinality: {card} unique values per feature")
        print("-" * 70)

        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randint(0, card, size=(n_samples, n_features))
        errors = np.random.random(n_samples)

        # Benchmark original implementation
        print("  Running original Slicefinder...", end=" ", flush=True)
        orig_result = profile_implementation(Slicefinder, X, errors)
        if orig_result["success"]:
            print(f"✓ ({orig_result['time_seconds']:.2f}s)")
        else:
            print(f"✗ ({orig_result.get('error', 'Unknown error')})")

        # Benchmark optimized implementation
        print("  Running OptimizedSlicefinder...", end=" ", flush=True)
        opt_result = profile_implementation(
            OptimizedSlicefinder, X, errors, max_features_per_column=500
        )
        if opt_result["success"]:
            print(f"✓ ({opt_result['time_seconds']:.2f}s)")
        else:
            print(f"✗ ({opt_result.get('error', 'Unknown error')})")

        # Calculate improvements
        if orig_result["success"] and opt_result["success"]:
            speedup = orig_result["time_seconds"] / opt_result["time_seconds"]
            memory_reduction = (
                1
                - opt_result["peak_memory_mb"] / orig_result["peak_memory_mb"]
            )

            print("\n  Performance Improvements:")
            print(f"    Speedup: {speedup:.2f}x")
            print(f"    Memory reduction: {memory_reduction*100:.1f}%")

            results.append(
                {
                    "cardinality": card,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "original": orig_result,
                    "optimized": opt_result,
                    "speedup": speedup,
                    "memory_reduction_percent": memory_reduction * 100,
                }
            )
        elif not orig_result["success"] and opt_result["success"]:
            print("\n  Performance Improvements:")
            print(f"    Original: FAILED ({orig_result.get('error')})")
            print("    Optimized: SUCCESS")

            results.append(
                {
                    "cardinality": card,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "original": orig_result,
                    "optimized": opt_result,
                    "speedup": float("inf"),
                    "memory_reduction_percent": 100,
                }
            )
        else:
            results.append(
                {
                    "cardinality": card,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "original": orig_result,
                    "optimized": opt_result,
                    "speedup": None,
                    "memory_reduction_percent": None,
                }
            )

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(
        f"{'Cardinality':<12} {'Original':<15} {'Optimized':<15} {'Speedup':<10}"
    )
    print("-" * 70)

    for result in results:
        card = result["cardinality"]
        orig = result["original"]
        opt = result["optimized"]

        if orig["success"]:
            orig_str = f"{orig['time_seconds']:.2f}s"
        else:
            orig_str = "FAILED"

        if opt["success"]:
            opt_str = f"{opt['time_seconds']:.2f}s"
        else:
            opt_str = "FAILED"

        if result["speedup"] is not None:
            if result["speedup"] == float("inf"):
                speedup_str = "∞"
            else:
                speedup_str = f"{result['speedup']:.2f}x"
        else:
            speedup_str = "N/A"

        print(f"{card:<12} {orig_str:<15} {opt_str:<15} {speedup_str:<10}")

    # Save results to JSON
    output_file = Path(__file__).parent / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    try:
        results = run_benchmarks()

        # Exit with success
        sys.exit(0)
    except Exception as e:
        print(f"\nError running benchmarks: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
