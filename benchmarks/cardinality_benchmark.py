"""
Cardinality Benchmark Script for Sliceline Performance Testing.

This script profiles the performance of Slicefinder across different
cardinality levels to measure the impact of optimizations.

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


def profile_implementation(X, errors, **kwargs):
    """Profile a Slicefinder implementation.

    Parameters
    ----------
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
        model = Slicefinder(k=10, max_l=2, verbose=False, **kwargs)
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
    print()

    # Define cardinality levels to test
    cardinalities = [10, 100, 500, 1000]
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

        # Benchmark implementation
        print("  Running Slicefinder...", end=" ", flush=True)
        result = profile_implementation(X, errors)
        if result["success"]:
            print(f"Done ({result['time_seconds']:.2f}s)")
        else:
            print(f"Failed ({result.get('error', 'Unknown error')})")

        result_entry = {
            "cardinality": card,
            "n_samples": n_samples,
            "n_features": n_features,
            "result": result,
        }
        results.append(result_entry)

        if result["success"]:
            print(f"    Time: {result['time_seconds']:.2f}s")
            print(f"    Memory: {result['peak_memory_mb']:.1f} MB")
            print(f"    Slices found: {result['n_slices_found']}")

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Cardinality':<12} {'Time (s)':<15} {'Memory (MB)':<15} {'Slices':<10}")
    print("-" * 70)

    for r in results:
        card = r["cardinality"]
        res = r["result"]

        if res["success"]:
            time_str = f"{res['time_seconds']:.2f}"
            mem_str = f"{res['peak_memory_mb']:.1f}"
            slices_str = str(res["n_slices_found"])
        else:
            time_str = "FAILED"
            mem_str = "-"
            slices_str = "-"

        print(f"{card:<12} {time_str:<15} {mem_str:<15} {slices_str:<10}")

    # Compare with baseline (lowest cardinality)
    if results[0]["result"]["success"] and len(results) > 1:
        baseline = results[0]["result"]
        print("\n" + "-" * 70)
        print("Scaling relative to cardinality=10:")
        print("-" * 70)

        for r in results[1:]:
            if r["result"]["success"]:
                time_ratio = r["result"]["time_seconds"] / baseline["time_seconds"]
                mem_ratio = r["result"]["peak_memory_mb"] / baseline["peak_memory_mb"]
                print(
                    f"  Cardinality {r['cardinality']}: "
                    f"{time_ratio:.1f}x time, {mem_ratio:.1f}x memory"
                )

    # Save results to JSON
    output_file = Path(__file__).parent / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
    print("=" * 70)

    return results


def run_dataset_size_benchmarks():
    """Run benchmarks varying dataset size."""
    print("\n" + "=" * 70)
    print("Dataset Size Scaling Benchmark")
    print("=" * 70)
    print()

    sizes = [1000, 5000, 10000, 25000, 50000]
    cardinality = 50
    n_features = 10

    results = []

    for n_samples in sizes:
        print(f"Testing {n_samples} samples...", end=" ", flush=True)

        np.random.seed(42)
        X = np.random.randint(0, cardinality, size=(n_samples, n_features))
        errors = np.random.random(n_samples)

        result = profile_implementation(X, errors, min_sup=max(10, n_samples // 100))

        if result["success"]:
            print(f"Done ({result['time_seconds']:.2f}s)")
            results.append(
                {"n_samples": n_samples, "time": result["time_seconds"], "memory": result["peak_memory_mb"]}
            )
        else:
            print(f"Failed ({result.get('error')})")

    if len(results) > 1:
        print("\nScaling Analysis:")
        print("-" * 50)
        baseline = results[0]
        for r in results[1:]:
            size_ratio = r["n_samples"] / baseline["n_samples"]
            time_ratio = r["time"] / baseline["time"]
            print(
                f"  {r['n_samples']} samples: "
                f"{size_ratio:.0f}x size -> {time_ratio:.1f}x time"
            )

    return results


if __name__ == "__main__":
    try:
        run_benchmarks()
        run_dataset_size_benchmarks()
        sys.exit(0)
    except Exception as e:
        print(f"\nError running benchmarks: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
