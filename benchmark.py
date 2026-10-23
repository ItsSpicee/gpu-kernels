
import argparse
import csv
import statistics
import sys
from pathlib import Path

IMPLEMENTATIONS_DIR = Path(__file__).resolve().parent / "Matmul Implementations"
sys.path[:0] = [
    str(IMPLEMENTATIONS_DIR),
    str(IMPLEMENTATIONS_DIR / "Kernels"),
    str(IMPLEMENTATIONS_DIR / "Libraries"),
]

import python_matmul
import numpy_matmul
import pytorch_matmul
import triton_blocked_matmul
import triton_naive_matmul
import cuda_bench_wrapper

DATA_DIR = Path(__file__).resolve().parent / "Data"


def positive_int(value):
    """Validate that the iteration count is a positive integer."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Value must be a positive integer."
        )

    if number <= 0:
        raise argparse.ArgumentTypeError(
            "Value must be greater than zero."
        )

    return number


def run_tests(iterations, skip_python=False):
    """Run each benchmark and collect its results."""
    benchmarks = [
        ("Python", python_matmul.main),
        ("NumPy", numpy_matmul.main),
        ("PyTorch", pytorch_matmul.main),
        ("Triton Blocked", triton_blocked_matmul.main),
        ("Triton Naive", triton_naive_matmul.main),
        ("CUDA Naive", cuda_bench_wrapper.naive_bench),
        ("CUDA Tiled", cuda_bench_wrapper.tiled_bench),
    ]

    if skip_python:
        benchmarks = [
        (name, fn) for name, fn in benchmarks
        if name != "Python"
    ]

    results = []

    for name, benchmark_function in benchmarks:
        print(f"\nRunning {name} benchmark...")

        try:
            result = benchmark_function(iterations)
            results.append((name, result))
            print(f"{name} benchmark completed.")

        except Exception as error:
            print(f"{name} benchmark failed: {error}")

    return results


def expand_results(results):
    """
    Normalize benchmark results into individual records.

    Each benchmark must return either:
        (warmup_seconds, [iteration_times_seconds])

    Or a dictionary:
        {
            "CPU": (warmup_seconds, [iteration_times_seconds]),
            "GPU": (warmup_seconds, [iteration_times_seconds]),
        }
    """
    for name, result in results:
        if isinstance(result, dict):
            result_items = result.items()
        else:
            result_items = [(None, result)]

        for device, measurements in result_items:
            implementation = (
                name if device is None else f"{name} {device}"
            )

            try:
                warmup, times = measurements
                times = list(times)

                if not times:
                    print(
                        f"Warning: {implementation} returned no "
                        "iteration timings; skipping it."
                    )
                    continue

                yield implementation, warmup, times

            except (TypeError, ValueError) as error:
                print(
                    f"Warning: Invalid results for {implementation}: "
                    f"{error}"
                )


def save_results_csv(
    results,
    filename=DATA_DIR / "benchmark_results.csv",
):
    """Save summary statistics for each implementation."""
    fieldnames = [
        "Implementation",
        "Iterations",
        "Warmup (ms)",
        "Mean (ms)",
        "Median (ms)",
        "Min (ms)",
        "Max (ms)",
        "Std Dev (ms)",
    ]

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for implementation, warmup, times in expand_results(results):
            mean_time = statistics.mean(times)

            std_dev = (
                statistics.stdev(times)
                if len(times) > 1
                else 0.0
            )

            writer.writerow({
                "Implementation": implementation,
                "Iterations": len(times),
                "Warmup (ms)": warmup * 1000,
                "Mean (ms)": mean_time * 1000,
                "Median (ms)": statistics.median(times) * 1000,
                "Min (ms)": min(times) * 1000,
                "Max (ms)": max(times) * 1000,
                "Std Dev (ms)": std_dev * 1000,
            })

    print(f"\nSummary results saved to {filename}")


def save_raw_logs(
    results,
    filename=DATA_DIR / "benchmark_raw_logs.csv",
):
    """Save warm-up and individual iteration timings."""
    fieldnames = [
        "Implementation",
        "Iteration",
        "Time (ms)",
    ]

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for implementation, warmup, times in expand_results(results):
            # Iteration 0 represents the warm-up measurement.
            writer.writerow({
                "Implementation": implementation,
                "Iteration": 0,
                "Time (ms)": warmup * 1000,
            })

            # Iterations 1 onward represent measured runs.
            for iteration, elapsed in enumerate(times, start=1):
                writer.writerow({
                    "Implementation": implementation,
                    "Iteration": iteration,
                    "Time (ms)": elapsed * 1000,
                })

    print(f"Raw timing logs saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark matrix multiplication implementations."
    )

    parser.add_argument(
        "--iterations",
        type=positive_int,
        default=100,
        help="Number of timed iterations per benchmark (default: 100).",
    )

    parser.add_argument(
        "--skip-python",
        action="store_true",
        default=False,
        help="Whether to skip the pure Python loop implementation."
    )

    args = parser.parse_args()

    

    print(f"Starting benchmarks with {args.iterations} iterations.")

    results = run_tests(args.iterations, args.skip_python)

    if not results:
        print("\nNo benchmarks completed successfully.")
        return

    save_results_csv(results)
    save_raw_logs(results)

    print("\nAll completed benchmarks have been processed.")


if __name__ == "__main__":
    main()
