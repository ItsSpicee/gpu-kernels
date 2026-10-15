import python_matmul
import numpy_matmul
import pytorch_matmul
import triton_blocked_matmul
import triton_naive_matmul
import cuda_bench_wrapper


import argparse


def run_tests(iterations):
    benchmarks = [
        ("Python", python_matmul.main),
        ("NumPy", numpy_matmul.main),
        ("PyTorch", pytorch_matmul.main),
        ("Triton Blocked", triton_blocked_matmul.main),
        ("Triton Naive", triton_naive_matmul.main),
        ("Cuda Naive", cuda_bench_wrapper.naive_bench),
        ("Cuda Tiled", cuda_bench_wrapper.tiled_bench)
    ]

    results = []

    for name, benchmark_fn in benchmarks:
        print(f"Running {name} benchmark...")
        result = benchmark_fn(iterations)
        results.append((name, result))

    return results

def positive_int(value):
    """Custom validator to ensure the value is an integer greater than 0."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer.")
        
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"'{value}' must be an integer greater than 0.")
        
    return ivalue

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--iterations", type=positive_int, default=100, help="Number of iterations > 0. Default is 100.")

    args = parser.parse_args()

    results = run_tests(args.iterations)
    
    print(results)



if __name__ == "__main__":
    main()


