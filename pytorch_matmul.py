
import time
import torch
import triton


def pytorch_matmul_cpu(A, B):
    return torch.matmul(A, B)


def benchmark_cpu(A, B, iterations):
    # Warm-up
    start = time.perf_counter()
    pytorch_matmul_cpu(A, B)
    warmup = time.perf_counter() - start

    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        pytorch_matmul_cpu(A, B)
        end = time.perf_counter()
        times.append(end - start)

    return warmup, times


def benchmark_gpu(A, B, iterations):
    # Keep input transfers outside the timed function.
    A_gpu = A.to("cuda")
    B_gpu = B.to("cuda")

    # Record one synchronized warm-up call separately from benchmark samples.
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.matmul(A_gpu, B_gpu)
    torch.cuda.synchronize()
    warmup = time.perf_counter() - start

    times = [
        triton.testing.do_bench_cudagraph(lambda: torch.matmul(A_gpu, B_gpu))
        for _ in range(iterations)
    ]
    return warmup, times


def main(iterations):
    size = 1024
    A = torch.full((size, size), 2.0, dtype=torch.float32)
    B = torch.full((size, size), 2.0, dtype=torch.float32)

    results = {
        "CPU": benchmark_cpu(A, B, iterations)
    }

    if torch.cuda.is_available():
        results["GPU"] = benchmark_gpu(A, B, iterations)
    else:
        print("CUDA is not available. Skipping GPU matmul.")

    return results
