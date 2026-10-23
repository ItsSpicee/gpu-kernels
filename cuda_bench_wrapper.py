from torch.utils.cpp_extension import load
import torch
import triton
import time
from pathlib import Path
import os

SIZE = 1024

CUDA_LIB_DIR = str(
    Path(os.environ["CONDA_PREFIX"]) / "Library" / "lib"
)

def naive_bench(iterations):

    A = torch.full((SIZE, SIZE), 2.0, dtype=torch.float32, device="cuda")
    B = torch.full((SIZE, SIZE), 2.0, dtype=torch.float32, device="cuda")
    C = torch.empty((SIZE, SIZE), device = A.device, dtype = A.dtype) #use the same device as A (same gpu)

    cuda_naive = load(
        name="cuda_naive",
        sources=["cuda_C++_naive_matmul.cu"],
        extra_ldflags=[f"/LIBPATH:{CUDA_LIB_DIR}", "cudart.lib"],
    )

    torch.cuda.synchronize()
    start = time.perf_counter()
    cuda_naive.naive_matmul(A, B, C)
    torch.cuda.synchronize()
    warmup = time.perf_counter() - start

    # A, B, and C are preallocated CUDA float32 tensors.
    times = [
        triton.testing.do_bench_cudagraph(
            lambda: cuda_naive.naive_matmul(A, B, C)
        ) / 1000.0  # CUDA graph milliseconds -> benchmark seconds.
        for _ in range(iterations)
    ]

    return warmup, times


def tiled_bench(iterations):

    A = torch.full((SIZE, SIZE), 2.0, dtype=torch.float32, device="cuda")
    B = torch.full((SIZE, SIZE), 2.0, dtype=torch.float32, device="cuda")
    C = torch.empty((SIZE, SIZE), device = A.device, dtype = A.dtype) #use the same device as A (same gpu)

    cuda_tiled = load(
        name="cuda_tiled",
        sources=["cuda_C++_tiled_matmul.cu"],
        extra_ldflags=[f"/LIBPATH:{CUDA_LIB_DIR}", "cudart.lib"],
    )

    torch.cuda.synchronize()
    start = time.perf_counter()
    cuda_tiled.tiled_matmul(A, B, C)
    torch.cuda.synchronize()
    warmup = time.perf_counter() - start

    # A, B, and C are preallocated CUDA float32 tensors.
    times = [
        triton.testing.do_bench_cudagraph(
            lambda: cuda_tiled.tiled_matmul(A, B, C)
        ) / 1000.0  # CUDA graph milliseconds -> benchmark seconds.
        for _ in range(iterations)
    ]

    return warmup, times




