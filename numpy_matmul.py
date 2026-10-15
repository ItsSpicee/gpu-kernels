import numpy as np
import time


def benchmark(A,B, iterations):
    warmup_time = 0
    times = []

    #warmup
    warmup_start = time.perf_counter()
    C = np.matmul(A, B)
    warmup_end = time.perf_counter()


    for _ in range(iterations):
        start = time.perf_counter()
        C = np.matmul(A, B)
        end = time.perf_counter()

        times.append(end - start)
    
    return warmup_end - warmup_start, times




def main(iterations):
    size = 1024
    #init arrays
    A = np.full((size, size), 2.0, dtype=np.float32)
    B = np.full((size, size), 2.0, dtype=np.float32)
    
    return benchmark(A, B, iterations)