
import time


def reference_matmul(A, B):
    M = len(A)
    K = len(A[0])
    K_2 = len(B)
    N = len(B[0])

    assert K == K_2, "Incompatible matrix dimensions"

    C = [[0.0 for _ in range(N)] for _ in range(M)]

    for i in range(M * N):
        m = i // N
        n = i % N

        accum = 0.0
        for k in range(K):
            accum += A[m][k] * B[k][n]

        C[m][n] = accum

    return C


def benchmark(A, B, iterations):
    # Warm-up
    start = time.perf_counter()
    reference_matmul(A, B)
    warmup = time.perf_counter() - start

    # Timed iterations
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        reference_matmul(A, B)
        end = time.perf_counter()

        times.append(end - start)

    return warmup, times


def main(iterations):
    size = 1024

    A = [[2.0 for _ in range(size)] for _ in range(size)]
    B = [[2.0 for _ in range(size)] for _ in range(size)]

    return benchmark(A, B, iterations)
