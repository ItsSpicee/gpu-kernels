import torch
import triton
import triton.language as tl

def reference_matmul(A, B):

    M, K = A.shape
    K_2, N = B.shape

    assert K == K_2

    C = torch.zeros(M, N)

    def dot_product(v1, v2):

        accum = 0
        for a,b in zip(v1, v2):
            accum += a*b

        return accum

    num_pids = list(range(M*N))

    for i in num_pids:

        m = i // N # current row
        n = i % N # current column

        C[m, n] = dot_product(A[m, :], B[:, n])
    return C