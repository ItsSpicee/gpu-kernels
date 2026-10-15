import torch
import triton
import triton.language as tl


BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 32

@triton.jit
def blocked_matmul_kernel(
    A_ptr, B_ptr, C_ptr, 
    M, N, K,
    a_stride_m, a_stride_k,
    b_stride_k, b_stride_n,
    c_stride_m, c_stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr 
):
    pid = tl.program_id(axis = 0)

    #Get how many columns of blocks we have
    num_block_cols = tl.cdiv(N, BLOCK_SIZE_N)

    block_m = pid // num_block_cols #what block of rows am I  responsible for?
    block_n = pid % num_block_cols #what block of columns am I responsible for?

    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    #calculate the individual pointers for the whole block, using combined horizontal and vertical strides for each starting index
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_m + offs_k[None, :] * a_stride_k)
    b_ptrs = B_ptr + (offs_k[:, None] * b_stride_k + offs_n[None, :] * b_stride_n)

    #init accumulator matrix
    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  

    #get the K-wise series of blocks needed to caculate the result in the C Matrix block
    for i in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        #ensure that the current block of A is within bounds
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - i * BLOCK_SIZE_K)
        #load block vals from a
        a_vals = tl.load(a_ptrs, mask = a_mask, other=0.0)

        #ensure that the current block of B is within bounds
        b_mask = (offs_k[:, None] < K - i * BLOCK_SIZE_K) & (offs_n[None, :] < N) 
        b_vals = tl.load(b_ptrs, mask = b_mask, other=0.0)

        #accumulate
        accum += tl.dot(a_vals,b_vals)

        #get the next K-wise block of A
        a_ptrs += BLOCK_SIZE_K * a_stride_k
        #get the next K-wise block of B
        b_ptrs += BLOCK_SIZE_K * b_stride_k


    #not sure the exact purpose of this, something about matching types?
    c = accum.to(A_ptr.dtype.element_ty)

    #calculate the memory locations of the resulting elements in c that should be set by the current pid calculation
    #by combining the row and column indices scaled by the stride factor in C
    c_offsets = c_stride_m * offs_m[:, None] + c_stride_n * offs_n[None, :]

    #another masks to prevent oopsies
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    #store the result as long as the mask allows is
    tl.store(C_ptr + c_offsets, c, mask = c_mask)




def main(iterations):
    size = 1024
    A = torch.full((size, size), 2.0, dtype=torch.float32, device="cuda")
    B = torch.full((size, size), 2.0, dtype=torch.float32, device="cuda")

    M, K = A.shape
    K_2, N = B.shape

    assert K == K_2

    C = torch.empty((M, N), device = A.device, dtype = A.dtype) #use the same device as A (same gpu)

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    #warmup benchmark using python time
    torch.cuda.synchronize()
    start = time.perf_counter()
    blocked_matmul_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K
            )
    torch.cuda.synchronize()
    warmup = time.perf_counter() - start

    #benchmark
    times = [
        triton.testing.do_bench_cudagraph( 
            lambda: blocked_matmul_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K
            )
        )
        for _ in range(iterations)
    ]

    return warmup, times
   