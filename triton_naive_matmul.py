    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def naive_matmul_kernel(
        A_ptr, B_ptr, C_ptr, 
        M, N, K,
        a_stride_m, a_stride_k,
        b_stride_k, b_stride_n,
        c_stride_m, c_stride_n
    ):
        pid = tl.program_id(axis = 0)

        m = pid // N #what row am I (pid) responsible for?
        n = pid % N #what column am I responsible for?

        accum = 0.0 #init 

        base_row_idx = A_ptr + (m * a_stride_m) #find the start of row m in memory
        base_col_idx = B_ptr + (n * b_stride_n) #find the start of row n in memory

        #loop through all K pairs of elements
        for i in range(K):

            a_val = tl.load(base_row_idx + (i * a_stride_k))
            b_val = tl.load((i * b_stride_k) + base_col_idx )

            accum += a_val * b_val
        
        tl.store(C_ptr + (m * c_stride_m) + (n * c_stride_n), accum)



    def naive_matmul(A, B):

        M, K = A.shape
        K_2, N = B.shape

        assert K == K_2

        C = torch.zeros((M, N), device = A.device, dtype = A.dtype)

        grid = (M * N,)

        naive_matmul_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1)

        )