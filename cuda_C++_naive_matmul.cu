#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#define TILE_SIZE 32

// Kernel definition
__global__ void MatMul_naive(const float* A, const float* B, float* C, int M, int N, int K)
{
    int row_idx  = threadIdx.y + blockDim.y * blockIdx.y;
    int col_idx = threadIdx.x + blockDim.x * blockIdx.x;

    
    if(row_idx < M && col_idx < N) {
        float sum = 0.0f;
        for(int i = 0; i < K; i++){

            sum += A[(row_idx * K) + i] * B[(i * N) + col_idx];

        }
        C[(row_idx * N) + col_idx]  = sum;
    }
    
        
}

void naive_matmul(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C
){

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    dim3 blocks(32, 32); //32 x 32 grid of blocks
    dim3 threads(32, 32); //each block has 32 x 32 threads 
    //total = 32 x 32 x 32 x 32 = 2^20 for 1024 x 1024 matmul

    auto stream = c10::cuda::getCurrentCUDAStream();


    MatMul_naive<<<blocks, threads, 0, stream.stream()>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_matmul", &naive_matmul);
}