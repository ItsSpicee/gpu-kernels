#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#define TILE_SIZE 32

// Kernel definition
__global__ void MatMul_Tiled(const float* A, const float* B, float* C, int M, int N, int K)
{
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_idx  = threadIdx.y + blockDim.y * blockIdx.y;
    int col_idx = threadIdx.x + blockDim.x * blockIdx.x;

    float sum = 0.0f;

    
    for(int phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; phase++){

        a_tile[ty][tx] = A[(row_idx * K) + (phase * TILE_SIZE) + tx];
        b_tile[ty][tx] = B[((phase * TILE_SIZE + ty) * N ) + col_idx];
        __syncthreads();
        
        for(int k = 0; k < TILE_SIZE; k++){
            sum += a_tile[ty][k] * b_tile[k][tx];
        }
        __syncthreads();

    }
    if (row_idx < M && col_idx < N){
        C[(row_idx * N) + col_idx]  = sum;
    }

    
        
}

void tiled_matmul(
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


    MatMul_Tiled<<<blocks, threads, 0, stream.stream()>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled_matmul", &tiled_matmul);
}