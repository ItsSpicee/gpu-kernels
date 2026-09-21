#include <cuda_runtime.h>
#define TILE_SIZE = 32

// Kernel definition
__global__ void MatMul_naive(float* A, float* B, float* C, int M, int N, int K)
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

int main()
{
    const int M = 1024; 
    const int N = 1024;
    const int K = 1024;

    size_t size_A = sizeof(float) * M * K
    size_t size_B = sizeof(float) * K * N
    size_t size_C = sizeof(float) * M * N


    float* host_A = new float[M * K];
    float* host_B = new float[K * N];
    float* host_C = new float[M * N];

    //initialize Matrix A and B with values
    for (int i = 0; i < M * K; i++) host_A[i] = 2.0f;
    for (int i = 0; i < K * N; i++) host_B[i] = 2.0f;

    float* device_A;
    float* device_B;
    float* device_C;

    cudaMalloc(&device_A, size_A);
    cudaMalloc(&device_B, size_B);
    cudaMalloc(&device_C, size_C);

    cudaMemcpy(device_A, host_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size_B, cudaMemcpyHostToDevice);

    dim3 blocks(32, 32); //32 x 32 grid of blocks
    dim3 threads(32, 32); //each block has 32 x 32 threads 
    //total = 32 x 32 x 32 x 32 = 2^20 for 1024 x 1024 matmul

    MatMul_naive<<<blocks, threads>>>(device_A, device_B, device_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(host_C, device_C, size_C, cudaMemcpyDeviceToHost);

    //clean up
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    delete[] host_A;
    delete[] host_B;
    delete[] host_C;
    
    

}