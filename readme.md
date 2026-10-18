GPU Kernels Exploratory Analysis

Implements 7 different forms of Matrix Multiplication:

1. Pure Python CPU Implementation Using for loops and arrays
2. Numpy Built in MatMul
3. PyTorch Matmul
4. OpenAI's Triton naive MatMul (1 pid per output element)
5. Triton Blocked MatMul (1 pid per "block" of output elements)
6. Nvidia CUDA C++ Naive MatMul (1 thread per output element)
7. Nvidia CUDA C++ tiled MatMul (1 thread per output element, but groups threads into localized "blocks" that cooperatively load tiles of input matrices to increase gpu cache usage and decrease global memory traffic)