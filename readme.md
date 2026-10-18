GPU Kernels Exploratory Analysis

Implements 7 different forms of Matrix Multiplication:

1. Pure Python CPU Implementation Using for loops and arrays
2. Numpy Built in MatMul
3. PyTorch Matmul
4. OpenAI's Triton naive MatMul (1 pid per output element)
5. Triton Blocked MatMul