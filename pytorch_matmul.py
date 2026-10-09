import torch


def pytorch_matmul_cpu(A, B):
    return torch.matmul(A, B)

def pytorch_matmul_gpu(A, B):
    A_gpu = A.to("cuda")
    B_gpu = B.to("cuda")

    C_gpu = torch.matmul(A_gpu, B_gpu)

    return C_gpu


def main():
    A = torch.full((1024, 1024), 2.0, dtype=torch.float32)
    B = torch.full((1024, 1024), 2.0, dtype=torch.float32)

    C_cpu = pytorch_matmul_cpu(A, B)

    if torch.cuda.is_available():
        C_gpu = pytorch_matmul_gpu(A, B)
        
    else:
        print("CUDA is not available. Skipping GPU matmul.")