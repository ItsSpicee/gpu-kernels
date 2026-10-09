import numpy as np 

def numpy_matmul(A, B):

    return np.matmul(A, B)


def main():

    A = np.full((1024, 1024), 2, dtype = np.float32)
    B = np.full((1024, 1024), 2, dtype = np.float32)

    C = numpy_matmul(A, B)
    
