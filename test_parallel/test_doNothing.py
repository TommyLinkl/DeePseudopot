import time
import torch
import multiprocessing as mp
import numpy as np
import os
import psutil 

def doNothing(matrix):
    start_time = time.time()
    tmp = matrix[:,5]
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Time: {total_time:.2f} seconds")

    return tmp

def generate_and_diagonalize_matrix(matrix_size=2000, num_processes=None):
    HList = []
    for _ in range(num_processes): 
        H = torch.randn(matrix_size, matrix_size, dtype=torch.complex128)
        HList.append(H)
    
    start_time = time.time()

    if num_processes is not None:   # parallel
        with mp.Pool(num_processes) as pool:
            args_list = [(matrix_size, num_processes) for _ in range(num_processes)]
            eigenvalues_list = pool.starmap(diagonalize_matrix, args_list)
            eigenvalues = np.concatenate(eigenvalues_list, axis=0)

    else:   # no parallel
        eigenvalues = diagonalize_matrix(matrix_size, num_processes)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time:.2f} seconds")

    return eigenvalues

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    print(psutil.cpu_count(logical=False))
    print(psutil.cpu_count(logical=True))

    for num_processes in range(1, 16): 
        print(f"\nNumber of processes = {num_processes}")
        print("Multiprocessing: ")
        generate_and_diagonalize_matrix(num_processes=num_processes)

    print("\nNo multiprocessing: ")
    generate_and_diagonalize_matrix()
