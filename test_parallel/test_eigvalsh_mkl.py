import time
import torch
import multiprocessing as mp
import numpy as np
import os
import psutil 

def diagonalize_matrix(size, num_processes):
    start_time = time.time()
    H = torch.randn(size, size, dtype=torch.complex128)
    eigenvalues = torch.linalg.eigvalsh(H).numpy()
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Time: {total_time:.2f} seconds")

    return eigenvalues

def generate_and_diagonalize_matrix(matrix_size=2000, num_processes=None):
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
    print(psutil.cpu_count(logical=False))
    print(psutil.cpu_count(logical=True))

    for num_processes in range(1, 16): 
        print(f"\nNumber of processes = {num_processes}")
        os.environ["OMP_NUM_THREADS"] = str(num_processes)
        os.environ["MKL_NUM_THREADS"] = str(num_processes)

        generate_and_diagonalize_matrix()