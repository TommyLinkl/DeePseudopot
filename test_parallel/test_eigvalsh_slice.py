import time
import torch
import multiprocessing as mp
import numpy as np
import os
import psutil 

def child_proc(H):
    start_time = time.time()
    eigenvalues = torch.linalg.eigvalsh(H)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Diagonalize the ham. Time: {total_time:.2f} seconds")

    return eigenvalues

def generate_and_diagonalize_matrix(globalH, num_processes=None):
    start_time = time.time()

    if num_processes is not None:   # parallel
        with mp.Pool(num_processes) as pool:
            args_list = [(globalH[:,:,idx]) for idx in range(num_processes)]
            eigenvalues_list = pool.starmap(child_proc, args_list)
        eigenvalues = torch.stack(eigenvalues_list)
    else:   # no parallel
        eigenvalues = child_proc(globalH[:,:,0])

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time:.2f} seconds")

    return eigenvalues

if __name__ == "__main__":
    # torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    print(psutil.cpu_count(logical=False))
    print(psutil.cpu_count(logical=True))

    globalH = torch.randn(2000, 2000, 100, dtype=torch.complex128)

    print("\nNo multiprocessing: ")
    generate_and_diagonalize_matrix(globalH)

    for num_processes in range(1, 9): 
        print(f"\nNumber of processes = {num_processes}")
        print("Multiprocessing: ")
        generate_and_diagonalize_matrix(globalH, num_processes=num_processes)

