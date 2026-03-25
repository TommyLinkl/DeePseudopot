# config_threads.py
import os
import torch

def configure_threads(n_threads=1):
    # Environment variables (apply before libs initialize)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

    # Torch runtime
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(1)  # reduce inter-op thread contention

    print(f"[config_threads] Using {n_threads} threads for BLAS + Torch.")
