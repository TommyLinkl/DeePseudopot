import sys
sys.path.append('..')
import torch
import os
from main import main

torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

if len(sys.argv) != 2:
    print("Usage: python test_memory.py <testCase>")
    sys.exit(1)

testCase = sys.argv[1]
print("Input argument:", testCase)
main(f"inputs_{testCase}/", f"results_{testCase}/")
print("\n"*10)
