import os
import torch
from test_memory.test_memory import run_memory_test
from utils.memory import set_debug_memory_flag
import pathlib

torch.set_default_dtype(torch.float32)
torch.manual_seed(24)
device = torch.device("cpu")
set_debug_memory_flag(True)

pwd = pathlib.Path(__file__).parent.resolve()
inputsFolder = f'{pwd}/checkpoint_inputs/'
resultsFolder = f'{pwd}/checkpoint_results/'
os.makedirs(resultsFolder, exist_ok=True)

run_memory_test(inputsFolder, resultsFolder, device)