import os
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import ExponentialLR
import pathlib

from constants.constants import *
from utils.nn_models import Net_relu_xavier_decay2
from utils.read import BulkSystem, read_NNConfigFile, read_PPparams
from utils.pp_func import pot_func, realSpacePot, plotBandStruct, plotPP, plot_training_validation_cost, FT_converge_and_write_pp
from utils.bandStruct import calcHamiltonianMatrix_GPU, calcBandStruct_GPU
from utils.init_NN_train import init_Zunger_data, init_Zunger_weighted_mse, init_Zunger_train_GPU
from utils.NN_train import weighted_mse_bandStruct, weighted_mse_energiesAtKpt, bandStruct_train_GPU
from utils.ham import Hamiltonian
from utils.memory import memory_usage_data, print_memory_usage, plot_memory_usage

torch.set_default_dtype(torch.float32)
torch.manual_seed(24)
DEBUG_MEMORY_FLAG = True

device = torch.device("cpu")

pwd = pathlib.Path(__file__).parent.resolve()
NNConfig = read_NNConfigFile(f"{pwd}/inputs/NN_config.par")
SHOWPLOTS = NNConfig['SHOWPLOTS']  # True or False
nSystem = NNConfig['nSystem']
checkpoint = NNConfig['checkpoint']  # True or False
separateKptGrad = NNConfig['separateKptGrad']  # True or False

system = BulkSystem()
system.setSystem(f"{pwd}/inputs/system_0.par")
system.setInputs(f"{pwd}/inputs/input_0.par")
system.setKPointsAndWeights(f"{pwd}/inputs/kpoints_0.par")
system.setExpBS(f"{pwd}/inputs/expBandStruct_0.par")
system.setBandWeights(f"{pwd}/inputs/bandWeights_0.par")
atomPPOrder = np.unique(system.atomTypes)
nPseudopot = len(atomPPOrder)
print("There are %d atomic pseudopotentials. They are in the order of: " % nPseudopot)
print(atomPPOrder)
PPparams, totalParams = read_PPparams(atomPPOrder, f"{pwd}/inputs/init_")
localPotParams = totalParams[:,:4]

layers = [1] + NNConfig['hiddenLayers'] + [nPseudopot]
PPmodel = Net_relu_xavier_decay2(layers)

print_memory_usage()

############# Initialize the NN to the local pot function form #############
train_dataset = init_Zunger_data(atomPPOrder, localPotParams, True)
val_dataset = init_Zunger_data(atomPPOrder, localPotParams, False)

ham = Hamiltonian(system, {}, atomPPOrder, device, NNConfig, SObool=False)
print_memory_usage()

PPmodel.load_state_dict(torch.load(f"{pwd}/inputs/init_PPmodel.pth", map_location=device))
print("\nDone with NN initialization to the file inputs/init_PPmodel.pth.")
print_memory_usage()

############# Fit NN to band structures ############# 
print("\n############################################\nStart training of the NN to fit to band structures. ")

criterion_singleSystem = weighted_mse_bandStruct
criterion_singleKpt = weighted_mse_energiesAtKpt
optimizer = torch.optim.Adam(PPmodel.parameters(), lr=NNConfig['optimizer_lr'])
scheduler = ExponentialLR(optimizer, gamma=NNConfig['scheduler_gamma'])

start_time = time.time()
(training_cost, validation_cost) = bandStruct_train_GPU(PPmodel, device, NNConfig, [system], [ham], atomPPOrder, localPotParams, criterion_singleSystem, criterion_singleKpt, optimizer, scheduler, val_dataset)

end_time = time.time()
elapsed_time = end_time - start_time
print("GPU training: elapsed time: %.2f seconds" % elapsed_time)
torch.cuda.empty_cache()

plot_memory_usage()