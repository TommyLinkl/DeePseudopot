import numpy as np
import time
import torch
import pathlib
import matplotlib.pyplot as plt

from utils.nn_models import *
from utils.ham import Hamiltonian
from utils.read import BulkSystem, read_PPparams, read_NNConfigFile
from utils.NN_train import weighted_mse_bandStruct
from utils.pp_func import plotPP, plotBandStruct

torch.set_default_dtype(torch.float64)
torch.manual_seed(24)

def truncate_float(value, numDecimal): 
    truncated_value = float(f"{value:.{numDecimal}f}")
    
def truncate_2d_tensor(original_tensor, numDecimal):
    rows, cols = original_tensor.size()
    truncated_tensor = torch.zeros_like(original_tensor)

    for i in range(rows):
        for j in range(cols):
            element = original_tensor[i, j].item()
            truncated_value = float(f"{element:.{numDecimal}f}")
            truncated_tensor[i, j] = torch.tensor(truncated_value)
    return truncated_tensor

def truncate_PPparam_dict(original_dict, numDecimal):
    truncated_dict = {}
    for key, tensor in original_dict.items():
        truncated_tensor = torch.tensor([float(f"{value:.{numDecimal}f}") for value in tensor])
        truncated_dict[key] = truncated_tensor
    return truncated_dict

'''
original_value = 88.83030750
for decimal_places in range(1, 10): 
    truncated_value = float(f"{original_value:.{decimal_places}f}")
    truncated_tensor = torch.tensor(truncated_value)
    print(f"Truncated to {decimal_places} decimal places: {truncated_value}. Tensor item: {truncated_tensor.item()}")
'''

device = torch.device("cpu")

pwd = pathlib.Path(__file__).parent.resolve()
NNConfig = read_NNConfigFile(f"{pwd}/inputs/sigFigs/NN_config.par")
system = BulkSystem()
system.setSystem(f"{pwd}/inputs/sigFigs/system_0.par")
system.setInputs(f"{pwd}/inputs/sigFigs/input_0.par")
system.setKPointsAndWeights(f"{pwd}/inputs/sigFigs/kpoints_0.par")
system.setExpBS(f"{pwd}/inputs/sigFigs/expBandStruct_0.par")
system.setBandWeights(f"{pwd}/inputs/sigFigs/bandWeights_0.par")
atomPPorder = np.unique(system.atomTypes)

bs_old = system.expBandStruct

PPparams, totalParams = read_PPparams(atomPPorder, f"{pwd}/inputs/sigFigs/init_")

ham1 = Hamiltonian(system, PPparams, atomPPorder, device, NNConfig=NNConfig, SObool=True)
f = open(f"{pwd}/sigFigs_results/decimal_MSE.dat", "w")
f.write("# decimalPoints       MSE(customedWeights)\n")

for decimal_places in range(8, 0, -1): 
    print(f"Truncated to {decimal_places} decimal places. ")
    PPparams = truncate_PPparam_dict(PPparams, decimal_places)
    totalParams = truncate_2d_tensor(totalParams, decimal_places)
    # print(PPparams)
    # print(totalParams)
    print(PPparams['Pb'][0].item())
    
    ham1.set_PPparams(PPparams)
    print(PPparams['Pb'][0].item())

    plot_BS_list = []
    start_time = time.time()
    bs_new = ham1.calcBandStruct_noGrad(cachedMats_info=None)
    bs_new.detach_()
    end_time = time.time()
    print(f"Finished calculating the SOC band structure... Elapsed time: {(end_time - start_time):.2f} seconds")
    plot_BS_list.append(system.expBandStruct)
    plot_BS_list.append(bs_new)
    MSE = weighted_mse_bandStruct(bs_new, system)
    f.write(f"{decimal_places:d}      {MSE:.3f}\n")

    fig = plotBandStruct([system], plot_BS_list, False)
    print("The total bandStruct MSE = %e " % MSE)
    fig.suptitle("The total bandStruct MSE = %e " % MSE)
    fig.savefig(f"{pwd}/sigFigs_results/plotBS_decimal{decimal_places}.png")
    plt.clf()
    plt.close(fig)

f.close()
