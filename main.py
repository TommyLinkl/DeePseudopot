import os, time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
from multiprocessing import shared_memory
from memory_profiler import profile
import numpy as np

from constants.constants import *
from utils.read import MEMORY_FLAG, RUNTIME_FLAG, read_NNConfigFile, setAllBulkSystems, setNN
from utils.pp_func import FT_converge_and_write_pp, plotBandStruct
from utils.init_NN_train import calcOldFuncBS, init_ZungerPP
from utils.NN_train import weighted_mse_bandStruct, weighted_mse_energiesAtKpt, bandStruct_train_GPU
from utils.ham import initAndCacheHams

def main(inputsFolder = 'inputs/', resultsFolder = 'results/'):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(24)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    device = torch.device("cpu")

    os.makedirs(resultsFolder, exist_ok=True)

    NNConfig = read_NNConfigFile(inputsFolder + 'NN_config.par')
    nSystem = NNConfig['nSystem']
    
    # Read and set up systems
    print(f"\nReading and setting up the BulkSystems.")
    systems, atomPPOrder, nPseudopot, PPparams, totalParams, localPotParams = setAllBulkSystems(nSystem, inputsFolder, resultsFolder)

    # Set up the neural network
    PPmodel = setNN(NNConfig, nPseudopot)

    # Initialize the ham class for each BulkSystem. Cache the SO and NL mats. 
    hams, cachedMats_info, shm_dict_SO, shm_dict_NL = initAndCacheHams(systems, NNConfig, PPparams, atomPPOrder, device)

    # Calculate bandStructure with the old function form with parameters given in PPparams
    oldFunc_totalMSE = calcOldFuncBS(systems, hams, NNConfig, cachedMats_info, resultsFolder)

    #Initialize the NN to the local pot function form
    PPmodel, ZungerPPFunc_val = init_ZungerPP(inputsFolder, PPmodel, atomPPOrder, localPotParams, nPseudopot, NNConfig, device, resultsFolder)





    # Evaluate the band structures and pseudopotentials for the initialized NN
    print("Plotting and write pseudopotentials in the real and reciprocal space.")
    qmax = np.array([10.0, 20.0, 30.0])
    nQGrid = np.array([2048, 4096])
    nRGrid = np.array([2048, 4096])
    torch.cuda.empty_cache()
    PPmodel.eval()
    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, f'{resultsFolder}initZunger_plotPP', f'{resultsFolder}initZunger_pot', NNConfig['SHOWPLOTS'])

    print("\nEvaluating band structures using the initialized pseudopotentials. ")
    plot_bandStruct_list = []
    init_totalMSE = 0
    for iSystem in range(nSystem): 
        hams[iSystem].NN_locbool = True
        hams[iSystem].set_NNmodel(PPmodel)
        start_time = time.time()
        init_bandStruct = hams[iSystem].calcBandStruct_noGrad(NNConfig, iSystem, cachedMats_info if cachedMats_info is not None else None)
        init_bandStruct.detach_()
        end_time = time.time()
        print(f"Finished calculating {iSystem}-th band structure in the initialized NN form... Elapsed time: {(end_time - start_time):.2f} seconds")
        plot_bandStruct_list.append(systems[iSystem].expBandStruct)
        plot_bandStruct_list.append(init_bandStruct)
        init_totalMSE += weighted_mse_bandStruct(init_bandStruct, systems[iSystem])
    fig = plotBandStruct(systems, plot_bandStruct_list, NNConfig['SHOWPLOTS'])
    print("The total bandStruct MSE = %e " % init_totalMSE)
    fig.suptitle("The total bandStruct MSE = %e " % init_totalMSE)
    fig.savefig(resultsFolder + 'initZunger_plotBS.png')
    plt.close('all')
    torch.cuda.empty_cache()

    ############# Fit NN to band structures ############# 
    print(f"\n{'#' * 40}\nStart training of the NN to fit to band structures. ")

    criterion_singleSystem = weighted_mse_bandStruct
    criterion_singleKpt = weighted_mse_energiesAtKpt
    optimizer = torch.optim.Adam(PPmodel.parameters(), lr=NNConfig['optimizer_lr'])
    scheduler = ExponentialLR(optimizer, gamma=NNConfig['scheduler_gamma'])

    start_time = time.time()
    (training_cost, validation_cost) = bandStruct_train_GPU(PPmodel, device, NNConfig, systems, hams, atomPPOrder, localPotParams, criterion_singleSystem, criterion_singleKpt, optimizer, scheduler, ZungerPPFunc_val, resultsFolder, cachedMats_info if cachedMats_info is not None else None)
    end_time = time.time()
    print(f"Training elapsed time: {end_time - start_time:.2f} seconds")
    torch.cuda.empty_cache()

    ############# Writing the trained NN PP ############# 
    print(f"\n{'#' * 40}\nWriting the NN pseudopotentials")
    PPmodel.eval()
    qmax = np.array([10.0, 20.0, 30.0])
    nQGrid = np.array([2048, 4096])
    nRGrid = np.array([2048, 4096])
    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, resultsFolder + 'final_plotPP', resultsFolder + 'final_pot', NNConfig['SHOWPLOTS'])

    if shm_dict_SO is not None: 
        for shm in shm_dict_SO.values():
            shm.close()
            shm.unlink()
    if shm_dict_NL is not None:
        for shm in shm_dict_NL.values():
            shm.close()
            shm.unlink()


if __name__ == "__main__":
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if MEMORY_FLAG:
        main = profile(main)

    main("CALCS/CsPbI3_test/inputs/", "CALCS/CsPbI3_test/results/")
