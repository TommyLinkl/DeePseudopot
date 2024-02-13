import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import gc
from multiprocessing import shared_memory
from memory_profiler import profile
import numpy as np

from constants.constants import *
from utils.read import MEMORY_FLAG, RUNTIME_FLAG, BulkSystem, read_NNConfigFile, setAllBulkSystems, setNN
from utils.pp_func import plotPP, FT_converge_and_write_pp, plotBandStruct
from utils.init_NN_train import init_Zunger_data, init_Zunger_weighted_mse, init_Zunger_train_GPU
from utils.NN_train import weighted_mse_bandStruct, weighted_mse_energiesAtKpt, bandStruct_train_GPU
from utils.ham import Hamiltonian, initAndCacheHams

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
    print(f"\n{'#' * 40}\nReading and setting up the BulkSystems.")
    systems, atomPPOrder, nPseudopot, PPparams, totalParams, localPotParams = setAllBulkSystems(nSystem, inputsFolder, resultsFolder)

    # Set up the neural network
    PPmodel = setNN(NNConfig, nPseudopot)

    # Initialize the ham class for each BulkSystem. Cache the SO and NL mats. 
    hams, cachedMats_info, shm_dict_SO, shm_dict_NL = initAndCacheHams(systems, NNConfig, PPparams, atomPPOrder, device)

    oldFunc_plot_bandStruct_list = []
    oldFunc_totalMSE = 0
    for iSys, sys in enumerate(systems):
        start_time = time.time()
        oldFunc_bandStruct = hams[iSys].calcBandStruct_noGrad(NNConfig, iSys, cachedMats_info if cachedMats_info is not None else None)
        oldFunc_bandStruct.detach_()
        end_time = time.time()
        print(f"Old Zunger BS: Finished calculating {iSys}-th band structure in the Zunger function form ... Elapsed time: {(end_time - start_time):.2f} seconds")
        oldFunc_plot_bandStruct_list.append(sys.expBandStruct)
        oldFunc_plot_bandStruct_list.append(oldFunc_bandStruct)
        oldFunc_totalMSE += weighted_mse_bandStruct(oldFunc_bandStruct, sys)
    fig = plotBandStruct(systems, oldFunc_plot_bandStruct_list, NNConfig['SHOWPLOTS'])
    print("The total bandStruct MSE = %e " % oldFunc_totalMSE)
    fig.suptitle("The total bandStruct MSE = %e " % oldFunc_totalMSE)
    fig.savefig(resultsFolder + 'oldFunc_plotBS.png')
    plt.close('all')

    ############# Initialize the NN to the local pot function form #############
    train_dataset = init_Zunger_data(atomPPOrder, localPotParams, train=True)
    val_dataset = init_Zunger_data(atomPPOrder, localPotParams, train=False)

    if os.path.exists(inputsFolder + 'init_PPmodel.pth'):
        print(f"\n{'#' * 40}\nInitializing the NN with file {inputsFolder}init_PPmodel.pth.")
        PPmodel.load_state_dict(torch.load(inputsFolder + 'init_PPmodel.pth'))
        print(f"Done with NN initialization to the file {inputsFolder}init_PPmodel.pth.")
    else:
        print(f"\n{'#' * 40}\nInitializing the NN by fitting to the latest function form of pseudopotentials. ")
        PPmodel.cpu()
        PPmodel.eval()
        NN_init = PPmodel(val_dataset.q)
        plotPP(atomPPOrder, val_dataset.q, val_dataset.q, val_dataset.vq_atoms, NN_init, "ZungerForm", "NN_init", ["-",":" ]*nPseudopot, False, NNConfig['SHOWPLOTS'])

        init_Zunger_criterion = init_Zunger_weighted_mse
        init_Zunger_optimizer = torch.optim.Adam(PPmodel.parameters(), lr=NNConfig['init_Zunger_optimizer_lr'])
        init_Zunger_scheduler = ExponentialLR(init_Zunger_optimizer, gamma=NNConfig['init_Zunger_scheduler_gamma'])
        trainloader = DataLoader(dataset = train_dataset, batch_size = int(train_dataset.len/4),shuffle=True)
        validationloader = DataLoader(dataset = val_dataset, batch_size =val_dataset.len, shuffle=False)

        start_time = time.time()
        init_Zunger_train_GPU(PPmodel, device, trainloader, validationloader, init_Zunger_criterion, init_Zunger_optimizer, init_Zunger_scheduler, 20, NNConfig['init_Zunger_num_epochs'], NNConfig['init_Zunger_plotEvery'], atomPPOrder, NNConfig['SHOWPLOTS'], resultsFolder)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Initialization elapsed time: %.2f seconds" % elapsed_time)

        torch.save(PPmodel.state_dict(), resultsFolder + 'initZunger_PPmodel.pth')

        print("Done with NN initialization to the latest function form.")

    print("Plotting and write pseudopotentials in the real and reciprocal space.")
    torch.cuda.empty_cache()
    PPmodel.eval()
    qmax = np.array([10.0, 20.0, 30.0])
    nQGrid = np.array([2048, 4096])
    nRGrid = np.array([2048, 4096])
    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, val_dataset, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, resultsFolder + 'initZunger_plotPP', resultsFolder + 'initZunger_pot', NNConfig['SHOWPLOTS'])

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
    (training_cost, validation_cost) = bandStruct_train_GPU(PPmodel, device, NNConfig, systems, hams, atomPPOrder, localPotParams, criterion_singleSystem, criterion_singleKpt, optimizer, scheduler, val_dataset, resultsFolder, cachedMats_info if cachedMats_info is not None else None)
    end_time = time.time()
    print(f"Training elapsed time: {end_time - start_time:.2f} seconds")
    torch.cuda.empty_cache()

    ############# Writing the trained NN PP ############# 
    print(f"\n{'#' * 40}\nWriting the NN pseudopotentials")
    PPmodel.eval()
    qmax = np.array([10.0, 20.0, 30.0])
    nQGrid = np.array([2048, 4096])
    nRGrid = np.array([2048, 4096])
    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, val_dataset, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, resultsFolder + 'final_plotPP', resultsFolder + 'final_pot', NNConfig['SHOWPLOTS'])

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
