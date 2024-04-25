import os, time, sys
import torch
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

from utils.read import read_NNConfigFile, setAllBulkSystems, setNN
from utils.pp_func import FT_converge_and_write_pp
from utils.init_NN_train import init_ZungerPP, init_optimizer
from utils.NN_train import weighted_mse_bandStruct, weighted_mse_energiesAtKpt, bandStruct_train_GPU, evalBS_noGrad, runMC_NN
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
    oldFunc_totalMSE = evalBS_noGrad(None, f'{resultsFolder}oldFunc_plotBS.png', 'Old Zunger BS', NNConfig, hams, systems, cachedMats_info, writeBS=True)

    # Initialize the NN to the local pot function form
    PPmodel, ZungerPPFunc_val = init_ZungerPP(inputsFolder, PPmodel, atomPPOrder, localPotParams, nPseudopot, NNConfig, device, resultsFolder)

    # Evaluate the band structures and pseudopotentials for the initialized NN
    print("\nEvaluating band structures using the initialized pseudopotentials. ")
    init_totalMSE = evalBS_noGrad(PPmodel, f'{resultsFolder}initZunger_plotBS.png', 'Init NN BS', NNConfig, hams, systems, cachedMats_info, writeBS=True)

    print("Converge the pseudopotentials in the real and reciprocal space for the initialized NN. ")
    qmax = np.array([10.0, 20.0, 30.0])
    nQGrid = np.array([2048, 4096])
    nRGrid = np.array([2048, 4096])
    torch.cuda.empty_cache()
    PPmodel.eval()
    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, f'{resultsFolder}initZunger_plotPP', f'{resultsFolder}initZunger_pot', NNConfig['SHOWPLOTS'])


    ############# Fit NN to band structures ############# 
    if (not NNConfig['mc_bool']): 
        print(f"\n{'#' * 40}\nStart training of the NN to fit to band structures. ")
        criterion_singleSystem = weighted_mse_bandStruct
        criterion_singleKpt = weighted_mse_energiesAtKpt
        optimizer = init_optimizer(inputsFolder, PPmodel, NNConfig)
        scheduler = ExponentialLR(optimizer, gamma=NNConfig['scheduler_gamma'])

        start_time = time.time()
        (training_cost, validation_cost) = bandStruct_train_GPU(PPmodel, device, NNConfig, systems, hams, atomPPOrder, criterion_singleSystem, criterion_singleKpt, optimizer, scheduler, ZungerPPFunc_val, resultsFolder, cachedMats_info)
        end_time = time.time()
        print(f"Total training + evaluation elapsed time: {end_time - start_time:.2f} seconds")
        torch.cuda.empty_cache()

    ############# Run Monte Carlo on NN ############# 
    else: 
        print(f"\n{'#' * 40}\nRunning Monte Carlo on the NN model. ")
        start_time = time.time()
        (trial_COST, accepted_COST) = runMC_NN(PPmodel, NNConfig, systems, hams, atomPPOrder, ZungerPPFunc_val, resultsFolder, cachedMats_info)
        end_time = time.time()
        print(f"Monte Carlo elapsed time: {end_time - start_time:.2f} seconds")
        torch.cuda.empty_cache()

    ############# Writing the trained NN PP ############# 
    print(f"\n{'#' * 40}\nWriting the NN pseudopotentials")
    PPmodel.eval()
    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, resultsFolder + 'final_plotPP', resultsFolder + 'final_pot', NNConfig['SHOWPLOTS'])

    # Free the shared data
    if shm_dict_SO is not None: 
        for shm in shm_dict_SO.values():
            shm.close()
            shm.unlink()
    if shm_dict_NL is not None:
        for shm in shm_dict_NL.values():
            shm.close()
            shm.unlink()
    for ham in hams:
        if ham.shm_eVec is not None:
            for shm in ham.shm_eVec.values():
                shm.close()
                shm.unlink()
    

if __name__ == "__main__":
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if len(sys.argv) != 3:
        print("Usage: python main.py <inputsFolder> <resultsFolder> ")
        sys.exit(1)

    inputsFolder = sys.argv[1]
    resultsFolder = sys.argv[2]
    main(inputsFolder, resultsFolder)
