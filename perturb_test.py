import os, time, sys
import torch
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from multiprocessing import shared_memory

from utils.read import read_NNConfigFile, setAllBulkSystems, setNN
from utils.pp_func import FT_converge_and_write_pp
from utils.init_NN_train import init_ZungerPP, init_optimizer
from utils.NN_train import evalBS_noGrad, perturb_model, bandStruct_train_GPU, weighted_mse_bandStruct, weighted_mse_energiesAtKpt
from utils.ham import Hamiltonian, initAndCacheHams

def perturb(inputsFolder = 'inputs_evalFullBand/', resultsFolder = 'results_evalFullBand/'):
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

    if not os.path.exists(inputsFolder + 'init_PPmodel.pth'):
        raise FileNotFoundError("""WARNING: Can't find init_PPmodel.pth file. 
              This routine perturbs an existing neural network model. Please provide init_PPmodel.pth in the input folder.""")

    # Initialize the ham class for each BulkSystem. Cache the SO and NL mats. 
    hams, cachedMats_info, shm_dict_SO, shm_dict_NL = initAndCacheHams(systems, NNConfig, PPparams, atomPPOrder, device)

    # Initialize the NN according to the provided file init_PPmodel.pth
    PPmodel, ZungerPPFunc_val = init_ZungerPP(inputsFolder, PPmodel, atomPPOrder, localPotParams, nPseudopot, NNConfig, device, resultsFolder)

    perturb_model(PPmodel, 0.02)

    # Run one epoch of fitting after perturbation
    NNConfig['max_num_epochs'] = 1
    criterion_singleSystem = weighted_mse_bandStruct
    criterion_singleKpt = weighted_mse_energiesAtKpt
    optimizer = init_optimizer(inputsFolder, PPmodel, NNConfig)
    scheduler = ExponentialLR(optimizer, gamma=NNConfig['scheduler_gamma'])
    (training_cost, validation_cost) = bandStruct_train_GPU(PPmodel, device, NNConfig, systems, hams, atomPPOrder, criterion_singleSystem, criterion_singleKpt, optimizer, scheduler, ZungerPPFunc_val, resultsFolder, cachedMats_info)
    
    print("Converge the pseudopotentials in the real and reciprocal space. ")
    qmax = np.array([10.0, 20.0, 30.0])
    nQGrid = np.array([2048, 4096])
    nRGrid = np.array([2048, 4096])
    torch.cuda.empty_cache()
    PPmodel.eval()
    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, f'{resultsFolder}initZunger_plotPP', f'{resultsFolder}initZunger_pot', NNConfig['SHOWPLOTS'])

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
        if ham.eVec_info is not None:
            for key in ham.eVec_info:
                shm_obj = shared_memory.SharedMemory(name=key)
                shm_obj.close()
                shm_obj.unlink()


if len(sys.argv) != 3:
    print("Usage: python perturb_test.py <inputsFolder> <resultsFolder> ")
    sys.exit(1)

inputsFolder = sys.argv[1]
resultsFolder = sys.argv[2]
perturb(inputsFolder, resultsFolder)
