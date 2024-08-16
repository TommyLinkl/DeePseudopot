import os, time, sys, glob
import torch
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import matplotlib.pyplot as plt 

from utils.read import read_NNConfigFile, setAllBulkSystems, setNN
from utils.pp_func import FT_converge_and_write_pp, plotPP
from utils.init_NN_train import init_ZungerPP, init_optimizer
from utils.NN_train import weighted_mse_bandStruct, weighted_mse_energiesAtKpt, weighted_relative_mse_bandStruct, weighted_relative_mse_energiesAtKpt, bandStruct_train_GPU, evalBS_noGrad, runMC_NN, print_and_inspect_NNParams
from utils.ham import initAndCacheHams
from utils.genMovie import genMovie

def norm_params_NN(model, mode='all'): 
    """
    Normalize each parameter (weights and biases) of a given neural network separately.
    
    Args:
        model (nn.Module): The neural network model whose parameters are to be normalized.
    """

    if mode == 'all': 
        # Gather all parameters into a single tensor
        all_params = []
        for param in model.parameters():
            all_params.append(param.data.view(-1)) 
        
        all_params = torch.cat(all_params)  # Concatenate into a single tensor

        mean = all_params.mean()
        std = all_params.std()

        # Normalize each parameter
        with torch.no_grad():
            for param in model.parameters():
                param.data = (param.data - mean) / std 

    elif mode == 'separately': 
        with torch.no_grad():
            for param in model.parameters():
                mean = param.data.mean()
                std = param.data.std()
                param.data = (param.data - mean) / (std + 1e-8)
    return


def norm_retrain_func(inputsFolder = 'inputs/', resultsFolder = 'results/'):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
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
    PPmodel, ZungerPPFunc_val = init_ZungerPP(inputsFolder, PPmodel, atomPPOrder, localPotParams, nPseudopot, NNConfig, device, resultsFolder)
    print_and_inspect_NNParams(PPmodel, f'{resultsFolder}loaded_0_params.dat', show=True)
    fig = plotPP(atomPPOrder, ZungerPPFunc_val.q, ZungerPPFunc_val.q, ZungerPPFunc_val.vq_atoms, PPmodel(ZungerPPFunc_val.q), "ZungerForm", f"loaded_0", ["-",":" ]*len(atomPPOrder), True, NNConfig['SHOWPLOTS']);
    fig.savefig(f'{resultsFolder}loaded_0_plotPP.png')

    for iRepeat in range(3):          # repeat 3 times? 
        # Normalize the parameters
        old_forScale = PPmodel(torch.tensor([0.0])).detach()
        norm_params_NN(PPmodel)
        print_and_inspect_NNParams(PPmodel, f'{resultsFolder}norm_{iRepeat}_params.dat', show=True)
        fig = plotPP(atomPPOrder, ZungerPPFunc_val.q, ZungerPPFunc_val.q, ZungerPPFunc_val.vq_atoms, PPmodel(ZungerPPFunc_val.q), "ZungerForm", f"norm_{iRepeat}", ["-",":" ]*len(atomPPOrder), True, NNConfig['SHOWPLOTS']);
        fig.savefig(f'{resultsFolder}norm_{iRepeat}_plotPP.png')
        new_forScale = PPmodel(torch.tensor([0.0])).detach()
        print(f"Before and after normalization: {old_forScale}, {new_forScale}. ")

        # Calculate the appropriate new scaling factors
        new_scale = old_forScale / (new_forScale + torch.full_like(new_forScale, 1e-8))
        print(new_scale)

        # Change the scaling factors in PPmodel and in NNConfig
        PPmodel.change_scale(new_scale)
        NNConfig['PPmodel_scale'] = new_scale.tolist()

        print_and_inspect_NNParams(PPmodel, f'{resultsFolder}rescale_{iRepeat}_params.dat', show=True)
        fig = plotPP(atomPPOrder, ZungerPPFunc_val.q, ZungerPPFunc_val.q, ZungerPPFunc_val.vq_atoms, PPmodel(ZungerPPFunc_val.q), "ZungerForm", f"rescale_{iRepeat}", ["-",":" ]*len(atomPPOrder), True, NNConfig['SHOWPLOTS']);
        fig.savefig(f'{resultsFolder}rescale_{iRepeat}_plotPP.png')



        # Now I actually want to retrain on the function
        PPmodel, ZungerPPFunc_val = init_ZungerPP(inputsFolder, PPmodel, atomPPOrder, localPotParams, nPseudopot, NNConfig, device, resultsFolder, force_retrain=True)
        print_and_inspect_NNParams(PPmodel, f'{resultsFolder}retrained_{iRepeat}_params.dat', show=True)
        fig = plotPP(atomPPOrder, ZungerPPFunc_val.q, ZungerPPFunc_val.q, ZungerPPFunc_val.vq_atoms, PPmodel(ZungerPPFunc_val.q), "ZungerForm", f"retrained_{iRepeat}", ["-",":" ]*len(atomPPOrder), True, NNConfig['SHOWPLOTS']);
        fig.savefig(f'{resultsFolder}retrained_{iRepeat}_plotPP.png')
        plt.close('all')

    return
    # Initialize the ham class for each BulkSystem. Cache the SO and NL mats. 
    hams, cachedMats_info, shm_dict_SO, shm_dict_NL = initAndCacheHams(systems, NNConfig, PPparams, atomPPOrder, device)

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

    return 
    ############# Fit NN to band structures ############# 
    if (not NNConfig['mc_bool']): 
        print(f"\n{'#' * 40}\nStart training of the NN to fit to band structures. ")
        if 'relE_bIdx' in NNConfig: 
            criterion_singleSystem = weighted_relative_mse_bandStruct
            criterion_singleKpt = weighted_relative_mse_energiesAtKpt
        else:
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
        (trial_COST, accepted_COST, bestModel, currModel) = runMC_NN(PPmodel, NNConfig, systems, hams, atomPPOrder, ZungerPPFunc_val, resultsFolder, cachedMats_info)
        end_time = time.time()
        print(f"Monte Carlo elapsed time: {end_time - start_time:.2f} seconds")
        torch.cuda.empty_cache()

        PPmodel = bestModel
        PPmodel.eval()
        FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, resultsFolder + 'best_plotPP', resultsFolder + 'best_pot', NNConfig['SHOWPLOTS'])

        PPmodel = currModel


    ############# Writing the trained NN PP ############# 
    print(f"\n{'#' * 40}\nWriting the NN pseudopotentials")
    PPmodel.eval()
    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, resultsFolder + 'final_plotPP', resultsFolder + 'final_pot', NNConfig['SHOWPLOTS'])

    ############# Creating animation ############# 
    start_time = time.time()
    genMovie(resultsFolder, f'{resultsFolder}movie_BS.mp4', NNConfig['max_num_epochs'])
    genMovie(resultsFolder, f'{resultsFolder}movie_PP.mp4', NNConfig['max_num_epochs'], type='PP')
    end_time = time.time()
    print(f"Creating animation, elapsed time: {end_time - start_time:.2f} seconds")
    [os.remove(file) for file in glob.glob(f'{resultsFolder}mc_iter_*_plotBS.png') if os.path.exists(file)]
    [os.remove(file) for file in glob.glob(f'{resultsFolder}mc_iter_*_plotPP.png') if os.path.exists(file)]

    ############# Free the shared data ############# 
    if shm_dict_SO is not None: 
        for shm in shm_dict_SO.values():
            shm.close()
            shm.unlink()
    if shm_dict_NL is not None:
        for shm in shm_dict_NL.values():
            shm.close()
            shm.unlink()





if len(sys.argv) != 3:
    print("Usage: python norm_retrain_func.py <inputsFolder> <resultsFolder> ")
    sys.exit(1)

inputsFolder = sys.argv[1]
resultsFolder = sys.argv[2]
norm_retrain_func(inputsFolder, resultsFolder)
