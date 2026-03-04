import os, time, sys, glob
import torch
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

from utils.read import read_NNConfigFile, setAllBulkSystems, setNN
from utils.pp_func import FT_converge_and_write_pp
from utils.init_NN_train import init_ZungerPP, init_optimizer
from utils.NN_train import weighted_mse_bandStruct, weighted_mse_energiesAtKpt, weighted_relative_mse_bandStruct, weighted_relative_mse_energiesAtKpt, bandStruct_train_GPU, evalBS_noGrad, runMC_NN, write_PP_qSpace
from utils.ham import initAndCacheHams
from utils.genMovie import genMovie
                
def main(inputsFolder = 'inputs/', resultsFolder = 'results/'):
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
    ##### add qpoint to setAllBulkSystems

    # Set up the neural network
    PPmodel = setNN(NNConfig, nPseudopot)

    # Initialize the ham class for each BulkSystem. Cache the SO and NL mats. 
    hams, cachedMats_info, shm_dict_SO, shm_dict_NL = initAndCacheHams(systems, NNConfig, PPparams, atomPPOrder, device)

    # Calculate bandStructure with the old function form with parameters given in PPparams
    # print("Evaluating band structures using the old Zunger form pseudopotentials in the init_xxx files. ")
    # oldFunc_totalMSE = evalBS_noGrad(None, f'{resultsFolder}oldFunc_plotBS.pdf', 'Old Zunger BS', NNConfig, hams, systems, cachedMats_info, writeBS=True, resultsFolder=resultsFolder)

    # Initialize the NN to the local pot function form or obtain the NN parameters from init-PPmodel.pth
    PPmodel, ZungerPPFunc_val = init_ZungerPP(inputsFolder, PPmodel, atomPPOrder, localPotParams, nPseudopot, NNConfig, device, resultsFolder)

    # Evaluate the band structures and pseudopotentials for the initialized NN
    # print("\nEvaluating band structures using the initialized pseudopotentials. ")
    # init_totalMSE = evalBS_noGrad(PPmodel, f'{resultsFolder}initZunger_plotBS.pdf', 'Init NN BS', NNConfig, hams, systems, cachedMats_info, writeBS=True, resultsFolder=resultsFolder)

    # print("Converge the pseudopotentials in the real and reciprocal space for the initialized NN. ")
    # qmax = np.array([30.0, 40.0, 50.0])
    # nQGrid = np.array([2048, 4096])
    # nRGrid = np.array([2048, 4096])
    # torch.cuda.empty_cache()
    # PPmodel.eval()
    # FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -4.0, 4.0, 40.0, 2048, 2048, f'{resultsFolder}initZunger_plotPP', f'{resultsFolder}initZunger_pot', NNConfig['SHOWPLOTS'])
    # write_PP_qSpace(f'{resultsFolder}initZunger_qSpace_pot.dat', PPmodel, atomPPOrder)


    ############# Fit NN to band structures ############# 
    # if (not NNConfig['mc_bool']): 
    #    print(f"\n{'#' * 40}\nStart training of the NN to fit to band structures. ")

    #    optimizer = init_optimizer(inputsFolder, PPmodel, NNConfig)
    #    scheduler = ExponentialLR(optimizer, gamma=NNConfig['scheduler_gamma'])

    #    start_time = time.time()
    #    (training_cost, validation_cost) = bandStruct_train_GPU(PPmodel, device, NNConfig, systems, hams, atomPPOrder, optimizer, scheduler, ZungerPPFunc_val, resultsFolder, cachedMats_info)
    #    end_time = time.time()
    #    print(f"Total training + evaluation elapsed time: {end_time - start_time:.2f} seconds")
    #    torch.cuda.empty_cache()

    ############# Run Monte Carlo on NN ############# 
    # else: 
    #    print(f"\n{'#' * 40}\nRunning Monte Carlo on the NN model. ")
    #    start_time = time.time()
    #    (trial_COST, accepted_COST, bestModel, currModel) = runMC_NN(PPmodel, NNConfig, systems, hams, atomPPOrder, ZungerPPFunc_val, resultsFolder, cachedMats_info)
    #    end_time = time.time()
    #    print(f"Monte Carlo elapsed time: {end_time - start_time:.2f} seconds")
    #    torch.cuda.empty_cache()

    #    PPmodel = bestModel
    #    PPmodel.eval()
    #    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -4.0, 4.0, 40.0, 2048, 2048, resultsFolder + 'best_plotPP', resultsFolder + 'best_pot', NNConfig['SHOWPLOTS'])

    #    PPmodel = currModel


    ############# Writing the trained NN PP ############# 
    # print(f"\n{'#' * 40}\nWriting the NN pseudopotentials")
    # PPmodel.eval()
    # FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -4.0, 4.0, 40.0, 2048, 2048, resultsFolder + 'final_plotPP', resultsFolder + 'final_pot', NNConfig['SHOWPLOTS'])

    ############# Calculate e-ph coupling #############   
    for iSys, sys in enumerate(systems):
        hams[iSys].NN_locbool = True
        hams[iSys].set_NNmodel(PPmodel)
        BS = hams[iSys].calcBandStruct_noGrad()
        # print("in main: ", hams[iSys].coupling)
        # print("in main: ", hams[iSys].cb_vecs)
        cpl_dict = hams[iSys].calcCouplings()
        output = os.path.join(resultsFolder, f"couplingBands_{iSys}.dat")
        with open(output, 'w') as fwrite:
            for atomidx in range(sys.getNAtoms()):
                print(f"Atom idx = {atomidx}   atom = {sys.atomTypes[atomidx]}   position = {sys.atomPos[atomidx]}", file=fwrite)

                for band in ["vb", "cb"]:
                    print(f"{band}-{band} coupling elements. ", file=fwrite, end="")
                    for gamma in range(3):
                        if gamma == 0:
                            print("\npolarization of derivative = x", file=fwrite)
                        elif gamma == 1:
                            print("polarization of derivative = y", file=fwrite)
                        else:
                           print("polarization of derivative = z", file=fwrite)

                        for qidx in range(sys.qpts.shape[0]):
                            if (atomidx, gamma, qidx, band) in cpl_dict:
                               print(f"{cpl_dict[(atomidx, gamma, qidx, band)]:.6e}   ", file=fwrite, end="")
                            else:
                                print("Not-fit   ", file=fwrite, end="")
                        print("\n", file=fwrite, end="")
                    print("\n", file=fwrite, end="")
                print("\n\n", file=fwrite, end="")

    ############# Free the shared data ############# 
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

    if len(sys.argv) != 3:
        print("Usage: python main.py <inputsFolder> <resultsFolder> ")
        sys.exit(1)

    inputsFolder = sys.argv[1]
    resultsFolder = sys.argv[2]
    main(inputsFolder, resultsFolder)
