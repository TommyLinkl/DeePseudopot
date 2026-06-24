import os, time, sys, glob, re

from utils.config_threads import configure_threads

# Must be called *before* importing numpy, scipy, torch, etc.
configure_threads(n_threads=1)

import torch
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

from utils.read import read_NNConfigFile, setAllBulkSystems, setNN, setNN_LSD
from utils.nn_models import zero_init_final_layer
from utils.pp_func import FT_converge_and_write_pp, plotPP_spin
from utils.init_NN_train import init_ZungerPP, init_ZungerPP_spin, init_optimizer
from utils.init_LSD_train import init_LSD_PP
from utils.NN_train import weighted_mse_bandStruct, weighted_mse_energiesAtKpt, weighted_relative_mse_bandStruct, weighted_relative_mse_energiesAtKpt, bandStruct_train_GPU, evalBS_noGrad, runMC_NN, write_PP_qSpace, write_PP_qSpace_spin
from utils.ham import initAndCacheHams, set_LSDModels
from utils.genMovie import genMovie
from utils.profiling import PROF, benchmark_eigensolve

def main(inputsFolder = 'inputs/', resultsFolder = 'results/'):
    torch.set_default_dtype(torch.float64)

    device = torch.device("cpu")
    
    os.makedirs(resultsFolder, exist_ok=True)

    NNConfig = read_NNConfigFile(inputsFolder + 'NN_config.par')
    # Enable the aggregating timing/memory profiler (utils/profiling.PROF) based
    # on the runtime_flag / memory_flag switches. When on, the training loop
    # prints a per-section breakdown (build vs diagonalize) plus RSS checkpoints.
    PROF.configure(NNConfig.get('runtime_flag', False), NNConfig.get('memory_flag', False))
    # os.environ["OMP_NUM_THREADS"] = f"{NNConfig["num_threads"]}"
    # os.environ["MKL_NUM_THREADS"] = f"{NNConfig["num_threads"]}"
    # torch.set_num_threads(NNConfig["num_threads"])
    print(f"Number of threads for multithreaded lin. alg = {NNConfig['num_threads']}", flush=True)
        
    nSystem = NNConfig['nSystem']
    
    # Read and set up systems
    print(f"\nReading and setting up the BulkSystems.")
    systems, atomPPOrder, nPseudopot, PPparams, totalParams, localPotParams = setAllBulkSystems(nSystem, inputsFolder, resultsFolder, NNConfig['local_env_corr'], descriptor_backend=NNConfig.get('descriptor_backend', 'handcrafted'))

    # Set up the neural network
    PPmodel = setNN(NNConfig, nPseudopot)

    # If spin-polarized (tot_magnetization != 0), set up the learned spin field
    # b(q): same architecture/output as PPmodel, with the final layer zero-init'd
    # so b == 0 at the start (training begins from the unpolarized solution). The
    # up/down local potentials are then V_up = V0 + b, V_down = V0 - b.
    if NNConfig.get('tot_magnetization', 0.0) != 0:
        print(f"\nSpin-polarized local potential ON (tot_magnetization = {NNConfig['tot_magnetization']}).")
        spinModel = setNN(NNConfig, nPseudopot)
        zero_init_final_layer(spinModel)
    else:
        spinModel = None

    # If local structure-dependent (LSD) corrections are required, then set up neural networks for LSD
    if NNConfig['local_env_corr']:
        # Set up the neural network to predict the LSD potential
        os.makedirs(f"{resultsFolder}LSD/", exist_ok=True)
        LSDmodels = {}

        print(f"atomPPOrder = {atomPPOrder}")
        for atom in atomPPOrder:
            n_descr = systems[0].env_descriptors[atom].shape[1]
            lsd_layers = [n_descr + 1] + NNConfig['LSD_hiddenLayers'] + [1]
            LSDmodels[atom] = setNN_LSD(NNConfig, layers=lsd_layers)
            print(f"\nLSDmodel[{atom}] = {LSDmodels[atom]}")

        LSDoptimizers = {atom: None for atom in atomPPOrder}
        LSDschedulers = {atom: None for atom in atomPPOrder}
    else:
        LSDmodels = None
        LSD_PPFunc_val = None
        LSDoptimizers = {atom: None for atom in atomPPOrder}
        LSDschedulers = {atom: None for atom in atomPPOrder}

    # Initialize the ham class for each BulkSystem. Cache the SO and NL mats.
    hams, cachedMats_info, shm_dict_SO, shm_dict_NL = initAndCacheHams(systems, NNConfig, PPparams, atomPPOrder, device, spinModel=spinModel)
    PROF.mem_checkpoint("after initAndCacheHams")

    # One-shot diagnostic: how does diagonalizing a representative Hamiltonian
    # scale with BLAS/torch threads? This directly answers "should I parallelize
    # the eigensolve (num_threads>1) vs keep BLAS single-threaded and parallelize
    # over k-points?" at the actual matrix size of this run.
    if NNConfig.get('runtime_flag', False) and len(hams) > 0:
        _nbv = hams[0].basis.shape[0]
        _ndim = 2 * _nbv if hams[0].spinor else _nbv
        benchmark_eigensolve(_ndim)

    if NNConfig['local_env_corr']:
        # Initialize the LSD correction to the potential differences
        LSDmodels, LSD_PPFunc_val = init_LSD_PP(inputsFolder, LSDmodels, systems, atomPPOrder, NNConfig, resultsFolder, force_retrain=NNConfig["init_LSD_force_retrain"])
        
        for iSys, system in enumerate(systems):
          hams[iSys].set_LSDmodels(LSDmodels)

    # Calculate bandStructure with the old function form with parameters given in PPparams
    # print("Evaluating band structures using the old Zunger form pseudopotentials in the init_xxx files. ")
    # oldFunc_totalMSE = evalBS_noGrad(None, f'{resultsFolder}oldFunc_plotBS.pdf', 'Old Zunger BS', NNConfig, hams, systems, cachedMats_info, writeBS=True, resultsFolder=resultsFolder)

    # Initialize the NN to the local pot function form. In a spin-polarized run
    # this also reads init_spinModel.pth (when present) the same way
    # init_PPmodel.pth is read for the local potential.
    if spinModel is not None:
        PPmodel, spinModel, ZungerPPFunc_val = init_ZungerPP_spin(inputsFolder, PPmodel, spinModel, atomPPOrder, localPotParams, nPseudopot, NNConfig, device, resultsFolder, force_retrain=NNConfig["force_retrain"])
    else:
        PPmodel, ZungerPPFunc_val = init_ZungerPP(inputsFolder, PPmodel, atomPPOrder, localPotParams, nPseudopot, NNConfig, device, resultsFolder, force_retrain=NNConfig["force_retrain"])
    
    # Evaluate the band structures and pseudopotentials for the initialized NN
    print("\nEvaluating band structures using the initialized pseudopotentials. ")
    PROF.reset()
    init_totalMSE = evalBS_noGrad(PPmodel, f'{resultsFolder}initZunger_plotBS.pdf', 'Init NN BS', NNConfig, hams, systems, cachedMats_info, writeBS=True, resultsFolder=resultsFolder, spinModel=spinModel)
    # Profile of the no-grad evaluation pass (serial path captures build+diag
    # fully; with num_cores>0 the eval runs in mp workers whose timing is not
    # folded back here -- use the training-loop report or num_cores=0 to profile).
    PROF.report("EVAL PROFILE (initial no-grad band structure)", reset=True)

    print("Converge the pseudopotentials in the real and reciprocal space for the initialized NN. ")
    Rmax = 30.0
    qmax = np.array([40.0])
    nQGrid = np.array([4096, 8192])
    nRGrid = np.array([4096, 8192])
    torch.cuda.empty_cache()
    PPmodel.eval()
    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -4.0, 4.0, 40.0, 2048, 2048, f'{resultsFolder}initZunger_plotPP', f'{resultsFolder}initZunger_pot', NNConfig['SHOWPLOTS'], PPparams, Rmax)
    write_PP_qSpace(f'{resultsFolder}initZunger_qSpace_pot.dat', PPmodel, atomPPOrder)
    torch.save(PPmodel.state_dict(), f"{resultsFolder}initZunger_PPmodel.pth")
    if spinModel is not None:
        write_PP_qSpace_spin(f'{resultsFolder}initZunger_qSpace_pot_spin.dat', PPmodel, spinModel, atomPPOrder)
        fig_spin = plotPP_spin(atomPPOrder, ZungerPPFunc_val.q, PPmodel(ZungerPPFunc_val.q), spinModel(ZungerPPFunc_val.q), "initZunger", NNConfig['SHOWPLOTS'])
        fig_spin.savefig(f'{resultsFolder}initZunger_plotPP_spin.pdf')
        fig_spin.savefig(f'{resultsFolder}initZunger_plotPP_spin.png')
        torch.save(spinModel.state_dict(), f"{resultsFolder}initZunger_spinModel.pth")

    ############# Fit NN to band structures ############# 
    if (not NNConfig['mc_bool']): 
        print(f"\n{'#' * 40}\nStart training of the NN to fit to band structures. ")

        optimizer = init_optimizer(inputsFolder, PPmodel, NNConfig)
        scheduler = ExponentialLR(optimizer, gamma=NNConfig['scheduler_gamma'])

        if NNConfig['local_env_corr'] == 1:
            for key in LSDmodels:
                LSDoptimizers[key] = init_optimizer(inputsFolder, LSDmodels[key], NNConfig, LSD_flag=True)
                LSDschedulers[key] = ExponentialLR(LSDoptimizers[key], gamma=NNConfig['LSD_scheduler_gamma'])

        # Spin-field optimizer. Use a fresh Adam state (do NOT load V0's
        # init_AdamState.pth) since b starts at zero with its own dynamics.
        if spinModel is not None:
            spinOptimizer = init_optimizer(inputsFolder, spinModel, NNConfig, load_adam_state=False)
            spinScheduler = ExponentialLR(spinOptimizer, gamma=NNConfig['scheduler_gamma'])
        else:
            spinOptimizer = None
            spinScheduler = None

        start_time = time.time()
        (training_cost, validation_cost) = bandStruct_train_GPU(PPmodel, device, NNConfig, systems, hams, atomPPOrder, optimizer, scheduler, ZungerPPFunc_val, resultsFolder, cachedMats_info, LSDmodels=LSDmodels, LSDoptimizers=LSDoptimizers, LSDscheduler=LSDschedulers, LSDval_dataset=LSD_PPFunc_val, spinModel=spinModel, spinOptimizer=spinOptimizer, spinScheduler=spinScheduler)
        end_time = time.time()
        print(f"Total training + evaluation elapsed time: {end_time - start_time:.2f} seconds")
        torch.cuda.empty_cache()

    ############# Run Monte Carlo on NN ############# 
    else: 
        print(f"\n{'#' * 40}\nRunning Monte Carlo on the NN model. ")
        start_time = time.time()
        (trial_COST, accepted_COST, bestModel, currModel, bestSpinModel, currSpinModel) = runMC_NN(PPmodel, NNConfig, systems, hams, atomPPOrder, ZungerPPFunc_val, resultsFolder, cachedMats_info, spinModel=spinModel)
        end_time = time.time()
        print(f"Monte Carlo elapsed time: {end_time - start_time:.2f} seconds")
        torch.cuda.empty_cache()

        PPmodel = bestModel
        PPmodel.eval()
        FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -4.0, 4.0, 40.0, 2048, 2048, resultsFolder + 'best_plotPP', resultsFolder + 'best_pot', NNConfig['SHOWPLOTS'], PPparams, Rmax)

        # spin-polarized run: write the best spin field b(q) and up/down potentials.
        if bestSpinModel is not None:
            bestSpinModel.eval()
            write_PP_qSpace_spin(f'{resultsFolder}best_qSpace_pot_spin.dat', bestModel, bestSpinModel, atomPPOrder, qmax=NNConfig.get('qmax', 40.0), nQGrid=NNConfig.get('nQGrid', 4096))
            fig_spin = plotPP_spin(atomPPOrder, ZungerPPFunc_val.q, bestModel(ZungerPPFunc_val.q), bestSpinModel(ZungerPPFunc_val.q), "best", NNConfig['SHOWPLOTS'])
            fig_spin.savefig(f'{resultsFolder}best_plotPP_spin.pdf')
            fig_spin.savefig(f'{resultsFolder}best_plotPP_spin.png')
            torch.save(bestSpinModel.state_dict(), f'{resultsFolder}best_spinModel.pth')

        PPmodel = currModel
        spinModel = currSpinModel


    ############# Writing the trained NN PP #############
    print(f"\n{'#' * 40}\nWriting the NN pseudopotentials")
    PPmodel.eval()
    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -4.0, 4.0, 40.0, 4096, 4096, resultsFolder + 'final_plotPP', resultsFolder + 'final_pot', NNConfig['SHOWPLOTS'], PPparams, Rmax)

    # spin-polarized run: write the final spin field b(q) and up/down potentials.
    if spinModel is not None:
        spinModel.eval()
        write_PP_qSpace_spin(f'{resultsFolder}final_qSpace_pot_spin.dat', PPmodel, spinModel, atomPPOrder, qmax=NNConfig.get('qmax', 40.0), nQGrid=NNConfig.get('nQGrid', 4096))
        fig_spin = plotPP_spin(atomPPOrder, ZungerPPFunc_val.q, PPmodel(ZungerPPFunc_val.q), spinModel(ZungerPPFunc_val.q), "final", NNConfig['SHOWPLOTS'])
        fig_spin.savefig(f'{resultsFolder}final_plotPP_spin.pdf')
        fig_spin.savefig(f'{resultsFolder}final_plotPP_spin.png')
        torch.save(spinModel.state_dict(), f'{resultsFolder}final_spinModel.pth')

    ############# Creating animation ############# 
    start_time = time.time()
    genMovie(resultsFolder, f'{resultsFolder}movie_BS.mp4', NNConfig['max_num_epochs'])
    genMovie(resultsFolder, f'{resultsFolder}movie_PP.mp4', NNConfig['max_num_epochs'], type='PP')
    end_time = time.time()
    print(f"Creating animation, elapsed time: {end_time - start_time:.2f} seconds")
    [os.remove(file) for file in glob.glob(f'{resultsFolder}mc_iter_*_plotBS.pdf') if os.path.exists(file)]
    [os.remove(file) for file in glob.glob(f'{resultsFolder}mc_iter_*_plotBS.png') if os.path.exists(file)]
    [os.remove(file) for file in glob.glob(f'{resultsFolder}mc_iter_*_plotPP.pdf') if os.path.exists(file)]
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

if __name__ == "__main__":
    # torch.set_num_threads(1)
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"

    if len(sys.argv) != 3:
        print("Usage: python main.py <inputsFolder> <resultsFolder> ")
        sys.exit(1)

    inputsFolder = sys.argv[1]
    resultsFolder = sys.argv[2]
    main(inputsFolder, resultsFolder)
