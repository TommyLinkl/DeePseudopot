import os
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import gc
from multiprocessing import shared_memory
from memory_profiler import profile

from constants.constants import *
from utils.nn_models import *
from utils.read import BulkSystem, read_NNConfigFile, read_PPparams
from utils.pp_func import plotPP, FT_converge_and_write_pp, plotBandStruct
from utils.init_NN_train import init_Zunger_data, init_Zunger_weighted_mse, init_Zunger_train_GPU
from utils.NN_train import weighted_mse_bandStruct, weighted_mse_energiesAtKpt, bandStruct_train_GPU
from utils.ham import Hamiltonian
from utils.memory import print_memory_usage, plot_memory_usage, set_debug_memory_flag

def main(inputsFolder = 'inputs/', resultsFolder = 'results/'):

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(24)

    '''
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available.\n")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.\n")
    '''
    device = torch.device("cpu")
    memory_usage_data = []
    set_debug_memory_flag(False)  # False

    ############## main ##############
    os.makedirs(resultsFolder, exist_ok=True)

    NNConfig = read_NNConfigFile(inputsFolder + 'NN_config.par')
    nSystem = NNConfig['nSystem']
    if 'memory_flag' in NNConfig:
        set_debug_memory_flag(NNConfig['memory_flag'])
    if 'num_cores' in NNConfig: 
        os.environ["OMP_NUM_THREADS"] = str(NNConfig['num_cores'])     # Accelerate numpy-related operations (eigh) through OpenMP
        os.environ["MKL_NUM_THREADS"] = "1"
        print(f"Setting OMP_NUM_THREADS = {NNConfig['num_cores']}, MKL_NUM_THREADS = 1")
    else: 
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        print(f"Setting OMP_NUM_THREADS = 1, MKL_NUM_THREADS = 1")



    # Read and set up systems
    print(f"{'#' * 40}\nReading and setting up the BulkSystems.")
    atomPPOrder = []
    systems = [BulkSystem() for _ in range(nSystem)]
    for iSys in range(nSystem): 
        systems[iSys].setSystem(inputsFolder + "system_%d.par" % iSys)
        systems[iSys].setInputs(inputsFolder + "input_%d.par" % iSys)
        systems[iSys].setKPointsAndWeights(inputsFolder + "kpoints_%d.par" % iSys)
        systems[iSys].setExpBS(inputsFolder + "expBandStruct_%d.par" % iSys)
        systems[iSys].setBandWeights(inputsFolder + "bandWeights_%d.par" % iSys)
        systems[iSys].print_basisStates(resultsFolder + "basisStates_%d.dat" % iSys)
        atomPPOrder.append(systems[iSys].atomTypes)

    # Calculate atomPPOrder. Read in initial PPparams. Set up NN accordingly
    atomPPOrder = np.unique(np.concatenate(atomPPOrder))
    nPseudopot = len(atomPPOrder)
    print(f"There are {nPseudopot} atomic pseudopotentials. They are in the order of: {atomPPOrder}")
    PPparams, totalParams = read_PPparams(atomPPOrder, inputsFolder + "init_")
    localPotParams = totalParams[:,:4]
    layers = [1] + NNConfig['hiddenLayers'] + [nPseudopot]
    # PPmodel = Net_relu_xavier_decay2(layers)
    if NNConfig['PPmodel'] in globals() and callable(globals()[NNConfig['PPmodel']]):
        if NNConfig['PPmodel']=='Net_relu_xavier_decay': 
            PPmodel = globals()[NNConfig['PPmodel']](layers, decay_rate=NNConfig['PPmodel_decay_rate'], decay_center=NNConfig['PPmodel_decay_center'])
        elif NNConfig['PPmodel']=='Net_relu_xavier_decayGaussian': 
            PPmodel = globals()[NNConfig['PPmodel']](layers, gaussian_std=NNConfig['PPmodel_gaussian_std'])
        else: 
            PPmodel = globals()[NNConfig['PPmodel']](layers)
    else:
        raise ValueError(f"Function {NNConfig['PPmodel']} does not exist.")
    print_memory_usage()

    # Initialize the ham class for each BulkSystem. 
    # dummy_ham is used to initialize and store the cached SOmats and NLmats in dict cachedMats. 
    # As I initialize dummy_ham, immediately load them into share memory
    # Use a dict "cachedMats_info" to store dtype and shape
    # Then remove dummy_ham, and any intermediate variables
    print("\nInitializing the ham class for each BulkSystem. ")
    hams = []
    cachedMats_info = None
    shm_dict_SO = None
    shm_dict_NL = None
    for iSys in range(nSystem): 
        start_time = time.time()

        # Here I should separate: 
        # 1. SObool = False --> Just initialize ham. No storage / moving is needed. 
        # 2. SObool = True, no parallel --> Initialize ham with cache. No storage / moving is needed.
        # 3. SObool = True, yes parallel --> Do the complicated storage / moving. 
        if not NNConfig['SObool']: 
            ham = Hamiltonian(systems[iSys], PPparams, atomPPOrder, device, NNConfig, SObool=NNConfig['SObool'])
        elif (NNConfig['SObool']) and ('num_cores' not in NNConfig): 
            ham = Hamiltonian(systems[iSys], PPparams, atomPPOrder, device, NNConfig, SObool=NNConfig['SObool'])
        else: 
            cachedMats_info = {}
            shm_dict_SO = {}
            shm_dict_NL = {}
            ham = Hamiltonian(systems[iSys], PPparams, atomPPOrder, device, NNConfig, SObool=True, cacheSO=False)
            dummy_ham = Hamiltonian(systems[iSys], PPparams, atomPPOrder, device, NNConfig, SObool=NNConfig['SObool'])

            if dummy_ham.SOmats is not None: 
                # reshape dummy_ham.SOmats into 4D arrays 
                # of shape (nkpt)*(nAtoms)*(2*nbasis) x (2*nbasis)
                tmpSOmats = dummy_ham.SOmats
                tmpSOmats_4d = np.array(tmpSOmats.tolist(), dtype=np.complex128).reshape((tmpSOmats.shape[0], tmpSOmats.shape[1], tmpSOmats[0,0].shape[0], tmpSOmats[0,0].shape[1]))
                for kidx in range(systems[iSys].getNKpts()):
                    SOkey = f"SO_{iSys}_{kidx}"
                    SOvalue = {'dtype': tmpSOmats_4d[kidx].dtype,
                        'shape': tmpSOmats_4d[kidx].shape,
                    }
                    cachedMats_info[SOkey] = SOvalue

                    # Move the SOmats to shared memory
                    shm_dict_SO[f"shm_SO_{iSys}_{kidx}"] = shared_memory.SharedMemory(create=True, size=tmpSOmats_4d[kidx].nbytes, name=f"SOmats_{iSys}_{kidx}")
                    tmp_arr = np.ndarray(cachedMats_info[f"SO_{iSys}_{kidx}"]['shape'], dtype=cachedMats_info[f"SO_{iSys}_{kidx}"]['dtype'], buffer=shm_dict_SO[f"shm_SO_{iSys}_{kidx}"].buf)  # Create a NumPy array backed by shared memory
                    tmp_arr[:] = tmpSOmats_4d[kidx][:]   # Copy the cached SOmat into shared memory

                del tmpSOmats, tmpSOmats_4d
            if dummy_ham.NLmats is not None: 
                # reshape dummy_ham.NLmats into 5D arrays 
                # of shape (nkpt)*(nAtoms)*(2)*(2*nbasis) x (2*nbasis)
                tmpNLmats = dummy_ham.NLmats
                tmpNLmats_5d = np.array(tmpNLmats.tolist(), dtype=np.complex128).reshape((tmpNLmats.shape[0], tmpNLmats.shape[1], tmpNLmats.shape[2], tmpNLmats[0,0,0].shape[0], tmpNLmats[0,0,0].shape[1]))
                for kidx in range(systems[iSys].getNKpts()):
                    NLkey = f"NL_{iSys}_{kidx}"
                    NLvalue = {'dtype': tmpNLmats_5d[kidx].dtype,
                        'shape': tmpNLmats_5d[kidx].shape,
                    }
                    cachedMats_info[NLkey] = NLvalue

                    # Move the NLmats to shared memory
                    shm_dict_NL[f"shm_NL_{iSys}_{kidx}"] = shared_memory.SharedMemory(create=True, size=tmpNLmats_5d[kidx].nbytes, name=f"NLmats_{iSys}_{kidx}")
                    tmp_arr = np.ndarray(cachedMats_info[f"NL_{iSys}_{kidx}"]['shape'], dtype=cachedMats_info[f"NL_{iSys}_{kidx}"]['dtype'], buffer=shm_dict_NL[f"shm_NL_{iSys}_{kidx}"].buf) 
                    tmp_arr[:] = tmpNLmats_5d[kidx][:] 

                del tmpNLmats, tmpNLmats_5d
            del dummy_ham
            gc.collect()
            print("Finished putting the cached SO and NLmats into shared memory ...")
        hams.append(ham)
        end_time = time.time()
        print(f"Elapsed time: {(end_time - start_time):.2f} seconds\n")
    print_memory_usage()

    oldFunc_plot_bandStruct_list = []
    oldFunc_totalMSE = 0
    for iSystem in range(nSystem): 
        start_time = time.time()
        # Ensure that when SOmats and NLmats are None, this still works! 
        oldFunc_bandStruct = hams[iSystem].calcBandStruct_noGrad(NNConfig, iSystem, cachedMats_info if cachedMats_info is not None else None)
        oldFunc_bandStruct.detach_()
        end_time = time.time()
        print(f"Old Zunger BS: Finished calculating {iSystem}-th band structure in the Zunger function form ... Elapsed time: {(end_time - start_time):.2f} seconds")
        oldFunc_plot_bandStruct_list.append(systems[iSystem].expBandStruct)
        oldFunc_plot_bandStruct_list.append(oldFunc_bandStruct)
        oldFunc_totalMSE += weighted_mse_bandStruct(oldFunc_bandStruct, systems[iSystem])
    fig = plotBandStruct(systems, oldFunc_plot_bandStruct_list, NNConfig['SHOWPLOTS'])
    print("The total bandStruct MSE = %e " % oldFunc_totalMSE)
    fig.suptitle("The total bandStruct MSE = %e " % oldFunc_totalMSE)
    fig.savefig(resultsFolder + 'oldFunc_plotBS.png')
    plt.close('all')
    print_memory_usage()

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
    print_memory_usage()

    print("Plotting and write pseudopotentials in the real and reciprocal space.")
    torch.cuda.empty_cache()
    PPmodel.eval()
    PPmodel.cpu()
    qmax = np.array([10.0, 20.0, 30.0])
    nQGrid = np.array([2048, 4096])
    nRGrid = np.array([2048, 4096])
    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, val_dataset, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, resultsFolder + 'initZunger_plotPP', resultsFolder + 'initZunger_pot', NNConfig['SHOWPLOTS'])
    print_memory_usage()

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
    print_memory_usage()

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
    PPmodel.cpu()
    qmax = np.array([10.0, 20.0, 30.0])
    nQGrid = np.array([2048, 4096])
    nRGrid = np.array([2048, 4096])
    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, val_dataset, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, resultsFolder + 'final_plotPP', resultsFolder + 'final_pot', NNConfig['SHOWPLOTS'])
    plot_memory_usage(resultsFolder)

    if shm_dict_SO is not None: 
        for shm in shm_dict_SO.values():
            shm.close()
            shm.unlink()
    if shm_dict_NL is not None:
        for shm in shm_dict_NL.values():
            shm.close()
            shm.unlink()


if __name__ == "__main__":
    MEMORY_FLAG = False
    if MEMORY_FLAG:
        main = profile(main)

    main("CALCS/CsPbI3_test/inputs/", "CALCS/CsPbI3_test/results/")
