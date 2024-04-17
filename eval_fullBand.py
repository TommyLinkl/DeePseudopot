import os, time, sys
import torch
import numpy as np

from utils.read import read_NNConfigFile, setAllBulkSystems, setNN
from utils.pp_func import FT_converge_and_write_pp
from utils.init_NN_train import init_ZungerPP
from utils.NN_train import evalBS_noGrad
from utils.ham import Hamiltonian

def eval_fullBand(inputsFolder = 'inputs_evalFullBand/', resultsFolder = 'results_evalFullBand/'):
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
              This routine performs a single-shot band structure calculation on an existing neural network model. 
              Please provide init_PPmodel.pth in the input folder.""")

    # I can't store and cache all the SO and NL mats ahead of time due to memory limitations. 
    # I will need to calculate the SO and NL mats on the fly 
    print("\nInitializing the ham class for each BulkSystem. Not cache-ing the corresponding SO and NL mats for memory issues. ")
    hams = []
    for iSys, sys in enumerate(systems):
        start_time = time.time()
        ham = Hamiltonian(sys, PPparams, atomPPOrder, device, NNConfig, iSys, SObool=NNConfig['SObool'], cacheSO=False)
        hams.append(ham)
        end_time = time.time()
        print(f"Elapsed time: {(end_time - start_time):.2f} seconds\n")

    # Initialize the NN according to the provided file init_PPmodel.pth
    PPmodel, ZungerPPFunc_val = init_ZungerPP(inputsFolder, PPmodel, atomPPOrder, localPotParams, nPseudopot, NNConfig, device, resultsFolder)

    # Calculate bandStructure with the old function form with parameters given in PPparams
    oldFunc_totalMSE = evalBS_noGrad(None, f'{resultsFolder}oldFunc_plotBS.png', 'Old Zunger BS', NNConfig, hams, systems, writeBS=True)

    # Evaluate the band structures and pseudopotentials for the initialized NN
    print("\nEvaluating band structures using the input NN pseudopotentials. ")
    init_totalMSE = evalBS_noGrad(PPmodel, f'{resultsFolder}eval_plotBS.png', 'Eval NN BS', NNConfig, hams, systems, writeBS=True)

    # Write BS to file

    print("Converge the pseudopotentials in the real and reciprocal space. ")
    qmax = np.array([10.0, 20.0, 30.0])
    nQGrid = np.array([2048, 4096])
    nRGrid = np.array([2048, 4096])
    torch.cuda.empty_cache()
    PPmodel.eval()
    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, f'{resultsFolder}eval_plotPP', f'{resultsFolder}eval_pot', NNConfig['SHOWPLOTS'])

    # Write pot.dat to file





if len(sys.argv) != 3:
    print("Usage: python eval_fullBand.py <inputsFolder> <resultsFolder> ")
    sys.exit(1)

inputsFolder = sys.argv[1]
resultsFolder = sys.argv[2]
eval_fullBand(inputsFolder, resultsFolder)
