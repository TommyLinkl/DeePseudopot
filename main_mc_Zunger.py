import os, sys

#from utils.config_threads import configure_threads

# Must be called *before* importing numpy, scipy, torch, etc.
#configure_threads(n_threads=4)

import numpy as np
import scipy.linalg
import torch
import pathlib

pwd = pathlib.Path(__file__).parent.resolve()

from utils.ham import Hamiltonian, initAndCacheHams
from utils.read import BulkSystem, setAllBulkSystems, read_NNConfigFile, read_ParamSteps, read_LSDparams, setNN
from utils.constants import *
from utils.fit_mc import MonteCarloFit, read_mc_opts
from utils.pp_func import FT_converge_and_write_pp, pot_func, realSpacePot, plotZungerPP, pot_funcLSD
from utils.NN_train import write_PP_qSpace
from utils.init_NN_train import init_ZungerPP
from utils.local_structure_correction import calcLocalSymmDescriptor

def main_mc_Zunger(inputsFolder = 'inputs/', resultsFolder = 'results/'):
    
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    device = torch.device("cpu")

    os.makedirs(resultsFolder, exist_ok=True)

    NNConfig = read_NNConfigFile(inputsFolder + 'NN_config.par')
    nSystem = NNConfig['nSystem']
    
    # Read and set up systems
    print(f"\nReading and setting up the BulkSystems.")
    systems, atomPPOrder, nPseudopot, PPparams, totalParams, localPotParams = setAllBulkSystems(nSystem, inputsFolder, resultsFolder)

    # If local environment corrections are required, then compute symmetry desriptors for each atom N_\alpha
    if NNConfig['local_env_corr']:
        nLSDBasis = systems[0].nLSDBasis
        # Set up the neural network to predict the LSD coefficients
        lsd_layers = [1] + NNConfig['lsd_hiddenLayers'] + [nLSDBasis]
        LSDmodels = {}
        for atom in atomPPOrder:
          LSDmodels[atom] = setNN(NNConfig, nLSDBasis, layers=lsd_layers)
    else:
        LSDmodels = None

    
    # Initialize the ham class for each BulkSystem. Cache the SO and NL mats. 
    hams, cachedMats_info, shm_dict_SO, shm_dict_NL = initAndCacheHams(systems, NNConfig, PPparams, atomPPOrder, device, LSDmodels=LSDmodels)
    print(f"PPparams:\n{PPparams}")

    # now read monte carlo options (except paramSteps)
    mc_opts = read_mc_opts(f"{inputsFolder}mcOpts1.par")
    print(f"mc_opts = {mc_opts}")
    
    # now read paramSteps, if there are any
    paramSteps = read_ParamSteps(atomPPOrder, inputsFolder)
    print(f"paramSteps: \n{paramSteps}")

    
    optimizer = MonteCarloFit(hams, f"{resultsFolder}", nSystems=NNConfig["nSystem"], paramSteps=paramSteps, **mc_opts)
    
    print(f"\n...writing output and chk files to {resultsFolder}")
    bestPP_params = optimizer.run_mc(cachedMats_info=cachedMats_info)
    print(bestPP_params)
    ############# Writing the trained NN PP ############# 
    print("Writing the pseudopotentials in the real and reciprocal space for the parametrized PPs.")
    qmax = 40.0
    nQGrid = 4096
    nRGrid = 4096
    
    print(f"\n{'#' * 40}\nWriting the pseudopotentials")
    qGrid = torch.linspace(0.0, qmax, nQGrid).view(-1, 1)
    v_qs = []
    for atom in atomPPOrder:
      PPs = bestPP_params[atom]
      vq = pot_func(qGrid, PPs)
      v_qs.append(vq.view(-1).detach().numpy())
      (vr, rSpacePot) = realSpacePot(qGrid, vq, nRGrid)
      pot = torch.cat((vr, rSpacePot), dim=1).detach().numpy()
      qpot = torch.cat((qGrid, vq.view(-1,1).detach()), dim=1).detach().numpy()
      np.savetxt(f"{resultsFolder}/final_pot_q_{atom}.dat", qpot, delimiter=' ', fmt='%e')
      np.savetxt(f"{resultsFolder}/final_pot_{atom}.dat", pot, delimiter=' ', fmt='%e')
    
    # v_qs = np.array(v_qs)
    
    # fig = plotZungerPP(atomPPOrder, qGrid, v_qs.T, nRGrid, NNConfig["SHOWPLOTS"])
    # fig.savefig(f"{resultsFolder}/final_pots.pdf", format='pdf')

    ############# Free the shared data ############# 
    if shm_dict_SO is not None: 
        for shm in shm_dict_SO.values():
            shm.close()
            shm.unlink()
    if shm_dict_NL is not None:
        for shm in shm_dict_NL.values():
            shm.close()
            shm.unlink()

    return



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main_mc_Zunger.py <inputsFolder> <resultsFolder> ")
        sys.exit(1)

    inputsFolder = sys.argv[1]
    resultsFolder = sys.argv[2]
    main_mc_Zunger(inputsFolder, resultsFolder)
    
