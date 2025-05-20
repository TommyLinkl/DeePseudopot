import numpy as np
import scipy.linalg
import torch
import pathlib
import os, sys
pwd = pathlib.Path(__file__).parent.resolve()

from utils.ham import Hamiltonian
from utils.read import BulkSystem, read_NNConfigFile
from utils.constants import *
from utils.fit_mc import MonteCarloFit, read_mc_opts

def main_mc_Zunger(inputsFolder = 'inputs/', resultsFolder = 'results/'): 
    device = torch.device("cpu")
    os.makedirs(resultsFolder, exist_ok=True)

    # read and set up system first system (no coupling)
    system1 = BulkSystem()
    system1.setSystem(f"{inputsFolder}system_0.par")
    system1.setInputs(f"{inputsFolder}input_0.par")
    system1.setKPointsAndWeights(f"{inputsFolder}kpoints_0.par")
    # system1.setQPointsAndWeights(f"{inputsFolder}qpoints_0.par")
    system1.setExpBS(f"{inputsFolder}expBandStruct_0.par")
    # system1.setExpCouplings(f"{inputsFolder}expCoupling_0.par")
    system1.setBandWeights(f"{inputsFolder}bandWeights_0.par")
    
    atomPPorder = np.unique(system1.atomTypes)

    # set up zunger potential
    PPparams = {}
    totalParams = torch.empty(0,9)
    for atomType in atomPPorder:
        file_path = f"{inputsFolder}init_{atomType}Params.par"
        with open(file_path, 'r') as file:
            a = torch.tensor([float(line.strip()) for line in file])
        totalParams = torch.cat((totalParams, a.unsqueeze(0)), dim=0)
        PPparams[atomType] = a

    # now read monte carlo options (except paramSteps)
    mc_opts = read_mc_opts(f"{inputsFolder}mcOpts1.par")

    # now read paramSteps, if there are any
    paramSteps = {}
    anyFile = 0
    for atomType in atomPPorder:
        file_path = f"{inputsFolder}{atomType}ParamSteps.par"
        if os.path.isfile(file_path):
            anyFile += 1
            with open(file_path, 'r') as file:
                steps = [float(line.strip()) for line in file]
                assert len(steps) == 9 or len(steps) == 5
                paramSteps[atomType] = steps
    if anyFile == 0:
        paramSteps = None
    elif anyFile != len(atomPPorder):
        raise ValueError("must supply a paramStep file for each atom type")



    NNConfig = read_NNConfigFile(f"{inputsFolder}NN_config.par")
    ham1 = Hamiltonian(system1, PPparams, atomPPorder, device, NNConfig=NNConfig, iSystem=0, SObool=True, coupling=False)
    optimizer = MonteCarloFit(ham1, f"{resultsFolder}", paramSteps=paramSteps, **mc_opts)
    print("\ntesting no coupling first\n")
    print(f"...writing output and chk files to {resultsFolder}")
    optimizer.run_mc()



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main_mc_Zunger_III-V.py <inputsFolder> <resultsFolder> ")
        sys.exit(1)

    inputsFolder = sys.argv[1]
    resultsFolder = sys.argv[2]
    main_mc_Zunger(inputsFolder, resultsFolder)
    
