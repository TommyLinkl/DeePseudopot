import numpy as np
import scipy.linalg
import torch
import pathlib
import os
pwd = pathlib.Path(__file__).parent.resolve()

from utils.ham import Hamiltonian
from utils.read import bulkSystem
from utils.constants import *
from utils.fit_mc import MonteCarloFit, read_mc_opts

# just test on cpu
device = torch.device("cpu")


# read and set up system first system (no coupling)
system1 = bulkSystem()
system1.setSystem(f"{pwd}/inputs/montecarlo/system_0.par")
system1.setInputs(f"{pwd}/inputs/montecarlo/input_0.par")
system1.setKPointsAndWeights(f"{pwd}/inputs/montecarlo/kpoints_0.par")
system1.setQPointsAndWeights(f"{pwd}/inputs/montecarlo/qpoints_0.par")
system1.setExpBS(f"{pwd}/inputs/montecarlo/expBandStruct_0.par")
system1.setExpCouplings(f"{pwd}/inputs/montecarlo/expCoupling_0.par")
atomPPorder = np.unique(system1.atomTypes)

# set up zunger potential
PPparams = {}
totalParams = torch.empty(0,9)
for atomType in atomPPorder:
    file_path = f"{pwd}/inputs/montecarlo/{atomType}Params.par"
    with open(file_path, 'r') as file:
        a = torch.tensor([float(line.strip()) for line in file])
    totalParams = torch.cat((totalParams, a.unsqueeze(0)), dim=0)
    PPparams[atomType] = a

# now read monte carlo options (except paramSteps)
mc_opts = read_mc_opts(f"{pwd}/inputs/montecarlo/mcOpts1.par")

# now read paramSteps, if there are any
paramSteps = {}
anyFile = 0
for atomType in atomPPorder:
    file_path = f"{pwd}/inputs/montecarlo/{atomType}ParamSteps.par"
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



ham1 = Hamiltonian(system1, PPparams, atomPPorder, device, SObool=False, coupling=False)
optimizer = MonteCarloFit(ham1, f"{pwd}/mc_out/nocpl/", paramSteps=paramSteps, **mc_opts)
print("\ntesting no coupling first\n")
print(f"...writing output and chk files to {pwd}/mc_out/nocpl/")
optimizer.run_mc()





# now do with coupling
print("\n\nnow testing with coupling...\n")
print(f"...writing output and chk files to {pwd}/mc_out/cpl/")

mc_opts = read_mc_opts(f"{pwd}/inputs/montecarlo/mcOpts2.par")

ham2 = Hamiltonian(system1, PPparams, atomPPorder, device, SObool=False, coupling=True)
optimizer = MonteCarloFit(ham2, f"{pwd}/mc_out/cpl/", paramSteps=paramSteps, **mc_opts)
optimizer.run_mc()