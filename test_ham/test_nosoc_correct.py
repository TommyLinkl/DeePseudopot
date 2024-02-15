import numpy as np
import torch
from torch.utils.data import DataLoader
import pathlib

from utils.nn_models import *
from utils.init_NN_train import init_Zunger_data
from utils.bandStruct import calcHamiltonianMatrix_GPU, calcBandStruct_GPU
from utils.ham import Hamiltonian
from utils.read import BulkSystem

# just test on cpu
device = torch.device("cpu")

# read test NN config
pwd = pathlib.Path(__file__).parent.resolve()
PPmodel = Net_relu_xavier_decay2([1,20,20,20,2])
PPmodel.load_state_dict(torch.load(f"{pwd}/epoch_199_PPmodel.pth", map_location=device) )

# read and set up system
system = BulkSystem()
system.setSystem(f"{pwd}/inputs/system_0.par")
system.setInputs(f"{pwd}/inputs/input_0.par")
system.setKPointsAndWeights(f"{pwd}/inputs/kpoints_0.par")
system.setExpBS(f"{pwd}/inputs/expBandStruct_0.par")
atomPPorder = np.unique(system.atomTypes)

# old band structure
bs_old = calcBandStruct_GPU(True, PPmodel, system, atomPPorder, [], device)

# new band structure
ham1 = Hamiltonian(system, {}, atomPPorder, device, SObool=False,
                   NN_locbool=True, model=PPmodel)
bs_new = ham1.calcBandStruct()

print(f"no SOC, NN: new and old band structure methods return same energies (in same format): {torch.allclose(bs_old, bs_new)}")


# now test zunger potential
PPparams = {}
totalParams = torch.empty(0,5)
for atomType in atomPPorder:
    file_path = f"{pwd}/inputs/init_{atomType}Params.par"
    with open(file_path, 'r') as file:
        a = torch.tensor([float(line.strip()) for line in file])
    totalParams = torch.cat((totalParams, a.unsqueeze(0)), dim=0)
    PPparams[atomType] = a

bs_old = calcBandStruct_GPU(False, PPmodel, system, atomPPorder, totalParams, device)

ham2 = Hamiltonian(system, PPparams, atomPPorder, device)
bs_new = ham2.calcBandStruct()

print(f"no SOC, zunger pot: new and old band structure methods return same energies (in same format): {torch.allclose(bs_old, bs_new)}")

