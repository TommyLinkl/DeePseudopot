import torch
import numpy as np
import time
import pathlib

from utils.nn_models import *
from utils.ham import Hamiltonian
from utils.read import BulkSystem, read_PPparams, read_NNConfigFile

print("***************************")
print("""for this test to pass, make sure the ...Integral_dan() functions
are called when the hamiltonian is constructing SO and NL matrices.""")
print("***************************")
# just test on cpu
device = torch.device("cpu")

# read and set up system
pwd = pathlib.Path(__file__).parent.resolve()
system = BulkSystem()
system.setSystem(f"{pwd}/inputs/soc/system_0.par")
system.setInputs(f"{pwd}/inputs/soc/input_0.par")
system.setKPointsAndWeights(f"{pwd}/inputs/soc/kpoints_0.par")
system.setExpBS(f"{pwd}/inputs/soc/bandStruct_0.dat")
atomPPorder = np.unique(system.atomTypes)

# reference band structure is loaded -- it is computed from c code
bs_old = system.expBandStruct


# now test zunger potential
'''
PPparams = {}
totalParams = torch.empty(0,9) # see the readme for definition of all 9 params.
                               # They are not all used in this test. Only
                               # params 0-3,5-7 are used (local pot, SOC,
                               # and nonlocal, no long range or strain)
for atomType in atomPPorder:
    file_path = f"{pwd}/inputs/soc/{atomType}Params.par"
    with open(file_path, 'r') as file:
        a = torch.tensor([float(line.strip()) for line in file])
    totalParams = torch.cat((totalParams, a.unsqueeze(0)), dim=0)
    PPparams[atomType] = a
'''
PPparams, totalParams = read_PPparams(atomPPorder, f"{pwd}/inputs/soc/")

NNConfig = read_NNConfigFile(f"{pwd}/inputs/NN_config.par")

start_time = time.time()
ham1 = Hamiltonian(system, PPparams, atomPPorder, device, NNConfig, iSystem=0, SObool=True)
bs_new = ham1.calcBandStruct()
end_time = time.time()
print(f"Finished calculating the SOC band structure... Elapsed time: {(end_time - start_time):.2f} seconds")

print(f"SOC, zunger-style pot: c code and python code return same band energies : {torch.allclose(bs_old, bs_new)}")

if not torch.allclose(bs_old, bs_new):
    print("bs_c[0,:]")
    print(bs_old[0,:])
    print("bs_new[0,:]")
    print(bs_new[0,:])

    print("\n\nbs_c - bs_new:")
    print(bs_old-bs_new)
