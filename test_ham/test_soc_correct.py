import os
import torch
import numpy as np
import time
import pathlib

torch.set_default_dtype(torch.float64)

from utils.nn_models import *
from utils.ham import Hamiltonian
from utils.read import BulkSystem, read_PPparams, read_NNConfigFile

print("***************************")
print("""for this test to pass, make sure the ...Integral_dan() functions
are called when the hamiltonian is constructing SO and NL matrices.
      
Actually, not switching to ...Integral_dan() also gives the correct results. 
This further demonstrates the integrals are calculated correctly. 
      
This test requires at least 64GB of memory to cache the SO and NL matrices. 
We recommend running this test on a single Perlmutter compute node interactively.""")
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

PPparams, totalParams = read_PPparams(atomPPorder, f"{pwd}/inputs/soc/")

NNConfig = read_NNConfigFile(f"{pwd}/inputs/NN_config.par")

print("\n\n***************************")
print("Test with cacheing")
print("***************************")
start_time = time.time()
ham1 = Hamiltonian(system, PPparams, atomPPorder, device, NNConfig=NNConfig, iSystem=0, SObool=True)
bs_new = ham1.calcBandStruct()
end_time = time.time()
print(f"Finished calculating the SOC band structure... Elapsed time: {(end_time - start_time):.2f} seconds")
print(f"SOC, zunger-style pot: c code and python code return same band energies: {torch.allclose(bs_old, bs_new)}")

if not torch.allclose(bs_old, bs_new):
    print("bs_c[0,:]")
    print(bs_old[0,:])
    print("bs_new[0,:]")
    print(bs_new[0,:])

    print("\n\nbs_c - bs_new:")
    print(bs_old-bs_new)


print("\n\n***************************")
print("Test without cacheing")
print("***************************")
start_time = time.time()
ham1 = Hamiltonian(system, PPparams, atomPPorder, device, NNConfig=NNConfig, iSystem=0, SObool=True, cacheSO=False)
bs_new = ham1.calcBandStruct()
end_time = time.time()
print(f"Finished calculating the SOC band structure... Elapsed time: {(end_time - start_time):.2f} seconds")
print(f"SOC, zunger-style pot: c code and python code return same band energies: {torch.allclose(bs_old, bs_new)}")

if not torch.allclose(bs_old, bs_new):
    print("bs_c[0,:]")
    print(bs_old[0,:])
    print("bs_new[0,:]")
    print(bs_new[0,:])

    print("\n\nbs_c - bs_new:")
    print(bs_old-bs_new)

