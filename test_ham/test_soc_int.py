import numpy as np
import torch
from torch.utils.data import DataLoader
import pathlib

from utils.nn_models import *
from utils.init_NN_train import init_Zunger_data
from utils.bandStruct import calcHamiltonianMatrix_GPU, calcBandStruct_GPU
from utils.ham import Hamiltonian
from utils.read import bulkSystem

# just test on cpu
device = torch.device("cpu")

# read and set up system
pwd = pathlib.Path(__file__).parent.resolve()
system = bulkSystem()
system.setSystem(f"{pwd}/inputs/soc/system_0.par")
system.setInputs(f"{pwd}/inputs/soc/input_0.par")
system.setKPointsAndWeights(f"{pwd}/inputs/kpoints_0.par")
system.setExpBS(f"{pwd}/inputs/soc/bandStruct_0.dat")
atomPPorder = np.unique(system.atomTypes)


# now set up pseudpot params
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


ham1 = Hamiltonian(system, PPparams, atomPPorder, device, SObool=False)


# compare daniels SOC integral style with analytic form
b1 = ham1.basis.numpy(force=True)
b2 = ham1.basis.numpy(force=True)
nb1 = np.linalg.norm(b1, axis=1)
nb2 = np.linalg.norm(b2, axis=1)
int_dan = ham1._soIntegral_dan(nb1, nb2, 0.7)
int_closed = ham1._soIntegral_vect(nb1, nb2, 4.2488, 0.7)

print(f"SOC integrals: dan c style and python analytic form are the same: {np.allclose(int_dan[1:,1:], int_closed[1:,1:])}")

if not np.allclose(int_dan, int_closed):
    print("daniel's bessel:")
    print(ham1._bessel1(nb1*1.0, 1/(nb1*1.0 + 1e-10)))
    print(f"\nk[1].mag = {nb1[1]}")
    print(f"kp[1].mag = {nb2[1]}")
    print(f"k[-1].mag = {nb1[-1]}")
    print("\nint_dan[1:,1:]")
    print(int_dan[1:,1:])
    print("\n\nint analytic[1:,1:]")
    print(int_closed[1:,1:])


# now check nonlocal integral
int2_dan = ham1._nlIntegral_dan(nb1, nb2, 1.0, 1.5)
int2_scipy = ham1._nlIntegral_vect(nb1, nb2, 4.2488, 1.0, 1.5)
print(f"\n\n\nNL integrals: dan c style and scipy numeric form are the same: {np.allclose(int2_dan[1:,1:], int2_scipy[1:,1:])}")

if not np.allclose(int2_dan, int2_scipy):
    print("\nint_dan[1:,1:]")
    print(int_dan[1:,1:])
    print("\n\n\nint analytic[1:,1:]")
    print(int_closed[1:,1:])


