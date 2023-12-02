import numpy as np
import scipy.linalg
import torch
import pathlib
pwd = pathlib.Path(__file__).parent.resolve()

from utils.nn_models import *
from utils.ham import Hamiltonian
from utils.read import bulkSystem
from constants.constants import AUTOEV

# just test on cpu
device = torch.device("cpu")


# read and set up system first system (no soc)
system1_nosoc = bulkSystem()
system1_nosoc.setSystem(f"{pwd}/inputs/defpot/system_0_nosoc.par")
system1_nosoc.setInputs(f"{pwd}/inputs/defpot/input_0_nosoc.par")
system1_nosoc.setKPointsAndWeights(f"{pwd}/inputs/defpot/kpoints_0_nosoc.par")
system1_nosoc.setExpBS(f"{pwd}/inputs/expBandStruct_0.par")
atomPPorder = np.unique(system1_nosoc.atomTypes)

# set up zunger potential
PPparams = {}
totalParams = torch.empty(0,5)
for atomType in atomPPorder:
    file_path = f"{pwd}/inputs/defpot/init_{atomType}Params.par"
    with open(file_path, 'r') as file:
        a = torch.tensor([float(line.strip()) for line in file])
    totalParams = torch.cat((totalParams, a.unsqueeze(0)), dim=0)
    PPparams[atomType] = a




ham1 = Hamiltonian(system1_nosoc, PPparams, atomPPorder, device)
bs1 = ham1.calcBandStruct()
print("\ntesting no SOC first\n")
print(f"no def vbm, cbm energies: {bs1[0,7]:.6f}, {bs1[0,8]:.6f}")

# now read deformed system file (lattice param multiplied by 1.0001)
system2_nosoc = bulkSystem()
system2_nosoc.setSystem(f"{pwd}/inputs/defpot/system_1_nosoc.par")
system2_nosoc.setInputs(f"{pwd}/inputs/defpot/input_0_nosoc.par")
system2_nosoc.setKPointsAndWeights(f"{pwd}/inputs/defpot/kpoints_0_nosoc.par")
system2_nosoc.setExpBS(f"{pwd}/inputs/expBandStruct_0.par")

ham2 = Hamiltonian(system2_nosoc, PPparams, atomPPorder, device)
bs2 = ham2.calcBandStruct()
print(f"DEF (system) vbm, cbm energies: {bs2[0,7]:.6f}, {bs2[0,8]:.6f}")


# now use the deformation potential routine
print(f"\nusing buildHtot_def() routine...")
Hdef = ham1.buildHtot_def(0, scale=1.0001)
vals, vecs = scipy.linalg.eigh(Hdef)
print(f"DEF (ham) vbm, cbm energies: {AUTOEV*vals[3]:.6f}, {AUTOEV*vals[4]:.6f}")

# confirm that original Htot build is okay
bs_tmp = ham1.calcBandStruct()
print(f"checking that everything returns to normal:")
print(f"new, no def vbm, cbm: {bs_tmp[0,7]:.6f}, {bs_tmp[0,8]:.6f}")





# now do with soc
print("\n\nnow testing with SOC...\n")
system1_soc = bulkSystem()
system1_soc.setSystem(f"{pwd}/inputs/defpot/system_0_soc.par")
system1_soc.setInputs(f"{pwd}/inputs/defpot/input_0_soc.par")
system1_soc.setKPointsAndWeights(f"{pwd}/inputs/defpot/kpoints_0_soc.par")
system1_soc.setExpBS(f"{pwd}/inputs/soc/bandStruct_0.dat")
atomPPorder = np.unique(system1_soc.atomTypes)


# new zunger potential
PPparams = {}
totalParams = torch.empty(0,9)
for atomType in atomPPorder:
    file_path = f"{pwd}/inputs/defpot/{atomType}Params.par"
    with open(file_path, 'r') as file:
        a = torch.tensor([float(line.strip()) for line in file])
    totalParams = torch.cat((totalParams, a.unsqueeze(0)), dim=0)
    PPparams[atomType] = a

ham1 = Hamiltonian(system1_soc, PPparams, atomPPorder, device, SObool=True)
bs1 = ham1.calcBandStruct()
print(f"no def vbm, cbm energies: {bs1[0,25]:.6f}, {bs1[0,26]:.6f}")


system2_soc = bulkSystem()
system2_soc.setSystem(f"{pwd}/inputs/defpot/system_1_soc.par")
system2_soc.setInputs(f"{pwd}/inputs/defpot/input_0_soc.par")
system2_soc.setKPointsAndWeights(f"{pwd}/inputs/defpot/kpoints_0_soc.par")
system2_soc.setExpBS(f"{pwd}/inputs/soc/bandStruct_0.dat")

ham2 = Hamiltonian(system2_soc, PPparams, atomPPorder, device, SObool=True)
bs2 = ham2.calcBandStruct()
print(f"DEF (system) vbm, cbm energies: {bs2[0,25]:.6f}, {bs2[0,26]:.6f}")

# now use the deformation potential routine
print(f"\nusing buildHtot_def() routine...")
Hdef = ham1.buildHtot_def(0, scale=1.0001)
vals, vecs = scipy.linalg.eigh(Hdef, subset_by_index=[0,30], driver='evr')
print(f"DEF (ham) vbm, cbm energies: {AUTOEV*vals[25]:.6f}, {AUTOEV*vals[26]:.6f}")

# confirm that original Htot build is okay
bs_tmp = ham1.calcBandStruct()
print(f"checking that everything returns to normal:")
print(f"new, no def vbm, cbm: {bs_tmp[0,25]:.6f}, {bs_tmp[0,26]:.6f}")
