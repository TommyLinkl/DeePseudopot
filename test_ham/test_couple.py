import numpy as np
import scipy.linalg
import torch
from torch.utils.data import DataLoader
import pathlib
import copy

from utils.nn_models import *
from utils.init_NN_train import init_Zunger_data
from utils.bandStruct import calcHamiltonianMatrix_GPU, calcBandStruct_GPU
from utils.ham import Hamiltonian
from utils.read import bulkSystem
from utils.constants import *

# just test on cpu
device = torch.device("cpu")
torch.set_printoptions(precision=8)

# read and set up system
pwd = pathlib.Path(__file__).parent.resolve()
system = bulkSystem()
system.setSystem(f"{pwd}/inputs/couple/system_0.par")
system.setInputs(f"{pwd}/inputs/couple/input_0.par")
system.setKPointsAndWeights(f"{pwd}/inputs/couple/kpoints_0.par")
system.setQPointsAndWeights(f"{pwd}/inputs/couple/qpoints_0.par")
system.setExpBS(f"{pwd}/inputs/couple/expBandStruct_0.par")
atomPPorder = np.unique(system.atomTypes)

print("initial atom positions:")
for i in range(2):
    print(f"{system.atomTypes[i]}: {system.atomPos[i]}")

# build zunger potential
PPparams = {}
totalParams = torch.empty(0,9) # see the readme for definition of all 9 params.
                               # They are not all used in this test. Only
                               # params 0-3,5-7 are used (local pot, SOC,
                               # and nonlocal, no long range or strain)
for atomType in atomPPorder:
    file_path = f"{pwd}/inputs/couple/{atomType}Params_tmp.par"
    with open(file_path, 'r') as file:
        a = torch.tensor([float(line.strip()) for line in file])
    totalParams = torch.cat((totalParams, a.unsqueeze(0)), dim=0)
    PPparams[atomType] = a


# construct initial hamiltonian for eigenvecs and for finite difference
ham1 = Hamiltonian(system, PPparams, atomPPorder, device, SObool=False, coupling=True)
h = ham1.buildHtot(ham1.idx_gap)
h = h.numpy(force=True)
vals, vecs = scipy.linalg.eigh(h, subset_by_index=[0,16], driver='evr')
#vb_vec = vecs[:,25]
#vb_vec = vecs[:,12]
vb_vec = vecs[:,7]
#cb_vec = vecs[:,26]
#cb_vec = vecs[:,13]
cb_vec = 1/np.sqrt(2) * (vecs[:,8] + vecs[:,9]) # avg over degen subspace
#e1s = [vals[25], vals[26]]
e1s = [vals[7], vals[8]]
print(f"\n\nInitial energies VBM: {e1s[0]}, CBM: {e1s[1]}")
print(f"vb-1 degen? {abs(vals[7] - vals[6]) < 1e-15}, {abs(vals[7] - vals[6])}")
print(f"vb-2 degen? {abs(vals[7] - vals[6]) < 1e-15}, {abs(vals[7] - vals[5])}")
print(f"cb+1 degen? {abs(vals[8] - vals[9]) < 1e-15}, {abs(vals[8] - vals[9])}")
print(f"cb+2 degen? {abs(vals[8] - vals[10]) < 1e-15}, {abs(vals[8] - vals[10])}")




# compute analytic derivs of potential
get_derivs = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
dV_dict = ham1.buildCouplingMats(1, atomgammaidxs=get_derivs) # qidx corresponds to 0,0,0



print("\n\nall analytic derivs")
# because the CB manifold is exactly degenerate, an arbitrary unitary rotation
# is allowed in the space of these eigenvecs. This means we need to average
# over the degenerate eigenvecs AND over the symmetry-equivalent derivative
# directions, which in this case is all 3 (x,y,z)
cd_cb_derivs = [0.0, 0.0, 0.0]
se_cb_derivs = [0.0, 0.0, 0.0]
for key in get_derivs:
    tmp = np.dot(np.conj(cb_vec), np.dot(dV_dict[key], cb_vec))
    if key[0] == 0:
        cd_cb_derivs[0] += tmp /3
        cd_cb_derivs[1] += tmp /3
        cd_cb_derivs[2] += tmp /3
    elif key[0] == 1:
        se_cb_derivs[0] += tmp /3
        se_cb_derivs[1] += tmp /3
        se_cb_derivs[2] += tmp /3

for key in get_derivs:
    if key[1] == 0:
        d = 'x'
    elif key[1] == 1:
        d = 'y'
    else:
        d = 'z'

    print(f"{system.atomTypes[key[0]]}, d/dR_{d}, vb-vb: {np.dot(np.conj(vb_vec), np.dot(dV_dict[key], vb_vec))}")
    if key[0] == 0:
        print(f"{system.atomTypes[key[0]]}, d/dR_{d}, cb-cb: {cd_cb_derivs[key[1]]}")
    elif key[0] == 1:
                print(f"{system.atomTypes[key[0]]}, d/dR_{d}, cb-cb: {se_cb_derivs[key[1]]}")




# compute dE/dy by loading a slightly deformed system along y
print("\n\nConverging d/dy finite diff for Cd...")
for sysid in range(7):
    system_dx = bulkSystem()
    system_dx.setSystem(f"{pwd}/inputs/couple/system_dx{sysid}.par")
    system_dx.setInputs(f"{pwd}/inputs/couple/input_0.par")
    system_dx.setKPointsAndWeights(f"{pwd}/inputs/couple/kpoints_0.par")
    system_dx.setQPointsAndWeights(f"{pwd}/inputs/couple/qpoints_0.par")
    system_dx.setExpBS(f"{pwd}/inputs/couple/expBandStruct_0.par")

    print("\nmanual system d/dy positions:")
    for i in range(2):
        print(f"{system_dx.atomTypes[i]}: {system_dx.atomPos[i]}")

    ham_dx = Hamiltonian(system_dx, PPparams, atomPPorder, device, SObool=False, coupling=False)
    hdx = ham_dx.buildHtot(ham1.idx_gap)
    hdx = hdx.numpy(force=True)
    vals_dx, vecs_dx = scipy.linalg.eigh(hdx, subset_by_index=[0,16], driver='evr')
    e1s_dx = [vals_dx[7], 0.5*(vals_dx[8] + vals_dx[9])]  # avg over degen cb subspace

    if sysid == 0:
        print(f"dy = 1.14485e-3")
        print(f"dE/dCd_y by finite diff. vbm: {(e1s_dx[0]-e1s[0])/(1.14485e-3)},  cbm: {(e1s_dx[1]-e1s[1])/(1.14485e-3)}")
    elif sysid == 1:
        print(f"dy = 5.72412e-4")
        print(f"dE/dCd_y by finite diff. vbm: {(e1s_dx[0]-e1s[0])/(5.72412e-4)},  cbm: {(e1s_dx[1]-e1s[1])/(5.72412e-4)}")
    elif sysid == 2:
        print(f"dy = 1.14485e-4")
        print(f"dE/dCd_y by finite diff. vbm: {(e1s_dx[0]-e1s[0])/(1.14485e-4)},  cbm: {(e1s_dx[1]-e1s[1])/(1.14485e-4)}")
    elif sysid == 3:
        print(f"dy = 5.72412e-5")
        print(f"dE/dCd_y by finite diff. vbm: {(e1s_dx[0]-e1s[0])/(5.72412e-5)},  cbm: {(e1s_dx[1]-e1s[1])/(5.72412e-5)}")
    elif sysid == 4:
        print(f"dy = 1.14485e-5")
        print(f"dE/dCd_y by finite diff. vbm: {(e1s_dx[0]-e1s[0])/(1.14485e-5)},  cbm: {(e1s_dx[1]-e1s[1])/(1.14485e-5)}")
    elif sysid == 5:
        print(f"dy = 1.14485e-6")
        print(f"dE/dCd_y by finite diff. vbm: {(e1s_dx[0]-e1s[0])/(1.14485e-6)},  cbm: {(e1s_dx[1]-e1s[1])/(1.14485e-6)}")
    elif sysid == 6:
        print(f"dy = 1.14485e-7")
        print(f"dE/dCd_y by finite diff. vbm: {(e1s_dx[0]-e1s[0])/(1.14485e-7)},  cbm: {(e1s_dx[1]-e1s[1])/(1.14485e-7)}")





# now testing the calcCoupling function directly
print("\n\n\n\n---------------------------------------------")
print("now redoing the same analysis using calcCouplings() function")
print("(this prints the magnitude of the coupling)\n")

avg_dirs = {}   # use this dict to tell the calcCouplings functions which
                # atoms and directions to average over due to the degenerate CB
                # manifold. 
avg_dirs[0] = ('x','y','z')
avg_dirs[1] = ('x','y','z')

# need to set the eigenvectors by calling calcBandStruct()
_ = ham1.calcBandStruct()
cpl_dict = ham1.calcCouplings(qlist=[1,], atomgammaidxs=get_derivs, symm_equiv=avg_dirs) 

print("(atomidx, gamma, qidx, vb/cb).    |cpl|")
for key, value in cpl_dict.items():
    print(f"{key}.   |cpl|: {value.real}")





# now add in arbitrary SOC potential and repeat
print("\n\n\n\n---------------------------------------------")
print("now redoing the same analysis for potential with abritrary SOC and nonlocal terms\n")

# build zunger potential
PPparams = {}
totalParams = torch.empty(0,9) # see the readme for definition of all 9 params.
                               # They are not all used in this test. Only
                               # params 0-3,5-7 are used (local pot, SOC,
                               # and nonlocal, no long range or strain)
for atomType in atomPPorder:
    file_path = f"{pwd}/inputs/couple/{atomType}Params_soc.par"
    with open(file_path, 'r') as file:
        a = torch.tensor([float(line.strip()) for line in file])
    totalParams = torch.cat((totalParams, a.unsqueeze(0)), dim=0)
    PPparams[atomType] = a


# construct initial hamiltonian for eigenvecs and for finite difference
ham1 = Hamiltonian(system, PPparams, atomPPorder, device, SObool=True, coupling=True)
h = ham1.buildHtot(ham1.idx_gap)
h = h.numpy(force=True)
vals, vecs = scipy.linalg.eigh(h, subset_by_index=[0,32], driver='evr')
#vb_vec = vecs[:,7]
vb_vec = vecs[:, 15]
#cb_vec = vecs[:,8]
cb_vec = vecs[:, 16]
e1s = [vals[15], vals[16]]
print(f"\n\nInitial energies VBM: {e1s[0]}, CBM: {e1s[1]}")
print(f"vb-1 degen? {abs(vals[15] - vals[14]) < 1e-15}, {abs(vals[15] - vals[14])}")
print(f"vb-2 degen? {abs(vals[15] - vals[13]) < 1e-15}, {abs(vals[15] - vals[13])}")
print(f"cb+1 degen? {abs(vals[16] - vals[17]) < 1e-15}, {abs(vals[16] - vals[17])}")
print(f"cb+2 degen? {abs(vals[16] - vals[18]) < 1e-15}, {abs(vals[16] - vals[18])}")



# compute analytic derivs of potential
get_derivs = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
dV_dict = ham1.buildCouplingMats(1, atomgammaidxs=get_derivs) # qidx corresponds to 0,0,0


print("\n\nall analytic derivs")
for key in get_derivs:
    if key[1] == 0:
        d = 'x'
    elif key[1] == 1:
        d = 'y'
    else:
        d = 'z'
    print(f"{system.atomTypes[key[0]]}, d/dR_{d}, vb-vb: {np.dot(np.conj(vb_vec), np.dot(dV_dict[key], vb_vec))}")
    print(f"{system.atomTypes[key[0]]}, d/dR_{d}, cb-cb: {np.dot(np.conj(cb_vec), np.dot(dV_dict[key], cb_vec))}")




# compute dE/dy by loading a slightly deformed system along y
print("\n\nConverging d/dy finite diff for Cd...")
for sysid in range(7):
    system_dx = bulkSystem()
    system_dx.setSystem(f"{pwd}/inputs/couple/system_dx{sysid}.par")
    system_dx.setInputs(f"{pwd}/inputs/couple/input_0.par")
    system_dx.setKPointsAndWeights(f"{pwd}/inputs/couple/kpoints_0.par")
    system_dx.setQPointsAndWeights(f"{pwd}/inputs/couple/qpoints_0.par")
    system_dx.setExpBS(f"{pwd}/inputs/couple/expBandStruct_0.par")
    print("\nmanual system d/dy positions:")
    for i in range(2):
        print(f"{system_dx.atomTypes[i]}: {system_dx.atomPos[i]}")

    ham_dx = Hamiltonian(system_dx, PPparams, atomPPorder, device, SObool=True, coupling=False)
    hdx = ham_dx.buildHtot(ham1.idx_gap)
    hdx = hdx.numpy(force=True)
    vals_dx, vecs_dx = scipy.linalg.eigh(hdx, subset_by_index=[0,32], driver='evr')
    #vb_vec_dx = vecs_dx[:,7]
    vb_vec_dx = vecs[:,15]
    #cb_vec_dx = vecs_dx[:,8]
    cb_vec_dx = vecs[:,16]
    #e1s_dx = [vals_dx[7], vals_dx[8]]
    e1s_dx = [vals_dx[15], vals_dx[16]]

    if sysid == 0:
        print(f"dy = 1.14485e-3")
        print(f"dE/dCd_y by finite diff. vbm: {(e1s_dx[0]-e1s[0])/(1.14485e-3)},  cbm: {(e1s_dx[1]-e1s[1])/(1.14485e-3)}")
    elif sysid == 1:
        print(f"dy = 5.72412e-4")
        print(f"dE/dCd_y by finite diff. vbm: {(e1s_dx[0]-e1s[0])/(5.72412e-4)},  cbm: {(e1s_dx[1]-e1s[1])/(5.72412e-4)}")
    elif sysid == 2:
        print(f"dy = 1.14485e-4")
        print(f"dE/dCd_y by finite diff. vbm: {(e1s_dx[0]-e1s[0])/(1.14485e-4)},  cbm: {(e1s_dx[1]-e1s[1])/(1.14485e-4)}")
    elif sysid == 3:
        print(f"dy = 5.72412e-5")
        print(f"dE/dCd_y by finite diff. vbm: {(e1s_dx[0]-e1s[0])/(5.72412e-5)},  cbm: {(e1s_dx[1]-e1s[1])/(5.72412e-5)}")
    elif sysid == 4:
        print(f"dy = 1.14485e-5")
        print(f"dE/dCd_y by finite diff. vbm: {(e1s_dx[0]-e1s[0])/(1.14485e-5)},  cbm: {(e1s_dx[1]-e1s[1])/(1.14485e-5)}")
    elif sysid == 5:
        print(f"dy = 1.14485e-6")
        print(f"dE/dCd_y by finite diff. vbm: {(e1s_dx[0]-e1s[0])/(1.14485e-6)},  cbm: {(e1s_dx[1]-e1s[1])/(1.14485e-6)}")
    elif sysid == 6:
        print(f"dy = 1.14485e-7")
        print(f"dE/dCd_y by finite diff. vbm: {(e1s_dx[0]-e1s[0])/(1.14485e-7)},  cbm: {(e1s_dx[1]-e1s[1])/(1.14485e-7)}")
