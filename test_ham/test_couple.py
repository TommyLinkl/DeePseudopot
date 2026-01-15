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
from utils.read import BulkSystem, read_NNConfigFile
from utils.constants import *

# just test on cpu
device = torch.device("cpu")
torch.set_printoptions(precision=8)
RUN_SOC = False

# read and set up system
pwd = pathlib.Path(__file__).parent.resolve()
system = BulkSystem()
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

NNConfig = read_NNConfigFile(f"{pwd}/inputs/NN_config.par")

# construct initial hamiltonian for eigenvecs and for finite difference
ham1 = Hamiltonian(system, PPparams, atomPPorder, device, NNConfig=NNConfig, iSystem=0, SObool=False, coupling=True)
h = ham1.buildHtot(ham1.idx_gap)
h = h.numpy(force=True)
vals, vecs = scipy.linalg.eigh(h, subset_by_index=[0,16], driver='evr')
# print(vals)
#vb_vec = vecs[:,25]
#vb_vec = vecs[:,12]
vb_vec = vecs[:,7]
#cb_vec = vecs[:,26]
#cb_vec = vecs[:,13]
cb_vec = 1/np.sqrt(2) * (vecs[:,8] + vecs[:,9]) # avg over degen subspace
#e1s = [vals[25], vals[26]]
e1s = [vals[7], vals[8]]
print(f"\n\nInitial energies VBM: {e1s[0]:.6f}, CBM: {e1s[1]:.6f} Hartree")
print(f"vb-1 degen? {abs(vals[7] - vals[6]) < 1e-10}, E_diff = {abs(vals[7] - vals[6]):.3e}")
print(f"vb-2 degen? {abs(vals[7] - vals[6]) < 1e-10}, E_diff = {abs(vals[7] - vals[5]):.3e}")
print(f"cb+1 degen? {abs(vals[8] - vals[9]) < 1e-10}, E_diff = {abs(vals[8] - vals[9]):.3e}")
print(f"cb+2 degen? {abs(vals[8] - vals[10]) < 1e-10}, E_diff = {abs(vals[8] - vals[10]):.3e}")



# compute analytic derivs of potential
get_derivs = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
dV_dict = ham1.buildCouplingMats(1, atomgammaidxs=get_derivs) # qidx corresponds to 0,0,0

print("\n\nanalytic derivs (averaged over degenerate cb subspace) calculated within this test script: ")
# because the CB manifold is exactly degenerate, an arbitrary unitary rotation
# is allowed in the space of these eigenvecs. This means we need to average
# over the degenerate eigenvecs AND over the symmetry-equivalent derivative
# directions, which in this case is all 3 (x,y,z)
cd_cb_derivs = [0.0, 0.0, 0.0]
se_cb_derivs = [0.0, 0.0, 0.0]
def _fmt_real_if_tiny_imag(val, tol=1e-12):
    if np.iscomplexobj(val) and abs(val.imag) < tol:
        return f"{val.real:.3e} (Re; |Im|<1e-12)"
    return f"{val}"

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

    vb_vb_val = np.dot(np.conj(vb_vec), np.dot(dV_dict[key], vb_vec))
    print(f"{system.atomTypes[key[0]]}, d/dR_{d}, vb-vb: {_fmt_real_if_tiny_imag(vb_vb_val)}")
    if key[0] == 0:
        print(f"{system.atomTypes[key[0]]}, d/dR_{d}, cb-cb: {_fmt_real_if_tiny_imag(cd_cb_derivs[key[1]])}")
    elif key[0] == 1:
                print(f"{system.atomTypes[key[0]]}, d/dR_{d}, cb-cb: {_fmt_real_if_tiny_imag(se_cb_derivs[key[1]])}")

print("\nanalytic derivs summary (Cd, d/dy, eV/Bohr):")
colw = 14
fmt = lambda v: f"{v:>{colw}.5e}"
auto_fd_cache = {}
print(
    f"{'dy':>{colw}} {'fd_vb':>{colw}} {'fd_cb':>{colw}} "
    f"{'an_vb':>{colw}} {'an_cb':>{colw}} "
    f"{'ham_fd_auto_vb':>{colw}} {'ham_fd_auto_cb':>{colw}}"
)
analytic_vb_cd_y = np.dot(np.conj(vb_vec), np.dot(dV_dict[(0,1)], vb_vec)).real * AUTOEV
analytic_cb_cd_y = np.real(cd_cb_derivs[1]) * AUTOEV
print(
    f"{'/':>{colw}} {'/':>{colw}} {'/':>{colw}} "
    f"{fmt(analytic_vb_cd_y)} {fmt(analytic_cb_cd_y)} "
    f"{'/':>{colw}} {'/':>{colw}}"
)



# compute dE/dy by loading a slightly deformed system along y
print("\n\nFinite diff, using manually set system.par, calculated within this test scipt. d/dy, for Cd...")
base_cd_pos = system.atomPos[0].clone()
base_scale = float(system.scale)
fd_results = []
for sysid in range(7):
    system_dx = BulkSystem()
    system_dx.setSystem(f"{pwd}/inputs/couple/system_dx{sysid}.par")
    system_dx.setInputs(f"{pwd}/inputs/couple/input_0.par")
    system_dx.setKPointsAndWeights(f"{pwd}/inputs/couple/kpoints_0.par")
    system_dx.setQPointsAndWeights(f"{pwd}/inputs/couple/qpoints_0.par")
    system_dx.setExpBS(f"{pwd}/inputs/couple/expBandStruct_0.par")

    print(f"{sysid}-th system.par for fd. ")
    for i in range(2):
        print(f"{system_dx.atomTypes[i]}: {system_dx.atomPos[i]}")
    dy = float((system_dx.atomPos[0][1] - base_cd_pos[1]))

    ham_dx = Hamiltonian(system_dx, PPparams, atomPPorder, device, NNConfig=NNConfig, iSystem=sysid, SObool=False, coupling=False)
    hdx = ham_dx.buildHtot(ham1.idx_gap)
    hdx = hdx.numpy(force=True)
    vals_dx, vecs_dx = scipy.linalg.eigh(hdx, subset_by_index=[0,16], driver='evr')
    e1s_dx = [vals_dx[7], 0.5*(vals_dx[8] + vals_dx[9])]  # avg over degen cb subspace
    fd_vb = (e1s_dx[0]-e1s[0]) / dy * AUTOEV
    fd_cb = (e1s_dx[1]-e1s[1]) / dy * AUTOEV
    fd_results.append((dy, fd_vb, fd_cb))
    print(f"Cd dy = {dy:.8e} Bohr")
    print(f"dE/dCd_y by finite diff for vbm: {fd_vb:.3e} eV/Bohr,  cbm: {fd_cb:.3e} eV/Bohr")


print("\nFinite-difference convergence (via system.par) summary (Cd, d/dy, eV/Bohr):")
fmt = lambda v: f"{v:>{colw}.5e}"
auto_fd_cache = {}
print(
    f"{'dy':>{colw}} {'fd_vb':>{colw}} {'fd_cb':>{colw}} "
    f"{'an_vb':>{colw}} {'an_cb':>{colw}} "
    f"{'ham_fd_auto_vb':>{colw}} {'ham_fd_auto_cb':>{colw}}"
)
for dy, fd_vb, fd_cb in fd_results:
    print(
        f"{fmt(dy)} {fmt(fd_vb)} {fmt(fd_cb)} "
        f"{fmt(analytic_vb_cd_y)} {fmt(analytic_cb_cd_y)} "
        f"{'/':>{colw}} {'/':>{colw}}"
    )


#######################################
print("\n\nham-coded finite difference for debugging (calcCouplings_diag_fd, eV/Bohr)")
def get_auto_fd(delta):
    if delta not in auto_fd_cache:
        auto_fd_cache[delta] = ham1.calcCouplings_diag_fd(delta=delta, debug=False, one_sided=True)
    return auto_fd_cache[delta]

def extract_auto(auto_dict, atomidx, gamma, band):
    for key, value in auto_dict.items():
        if key[0] == atomidx and key[1] == gamma and key[3] == band:
            return float(value)
    raise KeyError(f"Missing auto fd key for atom {atomidx}, gamma {gamma}, band {band}")

auto_fd_results = {}
print(
    f"{'dy':>{colw}} {'fd_vb':>{colw}} {'fd_cb':>{colw}} "
    f"{'ham_fd_auto_vb':>{colw}} {'ham_fd_auto_cb':>{colw}} "
    f"{'ham_fd_auto_vb_cen':>{colw}} {'ham_fd_auto_cb_cen':>{colw}}"
)
for dy, fd_vb, fd_cb in fd_results:
    auto_dict_one = get_auto_fd(abs(dy))
    auto_vb_one = extract_auto(auto_dict_one, 0, 1, 'vb')
    auto_cb_one = extract_auto(auto_dict_one, 0, 1, 'cb')
    auto_dict_cen = ham1.calcCouplings_diag_fd(delta=abs(dy), debug=False, one_sided=False)
    auto_vb_cen = extract_auto(auto_dict_cen, 0, 1, 'vb')
    auto_cb_cen = extract_auto(auto_dict_cen, 0, 1, 'cb')
    auto_fd_results[dy] = (auto_vb_one, auto_cb_one, auto_vb_cen, auto_cb_cen)
    print(f"{fmt(dy)} {fmt(fd_vb)} {fmt(fd_cb)} {fmt(auto_vb_one)} {fmt(auto_cb_one)} {fmt(auto_vb_cen)} {fmt(auto_cb_cen)}")


########################################
print("\n\nSummary (Cd, d/dy, eV/Bohr):")
fmt = lambda v: f"{v:>{colw}.5e}"
auto_fd_cache = {}
print(
    f"{'dy':>{colw}} {'fd_vb':>{colw}} {'fd_cb':>{colw}} "
    f"{'an_vb':>{colw}} {'an_cb':>{colw}} "
    f"{'ham_fd_auto_vb_cen':>{colw}} {'ham_fd_auto_cb_cen':>{colw}}"
)
analytic_vb_cd_y = np.dot(np.conj(vb_vec), np.dot(dV_dict[(0,1)], vb_vec)).real * AUTOEV
analytic_cb_cd_y = np.real(cd_cb_derivs[1]) * AUTOEV
for dy, fd_vb, fd_cb in fd_results:
    auto_vb, auto_cb, auto_vb_cen, auto_cb_cen = auto_fd_results[dy]
    print(
        f"{fmt(dy)} {fmt(fd_vb)} {fmt(fd_cb)} "
        f"{fmt(analytic_vb_cd_y)} {fmt(analytic_cb_cd_y)} "
        f"{fmt(auto_vb_cen)} {fmt(auto_cb_cen)}"
    )

# Using calcCouplings_diag_fd() directly at the smallest finite difference displacement, for 
# both Cd and Se atoms, in both x, y, z directions, for both VB and CB diagonal elements.
min_delta = min(abs(dy) for dy, _, _ in fd_results)
zero_vec = torch.zeros(3, dtype=system.qpts.dtype)
qidx_gamma = 0
for qid in range(system.getNQpts()):
    if torch.allclose(system.qpts[qid], zero_vec, atol=1e-12):
        qidx_gamma = qid
        break

print("\n\ncalcCouplings_diag_fd at smallest dy (eV/Bohr):")
print(f"delta = {min_delta:.3e} Bohr, qidx_gamma = {qidx_gamma}")
auto_min_dict = ham1.calcCouplings_diag_fd(delta=min_delta, debug=False)
print(f"{'atomidx':>7} {'atom':>6} {'gamma':>6} {'qidx':>6} {'vb':>{colw}} {'cb':>{colw}}")
for atomidx in range(system.getNAtoms()):
    for gamma, g_label in enumerate(["x", "y", "z"]):
        vb_val = auto_min_dict[(atomidx, gamma, qidx_gamma, "vb")]
        cb_val = auto_min_dict[(atomidx, gamma, qidx_gamma, "cb")]
        print(f"{atomidx:>7} {system.atomTypes[atomidx]:>6} {g_label:>6} {qidx_gamma:>6} {fmt(vb_val)} {fmt(cb_val)}")

# now testing the ham.calcCoupling() function directly
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

print("(atomidx, gamma, qidx, vb/cb).    |cpl| eV/Bohr")
for key, value in cpl_dict.items():
    val = float(value.real)
    if abs(val) < 1e-12:
        val_str = f"{0.0:.5e} (|val|<1e-12)"
    else:
        val_str = f"{val:.5e}"
    print(f"{key}.   |cpl|: {val_str}")









#######################################################
# now add in arbitrary SOC potential and repeat
if RUN_SOC:
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
    ham1 = Hamiltonian(system, PPparams, atomPPorder, device, NNConfig=NNConfig, iSystem=0, SObool=True, coupling=True)
    h = ham1.buildHtot(ham1.idx_gap)
    h = h.numpy(force=True)
    vals, vecs = scipy.linalg.eigh(h, subset_by_index=[0,32], driver='evr')
    #vb_vec = vecs[:,7]
    vb_vec = vecs[:, 15]
    #cb_vec = vecs[:,8]
    cb_vec = vecs[:, 16]
    e1s = [vals[15], vals[16]]
    print(f"\n\nInitial energies VBM: {e1s[0]:.6f}, CBM: {e1s[1]:.6f}")
    print(f"vb-1 degen? {abs(vals[15] - vals[14]) < 1e-10}, {abs(vals[15] - vals[14]):.3e}")
    print(f"vb-2 degen? {abs(vals[15] - vals[13]) < 1e-10}, {abs(vals[15] - vals[13]):.3e}")
    print(f"cb+1 degen? {abs(vals[16] - vals[17]) < 1e-10}, {abs(vals[16] - vals[17]):.3e}")
    print(f"cb+2 degen? {abs(vals[16] - vals[18]) < 1e-10}, {abs(vals[16] - vals[18]):.3e}")



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
    base_cd_pos = system.atomPos[0].clone()
    base_scale = float(system.scale)
    fd_results = []
    for sysid in range(7):
        system_dx = BulkSystem()
        system_dx.setSystem(f"{pwd}/inputs/couple/system_dx{sysid}.par")
        system_dx.setInputs(f"{pwd}/inputs/couple/input_0.par")
        system_dx.setKPointsAndWeights(f"{pwd}/inputs/couple/kpoints_0.par")
        system_dx.setQPointsAndWeights(f"{pwd}/inputs/couple/qpoints_0.par")
        system_dx.setExpBS(f"{pwd}/inputs/couple/expBandStruct_0.par")
        print("\nmanual system d/dy positions:")
        for i in range(2):
            print(f"{system_dx.atomTypes[i]}: {system_dx.atomPos[i]}")

        dy = float((system_dx.atomPos[0][1] - base_cd_pos[1]) * base_scale)
        ham_dx = Hamiltonian(system_dx, PPparams, atomPPorder, device, NNConfig=NNConfig, iSystem=0, SObool=True, coupling=False)
        hdx = ham_dx.buildHtot(ham1.idx_gap)
        hdx = hdx.numpy(force=True)
        vals_dx, vecs_dx = scipy.linalg.eigh(hdx, subset_by_index=[0,32], driver='evr')
        #e1s_dx = [vals_dx[7], vals_dx[8]]
        e1s_dx = [vals_dx[15], vals_dx[16]]

        fd_vb = (e1s_dx[0]-e1s[0]) / dy * AUTOEV
        fd_cb = (e1s_dx[1]-e1s[1]) / dy * AUTOEV
        fd_results.append((dy, fd_vb, fd_cb))
        print(f"dy = {dy:.8e}")
        print(f"dE/dCd_y by finite diff. vbm: {fd_vb} eV/Bohr,  cbm: {fd_cb} eV/Bohr")

    print("\nFinite-difference convergence summary (Cd, d/dy, eV/Bohr):")
    fmt = lambda v: f"{v:>{colw}.5e}"
    auto_fd_cache = {}
    print(
        f"{'dy':>{colw}} {'fd_vb':>{colw}} {'fd_cb':>{colw}} "
        f"{'fd_vb-an':>{colw}} {'fd_cb-an':>{colw}} "
        f"{'ham_fd_auto_vb':>{colw}} {'ham_fd_auto_cb':>{colw}} "
        f"{'ham_fd_auto_vb_cen':>{colw}} {'ham_fd_auto_cb_cen':>{colw}}"
    )
    analytic_vb_cd_y = np.dot(np.conj(vb_vec), np.dot(dV_dict[(0,1)], vb_vec)).real * AUTOEV
    analytic_cb_cd_y = np.dot(np.conj(cb_vec), np.dot(dV_dict[(0,1)], cb_vec)).real * AUTOEV
    print("\n\nham-coded finite difference for debugging (calcCouplings_diag_fd, eV/Bohr)")
    auto_fd_results_soc = {}
    print(
        f"{'dy':>{colw}} {'ham_fd_auto_vb':>{colw}} {'ham_fd_auto_cb':>{colw}} "
        f"{'ham_fd_auto_vb_cen':>{colw}} {'ham_fd_auto_cb_cen':>{colw}}"
    )
    for dy, _, _ in fd_results:
        auto_dict_one = get_auto_fd(abs(dy))
        auto_vb_one = extract_auto(auto_dict_one, 0, 1, 'vb')
        auto_cb_one = extract_auto(auto_dict_one, 0, 1, 'cb')
        auto_dict_cen = ham1.calcCouplings_diag_fd(delta=abs(dy), debug=False, one_sided=False)
        auto_vb_cen = extract_auto(auto_dict_cen, 0, 1, 'vb')
        auto_cb_cen = extract_auto(auto_dict_cen, 0, 1, 'cb')
        auto_fd_results_soc[dy] = (auto_vb_one, auto_cb_one, auto_vb_cen, auto_cb_cen)
        print(f"{fmt(dy)} {fmt(auto_vb_one)} {fmt(auto_cb_one)} {fmt(auto_vb_cen)} {fmt(auto_cb_cen)}")
    for dy, fd_vb, fd_cb in fd_results:
        auto_vb, auto_cb, auto_vb_cen, auto_cb_cen = auto_fd_results_soc[dy]
        print(
            f"{fmt(dy)} {fmt(fd_vb)} {fmt(fd_cb)} "
            f"{fmt(fd_vb-analytic_vb_cd_y)} {fmt(fd_cb-analytic_cb_cd_y)} "
            f"{fmt(auto_vb)} {fmt(auto_cb)} {fmt(auto_vb_cen)} {fmt(auto_cb_cen)}"
        )
