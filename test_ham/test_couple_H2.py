import numpy as np
import scipy.linalg
import torch
import pathlib
import os

from utils.ham import Hamiltonian
from utils.read import BulkSystem, read_NNConfigFile, read_PPparams, setNN
from utils.init_NN_train import init_ZungerPP
from utils.constants import AUTOEV

# just test on cpu
device = torch.device("cpu")
torch.set_printoptions(precision=8)
torch.set_grad_enabled(False)

pwd = pathlib.Path(__file__).parent.resolve()
inputs_dir = pwd / "eph_diag_fd_test_inputs"

# read and set up system
system = BulkSystem()
system.setSystem(f"{inputs_dir}/system_0.par")
system.setInputs(f"{inputs_dir}/input_0.par")
system.setKPointsAndWeights(f"{inputs_dir}/kpoints_0.par")
system.setQPointsAndWeights(f"{inputs_dir}/qpoints_0.par")
system.setExpBS(f"{inputs_dir}/expBandStruct_0.par")
atomPPorder = np.unique(system.atomTypes)

print("initial atom positions:")
for i in range(system.getNAtoms()):
    print(f"{system.atomTypes[i]}: {system.atomPos[i]}")

PPparams, totalParams = read_PPparams(atomPPorder, f"{inputs_dir}/init_")
NNConfig = read_NNConfigFile(f"{inputs_dir}/NN_config.par")

nPseudopot = len(atomPPorder)
localPotParams = totalParams[:, :4]
model = setNN(NNConfig, nPseudopot)
results_dir = "eph_diag_fd_test_results"
os.makedirs(results_dir, exist_ok=True)
model, _ = init_ZungerPP(
    str(inputs_dir) + os.sep,
    model,
    atomPPorder,
    localPotParams,
    nPseudopot,
    NNConfig,
    device,
    str(results_dir) + os.sep,
    force_retrain=False,
)
model.eval()

ham1 = Hamiltonian(
    system,
    PPparams,
    atomPPorder,
    device,
    NNConfig=NNConfig,
    iSystem=0,
    SObool=NNConfig["SObool"],
    NN_locbool=True,
    coupling=True,
    model=model,
)

h = ham1.buildHtot(ham1.idx_gap)
h = h.numpy(force=True)
max_band = system.nBands - 1
vals, vecs = scipy.linalg.eigh(h, subset_by_index=[0, max_band], driver="evr")
# print(f"evals (in eV) = {vals*AUTOEV}")
vb_idx = 0
cb_idx = 1

vb_vec = vecs[:, vb_idx]
cb_vec = vecs[:, cb_idx]
e1s = [vals[vb_idx], vals[cb_idx]]
print(f"\n\nInitial energies VBM: {e1s[0] * AUTOEV:.6f} eV, CBM: {e1s[1] * AUTOEV:.6f} eV")
print("Test script assumes no degeneracy")

# compute analytic derivs of potential
get_derivs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
zero_vec = torch.zeros(3, dtype=system.qpts.dtype)
qidx_gamma = 0
for qid in range(system.getNQpts()):
    if torch.allclose(system.qpts[qid], zero_vec, atol=1e-12):
        qidx_gamma = qid
        break

dV_dict = ham1.buildCouplingMats(qidx_gamma, atomgammaidxs=get_derivs)

print("\n\nanalytic derivs (averaged over degenerate cb subspace) calculated within this test script (eV/Bohr):")

with torch.no_grad():
    base_vals = torch.linalg.eigvalsh(ham1.buildHtot(0, requires_grad=False))[:system.nBands]

def _fmt_real_if_tiny_imag(val, tol=1e-12):
    if np.iscomplexobj(val) and abs(val.imag) < tol:
        real_val = val.real
        if abs(real_val) < tol:
            return f"{0.0:.3e} (|Re|,|Im|<1e-12)"
        return f"{real_val:.3e} (Re; |Im|<1e-12)"
    return f"{val}"

for key in get_derivs:
    if key[1] == 0:
        d = "x"
    elif key[1] == 1:
        d = "y"
    else:
        d = "z"

    vb_vb_val = np.dot(np.conj(vb_vec), np.dot(dV_dict[key], vb_vec)) * AUTOEV
    cb_cb_val = np.dot(np.conj(cb_vec), np.dot(dV_dict[key], cb_vec)) * AUTOEV
    print(f"{system.atomTypes[key[0]]}, d/dR_{d}, vb-vb: {_fmt_real_if_tiny_imag(vb_vb_val)}")
    print(f"{system.atomTypes[key[0]]}, d/dR_{d}, cb-cb: {_fmt_real_if_tiny_imag(cb_cb_val)}")

print("\nanalytic derivs summary (H, d/dz, eV/Bohr):")
colw = 14
fmt = lambda v: f"{v:>{colw}.5e}"
print(
    f"{'dy':>{colw}} {'fd_vb':>{colw}} {'fd_cb':>{colw}} "
    f"{'an_vb':>{colw}} {'an_cb':>{colw}}"
)
analytic_vb_z = (np.dot(np.conj(vb_vec), np.dot(dV_dict[(0, 2)], vb_vec))).real * AUTOEV
analytic_cb_z = (np.dot(np.conj(cb_vec), np.dot(dV_dict[(0, 2)], cb_vec))).real * AUTOEV
print(
    f"{'/':>{colw}} {'/':>{colw}} {'/':>{colw}} "
    f"{fmt(analytic_vb_z)} {fmt(analytic_cb_z)}"
)

# compute dE/dz by loading a slightly deformed system along z
print("\n\nFinite diff, using manually set system.par, calculated within this test script. d/dz, for H...")
base_pos = system.atomPos[0].clone()
fd_results = []
fd_deltas = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
for delta in fd_deltas:
    system_dx = BulkSystem()
    system_dx.setSystem(f"{inputs_dir}/system_0.par")
    system_dx.setInputs(f"{inputs_dir}/input_0.par")
    system_dx.setKPointsAndWeights(f"{inputs_dir}/kpoints_0.par")
    system_dx.setQPointsAndWeights(f"{inputs_dir}/qpoints_0.par")
    system_dx.setExpBS(f"{inputs_dir}/expBandStruct_0.par")

    system_dx.atomPos[0, 2] += delta
    dz = float(system_dx.atomPos[0, 2] - base_pos[2])

    ham_dx = Hamiltonian(
        system_dx,
        PPparams,
        atomPPorder,
        device,
        NNConfig=NNConfig,
        iSystem=0,
        SObool=NNConfig["SObool"],
        coupling=False,
        NN_locbool=True,
        model=model,
    )
    hdx = ham_dx.buildHtot(ham1.idx_gap)
    hdx = hdx.numpy(force=True)
    vals_dx, _ = scipy.linalg.eigh(hdx, subset_by_index=[0, max_band], driver="evr")
    e1s_dx = [vals_dx[vb_idx], vals_dx[cb_idx]]

    fd_vb = (e1s_dx[0] - e1s[0]) / dz * AUTOEV
    fd_cb = (e1s_dx[1] - e1s[1]) / dz * AUTOEV
    fd_results.append((dz, fd_vb, fd_cb))
    print(f"H dz = {dz:.3e} Bohr")
    print(f"dE/dH_z by finite diff for vbm: {fd_vb:.3e} eV/Bohr,  cbm: {fd_cb:.3e} eV/Bohr")

print("\nFinite-difference convergence (via system.par) summary (H0, d/dz, eV/Bohr):")
fmt = lambda v: f"{v:>{colw}.5e}"
print(
    f"{'dy':>{colw}} {'fd_vb':>{colw}} {'fd_cb':>{colw}} "
    f"{'an_vb':>{colw}} {'an_cb':>{colw}}"
)
for dz, fd_vb, fd_cb in fd_results:
    print(
        f"{fmt(dz)} {fmt(fd_vb)} {fmt(fd_cb)} "
        f"{fmt(analytic_vb_z)} {fmt(analytic_cb_z)}"
    )


print("\n\nham-coded finite difference for debugging (calcCouplings_diag_fd, H0, d/dz, eV/Bohr)")
def extract_auto(auto_dict, atomidx, gamma, band):
    for key, value in auto_dict.items():
        if key[0] == atomidx and key[1] == gamma and key[3] == band:
            return float(value)
    raise KeyError(f"Missing auto fd key for atom {atomidx}, gamma {gamma}, band {band}")

auto_fd_results = {}
print(
    f"{'dz':>{colw}} {'fd_vb':>{colw}} {'fd_cb':>{colw}} "
    f"{'ham_fd_auto_vb_cen':>{colw}} {'ham_fd_auto_cb_cen':>{colw}}"
)
for dz, fd_vb, fd_cb in fd_results:
    auto_dict_cen = ham1.calcCouplings_diag_fd(
        delta=abs(dz),
        debug=False,
        one_sided=False,
        select_gamma=2,
        select_atomidx=0, 
        base_vals=base_vals,
    )
    auto_vb_cen = extract_auto(auto_dict_cen, 0, 2, "vb")
    auto_cb_cen = extract_auto(auto_dict_cen, 0, 2, "cb")
    auto_fd_results[dz] = (auto_vb_cen, auto_cb_cen)
    print(f"{fmt(dz)} {fmt(fd_vb)} {fmt(fd_cb)} {fmt(auto_vb_cen)} {fmt(auto_cb_cen)}")


# now testing the calcCoupling function directly
print("\n\n\n\n---------------------------------------------")
print("now redoing the same analysis using calcCouplings() function")
print("(this prints the magnitude of the coupling)\n")

_ = ham1.calcBandStruct()
cpl_dict = ham1.calcCouplings(qlist=[qidx_gamma], atomgammaidxs=get_derivs)

print("(atomidx, gamma, qidx, vb/cb).    |cpl|  eV/Bohr")
for key, value in cpl_dict.items():
    val = float(value.real)
    if abs(val) < 1e-12:
        val_str = f"{0.0:.5e} (|val|<1e-12)"
    else:
        val_str = f"{val:.5e}"
    print(f"{key}.   |cpl|: {val_str}")
