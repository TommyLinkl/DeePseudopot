import numpy as np
import scipy.linalg
import torch
import pathlib
import os

from utils.ham import Hamiltonian
from utils.read import BulkSystem, read_NNConfigFile, read_PPparams, setNN
from utils.init_NN_train import init_ZungerPP
from utils.constants import AUTOEV

print(
    "Testing epc calculations on H2 with analytic vs finite-difference derivatives.\n"
    "Assumes no electronic degeneracy and Gamma-only k grids & q grids.\n"
    "Sections:\n"
    "1) Analytic dV/dR using eigenvectors from this script\n"
    "2) One-sided finite difference via ham.calcCouplings_diag_fd()\n"
    "3) calcCouplings() magnitude output\n"
)

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
print(f"All eigenvalues (eV): {[v * AUTOEV for v in vals]}") # I only want to print .6f precision here
print("Test script assumes no degeneracy! ")

# compute analytic derivs of potential
get_derivs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
zero_vec = torch.zeros(3, dtype=system.qpts.dtype)
qidx_gamma = 0
for qid in range(system.getNQpts()):
    if torch.allclose(system.qpts[qid], zero_vec, atol=1e-12):
        qidx_gamma = qid
        break

dV_dict = ham1.buildCouplingMats(qidx_gamma, atomgammaidxs=get_derivs)

########################################################################
print("\n\nAnalytic epc calculated from e-vecs extracted within this test script (eV/Bohr):")

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

print("\nAnalytic summary (H, d/dz, eV/Bohr):")
colw = 17
fmt = lambda v: f"{v:>{colw}.7e}"
print(
    f"{'dy':>{colw}} {'fd_vb':>{colw}} {'fd_cb':>{colw}} "
    f"{'|an_vb|':>{colw}} {'|an_cb|':>{colw}}"
)
analytic_vb_z = (np.dot(np.conj(vb_vec), np.dot(dV_dict[(0, 2)], vb_vec))).real * AUTOEV
analytic_cb_z = (np.dot(np.conj(cb_vec), np.dot(dV_dict[(0, 2)], cb_vec))).real * AUTOEV
analytic_vb_z_mag = abs(analytic_vb_z)
analytic_cb_z_mag = abs(analytic_cb_z)
print(
    f"{'/':>{colw}} {'/':>{colw}} {'/':>{colw}} "
    f"{fmt(analytic_vb_z_mag)} {fmt(analytic_cb_z_mag)}"
)

########################################################################
print("\n\nFinite difference by calling calcCouplings_diag_fd() (focusing only on H0, |d/dz|, eV/Bohr)")
def extract_auto(auto_dict, atomidx, gamma, band):
    for key, value in auto_dict.items():
        if key[0] == atomidx and key[1] == gamma and key[3] == band:
            return float(value)
    raise KeyError(f"Missing auto fd key for atom {atomidx}, gamma {gamma}, band {band}")

auto_fd_results = {}
fd_deltas = [1e-4, 1e-5, 1e-6, 1e-7]
print(
    f"{'dz':>{colw}} {'ham_fd_auto_vb':>{colw}} {'ham_fd_auto_cb':>{colw}}"
)
for dz in fd_deltas:
    auto_dict_cen = ham1.calcCouplings_diag_fd(
        delta=abs(dz),
        debug=False,
        select_gamma=2,
        select_atomidx=0, 
        base_vals=base_vals,
    )
    auto_vb_cen = extract_auto(auto_dict_cen, 0, 2, "vb")
    auto_cb_cen = extract_auto(auto_dict_cen, 0, 2, "cb")
    auto_fd_results[dz] = (auto_vb_cen, auto_cb_cen)
    print(f"{fmt(dz)} {fmt(auto_vb_cen)} {fmt(auto_cb_cen)}")


########################################################################
print("\n\nAnalytical epc calculated using calcCouplings() function (eV/Bohr):")

avg_dirs = {}   # use this dict to tell the calcCouplings functions which
                # atoms and directions to average over due to the degenerate CB
                # manifold. 
avg_dirs[0] = ('x','y','z')
avg_dirs[1] = ('x','y','z')

_ = ham1.calcBandStruct()
cpl_dict = ham1.calcCouplings(qlist=[qidx_gamma], atomgammaidxs=get_derivs)
cpl_dict_avg_wrong = ham1.calcCouplings(qlist=[qidx_gamma], atomgammaidxs=get_derivs, symm_equiv=avg_dirs)

def _label_atom(idx):
    if idx == 0:
        return "H0"
    if idx == 1:
        return "H1"
    return f"A{idx}"

def _label_gamma(idx):
    if idx == 0:
        return "x"
    if idx == 1:
        return "y"
    if idx == 2:
        return "z"
    return f"d{idx}"

def _fmt_cpl(val):
    if val is None:
        return "n/a"
    val = float(val.real)
    if abs(val) < 1e-12:
        return f"{0.0:.5e}"
    return f"{val:.5e}"

print("calcCouplings comparison (subset vs full). |cpl| in eV/Bohr")
print(f"{'atom':>4} {'dir':>4} {'band':>4} {'calcCouplings()':>14} {'<x,y,z> averaged (wrong!)':>14}")
band_order = ["vb", "cb"]
for atomidx in range(system.getNAtoms()):
    for gamma in range(3):
        for band in band_order:
            key = (atomidx, gamma, qidx_gamma, band)
            val_subset = cpl_dict.get(key)
            val_full = cpl_dict_avg_wrong.get(key)
            print(
                f"{_label_atom(atomidx):>4} {_label_gamma(gamma):>4} {band:>4} "
                f"{_fmt_cpl(val_subset):>14} {_fmt_cpl(val_full):>14}"
            )
