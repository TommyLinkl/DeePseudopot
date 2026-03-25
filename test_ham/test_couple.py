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

print(
    "\n".join(
        [
            "Testing epc calculations on zbCdSe.",
            "Normal 0.01 meV criteria would label CBM/VBM as non-degenerate.",
            "Here we pick two states to act as 'CB' and 'VB' for a degeneracy test.",
            "These are NOT the true band edges of the material.",
            "Indices (0-based): 'VB' = 7; 'CB' = 8,9.",
            "Degeneracy: 'CB' is 2-fold; 'VB' is non-degenerate.",
            "",
            "Sections:",
            "  1) calcCouplings() with symm_equiv=avg_dirs",
            "  2) calcCouplings() without symm_equiv",
            "  3) Same as (2) after random unitary rotation of the CB subspace",
            "  4) Manual average CB wavefunction: cb = (v8+v9)/sqrt(2) (not invariant)",
            "  5) Same as (4) but with a random unitary before averaging (not invariant)",
            "  6) Finite-difference epc from calcCouplings_diag_fd()",
            "",
        ]
    )
)


# test on cpu
device = torch.device("cpu")
torch.set_printoptions(precision=8)

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
print(f"\n\nInitial energies VBM: {e1s[0]}, CBM: {e1s[1]}")
print(f"vb-1 degen? {abs(vals[7] - vals[6]) < 1e-15}, {abs(vals[7] - vals[6])}")
print(f"vb-2 degen? {abs(vals[7] - vals[5]) < 1e-15}, {abs(vals[7] - vals[5])}")
print(f"cb+1 degen? {abs(vals[8] - vals[9]) < 1e-15}, {abs(vals[8] - vals[9])}")
print(f"cb+2 degen? {abs(vals[8] - vals[10]) < 1e-15}, {abs(vals[8] - vals[10])}")




# compute analytic derivs of potential
get_derivs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
zero_vec = torch.zeros(3, dtype=system.qpts.dtype)
qidx_gamma = None
for qid in range(system.getNQpts()):
    if torch.allclose(system.qpts[qid], zero_vec, atol=1e-12):
        qidx_gamma = qid
        break
if qidx_gamma is None:
    raise ValueError("Gamma q-point not found in q-point list")

dV_dict = ham1.buildCouplingMats(qidx_gamma, atomgammaidxs=get_derivs)

def _label_atom(idx):
    if idx == 0:
        return "Cd"
    if idx == 1:
        return "Se"
    return f"A{idx}"

def _label_gamma(idx):
    if idx == 0:
        return "dx"
    if idx == 1:
        return "dy"
    if idx == 2:
        return "dz"
    return f"d{idx}"

def _fmt_cpl(val):
    if val is None:
        return "n/a"
    val = float(val.real)
    if abs(val) < 1e-12:
        return f"{0.0:.6e}"
    return f"{val:.6e}"

def _random_unitary_2x2(rng):
    z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    return q * ph

def _rotate_cb_vecs(ham, unitary):
    for kidx, vec_list in ham.cb_vecs.items():
        if len(vec_list) < 2:
            continue
        vec_stack = torch.stack(vec_list[:2], dim=-1)
        u = torch.tensor(unitary, dtype=vec_stack.dtype)
        rot = vec_stack @ u
        ham.cb_vecs[kidx] = [rot[:, 0], rot[:, 1]] + vec_list[2:]

def _manual_avg_dict(vb_vec_np, cb_vec_np):
    out = {}
    for atomidx in range(system.getNAtoms()):
        for gamma in range(3):
            vb_val = np.dot(np.conj(vb_vec_np), np.dot(dV_dict[(atomidx, gamma)], vb_vec_np)) * AUTOEV
            cb_val = np.dot(np.conj(cb_vec_np), np.dot(dV_dict[(atomidx, gamma)], cb_vec_np)) * AUTOEV
            out[(atomidx, gamma, qidx_gamma, "vb")] = abs(vb_val)
            out[(atomidx, gamma, qidx_gamma, "cb")] = abs(cb_val)
    return out

avg_dirs = {0: ("x", "y", "z"), 1: ("x", "y", "z")}

_ = ham1.calcBandStruct()

section1 = ham1.calcCouplings(qlist=[qidx_gamma], atomgammaidxs=get_derivs, symm_equiv=avg_dirs)
section2 = ham1.calcCouplings(qlist=[qidx_gamma], atomgammaidxs=get_derivs)

rng = np.random.default_rng(1234)
unitary = _random_unitary_2x2(rng)
cb_vecs_orig = copy.deepcopy(ham1.cb_vecs)
_rotate_cb_vecs(ham1, unitary)
section3 = ham1.calcCouplings(qlist=[qidx_gamma], atomgammaidxs=get_derivs)
ham1.cb_vecs = cb_vecs_orig

cb_vecs_np = vecs[:, [8, 9]]
cb_vec_avg = (cb_vecs_np[:, 0] + cb_vecs_np[:, 1]) / np.sqrt(2.0)
section4 = _manual_avg_dict(vb_vec, cb_vec_avg)

cb_vecs_rot = cb_vecs_np @ unitary
cb_vec_avg_rot = (cb_vecs_rot[:, 0] + cb_vecs_rot[:, 1]) / np.sqrt(2.0)
section5 = _manual_avg_dict(vb_vec, cb_vec_avg_rot)
system_fd = copy.deepcopy(system)
system_fd.kpts = torch.zeros((1, 3), dtype=system.kpts.dtype)
system_fd.kptWeights = torch.ones(1, dtype=system.kptWeights.dtype)
system_fd.qpts = torch.zeros((1, 3), dtype=system.qpts.dtype)
system_fd.qptWeights = torch.ones(1, dtype=system.qptWeights.dtype)
system_fd.bandOrderMatrix = np.arange(system_fd.nBands)[np.newaxis, :]

ham_fd = Hamiltonian(
    system_fd,
    PPparams,
    atomPPorder,
    device,
    NNConfig=NNConfig,
    iSystem=0,
    SObool=False,
    coupling=False,
)
with torch.no_grad():
    base_vals_fd = torch.linalg.eigvalsh(ham_fd.buildHtot(0, requires_grad=False))[:system_fd.nBands]

delta = 1e-6
fd_dict = ham_fd.calcCouplings_diag_fd(delta=delta, debug=False, base_vals=base_vals_fd)
section6 = {
    (atomidx, gamma, qidx_gamma, band): fd_dict[(atomidx, gamma, 0, band)]
    for atomidx in range(system_fd.getNAtoms())
    for gamma in range(3)
    for band in ["vb", "cb"]
}

section_descriptions = [
    "Section 1: calcCouplings() with symm_equiv=avg_dirs",
    "Section 2: calcCouplings() without symm_equiv",
    "Section 3: calcCouplings() after random unitary rotation (no symm_equiv)",
    "Section 4: Manual dV/dR using averaged degenerate CB wavefunctions",
    "Section 5: Manual dV/dR using averaged, unitary-rotated CB wavefunctions",
    f"Section 6: calcCouplings_diag_fd() one-sided finite differences at Gamma, delta={delta:.1e} Bohr",
]
print("\n\nSection list:")
for desc in section_descriptions:
    print(desc)
print("[Unit = eV/Bohr]")

sections = [section1, section2, section3, section4, section5, section6]
print(
    f"{'atom':>4} {'dir':>4} {'band':>4} "
    f"{'S1 |cpl|':>12} {'S2 |cpl|':>12} {'S3 |cpl|':>12} "
    f"{'S4 |cpl|':>12} {'S5 |cpl|':>12} {'S6 |cpl|':>12}"
)
for atomidx in range(system.getNAtoms()):
    for gamma in range(3):
        for band in ["vb", "cb"]:
            key = (atomidx, gamma, qidx_gamma, band)
            row_vals = [sec.get(key) for sec in sections]
            print(
                f"{_label_atom(atomidx):>4} {_label_gamma(gamma):>4} {band:>4} "
                f"{_fmt_cpl(row_vals[0]):>12} {_fmt_cpl(row_vals[1]):>12} {_fmt_cpl(row_vals[2]):>12} "
                f"{_fmt_cpl(row_vals[3]):>12} {_fmt_cpl(row_vals[4]):>12} {_fmt_cpl(row_vals[5]):>12}"
            )
