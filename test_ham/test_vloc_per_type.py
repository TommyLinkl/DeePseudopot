"""
Test: the per-TYPE local-potential build in Hamiltonian.buildVlocMat (group the
structure factors by atom type, one grad-tracked multiply per type) is
numerically identical -- forward AND gradient -- to the legacy per-ATOM build.

The factorization used is exact:
    sum_{alpha in t} atomFF_t * sfact_alpha = atomFF_t * (sum_{alpha in t} sfact_alpha)
because same-type atoms share the form factor atomFF_t and the H sum is
associative; only floating-point summation order differs, so results must match
to round-off.

CsPbI3 has atoms [Pb, I, I, I, Cs] -> 5 atoms but 3 types, so the three I atoms
(distinct positions, same type) genuinely exercise the structure-factor sum.

Run from the repo root:
    python -m test_ham.test_vloc_per_type
"""
import pathlib
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

from utils.ham import Hamiltonian
from utils.read import BulkSystem, read_PPparams, read_NNConfigFile
from utils.pp_func import pot_funcLR, long_range_correction
from utils.nn_models import Net_sig

device = torch.device("cpu")
pwd = pathlib.Path(__file__).parent.resolve()


def build_system():
    system = BulkSystem()
    system.setSystem(f"{pwd}/inputs/soc/system_0.par")
    system.setInputs(f"{pwd}/inputs/soc/input_0.par")
    system.maxKE = 3.0   # small basis for speed
    system.setKPointsAndWeights(f"{pwd}/inputs/soc/kpoints_0.par")
    system.setExpBS(f"{pwd}/inputs/soc/bandStruct_0.dat")
    return system


def make_ham(SObool):
    system = build_system()
    atomPPorder = np.unique(system.atomTypes)
    PPparams, _ = read_PPparams(atomPPorder, f"{pwd}/inputs/soc/")
    NNConfig = read_NNConfigFile(f"{pwd}/inputs/NN_config.par", str(pwd) + "/")
    NNConfig['num_cores'] = 0
    NNConfig['local_env_corr'] = False
    NNConfig['checkpoint'] = 0
    ham = Hamiltonian(system, PPparams, atomPPorder, device, NNConfig=NNConfig,
                      iSystem=0, SObool=SObool, cacheSO=False)
    return ham, atomPPorder


def reference_vloc(ham, addMat=None):
    """Legacy per-ATOM local potential build (the formulation before grouping)."""
    nbv = ham.basis.shape[0]
    gdiff = torch.stack([ham.basis] * nbv, dim=1) - ham.basis.repeat(nbv, 1, 1)
    q = torch.norm(gdiff, dim=2).view(-1, 1)

    if ham.NN_locbool:
        atomFF_full = ham.model(q)
    if ham.magBool:
        bff_full = ham.spinModel(q)

    if addMat is not None:
        Vmat = addMat
    elif ham.spinor:
        Vmat = torch.zeros([2 * nbv, 2 * nbv], dtype=torch.complex128)
    else:
        Vmat = torch.zeros([nbv, nbv])

    for alpha in range(ham.system.getNAtoms()):
        atomType = ham.system.atomTypes[alpha]
        gdiffDotTau = torch.sum(gdiff * ham.system.atomPos[alpha], axis=2)
        sfact_re = 1 / ham.system.getCellVolume() * torch.cos(gdiffDotTau)
        sfact_im = 1 / ham.system.getCellVolume() * torch.sin(gdiffDotTau)
        thisAtomIndex = np.where(atomType == ham.atomPPorder)[0][0]

        if ham.NN_locbool:
            atomFF = atomFF_full[:, thisAtomIndex].view(nbv, nbv)
            lr_coeff = ham.PPparams[atomType][4]
            atomFF = atomFF + long_range_correction(torch.norm(gdiff, dim=2), ham.LRgamma, lr_coeff)
        else:
            atomFF = pot_funcLR(torch.norm(gdiff, dim=2), ham.PPparams[atomType], ham.LRgamma)

        if ham.magBool:
            bff = bff_full[:, thisAtomIndex].view(nbv, nbv)
            atomFF_up = atomFF + bff
            atomFF_dn = atomFF - bff
        else:
            atomFF_up = atomFF
            atomFF_dn = atomFF

        if ham.spinor:
            sfact = torch.complex(sfact_re, sfact_im)
            Vmat[:nbv, :nbv] = Vmat[:nbv, :nbv] + atomFF_up * sfact
            Vmat[nbv:, nbv:] = Vmat[nbv:, nbv:] + atomFF_dn * sfact
        else:
            Vmat = Vmat + atomFF * torch.complex(sfact_re, sfact_im)

    return Vmat


def grads_of(loss, model):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return torch.cat([p.grad.reshape(-1).clone() for p in model.parameters()])


all_ok = True
RTOL, ATOL = 0.0, 1e-10


def check(label, A, B):
    global all_ok
    A = A.detach()
    B = B.detach()
    ok = torch.allclose(A, B, rtol=RTOL, atol=ATOL)
    maxdiff = (A - B).abs().max().item()
    print(f"  [{'OK' if ok else 'FAIL'}] {label:42s} max|diff|={maxdiff:.3e}")
    all_ok = all_ok and ok


for SObool in (False, True):
    spin_label = "spinor (SObool=True)" if SObool else "non-spinor"
    print(f"\n=== {spin_label} ===")
    ham, atomPPorder = make_ham(SObool)
    n_types = len(atomPPorder)

    # --- algebraic local potential (NN_locbool=False) ----------------------
    ham.NN_locbool = False
    V_ref = reference_vloc(ham)
    V_new = ham.buildVlocMat()
    check("algebraic forward", V_new, V_ref)

    # --- NN local potential (NN_locbool=True): forward + gradient ----------
    ham.NN_locbool = True
    model = Net_sig([1, 8, n_types]).double()
    ham.set_NNmodel(model)

    V_ref = reference_vloc(ham)
    V_new = ham.buildVlocMat()
    check("NN forward", V_new, V_ref)

    # gradient: a real scalar that depends on the full complex matrix
    g_ref = grads_of(reference_vloc(ham).abs().pow(2).sum(), model)
    g_new = grads_of(ham.buildVlocMat().abs().pow(2).sum(), model)
    check("NN gradient (model params)", g_new, g_ref)

    # --- addMat path (mirrors buildHtot adding onto the kinetic matrix) -----
    nbv = ham.basis.shape[0]
    dim = 2 * nbv if ham.spinor else nbv
    base = torch.randn(dim, dim, dtype=torch.complex128)
    V_ref = reference_vloc(ham, addMat=base.clone())
    V_new = ham.buildVlocMat(addMat=base.clone())
    check("NN forward (addMat)", V_new, V_ref)

print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
assert all_ok, "per-type Vloc build does not match per-atom reference"
