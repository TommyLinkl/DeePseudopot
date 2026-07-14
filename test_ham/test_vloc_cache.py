"""
Test: caching the k-independent local potential Vloc ONCE per epoch and reusing
the same grad-carrying tensor across all k-points reproduces -- band structure
AND model gradients -- the legacy behavior of rebuilding Vloc at every k-point.

Two backward patterns are checked, matching the two in-process training paths:

  1. naive (separateKptGrad==0): a SINGLE backward over the whole band-structure
     loss. The shared Vloc subgraph is traversed once; autograd accumulates the
     model gradient correctly.  -> exercised via calcBandStruct_withGrad.

  2. separateKptGrad serial (num_cores==0): a SEPARATE backward per k-point with
     per-k weights. The shared Vloc subgraph must survive across those backwards,
     so retain_graph=True is used for every k except the last.

In both cases the cached result must equal the per-k-rebuild reference to
round-off (only the Vloc build COUNT changes, not the math).

Run from the repo root:
    python -m test_ham.test_vloc_cache
"""
import pathlib
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

from utils.ham import Hamiltonian
from utils.read import BulkSystem, read_PPparams, read_NNConfigFile
from utils.nn_models import Net_sig

device = torch.device("cpu")
pwd = pathlib.Path(__file__).parent.resolve()


def build_system():
    system = BulkSystem()
    system.setSystem(f"{pwd}/inputs/soc/system_0.par")
    system.setInputs(f"{pwd}/inputs/soc/input_0.par")
    system.maxKE = 3.0
    system.setKPointsAndWeights(f"{pwd}/inputs/soc/kpoints_0.par")
    system.setExpBS(f"{pwd}/inputs/soc/bandStruct_0.dat")
    return system


def make_ham():
    system = build_system()
    atomPPorder = np.unique(system.atomTypes)
    PPparams, _ = read_PPparams(atomPPorder, f"{pwd}/inputs/soc/")
    NNConfig = read_NNConfigFile(f"{pwd}/inputs/NN_config.par", str(pwd) + "/")
    NNConfig['num_cores'] = 0
    NNConfig['local_env_corr'] = False
    NNConfig['checkpoint'] = 0
    # SObool=False: Vloc caching is orthogonal to the SO/NL terms (those are
    # rebuilt per k regardless); a non-spinor ham isolates the Vloc grad path.
    ham = Hamiltonian(system, PPparams, atomPPorder, device, NNConfig=NNConfig,
                      iSystem=0, SObool=False, cacheSO=False)
    ham.NN_locbool = True
    return ham, atomPPorder


def model_grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def naive_reference(ham, model):
    """Rebuild Vloc per k, ONE summed backward (legacy naive behavior)."""
    model.zero_grad(set_to_none=True)
    rows = [ham.calcEigValsAtK(k, None, requires_grad=True)
            for k in range(ham.system.getNKpts())]
    bs = torch.stack(rows)
    (bs ** 2).sum().backward()
    return bs.detach(), model_grads(model)


def naive_cached(ham, model):
    """calcBandStruct_withGrad builds Vloc once and reuses it; ONE backward."""
    model.zero_grad(set_to_none=True)
    bs = ham.calcBandStruct_withGrad(None)
    (bs ** 2).sum().backward()
    return bs.detach(), model_grads(model)


def sepkpt_reference(ham, model, w):
    """Per-k backward, rebuild Vloc each k, accumulate weighted grads (legacy)."""
    grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    rows = []
    for k in range(ham.system.getNKpts()):
        e = ham.calcEigValsAtK(k, None, requires_grad=True)
        model.zero_grad(set_to_none=True)
        (e ** 2).sum().backward()
        for n, p in model.named_parameters():
            grads[n] += p.grad.detach().clone() * w[k]
        rows.append(e.detach().clone())
    return torch.stack(rows), grads


def sepkpt_cached(ham, model, w):
    """Per-k backward, shared Vloc + retain_graph (new separateKptGrad path)."""
    grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    rows = []
    nk = ham.system.getNKpts()
    precomp = ham.buildVlocMat()
    for k in range(nk):
        e = ham.calcEigValsAtK(k, None, requires_grad=True, precomp_Vloc=precomp)
        model.zero_grad(set_to_none=True)
        (e ** 2).sum().backward(retain_graph=(k < nk - 1))
        for n, p in model.named_parameters():
            grads[n] += p.grad.detach().clone() * w[k]
        rows.append(e.detach().clone())
    return torch.stack(rows), grads


all_ok = True
ATOL = 1e-10


def check(label, A, B):
    global all_ok
    ok = torch.allclose(A, B, rtol=0.0, atol=ATOL)
    print(f"  [{'OK' if ok else 'FAIL'}] {label:38s} max|diff|={(A - B).abs().max().item():.3e}")
    all_ok = all_ok and ok


def check_grads(label, ga, gb):
    for n in ga:
        check(f"{label}: grad[{n}]", ga[n], gb[n])


ham, atomPPorder = make_ham()
n_types = len(atomPPorder)
nk = ham.system.getNKpts()
w = ham.system.kptWeights
print(f"system: {nk} k-points, {n_types} atom types, nbv={ham.basis.shape[0]}")

# Shared model so both reference and cached see identical parameters.
model = Net_sig([1, 8, n_types]).double()
ham.set_NNmodel(model)

print("\n=== naive path (single summed backward) ===")
bs_ref, g_ref = naive_reference(ham, model)
bs_new, g_new = naive_cached(ham, model)
check("band structure", bs_new, bs_ref)
check_grads("naive", g_new, g_ref)

print("\n=== separateKptGrad path (per-k backward, retain_graph) ===")
bs_ref, g_ref = sepkpt_reference(ham, model, w)
bs_new, g_new = sepkpt_cached(ham, model, w)
check("band structure", bs_new, bs_ref)
check_grads("sepkpt", g_new, g_ref)

print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
assert all_ok, "cached Vloc does not match per-k rebuild"
