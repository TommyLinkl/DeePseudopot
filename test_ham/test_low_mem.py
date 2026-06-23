"""
Test: low_mem (cache SO/NL matrices grouped by atom TYPE) is numerically
IDENTICAL to the per-atom caching, for both band structures and NL/SOC
gradients.

The CsPbI3 system has atoms [Pb, I, I, I, Cs] -> 5 atoms but 3 types, so low_mem
collapses the 3 I matrices into one slot (nMatGroups 5 -> 3). Because the H sum
is associative and same-type atoms share a prefactor, the result must be bit-for
-bit identical.

Run from the repo root:
    python -m test_ham.test_low_mem
"""
import pathlib
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from utils.ham import Hamiltonian
from utils.read import BulkSystem, read_PPparams, read_NNConfigFile
from utils.NN_train import setup_nonlocal_grad

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


def make_ham(low_mem, cacheSO):
    system = build_system()
    atomPPorder = np.unique(system.atomTypes)
    PPparams, _ = read_PPparams(atomPPorder, f"{pwd}/inputs/soc/")
    NNConfig = read_NNConfigFile(f"{pwd}/inputs/NN_config.par")
    NNConfig['low_mem'] = low_mem
    NNConfig['num_cores'] = 0
    ham = Hamiltonian(system, PPparams, atomPPorder, device, NNConfig=NNConfig,
                      iSystem=0, SObool=True, cacheSO=cacheSO)
    return ham, atomPPorder, NNConfig


all_ok = True

# ---- 1) band structure: low_mem vs normal, cached path --------------------
ham_n, _, _ = make_ham(low_mem=False, cacheSO=True)
ham_l, _, _ = make_ham(low_mem=True,  cacheSO=True)
print(f"nMatGroups: normal={ham_n.nMatGroups} (per atom), low_mem={ham_l.nMatGroups} (per type)")
assert ham_n.nMatGroups == 5 and ham_l.nMatGroups == 3, "unexpected group counts"

bs_n = ham_n.calcBandStruct().detach()
bs_l = ham_l.calcBandStruct().detach()
max_bs_diff = (bs_n - bs_l).abs().max().item()
bs_ok = torch.allclose(bs_n, bs_l, atol=1e-10, rtol=0)
all_ok = all_ok and bs_ok
print(f"[cached]   band-structure max|Δ| = {max_bs_diff:.2e}  [{'OK' if bs_ok else 'FAIL'}]")

# ---- 2) band structure: on-the-fly path (cacheSO=False) -------------------
ham_n2, _, _ = make_ham(low_mem=False, cacheSO=False)
ham_l2, _, _ = make_ham(low_mem=True,  cacheSO=False)
bs_n2 = ham_n2.calcBandStruct().detach()
bs_l2 = ham_l2.calcBandStruct().detach()
max_bs_diff2 = (bs_n2 - bs_l2).abs().max().item()
bs_ok2 = torch.allclose(bs_n2, bs_l2, atol=1e-10, rtol=0)
all_ok = all_ok and bs_ok2
print(f"[on-the-fly] band-structure max|Δ| = {max_bs_diff2:.2e}  [{'OK' if bs_ok2 else 'FAIL'}]")

# ---- 3) NL/SOC gradients: low_mem vs normal -------------------------------
def nl_grads(low_mem):
    ham, atomPPorder, NNConfig = make_ham(low_mem=low_mem, cacheSO=True)
    NNConfig['nonlocal_grad'] = True
    NNConfig['nonlocal_grad_indices'] = [5, 6, 7]
    NNConfig['optimizer_lr'] = 1e-3
    nl_ctx = setup_nonlocal_grad([ham], atomPPorder, NNConfig)
    e = ham.calcEigValsAtK(0, cachedMats_info=None, requires_grad=True)
    loss = (e[:16] ** 2).sum()
    loss.backward()
    return {atom: ham.PPparams[atom].grad.detach().clone() for atom in nl_ctx['params']}

g_n = nl_grads(False)
g_l = nl_grads(True)
print("NL/SOC gradient comparison (normal vs low_mem):")
for atom in g_n:
    for i in [5, 6, 7]:
        d = abs(float(g_n[atom][i]) - float(g_l[atom][i]))
        ok = d < 1e-9
        all_ok = all_ok and ok
        if abs(float(g_n[atom][i])) > 1e-12 or abs(float(g_l[atom][i])) > 1e-12:
            print(f"  {atom}[{i}]: normal={float(g_n[atom][i]):+.6e}  low_mem={float(g_l[atom][i]):+.6e}  |Δ|={d:.2e}  [{'OK' if ok else 'FAIL'}]")

print("\n=========================================")
print(f"low_mem == normal (BS + NL grads): {'PASS' if all_ok else 'FAIL'}")
print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
print("=========================================")
