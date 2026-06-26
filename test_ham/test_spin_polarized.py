"""
Tests for the spin-polarized (spin-unrestricted) local pseudopotential.

V_up = V0 + b, V_down = V0 - b, where V0 is the existing local-potential model
and b is the learned spin field (self.spinModel). Toggled by tot_magnetization.

Conventions (important for reading the checks below):
  - The UNPOLARIZED path (magBool off, no SOC) builds the nbv x nbv block and
    returns each DISTINCT spatial band once: [e0, e1, e2, ...]. Spin degeneracy
    is implicit, so the spectrum is NOT doubled.
  - The SPIN-POLARIZED path (magBool on) builds the 2*nbv block-diagonal H and
    returns the spin-RESOLVED spectrum, interleaved [up0, dn0, up1, dn1, ...].
    At b == 0 the channels are degenerate, so this is exactly the unpolarized
    spectrum spin-doubled: [e0, e0, e1, e1, ...]. These two conventions are both
    correct (the SObool path uses the same spin-resolved convention); they must
    be compared like-for-like, i.e. by spin-doubling the unpolarized spectrum.

Checks:
  1. Zero-init invariant: with b == 0, the spin-polarized (2*nbv) band structure
     equals the unpolarized band structure SPIN-DOUBLED, to ~machine precision.
     Proves the sizing/doubling unification + zero-init.
  2. b -> -b symmetry: a nonzero b splits the bands, and flipping the sign of b
     (i.e. swapping which spin channel is up) leaves the physical SPECTRUM (the
     sorted set of eigenvalues) IDENTICAL. The per-channel interleaving swaps
     up<->dn within each pair, so the comparison is on the sorted spectrum.
     Proves the up/down block assignment + symmetric splitting.
  3. Gradient flow: the spin field receives nonzero gradients from a band loss.

Run from the DeePseudopot repo root:  PYTHONPATH=. python test_ham/test_spin_polarized.py
"""
import copy
import pathlib
import numpy as np
import torch

from utils.nn_models import Net_relu_xavier_decay2, zero_init_final_layer
from utils.ham import Hamiltonian
from utils.read import BulkSystem, read_NNConfigFile

torch.set_default_dtype(torch.float64)
device = torch.device("cpu")
pwd = pathlib.Path(__file__).parent.resolve()

# --- system + config + populated PPparams (needed for long_range_correction) ---
system = BulkSystem()
system.setSystem(f"{pwd}/inputs/system_0.par")
system.setInputs(f"{pwd}/inputs/input_0.par")
system.setKPointsAndWeights(f"{pwd}/inputs/kpoints_0.par")
system.setExpBS(f"{pwd}/inputs/expBandStruct_0.par")
atomPPorder = np.unique(system.atomTypes)

# Cap to a few k-points for a fast test (the spin invariants are per-k-point).
NKEEP = 4
system.kpts          = system.kpts[:NKEEP]
system.kptWeights    = system.kptWeights[:NKEEP]
system.bandOrderMatrix = system.bandOrderMatrix[:NKEEP]
system.expBandStruct = system.expBandStruct[:NKEEP]

NNConfig = read_NNConfigFile(f"{pwd}/inputs/NN_config.par")
NNConfig['separateKptGrad'] = False   # use the simple per-system grad path for calcBandStruct

PPparams = {}
for atomType in atomPPorder:
    with open(f"{pwd}/inputs/init_{atomType}Params.par", 'r') as f:
        a = torch.tensor([float(line.strip()) for line in f])
    # The Hamiltonian indexes PPparams[atom][5] (SOC) and [6],[7] (nonlocal).
    # The test init files only hold the 5 local-potential params (indices 0-4),
    # so pad SOC/NL with zeros (no SOC, no nonlocal) for this local-potential test.
    if a.shape[0] < 8:
        a = torch.cat([a, torch.zeros(8 - a.shape[0], dtype=a.dtype)])
    PPparams[atomType] = a

nPP = len(atomPPorder)
PPmodel = Net_relu_xavier_decay2([1, 20, 20, 20, nPP])
PPmodel.load_state_dict(torch.load(f"{pwd}/epoch_199_PPmodel.pth", map_location=device))


def make_spin_model(bias_const=None):
    m = Net_relu_xavier_decay2([1, 20, 20, 20, nPP])
    zero_init_final_layer(m)            # b == 0
    if bias_const is not None:
        # all weights are zero after zero-init, so setting the final-layer bias to
        # a constant makes b(q) = bias_const * decay(q): a nonzero, q-decaying field.
        with torch.no_grad():
            m.neural_network.hidden_l[-1].bias.fill_(bias_const)
    return m


def cfg_with_mag(mag):
    c = copy.deepcopy(NNConfig)
    c['tot_magnetization'] = mag
    return c


def build_ham(NNcfg, spinModel=None, SObool=False):
    return Hamiltonian(system, PPparams, atomPPorder, device, NNConfig=NNcfg,
                       iSystem=0, SObool=SObool, NN_locbool=True, model=PPmodel,
                       spinModel=spinModel)


# ----------------------------------------------------------------------------
# Test 1: zero-init invariant (spin-polarized with b==0 == unpolarized, doubled)
# ----------------------------------------------------------------------------
bs_plain = build_ham(cfg_with_mag(0.0)).calcBandStruct().detach()
bs_mag0  = build_ham(cfg_with_mag(1.0), spinModel=make_spin_model()).calcBandStruct().detach()
# The unpolarized spectrum lists each band once; spin-double it to match the
# spin-resolved [up0,dn0,up1,dn1,...] convention of the polarized path, then
# trim to the same nBands.
bs_plain_doubled = bs_plain.repeat_interleave(2, dim=1)[:, :bs_mag0.shape[1]]
ok1 = torch.allclose(bs_plain_doubled, bs_mag0, atol=1e-10)
print(f"[1] zero-init invariant (mag,b=0 == unpolarized spin-doubled): {ok1}  "
      f"max|diff|={(bs_plain_doubled-bs_mag0).abs().max().item():.2e}")

# ----------------------------------------------------------------------------
# Test 2: a nonzero b splits the bands, and b -> -b gives the same spectrum
# ----------------------------------------------------------------------------
bs_plus  = build_ham(cfg_with_mag(1.0), spinModel=make_spin_model(+0.05)).calcBandStruct().detach()
bs_minus = build_ham(cfg_with_mag(1.0), spinModel=make_spin_model(-0.05)).calcBandStruct().detach()
# Splitting: compare against the b==0 polarized spectrum (same convention), so
# the difference isolates the b-induced spin splitting, not a convention change.
split_happened = not torch.allclose(bs_plus, bs_mag0, atol=1e-6)
# b -> -b swaps the up/dn labels within each interleaved pair; the physical
# spectrum is invariant, so compare the sorted eigenvalues per k-point.
symmetric = torch.allclose(bs_plus.sort(dim=1).values,
                           bs_minus.sort(dim=1).values, atol=1e-10)
print(f"[2a] nonzero b splits the bands (differs from b=0 polarized): {split_happened}  "
      f"max|split|={(bs_plus-bs_mag0).abs().max().item():.2e}")
print(f"[2b] b -> -b symmetry (sorted spectrum invariant): {symmetric}  "
      f"max|diff|={(bs_plus.sort(dim=1).values-bs_minus.sort(dim=1).values).abs().max().item():.2e}")

# ----------------------------------------------------------------------------
# Test 3: gradient flow into the spin field from a band-structure loss
# ----------------------------------------------------------------------------
spin_grad_model = make_spin_model(0.02)
spin_grad_model.train()
ham_g = build_ham(cfg_with_mag(1.0), spinModel=spin_grad_model)
bs = ham_g.calcBandStruct(grad=True)
loss = (bs ** 2).sum()
loss.backward()
grads = [p.grad for p in spin_grad_model.parameters() if p.grad is not None]
total_grad = sum(g.abs().sum().item() for g in grads)
ok3 = total_grad > 0
print(f"[3] spin field receives nonzero gradients from band loss: {ok3}  "
      f"sum|grad|={total_grad:.3e}")

# ----------------------------------------------------------------------------
print()
allok = ok1 and split_happened and symmetric and ok3
print(f"ALL SPIN-POLARIZED TESTS PASSED: {allok}")
