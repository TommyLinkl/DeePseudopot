#!/usr/bin/env python3
"""
test_mace_coupling_fd.py -- finite-difference check of the LSD descriptor
coupling term in ham.buildCouplingMats.

The local part of the e-ph coupling matrix is the R-derivative of

    V^lsd_{ij}(R) = sum_beta SF_beta[i,j](R) * (base(Q_ij) + Delta v(N_beta(R), Q_ij))

with SF_beta = (1/Omega) e^{+i Q_ij . R_beta}, Q_ij = G_i - (G_j + q). Because the
LSD correction is environment-dependent, displacing atom alpha changes Delta v of
EVERY atom whose descriptor depends on R_alpha -- so the coupling has a chain-rule
term  sum_beta SF_beta * dDelta v(N_beta,Q)/dR_{alpha,gamma}  beyond the usual
structure-factor derivative. buildCouplingMats builds that term with a forward-mode
JVP through the (now differentiable) descriptors; this script verifies it against a
central finite difference of V^lsd, for both descriptor backends.

It also checks that the coupling stays differentiable w.r.t. the LSD network
parameters (needed when fitting couplings).

Run (needs the mace env for backend='mace'):
    python test_mace_coupling_fd.py [mace|handcrafted] [inputs_dir] [results_dir]
"""
import os, sys
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from utils.read import BulkSystem
from utils.ham import Hamiltonian
from utils.pp_func import pot_funcLR
from utils.nn_models import Net_celu_HeInit_decayGaussian_LSD

BACKEND = sys.argv[1] if len(sys.argv) > 1 else "mace"
INP     = sys.argv[2] if len(sys.argv) > 2 else "inputs_lsd_train_mace"
RES     = sys.argv[3] if len(sys.argv) > 3 else "results_lsd_train_mace_2"
ELEMENTS = ["Cs", "I", "Pb"]
GAUSS_STD = 3.5


def build_lsd_models(n_descr, ref_system):
    """Per-element LSD nets. Load trained weights for MACE (D=256) if present,
    else random; N_ref (cubic reference descriptor) is set from ref_system since
    it is not stored in the saved state_dict."""
    models = {}
    for el in ELEMENTS:
        nd = n_descr[el]
        m = Net_celu_HeInit_decayGaussian_LSD([nd + 1, min(256, 16 * nd), 16, 1]
                                              if nd <= 5 else [nd + 1, 256, 128, 64, 1],
                                              gaussian_std=GAUSS_STD)
        ckpt = os.path.join(RES, f"final_{el}_LSDmodel.pth")
        if os.path.exists(ckpt) and nd == 256:
            m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        m.N_ref = ref_system.env_descriptors[el][0:1].detach().clone()
        m.eval()
        models[el] = m
    return models


def main():
    bs = BulkSystem(); bs.systemName = "CsPbI3"
    bs.setSystem(f"{INP}/system_1.par")
    bs.compute_descriptors(backend=BACKEND, differentiable=True)
    bs.qpts = torch.zeros(1, 3)                       # single Gamma phonon q
    natom = bs.getNAtoms()

    ref = BulkSystem(); ref.systemName = "CsPbI3"
    ref.setSystem(f"{INP}/system_0.par")
    ref.compute_descriptors(backend=BACKEND, differentiable=False)

    LSDmodels = build_lsd_models(bs.n_descr, ref)

    ham = object.__new__(Hamiltonian)
    ham.system      = bs
    ham.NNConfig    = {"local_env_corr": 1, "descriptor_backend": BACKEND, "checkpoint": 0}
    ham.SObool      = False
    ham.NN_locbool  = False
    ham.LRgamma     = 1.0
    ham.atomPPorder = np.array(ELEMENTS)
    ham.LSDmodels   = LSDmodels
    ham.PPparams    = {el: torch.tensor([0.5, 2.0, 1.3, 0.4, 0.05, 0., 0., 0.])
                       for el in ELEMENTS}
    g = torch.Generator().manual_seed(0)
    nbv = 6
    ham.basis = (torch.rand(nbv, 3, generator=g) - 0.5) * 2.0

    qpt = bs.qpts[0]
    Omega = bs.getCellVolume()

    def Vlocal(atomPos):
        desc = bs.descriptors_from_pos(atomPos, backend=BACKEND)
        Q  = ham.basis[:, None, :] - (ham.basis[None, :, :] + qpt)
        Qn = torch.norm(Q, dim=2)
        V  = torch.zeros(nbv, nbv, dtype=torch.complex128)
        for b in range(natom):
            tb = bs.atomTypes[b]
            phase = torch.sum(Q * atomPos[b], dim=2)
            SF = (1.0 / Omega) * (torch.cos(phase) + 1j * torch.sin(phase))
            base = pot_funcLR(Qn, ham.PPparams[tb], ham.LRgamma)
            ib = torch.where(bs.atom_indices[tb] == b)[0].squeeze(0)
            Nb = desc[tb][ib]
            xin = torch.cat([Nb.unsqueeze(0).expand(nbv * nbv, -1), Qn.reshape(-1, 1)], dim=1)
            V = V + SF * (base + LSDmodels[tb](xin).view(nbv, nbv))
        return V

    pairs = [(0, 0), (4, 1), (8, 2), (12, 1)]
    with torch.no_grad():
        dV = ham.buildCouplingMats(0, atomgammaidxs=pairs)

    h = 1e-5 if BACKEND == "mace" else 1e-6
    print(f"\nbackend={BACKEND}  | central-difference check of buildCouplingMats")
    print(f"{'atom,gamma':>10} | {'max|analytic-FD|':>18} | {'max|FD|':>12} | rel.err")
    worst = 0.0
    for (mu, gam) in pairs:
        Rp = bs.atomPos.detach().clone(); Rp[mu, gam] += h
        Rm = bs.atomPos.detach().clone(); Rm[mu, gam] -= h
        with torch.no_grad():
            fd = (Vlocal(Rp) - Vlocal(Rm)) / (2 * h)
        ana = dV[(mu, gam)]
        err = (ana - fd).abs().max().item()
        scale = fd.abs().max().item()
        rel = err / (scale + 1e-30)
        worst = max(worst, rel)
        print(f"{str((mu, gam)):>10} | {err:18.3e} | {scale:12.3e} | {rel:.2e}")
    tol = 1e-5
    print(f"WORST relative error: {worst:.2e}  -> {'PASS' if worst < tol else 'FAIL'}")

    # coupling must stay differentiable w.r.t. LSD params for fit_couplings
    for el in ELEMENTS:
        for p in LSDmodels[el].parameters():
            p.requires_grad_(True)
    dVg = ham.buildCouplingMats(0, atomgammaidxs=[(8, 2)])
    loss = dVg[(8, 2)].abs().pow(2).sum()
    loss.backward()
    gI = sum((p.grad.norm().item() if p.grad is not None else 0.0)
             for p in LSDmodels["I"].parameters())
    print(f"[grad] coupling differentiable w.r.t. LSD params; sum|grad|(I)={gI:.3e} "
          f"-> {'PASS' if gI > 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
