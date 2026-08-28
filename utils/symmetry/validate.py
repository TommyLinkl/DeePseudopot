"""
Validation suite for the symmetry-adapted basis. These tests ACTUALLY RUN against
the real DeePseudopot Hamiltonian pipeline (kinetic + local NN potential + KB
nonlocal + SOC), which is the sharpest check.

Run:
  PYTHONPATH=<DeePseudopot> python -m utils.symmetry.validate <test_inputs_folder>

Tests
  1. completeness            sum_lambda n_cols == 2 * N_complete (== 2*nbv at Gamma)
  2. B orthonormality        B_l^dag B_l = I, B_l^dag B_m = 0  (l != m)
  3. block-diagonality       ||B_l^dag H B_m|| ~ 0 for l != m  (the sharp test)
  4. spectrum equivalence    block spectrum == eigvalsh(H) (exact at Gamma)
  5. free-electron (V=0)     empty-lattice degeneracies + integer irrep multiplicities
  6. boundary leakage        quantify the |G|-sphere truncation at boundary k
"""
import sys
import numpy as np
import torch

from utils.symmetry.group import build_Oh
from utils.symmetry.interface import build_sak_for_k, HIGH_SYM_POINTS
from utils.symmetry.hot_path import block_eigvals, _block_matrix
from utils.constants import AUTOEV


# ---------------------------------------------------------------------------
def _setup(inputs):
    """Replicate main.py's minimal setup: systems + random NN model + cached
    SO/NL mats, so ham.buildHtot(kidx) yields a symmetry-respecting spinor H."""
    from utils.config_threads import configure_threads
    configure_threads(n_threads=1)
    torch.set_default_dtype(torch.float64)
    from utils.read import read_NNConfigFile, setAllBulkSystems, setNN
    from utils.ham import initAndCacheHams
    results = inputs.rstrip('/') + '/_valout/'
    import os
    os.makedirs(results, exist_ok=True)
    NNConfig = read_NNConfigFile(inputs + 'NN_config.par', results)
    NNConfig['inputsFolder'] = inputs
    NNConfig['resultsFolder'] = results
    systems, atomPPOrder, nPseudopot, PPparams, totalParams, localPotParams = \
        setAllBulkSystems(1, inputs, results, NNConfig['local_env_corr'],
                          descriptor_backend=NNConfig.get('descriptor_backend', 'handcrafted'))
    torch.manual_seed(1)
    PPmodel = setNN(NNConfig, nPseudopot)     # random-init NN = random symmetric V(|G|)
    hams, cachedMats_info, shmSO, shmNL = initAndCacheHams(
        systems, NNConfig, PPparams, atomPPOrder, torch.device('cpu'), model=PPmodel)
    # attach the NN model exactly as the training/eval paths do (ham.set_NNmodel)
    for h in hams:
        h.NN_locbool = True
        h.set_NNmodel(PPmodel)
    return systems[0], hams[0], NNConfig


def _H_at(ham, kidx):
    with torch.no_grad():
        H = ham.buildHtot(kidx, requires_grad=False)
    return H.detach()


# ---------------------------------------------------------------------------
def test_orthonormality(sak, tol=1e-10):
    print(f"\n[{sak.name}] 2. B orthonormality (B_l^dag B_m):")
    ok = True
    blks = sak.irreps
    for a, ba in enumerate(blks):
        if ba['ncol'] == 0:
            continue
        for b, bb in enumerate(blks):
            if bb['ncol'] == 0:
                continue
            # dense small: B_a^dag B_b
            Ba = ba['B'].to_dense()
            Bb = bb['B'].to_dense()
            G = Ba.conj().transpose(-2, -1) @ Bb
            if a == b:
                err = (G - torch.eye(ba['ncol'], dtype=G.dtype)).abs().max().item()
                tag = "I"
            else:
                err = G.abs().max().item()
                tag = "0"
            if err > 1e-9:
                ok = False
                print(f"    {ba['label']:8s} x {bb['label']:8s} -> {tag}  max dev {err:.2e}  FAIL")
    print("    PASS" if ok else "    *** FAIL ***")
    return ok


def test_block_diagonality(H, sak):
    print(f"\n[{sak.name}] 3. block-diagonality ||B_l^dag H B_m|| (l != m):")
    blks = [b for b in sak.irreps if b['ncol'] > 0]
    # precompute H B_m as dense via sparse mm
    HB = {}
    for b in blks:
        HB[b['label']] = torch.sparse.mm(b['Bdag'], H)   # B^dag H  (ncol,dim)
    worst_off = 0.0
    worst_diag = 0.0
    for a in blks:
        Za = torch.sparse.mm(a['Bdag'], H)              # (ncol_a, dim)
        for b in blks:
            # A_ab = B_a^dag H B_b = Za @ B_b
            Aab = torch.sparse.mm(b['Bt'], Za.transpose(-2, -1)).transpose(-2, -1)
            m = Aab.abs().max().item()
            if a['label'] == b['label']:
                worst_diag = max(worst_diag, m)
            else:
                worst_off = max(worst_off, m)
    print(f"    max |off-block| = {worst_off:.3e}   (typical diag scale {worst_diag:.3e})")
    print(f"    ratio off/diag  = {worst_off/max(worst_diag,1e-300):.3e}")
    return worst_off, worst_diag


def test_spectrum_equivalence(H, sak, atol=1e-8):
    print(f"\n[{sak.name}] 4. spectrum equivalence (blocks vs full eigvalsh):")
    # full spectrum on the COMPLETE-star subspace: project H there
    keep_rows = _complete_row_indices(sak)
    Hc = H[keep_rows][:, keep_rows]
    full = torch.linalg.eigvalsh(Hc).tolist()
    # block spectrum: expand each physical eigenvalue by its d_lambda multiplicity
    block = []
    ev = block_eigvals(H, sak, collapse=True, merge_coreps=False)
    for blk in sak.irreps:
        d = blk['dim']
        for e in ev[blk['label']].tolist():
            block.extend([e] * d)
    full = np.sort(np.array(full))
    block = np.sort(np.array(block))
    assert len(full) == len(block), f"dim mismatch full {len(full)} block {len(block)}"
    dev = np.abs(full - block).max()
    print(f"    dim = {len(full)}   max|dE| = {dev*AUTOEV:.3e} eV  ({dev:.3e} Ha)")
    return dev


def test_boundary_leakage(H, sak, nlow=42):
    """Does excluding incomplete boundary stars change the LOW (fitted) bands?
    Compare lowest nlow eigenvalues of the FULL H vs the complete-star-restricted
    H. This is the physically decisive number for the sector loss."""
    keep = _complete_row_indices(sak)
    Hc = H[keep][:, keep]
    full = torch.linalg.eigvalsh(H)[:nlow]
    restr = torch.linalg.eigvalsh(Hc)[:nlow]
    dev = (full - restr).abs()
    print(f"[{sak.name}] 6. boundary leakage on lowest {nlow} bands: "
          f"max|dE| = {dev.max().item()*AUTOEV:.3e} eV, "
          f"mean = {dev.mean().item()*AUTOEV:.3e} eV")
    return dev.max().item() * AUTOEV


def _complete_row_indices(sak):
    """Global spin-major row indices (spin*nbv+iG) spanned by complete stars =
    exactly the rows covered by the SAPW columns."""
    rows = set()
    for blk in sak.irreps:
        B = blk['B'].coalesce()
        idx = B.indices()[0].tolist()
        rows.update(idx)
    return torch.tensor(sorted(rows), dtype=torch.long)


def test_completeness(sak):
    nbv = sak.nbv
    ncols = sum(b['ncol'] for b in sak.irreps)
    ncomplete_rows = len(_complete_row_indices(sak))
    print(f"\n[{sak.name}] 1. completeness: sum cols = {ncols}, "
          f"2*nbv = {2*nbv}, rows in complete stars = {ncomplete_rows}")
    assert ncols == ncomplete_rows, f"cols {ncols} != complete rows {ncomplete_rows}"
    deficit = 2 * nbv - ncols
    print(f"    boundary deficit = {deficit} spinor-components "
          f"({100*deficit/(2*nbv):.2f}% of basis, excluded incomplete stars)")
    return ncols, 2 * nbv


def test_free_electron(ham, oh):
    """V=0 at Gamma: H = 0.5|G|^2 (x) I_2. Check empty-lattice shells decompose
    into integer irrep multiplicities and doubled (spinor) degeneracies."""
    print(f"\n[free-electron] V=0 empty lattice at Gamma:")
    basis = ham.basis.detach().cpu().numpy()
    nbv = basis.shape[0]
    kin = 0.5 * (basis ** 2).sum(1)                       # (nbv,)
    diag = np.concatenate([kin, kin])                     # spin-major
    H = torch.tensor(np.diag(diag), dtype=torch.complex128)
    sak = build_sak_for_k(ham, [0, 0, 0], name='Gamma_free', oh=oh)
    ev = block_eigvals(H, sak, collapse=True, merge_coreps=False)
    # gather all physical levels with labels, group by energy shell
    levels = []
    for blk in sak.irreps:
        for e in ev[blk['label']].tolist():
            levels.append((round(e, 8), blk['label'], blk['dim']))
    levels.sort()
    # summarize lowest few shells
    from collections import defaultdict
    shells = defaultdict(lambda: defaultdict(int))
    for e, lab, d in levels:
        shells[e][lab] += 1
    print(f"    lowest empty-lattice shells (energy Ha : {{irrep: count}}, degeneracy):")
    nshow = 0
    for e in sorted(shells)[:6]:
        decomp = dict(shells[e])
        deg = sum(sak_dim(sak, lab) * c for lab, c in decomp.items())
        print(f"      {e:10.5f} : {decomp}   total spinor deg = {deg}")
        nshow += 1
    return sak


def sak_dim(sak, label):
    for b in sak.irreps:
        if b['label'] == label:
            return b['dim']
    return 0


# ---------------------------------------------------------------------------
def main(inputs):
    oh = build_Oh()
    system, ham, NNConfig = _setup(inputs)
    nbv = ham.basis.shape[0]
    print(f"\n=== VALIDATION (nbv={nbv}, dim=2*nbv={2*nbv}, spinor={ham.spinor}) ===")
    # k-points present in this test system: Gamma(0),R(1),X(2),M(3)
    kmap = {'Gamma': (0, [0, 0, 0]), 'R': (1, [.5, .5, .5]),
            'X': (2, [.5, 0, 0]), 'M': (3, [.5, .5, 0])}
    for name, (kidx, kfrac) in kmap.items():
        sak = build_sak_for_k(ham, kfrac, name=name, oh=oh)
        print("\n" + "=" * 70)
        print(sak.sector_report())
        test_completeness(sak)
        test_orthonormality(sak)
        H = _H_at(ham, kidx)
        # Hermiticity of H
        herm = (H - H.conj().transpose(-2, -1)).abs().max().item()
        print(f"[{name}] H Hermiticity max|H-H^dag| = {herm:.2e}")
        test_block_diagonality(H, sak)
        test_spectrum_equivalence(H, sak)
        test_boundary_leakage(H, sak, nlow=min(42, len(_complete_row_indices(sak))))
    # free-electron at Gamma
    test_free_electron(ham, oh)
    print("\n=== VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
