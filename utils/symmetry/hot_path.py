"""
Training HOT PATH: given the dense spinor Hamiltonian H at a symmetry k-point and
the precomputed SAPW basis, return eigenvalues pre-labeled by irrep -- sparse
transform, one eigvalsh per block, NO eigenvectors, fully differentiable in H.

Everything here is pure and cheap; B is a constant (no grad). Only H carries the
gradient to the potential parameters, and it flows through the two sparse mms and
eigvalsh (degeneracy-safe: eigenVALUE autograd has no 1/(l_i-l_j) terms).
"""
import torch


def _block_matrix(H, blk):
    """A = B^dag H B for one irrep block, using sparse B (never densified).
    Cached on blk: Bdag = B^dag (ncol,dim) sparse; Bt = B^T (ncol,dim) sparse.

      Z  = Bdag @ H                    (ncol, dim)  = B^dag H
      A^T = Bt @ Z^T                   (ncol,ncol)  = (Z B)^T = (B^dag H B)^T
    (Bt is a plain transpose, no conjugation, so Bt @ Z^T contracts the shared
    G-index correctly; the conjugation of B already lives in Z via Bdag.)"""
    Bdag = blk['Bdag']
    Bt = blk['Bt']
    Z = torch.sparse.mm(Bdag, H)                       # (ncol, dim)
    At = torch.sparse.mm(Bt, Z.transpose(-2, -1))       # (ncol, ncol) = A^T
    A = At.transpose(-2, -1)
    return 0.5 * (A + A.conj().transpose(-2, -1))        # Hermitize away roundoff


def block_eigvals(H, sak, collapse=True, need=None, merge_coreps=True):
    """Return {sector_label: eigenvalues_tensor} at this k-point (H units, ascending).

    H        : dense (2*nbv, 2*nbv) complex Hermitian (from ham.buildHtot), grad-ok.
    sak      : SymmetryAdaptedK for this k.
    collapse : average the exact d_lambda-fold degeneracy inside each block back to
               one eigenvalue per PHYSICAL state (the full-isotypic-projector gives
               d_lambda copies). If False, returns the raw block spectrum.
    merge_coreps : merge time-reversal co-rep partners (a complex-conjugate irrep
               pair) into one physical sector -- required so that lines like Lambda
               (C_3v), whose two 1-dim spinor irreps are Kramers partners, give one
               2-fold sector matching the reference. Points (Gamma/R/X/M) have no
               such pairs, so this is a no-op there.
    need     : if set, keep only the lowest `need` physical eigenvalues per sector.
    Returns {label: 1D tensor}.
    """
    raw = {}
    for blk in sak.irreps:
        if blk['ncol'] == 0:
            raw[blk['label']] = H.new_zeros(0, dtype=torch.float64)
            continue
        A = _block_matrix(H, blk)
        w = torch.linalg.eigvalsh(A)             # ascending, real
        if collapse:
            d = blk['dim']
            n_lambda = blk['ncol'] // d
            w = w.view(n_lambda, d).mean(dim=1)  # collapse exact d-fold degeneracy
        raw[blk['label']] = w

    if not merge_coreps:
        return {k: (v[:need] if need is not None else v) for k, v in raw.items()}

    # merge into physical co-rep sectors (sak.eff_sectors)
    out = {}
    for sec in sak.eff_sectors:
        members = sec['members']
        if len(members) == 1:
            w = raw[members[0]]
        else:
            # complex-conjugate pair: partners are Kramers-degenerate (TRS), so the
            # two sorted spectra coincide; average them into one physical level set.
            parts = [torch.sort(raw[m])[0] for m in members if raw[m].numel() > 0]
            m = min((p.numel() for p in parts), default=0)
            w = sum(p[:m] for p in parts) / len(parts) if parts else raw[members[0]]
        out[sec['label']] = w[:need] if need is not None else w
    return out


def irrep_sequence(H, sak, n_show=None, unit=27.2114):
    """Cheap diagnostic / freeze input: ascending list of (energy, irrep_label,
    dim) for the PHYSICAL states at this k-point (one entry per multiplet).
    Energies scaled by `unit` (default Hartree->eV). Use every epoch at Gamma/R."""
    ev = block_eigvals(H, sak, collapse=True)
    dim_of = sak.eff_dim          # merged co-rep sector dims (labels match ev keys)
    tagged = []
    for label, w in ev.items():
        for e in w.tolist():
            tagged.append((e * unit, label, dim_of[label]))
    tagged.sort(key=lambda t: t[0])
    if n_show is not None:
        tagged = tagged[:n_show]
    return tagged
