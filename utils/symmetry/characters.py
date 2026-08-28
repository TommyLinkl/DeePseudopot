"""
Character machinery (SETUP-TIME / VALIDATION ONLY -- never called per training
step). Implements the group action on plane-wave spinors and the character of a
degenerate eigen-subspace, used to (a) validate the SAPW basis by independently
labeling the model's own eigenvectors, and (b) label externally supplied
spinor wavefunctions (e.g. QE) in the SAME convention as the model.

Group action (symmorphic; a fractional translation t enters only as the phase
noted below, t=0 for Pm-3m at these sites):

    (U({R|t}) psi)(G', s') = sum_s D^{1/2}(R)_{s',s} * e^{-i (k+G')·t} * psi(G, s)
    with  G = R^{-1}(G' - G0),  R k = k + G0.

CONVENTIONS: spin-major global index = s*nbv + iG (s=0 up); G-list Cartesian;
D^{1/2} from group.su2 (active). See group.py header.
"""
import numpy as np


def build_global_action(dlg, el, mill, index_of, nbv):
    """Return (src, M, valid) for double-group element index `el`:
      src[g']  = source G-index  R^{-1}(G'-G0)  (or -1 if out of sphere)
      M        = 2x2 spin matrix s * D^{1/2}(R)
      valid    = bool mask over G' (whether the source is in the sphere)
    so that (U psi)[s'*nbv+g'] = sum_s M[s',s] psi[s*nbv+src[g']] for valid g'."""
    i, s = dlg.elements[el]
    R = dlg.oh[i]['R']
    G0 = dlg.G0[i]
    Rinv = np.rint(np.linalg.inv(R)).astype(np.int64)
    src = -np.ones(nbv, dtype=np.int64)
    for gprime in range(nbv):
        m = Rinv @ (mill[gprime] - G0)
        src[gprime] = index_of.get(tuple(int(x) for x in m), -1)
    valid = src >= 0
    M = s * dlg.oh[i]['su2']
    return src, M, valid


def apply_U(psi, src, M, valid, nbv):
    """Apply U(el) to a (2*nbv,) or (2*nbv, nvec) complex array. Out-of-sphere
    sources contribute 0 (only exact on complete-star subspace)."""
    psi = np.asarray(psi)
    single = psi.ndim == 1
    if single:
        psi = psi[:, None]
    up_in, dn_in = psi[:nbv], psi[nbv:]
    out = np.zeros_like(psi)
    gp = np.where(valid)[0]
    sg = src[gp]
    # spinor mixing: out_up = M00 up_src + M01 dn_src ; out_dn = M10 up_src + M11 dn_src
    out[gp] = M[0, 0] * up_in[sg] + M[0, 1] * dn_in[sg]
    out[nbv + gp] = M[1, 0] * up_in[sg] + M[1, 1] * dn_in[sg]
    return out[:, 0] if single else out


def subspace_character(vectors, src, M, valid, nbv):
    """chi(R) = tr_subspace U(R) = sum_i <v_i| U(R) |v_i> for orthonormal columns
    v_i (a degenerate multiplet). vectors: (2*nbv, d)."""
    Uv = apply_U(vectors, src, M, valid, nbv)
    return np.trace(vectors.conj().T @ Uv)


def group_degenerate(energies, tol=1e-4):
    """Group ascending eigenvalues into degenerate multiplets (index lists)."""
    energies = np.asarray(energies)
    groups = []
    cur = [0]
    for i in range(1, len(energies)):
        if abs(energies[i] - energies[cur[-1]]) < tol:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups


def characters_of_multiplet(vectors, dlg, mill, index_of, nbv):
    """Per-element characters chi(g) for one degenerate multiplet (columns of
    `vectors`). Returns complex array length |DLG|."""
    chi = np.zeros(len(dlg.elements), dtype=np.complex128)
    for el in range(len(dlg.elements)):
        src, M, valid = build_global_action(dlg, el, mill, index_of, nbv)
        chi[el] = subspace_character(vectors, src, M, valid, nbv)
    return chi


def class_average(chi, dlg):
    """Average chi over conjugacy classes; return (avg per class, max spread)."""
    avg = np.zeros(dlg.nclass, dtype=np.complex128)
    spread = 0.0
    for ci, cl in enumerate(dlg.classes):
        vals = chi[cl]
        avg[ci] = vals.mean()
        spread = max(spread, np.abs(vals - avg[ci]).max())
    return avg, spread


def decompose(chi, dlg, spin_irreps, tol=1e-3):
    """Decompose a (per-element) character into spinor irreps:
        m_lambda = (1/|DLG|) sum_g chi(g) chi^lambda(g)*.
    Returns {label: multiplicity}; flags non-integer multiplicities (a sign that
    degenerate grouping failed or conventions are inconsistent)."""
    n = len(dlg.elements)
    out = {}
    flags = []
    for r in spin_irreps:
        m = np.sum(chi * r['char_elem'].conj()) / n
        mr = m.real
        out[r['label']] = mr
        if abs(mr - round(mr)) > tol or abs(m.imag) > tol:
            flags.append((r['label'], complex(m)))
    return out, flags
