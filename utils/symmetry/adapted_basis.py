"""
Symmetry-adapted plane-wave-spinor (SAPW) basis -- THE main deliverable.

For a k-point with double little group DLG we block-diagonalize the spinor
Hamiltonian H (dim 2*nbv, spin-major index = spin*nbv + iG, up block first --
matches ham.py) by irrep, WITHOUT ever computing eigenvectors in the hot path.

CONSTRUCTION (setup, run once, cached)
--------------------------------------
The little group acts on plane waves by  G -> R G + G0  (R k = k + G0), and on
spin by  s * D^{1/2}(R)  (s = SU(2) double-group sign). Orbits of G under this
action are STARS; each star has <= |little group| <= 48 G-vectors, so a SAPW
lives on <= 2*48 = 96 components (spinor). We therefore work STAR-BY-STAR:

  * Build the small (2L x 2L) rep u_S(g) of DLG on one star S (L = |S|).
  * For each spinor irrep lambda, the isotypic projector
        P^lambda_S = (d_lambda / |DLG|) sum_g chi^lambda(g)* u_S(g)
    is an orthogonal projector (Hermitian, idempotent) onto the lambda-isotypic
    component; its range (eigenvalue-1 eigenvectors) are orthonormal SAPWs.
  * Embed each SAPW (length 2L) as a SPARSE column in the full 2*nbv space.

We use the FULL isotypic projector (characters only -- no error-prone irrep
MATRICES), so each physical lambda-multiplet appears d_lambda times inside its
block. block_eigvals() collapses those exact d_lambda-fold degeneracies back to
one eigenvalue per physical state and the loss weights each by d_lambda (exactly
the prompt's L = sum_lambda d_lambda sum_n w_n dE^2). This gives identical
physics to the single-row projector with zero irrep-matrix transcription risk.

BOUNDARY-k TRUNCATION (important, honest)
-----------------------------------------
The model uses a k-INDEPENDENT |G|-sphere basis. At Gamma every G0 = 0, so every
star is closed and block-diagonalization is EXACT to machine precision. At
R/X/M/lines some ops have G0 != 0, so a thin boundary shell of stars maps partly
outside the sphere (INCOMPLETE stars). Those are excluded from the SAPW basis and
reported; the low bands we fit live entirely in complete interior stars, where
block-diagonalization is exact. Completeness then reads
    sum_lambda n_lambda == 2 * N_complete   (== 2*nbv only when no star is cut).
The excluded weight is quantified by validate.measure_boundary_leakage().

B is a CONSTANT tensor: built once, stored as a torch sparse tensor with
requires_grad=False. Only H carries gradient to the potential parameters.
"""
import numpy as np
import torch

from .group import build_Oh
from .irreps import DoubleLittleGroup


def _corep_name(members):
    """Readable label for a merged time-reversal co-rep sector. For a
    complex-conjugate 1-dim pair sharing a stem (e.g. 'E1dim_a', 'E1dim_a#1')
    return '<stem>_cc'; otherwise join the two labels."""
    import re
    stems = {re.split(r'[_#]', m)[0] for m in members}
    if len(stems) == 1:
        return f"{stems.pop()}_cc"
    return "+".join(members)


class StarDecomposition:
    """Orbits (stars) of the G-list under the little group's G-action G->R G+G0,
    with per-star, per-element within-star source permutation and completeness."""

    def __init__(self, dlg, mill, index_of):
        self.dlg = dlg
        self.mill = mill                    # (nbv,3) integer Miller indices
        self.nbv = mill.shape[0]
        self.index_of = index_of            # {(m0,m1,m2): iG}
        self._build()

    def _image(self, iG, el_local):
        """Global G-index of R G + G0 for op element index el_local, or None if
        it leaves the sphere."""
        i, s = self.dlg.elements[el_local]
        R = self.dlg.oh[i]['R']
        G0 = self.dlg.G0[i]
        m = R @ self.mill[iG] + G0
        return self.index_of.get(tuple(int(x) for x in m), None)

    def _build(self):
        n = self.nbv
        assigned = -np.ones(n, dtype=np.int64)
        stars = []
        nlg = len(self.dlg.elements)
        for seed in range(n):
            if assigned[seed] >= 0:
                continue
            # orbit of seed under all elements (spatial action only; the +1/-1
            # spin sign does not move G, so iterate over spatial ops = first half)
            orbit = {}
            complete = True
            frontier = [seed]
            orbit[seed] = True
            while frontier:
                g = frontier.pop()
                for el in range(nlg):
                    img = self._image(g, el)
                    if img is None:
                        complete = False
                        continue
                    if img not in orbit:
                        orbit[img] = True
                        frontier.append(img)
            members = sorted(orbit.keys())
            for gidx in members:
                assigned[gidx] = len(stars)
            stars.append({'members': members, 'complete': complete})
        self.stars = stars
        self.n_complete_G = sum(len(s['members']) for s in stars if s['complete'])
        self.n_incomplete_G = n - self.n_complete_G

    def within_star_perm(self, star, el_local):
        """For a COMPLETE star, return array src[a'] = local index a such that
        G_member[a] == R^{-1}(G_member[a'] - G0), i.e. the SOURCE component that
        maps INTO target a' under U(el). (U(el)c)(a') = sum_sp M c(src[a'])."""
        members = star['members']
        pos = {g: a for a, g in enumerate(members)}
        src = np.empty(len(members), dtype=np.int64)
        for aprime, gt in enumerate(members):
            # find source g with image(g)=gt: image is a bijection on the star,
            # so invert by scanning (stars are tiny, <=48).
            found = -1
            for a, g in enumerate(members):
                if self._image(g, el_local) == gt:
                    found = a
                    break
            assert found >= 0, "within-star permutation not bijective on complete star"
            src[aprime] = found
        return src


def _star_rep(dlg, star, star_dec):
    """Small (nel, 2L, 2L) rep u_S(el) of the double group on one complete star.
    Local basis order: (component a, spin sp) -> row a*2+sp, sp in {0=up,1=dn}."""
    members = star['members']
    L = len(members)
    nel = len(dlg.elements)
    u = np.zeros((nel, 2 * L, 2 * L), dtype=np.complex128)
    for el in range(nel):
        i, s = dlg.elements[el]
        M = s * dlg.oh[i]['su2']                 # 2x2 spin action (up,dn)
        src = star_dec.within_star_perm(star, el)
        for aprime in range(L):
            a = src[aprime]
            for spp in range(2):
                for sp in range(2):
                    u[el, aprime * 2 + spp, a * 2 + sp] = M[spp, sp]
    return u


class SymmetryAdaptedK:
    """Cached SAPW basis at one k-point: sparse B per irrep + sector metadata.
    Everything here is a constant (requires_grad=False)."""

    def __init__(self, dlg, star_dec, irrep_blocks, nbv, device=None,
                 dtype=torch.complex128, corep=None):
        self.dlg = dlg
        self.name = dlg.name
        self.nbv = nbv
        self.dim = 2 * nbv
        self.device = device
        self.dtype = dtype
        # irrep_blocks: list of dicts {label,dim(d_lambda),parity,cols(list of
        #   (rows_global, values)), n_lambda}
        self.irreps = irrep_blocks
        # time-reversal co-representation partner map {label: partner or None}
        self.corep = corep or {blk['label']: None for blk in irrep_blocks}
        self._build_sparse()
        self._build_effective_sectors()

    def _build_effective_sectors(self):
        """Group unitary irreps into physical TIME-REVERSAL co-rep sectors: a
        complex-conjugate pair (lambda, lambda*) becomes one sector of dim
        d_lambda+d_lambda*; every other irrep is its own sector. Populates
        self.eff_sectors = [{'label','dim','members':[irrep labels]}] and the
        lookup self.eff_dim = {eff_label: dim}."""
        dim_of = {b['label']: b['dim'] for b in self.irreps}
        eff = []
        done = set()
        for blk in self.irreps:
            lab = blk['label']
            if lab in done:
                continue
            partner = self.corep.get(lab)
            if partner is not None and partner in dim_of:
                members = sorted([lab, partner])
                eff.append({'label': _corep_name(members), 'members': members,
                            'dim': dim_of[lab] + dim_of[partner]})
                done.update(members)
            else:
                eff.append({'label': lab, 'members': [lab], 'dim': dim_of[lab]})
                done.add(lab)
        self.eff_sectors = eff
        self.eff_dim = {s['label']: s['dim'] for s in eff}

    def _build_sparse(self):
        for blk in self.irreps:
            rows = []
            cols = []
            vals = []
            for cidx, (rlist, vlist) in enumerate(blk['cols']):
                rows.extend(rlist)
                cols.extend([cidx] * len(rlist))
                vals.extend(vlist)
            ncol = len(blk['cols'])
            idx = torch.tensor([rows, cols], dtype=torch.long)
            v = torch.tensor(vals, dtype=self.dtype)
            B = torch.sparse_coo_tensor(idx, v, size=(self.dim, ncol)).coalesce()
            B.requires_grad_(False)
            if self.device:
                B = B.to(self.device)
            # cache B^dag and B^T (both ncol x dim) for the hot-path B^dag H B.
            Bdag = torch.sparse_coo_tensor(idx.flip(0), v.conj(), size=(ncol, self.dim)).coalesce()
            Bt = torch.sparse_coo_tensor(idx.flip(0), v, size=(ncol, self.dim)).coalesce()
            for t in (Bdag, Bt):
                t.requires_grad_(False)
            blk['B'] = B
            blk['Bdag'] = Bdag.to(self.device) if self.device else Bdag
            blk['Bt'] = Bt.to(self.device) if self.device else Bt
            blk['ncol'] = ncol
            blk['n_lambda'] = ncol // blk['dim']   # physical multiplicity

    def sector_report(self):
        lines = []
        tot = 0
        for blk in self.irreps:
            tot += blk['ncol']
            lines.append(f"    {blk['label']:9s} d={blk['dim']}  n_lambda(phys)={blk['n_lambda']:4d}"
                         f"  cols={blk['ncol']:4d}")
        return f"[{self.name}] sectors (sum cols = {tot} = 2*N_complete):\n" + "\n".join(lines)


def build_symmetry_adapted_k(oh, kfrac, mill, index_of, nbv, name="",
                             device=None, dtype=torch.complex128, tol=1e-9):
    """Construct the SAPW basis at one k-point. Pure setup; returns SymmetryAdaptedK.

    mill      : (nbv,3) int Miller indices of the G-list (G = mill @ b).
    index_of  : {(m0,m1,m2): iG} hash for the G-list.
    """
    dlg = DoubleLittleGroup(oh, kfrac, name=name)
    dlg.character_table()
    spin_irreps = dlg.spinor_irreps()
    star_dec = StarDecomposition(dlg, mill, index_of)

    # accumulate SAPW columns per irrep
    blocks = {r['lam']: {'label': r['label'], 'dim': r['dim'], 'parity': r['parity'],
                         'char_elem': r['char_elem'], 'cols': []} for r in spin_irreps}

    nel = len(dlg.elements)
    for star in star_dec.stars:
        if not star['complete']:
            continue
        members = star['members']
        L = len(members)
        u = _star_rep(dlg, star, star_dec)         # (nel,2L,2L)
        for r in spin_irreps:
            lam = r['lam']
            d = r['dim']
            chi = r['char_elem']                   # (nel,)
            # isotypic projector P = (d/|G~|) sum_g chi(g)* u(g)
            P = (d / nel) * np.tensordot(chi.conj(), u, axes=([0], [0]))  # (2L,2L)
            # Hermitian orthogonal projector: eigen-decompose, keep eval~1
            P = 0.5 * (P + P.conj().T)
            w, V = np.linalg.eigh(P)
            keep = np.where(w > 0.5)[0]
            if keep.size == 0:
                continue
            # sanity: kept eigenvalues ~1
            assert np.all(np.abs(w[keep] - 1.0) < 1e-6), \
                f"projector not idempotent on star (evals {w[keep]})"
            for c in keep:
                vec = V[:, c]                       # length 2L, local (a,sp)
                # embed to global: row = sp*nbv + members[a]
                rlist, vlist = [], []
                for a in range(L):
                    for sp in range(2):
                        amp = vec[a * 2 + sp]
                        if abs(amp) > 1e-12:
                            rlist.append(sp * nbv + members[a])
                            vlist.append(complex(amp))
                blocks[lam]['cols'].append((rlist, vlist))

    irrep_blocks = [blocks[r['lam']] for r in spin_irreps]
    # time-reversal co-rep partners, keyed by irrep LABEL (for the sector merge)
    partner_by_lam = dlg.corep_classify(spin_irreps)   # {label: partner_label|None}
    corep = {r['label']: partner_by_lam[r['label']] for r in spin_irreps}
    return SymmetryAdaptedK(dlg, star_dec, irrep_blocks, nbv, device=device,
                            dtype=dtype, corep=corep)
