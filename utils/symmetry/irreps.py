"""
Double-group representation theory for a little group, generated AUTOMATICALLY
from the group multiplication table (no hand-transcribed character tables -- the
"convention trap" the physics of this problem is famous for).

Pipeline
--------
1. Build the double little group DLG as elements (oh_index, su2_sign) and its
   multiplication table via the SU(2) cocycle (group.su2_cocycle).
2. Conjugacy classes.
3. Character table of ALL irreps via the Burnside / class-algebra eigenvalue
   method (simultaneous eigenvectors of the class-sum matrices).
4. Validate: row & column orthogonality, sum_lambda dim^2 == |DLG|.
5. Keep the SPINOR (double-valued) irreps: chi(E-bar) = -dim (the 2*pi rotation
   is represented by -1). Single-valued irreps (chi(E-bar)=+dim) are the ordinary
   ones and are irrelevant for spinor eigenstates.
6. LABEL each spinor irrep by STRUCTURE, not name strings: dimension and parity
   (character of inversion) are unambiguous; Gamma6 vs Gamma7 is separated by the
   character of a 3-fold rotation (C3), cross-checked against the physical
   Wigner-D characters chi_{1/2}, chi_{3/2}. This yields the standard
   Koster labels Gamma6+/-, Gamma7+/-, Gamma8+/- for O_h and the E_{1/2}/E_{3/2}
   (+ parity) doublets for the subgroups, in ONE self-consistent convention.

The projector code (adapted_basis.py) needs ONLY the characters chi^lambda(g) for
every group element g -- provided here as `char_of_element[lambda][g_index]`.
"""
import numpy as np
from .group import build_Oh, little_group_indices, su2_cocycle, wigner_D, _chi_A2_of_O


# ---------------------------------------------------------------------------
#  Double little group as an abstract finite group with a multiplication table
# ---------------------------------------------------------------------------
class DoubleLittleGroup:
    """Elements are (oh_index, sign) with sign in {+1,-1}; the double group has
    2*|little group| elements. Element 0 is the identity (E,+1)."""

    def __init__(self, oh, kfrac, name=""):
        self.oh = oh
        self.name = name
        self.kfrac = np.asarray(kfrac, float)
        lg = little_group_indices(oh, kfrac)
        self.lg_idx = [i for (i, G0) in lg]          # indices into oh
        self.G0 = {i: G0 for (i, G0) in lg}          # umklapp per op
        # elements: (oh_index, sign)
        self.elements = [(i, +1) for i in self.lg_idx] + [(i, -1) for i in self.lg_idx]
        self.n = len(self.elements)
        self._index = {el: k for k, el in enumerate(self.elements)}
        # identity first
        eident = (self.lg_idx[0], +1)
        assert oh[self.lg_idx[0]]['det'] == 1 and np.array_equal(
            oh[self.lg_idx[0]]['R'], np.eye(3, dtype=np.int64)), "identity not first in lg"
        self._build_mult_table()
        self._build_classes()

    # ---- multiplication ----------------------------------------------------
    def mult(self, a, b):
        """product index of elements[a] * elements[b]."""
        (i, si), (j, sj) = self.elements[a], self.elements[b]
        eps, k = su2_cocycle(self.oh, i, j)          # D(Ri)D(Rj) = eps D(RiRj)
        return self._index[(k, si * sj * eps)]

    def _build_mult_table(self):
        n = self.n
        self.table = np.zeros((n, n), dtype=np.int64)
        for a in range(n):
            for b in range(n):
                self.table[a, b] = self.mult(a, b)
        # sanity: each row/col a permutation
        for a in range(n):
            assert len(set(self.table[a])) == n and len(set(self.table[:, a])) == n
        # inverses
        self.inv = np.zeros(n, dtype=np.int64)
        for a in range(n):
            self.inv[a] = int(np.where(self.table[a] == 0)[0][0])
        # E-bar (2*pi rotation): (identity op, -1)
        self.ebar = self._index[(self.lg_idx[0], -1)]

    def _build_classes(self):
        n = self.n
        seen = [False] * n
        classes = []
        for a in range(n):
            if seen[a]:
                continue
            orb = set()
            for g in range(n):
                orb.add(self.table[self.table[g, a], self.inv[g]])   # g a g^-1
            for x in orb:
                seen[x] = True
            classes.append(sorted(orb))
        self.classes = classes
        self.nclass = len(classes)
        self.class_of = np.zeros(n, dtype=np.int64)
        for ci, cl in enumerate(classes):
            for x in cl:
                self.class_of[x] = ci

    # ---- Burnside character table -----------------------------------------
    def character_table(self):
        """Return (chars, dims) where chars is (nirr, nclass) complex and dims is
        (nirr,) ints. Also caches per-ELEMENT characters (nirr, n)."""
        n, nc = self.n, self.nclass
        classes = self.classes
        # class-sum multiplication constants c_{r,s,t}: (sum over class r)(class s)
        # decomposed into classes. Build matrices M_r with (M_r)_{t,s}=c_{r,s,t}.
        # Representative-based: pick rep of class s, multiply by all of class r.
        Ms = []
        rep_s = [classes[s][0] for s in range(nc)]
        for r in range(nc):
            Mr = np.zeros((nc, nc))
            for s in range(nc):
                counts = np.zeros(nc)
                bs = rep_s[s]
                for a in classes[r]:
                    counts[self.class_of[self.table[a, bs]]] += 1
                Mr[:, s] = counts
            Ms.append(Mr)
        # Simultaneous eigenvectors of all M_r -> columns give irreducible
        # characters. Use a random combination for nondegenerate spectrum.
        rng = np.random.default_rng(0)
        Mcomb = sum(rng.standard_normal() * Ms[r] for r in range(nc))
        evals, V = np.linalg.eig(Mcomb)
        # columns of V are common eigenvectors; character ratios chi_r/chi_0 are
        # the eigenvalues of M_r on that eigenvector: M_r v = (|class_r| chi_r/chi_0) v
        class_sizes = np.array([len(c) for c in classes], float)
        chars = np.zeros((nc, nc), dtype=np.complex128)
        for a in range(nc):
            v = V[:, a]
            # eigenvalue of each M_r on v
            lam = np.array([(Ms[r] @ v)[np.argmax(np.abs(v))] / v[np.argmax(np.abs(v))]
                            for r in range(nc)])
            chi_over_chi0 = lam / class_sizes             # chi_r/chi_0
            # normalize: sum_r |class_r| |chi_r|^2 = |G|  => chi_0^2 * sum |c||ratio|^2=|G|
            denom = np.sum(class_sizes * np.abs(chi_over_chi0) ** 2)
            chi0 = np.sqrt(self.n / denom)
            chars[a] = chi0 * chi_over_chi0
        dims = np.rint(chars[:, 0].real).astype(int)
        # order by dim then something stable
        order = np.lexsort((chars[:, 1].real, dims))
        chars = chars[order]
        dims = dims[order]
        self.chars = chars
        self.dims = dims
        # per-element characters
        self.char_of_element = np.zeros((nc, n), dtype=np.complex128)
        for lam in range(nc):
            for g in range(n):
                self.char_of_element[lam, g] = chars[lam, self.class_of[g]]
        self._validate_orthogonality()
        return chars, dims

    def _validate_orthogonality(self, tol=1e-8):
        nc = self.nclass
        class_sizes = np.array([len(c) for c in self.classes], float)
        # row orthogonality: (1/|G|) sum_g chi_a(g) chi_b(g)* = delta_ab
        Gram = np.zeros((nc, nc), dtype=np.complex128)
        for a in range(nc):
            for b in range(nc):
                Gram[a, b] = np.sum(class_sizes * self.chars[a] * self.chars[b].conj()) / self.n
        assert np.allclose(Gram, np.eye(nc), atol=1e-6), \
            f"[{self.name}] character rows not orthonormal:\n{np.round(Gram,3)}"
        # dimension theorem
        assert abs(np.sum(self.dims ** 2) - self.n) < 1e-6, \
            f"[{self.name}] sum dim^2 = {np.sum(self.dims**2)} != |G|={self.n}"

    # ---- select & label the spinor (double-valued) irreps ------------------
    def spinor_irreps(self):
        """Return a list of dicts for the DOUBLE-VALUED irreps only:
            {'label', 'dim', 'parity' (+1/-1/None), 'char_elem' (len n complex)}.
        Labels assigned by structure (dim, parity, C3/C4 character), cross-checked
        against physical Wigner-D characters."""
        if not hasattr(self, 'chars'):
            self.character_table()
        # double-valued: chi(E-bar) = -dim
        spinor = [lam for lam in range(self.nclass)
                  if self.chars[lam, self.class_of[self.ebar]].real < 0]
        # parity: does the group contain inversion? find element (I,+1)
        inv_elem = self._find_inversion()
        results = []
        for lam in spinor:
            dim = int(self.dims[lam])
            parity = None
            if inv_elem is not None:
                # chi(inversion)/dim = +1 (g) or -1 (u) for these irreps
                cval = self.char_of_element[lam, inv_elem].real / dim
                parity = 1 if cval > 0 else -1
            results.append({'lam': lam, 'dim': dim, 'parity': parity,
                            'char_elem': self.char_of_element[lam].copy()})
        self._assign_labels(results)
        return results

    def corep_classify(self, spin_irreps, tol=1e-6):
        """Time-reversal CO-REPRESENTATION classification of the spinor irreps.

        The model Hamiltonian is time-reversal invariant with T^2 = -1 (spin-1/2),
        so extra Kramers degeneracies beyond the unitary-irrep dimension can occur.
        For these little groups the only such case is a COMPLEX (case-c) irrep,
        whose character is not real: it is inequivalent to its conjugate, and time
        reversal glues lambda and lambda* into ONE physically-degenerate co-rep of
        dimension dim(lambda)+dim(lambda*). (This is exactly the C_3v line Lambda,
        whose two 1-dim spinor irreps are a complex-conjugate pair.)

        Real-character spinor irreps are treated as case-a (physical degeneracy =
        irrep dim), which is correct for every spinor irrep of Gamma/R (O_h),
        X/M (D_4h), Sigma (C_2v) and T (C_4v): they are pseudoreal, and with T^2=-1
        pseudoreal irreps are NOT further doubled. The rare pseudoreal-DOUBLING
        case-b does not occur for these groups; if it ever did, it would surface as
        a degeneracy-spread warning at reference-freeze time.

        Returns {label: partner_label_or_None}; partner is set (symmetrically) for a
        complex-conjugate pair, None otherwise."""
        partner = {r['label']: None for r in spin_irreps}
        for r in spin_irreps:
            if partner[r['label']] is not None:
                continue
            ce = r['char_elem']
            if np.abs(ce.imag).max() < tol:
                continue                                   # real character -> case a
            for s in spin_irreps:
                if s['label'] == r['label'] or partner[s['label']] is not None:
                    continue
                if np.allclose(s['char_elem'], ce.conj(), atol=tol):
                    partner[r['label']] = s['label']
                    partner[s['label']] = r['label']
                    break
        return partner

    def _find_inversion(self):
        for k, (i, s) in enumerate(self.elements):
            if s == +1 and self.oh[i]['det'] == -1 and \
               np.array_equal(self.oh[i]['R'], -np.eye(3, dtype=np.int64)):
                return k
        return None

    def _physical_char(self, j, g):
        """chi of the pure-rotation Wigner-D^j on double-group element g, i.e.
        sign * trace(D^j(R_proper))."""
        i, s = self.elements[g]
        D = wigner_D(j, self.oh[i]['R'].astype(float))
        return s * np.trace(D)

    def _assign_labels(self, results):
        """Assign Koster-style labels using dimension + parity + rotation
        characters -- all STRUCTURAL invariants, never label strings.

        Two 2-dim spinor irreps (Gamma6 vs Gamma7 in O_h; E_{1/2} vs E_{3/2} in
        the D4/C4v subgroups) are separated by the character of the FOUR-fold
        rotation C4: the physical Gamma6 = D^{1/2} has chi(C4) = 2cos(45deg) =
        +sqrt(2); Gamma7 = Gamma6 (x) A2 has chi(C4) = -sqrt(2). (They do NOT
        differ on C3, where both give +1 -- a classic mislabeling trap.) We
        cross-check the sign against the physical Wigner-D^{1/2} character so the
        label is pinned to one convention by construction. Groups with a 4-dim
        spinor irrep are named O_h-style (Gamma6/7/8); the rest use E_{1/2},
        E_{3/2}. Groups with only a 3-fold axis (C_3v) have a 2-dim E_{1/2} plus
        a conjugate pair of 1-dim spinor irreps."""
        c3 = self._find_rotation_order(3)
        c4 = self._find_rotation_order(4)
        has_dim4 = any(r['dim'] == 4 for r in results)
        for r in results:
            dim, par = r['dim'], r['parity']
            suf = '' if par is None else ('+' if par == 1 else '-')
            if dim == 4:
                r['label'] = f"Gamma8{suf}"
            elif dim == 2:
                if c4 is not None:
                    chi_c4 = r['char_elem'][c4].real          # +sqrt2 (6) or -sqrt2 (7)
                    phys = (self._physical_char(0.5, c4)).real  # = +sqrt2 for D^{1/2}
                    is_g6 = (chi_c4 * phys > 0)                 # matches D^{1/2}?
                    if has_dim4:
                        r['label'] = ("Gamma6" if is_g6 else "Gamma7") + suf
                    else:
                        r['label'] = ("E1/2" if is_g6 else "E3/2") + suf
                else:
                    # only a 3-fold axis (C_3v): single 2-dim spinor irrep E_{1/2}
                    r['label'] = "E1/2" + suf
            elif dim == 1:
                # C_3v conjugate pair: distinguish by sign of Im chi(C3)
                tag = 'a'
                if c3 is not None and abs(r['char_elem'][c3].imag) > 1e-6:
                    tag = 'a' if r['char_elem'][c3].imag > 0 else 'b'
                r['label'] = f"E1dim_{tag}{suf}"
            else:
                r['label'] = f"dim{dim}{suf}"
        # disambiguate accidental duplicates deterministically
        seen = {}
        for r in results:
            base = r['label']
            if base in seen:
                seen[base] += 1
                r['label'] = f"{base}#{seen[base]}"
            else:
                seen[base] = 0

    def _find_rotation_order(self, order):
        """Return a double-group element index that is a PROPER rotation of the
        given order with sign +1, or None."""
        for k, (i, s) in enumerate(self.elements):
            if s != +1 or self.oh[i]['det'] != 1:
                continue
            R = self.oh[i]['R']
            if np.array_equal(R, np.eye(3, dtype=np.int64)):
                continue
            # order of R
            P = R.copy()
            o = 1
            while not np.array_equal(P, np.eye(3, dtype=np.int64)) and o <= 6:
                P = P @ R
                o += 1
            if o == order:
                return k
        return None
