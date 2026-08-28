"""
Double-group machinery for the cubic Pm-3m (O_h) point group and its little-group
subgroups, for a spinor (SOC) pseudopotential Hamiltonian.

=========================  CONVENTIONS (read me)  =========================
SPACE GROUP / ORIGIN
  Pm-3m (#221), symmorphic. In THIS code's structure (system_0.par) the origin
  is on the Pb site: Pb (0,0,0), Cs (1/2,1/2,1/2), X at (1/2,0,0)+perms. The
  point group O_h is centered on Pb. There are NO fractional translations at
  these Wyckoff sites, but every operation carries a translation vector (all
  zero here) so the data structure generalizes to nonsymmorphic settings.
  (NOTE: the source prompt used the opposite origin, Cs at 0; we follow the
  actual input files.)

POINT-GROUP MATRICES: CRYSTAL vs CARTESIAN
  Operations are stored as INTEGER 3x3 matrices acting on CRYSTAL (fractional)
  coordinates: a fractional vector f transforms as f -> R_crys @ f, and a
  reciprocal (Miller) triple m (G = m @ b, b = reciprocal rows) transforms the
  SAME way, m -> R_crys @ m, because the lattice here is simple cubic so the
  reciprocal lattice is also simple cubic. For a simple-cubic lattice the
  reciprocal basis is b = (2*pi/a) * I, hence the CARTESIAN action equals the
  crystal action numerically: R_cart == R_crys (an integer signed-permutation
  matrix). We use that identity (asserted in build_Oh) to get the Cartesian
  rotation needed for the SU(2)/Wigner-D spin rep. For a non-cubic cell this
  identity would break and R_cart = A^T R_crys A^{-T} would be needed.

ACTIVE rotations. D^{j}(R) rotates kets actively: D^j(R) = exp(-i * theta * n.J),
  n the Cartesian rotation axis (unit), theta the angle, J the spin-j generators.

SPINOR COMPONENT ORDERING: |up> = (1,0) = m_s=+1/2, |down> = (0,1) = m_s=-1/2.
  sigma_z = diag(+1,-1). This matches ham.py (kin.repeat(2): up block first;
  SOC up-up block carries +1/2 g_z). Wigner-D basis is ordered m = +j, ..., -j,
  so for j=1/2 the 2x2 D^{1/2} rows/cols are (up, down) -- consistent.

IMPROPER OPERATIONS: spatial inversion I acts TRIVIALLY on spin, D^{1/2}(I)=Id.
  So for an improper op R (det R = -1) the spin rep uses its proper part
  R_proper = -R (a pure rotation): D^{j}(R) = D^{j}(R_proper). Spatial parity
  (g/u) is carried SEPARATELY as an irrep label, not in the rotation matrix.

DOUBLE GROUP: each spatial op R lifts to two elements (R,+1) and (R,-1) of the
  double group; the label s in {+1,-1} is the SU(2) sign, with the 2*pi rotation
  E-bar = (E,-1). |double group| = 2 * |point group|.
===========================================================================
"""
import itertools
import numpy as np

# ---- Pauli matrices, |up>=(1,0), |down>=(0,1) ----------------------------
SIGMA = np.array([
    [[0, 1], [1, 0]],          # sigma_x
    [[0, -1j], [1j, 0]],       # sigma_y
    [[1, 0], [0, -1]],         # sigma_z
], dtype=np.complex128)


# ============================================================================
#  Spin-j generators and Wigner-D matrices (ACTIVE, basis m = +j..-j)
# ============================================================================
def spin_matrices(j):
    """Angular-momentum matrices (Jx, Jy, Jz) for spin j, in the |j,m> basis
    ordered m = +j, +j-1, ..., -j. Returns a (3, d, d) complex array, d=2j+1."""
    d = int(round(2 * j + 1))
    ms = np.array([j - k for k in range(d)])   # +j .. -j
    Jz = np.diag(ms).astype(np.complex128)
    Jp = np.zeros((d, d), dtype=np.complex128)  # raising: J+ |m> = sqrt(j(j+1)-m(m+1)) |m+1>
    for col in range(d):
        m = ms[col]
        if col - 1 >= 0:                        # target row has m+1 (one slot up)
            Jp[col - 1, col] = np.sqrt(j * (j + 1) - m * (m + 1))
    Jm = Jp.conj().T
    Jx = 0.5 * (Jp + Jm)
    Jy = (Jp - Jm) / (2j)
    return np.array([Jx, Jy, Jz])


def _rotation_axis_angle(Rcart):
    """Axis (unit, Cartesian) and angle for a PROPER rotation matrix (det=+1).
    Handles the theta=0 and theta=pi special cases robustly."""
    Rcart = np.asarray(Rcart, dtype=float)
    cos_t = (np.trace(Rcart) - 1.0) / 2.0
    cos_t = max(-1.0, min(1.0, cos_t))
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0        # identity: axis irrelevant
    if abs(theta - np.pi) < 1e-8:
        # R = 2 n n^T - I  => n n^T = (R + I)/2; take the largest diagonal for stability
        A = (Rcart + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        n = A[:, k] / np.sqrt(A[k, k])
        n = n / np.linalg.norm(n)
        return n, np.pi
    w = np.array([Rcart[2, 1] - Rcart[1, 2],
                  Rcart[0, 2] - Rcart[2, 0],
                  Rcart[1, 0] - Rcart[0, 1]])          # = 2 sin(theta) * n
    n = w / (2.0 * np.sin(theta))
    return n / np.linalg.norm(n), theta


def wigner_D(j, Rcart):
    """Active Wigner-D matrix D^j(R) = exp(-i theta n.J) for the ROTATION part.
    Rcart may be proper or improper; the improper case uses R_proper = -R
    (inversion is trivial on spin/angular rotation). Basis m = +j..-j."""
    Rcart = np.asarray(Rcart, dtype=float)
    det = round(np.linalg.det(Rcart))
    Rproper = Rcart if det > 0 else -Rcart
    n, theta = _rotation_axis_angle(Rproper)
    J = spin_matrices(j)
    nJ = n[0] * J[0] + n[1] * J[1] + n[2] * J[2]
    # matrix exponential via eigendecomposition of the Hermitian n.J
    evals, evecs = np.linalg.eigh(nJ)
    D = evecs @ np.diag(np.exp(-1j * theta * evals)) @ evecs.conj().T
    return D


def su2(Rcart):
    """SU(2) spin-1/2 rep D^{1/2}(R). 2x2. Improper -> proper part (see above)."""
    return wigner_D(0.5, Rcart)


# ============================================================================
#  Build O_h (48 ops) as integer crystal matrices (= Cartesian for cubic)
# ============================================================================
def build_Oh():
    """Return a list of 48 dicts, each:
        {'R': int 3x3 crystal matrix (== Cartesian here),
         't': length-3 translation (zeros; carried for generality),
         'det': +1/-1,
         'su2': 2x2 SU(2) matrix D^{1/2}(R),
         'chiA2': +/-1 the O single-group A2 (=sign rep of S4 on body diagonals)}.
    Ordered with the identity first."""
    ops = []
    seen = set()
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3), dtype=np.int64)
        for r, c in enumerate(perm):
            P[r, c] = 1
        for signs in itertools.product((1, -1), repeat=3):
            R = P * np.array(signs, dtype=np.int64)[None, :]
            key = tuple(R.flatten())
            if key in seen:
                continue
            seen.add(key)
            ops.append(R)
    assert len(ops) == 48, f"expected 48 O_h ops, got {len(ops)}"
    # put identity first
    ops.sort(key=lambda R: (0 if np.array_equal(R, np.eye(3, dtype=np.int64)) else 1))

    out = []
    for R in ops:
        det = int(round(np.linalg.det(R)))
        # For simple cubic, Cartesian action == crystal action (integer matrix).
        out.append({
            'R': R,
            't': np.zeros(3),
            'det': det,
            'su2': su2(R.astype(float)),
            'chiA2': _chi_A2_of_O(R),
        })
    return out


def _chi_A2_of_O(R):
    """O single-group A2 character = sign of the permutation R induces on the
    four cubic body-diagonal AXES (O ~= S4 acting on body diagonals; A2 = sign
    rep). Improper ops are reduced to their proper part first (A2 is a rotation-
    group label; the g/u parity is separate)."""
    R = np.asarray(R, dtype=np.int64)
    if round(np.linalg.det(R)) < 0:
        R = -R
    diags = [np.array(v) for v in [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]]
    perm = []
    for d in diags:
        Rd = R @ d
        found = None
        for k, e in enumerate(diags):
            if np.array_equal(Rd, e) or np.array_equal(Rd, -e):
                found = k
                break
        assert found is not None, "body-diagonal not permuted -- not a cubic op?"
        perm.append(found)
    # sign of permutation
    perm = list(perm)
    sign = 1
    visited = [False] * 4
    for i in range(4):
        if visited[i]:
            continue
        L = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            L += 1
        if L % 2 == 0:
            sign = -sign
    return sign


# ============================================================================
#  Little group of k and its double group
# ============================================================================
def little_group_indices(oh, kfrac, tol=1e-9):
    """Indices (into oh) of ops R with R k = k + G0, G0 integer (crystal recip).
    Returns list of (op_index, G0) with G0 an int length-3 array (the umklapp)."""
    k = np.asarray(kfrac, dtype=float)
    out = []
    for idx, op in enumerate(oh):
        dk = op['R'] @ k - k
        G0 = np.rint(dk)
        if np.abs(dk - G0).max() < tol:
            out.append((idx, G0.astype(np.int64)))
    return out


def su2_cocycle(oh, i, jdx):
    """Sign eps in D^{1/2}(R_i) D^{1/2}(R_j) = eps * D^{1/2}(R_i R_j), eps=+/-1.
    (The double-group multiplication rule; R_i R_j is another O_h op.)"""
    Ri, Rj = oh[i]['R'], oh[jdx]['R']
    Rprod = Ri @ Rj
    prod_su2 = oh[i]['su2'] @ oh[jdx]['su2']
    # find index of Rprod
    for k, op in enumerate(oh):
        if np.array_equal(op['R'], Rprod):
            ratio = prod_su2 @ np.linalg.inv(op['su2'])
            # ratio should be +Id or -Id
            s = ratio[0, 0].real
            assert np.allclose(ratio, s * np.eye(2), atol=1e-8), "SU(2) not projective +/-1"
            return int(round(s)), k
    raise RuntimeError("product op not found in O_h (group not closed)")
