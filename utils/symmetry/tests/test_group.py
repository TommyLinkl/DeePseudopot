"""Self-tests for utils/symmetry/group.py -- run directly:
   PYTHONPATH=<DeePseudopot> python utils/symmetry/tests/test_group.py
"""
import numpy as np
from utils.symmetry.group import (
    build_Oh, little_group_indices, su2, wigner_D, spin_matrices,
    su2_cocycle, _chi_A2_of_O,
)


def test_oh_basic():
    oh = build_Oh()
    assert len(oh) == 48
    assert np.array_equal(oh[0]['R'], np.eye(3, dtype=np.int64)), "identity not first"
    dets = [op['det'] for op in oh]
    assert dets.count(1) == 24 and dets.count(-1) == 24
    # closure of the 48 integer matrices
    keys = {tuple(op['R'].flatten()) for op in oh}
    for a in oh:
        for b in oh:
            assert tuple((a['R'] @ b['R']).flatten()) in keys
    print("  [ok] O_h: 48 ops, 24/24 proper/improper, closed under multiplication")


def test_chiA2_is_1d_rep():
    oh = build_Oh()
    # chiA2 must be a homomorphism to {+1,-1} (proper part), 1-dim rep of O
    for a in oh:
        for b in oh:
            Rab = a['R'] @ b['R']
            assert _chi_A2_of_O(Rab) == _chi_A2_of_O(a['R']) * _chi_A2_of_O(b['R'])
    # A2 is nontrivial: some ops give -1
    vals = [_chi_A2_of_O(op['R']) for op in oh]
    assert set(vals) == {1, -1}
    print("  [ok] chi_A2: valid nontrivial 1-dim rep of O (sign rep of S4)")


def test_su2_projective_homomorphism():
    oh = build_Oh()
    # D^{1/2}(Ri) D^{1/2}(Rj) = +/- D^{1/2}(Ri Rj); collect the cocycle and check
    # E-bar structure: there must exist ops whose product carries -1.
    saw_minus = False
    for i in range(len(oh)):
        for j in range(len(oh)):
            s, k = su2_cocycle(oh, i, j)
            if s == -1:
                saw_minus = True
    assert saw_minus, "SU(2) cocycle never -1 -- double group would be trivial"
    # unitarity + determinant 1 (proper part) of each SU(2)
    for op in oh:
        U = op['su2']
        assert np.allclose(U.conj().T @ U, np.eye(2), atol=1e-10)
        assert abs(abs(np.linalg.det(U)) - 1.0) < 1e-10
    print("  [ok] SU(2): unitary, projective homomorphism with genuine -1 cocycle")


def test_su2_improper_equals_proper():
    oh = build_Oh()
    inv = -np.eye(3)
    D_inv = su2(inv)
    assert np.allclose(D_inv, np.eye(2), atol=1e-12), "D^1/2(I) must be identity"
    # improper op R = I * (-R): su2(R) == su2(-R)
    for op in oh:
        if op['det'] < 0:
            assert np.allclose(su2(op['R'].astype(float)), su2(-op['R'].astype(float)), atol=1e-12)
    print("  [ok] SU(2): inversion trivial on spin, improper = proper part")


def test_wigner_j32_homomorphism():
    oh = build_Oh()
    # D^{3/2} must be a projective rep too (rotation part); check on proper ops
    for i in range(len(oh)):
        for j in range(len(oh)):
            Ri, Rj = oh[i]['R'], oh[j]['R']
            if oh[i]['det'] < 0 or oh[j]['det'] < 0:
                continue
            Di, Dj = wigner_D(1.5, Ri.astype(float)), wigner_D(1.5, Rj.astype(float))
            Rprod = (Ri @ Rj).astype(float)
            Dprod = wigner_D(1.5, Rprod)
            ratio = (Di @ Dj) @ np.linalg.inv(Dprod)
            s = ratio[0, 0]
            assert np.allclose(ratio, s * np.eye(4), atol=1e-8), "j=3/2 not projective scalar"
            assert abs(abs(s) - 1.0) < 1e-8
    # unitarity
    for op in oh:
        D = wigner_D(1.5, op['R'].astype(float))
        assert np.allclose(D.conj().T @ D, np.eye(4), atol=1e-9)
    print("  [ok] Wigner D^{3/2}: unitary, projective homomorphism (Gamma8 rotation part)")


def test_little_groups():
    oh = build_Oh()
    expected = {
        'Gamma': ([0, 0, 0], 48),
        'R':     ([0.5, 0.5, 0.5], 48),
        'X':     ([0.5, 0.0, 0.0], 16),
        'M':     ([0.5, 0.5, 0.0], 16),
        'Lambda':([0.3, 0.3, 0.3], 6),
        'Sigma': ([0.3, 0.3, 0.0], 4),
        'T':     ([0.5, 0.5, 0.3], 8),
    }
    for name, (k, n) in expected.items():
        lg = little_group_indices(oh, k)
        assert len(lg) == n, f"{name}: |LG|={len(lg)} expected {n}"
    print("  [ok] little groups: Gamma/R=48, X/M=16, Lambda=6, Sigma=4, T=8")


if __name__ == "__main__":
    print("group.py self-tests:")
    test_oh_basic()
    test_chiA2_is_1d_rep()
    test_su2_projective_homomorphism()
    test_su2_improper_equals_proper()
    test_wigner_j32_homomorphism()
    test_little_groups()
    print("ALL group.py TESTS PASSED")
