"""Bridge between the DeePseudopot Hamiltonian/BulkSystem objects and the
symmetry module: extract the integer Miller indices of the G-list and build the
SAPW basis at a k-point (given in FRACTIONAL reciprocal / crystal coordinates)."""
import numpy as np

from .group import build_Oh
from .adapted_basis import build_symmetry_adapted_k


def mill_index_from_basis(basis_cart, Grecip):
    """basis_cart: (nbv,3) Cartesian G-list (1/Bohr). Grecip: rows b_1,b_2,b_3
    (Cartesian). Return integer Miller indices mill (G = mill @ Grecip) and the
    hash {(m0,m1,m2): iG}."""
    Binv = np.linalg.inv(np.asarray(Grecip, float))
    millf = np.asarray(basis_cart, float) @ Binv
    mill = np.rint(millf).astype(np.int64)
    err = np.abs(millf - mill).max()
    assert err < 1e-6, f"G-list not on integer reciprocal lattice (max dev {err})"
    index_of = {tuple(int(x) for x in m): i for i, m in enumerate(mill)}
    assert len(index_of) == mill.shape[0], "duplicate G integer triples"
    return mill, index_of


def build_sak_for_k(ham, kfrac, name="", oh=None, device=None):
    """Build the SymmetryAdaptedK for a Hamiltonian `ham` at fractional-recip k.
    Uses ham.basis (Cartesian G-list) and ham.system.getGVectors()."""
    import torch
    basis = ham.basis.detach().cpu().numpy()
    Grecip = ham.system.getGVectors().detach().cpu().numpy()
    mill, index_of = mill_index_from_basis(basis, Grecip)
    if oh is None:
        oh = build_Oh()
    return build_symmetry_adapted_k(oh, kfrac, mill, index_of, basis.shape[0],
                                    name=name, device=device, dtype=torch.complex128)


# Standard Pm-3m high-symmetry points in fractional reciprocal coordinates.
HIGH_SYM_POINTS = {
    'Gamma': [0.0, 0.0, 0.0],
    'R':     [0.5, 0.5, 0.5],
    'X':     [0.5, 0.0, 0.0],
    'M':     [0.5, 0.5, 0.0],
}
