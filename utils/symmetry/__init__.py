"""
Symmetry-adapted (double-group) basis for the spinor pseudopotential Hamiltonian.

Setup (run once, cached) vs hot path (every training step) are kept in separate
modules per the design requirement:

  SETUP:    group.py        - O_h ops, SU(2)/Wigner-D, double group, little groups
            irreps.py       - auto double-group character tables (Burnside) + labels
            adapted_basis.py- sparse symmetry-adapted plane-wave-spinor basis B_lambda
            interface.py    - bridge to ham/BulkSystem; build_sak_for_k
            characters.py   - group action + character-of-subspace (validation/QE)
            loss.py         - reference labeling + sector-loss builders (setup part)
            validate.py     - the validation test suite

  HOT PATH: hot_path.py     - block_eigvals(H, sak): {irrep: eigenvalues}, no eigvecs
            loss.py         - sector_loss_at_k (differentiable)

See module headers for the (heavily documented) convention choices.
"""
from .group import build_Oh, little_group_indices
from .irreps import DoubleLittleGroup
from .adapted_basis import build_symmetry_adapted_k, SymmetryAdaptedK
from .interface import build_sak_for_k, HIGH_SYM_POINTS
from .hot_path import block_eigvals, irrep_sequence
from .loss import (freeze_reference_labels, reference_sector_energies,
                   assert_sector_counts, sector_loss_at_k)

__all__ = [
    'build_Oh', 'little_group_indices', 'DoubleLittleGroup',
    'build_symmetry_adapted_k', 'SymmetryAdaptedK', 'build_sak_for_k',
    'HIGH_SYM_POINTS', 'block_eigvals', 'irrep_sequence',
    'freeze_reference_labels', 'reference_sector_energies',
    'assert_sector_counts', 'sector_loss_at_k',
]
