"""
G2 Behler-Parrinello descriptor for nanocrystals (non-periodic).

Usage:
    python compute_G2_nanocrystal.py conf.par

The conf.par file format:
    N_atoms
    Symbol x y z
    Symbol x y z
    ...

Returns:
    (N_atoms,) numpy array with G2 values (also printed to stdout).
"""

import sys
import numpy as np
import torch
from .local_structure_correction import compute_Cs_descriptors_nano, compute_Pb_descriptors_nano, compute_I_descriptors_nano

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_conf_par(filepath: str):
    """Parse a conf.par file.

    Returns
    -------
    symbols : list[str]   – element symbols, length N
    positions : np.ndarray – shape (N, 3), Cartesian coordinates
    """
    with open(filepath) as fh:
        lines = [l.strip() for l in fh if l.strip()]

    n_atoms = int(lines[0])
    symbols = []
    positions = []

    for line in lines[1:n_atoms + 1]:
        parts = line.split()
        symbols.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(positions) != n_atoms:
        raise ValueError(
            f"Header says {n_atoms} atoms but {len(positions)} coordinate "
            "lines were found."
        )

    return symbols, np.array(positions, dtype=float)


# ---------------------------------------------------------------------------
# Cutoff function  (Behler-Parrinello cosine cutoff)
# ---------------------------------------------------------------------------

def cutoff_fc(dist: np.ndarray, Rc: float) -> np.ndarray:
    """Smooth cosine cutoff function.

    f_c(r) = 0.5 * [cos(pi * r / Rc) + 1]  for r <= Rc
           = 0                               for r > Rc

    Parameters
    ----------
    dist : (N, N) array of pairwise distances
    Rc   : cutoff radius

    Returns
    -------
    fc : (N, N) array
    """
    fc = np.where(
        dist <= Rc,
        0.5 * (np.cos(np.pi * dist / Rc) + 1.0),
        0.0,
    )
    return fc


# ---------------------------------------------------------------------------
# Pairwise distances (no periodic images for a nanocrystal)
# ---------------------------------------------------------------------------

def compute_distances(positions: np.ndarray):
    """Compute all pairwise displacement vectors and distances.

    Parameters
    ----------
    positions : (N, 3)

    Returns
    -------
    dR   : (N, N, 3) – displacement vectors  r_j - r_i
    dist : (N, N)    – Euclidean distances
    """
    # dR[i,j] = positions[j] - positions[i]
    dR = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # (N,N,3)
    dist = np.linalg.norm(dR, axis=-1)                               # (N,N)
    return dR, dist


# ---------------------------------------------------------------------------
# Equilibrium / reference distance Rs  (CsPbX3 perovskite lookup table)
# All values in Bohr.
# ---------------------------------------------------------------------------

def retEquilDist(atom1: str, atom2: str, material: str) -> float:
    """Return the equilibrium interatomic distance (Bohr) for a pair of
    element symbols in a given CsPbX3 perovskite material.

    Parameters
    ----------
    atom1, atom2 : element symbols (order-insensitive)
    material     : one of 'CsPbI3', 'CsPbBr3', 'CsPbCl3'

    Returns
    -------
    Equilibrium distance in Bohr.
    """
    if atom1 == "I" and atom2 == "I":
        return 8.40493990176          # I-I in CsPbI3
    if atom1 == "Br" and atom2 == "Br":
        return 7.8437163206           # Br-Br in CsPbBr3
    if atom1 == "Cl" and atom2 == "Cl":
        return 7.48961492225          # Cl-Cl in CsPbCl3

    if set([atom1, atom2]) == {"Pb", "I"}:
        return 5.94319                # Pb-I
    if set([atom1, atom2]) == {"Pb", "Br"}:
        return 5.546345               # Pb-Br
    if set([atom1, atom2]) == {"Pb", "Cl"}:
        return 5.2959575              # Pb-Cl

    if set([atom1, atom2]) == {"I", "Cs"}:
        return 8.40493990176          # I-Cs
    if set([atom1, atom2]) == {"Br", "Cs"}:
        return 7.8437163206           # Br-Cs
    if set([atom1, atom2]) == {"Cl", "Cs"}:
        return 7.48961492225          # Cl-Cs

    if set([atom1, atom2]) == {"Pb", "Cs"}:
        if material == "CsPbI3":
            return 10.293907039
        if material == "CsPbBr3":
            return 9.60655133631
        if material == "CsPbCl3":
            return 9.17286746472

    if atom1 == "Pb" and atom2 == "Pb":
        if material == "CsPbI3":
            return 11.88638
        if material == "CsPbBr3":
            return 11.09269
        if material == "CsPbCl3":
            return 10.591915

    if atom1 == "Cs" and atom2 == "Cs":
        if material == "CsPbI3":
            return 11.88638
        if material == "CsPbBr3":
            return 11.09269
        if material == "CsPbCl3":
            return 10.591915

    print(f"Undefined atom pair in retEquilDist: {atom1} - {atom2}")
    raise ValueError(f"No equilibrium distance defined for {atom1}-{atom2} in {material}")


def computeEquilDist(symbols: list, material: str) -> np.ndarray:
    """Build the (N, N) Rs matrix using the perovskite lookup table.

    Fully vectorized: unique element pairs are looked up once into a small
    (n_unique x n_unique) table, then the full (N, N) matrix is filled via
    NumPy integer indexing — O(n_unique^2) scalar calls regardless of N.
    For CsPbX3 that is at most 16 calls instead of N^2.

    Parameters
    ----------
    symbols  : list[str] of length N  – element symbols
    material : 'CsPbI3' | 'CsPbBr3' | 'CsPbCl3'

    Returns
    -------
    Rs : (N, N) numpy array  [Bohr]
    """
    # Map each unique element to a compact integer index
    unique_elems = list(dict.fromkeys(symbols))           # dedup, preserve order
    elem_to_idx  = {e: i for i, e in enumerate(unique_elems)}
    n_unique     = len(unique_elems)

    # Build a small (n_unique, n_unique) table — at most 4x4 for CsPbX3
    lookup = np.zeros((n_unique, n_unique), dtype=float)
    for i, e1 in enumerate(unique_elems):
        for j, e2 in enumerate(unique_elems):
            lookup[i, j] = retEquilDist(e1, e2, material)

    # Map full symbol list to indices, then broadcast-index into lookup
    idx = np.array([elem_to_idx[s] for s in symbols], dtype=int)  # (N,)
    Rs  = lookup[np.ix_(idx, idx)]                                 # (N, N)

    return Rs

def compute_surface_mask(positions: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Identify surface atoms as those within `threshold` Bohr of the extrema
    in any Cartesian dimension. Handles concave surfaces correctly.

    Parameters
    ----------
    positions : (N, 3) array of Cartesian coordinates [Bohr]
    threshold : float, distance from extrema to consider surface [Bohr]

    Returns
    -------
    surface_mask : (N,) bool array, True for surface atoms
    """
    positions    = np.asarray(positions)
    
    x_min = positions[:, 0].min() + threshold
    x_max = positions[:, 0].max() - threshold
    y_min = positions[:, 1].min() + threshold
    y_max = positions[:, 1].max() - threshold
    z_min = positions[:, 2].min() + threshold
    z_max = positions[:, 2].max() - threshold

    surface_list = []
    for coord in positions:
        x = coord[0]
        y = coord[1]
        z = coord[2]

        if (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max) | (z < z_min) | (z > z_max):
            surface_list.append(1)
        else:
            surface_list.append(0)

    surface_mask = np.array(surface_list, dtype=bool)
    return surface_mask

# ---------------------------------------------------------------------------
# Main G2 computation
# ---------------------------------------------------------------------------

def compute_nanocrystal_descriptors(
    symbols:   list,
    positions: np.ndarray,
    material:  str,
) -> tuple[dict, dict]:
    """
    Compute per-atom structural descriptors for a nanocrystal (no periodic boundary).

    Surface atoms are identified via extrema condition and assigned cubic reference
    descriptor values (all zeros for normalized descriptors).

    Parameters
    ----------
    symbols   : list[str] of length N
    positions : (N, 3) array of Cartesian coordinates [Bohr]
    material  : 'CsPbI3' | 'CsPbBr3' | 'CsPbCl3'

    Returns
    -------
    descriptors  : dict with keys 'I' (or 'Br'/'Cl'), 'Pb', 'Cs'
                   Each value is a (N_species, n_descriptors) tensor.
    atom_indices : dict with keys 'I' (or 'Br'/'Cl'), 'Pb', 'Cs'
                   Each value is a (N_species,) tensor of atom indices.
    """
    positions = np.asarray(positions, dtype=float)
    n_atoms   = positions.shape[0]

    # ------------------------------------------------------------------
    # Surface detection
    # ------------------------------------------------------------------
    surface_mask = compute_surface_mask(positions, threshold=1.5)
    n_surface    = surface_mask.sum()
    print(f"Surface atoms: {n_surface} / {n_atoms}")
    print(f"surface mask = {surface_mask}")
    # ------------------------------------------------------------------
    # Cubic reference descriptor values (assigned to surface atoms)
    # ------------------------------------------------------------------
    cubic_refs = {
        'I':  np.zeros(2),
        'Pb': np.array([0.987559, 0.0, 0.0, 0.0, 0.0]),
        'Cs': np.zeros(3),
    }

    halide = [t for t in set(symbols) if t not in ('Cs', 'Pb')][0]

    # ------------------------------------------------------------------
    # Compute descriptors per atom
    # ------------------------------------------------------------------
    descriptors  = {'I': [], 'Pb': [], 'Cs': []}
    atom_indices = {'I': [], 'Pb': [], 'Cs': []}

    for i, atype in enumerate(symbols):

        species = 'I' if atype == halide else atype

        if surface_mask[i]:
            d = cubic_refs[species].copy()
            print(f"  atom {i:4d} ({atype}): surface -> cubic ref {d}")
        else:
            if atype == halide:
                d = compute_I_descriptors_nano (i, symbols, positions, material)
            elif atype == 'Pb':
                d = compute_Pb_descriptors_nano(i, symbols, positions, material)
            elif atype == 'Cs':
                d = compute_Cs_descriptors_nano(i, symbols, positions, material)

        descriptors [species].append(d)
        atom_indices[species].append(i)

    descriptors  = {k: torch.tensor(np.array(v)) for k, v in descriptors.items()}
    atom_indices = {k: torch.tensor(np.array(v)) for k, v in atom_indices.items()}

    for k, v in descriptors.items():
        n_surf = int(surface_mask[atom_indices[k].numpy()].sum())
        print(f"\n{k}: {v.shape}  ({n_surf} surface atoms at cubic ref)\n{v}")

    return descriptors, atom_indices



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python compute_G2_nanocrystal.py <conf.par> <material>")
        print("  material: CsPbI3 | CsPbBr3 | CsPbCl3")
        sys.exit(1)

    filepath = sys.argv[1]
    material = sys.argv[2]
    symbols, positions = read_conf_par(filepath)

    print(f"Read {len(symbols)} atoms from '{filepath}'  (material: {material})")

    descriptors, atom_indices = compute_nanocrystal_descriptors(symbols, positions, material)

    for k, v in descriptors.items():
        print(f"\n{k} descriptors: {v.shape}\n{v}")

    return descriptors, atom_indices


if __name__ == "__main__":
    main()