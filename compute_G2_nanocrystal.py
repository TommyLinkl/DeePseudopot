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


# ---------------------------------------------------------------------------
# Main G2 computation
# ---------------------------------------------------------------------------

def compute_G2(
    symbols: list,
    positions: np.ndarray,
    material: str,
    eta: float = 0.5
) -> np.ndarray:
    """Compute the G2 Behler-Parrinello descriptor for every atom.

    G2[alpha] = sum_{j != alpha} exp(-eta * (r_{alpha j} - Rs_{alpha j})^2)
                                  * f_c(r_{alpha j})

    Note: following the pseudocode convention, a constant –1 shift is applied
    inside the Gaussian term  (exp_term - 1), matching the original reference
    implementation provided.

    Parameters
    ----------
    symbols   : list[str] of length N
    positions : (N, 3) array of Cartesian coordinates  [Bohr]
    material  : 'CsPbI3' | 'CsPbBr3' | 'CsPbCl3'
    eta       : Gaussian width parameter  [Bohr^{-2}]
    
    Returns
    -------
    G2 : (N,) numpy array
    """
    positions = np.asarray(positions, dtype=float)
    n_atoms = positions.shape[0]

    # Pairwise distances (no periodic images)
    _, dist = compute_distances(positions)         # (N,N)

    # Reference distances Rs  [Bohr]
    Rs = computeEquilDist(symbols, material)       # (N,N)

    # Per-species parameters
    Rc_map = {
        'CsPbI3':  {'Cs': 8.5, 'Pb': 7.0, 'I':  7.0},
        'CsPbBr3': {'Cs': 8.2, 'Pb': 6.5, 'Br': 6.5},
        'CsPbCl3': {'Cs': 7.8, 'Pb': 6.0, 'Cl': 6.0},
    }
    
    Rc_per_atom = np.array([Rc_map[material][t] for t in symbols])   # (N,)
    
    # Per-atom cutoff: fc[alpha, beta] uses Rc of alpha
    fc  = np.zeros((n_atoms, n_atoms))
    fcp = np.zeros((n_atoms, n_atoms))
    for alpha in range(n_atoms):
        fc [alpha] = cutoff_fc      (dist[alpha], Rc_per_atom[alpha])
    
    # Exclude self-interaction: set Rs diagonal to 0 (dist diagonal is 0 too,
    # but fc(0) ≠ 0 in general, so we explicitly zero out the diagonal below)
    np.fill_diagonal(Rs, 0.0)
    np.fill_diagonal(fc, 0.0)

    neighbor_mask = (fc > 0) & (dist > 1e-12)       # (N,N) bool
    N_neighbors = neighbor_mask.sum(axis=1)          # (N,)  int
    N_neighbors = np.maximum(N_neighbors, 1)         # guard against isolated atoms
    for i in range(len(symbols)):
        print(f"{symbols[i]} {N_neighbors[i]}")

    # Gaussian term
    exp_term = np.exp(-eta * (dist - Rs) ** 2)    # (N,N)

    # Compute G2
    S  = np.sum((exp_term - 1.0) * fc, axis=1)

    # G2 = S / Z_per_atom                                       # (N,)
    G2 = np.log(np.abs(S) + 1e-8) / N_neighbors 
    
    return G2


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

    G2 = compute_G2(symbols, positions, material)

    print("\nG2 descriptor (one value per atom):")
    for i, (sym, val) in enumerate(zip(symbols, G2)):
        print(f"  Atom {i:4d}  {sym:3s}  G2 = {val: .6f}")

    print(f"\nG2 array shape : {G2.shape}")
    print(f"G2 min / max   : {G2.min():.6f} / {G2.max():.6f}")

    return G2


if __name__ == "__main__":
    main()