import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from .constants import *

torch.set_default_dtype(torch.float64)

import numpy as np

# =============================================================================
# Reference values (orthorhombic phase)
# =============================================================================

ORTHO_REF = {
    'CsPbI3': {
        'I': {
            'angle_axial':      160.6,    # Pb-I-Pb axial (degrees)
            'angle_equatorial': 150.8,    # Pb-I-Pb equatorial (degrees)
            'angle_ref':        150.8,    # normalization reference (most distorted)
        },
        'Pb': {
            'bond_axial':       5.94840,  # Bohr
            'bond_equatorial':  6.087721,  # Bohr
            'nn_dists_ref':     [5.94804919, 5.94883054, 5.97731908, 5.97751044, 6.08702227, 6.08772055], # Bohr
            'angle_cis_refs':   [90.16758216, 89.84042447, 90.40166034, 89.60573902, 89.83786228, 90.15413018, 89.60511724, 90.38748283, 92.61711464, 87.38848998, 87.39016994, 92.6042244],  # degrees
            'angle_trans_ref':  180.0,    # degrees (same in cubic and ortho)
        },
        'Cs': {
            'dist_mean':        8.3458608,     
            'dist_spread':      1.02919792,    
            'dist_skew':        0.21045231,    
            'angle_CsIPb_mean': 88.97281247,   
            'angle_CsIPb_std':  7.51809556,    
        },
    },
    'CsPbBr3': {
        'Br': {
            'angle_axial':      160.321,    # Pb-Br-Pb axial (degrees)
            'angle_equatorial': 150.862,    # Pb-Br-Pb equatorial (degrees)
            'angle_ref':        150.862,    # normalization reference (most distorted)
        },
        'Pb': {
            'bond_axial':       5.62675524,  # Bohr
            'bond_equatorial':  5.71507342,  # Bohr
            'nn_dists_ref':     [5.62675524, 5.62675799, 5.64411382, 5.64411771, 5.71503913, 5.71507342], # Bohr
            'angle_cis_refs':   [90.10809608, 89.89193187, 92.44763632, 87.55240593, 89.89194351, 90.10802854, 87.5527143, 92.44724345, 90.93811372, 89.06193085, 89.062235, 90.93772043],  # degrees
            'angle_trans_ref':  180.0,    # degrees (same in cubic and ortho)
        },
        'Cs': {
            'dist_mean':        None,     
            'dist_spread':      0.9673,    
            'dist_skew':        0.2187,    
            'angle_CsIPb_mean': None,   
            'angle_CsIPb_std':  8.1972,    
        },
    },
    # Add CsPbBr3, CsPbCl3 when reference values available
}

# Cubic reference values (analytic)
CUBIC_REF = {
    'I':  {'angle': 180.0},
    'Pb': {'bond_mean': None,       # TODO: from cubic reference calc
           'angle_cis':  90.0,
           'angle_trans': 180.0},
    'Cs': {'dist_mean':  None,      # TODO: from cubic reference calc
           'dist_spread': 0.0,
           'angle_mean':  None},    # TODO: from cubic reference calc
}

# =============================================================================
# Neighbor finding utilities
# =============================================================================

def find_nearest_neighbors(center_idx, target_type, atomTypes, dist, n_neighbors,
                                  dR=None, self_mask=None):
    """
    Find the n_neighbors nearest neighbors of a given type to atom center_idx,
    across all periodic images. Uses torch throughout to preserve autograd graph.

    Parameters
    ----------
    center_idx  : int
    target_type : str
    atomTypes   : list of str
    dist        : (N, N, M) torch tensor
    n_neighbors : int
    dR          : (N, N, M, 3) torch tensor, optional
    self_mask   : (N, N, M) bool torch tensor, optional

    Returns
    -------
    nn_dists  : (n_neighbors,) tensor  — differentiable
    nn_dR     : (n_neighbors, 3) tensor — differentiable (if dR provided)
    nn_indices: (n_neighbors,) long tensor — integer indices, no grad
    """
    N, _, M = dist.shape

    # Type mask — pure python/bool, no grad needed
    type_mask = torch.tensor([t == target_type for t in atomTypes],
                              dtype=torch.bool)               # (N,)

    # Extract distances from center_idx: (N, M)
    # Use clone to avoid in-place modification of the graph
    d_row = dist[center_idx].clone()                          # (N, M)

    # Mask out wrong types and self-interactions
    # We need a float copy for masking since we can't set inf on a grad tensor
    d_row_masked = d_row.detach().clone()                     # (N, M) — for index finding only
    d_row_masked[~type_mask, :] = float('inf')
    if self_mask is not None:
        d_row_masked[self_mask[center_idx]] = float('inf')

    # Find n_neighbors smallest by sorting the detached copy
    d_flat     = d_row_masked.flatten()                       # (N*M,)
    sorted_idx = torch.argsort(d_flat)
    nn_flat_idx = sorted_idx[:n_neighbors]                    # top-k indices

    atom_idx  = nn_flat_idx // M                              # (n_neighbors,) long
    image_idx = nn_flat_idx %  M                              # (n_neighbors,) long

    # Index into the original differentiable tensors
    nn_dists  = dist[center_idx, atom_idx, image_idx]         # (n_neighbors,) — has grad
    nn_indices = atom_idx                                      # (n_neighbors,) — no grad

    if dR is not None:
        nn_dR = dR[center_idx, atom_idx, image_idx, :]        # (n_neighbors, 3) — has grad
        return nn_dists, nn_dR, nn_indices

    return nn_dists, nn_indices


def find_nearest_neighbors_nano(center_idx, target_type, atomTypes, positions, n_neighbors):
    """
    Find the n_neighbors nearest neighbors of a given type to atom center_idx.
    No periodic images — operates on a plain (N, 3) position array.

    Parameters
    ----------
    center_idx  : int
        Index of the central atom.
    target_type : str
        Atom type to search for (e.g. 'I').
    atomTypes   : list of str
        Atom type for each atom index.
    positions   : (N, 3) array
        Cartesian coordinates.
    n_neighbors : int
        Number of nearest neighbors to return.

    Returns
    -------
    nn_dists   : (n_neighbors,) array of distances
    nn_dR      : (n_neighbors, 3) array of displacement vectors (center -> neighbor)
    nn_indices : (n_neighbors,) array of atom indices
    """
    positions = np.asarray(positions)

    dR   = positions - positions[center_idx]   # (N, 3)
    dist = np.linalg.norm(dR, axis=-1)         # (N,)

    # True for target type, False for self and wrong types
    type_mask = np.array([t == target_type for t in atomTypes])
    type_mask[center_idx] = False

    d_masked = dist.copy()
    d_masked[~type_mask] = np.inf

    # Guard: can't request more neighbors than valid targets
    n_valid = int(np.isfinite(d_masked).sum())
    k = min(n_neighbors, n_valid)

    if k == 0:
        raise ValueError(f"No valid neighbors of type {target_type} found for atom {center_idx}")
    if k < n_neighbors:
        print(f"Warning: atom {center_idx} has only {k} neighbors of type {target_type}, requested {n_neighbors}")

    # argsort puts inf entries last, take first k
    sorted_idx = np.argsort(d_masked)
    nn_idx     = sorted_idx[:k]

    return dist[nn_idx], dR[nn_idx], nn_idx

def find_Pb_octahedron(pb_idx, atomTypes, dist, dR, self_mask, halide):
    """
    Returns the 6 nearest I neighbors of a Pb atom and their displacement vectors.
    """
    nn_dists, nn_dR, nn_idx = find_nearest_neighbors(
        pb_idx, halide, atomTypes, dist, 6, dR=dR, self_mask=self_mask
    )
    return nn_dists, nn_dR, nn_idx


def find_Cs_cage(cs_idx, atomTypes, dist, dR, self_mask, halide):
    """
    Returns the 12 nearest I neighbors of a Cs atom and their displacement vectors.
    """
    nn_dists, nn_dR, nn_idx = find_nearest_neighbors(
        cs_idx, halide, atomTypes, dist, 12, dR=dR, self_mask=self_mask
    )
    return nn_dists, nn_dR, nn_idx

def find_Pb_octahedron_nano(pb_idx, atomTypes, positions, halide):
    return find_nearest_neighbors_nano(pb_idx, halide, atomTypes, positions, 6)

def find_Cs_cage_nano(cs_idx, atomTypes, positions, halide):
    return find_nearest_neighbors_nano(cs_idx, halide, atomTypes, positions, 12)

# =============================================================================
# Angle utilities
# =============================================================================

def angle_between(v1, v2):
    """Angle in degrees between two vectors or arrays of vectors."""
    # v1, v2: (..., 3)
    cos_theta = np.sum(v1 * v2, axis=-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-12
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def angle_between_torch(v1, v2):
    cross = torch.linalg.cross(v1, v2)
    dot   = (v1 * v2).sum(dim=-1)
    return torch.atan2(torch.linalg.norm(cross, dim=-1), dot) * (180.0 / torch.pi)

# =============================================================================
# Per-species descriptor functions
# =============================================================================

def compute_halide_descriptors(i_idx, atomTypes, dist, dR, self_mask, material, halide):
    """
    Descriptors for a halide atom.

    d1: (180 - theta_PbIPb) / (180 - angle_ref)   — Pb-I-Pb angle distortion
    d2: (r1 - r2) / (r1 + r2)                      — Pb-I bond asymmetry (near-dead here)
    d3: (mean(r1,r2) - r_ref) / dr_ref            — mean Pb-I bond length (compression/
                                                     stretch); sets the on-site potential
                                                     shift, absent from d1/d2
    d4: (mean Cs-I dist - cs_ref) / cs_spread_ref — size of the surrounding Cs cage

    Returns: (4,) tensor with grad
    """
    ref       = ORTHO_REF[material][halide]
    angle_ref = ref['angle_ref']

    pb_dists, pb_dR, _ = find_nearest_neighbors(
        i_idx, 'Pb', atomTypes, dist, 2, dR=dR, self_mask=self_mask
    )

    r1, r2 = pb_dists[0], pb_dists[1]

    # Pb-I-Pb angle using atan2 for numerical stability near 180 degrees
    v1    = pb_dR[0]   # (3,) I -> Pb1
    v2    = pb_dR[1]   # (3,) I -> Pb2
    theta = angle_between_torch(v1, v2)

    d1 = (180.0 - theta) / (180.0 - angle_ref)
    d2 = (r1 - r2)       / (r1 + r2 + 1e-12)

    # --- d3: mean Pb-I bond length, normalized by the ortho Pb-I bond stats ---
    pb_ref     = ORTHO_REF[material]['Pb']
    r_ref      = 0.5 * (pb_ref['bond_axial'] + pb_ref['bond_equatorial'])      # ~mean Pb-I bond
    dr_ref     = (pb_ref['bond_equatorial'] - pb_ref['bond_axial'])            # axial/equ split (scale)
    mean_r     = 0.5 * (r1 + r2)
    d3 = (mean_r - r_ref) / (dr_ref + 1e-12)

    # --- d4: mean halide->Cs distance over the 4 nearest Cs (Cs-cage size) ---
    cs_ref       = ORTHO_REF[material]['Cs']
    cs_dist_ref  = cs_ref['dist_mean']   if cs_ref['dist_mean']   else 8.35    # Bohr (cubic Cs-I)
    cs_spread    = cs_ref['dist_spread'] if cs_ref['dist_spread'] else 1.0     # Bohr (scale)
    cs_dists, _, _ = find_nearest_neighbors(
        i_idx, 'Cs', atomTypes, dist, 4, dR=dR, self_mask=self_mask
    )
    mean_cs = torch.mean(cs_dists)
    d4 = (mean_cs - cs_dist_ref) / (cs_spread + 1e-12)

    return torch.stack([d1, d2, d3, d4])


def compute_Pb_descriptors(pb_idx, atomTypes, dist, dR, self_mask, material, halide):
    """
    Descriptors for a Pb atom (octahedral environment).

    Bond length invariants:
        d1: mean Pb-I bond length (normalized by ortho mean)
        d2: axial vs equatorial spread (r_max - r_min), normalized
        d3: bond length std (breathing/Jahn-Teller), normalized

    Angular invariants:
        d4: mean deviation of cis angles from 90 (normalized)
        d5: std of cis angles (symmetry breaking), normalized

    Returns: (5,) tensor with grad
    """
    ref      = ORTHO_REF[material]['Pb']
    r_ref    = torch.tensor((ref['bond_axial'] + ref['bond_equatorial']) / 2.0, dtype=dist.dtype)
    dr_ref   = torch.tensor(ref['bond_equatorial'] - ref['bond_axial'], dtype=dist.dtype)
    std_ref  = torch.tensor(np.std(ref['nn_dists_ref']),   dtype=dist.dtype)
    cis_ref_mean_dev = torch.tensor( np.mean(np.abs(np.array(ref['angle_cis_refs']) - 90.0)), dtype=dist.dtype)
    cis_ref_std = torch.tensor(np.std(ref['angle_cis_refs']), dtype=dist.dtype)

    nn_dists, nn_dR, _ = find_Pb_octahedron(pb_idx, atomTypes, dist, dR, self_mask, halide)
    # nn_dists: (6,)  nn_dR: (6, 3) — both differentiable

    # --- Bond length descriptors ---
    d1 = torch.mean(nn_dists) / r_ref
    d2 = (torch.max(nn_dists) - torch.min(nn_dists)) / (dr_ref  + 1e-12)
    d3 = torch.std(nn_dists, unbiased=False)         / (std_ref + 1e-12)

    # --- All 15 pairwise angles ---
    angles = torch.stack([
        angle_between_torch(nn_dR[j], nn_dR[k])
        for j in range(6)
        for k in range(j+1, 6)
    ])                                                  # (15,) — differentiable

    # --- Classify cis/trans on detached copy ---
    angles_np   = angles.detach()
    cis_mask    = angles_np <= 150.0                    # (15,) bool
    cis_angles  = angles[cis_mask]                      # differentiable, variable length

    # --- Angular descriptors ---
    d4 = torch.mean(torch.abs(cis_angles - 90.0)) / (cis_ref_mean_dev + 1e-12)
    d5 = torch.std(cis_angles, unbiased=False)    / (cis_ref_std      + 1e-12)

    return torch.stack([d1, d2, d3, d4, d5])


def compute_Cs_descriptors(cs_idx, atomTypes, dist, dR, self_mask, material, halide):
    """
    Descriptors for a Cs atom (cuboctahedral cage of 12 I neighbors).

    d1: mean Cs-I distance (normalized)         — cage size
    d2: std of Cs-I distances (normalized)      — cage distortion, 0=cubic
    d3: (max - mean) / mean                     — asymmetric Cs displacement
    d4: mean Cs-I-Pb angle (normalized)         — placeholder
    d5: std of Cs-I-Pb angles (normalized)      — symmetry breaking, 0=cubic

    Returns: (5,) array
    """
    ref = ORTHO_REF[material]['Cs']

    nn_dists, nn_dR, nn_idx = find_Cs_cage(cs_idx, atomTypes, dist, dR, self_mask, halide)

    # --- Distance descriptors ---
    mean_d = torch.mean(nn_dists)
    std_d  = torch.std(nn_dists, unbiased=False)
    skew_d = (torch.max(nn_dists) - mean_d) / (mean_d + 1e-12)

    # Normalize — use placeholder 1.0 until reference values available
    d1 = std_d  / (ref['dist_spread'] if ref['dist_spread'] else 1.0)
    d2 = skew_d / (ref['dist_skew'] if ref['dist_skew'] else 1.0)

    # --- Cs-I-Pb angle descriptors ---
    # For each I neighbor, find its nearest Pb and compute Cs-I-Pb angle
    CsIPb_angles = []
    for n, (i_idx, i_dR) in enumerate(zip(nn_idx, nn_dR)):
        pb_dists_of_I, pb_dR_of_I, _ = find_nearest_neighbors(
            i_idx, 'Pb', atomTypes, dist, 1,  dR=dR, self_mask=self_mask
        )
        
        v_ICs = -i_dR                    # vector from I toward Cs
        v_IPb = pb_dR_of_I[0]            # vector from I toward Pb
        CsIPb_angles.append(angle_between_torch(v_ICs, v_IPb))
    
    CsIPb_angles = torch.stack(CsIPb_angles)
    std_angle    = torch.std(CsIPb_angles, unbiased=False)

    d3 = std_angle  / (ref['angle_CsIPb_std']  if ref['angle_CsIPb_std']  else 1.0)

    return torch.stack([d1, d2, d3])

def compute_I_descriptors_nano(i_idx, atomTypes, positions, material):
    """
    Descriptors for an I atom (no periodic boundary).
    d1: (180 - theta_PbIPb) / (180 - 150.8)
    d2: (r1 - r2) / (r1 + r2)
    """
    ref = ORTHO_REF[material]['I']

    pb_dists, pb_dR, _ = find_nearest_neighbors_nano(i_idx, 'Pb', atomTypes, positions, 2)

    r1, r2 = pb_dists[0], pb_dists[1]
    theta  = angle_between(pb_dR[0], pb_dR[1])

    d1 = (180.0 - theta) / (180.0 - ref['angle_ref'])
    d2 = (r1 - r2) / (r1 + r2 + 1e-12)

    return np.array([d1, d2])


def compute_Pb_descriptors_nano(pb_idx, atomTypes, positions, material):
    """
    Descriptors for a Pb atom (no periodic boundary).
    d1: mean Pb-I bond length (normalized)
    d2: bond length spread (normalized)
    d3: bond length std (normalized)
    d4: mean cis angle deviation from 90 (normalized)
    d5: std of cis angles (normalized)
    """
    ref   = ORTHO_REF[material]['Pb']
    r_ref = np.mean([ref['bond_axial'], ref['bond_equatorial']])
    dr_ref = ref['bond_equatorial'] - ref['bond_axial']

    nn_dists, nn_dR, _ = find_nearest_neighbors_nano(pb_idx, 'I', atomTypes, positions, 6)

    # Bond length descriptors
    d1 = np.mean(nn_dists) / r_ref
    d2 = (np.max(nn_dists) - np.min(nn_dists)) / (dr_ref + 1e-12)
    d3 = np.std(nn_dists)  / (dr_ref + 1e-12)

    # Angular descriptors
    angles = []
    for j in range(len(nn_dR)):
        for k in range(j+1, len(nn_dR)):
            angles.append(angle_between(nn_dR[j], nn_dR[k]))
    angles = np.array(angles)

    cis_angles  = angles[angles <= 150.0]
    trans_angles = angles[angles > 150.0]

    cis_ref_mean_dev = np.mean(np.abs(np.array(ref['angle_cis_refs']) - 90.0))

    d4 = np.mean(np.abs(cis_angles - 90.0)) / (cis_ref_mean_dev + 1e-12) if len(cis_angles)  > 0 else 0.0
    d5 = np.std(cis_angles)                 / (np.std(ref['angle_cis_refs']) + 1e-12) if len(cis_angles) > 0 else 0.0

    return np.array([d1, d2, d3, d4, d5])


def compute_Cs_descriptors_nano(cs_idx, atomTypes, positions, material):
    """
    Descriptors for a Cs atom (no periodic boundary).
    d1: mean Cs-I distance (normalized)
    d2: std of Cs-I distances (normalized)
    d3: asymmetric displacement (max - mean) / mean
    """
    ref = ORTHO_REF[material]['Cs']

    nn_dists, nn_dR, nn_idx = find_nearest_neighbors_nano(cs_idx, 'I', atomTypes, positions, 12)

    mean_d = np.mean(nn_dists)
    std_d  = np.std(nn_dists)
    skew_d = (np.max(nn_dists) - mean_d) / (mean_d + 1e-12)

    d1 = mean_d / (ref['dist_mean']   if ref['dist_mean']   else 1.0)
    d2 = std_d  / (ref['dist_spread'] if ref['dist_spread'] else 1.0)
    d3 = skew_d / (ref['dist_skew']   if ref['dist_skew']   else 1.0)

    return np.array([d1, d2, d3])

def calcLocalSymmDescriptor(system, style="Behler-Parrinello"):
  """
  Compute Behler-Parrinello G2, G4/G5 symmetry funcs
  """

  if style == "Behler-Parrinello":
    symm_funcs = calcBehlerParrinelloDescriptor(system)

  return symm_funcs


def retEquilDist(atom1, atom2):
    if ((atom1 == 'Pb') and (atom2 == 'I')) or ((atom2 == 'Pb') and (atom1 == 'I')):
        return 5.94319 # Bohr Pb-I distance

def retMinImageDist(atomPos, unitCellVectors):
    """
    Compute minimum-image convention pairwise displacement vectors and distances.

    Parameters
    ----------
    atomPos : (N,3) array
        Atomic positions in Cartesian coordinates.
    unitCellVectors : (3,3) array
        Lattice vectors as rows: row 0 = a, row 1 = b, row 2 = c.

    Returns
    -------
    dR : (N,N,3) array
        Minimum-image displacement vectors r_j - r_i.
    dist : (N,N) array
        Pairwise minimum-image distances.
    """
    atomPos = np.asarray(atomPos)
    cell    = np.asarray(unitCellVectors)   # rows are a,b,c
    
    # Correct Cartesian -> fractional transform for row-convention cell
    invCell = np.linalg.inv(cell)           # (3,3)
    fracPos = atomPos @ invCell             # (N,3)

    # Fractional displacements
    dFrac = fracPos[None, :, :] - fracPos[:, None, :]   # (N,N,3)

    # Wrap to [-0.5, 0.5): minimum image in fractional space
    dFrac -= np.round(dFrac)

    # Back to Cartesian
    dR   = dFrac @ cell                     # (N,N,3)
    dist = np.linalg.norm(dR, axis=-1)      # (N,N)
    return dR, dist

def retAllImageDist(atomPos, unitCellVectors, n_images=2):
    """
    Generate all periodic images within ±n_images cells in each direction,
    and return ALL displacement vectors and distances (not just minimum image).

    Parameters
    ----------
    atomPos : (N,3) array
        Atomic positions in Cartesian coordinates.
    unitCellVectors : (3,3) array
        Lattice vectors as rows: row 0 = a, row 1 = b, row 2 = c.
    n_images : int
        Number of periodic images in each direction (±n_images).

    Returns
    -------
    dR   : (N, N, M, 3) array
        Displacement vectors r_j + T - r_i for all image translations T.
    dist : (N, N, M) array
        Corresponding distances. M = (2*n_images+1)**3 total translations.
    """
    atomPos = np.asarray(atomPos, dtype=float)
    cell    = np.asarray(unitCellVectors, dtype=float)

    # Build all translation vectors T = n0*a + n1*b + n2*c
    ns = np.arange(-n_images, n_images + 1)
    n0, n1, n2 = np.meshgrid(ns, ns, ns, indexing='ij')
    offsets = np.stack([n0.ravel(), n1.ravel(), n2.ravel()], axis=1)  # (M, 3)
    translations = offsets @ cell                                       # (M, 3)  Cartesian

    # Raw displacements r_j - r_i, shape (N, N, 3)
    dR_base = atomPos[None, :, :] - atomPos[:, None, :]  # (N, N, 3)

    # Add all translations: (N, N, 1, 3) + (1, 1, M, 3) -> (N, N, M, 3)
    dR   = dR_base[:, :, None, :] + translations[None, None, :, :]
    dist = np.linalg.norm(dR, axis=-1)                                 # (N, N, M)

    return dR, dist

def cutoff_fc(r, Rc):
    """
    Smooth cutoff function f_c(r). Returns 0 for r>=Rc.
    Uses cosine cutoff:
        f_c(r) = 0.5 * (cos(pi * r / Rc) + 1)   for r < Rc
               = 0                                  for r >= Rc
    Works for scalar, array, or broadcast-shaped Rc.
    """
    r  = np.asarray(r,  dtype=float)
    Rc = np.asarray(Rc, dtype=float)
    val = 0.5 * (np.cos(np.pi * r / Rc) + 1.0)
    return np.where(r < Rc, val, 0.0)


def cutoff_fc_prime(r, Rc):
    """
    Derivative of smooth cutoff function df_c/dr.
        f_c'(r) = -0.5 * (pi/Rc) * sin(pi * r / Rc)   for r < Rc
                = 0                                      for r >= Rc
    Works for scalar, array, or broadcast-shaped Rc.
    """
    r  = np.asarray(r,  dtype=float)
    Rc = np.asarray(Rc, dtype=float)
    val = -0.5 * (np.pi / Rc) * np.sin(np.pi * r / Rc)
    return np.where(r < Rc, val, 0.0)

# def cutoff_fc_prime(r, Rc):
#     """
#     Derivative of cosine cutoff function f_c(r)
#     """
#     r = np.asarray(r)
#     fcp = np.zeros_like(r, dtype=float)
#     mask = (r < Rc)
#     x = r[mask] * np.pi / Rc
#     fcp[mask] = -0.5 * (np.pi / Rc) * np.sin(x)
#     return fcp

def setDefaultBPParams(nAtoms, G2_params, G4_params, G5_params):
    # default parameter sets if none provided
    if G2_params is None:
        G2_params = [{'eta': 0.5, 'Rs': 5.5407}] * nAtoms
        # 5.5407 Bohr is the cubic equilibrium distance in CsPbI3
        # This value should be adjusted for bromide/general case if accidentally left behind!
        # This will be a bug if not made into a variable in later editions of the code! - Daniel C 9.22.25
    if G4_params is None:
        G4_params = [{'eta': 0.005, 'zeta': 1.0, 'lambda':  1.0}] * nAtoms
    if G5_params is None:
        # G5 channels: include Rs like G2 does, hybrid radial-angular
        G5_params = [{'eta': 0.005, 'zeta': 1.0, 'lambda':  1.0, 'Rs': 0.0}] * nAtoms
    
    return G2_params, G4_params, G5_params

def calcBehlerParrinelloDescriptor(
    system,
    Rc=12.0,
    G2_params=None,
    G4_params=None,
    G5_params=None,
    calcAngular=False
):
    """
    Compute BP descriptors (G2, G4, G5) for a list of systems.

    Parameters
    ----------
    system : BulkSystem object
        Bulk system object with attrs:
          - atomPos : (N,3) ndarray (Cartesian)
          - unitCellVectors : (3,3) ndarray
          - atomTypes : (N,) array-like of ints or strings (optional but helpful)
    Rc : float
        Cutoff radius for neighbors.
    G2_params : list of dict
        Each dict has keys 'eta' and 'Rs' for a G2 channel. Example:
           [{'eta': 0.005, 'Rs': 0.0}, {'eta': 0.5, 'Rs': 0.5}, ...]
    G4_params : list of dict
        Each dict has keys 'eta', 'zeta', 'lambda' for a G4 channel. Example:
           [{'eta': 0.001, 'zeta': 1.0, 'lambda': +1.0}, ...]
    G5_params : list of dict
        Each dict has keys 'eta', 'zeta', 'lambda', 'Rs' for G5 channels.
        (G5 is an alternative angular/radial hybrid; see comments below.)

    Returns
    -------
    all_desc : list
        List of dicts (one per system) with keys:
           - 'G2' : (N, n_G2) ndarray
           - 'G4' : (N, n_G4) ndarray
           - 'G5' : (N, n_G5) ndarray
           - optionally 'atomTypes'
    """

    all_desc = []

    atomPos = np.asarray(system.atomPos)
    cell = np.asarray(system.unitCellVectors)
    atomTypes = np.asarray(getattr(system, 'atomTypes', np.arange(atomPos.shape[0])))
    nAtoms = atomPos.shape[0]

    G2_params, G4_params, G5_params = setDefaultBPParams(nAtoms, G2_params, G4_params, G5_params)
    
    dR, dist = retMinImageDist(atomPos, cell)  # dR: (N,N,3), dist: (N,N)
    # Precompute cutoff matrix (N,N)
    fc_mat = cutoff_fc(dist, Rc)
    # zero self-interactions
    np.fill_diagonal(fc_mat, 0.0)
    # mask of neighbors (bool)
    neigh_mask = (dist > 1e-12) & (dist < Rc)

    # --- G2 computation -------------------------------------------------
    nG2 = len(G2_params)
    G2 = np.zeros((nAtoms, nG2), dtype=float)
    # assert(nG2 - nAtoms < 1e-15)
    for ig2, p in enumerate(G2_params):
        eta = p['eta']
        Rs = p['Rs']
        # apply for each central atom i: G2_i = sum_j exp(-eta*(R_ij - Rs)^2) * f_c(R_ij)
        # vectorized:
        term = np.exp(-eta * (dist - Rs)**2) * fc_mat
        # sum over neighbor j
        G2[:, ig2] = term.sum(axis=1)
    
    # C2 is a [nAtoms] tensor of BP descriptors for each atom in the system.
    G2 = torch.tensor(G2[:, 0])
    
    # This dict, G2_dict contains the unique descriptors for each atomType
    # It was developed to decrease the expense of evaluating the Vlsd matrix (ham.py buildVlocmat)
    # However, that routine must loop over all atoms anyway to obtain the correct form factors, so
    # there was no savings. Commenting out and removing from implementation on 10.23.25 - Daniel C.
    # G2_dict = {elem: [] for elem in set(atomTypes)}
    # for alpha, atom in enumerate(atomTypes):
    #     G2_dict.setdefault(atom, []).append(G2[alpha])
    # for atom in set(atomTypes):
    #     unique_vals = sorted(set(round(float(x), 6) for x in G2_dict[atom]))
    #     G2_dict[atom] = torch.tensor(unique_vals)
        

    if calcAngular:
        # --- G4 computation (angular 3-body) ------------------------------
        # Chosen variant (common): G4_i = 2^{1-zeta} sum_{j,k != i}
        #    (1 + lambda * cos(theta_ijk))^zeta * exp(-eta*(R_ij^2 + R_ik^2 + R_jk^2)) * fc(R_ij) fc(R_ik) fc(R_jk)
        # Note: this is the "heavy" triple-sum variant. We compute per center i looping over i,
        # and vectorizing over pairs (j,k) using broadcasting. For typical neighbor counts this is fine.
        nG4 = len(G4_params)
        G4 = np.zeros((nAtoms, nG4), dtype=float)

        for i in range(nAtoms):
            # neighbors indices for central i
            neigh_idx = np.nonzero(neigh_mask[i])[0]
            nj = neigh_idx.size
            if nj < 2:
                continue
            # displacement vectors from i to j: r_ij shape (nj,3)
            r_ij = dR[i, neigh_idx, :]   # (nj,3)
            R_ij = dist[i, neigh_idx]    # (nj,)
            fc_ij = fc_mat[i, neigh_idx] # (nj,)

            # prepare pairwise arrays between neighbors j,k
            # r_ij[:,None,:] and r_ij[None,:,:] -> (nj,nj,3)
            rj = r_ij[:, None, :]    # (nj,1,3)
            rk = r_ij[None, :, :]    # (1,nj,3)
            # pairwise dot product r_ij . r_ik -> (nj,nj)
            # dot = np.einsum('aij,akj->aiak', rj, rk)  # this is wrong shape; will do simpler:
            # simpler: compute with broadcasting
            dot = np.sum(rj * rk, axis=-1)  # (nj,nj)

            R_j = R_ij[:, None]  # (nj,1)
            R_k = R_ij[None, :]  # (1,nj)
            # cos(theta) = (r_ij · r_ik) / (R_ij * R_ik)
            denom = (R_j * R_k)
            # avoid div by zero (shouldn't happen because neighbours excluded 0)
            cos_theta = np.zeros_like(dot)
            mask_nonzero = denom > 1e-12
            cos_theta[mask_nonzero] = dot[mask_nonzero] / denom[mask_nonzero]
            # clip numerical noise
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            # pairwise distance between j and k (neighbor-neighbor) under PBC:
            # We can compute using dR[neigh_idx][:,neigh_idx,:]
            # which yields (nj,nj,3)
            dR_jk = dR[np.ix_(neigh_idx, neigh_idx, [0,1,2])]  # shape (nj,nj,3)
            R_jk = np.linalg.norm(dR_jk, axis=-1)  # (nj,nj)

            # triple cutoff product fc_ij * fc_ik * fc_jk
            fc_j = fc_ij[:, None]  # (nj,1)
            fc_k = fc_ij[None, :]  # (1,nj)
            fc_jk = cutoff_fc(R_jk, Rc)  # (nj,nj)
            fc_trip = fc_j * fc_k * fc_jk  # (nj,nj)

            # Now compute G4 channels
            for ig4, p in enumerate(G4_params):
                eta = p['eta']
                zeta = p['zeta']
                lamb = p['lambda']  # +1 or -1 commonly
                prefac = 2.0**(1.0 - zeta)
                ang_term = (1.0 + lamb * cos_theta)**zeta  # (nj,nj)
                # radial exponential: exp(-eta*(R_ij^2 + R_ik^2 + R_jk^2))
                radial = np.exp(-eta * (R_j**2 + R_k**2 + R_jk**2))
                # product
                contrib = prefac * ang_term * radial * fc_trip
                # sum over j,k (note that j=k terms are included but fc_jk==0 there; safe)
                G4[i, ig4] = contrib.sum()

        # --- G5 computation (alternative angular/radial hybrid) -----------
        # One common G5 variant:
        #   G5_i = sum_{j,k != i} (1 + lambda*cos(theta_ijk))^zeta * exp(-eta * ((R_ij + R_ik)/2 - Rs)^2) * fc(R_ij) fc(R_ik)
        # This variant omits the fc(R_jk) and R_jk in the exponent; it is lighter-weight.
        nG5 = len(G5_params)
        G5 = np.zeros((nAtoms, nG5), dtype=float)

        for i in range(nAtoms):
            neigh_idx = np.nonzero(neigh_mask[i])[0]
            nj = neigh_idx.size
            if nj < 2:
                continue
            r_ij = dR[i, neigh_idx, :]
            R_ij = dist[i, neigh_idx]
            fc_ij = fc_mat[i, neigh_idx]

            rj = r_ij[:, None, :]
            rk = r_ij[None, :, :]
            dot = np.sum(rj * rk, axis=-1)
            R_j = R_ij[:, None]
            R_k = R_ij[None, :]
            denom = R_j * R_k
            cos_theta = np.zeros_like(dot)
            mask_nonzero = denom > 1e-12
            cos_theta[mask_nonzero] = dot[mask_nonzero] / denom[mask_nonzero]
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            fc_pair = fc_ij[:, None] * fc_ij[None, :]

            for ig5, p in enumerate(G5_params):
                eta = p['eta']
                zeta = p['zeta']
                lamb = p['lambda']
                Rs = p.get('Rs', 0.0)
                ang_term = (1.0 + lamb * cos_theta)**zeta
                # radial factor uses average distance (R_ij+R_ik)/2 and a shift Rs
                Rmean = 0.5 * (R_j + R_k)
                radial = np.exp(-eta * (Rmean - Rs)**2)
                contrib = ang_term * radial * fc_pair
                G5[i, ig5] = contrib.sum()

    # Package
    desc = {
        'G2': G2
        # 'G4': G4,
        # 'G5': G5,
        # 'atomTypes': atomTypes
    }

    return desc


def compute_gradN(pos, neighbor_list, eta, R0, fc, fc_prime):
    N = pos.shape[0]
    gradN = np.zeros((N,3))
    for (a,b) in neighbor_list:            # each neighbor pair once
        r_ab = pos[a] - pos[b]
        R = np.linalg.norm(r_ab)
        if R == 0: 
            continue
        e_ab = r_ab / R
        expfac = np.exp(-eta*(R - R0)**2)
        w = expfac*(fc_prime(R) - 2*eta*(R - R0)*fc(R))
        # contribution to grad N_a (positive) and to grad N_b (negative)
        vec = w * e_ab
        gradN[a] += vec
        gradN[b] -= vec
    
    return gradN