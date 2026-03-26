import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from .constants import *

torch.set_default_dtype(torch.float64)

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

# def retMinImageDist(atomPos, unitCellVectors):
#     """
#     Compute minimum-image convention pairwise displacement vectors and distances.

#     Parameters
#     ----------
#     atomPos : (N,3) array
#         Atomic positions in Cartesian coordinates.
#     unitCellVectors : (3,3) array
#         Lattice vectors (a,b,c) in Cartesian coords.

#     Returns
#     -------
#     dR : (N,N,3) array
#         Minimum-image displacement vectors r_j - r_i.
#     dist : (N,N) array
#         Pairwise minimum-image distances.
#     """
#     nAtoms = atomPos.shape[0]
#     # Convert Cartesian -> fractional
#     invCell = np.linalg.inv(unitCellVectors.T)  # (3,3)
#     fracPos = atomPos @ invCell  # (N,3)

#     # Fractional differences
#     dFrac = fracPos[:, None, :] - fracPos[None, :, :]  # (N,N,3)

#     # Apply minimum image: wrap to [-0.5,0.5)
#     dFrac -= np.round(dFrac)

#     # Back to Cartesian
#     dR = dFrac @ unitCellVectors.T  # (N,N,3)
#     # print(f"dR = \n {dR[0]}")
#     dist = np.linalg.norm(dR, axis=-1)  # (N,N)
#     return dR, dist

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
    print(f"cell:\n{cell}")
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

def cutoff_fc(r, Rc):
    """
    Smooth cutoff function f_c(r). Returns 0 for r>=Rc.
    Uses cosine cutoff:
        f_c(r) = 0.5 * (cos(pi * r / Rc) + 1)   for r < Rc
               = 0                                  for r >= Rc
    """
    r = np.asarray(r)
    fc = np.zeros_like(r, dtype=float)
    mask = (r < Rc)
    x = r[mask] * np.pi / Rc
    fc[mask] = 0.5 * (np.cos(x) + 1.0)
    return fc

def cutoff_fc_prime(r, Rc):
    """
    Derivative of cosine cutoff function f_c(r)
    """
    r = np.asarray(r)
    fcp = np.zeros_like(r, dtype=float)
    mask = (r < Rc)
    x = r[mask] * np.pi / Rc
    fcp[mask] = -0.5 * (np.pi / Rc) * np.sin(x)
    return fcp

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