"""
utils/mace_descriptors.py
=========================

MACE-MP-0 per-atom invariant descriptors, used as a drop-in replacement for the
hand-crafted ``BulkSystem.env_descriptors``.

The frozen MACE-MP-0 foundation model is used purely as a feature extractor (no
training): for each atom we take its rotation-invariant node features
(``get_descriptors(atoms, invariants_only=True)``, D=256 for the "medium" model).
These feed the existing per-element LSD MLP as the N_alpha inputs.

This backend does NOT provide dN_dR (the band-structure-stage gradient of the
descriptors w.r.t. atomic positions). It is for the init_LSD pretraining only.

Determinism / cost: MACE is far costlier than the hand-crafted descriptors, and
the values must be byte-identical across dataset generation, training, and
diagnosis (otherwise the per-atom indexing desyncs). So we compute on CPU/float64
and cache each structure's descriptors to disk, keyed by a geometry hash.
"""
import os
import re
import hashlib

import numpy as np
import torch

from .constants import AUTOAA   # Bohr -> Angstrom

torch.set_default_dtype(torch.float64)

_MODEL_CACHE = {}
_DESC_CACHE_DIR = os.environ.get(
    "MACE_DESC_CACHE", os.path.expanduser("~/.cache/lsd_mace_desc")
)


def load_mace_model(model="medium", device="cpu"):
    """Load (and memoize) a MACE-MP-0 calculator. Frozen; CPU/float64."""
    key = (model, device)
    if key not in _MODEL_CACHE:
        from mace.calculators import mace_mp
        _MODEL_CACHE[key] = mace_mp(model=model, device=device, default_dtype="float64")
    return _MODEL_CACHE[key]


def bulk_to_ase(system):
    """BulkSystem (Bohr) -> ASE Atoms (Angstrom, periodic). Element symbols are
    the stripped atomTypes (already 'Cs'/'I'/'Pb' from setSystem)."""
    from ase import Atoms
    cell_A = system.unitCellVectors.detach().cpu().numpy() * AUTOAA
    pos_A  = system.atomPos.detach().cpu().numpy() * AUTOAA
    syms   = [re.match(r"[A-Za-z]+", str(t)).group(0) for t in system.atomTypes]
    return Atoms(symbols=syms, positions=pos_A, cell=cell_A, pbc=True)


def _hash_atoms(atoms, model):
    h = hashlib.sha1()
    h.update(model.encode())
    h.update(np.round(np.asarray(atoms.cell, dtype=np.float64), 6).tobytes())
    h.update(np.round(atoms.get_positions(), 6).tobytes())
    h.update("".join(atoms.get_chemical_symbols()).encode())
    return h.hexdigest()[:16]


def compute_mace_descriptors(atoms, model="medium"):
    """Per-atom invariant descriptors [n_atoms, D]; cached to disk by geometry."""
    os.makedirs(_DESC_CACHE_DIR, exist_ok=True)
    path = os.path.join(_DESC_CACHE_DIR, f"{model}_{_hash_atoms(atoms, model)}.npy")
    if os.path.exists(path):
        return np.load(path)
    calc = load_mace_model(model=model)
    desc = np.asarray(calc.get_descriptors(atoms, invariants_only=True), dtype=np.float64)
    np.save(path, desc)
    return desc


def mace_env_descriptors(system, model="medium"):
    """Return {element: tensor[n_atoms_of_element, D]} in atomTypes order, the
    drop-in for BulkSystem.env_descriptors."""
    atoms = bulk_to_ase(system)
    desc = compute_mace_descriptors(atoms, model=model)        # [n_atoms, D]
    syms = np.array([re.match(r"[A-Za-z]+", str(t)).group(0) for t in system.atomTypes])
    out = {}
    for el in np.unique(syms):
        out[str(el)] = torch.tensor(desc[syms == el], dtype=torch.float64)   # preserves order
    return out
