"""
utils/mace_descriptors.py
=========================

MACE-MP-0 per-atom invariant descriptors, used as a drop-in replacement for the
hand-crafted ``BulkSystem.env_descriptors``.

The frozen MACE-MP-0 foundation model is used purely as a feature extractor (no
training): for each atom we take its rotation-invariant node features
(``get_descriptors(atoms, invariants_only=True)``, D=256 for the "medium" model).
These feed the existing per-element LSD MLP as the N_alpha inputs.

Two paths are provided:

* ``mace_env_descriptors`` / ``compute_mace_descriptors`` -- the frozen, *detached*
  descriptors used for the init_LSD pretraining. These are computed on CPU/float64
  and cached to disk keyed by a geometry hash, because the values must be
  byte-identical across dataset generation, training, and diagnosis (otherwise
  the per-atom indexing desyncs).

* ``mace_env_descriptors_grad`` / ``compute_mace_descriptors_grad`` -- a
  *differentiable* forward pass that returns descriptors carrying an autograd
  graph back to ``atomPos``. The neighbour list (edge_index / shifts) is built
  once from the detached positions; gradients then flow through the injected
  ``positions`` tensor via ``vectors = pos[recv] - pos[send] + shifts``. This is
  what the band-structure / coupling stage uses so that ham.buildCouplingMats
  picks up the dN/dR contribution. The values are identical (to round-off) to the
  cached path -- only the graph differs.
"""
import os
import re
import hashlib

import numpy as np
import torch

from .constants import AUTOAA   # Bohr -> Angstrom

torch.set_default_dtype(torch.float64)

_MODEL_CACHE = {}
_META_CACHE = {}
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


# =============================================================================
# Differentiable path (graph back to atomPos) -- for the coupling/band stage
# =============================================================================

def _invariant_metadata(calc):
    """Cache (num_interactions, l_max, num_invariant_features, to_keep) needed to
    extract the invariant node features, exactly as MACECalculator.get_descriptors
    does. Keyed by the model object so repeated calls are free."""
    key = id(calc.models[0])
    if key not in _META_CACHE:
        from e3nn import o3
        mdl = calc.models[0]
        num_interactions = int(mdl.num_interactions)
        irreps_out = o3.Irreps(str(mdl.products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_inv = irreps_out.dim // (l_max + 1) ** 2
        per_layer = [irreps_out.dim] * num_interactions
        per_layer[-1] = num_inv          # last layer carries only invariants
        to_keep = int(sum(per_layer[:num_interactions]))
        _META_CACHE[key] = (num_interactions, l_max, num_inv, to_keep)
    return _META_CACHE[key]


def _ase_atoms_from_pos(system, pos_bohr_np):
    """ASE Atoms (Angstrom, periodic) at the given Bohr positions; cell/symbols
    from `system`. Used only to build the (non-differentiable) neighbour list."""
    from ase import Atoms
    cell_A = system.unitCellVectors.detach().cpu().numpy() * AUTOAA
    syms   = [re.match(r"[A-Za-z]+", str(t)).group(0) for t in system.atomTypes]
    return Atoms(symbols=syms, positions=pos_bohr_np * AUTOAA, cell=cell_A, pbc=True)


def compute_mace_descriptors_grad(system, atomPos, model="medium"):
    """Per-atom invariant descriptors [n_atoms, D] computed by a differentiable
    forward pass, so the result carries an autograd graph back to `atomPos`
    (Bohr). Values match compute_mace_descriptors to round-off.

    The neighbour list (edge_index, shifts) is built from atomPos.detach(); the
    gradient flows only through the injected positions, exactly like the
    hand-crafted descriptors (which also hold the periodic-image topology fixed
    and differentiate the distances)."""
    from mace.modules.utils import extract_invariant
    calc = load_mace_model(model=model)
    mdl  = calc.models[0]
    num_interactions, l_max, num_inv, to_keep = _invariant_metadata(calc)
    n_atoms = atomPos.shape[0]

    atoms = _ase_atoms_from_pos(system, atomPos.detach().cpu().numpy())
    batch = calc._atoms_to_batch(atoms)
    d = batch.to_dict()
    if d["positions"].shape[0] != n_atoms:
        raise RuntimeError(
            "MACE batch was padded (use_compile enabled?); the differentiable "
            "descriptor path needs an unpadded graph.")
    # inject differentiable positions (Angstrom) that trace back to atomPos (Bohr).
    # prepare_graph()'s in-place positions.requires_grad_(True) is a no-op on this
    # (already-requires-grad, non-leaf) tensor, so the graph is preserved.
    d["positions"] = atomPos.to(torch.float64) * AUTOAA
    out = mdl(d, compute_force=False)
    inv = extract_invariant(out["node_feats"], num_layers=num_interactions,
                            num_features=num_inv, l_max=l_max)
    return inv[:, :to_keep][:n_atoms]


def mace_env_descriptors_grad(system, atomPos=None, model="medium"):
    """Differentiable drop-in for mace_env_descriptors: returns
    {element: tensor[n_atoms_of_element, D]} (atomTypes order) carrying a graph
    back to `atomPos` (defaults to system.atomPos)."""
    if atomPos is None:
        atomPos = system.atomPos
    desc = compute_mace_descriptors_grad(system, atomPos, model=model)   # [n_atoms, D], graph
    syms = np.array([re.match(r"[A-Za-z]+", str(t)).group(0) for t in system.atomTypes])
    out = {}
    for el in np.unique(syms):
        idx = torch.from_numpy(np.where(syms == el)[0]).long()
        out[str(el)] = desc.index_select(0, idx)        # preserves order & graph
    return out
