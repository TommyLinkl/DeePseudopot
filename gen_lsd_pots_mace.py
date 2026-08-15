#!/usr/bin/env python3
"""
gen_lsd_pots_mace.py
====================

Standalone generator of the local-structure-dependent (LSD) potential correction
Delta v^loc for every atom of a *nanocrystal*, using the MACE-descriptor LSD nets
trained on bulk. This is the open-boundary (non-periodic) counterpart of the bulk
machinery: it reads a `conf.par` nanocrystal geometry, computes each atom's MACE
invariant descriptor, runs it through the trained per-element LSD network, and
writes the resulting Delta v^loc(q) and its real-space radial transform Delta
v^loc(r) -- one pair of files per atom -- ready to be added to the nanocrystal's
total local potential.

It mirrors gen_lsd_pots.py (the hand-crafted-descriptor version) so the output
format is a drop-in for the same downstream nanocrystal code:
    LSD/pot_q_LSD_<sym><idx>.dat   columns: q[Bohr^-1]   Delta v(q)
    LSD/pot_LSD_<sym><idx>.dat     columns: r[Bohr]      Delta v(r)

Three things this script gets right for the bulk->nanocrystal transfer:

  1. Non-periodic descriptors. NC atoms are embedded with pbc=False, so MACE sees
     the real (open) environment. Deep-interior atoms reproduce the bulk periodic
     descriptor (the descriptor is local / receptive-field-converged); near-surface
     atoms do not.

  2. Correct reference subtraction. The trained net outputs
     Delta v = NN([N, q]) - NN([N_ref, q]), and N_ref is the *bulk cubic* descriptor.
     N_ref is NOT stored in the checkpoint (it is set at runtime during training),
     so we reconstruct it here from a periodic cubic reference cell -- exactly the
     reference the bulk fit used. Get this wrong and every Delta v is offset.

  3. Surface exemption. The bulk-trained model is only valid where the NC
     descriptor matches a bulk environment, i.e. the interior. Surface atoms
     (within --surface-threshold of the cluster boundary) are given Delta v = 0.
     Use --no-surface-exempt to override.

Usage
-----
    python gen_lsd_pots_mace.py conf.par CsPbI3 --models results_lsd_train_mace_2

    # tune the surface depth / grids / reference, or apply to all atoms
    python gen_lsd_pots_mace.py conf.par CsPbI3 --models DIR \
        --surface-threshold 11.3 --qmax 30 --nq 4096 --rmax 120 --nr 4096
    python gen_lsd_pots_mace.py conf.par CsPbI3 --models DIR --ref-par system_0.par
    python gen_lsd_pots_mace.py conf.par CsPbI3 --models DIR --no-surface-exempt --plot
"""
import os
import sys
import argparse

import numpy as np
import torch
torch.set_default_dtype(torch.float64)

# allow running from anywhere: make the package importable
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.constants import AUTOAA
from utils.nn_models import Net_celu_HeInit_decayGaussian_LSD
from utils.mace_descriptors import compute_mace_descriptors
from utils.compute_nanocrystal_descriptors import read_conf_par, compute_surface_mask


def coordination_surface_mask(positions, r_cut, frac=0.95):
    """Shape-agnostic surface detector: an atom is 'interior' iff its neighbour
    count within r_cut (Bohr) is >= frac * bulk coordination, where the bulk
    coordination is the 95th percentile of counts in the cluster (the interior
    value). Unlike the bounding-box test this is correct for spherical / faceted
    nanocrystals, not just cuboids. An atom is exempted (surface) when part of its
    receptive-field environment is missing, which is exactly when the bulk-trained
    descriptor stops being transferable."""
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pos)
        counts = tree.query_ball_point(pos, r=r_cut, return_length=True) - 1   # exclude self
    except Exception:
        counts = np.empty(n, dtype=int)
        for i in range(n):                      # O(N^2) fallback, chunk-free
            counts[i] = int((np.linalg.norm(pos - pos[i], axis=1) <= r_cut).sum()) - 1
    bulk_coord = np.percentile(counts, 95)
    return counts < frac * bulk_coord

# cubic (high-symmetry) lattice constant a = |a1| (Bohr) per material; equals the
# Pb-Pb / Cs-Cs cubic spacing used to build the bulk reference (see system_0.par).
CUBIC_A = {"CsPbI3": 11.88638, "CsPbBr3": 11.09269, "CsPbCl3": 10.591915}
HALIDE  = {"CsPbI3": "I", "CsPbBr3": "Br", "CsPbCl3": "Cl"}


# ---------------------------------------------------------------------------
# Model loading (architecture inferred from the checkpoint)
# ---------------------------------------------------------------------------

def load_lsd_model(ckpt_path):
    """Build a Net_celu_HeInit_decayGaussian_LSD whose layer widths and
    gaussian_std are inferred from the saved state_dict, then load the weights.
    N_ref (absent from the checkpoint) is left unset and filled in later."""
    sd = torch.load(ckpt_path, map_location="cpu")
    # collect the sub-network linear weights in layer order
    ws = []
    i = 0
    while f"neural_network.hidden_l.{i}.weight" in sd:
        ws.append(sd[f"neural_network.hidden_l.{i}.weight"])
        i += 1
    if not ws:
        raise ValueError(f"{ckpt_path}: no 'neural_network.hidden_l.*.weight' found "
                         "-- is this a Net_celu_HeInit_decayGaussian_LSD checkpoint?")
    layers = [int(ws[0].shape[1])] + [int(w.shape[0]) for w in ws]   # [n_descr+1, ..., 1]
    gstd = float(sd["gaussian_std"]) if "gaussian_std" in sd else 1.0
    model = Net_celu_HeInit_decayGaussian_LSD(layers, gaussian_std=gstd)
    model.load_state_dict(sd, strict=False)   # N_ref not in ckpt -> set below
    model.eval()
    return model, layers[0] - 1               # (model, n_descr)


def find_ckpt(models_dir, el):
    for name in (f"final_{el}_LSDmodel.pth", f"{el}_LSDmodel.pth", f"init_{el}_LSDmodel.pth"):
        p = os.path.join(models_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No LSD checkpoint for element '{el}' in {models_dir} "
        f"(looked for final_{el}_LSDmodel.pth / {el}_LSDmodel.pth / init_{el}_LSDmodel.pth)")


# ---------------------------------------------------------------------------
# Bulk cubic reference descriptor  ->  N_ref  (periodic, matches the bulk fit)
# ---------------------------------------------------------------------------

def cubic_reference_descriptors(material, mace_model):
    """Per-element bulk reference descriptor N_ref, computed from the periodic
    cubic CsPbX3 cell -- the same reference the bulk LSD fit subtracted."""
    from ase import Atoms
    a = CUBIC_A[material]; X = HALIDE[material]
    cell_A = a * AUTOAA * np.eye(3)
    frac = np.array([[0.5, 0.5, 0.5],     # Cs
                     [0.0, 0.0, 0.0],     # Pb
                     [0.5, 0.0, 0.0],     # X
                     [0.0, 0.5, 0.0],
                     [0.0, 0.0, 0.5]])
    syms = ["Cs", "Pb", X, X, X]
    atoms = Atoms(symbols=syms, scaled_positions=frac, cell=cell_A, pbc=True)
    desc = compute_mace_descriptors(atoms, model=mace_model)        # [5, D]
    return {"Cs": torch.tensor(desc[0:1]),                          # [1, D] each
            "Pb": torch.tensor(desc[1:2]),
            X:    torch.tensor(desc[2:3])}


def cubic_reference_from_par(par_path, mace_model):
    """N_ref from an explicit bulk cubic system_*.par (exactly the bulk fit's
    reference geometry)."""
    from utils.read import BulkSystem
    bs = BulkSystem(); bs.systemName = "ref"
    bs.setSystem(par_path)
    from utils.mace_descriptors import bulk_to_ase
    desc = compute_mace_descriptors(bulk_to_ase(bs), model=mace_model)
    syms = np.array([s for s in bs.atomTypes])
    out = {}
    for el in np.unique(syms):
        out[str(el)] = torch.tensor(desc[syms == el][0:1])
    return out


# ---------------------------------------------------------------------------
# Real-space radial transform of the q-space correction (matches gen_lsd_pots.py)
# ---------------------------------------------------------------------------

def realSpacePot(vq, qSpacePot, nRGrid, rmax):
    dq = vq[1] - vq[0]
    vr = torch.linspace(0, rmax, nRGrid, device=vq.device)
    vq_ = vq.flatten(); qp_ = qSpacePot.flatten()
    sin_term  = torch.sin(vr[1:, None] * vq_[None, :])
    prefactor = 4 * np.pi * dq / (8 * np.pi**3 * vr[1:])
    bulk      = prefactor * (sin_term * (vq_ * qp_)[None, :]).sum(dim=1)
    r0 = (4 * np.pi * dq / (8 * np.pi**3)) * (vq_**2 * qp_).sum()
    rSpacePot = torch.cat([r0.unsqueeze(0), bulk])
    return vr.view(-1, 1), rSpacePot.view(-1, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("conf", help="nanocrystal geometry: conf.par (N_atoms; then 'Sym x y z' in Bohr)")
    ap.add_argument("material", choices=sorted(CUBIC_A), help="CsPbI3 | CsPbBr3 | CsPbCl3")
    ap.add_argument("--models", default="results_lsd_train_mace_2",
                    help="dir with final_<el>_LSDmodel.pth (default: results_lsd_train_mace_2)")
    ap.add_argument("--ref-par", default=None,
                    help="bulk cubic system_*.par for N_ref (default: build cubic cell from lattice const)")
    ap.add_argument("--mace-model", default="medium", help="MACE-MP-0 size (default: medium)")
    ap.add_argument("--surface-method", choices=["coordination", "bbox"], default="coordination",
                    help="how to find surface atoms: 'coordination' (shape-agnostic, default) "
                         "or 'bbox' (bounding-box extrema; matches the hand-crafted pipeline, "
                         "only valid for cuboidal NCs)")
    ap.add_argument("--surface-threshold", type=float, default=None,
                    help="bbox method: exempt atoms within this many Bohr of the bounding box "
                         "(default: one MACE cutoff radius)")
    ap.add_argument("--coord-rcut", type=float, default=None,
                    help="coordination method: neighbour radius in Bohr (default: one MACE cutoff radius)")
    ap.add_argument("--coord-frac", type=float, default=0.95,
                    help="coordination method: interior if neighbour count >= frac*bulk (default 0.95)")
    ap.add_argument("--no-surface-exempt", action="store_true",
                    help="apply the correction to ALL atoms (no surface exemption)")
    ap.add_argument("--qmax", type=float, default=30.0)
    ap.add_argument("--nq", type=int, default=4096)
    ap.add_argument("--rmax", type=float, default=120.0)
    ap.add_argument("--nr", type=int, default=4096)
    ap.add_argument("--outdir", default="LSD")
    ap.add_argument("--plot", action="store_true", help="also write a per-atom PDF (slow for big NCs)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    symbols, positions = read_conf_par(args.conf)     # positions in Bohr
    n_atoms = len(symbols)
    elems = sorted(set(symbols))
    print(f"Read {n_atoms} atoms from '{args.conf}'  (material {args.material}, elements {elems})")

    # --- load per-element LSD nets --------------------------------------------
    models, n_descr = {}, None
    for el in elems:
        models[el], nd = load_lsd_model(find_ckpt(args.models, el))
        n_descr = nd if n_descr is None else n_descr
        print(f"  loaded LSD net for {el}: input n_descr={nd}, gaussian_std={float(models[el].gaussian_std):.3g}")

    # --- bulk cubic reference -> N_ref ---------------------------------------
    if args.ref_par:
        Nref = cubic_reference_from_par(args.ref_par, args.mace_model)
        print(f"  N_ref from {args.ref_par}")
    else:
        Nref = cubic_reference_descriptors(args.material, args.mace_model)
        print(f"  N_ref from analytic cubic {args.material} cell (a={CUBIC_A[args.material]} Bohr)")
    for el in elems:
        if el not in Nref:
            raise KeyError(f"reference has no descriptor for element '{el}'")
        models[el].N_ref = Nref[el].detach().clone()

    # --- MACE descriptors for the nanocrystal (non-periodic) -----------------
    from ase import Atoms
    nc = Atoms(symbols=symbols, positions=positions * AUTOAA, pbc=False)
    desc = compute_mace_descriptors(nc, model=args.mace_model)       # [n_atoms, D]
    desc = torch.tensor(desc)
    if desc.shape[1] != n_descr:
        raise ValueError(f"MACE descriptor width {desc.shape[1]} != net input width {n_descr}")

    # --- surface exemption ----------------------------------------------------
    # one MACE cutoff radius (Bohr): the receptive-field scale over which the
    # bulk-trained descriptor must be intact for the correction to be valid.
    from utils.mace_descriptors import load_mace_model
    r_cut_bohr = float(getattr(load_mace_model(model=args.mace_model), "r_max", 6.0)) / AUTOAA
    if args.no_surface_exempt:
        surf = np.zeros(n_atoms, dtype=bool)
        print("  surface exemption: OFF (correcting all atoms)")
    elif args.surface_method == "bbox":
        thr = args.surface_threshold if args.surface_threshold is not None else r_cut_bohr
        surf = compute_surface_mask(positions, threshold=thr)
        print(f"  surface exemption: bbox, threshold={thr:.2f} Bohr -> "
              f"{int(surf.sum())}/{n_atoms} exempted (Delta v = 0)")
    else:
        rc = args.coord_rcut if args.coord_rcut is not None else r_cut_bohr
        surf = coordination_surface_mask(positions, r_cut=rc, frac=args.coord_frac)
        print(f"  surface exemption: coordination, r_cut={rc:.2f} Bohr, frac={args.coord_frac} -> "
              f"{int(surf.sum())}/{n_atoms} exempted (Delta v = 0)")

    # --- per-atom Delta v(q) and Delta v(r) ----------------------------------
    qGrid = torch.linspace(0.0, args.qmax, args.nq).view(-1, 1)
    manifest = []
    print(f"Writing per-atom LSD potentials to '{args.outdir}/' ...")
    for a, sym in enumerate(symbols):
        if surf[a]:
            qSpacePot = torch.zeros_like(qGrid)
        else:
            N = desc[a:a+1].repeat(args.nq, 1)                      # [nq, D]
            x = torch.cat([N, qGrid], dim=1)                       # [nq, D+1]
            with torch.no_grad():
                qSpacePot = models[sym](x)                          # [nq, 1]
        vr, rSpacePot = realSpacePot(qGrid.view(-1), qSpacePot, args.nr, args.rmax)

        potq = torch.cat((qGrid, qSpacePot), dim=1).detach().numpy()
        potr = torch.cat((vr, rSpacePot), dim=1).detach().numpy()
        np.savetxt(f"{args.outdir}/pot_q_LSD_{sym}{a}.dat", potq, delimiter='  ', fmt='%e')
        np.savetxt(f"{args.outdir}/pot_LSD_{sym}{a}.dat",   potr, delimiter='  ', fmt='%e')
        manifest.append((a, sym, int(surf[a]), float(qSpacePot.abs().max())))

        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(potr[:, 0], potr[:, 1], lw=2, label=f"{sym}{a}{' (surf)' if surf[a] else ''}")
            ax.set_xlim(0, 10); ax.set_xlabel("r [Bohr]"); ax.set_ylabel(r"$\Delta v^{loc}$ [a.u.]")
            ax.legend(frameon=False); fig.tight_layout()
            fig.savefig(f"{args.outdir}/pot_LSD_{sym}{a}.pdf"); plt.close(fig)

    # manifest: one line per atom -> easy to see which atoms were corrected
    with open(f"{args.outdir}/lsd_manifest.dat", "w") as fh:
        fh.write("# idx  symbol  surface(1=exempt)  max|Delta v(q)|\n")
        for a, sym, s, mx in manifest:
            fh.write(f"{a:6d}  {sym:3s}  {s:d}  {mx:.6e}\n")
    n_corr = sum(1 for _, _, s, _ in manifest if s == 0)
    print(f"Done. Corrected {n_corr}/{n_atoms} (interior) atoms; "
          f"wrote {2*n_atoms} .dat files + lsd_manifest.dat to '{args.outdir}/'.")


if __name__ == "__main__":
    main()
