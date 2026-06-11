#!/usr/bin/env python3
"""
gen_lsd_dataset.py
==================

Build the LSD pre-training dataset for the iodide (CsPbI3) system.

The LSD ("local structure dependent") potential is a per-atom correction to the
base *cubic* pseudopotential.  Rather than learning it directly from (expensive)
band-structure fits, we pre-train the LSD neural networks to reproduce the
difference between

    v_atom(q)   in a distorted structure        (read from a per-structure fit)
    v_type(q)   in the cubic reference geometry  (one curve per element)

i.e.  Delta v(q) = v_atom(q) - v_cubic_type(q).

Each distorted atom is tagged with its structural descriptors N_alpha (the same
descriptors the LSD network takes as input), so the network learns the map
N_alpha  ->  Delta v(q).

This script:
  1. Reads the cubic reference v(q) per element from RESULTS_CUBIC/final_pot_q_*.dat
  2. For every system (cubic = system 0, then each distorted fit = system 1..N):
       a. converts its POSCAR -> inputs_lsd_train/system_<iSys>.par
       b. computes per-atom descriptors with the *production* code
          (BulkSystem.compute_descriptors), so ordering/dedup match training time
       c. reads each atom's converged v(q) from its own
          <results_dir>/final_pot_q_<el><idx>.dat file
       d. dedups atoms by descriptor (unique_within_tolerance, rtol=RTOL) -- the
          identical call init_LSD_PP makes -- and writes one difference file
          per unique environment:  pot_q_<el>_diff_<iSys>_<i>.par
       e. writes a diagnostic .pdf overlay per element
  3. Copies the static inputs (NN_config, base PP model + params, and placeholder
     band-structure inputs) so the directory is runnable with main.py.

Run from this directory:   python gen_lsd_dataset.py

NOTE on grids: every fit's final_pot_q_*.dat is written by
FT_converge_and_write_pp on the q grid set in NN_config (qmax / nQGrid).  As long
as the distorted fits and the cubic reference used the same qmax / nQGrid, all
potentials share one grid and are subtracted directly -- no interpolation.  This
script asserts each distorted final_pot_q grid matches the cubic reference grid
and errors out otherwise.
"""

import os
import re
import glob
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

torch.set_default_dtype(torch.float64)

from utils.read import BulkSystem
from utils.init_LSD_train import env_keep_mask
from utils.constants import AUTOAA

# ----------------------------------------------------------------------------- #
#  Configuration                                                                #
# ----------------------------------------------------------------------------- #
HERE          = os.path.dirname(os.path.abspath(__file__))
SYSTEM_NAME   = "CsPbI3"
ELEMENTS      = ["Cs", "I", "Pb"]          # == atomPPOrder = np.unique(atomTypes)
RTOL          = 0.05                        # MUST match init_LSD_PP's dedup tolerance

CUBIC_DIR     = os.path.join(HERE, "results_cubic_q-0.95_iter1_all")
DISTORTED_DIRS = sorted(glob.glob(os.path.join(HERE, "results_0[0-9][0-9]_g_1")))

OUT_DIR       = os.path.join(HERE, "inputs_lsd_train")

# Static files copied from the worked example so the directory runs end-to-end.
EXAMPLE_DIR   = os.path.join(HERE, "inputs_lsd")


def _read_backend():
    """descriptor_backend from the run config (OUT_DIR) if it exists, else the
    template -- so dataset generation matches what training will read."""
    for cfg in (os.path.join(OUT_DIR, "NN_config.par"),
                os.path.join(EXAMPLE_DIR, "NN_config.par")):
        if os.path.exists(cfg):
            for line in open(cfg):
                s = line.split('#')[0].strip()
                if s.startswith('descriptor_backend'):
                    return s.split('=', 1)[1].strip()
    return 'handcrafted'

BACKEND = _read_backend()
# Per-system band-structure inputs are placeholders for the *pre-training* stage
# (NN_config max_num_epochs = 0); replace them with the real targets before any
# band-structure (max_num_epochs > 0) training.
BS_TEMPLATE_CUBIC     = "0"   # example index whose geometry is the 5-atom cubic cell
BS_TEMPLATE_DISTORTED = "1"   # example index whose geometry is a 20-atom cell


# ----------------------------------------------------------------------------- #
#  POSCAR  ->  system_<i>.par                                                    #
# ----------------------------------------------------------------------------- #
def strip_type(label):
    """'Cs0' -> 'Cs', 'I11' -> 'I', 'Pb3' -> 'Pb'."""
    return re.match(r"[A-Za-z]+", label).group(0)


def read_poscar(poscar_path):
    """Return (cell_AA [3,3], types [N] stripped to element, cart_AA [N,3])."""
    with open(poscar_path) as f:
        lines = [ln.rstrip("\n") for ln in f]

    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale

    name_tokens  = lines[5].split()
    count_tokens = [int(x) for x in lines[6].split()]

    # Coordinate mode line (Cartesian / Direct), then the positions.
    mode = lines[7].strip().lower()
    n_atoms = sum(count_tokens)
    coords = np.array([[float(x) for x in lines[8 + i].split()[:3]] for i in range(n_atoms)])

    # Expand the (element-name, count) header into a per-atom type list.
    types = []
    for tok, cnt in zip(name_tokens, count_tokens):
        types.extend([strip_type(tok)] * cnt)
    types = np.array(types)

    if mode.startswith("d"):                     # direct/fractional -> cartesian
        coords = coords @ cell

    return cell, types, coords


def write_system_par(poscar_path, out_path):
    """Convert a POSCAR (Angstrom, Cartesian) to a system_*.par (Bohr, fractional).

    Follows the QE-celldm convention BulkSystem.setSystem expects: `scale` is the
    lattice parameter a = |a1| (in Bohr) and the cell card lists the lattice
    vectors in units of a (identity for a cubic cell). The reader rebuilds the
    physical cell as scale * cell, so the geometry is identical either way.
    """
    cell_AA, types, cart_AA = read_poscar(poscar_path)

    # Fractional coordinates are unit independent.
    frac = cart_AA @ np.linalg.inv(cell_AA)
    cell_bohr = cell_AA / AUTOAA                  # Angstrom -> Bohr (a.u.)

    # scale = a = |a1| (Bohr); express the lattice vectors in units of a.
    scale = np.linalg.norm(cell_bohr[0])
    cell_scaled = cell_bohr / scale

    with open(out_path, "w") as f:
        f.write(f"scale = {scale:.12f}\n\n")
        f.write("cell\n")
        for row in cell_scaled:
            f.write(f"{row[0]:.12f}\t{row[1]:.12f}\t{row[2]:.12f}\n")
        f.write("\natoms\n")
        for t, fr in zip(types, frac):
            f.write(f"{t} {fr[0]:.12f} {fr[1]:.12f} {fr[2]:.12f}\n")
    return types


# ----------------------------------------------------------------------------- #
#  final_pot_q reading                                                           #
# ----------------------------------------------------------------------------- #
# Each fit writes one final_pot_q_<label>.dat per pseudopotential channel
# (FT_converge_and_write_pp).  Distorted fits use a per-atom atomPPOrder, so the
# labels are 'Cs0'..'Cs3', 'I0'..'I11', 'Pb0'..'Pb3'.  The cubic reference fit
# uses per-element channels, so its labels are 'Cs', 'I', 'Pb'.  All are sampled
# on the shared q grid (NN_config qmax / nQGrid).
def read_pot_q(path, master_q=None):
    """Load a 2-column final_pot_q_*.dat -> (q, v); optionally assert grid match."""
    d = np.loadtxt(path)
    q, v = d[:, 0], d[:, 1]
    if master_q is not None and (q.shape[0] != master_q.shape[0]
                                 or not np.allclose(q, master_q)):
        raise ValueError(
            f"{os.path.basename(path)} grid (n={q.shape[0]}, qmax={q[-1]:.4f}) does "
            f"not match the master grid (n={master_q.shape[0]}, qmax={master_q[-1]:.4f}). "
            f"Regenerate all fits with the same NN_config qmax/nQGrid."
        )
    return q, v


def read_cubic_reference(master_q):
    """Cubic v_type(q) per element on the master grid (no interpolation)."""
    ref = {}
    for el in ELEMENTS:
        _, v = read_pot_q(os.path.join(CUBIC_DIR, f"final_pot_q_{el}.dat"), master_q)
        ref[el] = v
    return ref


# ----------------------------------------------------------------------------- #
#  Descriptors                                                                   #
# ----------------------------------------------------------------------------- #
def descriptors_for(system_par_path):
    """Build a BulkSystem from a system_*.par and return env_descriptors (numpy)."""
    sys = BulkSystem()
    sys.systemName = SYSTEM_NAME                  # needed: descriptors index ORTHO_REF[material]
    sys.setSystem(system_par_path)
    sys.compute_descriptors(backend=BACKEND)
    return {el: sys.env_descriptors[el] for el in ELEMENTS if el in sys.env_descriptors}


# ----------------------------------------------------------------------------- #
#  Main dataset generation                                                       #
# ----------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Master q grid = the shared fit grid (NN_config qmax/nQGrid), taken from the
    # cubic reference; every distorted final_pot_q file is checked against it.
    master_q, _ = read_pot_q(os.path.join(CUBIC_DIR, f"final_pot_q_{ELEMENTS[0]}.dat"))
    cubic_ref = read_cubic_reference(master_q)

    # System list: cubic is system 0 (sets the LSD reference N_ref); distorted follow.
    systems = [("cubic", os.path.join(CUBIC_DIR, "0.POSCAR"), None)]
    for d in DISTORTED_DIRS:
        systems.append((os.path.basename(d), os.path.join(d, "0.POSCAR"), d))

    summary = []
    for iSys, (tag, poscar, results_dir) in enumerate(systems):
        sys_par = os.path.join(OUT_DIR, f"system_{iSys}.par")
        write_system_par(poscar, sys_par)
        descr = descriptors_for(sys_par)

        for el in ELEMENTS:
            desc_el = descr[el]                              # torch [n_atoms_el, n_descr]
            keep = env_keep_mask(desc_el, BACKEND, rtol=RTOL)   # same call as init_LSD_PP
            kept_positions = [r for r in range(keep.shape[0]) if bool(keep[r])]

            fig, ax = plt.subplots()
            for i, r in enumerate(kept_positions):
                if results_dir is None:           # cubic: every atom == reference -> diff 0
                    diff = np.zeros_like(master_q)
                else:
                    # per-atom potential of the r-th atom of element el
                    _, v_atom = read_pot_q(
                        os.path.join(results_dir, f"final_pot_q_{el}{r}.dat"), master_q)
                    diff = v_atom - cubic_ref[el]

                out = np.column_stack([master_q, diff])
                np.savetxt(os.path.join(OUT_DIR, f"pot_q_{el}_diff_{iSys}_{i}.par"), out)

                ax.plot(master_q, diff, label=f"{el}{r}")

            ax.set_xlim(0.0, 8.0)
            ax.set_xlabel(r"q (a.u.$^{-1}$)")
            ax.set_ylabel(r"$\Delta v(q)$ (a.u.)")
            ax.set_title(f"system {iSys} ({tag}) — {el}: {len(kept_positions)} unique env")
            ax.legend(fontsize=6, ncol=2)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"pot_q_{el}_diff_{iSys}.pdf"))
            plt.close(fig)

            summary.append((iSys, tag, el, desc_el.shape[0], len(kept_positions)))

    # ---- static / template inputs so the directory runs with main.py ---------- #
    write_static_inputs(n_systems=len(systems))

    # ---- report --------------------------------------------------------------- #
    print("\n" + "=" * 64)
    print(f"Wrote LSD pre-training dataset to {OUT_DIR}")
    print(f"{'sys':>3} {'tag':<16} {'el':<3} {'n_atoms':>7} {'n_unique':>8}")
    for iSys, tag, el, n_at, n_uni in summary:
        print(f"{iSys:>3} {tag:<16} {el:<3} {n_at:>7} {n_uni:>8}")
    print("=" * 64)


def write_static_inputs(n_systems):
    """Copy NN_config (patched), base PP model + params, and per-system BS inputs."""
    # NN_config.par: write from the template only if OUT_DIR doesn't already have
    # one -- preserve a hand-tuned run config across regenerations.
    out_cfg = os.path.join(OUT_DIR, "NN_config.par")
    if os.path.exists(out_cfg):
        print(f"Keeping existing {out_cfg} (not overwriting your tuned config).")
    else:
        with open(os.path.join(EXAMPLE_DIR, "NN_config.par")) as f:
            cfg = f.read()
        cfg = re.sub(r"^nSystem\s*=.*$", f"nSystem = {n_systems}", cfg, flags=re.M)
        with open(out_cfg, "w") as f:
            f.write(cfg)

    # Base cubic pseudopotential (3 channels: Cs, I, Pb) and its Zunger params.
    shutil.copy(os.path.join(CUBIC_DIR, "initZunger_PPmodel.pth"),
                os.path.join(OUT_DIR, "init_PPmodel.pth"))
    for el in ELEMENTS:
        shutil.copy(os.path.join(EXAMPLE_DIR, f"init_{el}Params.par"),
                    os.path.join(OUT_DIR, f"init_{el}Params.par"))

    # Per-system band-structure inputs (PLACEHOLDERS for pre-training).
    for iSys in range(n_systems):
        tmpl = BS_TEMPLATE_CUBIC if iSys == 0 else BS_TEMPLATE_DISTORTED
        for kind in ("input", "kpoints", "expBandStruct", "bandWeights"):
            shutil.copy(os.path.join(EXAMPLE_DIR, f"{kind}_{tmpl}.par"),
                        os.path.join(OUT_DIR, f"{kind}_{iSys}.par"))

    # Do NOT copy init_<el>_LSDmodel.pth: their absence forces fresh pre-training
    # (NN_config also has init_LSD_force_retrain = 1).


if __name__ == "__main__":
    main()
