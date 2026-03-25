import numpy as np
import os
import re
import matplotlib.pyplot as plt

# ============================================================
#  NEW: q-dependent coupling plots (one PDF per atom)
# ============================================================

def parse_coupling_file_all_q(filename, band_choice):
    """
    Parse coupling file where each polarization line contains
    values for ALL q-points on one line.

    Returns:
        atoms: list of atom names
        vals: array of shape (Natoms, 3, Nq)
              vals[i_atom, comp, iq]
    """
    atoms = []
    vals = []

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if "atom =" in lines[i]:
            atom = re.search(r"atom\s=\s(\w+)", lines[i]).group(1)
            atoms.append(atom)

            # move to cb-cb or vb-vb block based on band choice
            while i < len(lines) and f"{band_choice}-{band_choice} coupling elements" not in lines[i]:
                i += 1

            comps = []
            for _ in range(3):
                i += 1  # skip "polarization = ..."
                qvals = [float(x) for x in lines[i].split()]
                comps.append(qvals)
                i += 1

            vals.append(comps)
        else:
            i += 1

    return atoms, np.array(vals)


def load_q_distances(qfile):
    """
    Load fractional q-vectors and return cumulative |dq|
    """
    q = np.loadtxt(qfile)[:, :3]
    q = q[::-1]
    dq = np.linalg.norm(np.diff(q, axis=0), axis=1)
    qdist = np.concatenate([[0.0], np.cumsum(dq)])
    return qdist


# --- Load q-point distances ---
qdist = load_q_distances("qpoints_0.par")

# --- Loop over bands (vb/cb) --- 
band_choices = ["vb", "cb"]
for band in band_choices:
    print(f"Plotting {band} couplings...")
    # --- Load full-q coupling data ---
    ref_atoms_q, ref_vals_q = parse_coupling_file_all_q("expCoupling_0.par", band)
    cmp_atoms_q, cmp_vals_q = parse_coupling_file_all_q("initZunger_couplingBands_0.dat", band)

    if ref_atoms_q != cmp_atoms_q:
        print("WARNING: atom lists differ between reference and computed!")

    atoms = ref_atoms_q
    Nat, _, Nq = ref_vals_q.shape

    pol_labels = ["x", "y", "z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    
    # --- One plot per atom ---
    found_atom_dict = {atom: 0 for atom in atoms}
    for ia, atom in enumerate(atoms):
        found_atom_dict[atom] += 1 
        atom_str = f"{atom}{found_atom_dict[atom]}"

        fig, ax = plt.subplots(figsize=(4, 5))

        for comp in range(3):
            # Reference
            ax.scatter(
                qdist,
                ref_vals_q[ia, comp, ::-1],
                color=colors[comp],
                marker="^",
                label=f"Ref {pol_labels[comp]}"
            )

            # Computed
            ax.plot(
                qdist,
                cmp_vals_q[ia, comp, ::-1],
                color=colors[comp],
                linestyle="--",
                label=f"Computed {pol_labels[comp]}"
            )

        ax.set_xlabel(r"$|q|$", fontsize=18)
        ax.set_ylabel(rf"g$^{{{atom_str}}}_{{{band}}}$", fontsize=18)
        ax.set_title(f"Coupling vs q for atom {atom_str}", fontsize=15)
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"coupling_vs_q_{atom_str}_{band}.pdf")
        plt.close()

print(f"Done.")