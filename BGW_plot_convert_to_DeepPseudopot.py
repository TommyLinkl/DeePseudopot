#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_bgw_bandstructure(filename, energy_col="eqp", spin=None):
    """
    Load BGW bandstructure.dat and return k-points and band energies.

    Expected columns:
      spin, band, kx, ky, kz, E(MF), E(QP), Delta E
    """
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 8:
        raise ValueError("Expected at least 8 columns in bandstructure.dat.")

    if spin is not None:
        data = data[data[:, 0] == spin]
        if data.size == 0:
            raise ValueError(f"No rows found for spin {spin}.")

    band_vals = data[:, 1].astype(int)
    _, first_indices = np.unique(band_vals, return_index=True)
    band_order = band_vals[np.sort(first_indices)]

    first_band = band_order[0]
    kpoints = data[band_vals == first_band][:, 2:5]
    nk = len(kpoints)
    nb = len(band_order)

    if nk == 0 or nb == 0:
        raise ValueError("No k-points or bands detected in the file.")

    expected_rows = nb * nk
    if data.shape[0] != expected_rows:
        raise ValueError(
            f"Unexpected row count. Got {data.shape[0]}, expected {expected_rows} "
            f"(nb={nb}, nk={nk})."
        )

    if energy_col.lower() == "eqp":
        energies = data[:, 6].reshape((nb, nk))
    elif energy_col.lower() in ("emf", "mf"):
        energies = data[:, 5].reshape((nb, nk))
    else:
        raise ValueError("energy_col must be 'eqp' or 'emf'.")

    return kpoints, energies, band_order


def compute_k_distances(kpoints):
    distances = [0.0]
    for i in range(1, len(kpoints)):
        dk = np.linalg.norm(kpoints[i] - kpoints[i - 1])
        distances.append(distances[-1] + dk)
    return np.array(distances)


def convert_for_deeppseudopot(k_dist, bands, write_to="forDeepPseudopot.dat", max_bands=None):
    nb = bands.shape[0]
    use_bands = nb if max_bands is None else min(max_bands, nb)
    out = np.zeros((len(k_dist), use_bands + 1))
    out[:, 0] = k_dist
    for ib in range(use_bands):
        out[:, ib + 1] = bands[ib]
    np.savetxt(write_to, out, fmt="%.5f")


def plot_bandstructure(k_dist, bands, out_plot, y_limits=None):
    for band in bands:
        plt.plot(k_dist, band, color="blue", lw=1)
    plt.xlabel("k-path distance")
    plt.ylabel("Energy (eV)")
    plt.xlim(k_dist[0], k_dist[-1])
    if y_limits is not None:
        plt.ylim(y_limits[0], y_limits[1])
    ax = plt.gca()
    ax.grid(True, axis="y", color="0.8", linestyle="-")
    ax.grid(True, axis="x", color="0.8", linestyle="--")
    plt.tight_layout()
    plt.savefig(out_plot)


def parse_qe_cell_and_kpoints(qe_input):
    cell = []
    kpoints = []
    weights = []
    labels = []
    cell_units = None
    with open(qe_input, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("CELL_PARAMETERS"):
            if "{" in line and "}" in line:
                cell_units = line.split("{", 1)[1].split("}", 1)[0].strip().lower()
            i += 1
            for _ in range(3):
                parts = lines[i].split()
                cell.append([float(parts[0]), float(parts[1]), float(parts[2])])
                i += 1
            continue
        if line.startswith("K_POINTS"):
            if "crystal_b" not in line:
                raise ValueError("Expected K_POINTS crystal_b in qe input.")
            i += 1
            nk = int(lines[i].split()[0])
            i += 1
            for _ in range(nk):
                raw = lines[i].strip()
                i += 1
                if not raw:
                    continue
                if "!" in raw:
                    coords_part, label_part = raw.split("!", 1)
                    label = label_part.strip()
                else:
                    coords_part = raw
                    label = ""
                parts = coords_part.split()
                if len(parts) < 4:
                    continue
                kpoints.append([float(parts[0]), float(parts[1]), float(parts[2])])
                weights.append(int(float(parts[3])))
                labels.append(label)
            continue
        i += 1

    if len(cell) != 3:
        raise ValueError("CELL_PARAMETERS block not found or incomplete.")
    if len(kpoints) == 0:
        raise ValueError("No K_POINTS crystal_b found in qe input.")

    return np.array(cell), np.array(kpoints), np.array(weights), labels, cell_units


def parse_qe_structure(qe_input):
    cell = []
    cell_units = None
    atoms = []
    pos_units = None
    with open(qe_input, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("CELL_PARAMETERS"):
            if "{" in line and "}" in line:
                cell_units = line.split("{", 1)[1].split("}", 1)[0].strip().lower()
            i += 1
            for _ in range(3):
                parts = lines[i].split()
                cell.append([float(parts[0]), float(parts[1]), float(parts[2])])
                i += 1
            continue
        if line.startswith("ATOMIC_POSITIONS"):
            if "{" in line and "}" in line:
                pos_units = line.split("{", 1)[1].split("}", 1)[0].strip().lower()
            i += 1
            while i < len(lines):
                raw = lines[i].strip()
                if raw == "" or raw.startswith("K_POINTS"):
                    break
                parts = raw.split()
                if len(parts) >= 4:
                    atoms.append(
                        (parts[0], float(parts[1]), float(parts[2]), float(parts[3]))
                    )
                i += 1
            continue
        i += 1

    if len(cell) != 3:
        raise ValueError("CELL_PARAMETERS block not found or incomplete.")
    if len(atoms) == 0:
        raise ValueError("ATOMIC_POSITIONS block not found or empty.")
    if pos_units is None:
        raise ValueError("ATOMIC_POSITIONS units not found.")

    return np.array(cell), cell_units, atoms, pos_units


def reciprocal_from_cell(cell, cell_units):
    a1, a2, a3 = cell
    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)
    volume = np.dot(a1, np.cross(a2, a3))
    b1 = 2.0 * np.pi * np.cross(a2, a3) / volume
    b2 = 2.0 * np.pi * np.cross(a3, a1) / volume
    b3 = 2.0 * np.pi * np.cross(a1, a2) / volume
    if cell_units is None:
        raise ValueError("CELL_PARAMETERS units not found; specify --cell-units.")
    if cell_units == "angstrom":
        b1 = b1 * 0.529177210903
        b2 = b2 * 0.529177210903
        b3 = b3 * 0.529177210903
    elif cell_units == "bohr":
        pass
    else:
        raise ValueError(f"Unsupported CELL_PARAMETERS units: {cell_units}")
    return np.column_stack((b1, b2, b3))


def cell_to_bohr(cell, cell_units):
    if cell_units is None:
        raise ValueError("CELL_PARAMETERS units not found; specify --cell-units.")
    if cell_units == "angstrom":
        return cell * 1.889726124565062
    if cell_units == "bohr":
        return cell
    raise ValueError(f"Unsupported CELL_PARAMETERS units: {cell_units}")


def positions_to_fractional(atoms, pos_units, cell_bohr):
    if pos_units == "crystal":
        return atoms
    if pos_units == "angstrom":
        cart = np.array([[a[1], a[2], a[3]] for a in atoms]) * 1.889726124565062
    elif pos_units == "bohr":
        cart = np.array([[a[1], a[2], a[3]] for a in atoms])
    else:
        raise ValueError(f"Unsupported ATOMIC_POSITIONS units: {pos_units}")
    frac = np.linalg.inv(cell_bohr.T) @ cart.T
    frac = frac.T
    return [(atoms[i][0], frac[i][0], frac[i][1], frac[i][2]) for i in range(len(atoms))]


def write_system_par(qe_input, out_file="system_0.par", cell_units_override=None):
    cell, cell_units, atoms, pos_units = parse_qe_structure(qe_input)
    if cell_units_override is not None:
        cell_units = cell_units_override
    cell_bohr = cell_to_bohr(cell, cell_units)
    atoms_frac = positions_to_fractional(atoms, pos_units, cell_bohr)
    scale = 1.0
    with open(out_file, "w") as f:
        f.write(f"scale = {scale:.6f}\n\n")
        f.write("cell\n")
        for row in cell_bohr:
            f.write(f"{row[0]:.8f} {row[1]:.8f} {row[2]:.8f}\n")
        f.write("\n")
        f.write("atoms\n")
        for sym, x, y, z in atoms_frac:
            f.write(f"{sym} {x:.8f} {y:.8f} {z:.8f}\n")


def expand_qe_kpath(kpoints_red, weights):
    if len(kpoints_red) != len(weights):
        raise ValueError("K_POINTS and weights length mismatch.")
    if len(kpoints_red) == 0:
        return np.zeros((0, 3)), []
    expanded = []
    special_indices = []
    idx = 0
    for i in range(len(kpoints_red) - 1):
        start = kpoints_red[i]
        end = kpoints_red[i + 1]
        n = int(weights[i])
        if n < 1:
            continue
        special_indices.append(idx)
        for j in range(n):
            # QE crystal_b: n points from start toward end, excluding the end.
            t = 0.0 if n == 1 else j / float(n)
            kp = (1.0 - t) * start + t * end
            expanded.append(kp)
            idx += 1
    special_indices.append(idx)
    expanded.append(kpoints_red[-1])
    return np.array(expanded), special_indices


def qe_kpath_to_cart(qe_input, cell_units_override=None):
    cell, kpoints_red, weights, labels, cell_units = parse_qe_cell_and_kpoints(qe_input)
    if cell_units_override is not None:
        cell_units = cell_units_override
    recip = reciprocal_from_cell(cell, cell_units)
    expanded_red, special_indices = expand_qe_kpath(kpoints_red, weights)
    expanded_cart = (recip @ expanded_red.T).T
    return expanded_cart, expanded_red, special_indices, labels, recip

def write_kpoints_par(kpoints, filename="kpoints.par"):
    ones = np.ones((kpoints.shape[0], 1))
    data = np.hstack((kpoints, ones))
    np.savetxt(filename, data, fmt="%.8f")

def sparse_indices(n, factor, special_indices):
    if factor is None or factor <= 1.0:
        return np.arange(n, dtype=int)
    special_set = set(int(i) for i in special_indices)
    special_set.update([0, n - 1])
    target = int(np.round(n / factor))
    target = max(target, len(special_set))
    base = np.unique(np.round(np.linspace(0, n - 1, target)).astype(int))
    selected = sorted(set(base).union(special_set))
    if len(selected) > target:
        while len(selected) > target:
            best_pos = None
            best_gap = None
            for pos in range(1, len(selected) - 1):
                idx = selected[pos]
                if idx in special_set:
                    continue
                gap = selected[pos + 1] - selected[pos - 1]
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_pos = pos
            if best_pos is None:
                break
            selected.pop(best_pos)
    return np.array(selected, dtype=int)

def pad_missing_bands(bands, missing, fill_value=-1000.0):
    if missing <= 0:
        return bands
    pad = np.full((missing, bands.shape[1]), fill_value, dtype=bands.dtype)
    return np.vstack([pad, bands])


def interleave_energy_columns(bands, repeats=2):
    """
    Repeat each band (column in the expBandStruct files) along axis 0.

    Parameters
    ----------
    bands : np.ndarray
        Array with shape (nbands, nkpoints).
    repeats : int
        How many times to repeat each band. Default is 2.

    Returns
    -------
    np.ndarray
        Array with nbands * repeats rows (unless repeats <= 1).
    """
    if repeats <= 1 or bands.size == 0:
        return bands
    return np.repeat(bands, repeats, axis=0)


def apply_scissors_operator(bands, threshold=None, shift=0.0):
    """
    Apply a scissors operator: energies strictly above threshold are shifted by shift.
    """
    if threshold is None or shift == 0.0:
        return bands
    return np.where(bands > threshold, bands + shift, bands)


def main():
    parser = argparse.ArgumentParser(
        description="Plot BGW bandstructure.dat and convert to DeepPseudopot format."
    )
    parser.add_argument(
        "--input",
        default="bandstructure.dat",
        help="Input BGW bandstructure.dat file.",
    )
    parser.add_argument(
        "--qe-input",
        default="qe_bands.in",
        help="QE input file with CELL_PARAMETERS and K_POINTS crystal_b.",
    )
    parser.add_argument(
        "--cell-units",
        choices=["angstrom", "bohr"],
        default=None,
        help="Override CELL_PARAMETERS units if not specified in qe input.",
    )
    parser.add_argument(
        "--energy",
        default="eqp",
        choices=["eqp", "emf"],
        help="Which energy column to use for plotting/conversion.",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=None,
        help="Spin channel to select (default: use all).",
    )
    parser.add_argument(
        "--max-bands",
        type=int,
        default=None,
        help="Maximum number of bands to write to DeepPseudopot file.",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=0.0,
        help="Constant energy shift applied to all bands (eV).",
    )
    parser.add_argument(
        "--scissors-threshold",
        type=float,
        default=None,
        help="Apply scissors shift to energies strictly above this threshold (eV). "
        "Threshold is evaluated after --shift.",
    )
    parser.add_argument(
        "--scissors-shift",
        type=float,
        default=0.0,
        help="Scissors energy shift (eV) applied above --scissors-threshold.",
    )
    parser.add_argument(
        "--out-plot",
        default="bandstructure.pdf",
        help="Output plot filename.",
    )
    parser.add_argument(
        "--out-data",
        default="forDeepPseudopot_expBandStruct_0.par",
        help="Output DeepPseudopot data filename.",
    )
    parser.add_argument(
        "--out-kpoints",
        default="forDeepPseudopot_kpoints_0.par",
        help="Output k-points filename for DeepPseudopot.",
    )
    parser.add_argument(
        "--out-system",
        default="system_0.par",
        help="Output system file with lattice and atoms.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Y-axis minimum (optional).",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Y-axis maximum (optional).",
    )
    parser.add_argument(
        "--sparse-factor",
        type=float,
        default=None,
        help="Downsample factor for DeepPseudopot outputs only (e.g., 2, 2.5, 3).",
    )
    parser.add_argument(
        "--missing-bands",
        type=int,
        default=None,
        help="Override number of missing low bands to pad with -1000.0.",
    )
    parser.add_argument(
        "--remove-deepest-bands",
        type=int,
        default=0,
        help="Number of lowest-energy bands to remove from the input before output.",
    )
    parser.add_argument(
        "--interleave-energies",
        action="store_true",
        help="Interleave (repeat) every band column twice in expBandStruct outputs. "
        "The k-path column is never modified.",
    )
    args = parser.parse_args()

    kpoints, bands, band_order = load_bgw_bandstructure(
        args.input, energy_col=args.energy, spin=args.spin
    )
    k_dist = compute_k_distances(kpoints)
    original_min_band = int(np.min(band_order))

    if args.missing_bands is not None and args.missing_bands < 0:
        raise ValueError("--missing-bands must be >= 0.")
    if args.remove_deepest_bands < 0:
        raise ValueError("--remove-deepest-bands must be >= 0.")
    if args.remove_deepest_bands >= bands.shape[0]:
        raise ValueError(
            f"--remove-deepest-bands ({args.remove_deepest_bands}) must be less than "
            f"the number of available bands ({bands.shape[0]})."
        )
    if args.remove_deepest_bands > 0:
        bands = bands[args.remove_deepest_bands :, :]
        band_order = band_order[args.remove_deepest_bands :]

    if args.shift != 0.0:
        bands = bands + args.shift

    if args.scissors_shift != 0.0 and args.scissors_threshold is None:
        raise ValueError(
            "--scissors-threshold must be set when --scissors-shift is non-zero."
        )
    bands = apply_scissors_operator(
        bands, threshold=args.scissors_threshold, shift=args.scissors_shift
    )

    y_limits = None
    if args.ymin is not None or args.ymax is not None:
        if args.ymin is None or args.ymax is None:
            raise ValueError("Both --ymin and --ymax must be set to use y-limits.")
        y_limits = (args.ymin, args.ymax)

    qe_cart, qe_frac, indices, labels, recip = qe_kpath_to_cart(
        args.qe_input, cell_units_override=args.cell_units
    )
    write_system_par(args.qe_input, out_file=args.out_system, cell_units_override=args.cell_units)

    kpoints_frac = qe_frac

    if len(qe_cart) != len(kpoints):
        print(
            f"Warning: QE-expanded kpoints ({len(qe_cart)}) != BGW kpoints ({len(kpoints)}). "
            "Using the minimum length to align."
        )
        min_len = min(len(qe_cart), len(kpoints))
        k_dist = k_dist[:min_len]
        bands = bands[:, :min_len]
        kpoints = kpoints[:min_len]
        kpoints_frac = kpoints_frac[:min_len]
        qe_cart = qe_cart[:min_len]
        indices = [i for i in indices if i < min_len]

    def maybe_interleave(band_array):
        return (
            interleave_energy_columns(band_array, repeats=2)
            if args.interleave_energies
            else band_array
        )

    for band in bands:
        plt.plot(k_dist, band, color="blue", lw=1)
    plt.xlabel("k-path distance")
    plt.ylabel("Energy (eV)")
    plt.xlim(k_dist[0], k_dist[-1])
    if y_limits is not None:
        plt.ylim(y_limits[0], y_limits[1])
    ax = plt.gca()
    ax.grid(True, axis="y", color="0.8", linestyle="-")
    ax.grid(True, axis="x", color="0.8", linestyle="--")

    if indices:
        special_x = k_dist[indices]
        pretty_labels = []
        for label in labels:
            if label.lower().startswith("gamma") or label == "G":
                pretty_labels.append(r"$\Gamma$")
            else:
                pretty_labels.append(label)
        plt.xticks(special_x, pretty_labels)
        for sx in special_x[1:-1]:
            plt.axvline(sx, color="0.6", lw=0.8, ls="--")

    plt.tight_layout()
    plt.savefig(args.out_plot)

    bands_for_output = maybe_interleave(bands)
    convert_for_deeppseudopot(
        k_dist, bands_for_output, write_to=args.out_data, max_bands=args.max_bands
    )
    write_kpoints_par(kpoints_frac, filename=args.out_kpoints)

    missing = None
    if args.missing_bands is not None:
        missing = args.missing_bands
    else:
        missing = max(0, original_min_band - 1)

    if args.sparse_factor is not None:
        if args.sparse_factor < 1.0:
            raise ValueError("--sparse-factor must be >= 1.0")
        sparse_idx = sparse_indices(len(k_dist), args.sparse_factor, indices)
        k_dist_out = k_dist[sparse_idx]
        bands_out = bands[:, sparse_idx]
        bands_out = maybe_interleave(bands_out)
        kpoints_out = kpoints_frac[sparse_idx]
        bands_out = pad_missing_bands(bands_out, missing)
        convert_for_deeppseudopot(
            k_dist_out,
            bands_out,
            write_to="sparser_expBandStruct_0.par",
            max_bands=args.max_bands,
        )
        write_kpoints_par(kpoints_out, filename="sparser_kpoints_0.par")
    bands_full_prepad = bands_for_output
    bands_full_out = pad_missing_bands(bands_full_prepad, missing)
    convert_for_deeppseudopot(
        k_dist, bands_full_out, write_to=args.out_data, max_bands=args.max_bands
    )
    write_kpoints_par(kpoints_frac, filename=args.out_kpoints)


if __name__ == "__main__":
    main()
