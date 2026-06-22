"""
Plot the spin-orbit / non-local (NL) pseudopotential prefactors vs training epoch.

Reads the `epoch_*_nonlocalParams.dat` files emitted by bandStruct_train_GPU
(written by write_nonlocal_params). Each file holds, per trained atom, a header
    # <atom>  (trained indices = [5, 6, 7])
followed by the full 9-entry PPparams vector. Only the trained indices are
plotted (idx 5 = SOC, 6 = NL1, 7 = NL2).

Usage:
    python plot_nonlocal_params.py [resultsFolder] [-o out.pdf] [--show]

    resultsFolder   directory containing epoch_*_nonlocalParams.dat (default: ./)
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt

# Human-readable names for the PPparams indices that can be trained.
INDEX_NAMES = {5: "SOC", 6: "NL1", 7: "NL2"}

HEADER_RE = re.compile(r"#\s*(\S+)\s*\(trained indices\s*=\s*\[([0-9,\s]*)\]\)")


def parse_file(path):
    """Return ({atom: full param vector (np.array)}, {atom: [trained indices]})."""
    params, trained = {}, {}
    atom = None
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = HEADER_RE.match(line)
            if m:
                if atom is not None:
                    params[atom] = np.array(vals, dtype=float)
                atom = m.group(1)
                idx_str = m.group(2).strip()
                trained[atom] = [int(x) for x in idx_str.split(",") if x.strip() != ""]
                vals = []
            elif line.startswith("#"):
                continue
            else:
                vals.append(float(line))
    if atom is not None:
        params[atom] = np.array(vals, dtype=float)
    return params, trained


def epoch_of(path):
    m = re.search(r"epoch_(\d+)_nonlocalParams\.dat", os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("resultsFolder", nargs="?", default="./",
                    help="directory with epoch_*_nonlocalParams.dat (default: ./)")
    ap.add_argument("-o", "--output", default=None,
                    help="output figure path (default: <resultsFolder>/nonlocal_params_vs_epoch.pdf)")
    ap.add_argument("--show", action="store_true", help="display the figure interactively")
    args = ap.parse_args()

    folder = args.resultsFolder
    files = glob.glob(os.path.join(folder, "epoch_*_nonlocalParams.dat"))
    files = sorted((f for f in files if epoch_of(f) is not None), key=epoch_of)
    if not files:
        raise SystemExit(f"No epoch_*_nonlocalParams.dat files found in '{folder}'.")

    epochs = [epoch_of(f) for f in files]
    # series[atom][idx] -> list of values aligned with `epochs`
    series = {}
    trained_idx = {}
    for path in files:
        params, trained = parse_file(path)
        for atom, vec in params.items():
            trained_idx.setdefault(atom, trained.get(atom, []))
            for idx in trained_idx[atom]:
                series.setdefault(atom, {}).setdefault(idx, []).append(vec[idx])

    atoms = sorted(series.keys())
    # Columns = the union of all trained indices (each gets its OWN y-scale so a
    # small drift on top of a large offset is still visible). Rows = atoms.
    all_idx = sorted({idx for atom in atoms for idx in series[atom]})
    print(f"Found {len(files)} epoch files (epochs {epochs[0]}-{epochs[-1]}) "
          f"for atoms {atoms}, indices {all_idx}.")

    nrow, ncol = len(atoms), len(all_idx)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow),
                             sharex=True, squeeze=False)
    for r, atom in enumerate(atoms):
        for c, idx in enumerate(all_idx):
            ax = axes[r, c]
            name = INDEX_NAMES.get(idx, f"idx{idx}")
            if idx in series[atom]:
                y = series[atom][idx]
                ax.plot(epochs, y, marker=".", markersize=4, linewidth=1.0)
                # annotate net change to make "still moving vs flat" quantitative
                ax.set_title(f"{atom}: {name} (PPparams[{idx}])  Δ={y[-1]-y[0]:+.4g}",
                             fontsize=9)
            else:
                ax.set_visible(False)
                continue
            ax.grid(True, alpha=0.3)
            if c == 0:
                ax.set_ylabel("prefactor value")
            if r == nrow - 1:
                ax.set_xlabel("epoch")
    fig.suptitle("SOC / non-local prefactors vs epoch")
    fig.tight_layout()

    out = args.output or os.path.join(folder, "nonlocal_params_vs_epoch.pdf")
    fig.savefig(out)
    png = os.path.splitext(out)[0] + ".png"
    fig.savefig(png, dpi=150)
    print(f"Saved {out} and {png}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
