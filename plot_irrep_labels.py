"""
Put double-group irrep labels on a band-structure plot at the high-symmetry
points. Reads band structures that ALREADY EXIST (the .dat/.par files) and the
irrep labels straight out of a QE bands.x output (bands_post.out). No Hamiltonian,
no model -- this is just parsing + plotting.

USAGE (run from the DeePseudopot directory):
  <python> plot_irrep_labels.py \
      --ref      ../bromide/inputs_cubic_all/expBandStruct_0.par \
      --model    ../bromide/results_cubic_all/initZunger_BS_sys0.dat \
      --kpoints  ../bromide/inputs_cubic_all/kpoints_0.par \
      --qe       /.../calc_bands/bromide/cubic/bands_post.out \
      --emin -4 --emax 4 --out ../bromide/results_cubic_all/irrep_bands.pdf

Labels are placed at the REFERENCE band energies (from --ref), which are the QE
symmetries; the model bands lie next to them. Where the QE alignment is clean the
label is QE-anchored; otherwise it falls back to bottom-up multiplet pairing (the
point is flagged '~' in the printed table and legend).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.symmetry.qe_labels import parse_bands_post, _match_kfrac, align_reference_to_qe


def _kname(kf):
    """Pm-3m point name from fractional coords (fold each comp to [0,1/2], sort)."""
    red = np.array([min(abs(x % 1.0), 1.0 - abs(x % 1.0)) for x in kf])
    a, b, c = np.sort(red)
    H, t = 0.5, 1e-3
    eq = lambda u, v: abs(u - v) < t
    if eq(a, 0) and eq(b, 0) and eq(c, 0): return 'Gamma'
    if eq(a, 0) and eq(b, 0) and eq(c, H): return 'X'
    if eq(a, 0) and eq(b, H) and eq(c, H): return 'M'
    if eq(a, H) and eq(b, H) and eq(c, H): return 'R'
    return None


def _pretty(label):
    """'Gamma8+' -> $\\Gamma_8^+$, 'E3/2-' -> $E_{3/2}^-$. Falls back to raw."""
    try:
        s, sup = label.strip(), ''
        if s.endswith('+'): sup, s = '^+', s[:-1]
        elif s.endswith('-'): sup, s = '^-', s[:-1]
        s = s.replace('Gamma', r'\Gamma')
        i = len(s)
        while i > 0 and (s[i - 1].isdigit() or s[i - 1] == '/'):
            i -= 1
        head, sub = s[:i], s[i:]
        return f'${head}{("_{"+sub+"}") if sub else ""}{sup}$'
    except Exception:
        return label


def _labels_at_point(E_ref_row, blk):
    """[(energy, label)] for the reference bands at one high-sym point, plus a flag
    'QE' (clean align) or '~' (bottom-up fallback). E_ref_row: ascending ref
    energies at this k; blk: the matched QE block from parse_bands_post."""
    E = np.sort(np.asarray(E_ref_row, float))
    try:
        frozen, _diag = align_reference_to_qe(E, blk['multiplets'],
                                              warn=(lambda *a, **k: None))
        return [(m['energy'], m['label']) for m in frozen], 'QE'
    except Exception:
        # bottom-up: consume `degen` reference bands per QE multiplet, in order.
        out, i = [], 0
        for m in blk['multiplets']:
            d = m['degen']
            if i + d > len(E):
                break
            out.append((float(E[i:i + d].mean()), m['label']))
            i += d
        return out, '~'


def _kpath_x(kpoints_frac):
    """Cumulative distance along the path (matches plot_bands.py)."""
    d = np.linalg.norm(np.diff(kpoints_frac, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(d)))


def main(a):
    ref = np.loadtxt(a.ref)
    mod = np.loadtxt(a.model)
    kpts = np.loadtxt(a.kpoints)[:, :3]
    ref[:, 1:] -= a.fermi
    mod[:, 1:] -= a.fermi
    xk = _kpath_x(kpts)                                   # one x per k-point

    qe_blocks = parse_bands_post(a.qe)
    points = [(i, _kname(kf), kf) for i, kf in enumerate(kpts) if _kname(kf)]
    print(f"High-symmetry points: {[(i, n) for (i, n, _k) in points]}")

    labels_by_i = {}
    for (i, name, kf) in points:
        blk = _match_kfrac(kf, qe_blocks)
        if blk is None:
            print(f"  {name} (k{i}): no QE block found -- skipping labels here.")
            continue
        E_ref_row = ref[i, 1:] + a.fermi                 # absolute frame for QE align
        labeled, src = _labels_at_point(E_ref_row, blk)
        labeled = [(e - a.fermi, lab) for (e, lab) in labeled]   # back to plot frame
        labels_by_i[i] = (name, labeled, src)
        print(f"\n=== {name} (k{i})  [{src}] ===")
        for (e, lab) in labeled:
            if a.emin <= e <= a.emax:
                print(f"   {e:8.3f}  {lab}")

    # ---- one plot: model bands (blue) + reference bands (red), with labels -------
    fig, ax = plt.subplots(figsize=(9, 6))
    done = False
    for ib in range(1, mod.shape[1]):
        col = mod[:, ib]
        if (col >= a.emin).any() and (col <= a.emax).any():
            ax.plot(xk, col, '-', color='tab:blue', lw=0.8,
                    label=None if done else 'model')
            done = True
    done = False
    for ib in range(1, ref.shape[1]):
        col = ref[:, ib]
        if (col >= a.emin).any() and (col <= a.emax).any():
            ax.plot(xk, col, '.', color='tab:red', ms=6,
                    label=None if done else 'reference')
            done = True

    tick_x = [xk[i] for (i, _n, _k) in points]
    ax.set_xticks(tick_x)
    ax.set_xticklabels([_pretty(n) for (_i, n, _k) in points])
    for x in tick_x:
        ax.axvline(x, color='0.85', lw=0.8, zorder=0)
    ax.set_xlim(xk.min(), xk.max())
    ax.set_ylim(a.emin, a.emax)
    ax.set_ylabel(f"E{'' if a.fermi == 0 else f' - {a.fermi:g}'}  (eV)")
    ax.legend(loc='upper right', fontsize=8)

    def _spread(ys, gap):
        """Nudge sorted y-positions apart to at least `gap`, preserving order and
        staying near the originals (so labels don't overprint)."""
        ys = list(ys)
        for _ in range(200):
            moved = False
            for k in range(1, len(ys)):
                if ys[k] - ys[k - 1] < gap:
                    shift = (gap - (ys[k] - ys[k - 1])) / 2
                    ys[k - 1] -= shift
                    ys[k] += shift
                    moved = True
            if not moved:
                break
        return ys

    xmid = 0.5 * (xk.min() + xk.max())
    xspan = xk.max() - xk.min()
    gap = (a.emax - a.emin) / 34.0                       # min vertical label spacing
    any_fallback = False
    for (i, (name, labeled, src)) in labels_by_i.items():
        any_fallback |= (src == '~')
        vis = sorted([(e, lab) for (e, lab) in labeled if a.emin <= e <= a.emax])
        if not vis:
            continue
        right = xk[i] > xmid
        x_text = xk[i] + (-0.03 if right else 0.03) * xspan   # text column beside the point
        ha = 'right' if right else 'left'
        ys = _spread([e for (e, _l) in vis], gap)
        for (ty, (e, lab)) in zip(ys, vis):
            ax.plot([xk[i], x_text], [e, ty], '-', lw=0.4, color='0.6', zorder=2)
            ax.text(x_text, ty, _pretty(lab), fontsize=7, color='black',
                    va='center', ha=ha, zorder=3)

    title = "Double-group irreps at high-symmetry points"
    if any_fallback:
        title += "  (~ = bottom-up fallback, verify)"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="reference band structure (expBandStruct_0.par)")
    p.add_argument("--model", required=True, help="model band structure (initZunger_BS_sys0.dat)")
    p.add_argument("--kpoints", required=True, help="kpoints_0.par (fractional coords)")
    p.add_argument("--qe", required=True, help="QE bands.x output with symmetry labels (bands_post.out)")
    p.add_argument("--emin", type=float, default=-4.0)
    p.add_argument("--emax", type=float, default=4.0)
    p.add_argument("--fermi", type=float, default=0.0, help="energy zero shift (eV)")
    p.add_argument("--out", default="irrep_bands.pdf")
    main(p.parse_args())
