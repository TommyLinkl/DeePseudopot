"""
Generate ``expBandIrreps_0.par`` -- the QE-anchored irrep label of every REFERENCE
band, aligned to ``expBandStruct_0.par``, so the reference band symmetries are a
tracked, diff-able artifact instead of a value recomputed at freeze time.

This is a SETUP-TIME, torch-free step: it needs only the reference spectrum
(``expBandStruct_0.par``), the k-points (``kpoints_0.par``), and the cubic QE
``bands.x`` output (``lsym=.true.``). No Hamiltonian, no model -- the label of a
reference band comes purely from :func:`qe_labels.align_reference_to_qe`, exactly
the same call the training freeze uses when ``irrep_reference_labels`` is set.

Only the high-symmetry POINTS (Gamma, X, M, R) carry QE point-group labels; a
k-point on a high-symmetry LINE is written as ``line:<Name>`` (its little group)
and a general k as ``-``. The file mirrors ``expBandStruct_0.par`` row-for-row
(one row per k-point, ascending band order), so column j of row i is the irrep of
the energy in the same cell of ``expBandStruct_0.par``.

Usage:
  PYTHONPATH=<DeePseudopot> python -m utils.symmetry.gen_expbandirreps \
      <inputs_folder> [path/to/bands_post.out] [system_index]

If the QE path is omitted it is read from NN_config.par's
``irrep_reference_labels`` (resolved relative to the inputs folder, like the fit).
"""
import os
import sys

import numpy as np

from .qe_labels import parse_bands_post, _match_kfrac, align_reference_to_qe


# ---- standard Pm-3m k-point name from fractional coords (points AND lines) ----
def _kname(kf):
    """Fold each component to [0,1/2] and sort -> symmetry-robust Pm-3m label.
    Returns (name, is_point). Mirrors train_hooks.IrrepContext._kname so the file
    and the training freeze agree on which k is which."""
    red = np.array([min(abs(x % 1.0), 1.0 - abs(x % 1.0)) for x in kf])
    a, b, c = np.sort(red)
    H, t = 0.5, 1e-3
    def eq(u, v):
        return abs(u - v) < t
    # high-symmetry POINTS
    if eq(a, 0) and eq(b, 0) and eq(c, 0):    return 'Gamma', True
    if eq(a, 0) and eq(b, 0) and eq(c, H):    return 'X', True
    if eq(a, 0) and eq(b, H) and eq(c, H):    return 'M', True
    if eq(a, H) and eq(b, H) and eq(c, H):    return 'R', True
    # high-symmetry LINES
    if eq(a, 0) and eq(b, 0):                 return 'Delta', False    # G-X
    if eq(a, 0) and eq(b, c):                 return 'Sigma', False    # G-M
    if eq(a, b) and eq(c, H):                 return 'S', False        # X-R
    if eq(a, 0) and eq(c, H):                 return 'Z', False        # X-M
    if eq(a, b) and eq(b, c):                 return 'Lambda', False   # G-R
    if eq(b, H) and eq(c, H):                 return 'T', False        # M-R
    return f"k={np.round(kf, 3)}", False


def _read_expbandstruct(path):
    """Rows = k-points; col 0 = k-path distance, cols 1: = ascending band energies."""
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:]        # (nk,), (nk, nBands)


def _read_kpoints(path):
    kp = np.loadtxt(path)
    return kp[:, :3]                      # (nk, 3) fractional


def build_labels(inputs, qe_path=None, degen_tol=1e-4):
    """Return (kdist, kfrac, labels, info) where labels is an (nk, nBands) array of
    label strings and info[kidx] holds the per-point alignment diagnostics."""
    inputs = inputs.rstrip('/') + '/'
    kdist, E = _read_expbandstruct(inputs + 'expBandStruct_0.par')
    kfrac = _read_kpoints(inputs + 'kpoints_0.par')
    nk, nBands = E.shape
    if kfrac.shape[0] != nk:
        raise ValueError(f"kpoints ({kfrac.shape[0]}) and expBandStruct ({nk}) "
                         f"disagree on the number of k-points.")

    qe_blocks = parse_bands_post(qe_path)
    labels = np.full((nk, nBands), '-', dtype=object)
    info = {}
    for kidx in range(nk):
        name, is_point = _kname(kfrac[kidx])
        if not is_point:
            labels[kidx, :] = f"line:{name}"
            info[kidx] = {'name': name, 'point': False}
            continue
        blk = _match_kfrac(kfrac[kidx], qe_blocks)
        if blk is None:
            labels[kidx, :] = f"{name}:noQE"
            info[kidx] = {'name': name, 'point': True, 'matched': False}
            continue
        frozen, diag = align_reference_to_qe(E[kidx], blk['multiplets'],
                                             degen_tol=degen_tol, warn=None)
        row = np.full(nBands, f"{name}:?", dtype=object)
        for m in frozen:
            for b in m['members']:
                row[b] = m['label']
        labels[kidx, :] = row
        info[kidx] = {'name': name, 'point': True, 'matched': True,
                      'point_group': blk['point_group'], 'diag': diag,
                      'n_labeled': sum(len(m['members']) for m in frozen)}
    return kdist, kfrac, labels, info


def write_par(path, kdist, labels):
    """Write the label matrix, one row per k-point, whitespace-separated, with the
    k-path distance as the first column (mirrors expBandStruct_0.par)."""
    nk, nBands = labels.shape
    width = max(8, max(len(str(x)) for x in labels.ravel()) + 1)
    with open(path, 'w') as fh:
        fh.write(f"# QE-anchored irrep label of each reference band, aligned row-for-row\n")
        fh.write(f"# to expBandStruct_0.par ({nk} k-points x {nBands} bands). "
                 f"'line:<N>' = high-sym line; '-' = general k.\n")
        fh.write(f"# col0 = k-path distance; cols 1..{nBands} = band irreps (ascending).\n")
        for kidx in range(nk):
            cells = "".join(f"{str(labels[kidx, b]):>{width}}" for b in range(nBands))
            fh.write(f"{kdist[kidx]:>10.6f}{cells}\n")


def _print_point_summary(kfrac, labels, info, E=None, window=None):
    print("\n=== reference band irreps at the high-symmetry POINTS ===")
    for kidx, meta in info.items():
        if not meta.get('point') or not meta.get('matched'):
            continue
        d = meta['diag']
        flag = "" if d['ok'] else "  [!] loose energy alignment -- CHECK"
        print(f"\n  {meta['name']} (kidx {kidx}, {meta['point_group']}): "
              f"QE offset {d['offset']} (dropped {d['n_qe_dropped']} deep multiplets), "
              f"shift {d['shift']:.3f} eV, residual spread {d['spread']:.3g} eV{flag}")
        seq = labels[kidx]
        lo, hi = window if window else (0, len(seq) - 1)
        toks = " ".join(f"{b}:{seq[b]}" for b in range(lo, hi + 1))
        print(f"      bands {lo}-{hi}: {toks}")


def main(inputs, qe_path=None, isys=0, window=None):
    inputs = inputs.rstrip('/') + '/'
    if qe_path is None:
        # resolve from NN_config.par, relative to the inputs folder like the fit
        cfg = {}
        with open(inputs + 'NN_config.par') as fh:
            for line in fh:
                if '=' in line and not line.strip().startswith('#'):
                    k, v = line.split('=', 1)
                    cfg[k.strip()] = v.split('#')[0].strip()
        rel = cfg.get('irrep_reference_labels')
        if not rel:
            raise SystemExit("No QE path: pass it as arg 2 or set "
                             "irrep_reference_labels in NN_config.par")
        for cand in (rel, os.path.join(inputs, rel)):
            if os.path.exists(cand):
                qe_path = cand
                break
        if qe_path is None:
            raise SystemExit(f"irrep_reference_labels='{rel}' not found relative to "
                             f"'{inputs}' or cwd.")
    print(f"[gen] inputs   : {inputs}")
    print(f"[gen] QE labels: {qe_path}")

    kdist, kfrac, labels, info = build_labels(inputs, qe_path)
    out = f"{inputs}expBandIrreps_{isys}.par"
    write_par(out, kdist, labels)
    _print_point_summary(kfrac, labels, info, window=window)
    print(f"\n[gen] wrote {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    args = sys.argv[1:]
    win = None
    # optional --window lo hi
    if '--window' in args:
        i = args.index('--window')
        win = (int(args[i + 1]), int(args[i + 2]))
        del args[i:i + 3]
    inputs = args[0]
    qe = args[1] if len(args) > 1 else None
    isys = int(args[2]) if len(args) > 2 else 0
    main(inputs, qe, isys, window=win)
