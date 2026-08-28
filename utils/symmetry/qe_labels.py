"""
QE-anchored reference irrep labels (SETUP-TIME ONLY -- never in the hot path).

Parse a Quantum ESPRESSO ``bands.x`` output produced with ``lsym=.true.`` for the
cubic Pm-3m SOC cell and return, at each high-symmetry k-point, the ascending
sequence of double-group irrep multiplets ``(energy_eV, degeneracy, label)`` in
the SAME label convention as :mod:`utils.symmetry.irreps` -- Koster Gamma6/7/8
for O_h (Gamma, R) and E1/2 / E3/2 for the D_4h subgroups (X, M).

WHY THIS EXISTS
---------------
The sector loss needs every REFERENCE band tagged by irrep. The default freeze
(:func:`loss.freeze_reference_labels`) inherits each label from the *model* in
energy order, which is only as trustworthy as the initial potential's Gamma6 vs
Gamma7 ordering. When a genuine cubic QE ``bands.x`` run with ``lsym=.true.``
exists -- here the 5-atom Pm-3m cell, Pb at origin, ``lspinorb`` (exactly the fit
cell: see ``system_0.par``) -- its "Band symmetry" section carries the TRUE
labels. This module reads them and aligns them to the fit's ``expBandStruct`` so
the freeze is anchored to first-principles symmetry instead of the model.

CONVENTION MATCH (verified against QE's own printed character tables, not assumed)
---------------------------------------------------------------------------------
QE prints the double-group character table inside each block. In BOTH O_h and
D_4h, QE's ``G_6`` is the branch with chi(C4) = +sqrt(2) and ``G_7`` the
chi(C4) = -sqrt(2) branch (read straight off the "2C4"/"6C4" column of the
tables in the output). :func:`irreps.DoubleGroupIrreps._assign_labels` DEFINES
Gamma6 (= E1/2 in the subgroups) as the physical D^{1/2} branch, chi(C4)=+sqrt(2),
and Gamma7 (= E3/2) as its A2 partner, chi(C4)=-sqrt(2). The conventions
therefore coincide exactly:

    O_h  : G_6+/- -> Gamma6+/- ,  G_7+/- -> Gamma7+/- ,  G_8+/- -> Gamma8+/-
    D_4h : G_6+/- -> E1/2+/-   ,  G_7+/- -> E3/2+/-

The +/- superscript is QE's g/u inversion parity, identical to the model's
inversion-character suffix.

ALIGNMENT
---------
``expBandStruct`` is QE's spectrum with the deep semicore bands dropped from the
bottom and a rigid energy shift applied (VBM/work-function alignment; an optional
scissor on the conduction bands). The reference multiplet sequence is therefore a
*contiguous sub-sequence* of QE's multiplet sequence with the SAME order and the
SAME degeneracies. :func:`align_reference_to_qe` finds the drop offset by matching
the full reference degeneracy pattern to QE's and checking that the implied
per-multiplet energy shift is (piecewise-)constant -- so no shift/scissor value
ever has to be reconstructed. At an all-doublet point (M, X: D_4h has no 4-dim
irrep) the degeneracy pattern is uniform, so the offset is pinned by the energy
shift alone.
"""
import re

import numpy as np

from .characters import group_degenerate

# --- line parsers ----------------------------------------------------------
_XK = re.compile(r"xk=\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*\)")
_PG = re.compile(r"double point group\s+(\S+)")
_ELINE = re.compile(
    r"e\(\s*(\d+)\s*-\s*(\d+)\)\s*=\s*([-\d.]+)\s*eV\s+(\d+)\s*-->\s*(.*)")
_GLABEL = re.compile(r"G_([678])([+-]?)")


def _qe_to_model_label(nchar, parity, point_group):
    """Map a QE Koster label (n in {6,7,8}, parity in {'+','-',''}) to the model
    convention for the given little group. See module header for the proof."""
    is_Oh = point_group.startswith("O_h")
    if nchar == "8":
        return f"Gamma8{parity}"
    if is_Oh:
        return f"Gamma{nchar}{parity}"           # Gamma6/Gamma7
    return ("E1/2" if nchar == "6" else "E3/2") + parity


def parse_bands_post(path):
    """Parse a ``bands.x`` output. Return a list of block dicts, one per k-point:

        {'kfrac': (a,b,c), 'point_group': str,
         'multiplets': [ {'energy': eV, 'degen': int,
                          'qe_label': 'G_6+', 'label': 'Gamma6+'} , ... ]}

    Only blocks that actually carry a "Band symmetry" listing are returned.
    """
    blocks = []
    cur = None
    with open(path) as fh:
        for line in fh:
            mxk = _XK.search(line)
            if mxk:
                if cur is not None and cur["multiplets"]:
                    blocks.append(cur)
                cur = {"kfrac": tuple(round(float(x), 6) for x in mxk.groups()),
                       "point_group": None, "multiplets": []}
                continue
            if cur is None:
                continue
            mpg = _PG.search(line)
            if mpg:
                cur["point_group"] = mpg.group(1)
                continue
            mel = _ELINE.search(line)
            if mel:
                _n1, _n2, energy, degen, tail = mel.groups()
                mg = _GLABEL.search(tail)
                if mg is None:
                    # a labelled line without a G_n token (e.g. a subgroup along a
                    # high-sym LINE); skip -- QE-anchored labels are for POINTS.
                    continue
                nchar, parity = mg.group(1), mg.group(2)
                cur["multiplets"].append({
                    "energy": float(energy),
                    "degen": int(degen),
                    "qe_label": mg.group(0),
                    "label": _qe_to_model_label(nchar, parity, cur["point_group"] or ""),
                })
    if cur is not None and cur["multiplets"]:
        blocks.append(cur)
    return blocks


def _match_kfrac(kfrac, blocks, tol=1e-3):
    """Find the QE block whose kfrac matches ``kfrac`` up to a reciprocal-lattice
    vector and the cubic point group (fold each component to [0,1/2], sort)."""
    def canon(k):
        red = [min(abs(x % 1.0), 1.0 - abs(x % 1.0)) for x in k]
        return tuple(sorted(red))
    target = canon(kfrac)
    for b in blocks:
        if all(abs(u - v) < tol for u, v in zip(canon(b["kfrac"]), target)):
            return b
    return None


def align_reference_to_qe(E_ref_k, qe_multiplets, degen_tol=1e-4,
                          shift_tol=0.05, warn=print):
    """Assign each REFERENCE band its QE irrep label.

    E_ref_k       : 1D ascending reference energies at this k (eV).
    qe_multiplets : that k-point's ``block['multiplets']`` from parse_bands_post
                    (ascending; each has 'energy','degen','label').
    Returns (frozen_labels, diag) where frozen_labels matches the shape of
    :func:`loss.freeze_reference_labels`:
        [{'members':[ref band idx...], 'energy':mean, 'label':str, 'dim':int}, ...]
    and diag = {'offset','shift','spread','n_qe_dropped','ok'}.

    Method: group the reference into multiplets, then slide that degeneracy
    pattern along the QE multiplet list. Accept the offset whose degeneracy
    pattern matches exactly AND whose per-multiplet energy residual
    (E_ref - E_qe) is most nearly constant (allowing ONE break for a
    valence/conduction scissor). The reference's TOP multiplet may be a partial
    tail (the fixed nBands window cuts a QE degenerate set): it is allowed to
    have fewer bands than its QE partner and is dropped, exactly as
    :func:`loss.freeze_reference_labels` does. No shift value is assumed.
    """
    E_ref_k = np.asarray(E_ref_k, float)
    groups = group_degenerate(E_ref_k, tol=degen_tol)
    ref_deg = [len(g) for g in groups]
    ref_e = np.array([E_ref_k[g].mean() for g in groups])
    nref = len(groups)

    qe_deg = [m["degen"] for m in qe_multiplets]
    qe_e = np.array([m["energy"] for m in qe_multiplets])
    nqe = len(qe_multiplets)
    if nqe < nref:
        raise ValueError(f"QE block has {nqe} multiplets but the reference needs "
                         f"{nref}. Wrong k-block or truncated bands.x output?")

    def residual_spread(shifts):
        """Spread of a shift array, allowing one contiguous break (scissor):
        min over split point of max(intra-segment peak-to-peak)."""
        best = np.ptp(shifts)
        for cut in range(1, len(shifts)):
            best = min(best, max(np.ptp(shifts[:cut]), np.ptp(shifts[cut:])))
        return best

    def deg_ok(o):
        """All reference multiplets match QE degen exactly, except the LAST which
        may be a partial tail (ref_deg[-1] <= qe_deg). Returns (ok, last_partial)."""
        if qe_deg[o:o + nref - 1] != ref_deg[:nref - 1]:
            return False, False
        last = qe_deg[o + nref - 1]
        if ref_deg[-1] == last:
            return True, False
        if ref_deg[-1] < last:                    # window cuts a QE multiplet
            return True, True
        return False, False

    best = None
    for o in range(0, nqe - nref + 1):
        ok_deg, last_partial = deg_ok(o)
        if not ok_deg:
            continue
        shifts = ref_e - qe_e[o:o + nref]
        # a partial top multiplet still sits at the same energy (all members are
        # degenerate in QE), so its residual is meaningful and kept in the spread.
        spread = residual_spread(shifts)
        if best is None or spread < best[1]:
            best = (o, spread, shifts, last_partial)

    if best is None:
        raise ValueError(
            "no QE offset reproduces the reference degeneracy pattern "
            f"{ref_deg[:12]}... -- QE run and reference disagree on symmetry "
            "(wrong k-point, wrong cell, or a genuine wrong-basin reference).")

    o, spread, shifts, last_partial = best
    ok = spread <= shift_tol
    if not ok and warn:
        warn(f"    [qe-label] WARNING: residual shift spread {spread:.4g} eV > "
             f"{shift_tol} at this k -- degeneracy pattern matched at offset {o} "
             f"but energies drift; check the QE<->reference correspondence.")

    n_keep = nref - 1 if last_partial else nref
    if last_partial and warn:
        m = qe_multiplets[o + nref - 1]
        warn(f"    [qe-label] reference window cuts QE multiplet '{m['label']}' "
             f"(QE dim {m['degen']}, have {ref_deg[-1]}); dropping partial tail.")
    frozen = []
    for i in range(n_keep):
        m = qe_multiplets[o + i]
        frozen.append({"members": list(groups[i]), "energy": float(ref_e[i]),
                       "label": m["label"], "dim": ref_deg[i]})
    diag = {"offset": o, "shift": float(np.median(shifts[:n_keep])),
            "spread": float(spread), "n_qe_dropped": o,
            "last_partial": last_partial, "ok": ok}
    return frozen, diag


def compare_labels(frozen_qe, model_seq, degen_tol=1e-4):
    """Compare QE-anchored labels to the model's energy-ordered labels multiplet
    by multiplet. Returns (n_match, n_total, disagreements) where each
    disagreement is (index, qe_label, model_label). model_seq is the
    (energy,label,dim) list from :func:`hot_path.irrep_sequence` (ascending)."""
    disagreements = []
    n = min(len(frozen_qe), len(model_seq))
    for i in range(n):
        ql = frozen_qe[i]["label"]
        ml = model_seq[i][1]
        if ql != ml:
            disagreements.append((i, ql, ml))
    return n - len(disagreements), n, disagreements
