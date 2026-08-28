"""
Reference labeling (section 6) + sector-wise loss (section 7).

REFERENCE LABELING -- "the convention trap"
-------------------------------------------
We do NOT match on label strings and we do NOT trust QE's double-group labels
(the reference here comes from a 20-atom orthorhombic SOC supercell run with
nosym/noinv, so its .rap file is unusable; see project notes). Instead:

  * dimension is unambiguous: a 4-fold reference multiplet at Gamma/R MUST be
    Gamma8; 2-folds are Gamma6 or Gamma7 (or E_{1/2}/E_{3/2} in the subgroups).
  * Gamma6 vs Gamma7 and parity g/u come from the MODEL's block labels: we align
    the reference multiplets to the model's irrep sequence IN ENERGY ORDER, once,
    using a physically-sensible initial potential, and FREEZE the assignment.

This is done at setup (freeze_reference_labels). At train time the sector loss
uses the frozen assignment; if the model wanders into a wrong basin the per-sector
STATE COUNTS stop matching and assert_sector_counts raises loudly -- which both
detects ghosts and localizes them to one sector (section 7/8).

BLOCK-MATCHED BAND MSE (replaces the old separate "sector loss")
----------------------------------------------------------------
There is NO separate irrep objective. Irreducibility enters the ONE band-structure
MSE only by changing HOW model eigenvalues are matched to reference bands at the
high-symmetry k-points: within each irrep block (by ascending energy) instead of on
the globally-sorted list. Concretely, `windowed_band_assignment` builds a per-band
(label, rank) map; the training hook (train_hooks.irrep_loss_term) uses it to
REORDER the model spectrum and then feeds it through the system's own band MSE --
same bandWeights, same kptWeights, same energy convention (absolute or
relative-to-band via relE_bIdx). No irrep_lambda, no per-sector d_lambda weight,
no separate energy zero: those knobs are gone by design.

CRITICAL: the within-block rank is counted only over the FITTING WINDOW
(bandWeights > 0) and offset by the MODEL's own below-window state count. The
earlier `frozen_band_assignment` (kept for reference) ranked every block from band
0 using the REFERENCE (QE) labels, so a weight-0 deep band whose QE parity the
smooth pseudopotential cannot reproduce shifted every frontier band of that irrep
by one rank -- pairing it with the wrong model level and blowing the "small
correction" up to ~10x the band loss. `windowed_band_assignment` fixes this.

`sector_loss_at_k` below is the OLD standalone weighted-MSE objective and is
DEPRECATED / no longer wired into training; kept only for reference and tests.
"""
import numpy as np
import torch

from .characters import group_degenerate


def freeze_reference_labels(E_ref_k, model_seq, degen_tol=1e-4, warn=print):
    """Label a reference spectrum at ONE high-symmetry k by assigning each
    reference BAND the model's per-band irrep label IN ENERGY ORDER, then
    collapsing each physical multiplet.

    This is robust to ACCIDENTAL degeneracies (two different irreps landing at
    nearly the same energy): we never group the reference by its own degeneracy
    pattern -- the model labels every band by construction, so we go purely by
    energy order. The reference window is taken from band 1 (the bottom), as the
    index-to-index matching requires.

    E_ref_k   : 1D array of reference energies (ascending) at this k.
    model_seq : list of (energy, label, dim) for the model's PHYSICAL multiplets
                at this k, ascending (from hot_path.irrep_sequence).
    Returns list of dicts, one per reference physical multiplet:
        {'members': [ref band indices], 'energy': mean, 'label': str, 'dim': int}.
    Raises only if the model sequence cannot cover the reference window.
    """
    E_ref_k = np.asarray(E_ref_k, float)
    nref = len(E_ref_k)
    # expand model multiplets into a per-band label/dim stream (dim copies each)
    per_band = []
    for (e, label, dim) in model_seq:
        per_band.extend([(label, dim)] * dim)
    if len(per_band) < nref:
        raise ValueError(
            f"model sequence covers only {len(per_band)} bands but the reference "
            f"window needs {nref}. Is this a symmetry k-point / is nBands right?")
    labels_out = []
    i = 0
    while i < nref:
        label, dim = per_band[i]
        members = list(range(i, min(i + dim, nref)))
        if len(members) < dim:
            # window cuts a multiplet -- drop the partial tail (health note)
            if warn:
                warn(f"    [irrep freeze] reference window cuts multiplet '{label}' "
                     f"(need {dim}, have {len(members)}); dropping partial tail at band {i}.")
            break
        # sanity: the d bands assigned to one multiplet should be near-degenerate
        spread = float(E_ref_k[members].max() - E_ref_k[members].min())
        if warn and dim > 1 and spread > 50 * degen_tol:
            warn(f"    [irrep freeze] WARNING: reference multiplet '{label}' at band {i} "
                 f"has spread {spread:.4g} > {50*degen_tol:.4g}; degeneracy/ordering "
                 f"may disagree with the model (possible wrong basin).")
        labels_out.append({'members': members, 'energy': float(E_ref_k[members].mean()),
                           'label': label, 'dim': dim})
        i += dim
    return labels_out


def reference_sector_energies(frozen_labels):
    """From frozen per-multiplet labels build {label: ascending energies array}
    (one entry per physical multiplet). Reference E_ref[lambda,n]."""
    sect = {}
    for m in frozen_labels:
        sect.setdefault(m['label'], []).append(m['energy'])
    return {lab: np.sort(np.array(v)) for lab, v in sect.items()}


def frozen_band_assignment(frozen_labels):
    """Per-reference-band map (ref_band_index, label, block_rank) used to REORDER
    the model spectrum for the band-structure MSE (section 7, block-matched form).

    Each reference band inherits the (irrep label, within-block rank) of its frozen
    multiplet, so at train time reference band i is compared to its OWN irrep block's
    rank-r model eigenvalue -- instead of the i-th globally-sorted one. This is what
    makes the ordinary band MSE robust to inter-irrep level crossings: a block-B
    state that slides below a block-A state stays matched to block-B, never to the
    reference band frozen as block-A.

    frozen_labels is ascending in energy (from freeze_reference_labels /
    align_reference_to_qe); the rank of a multiplet is its position among the
    same-label multiplets, ascending -- i.e. the index into
    block_eigvals(collapse=True)[label]. All members of one multiplet (degenerate
    bands) share the same (label, rank), so they map to the same model level.
    """
    assign = []
    rank_of = {}
    for m in frozen_labels:
        lab = m['label']
        r = rank_of.get(lab, 0)
        for i in m['members']:
            assign.append((int(i), lab, r))
        rank_of[lab] = r + 1
    return assign


def windowed_band_assignment(frozen_labels, model_seq, band_mask, warn=print):
    """Per-reference-band (band_idx, label, model_rank) map for the block-matched
    band MSE, with the within-block RANK counted only over the FITTING WINDOW and
    offset by the MODEL's own below-window state count.

    Why not just frozen_band_assignment: that ranks each irrep block from band 0
    using the REFERENCE (QE) labels. The weight-0 deep bands are then allowed to
    set the ranks of the bands we actually fit. When the reference's deep labels
    disagree with the model's -- e.g. a semicore parity a smooth local
    pseudopotential cannot reproduce -- each phantom deep member shifts every
    frontier band of that irrep by one rank, so block-matching pairs a frontier
    reference band with the WRONG (next-higher) model eigenvalue and the "small
    correction" explodes. Here the deep bands never enter the rank:

        rank(reference multiplet) = m0[label] + j

    where j is its 0-based position among the SAME-label multiplets INSIDE the
    window and m0[label] is the number of MODEL multiplets of that label BELOW the
    window floor (read off model_seq, the model's own energy-ordered blocks). The
    reference's deep-band labels drop out entirely; only the model's own
    below-window structure (self-consistent with model_ev) sets the offset.

    frozen_labels : ascending [{'members','label','dim'}, ...] (reference bands).
    model_seq     : ascending (energy, label, dim) of the MODEL's blocks at this k
                    (hot_path.irrep_sequence -- same block_eigvals as the hot path).
    band_mask     : bool array over reference bands, True where bandWeight > 0.
    Returns list of (ref_band_idx, label, model_rank), for in-window members only.
    Empty when nothing is weighted at this k.
    """
    mask = np.asarray(band_mask, bool)
    if not mask.any():
        return []                                    # no weighted bands here
    win = np.flatnonzero(mask)
    lo = int(win[0])                                 # window floor = first weighted band
    if warn and not np.array_equal(win, np.arange(win[0], win[-1] + 1)):
        warn(f"    [irrep freeze] non-contiguous band window {win.tolist()}; "
             f"rank offset uses the first weighted band ({lo}) as the floor.")
    # m0[label] = number of MODEL multiplets of this label strictly below band `lo`
    m0, nb = {}, 0
    for (_e, lab, dim) in model_seq:
        if nb >= lo:
            break
        if nb + dim > lo and warn:                   # floor splits a model multiplet
            warn(f"    [irrep freeze] window floor at band {lo} splits model "
                 f"multiplet '{lab}' (dim {dim} at band {nb}); counting it as below.")
        m0[lab] = m0.get(lab, 0) + 1
        nb += dim
    # reference in-window rank j per label, offset by the model's below-window count
    assign, jrank = [], {}
    for m in frozen_labels:
        lab = m['label']
        in_win = [i for i in m['members'] if mask[i]]
        if not in_win:
            continue                                 # multiplet entirely outside window
        j = jrank.get(lab, 0)
        rank = m0.get(lab, 0) + j
        for i in in_win:
            assign.append((int(i), lab, rank))
        jrank[lab] = j + 1
    return assign


def assert_sector_counts(model_ev, ref_sect, window=None, kname=""):
    """Raise if the number of model vs reference states per sector differ within
    the fitting window. This is the ghost-state detector (section 7)."""
    for lab, ref_e in ref_sect.items():
        n_ref = len(ref_e) if window is None else min(window, len(ref_e))
        n_mod = len(model_ev.get(lab, [])) if window is None else min(window, len(model_ev.get(lab, [])))
        if n_mod < n_ref:
            raise ValueError(
                f"[{kname}] sector '{lab}': model has {n_mod} states but reference "
                f"needs {n_ref}. A state is missing from this irrep block -- likely a "
                f"ghost or a wrong-basin reordering localized to sector '{lab}'.")


def sector_loss_at_k(model_ev, ref_sect, band_weight_by_label=None, dims=None,
                     energy_zero='none', vbm_ref=None, kname=""):
    """Sector-wise weighted MSE at one k-point (differentiable in model_ev).

    model_ev : {label: 1D tensor of model eigenvalues (ascending, collapsed)}.
    ref_sect : {label: 1D np array of reference energies (ascending)}.
    band_weight_by_label : optional {label: 1D tensor of per-state weights}.
    dims     : {label: d_lambda}; each sector weighted by d_lambda (prompt formula).
    energy_zero : 'none' | 'global' | 'vbm_R'.
    vbm_ref  : (E_model_vbm, E_ref_vbm) scalars, required if energy_zero=='vbm_R'.
    """
    # optional energy-zero handling
    dshift_model = 0.0
    dshift_ref = 0.0
    if energy_zero == 'vbm_R':
        if vbm_ref is None:
            raise ValueError("energy_zero='vbm_R' requires vbm_ref=(E_model_vbm,E_ref_vbm)")
        dshift_model, dshift_ref = vbm_ref

    total = model_ev[next(iter(model_ev))].new_zeros(())
    # 'global' additive constant = LS-optimal weighted mean residual (closed form)
    if energy_zero == 'global':
        num = model_ev[next(iter(model_ev))].new_zeros(())
        den = 0.0
        for lab, ref_e in ref_sect.items():
            n = min(len(ref_e), len(model_ev.get(lab, [])))
            if n == 0:
                continue
            d = dims[lab] if dims else 1
            w = (band_weight_by_label or {}).get(lab)
            w = w[:n] if w is not None else torch.ones(n, dtype=torch.float64)
            res = model_ev[lab][:n] - torch.as_tensor(ref_e[:n], dtype=torch.float64)
            num = num + d * (w * res).sum()
            den += d * float(w.sum())
        c = num / den if den > 0 else 0.0 * num
        dshift_model = c   # subtract from model

    for lab, ref_e in ref_sect.items():
        n = min(len(ref_e), len(model_ev.get(lab, [])))
        if n == 0:
            continue
        d = dims[lab] if dims else 1
        w = (band_weight_by_label or {}).get(lab)
        w = w[:n] if w is not None else torch.ones(n, dtype=torch.float64)
        e_mod = model_ev[lab][:n] - dshift_model
        e_ref = torch.as_tensor(ref_e[:n], dtype=torch.float64) - dshift_ref
        total = total + d * (w * (e_mod - e_ref) ** 2).sum()
    return total
