"""
Training-loop hooks tying the symmetry module into NN_train.py. Everything here
is OPT-IN (gated by NNConfig['irrep_loss'] / ['irrep_diagnostic']); when off,
nothing in this file runs and the production path is untouched.

SETUP (once):   build_irrep_context() detects the high-symmetry k-points present
                in the fit, builds the cached SAPW basis at each, and FREEZES the
                reference irrep labels by energy-ordered matching to the model's
                block spectrum (see loss.freeze_reference_labels).

HOT PATH:       irrep_loss_term() -> differentiable CORRECTION that replaces the
                globally-sorted band MSE with a block-matched one at the high-sym
                k-points. It is NOT a separate objective: it inherits the system's
                bandWeights, kptWeights, and energy convention, so the total is just
                the band-structure MSE with irrep-aware matching at those k. The old
                irrep_lambda / irrep_energy_zero / per-sector d_lambda knobs are
                deprecated (ignored).

DIAGNOSTIC:     irrep_diagnostic_lines() -> cheap per-epoch irrep sequence at
                Gamma and R (and any other high-symmetry point/line requested).

SCOPE: the quantitative sector loss is wired for BOTH the high-symmetry POINTS
(little group O_h at Gamma/R, D_4h at X/M) and the high-symmetry LINES
(C_4v on T, C_3v on Lambda, C_2v on Sigma). Time-reversal co-representations are
merged (adapted_basis / hot_path): the only complex-conjugate spinor-irrep pair
here is C_3v's two 1-dim irreps on Lambda, which Kramers-pair into one 2-fold
physical sector so the model degeneracies match the (all-doublet) reference along
the lines. General k (little group order < 4) falls back to the plain loss.
"""
import numpy as np
import torch

from utils.constants import AUTOEV
from .group import build_Oh, little_group_indices
from .interface import build_sak_for_k
from .hot_path import block_eigvals, irrep_sequence
from .loss import (freeze_reference_labels, reference_sector_energies,
                   assert_sector_counts, windowed_band_assignment)
from .qe_labels import (parse_bands_post, _match_kfrac, align_reference_to_qe,
                        compare_labels)

# little-group orders that are high-symmetry POINTS with TRS-safe (>=2-dim) irreps
_POINT_LG_ORDERS = {16, 48}     # D_4h (X,M) and O_h (Gamma,R)


class IrrepContext:
    """Per-system cached symmetry context for the irrep loss + diagnostic."""

    def __init__(self, system, ham, NNConfig, cachedMats_info, oh=None):
        self.system = system
        self.ham = ham
        self.NNConfig = NNConfig
        self.cachedMats_info = cachedMats_info
        self.oh = oh if oh is not None else build_Oh()
        self.degen_tol = float(NNConfig.get('irrep_degen_tol', 1e-4))          # eV
        # DEPRECATED knobs (kept only so old configs still parse): the block-matched
        # band MSE has no separate scale or energy zero -- it inherits the system's.
        self.energy_zero = NNConfig.get('irrep_energy_zero', None)
        self.lam = float(NNConfig.get('irrep_lambda', 1.0))
        # Optional: anchor the reference irrep labels to a real cubic QE bands.x
        # run (lsym=.true.) instead of the model's energy-ordered labels. Path to
        # the bands.x output (e.g. .../cubic/bands_post.out). See qe_labels.py.
        # Optional band window (global 0-indexed band indices, inclusive) that
        # restricts the sector LOSS to a band range -- e.g. the gap region. Deep
        # valence / high conduction bands, where a smooth local pseudopotential
        # cannot reproduce QE's symmetry ordering, are given zero weight: they
        # stay in the spectrum for sorted-matching and count checks, but do not
        # contribute to the loss. Config 'irrep_band_window = lo hi'.
        self.band_window = None
        bw_win = NNConfig.get('irrep_band_window', None)
        if bw_win:
            parts = str(bw_win).split()
            if len(parts) == 2:
                self.band_window = (int(parts[0]), int(parts[1]))
        self.qe_labels_path = self._resolve_qe_path(
            NNConfig.get('irrep_reference_labels', None))
        self.qe_blocks = None
        if self.qe_labels_path:
            self.qe_blocks = parse_bands_post(self.qe_labels_path)
        self._detect_kpoints()
        self.sak = {}
        self.frozen = {}          # kidx -> {'ref_sect','dims','weights','name','kfrac'}
        self.R_kidx = None
        self._build_saks()

    def _resolve_qe_path(self, path):
        """Resolve the QE bands.x path: absolute as-is, else tried relative to the
        inputs folder (robust to the fit's working directory) then the cwd."""
        if not path:
            return None
        import os
        if os.path.isabs(path) and os.path.exists(path):
            return path
        inp = self.NNConfig.get('inputsFolder', '')
        for cand in (os.path.join(inp, path), path):
            if os.path.exists(cand):
                return cand
        print(f"[irrep] WARNING: irrep_reference_labels='{path}' not found "
              f"(inputsFolder='{inp}'); falling back to model-ordered labels.")
        return None

    # ---- which k-points carry a nontrivial little group (points AND lines)? -
    def _detect_kpoints(self):
        Grecip = self.system.getGVectors().detach().cpu().numpy()
        self._Binv_k = np.linalg.inv(Grecip)
        kpts_cart = self.system.kpts.detach().cpu().numpy()
        self.kfrac_all = kpts_cart @ self._Binv_k
        # Optional restriction of the sector LOSS to named high-symmetry points
        # (config 'irrep_sector_points', space-separated, e.g. "Gamma" or
        # "Gamma R M"). Default None -> loss on every detected high-sym k. Use
        # this when a boundary k's complete-star subspace does not capture the
        # fitted bands (small maxKE -> nonzero boundary leakage): restrict the
        # loss to the exact points (Gamma always, others once maxKE is large
        # enough). The basis is still BUILT at every detected point so the
        # per-epoch diagnostic can label all of them.
        raw = self.NNConfig.get('irrep_sector_points', None)
        self.allowed_names = set(str(raw).split()) if raw else None
        self.point_kidx = []      # O_h / D_4h high-symmetry POINTS
        self.line_kidx = []       # C_4v / C_3v / C_2v high-symmetry LINES
        self.all_sym_kidx = []    # every detected high-sym k (saks + diagnostic)
        self.sector_kidx = []     # subset that actually gets a sector loss
        for kidx, kf in enumerate(self.kfrac_all):
            order = len(little_group_indices(self.oh, kf))
            if order in _POINT_LG_ORDERS:
                self.point_kidx.append(kidx)
            elif order >= 4:                 # C_2v(4), C_3v(6), C_4v(8)
                self.line_kidx.append(kidx)
            else:
                continue
            self.all_sym_kidx.append(kidx)
            if self.allowed_names is None or self._kname(kf) in self.allowed_names:
                self.sector_kidx.append(kidx)

    def _kname(self, kf):
        """Standard Pm-3m label from the fractional coords (points AND lines),
        by folding each component to [0,1/2] and sorting: this is symmetry-robust
        and no longer snaps line points to the nearest corner."""
        red = np.array([min(abs(x % 1.0), 1.0 - abs(x % 1.0)) for x in kf])
        a, b, c = np.sort(red)               # a <= b <= c, each in [0, 0.5]
        H = 0.5
        t = 1e-3
        def eq(u, v):
            return abs(u - v) < t
        # points
        if eq(a, 0) and eq(b, 0) and eq(c, 0):    return 'Gamma'
        if eq(a, 0) and eq(b, 0) and eq(c, H):    return 'X'
        if eq(a, 0) and eq(b, H) and eq(c, H):    return 'M'
        if eq(a, H) and eq(b, H) and eq(c, H):    return 'R'
        # lines (0<u<1/2)
        if eq(a, 0) and eq(b, 0):                 return 'Delta'    # Gamma-X  (0,0,u)
        if eq(a, 0) and eq(b, c):                 return 'Sigma'    # Gamma-M  (0,u,u)
        if eq(a, b) and eq(c, H):                 return 'S'        # X-R      (u,u,1/2)
        if eq(a, 0) and eq(c, H):                 return 'Z'        # X-M      (0,u,1/2)
        if eq(a, b) and eq(b, c):                 return 'Lambda'   # Gamma-R  (u,u,u)
        if eq(b, H) and eq(c, H):                 return 'T'        # M-R      (u,1/2,1/2)
        return f"k={np.round(kf, 3)}"

    def _build_saks(self):
        for kidx in self.all_sym_kidx:
            kf = self.kfrac_all[kidx]
            name = self._kname(kf)
            self.sak[kidx] = build_sak_for_k(self.ham, list(kf), name=name, oh=self.oh)
            if name == 'R':
                self.R_kidx = kidx

    # ---- freeze reference irrep labels from the current model (once) --------
    def freeze_reference(self, cachedMats_info=None, verbosity=1):
        cmi = cachedMats_info if cachedMats_info is not None else self.cachedMats_info
        for kidx in self.sector_kidx:
            sak = self.sak[kidx]
            name = sak.name
            with torch.no_grad():
                H = self.ham.buildHtot_cached(kidx, cmi, requires_grad=False)
            model_seq = irrep_sequence(H, sak, unit=AUTOEV)   # (E_eV, label, dim) ascending
            E_ref = self.system.expBandStruct[kidx].detach().cpu().numpy()

            qe_blk = self._qe_block_for(kidx)
            source = 'model-energy-order'
            if qe_blk is not None:
                # anchor labels to the first-principles QE symmetry assignment
                frozen_labels, diag = align_reference_to_qe(
                    E_ref, qe_blk['multiplets'], degen_tol=self.degen_tol)
                source = (f"QE {qe_blk['point_group']} "
                          f"(dropped {diag['n_qe_dropped']} deep multiplets, "
                          f"shift {diag['shift']:.3f} eV)")
                if not diag['ok'] and verbosity:
                    print(f"    [irrep freeze] {name}: QE/reference energy match is "
                          f"loose (spread {diag['spread']:.3g} eV) -- verify alignment.")
            else:
                frozen_labels = freeze_reference_labels(E_ref, model_seq,
                                                        degen_tol=self.degen_tol)

            # always report QE-vs-model agreement so a wrong initial basin / a
            # swapped Gamma6-Gamma7 shows up at freeze time, not as a silent fit.
            if verbosity:
                self._report_labels(name, kidx, frozen_labels, model_seq, source)

            ref_sect = reference_sector_energies(frozen_labels)
            dims = {m['label']: m['dim'] for m in frozen_labels}
            weights = self._sector_weights(frozen_labels)
            # Per-band (ref_idx, label, block_rank) map that REORDERS the model
            # spectrum into block-matched order for the ordinary band MSE. Ranks
            # are counted only over the fitting window (bandWeights > 0) and
            # offset by the MODEL's own below-window state count, so the weight-0
            # deep bands -- whose QE labels the smooth pseudopotential may not
            # reproduce -- cannot shift the frontier match (see loss.py).
            band_mask = self.system.bandWeights.detach().cpu().numpy() > 0
            band_assign = windowed_band_assignment(frozen_labels, model_seq, band_mask)
            # The reorder places block values onto expBandStruct band indices, so it
            # is only valid when calcEigValsAtK returns bands in that same index
            # order (bandOrderMatrix == identity at this k). If a manual band order
            # is set here, keep the sector for the diagnostic/count-check but do NOT
            # block-match (fall back to the plain ordered MSE at this k).
            if not self._is_identity_bandorder(kidx):
                if verbosity:
                    print(f"    [irrep freeze] {name}: non-identity bandOrderMatrix "
                          f"at this k -- block-matching disabled here (plain ordered "
                          f"MSE retained); diagnostic/count-check still active.")
                band_assign = None
            self.frozen[kidx] = {'ref_sect': ref_sect, 'dims': dims,
                                 'weights': weights, 'name': name,
                                 'kfrac': self.kfrac_all[kidx], 'band_assign': band_assign}

    def _is_identity_bandorder(self, kidx):
        """True if this k's band ordering is the plain ascending arange, so a
        model eigenvalue at ascending position i lands on reference band i.
        Robust to bandOrderMatrix being a numpy array or a torch tensor."""
        bom = getattr(self.system, 'bandOrderMatrix', None)
        if bom is None:
            return True
        row = bom[kidx]
        arr = row.detach().cpu().numpy() if torch.is_tensor(row) else np.asarray(row)
        return bool(np.array_equal(arr, np.arange(arr.shape[0])))

    def _qe_block_for(self, kidx):
        """The parsed QE band-symmetry block matching this k-point, or None (no
        QE file, or this k is a high-sym LINE -- QE labels are for POINTS only)."""
        if self.qe_blocks is None or kidx not in self.point_kidx:
            return None
        return _match_kfrac(self.kfrac_all[kidx], self.qe_blocks)

    def _report_labels(self, name, kidx, frozen_labels, model_seq, source):
        seq = " ".join(m['label'] for m in frozen_labels[:12])
        print(f"    [irrep freeze] {name} (kidx {kidx}): {len(frozen_labels)} "
              f"multiplets from {source}:")
        print(f"        ref : {seq} ...")
        n_ok, n_tot, bad = compare_labels(frozen_labels, model_seq)
        mseq = " ".join(lab for (_, lab, _d) in model_seq[:12])
        print(f"        model: {mseq} ...")
        if bad:
            detail = ", ".join(f"#{i}: ref {q} vs model {m}" for (i, q, m) in bad[:6])
            print(f"        [!] {n_tot - n_ok}/{n_tot} multiplets DISAGREE "
                  f"(initial model basin): {detail}"
                  + (" ..." if len(bad) > 6 else ""))
        else:
            print(f"        [ok] all {n_tot} multiplets agree with the initial model.")

    def _sector_weights(self, frozen_labels):
        """Per-state weights grouped by sector, in energy order, from bulkSystem
        bandWeights (mean over each multiplet's member bands). If a band window
        is set, multiplets with no member band inside [lo, hi] get weight 0 -- so
        the sector loss ignores them while they still count for sorted-matching
        and the ghost-state check (assert_sector_counts)."""
        bw = self.system.bandWeights.detach().cpu().numpy()
        lo, hi = self.band_window if self.band_window is not None else (None, None)
        out = {}
        for m in frozen_labels:
            w = float(np.mean([bw[i] for i in m['members']]))
            if self.band_window is not None and not any(lo <= i <= hi for i in m['members']):
                w = 0.0
            out.setdefault(m['label'], []).append(w)
        return {lab: torch.tensor(v, dtype=torch.float64) for lab, v in out.items()}

    # ---- hot path: block-matching CORRECTION to the band-structure MSE ------
    def irrep_loss_term(self, cachedMats_info=None, requires_grad=True):
        """Correction that turns the ordinary (globally-sorted) band MSE into a
        block-matched one at the high-symmetry k-points. There is NO separate irrep
        objective: this returns, summed over those k,

            kptWeight_k * relu( bandMSE(block_matched_k) - bandMSE(ordered_k) ),

        where both MSEs use the SYSTEM's own band loss (bandWeights + energy
        convention). The main loss already adds bandMSE(ordered_k) at every k with
        weight kptWeight_k, so the total contribution at these k becomes
        max(bandMSE(block_matched_k), bandMSE(ordered_k)) -- the ordered term cancels
        bit-for-bit while the correction is positive (the `ordered` spectrum here is
        the SAME calcEigValsAtK the main loop uses), and the k falls back to the plain
        ordered MSE when the correction would go negative. The clamp is NOT cosmetic:
        the block eigenvalues are Rayleigh-Ritz upper bounds on an incomplete
        symmetry-adapted subspace, and an un-clamped negative correction lets the fit
        decouple them from the true spectrum (see the clamp comment below). This term
        is therefore strictly >= 0. Every other k is untouched, and when nothing is
        frozen this is a zero tensor."""
        cmi = cachedMats_info if cachedMats_info is not None else self.cachedMats_info
        device = self.system.kpts.device
        total = torch.zeros((), dtype=torch.float64, device=device)
        if not self.frozen:
            return total
        nBands = int(self.system.nBands)
        for kidx, fr in self.frozen.items():
            sak = self.sak[kidx]
            # ordinary spectrum -- identical to the main band-structure term, so its
            # contribution cancels exactly in the correction below.
            E_ord = self.ham.calcEigValsAtK(kidx, cmi, requires_grad=requires_grad)
            # block-diagonalized model spectrum, one ascending list per irrep.
            H = self.ham.buildHtot_cached(kidx, cmi, requires_grad=requires_grad)
            model_ev = block_eigvals(H, sak, collapse=True)          # Hartree
            model_ev = {lab: v * AUTOEV for lab, v in model_ev.items()}  # -> eV
            # ghost / wrong-basin guard: per-irrep state counts must match reference.
            assert_sector_counts(model_ev, fr['ref_sect'], window=None,
                                 kname=fr['name'])
            if fr['band_assign'] is None:            # ordering guard: no reorder here
                continue
            # gather block-matched model energies onto their reference band indices
            idxs, vals = [], []
            for (i, lab, rank) in fr['band_assign']:
                if i >= nBands:
                    continue
                col = model_ev.get(lab)
                if col is None or rank >= len(col):  # guarded above; skip defensively
                    continue
                idxs.append(i)
                vals.append(col[rank])
            if not idxs:
                continue
            idx_t = torch.as_tensor(idxs, dtype=torch.long, device=E_ord.device)
            matched = E_ord.index_copy(0, idx_t, torch.stack(vals).to(E_ord.dtype))
            kw = float(self.system.kptWeights[kidx])
            # Block-matched vs ordered MSE at this k. The block eigenvalues feeding
            # `matched` are Rayleigh-Ritz values on the symmetry-adapted subspace
            # (B^dag H B) -- variational UPPER bounds on the true spectrum. When that
            # subspace is complete they are a REORDERING of the true eigenvalues, so
            # block-matched >= ordered (rearrangement inequality) and this correction
            # is >= 0. When the subspace LEAKS (finite maxKE at a boundary k such as
            # R), the projected eigenvalues decouple from the true spectrum and the
            # raw correction can go NEGATIVE: the optimizer then "improves" the
            # projected eigenvalues while the true bands (the reported bandStruct)
            # drift away -- the divergence seen in run_cubic_irrep_noLR_2.dat, where
            # irrep goes negative in lockstep with bandStruct rising.
            #
            # Clamp the per-k correction at 0 so this k falls back to the plain
            # (true-spectrum) ordered band MSE in that regime: the term is strictly
            # >= 0 and the fit is never rewarded for the projection artifact. The
            # effective per-k objective becomes max(blockMSE, orderedMSE) -- identical
            # to the intended block-matched MSE wherever the subspace is healthy, and
            # a safe fallback where it leaks. Per-k (not on the summed total) so a
            # leaking boundary point is disabled without also killing a healthy Gamma
            # correction. relu, NOT abs: abs would flip the gradient sign and actively
            # push the block eigenvalues away from the true spectrum.
            correction = (self._kpt_band_mse(matched, kidx)
                          - self._kpt_band_mse(E_ord, kidx))
            total = total + kw * torch.clamp(correction, min=0.0)
        return total

    def _kpt_band_mse(self, E_pred, kidx):
        """The system's own single-k band-structure MSE (bandWeights + optional
        relative-to-band energy zero), mirroring NN_train.bandStruct_kpt_loss so the
        block-matched term is measured in exactly the fit's units."""
        sys = self.system
        bw = sys.bandWeights
        relE = int(getattr(sys, 'relE_bIdx', -1))
        if relE != -1:
            ref = sys.expBandStruct[kidx] - sys.expBandStruct[kidx, relE]
            pred = E_pred - E_pred[relE]
            return torch.sum((ref - pred) ** 2 * bw)
        return torch.sum((E_pred - sys.expBandStruct[kidx]) ** 2 * bw)

    # ---- one-time QE-vs-model label report (all high-sym POINTS) -----------
    def qe_label_report(self, cachedMats_info=None):
        """Print, at every high-symmetry POINT, the QE-anchored reference irrep
        labels beside the model's energy-ordered labels -- independent of which
        points carry the sector loss. Requires irrep_reference_labels to be set.
        Returns a dict {name: {'agree':int,'total':int,'disagreements':[...]}}."""
        cmi = cachedMats_info if cachedMats_info is not None else self.cachedMats_info
        if self.qe_blocks is None:
            print("[qe-label] no irrep_reference_labels set -- nothing to report.")
            return {}
        out = {}
        for kidx in self.point_kidx:
            sak = self.sak[kidx]
            qe_blk = self._qe_block_for(kidx)
            if qe_blk is None:
                print(f"    [qe-label] {sak.name}: no matching QE block in the file.")
                continue
            with torch.no_grad():
                H = self.ham.buildHtot_cached(kidx, cmi, requires_grad=False)
            model_seq = irrep_sequence(H, sak, unit=AUTOEV)
            E_ref = self.system.expBandStruct[kidx].detach().cpu().numpy()
            frozen_labels, diag = align_reference_to_qe(
                E_ref, qe_blk['multiplets'], degen_tol=self.degen_tol)
            source = (f"QE {qe_blk['point_group']} "
                      f"(dropped {diag['n_qe_dropped']}, shift {diag['shift']:.3f} eV, "
                      f"spread {diag['spread']:.2g} eV)")
            self._report_labels(sak.name, kidx, frozen_labels, model_seq, source)
            n_ok, n_tot, bad = compare_labels(frozen_labels, model_seq)
            out[sak.name] = {'agree': n_ok, 'total': n_tot, 'disagreements': bad}
        return out

    # ---- diagnostic --------------------------------------------------------
    def diagnostic_lines(self, cachedMats_info=None, n_show=14, which=('Gamma', 'R')):
        cmi = cachedMats_info if cachedMats_info is not None else self.cachedMats_info
        lines = []
        for kidx in self.point_kidx + self.line_kidx:
            sak = self.sak[kidx]
            if which and sak.name not in which:
                continue
            with torch.no_grad():
                H = self.ham.buildHtot_cached(kidx, cmi, requires_grad=False)
            seq = irrep_sequence(H, sak, n_show=n_show, unit=AUTOEV)
            labels = " ".join(f"{lab}" for (_, lab, _d) in seq)
            lines.append(f"  irrep@{sak.name}: {labels}")
        return "\n".join(lines)


def build_irrep_context(system, ham, NNConfig, cachedMats_info, freeze=True, verbosity=1):
    """Convenience: build the context and (optionally) freeze reference labels."""
    ctx = IrrepContext(system, ham, NNConfig, cachedMats_info)
    if verbosity:
        detected = [ctx.sak[k].name for k in ctx.all_sym_kidx]
        loss_on = [ctx.sak[k].name for k in ctx.sector_kidx]
        print(f"[irrep] high-sym k detected: {detected}; sector loss on {loss_on}"
              + (f" (restricted by irrep_sector_points={sorted(ctx.allowed_names)})"
                 if ctx.allowed_names is not None else "")
              + f"; other detected points get the diagnostic only")
        if ctx.band_window is not None:
            print(f"[irrep] sector loss WINDOWED to bands {ctx.band_window[0]}-"
                  f"{ctx.band_window[1]} (0-indexed, inclusive); bands outside get "
                  f"zero weight (kept only for sorted-matching / count checks)")
    if freeze:
        ctx.freeze_reference(cachedMats_info, verbosity=verbosity)
    return ctx
