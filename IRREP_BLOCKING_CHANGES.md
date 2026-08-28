# Irrep blocking for DeePseudopot — change summary (`irrep-block` branch)

Adds a **spinor-aware double-group symmetry-adapted basis** to the NN pseudopotential
fit. At the high-symmetry k-points the SOC Hamiltonian is block-diagonalized by irrep,
so model eigenvalues come out **pre-labeled by irrep** and can be matched sector-by-sector
to the reference bands — **without ever computing eigenvectors in the training hot path**.

Everything is **opt-in** and **default-OFF**: with the new config keys absent, not one line
of this code runs and the production eigenvalue path is byte-for-byte unchanged.

Target system: cubic Pm-3m CsPbX3 (Pb at origin), spinor states, double group throughout.

---

## 1. New package: `utils/symmetry/`

Setup (run once, cached) and the training hot path live in separate modules by design.

| module | when | what it does |
|---|---|---|
| `group.py` | setup | O_h (48 ops; on integer-crystal G = Cartesian G they are signed permutations), SU(2) `D^{1/2}` + Wigner `D^{3/2}`, the 96-element double group, and little-group extraction |
| `irreps.py` | setup | double-group character tables **generated** by the Burnside class-algebra method, orthogonality-validated; irreps labeled Γ6/Γ7/Γ8± by structure (dim, parity, C4 character), not by hard-coded strings |
| `adapted_basis.py` | setup | per-**star** sparse symmetry-adapted plane-wave-spinor basis `B_λ` (each column ≤ 96 nonzeros, `requires_grad=False`); complete/incomplete-star detection; time-reversal co-rep sectors (`eff_sectors`/`eff_dim`) |
| `interface.py` | setup | `build_sak_for_k(ham, kfrac)` — bridges the DeePseudopot `Hamiltonian`/`BulkSystem` to the symmetry module (Miller indices from the Cartesian G-list); `HIGH_SYM_POINTS` |
| `characters.py` | setup/validate | group action `U(R)` and character-of-a-degenerate-subspace (used to label QE/model eigenvectors during validation) |
| `hot_path.py` | **hot** | `block_eigvals(H, sak) -> {irrep: eigenvalues}` via sparse `B†HB`, one `eigvalsh` per block, **no eigenvectors, fully differentiable in H**; co-rep merging; `irrep_sequence` diagnostic |
| `loss.py` | setup + hot | `freeze_reference_labels` (energy-ordered, robust to accidental degeneracies), differentiable `sector_loss_at_k`, and `assert_sector_counts` ghost-state detection |
| `qe_labels.py` | setup | parse a cubic QE `bands.x` output (`lsym=.true.`), map its Koster labels to the model convention (verified against QE's printed character tables), and align them to `expBandStruct` by degeneracy-pattern fingerprint (VBM-shift/scissor/deep-band-drop robust) → **QE-anchored** reference labels; `compare_labels` model-vs-QE diff |
| `report_qe_labels.py` | tool | one-time report: QE-anchored labels vs the initial model at every high-sym point, no training run needed |
| `train_hooks.py` | both | `IrrepContext`: detects the high-sym k in the fit, caches the SAPW basis, freezes reference labels, exposes `irrep_loss_term()` and `diagnostic_lines()` |
| `validate.py` | test | validation suite run against the **real** `buildHtot` pipeline (6 tests, see §4) |
| `tests/` | test | `test_group.py`, `test_irreps.py` — pure unit tests (no ham needed) |
| `README.md` | — | reference doc for the package |

**Why it's cheap in the hot path:** `B_λ` is a constant sparse matrix (no grad). Each step
does two sparse mm's (`B†H`, then `·B`) and an `eigvalsh` per block. Gradients flow to the
potential parameters only through `H`; eigenvalue autograd has no `1/(λ_i−λ_j)` terms, so it is
degeneracy-safe. No eigenvectors are ever formed.

---

## 2. Hot-path integration (edits to existing files)

### `utils/ham.py` — `Hamiltonian.buildHtot_cached(kidx, cachedMats_info, requires_grad, precomp_Vloc)` (new)
Returns the **dense** `Htot` at k-index `kidx`, loading SO/NL matrices from the shared-memory /
disk cache exactly as `calcEigValsAtK` does. The symmetry path needs the dense `H` (to form
`B†HB`), whereas `calcEigValsAtK` only returns eigenvalues — so this is factored out as a
sibling loader. **The production eigenvalue path is untouched**; the shm/disk load block here
mirrors `calcEigValsAtK`'s loader and must be kept in sync with it.

### `utils/read.py` — config keys (parser only)
Registered the new `NN_config.par` keys so they parse to the right type:
- **bool:** `irrep_loss`, `irrep_diagnostic`
- **float:** `irrep_lambda`, `irrep_degen_tol`
- string (fallback, no parser change needed): `irrep_energy_zero` (`vbm_R` | `global` | `none`)
  and `irrep_sector_points` (space-separated point names, or absent = all detected)

### `utils/NN_train.py` — loss term + diagnostic
- Added `"irrep"` to `LOSS_TERM_NAMES` (so it's tracked, printed, and plotted alongside the others).
- `_irrep_term(model, ham, bulkSystem, cachedMats_info, requires_grad, device)`: returns a
  `0.0` tensor when `irrep_loss` is off; otherwise lazily builds and caches an `IrrepContext`
  on the ham (`ham._irrep_ctx`, frozen from the initial potential) and returns
  `ctx.irrep_loss_term(...)`. Wired into `compute_global_system_losses` as the `"irrep"` entry.
- `_log_irrep_diagnostic(systems, hams, NNConfig, cachedMats_info)`: prints the ascending irrep
  sequence at Γ and R for each system. Called once per epoch from `bandStruct_train_GPU` when
  `irrep_diagnostic` is on — so a drift into a **wrong symmetry basin** shows up immediately,
  not just as a loss plateau. Reuses the cached SAPW basis (cheap).

**Reference labeling.** Two modes, both frozen once at setup:

- **QE-anchored (preferred)** — set `irrep_reference_labels = <path to a cubic QE bands.x
  output>` (run with `lsym=.true.` on the 5-atom Pm-3m SOC cell — exactly the fit cell). The
  "Band symmetry" section then carries the *true* double-group labels. `utils/symmetry/qe_labels.py`
  parses them, maps QE's Koster labels to the model's convention (`G_6/7/8± → Gamma6/7/8±` for
  O_h, `→ E1/2/E3/2±` for D_4h — a mapping **verified** against QE's own printed character
  tables: in both groups QE's `G_6` is the χ(C4)=+√2 branch, which is exactly how
  `irreps._assign_labels` defines `Gamma6`/`E1/2`), and aligns them to `expBandStruct` by
  degeneracy-pattern fingerprint (robust to the VBM shift, the CB scissor, and the dropped deep
  bands). The reference is then labeled by **first-principles symmetry**, independent of the
  model. (The older orthorhombic-supercell reference had `nosym/noinv`, so *its* `.rap` was
  unusable — see `qe-reference-provenance`; this is a separate, purpose-run cubic bands.x.)

- **Model-ordered (fallback)** — with no QE file set, labels are frozen from the initial model by
  matching each reference band to the model's per-band irrep in energy order. This inherits the
  initial potential's Gamma6-vs-Gamma7 ordering, so it is only as trustworthy as that basin.

Either way, freeze prints the reference labels beside the model's own (`compare_labels`), so a
swapped Gamma6/Gamma7 or wrong parity — a wrong initial basin — is visible immediately, not just
as a later loss plateau. A drift during training still surfaces as a per-sector state-count
mismatch (`assert_sector_counts`) or in the per-epoch diagnostic. Energy zero is a config choice:
`vbm_R` (VBM at R, model & reference), `global`, or `none`.

To see the labeling without launching a fit:
`python -m utils.symmetry.report_qe_labels <inputs> <bands_post.out>` prints the QE-anchored
labels vs the initial model at every high-symmetry point.

---

## 3. How to enable it in a fit

In `NN_config.par` (all default OFF; requires SO/NL caching so the extra high-sym `buildHtot`
calls are cheap):

```
cacheSO = 1
cacheNL = 1
irrep_loss = 1            # add the symmetry sector-wise loss at high-sym k
irrep_diagnostic = 1      # log the irrep sequence at Gamma/R every epoch
irrep_lambda = 1.0        # weight of the sector-loss term
irrep_degen_tol = 1e-4    # multiplet grouping tolerance (eV)
irrep_energy_zero = global      # 'vbm_R' | 'global' | 'none'
irrep_sector_points = Gamma     # optional: restrict the LOSS to named points
```

The high-symmetry k-points are auto-detected from `kpoints_0.par` by little-group order, so no
extra k-point bookkeeping is needed — the fit path just has to actually sample Γ/R/M (and the
lines) for the loss to have anything to act on.

**`irrep_sector_points`** (new) restricts the sector *loss* to the named points; the basis is
still built at every detected point so the per-epoch *diagnostic* still labels all of them.
Absent = loss on every detected point. Use it when a boundary point's complete-star subspace
does not capture the fitted bands at the chosen cutoff (see §5). `irrep_energy_zero = global`
fits one additive constant from each loss-point's own sectors and never reads the R block — so a
Γ-only run stays independent of R's (possibly truncated) spectrum.

---

## 4. Validation status

Suite (`python -m utils.symmetry.validate <inputs>`) runs against the real kinetic + local-NN +
KB-nonlocal + SOC Hamiltonian. At the **production cutoff maxKE=14** all six tests pass:

1. **completeness** — Σ_λ n_cols == 2·N_complete (== 2·nbv at Γ)
2. **B orthonormality** — `B_l† B_l = I`, `B_l† B_m = 0` (l≠m)
3. **block-diagonality** — `||B_l† H B_m|| ~ 0` (l≠m) to machine precision — the sharp test
4. **spectrum equivalence** — block spectrum == `eigvalsh(H)` (~1e-14 Ha)
5. **free-electron (V=0)** — empty-lattice degeneracies + integer irrep multiplicities
6. **boundary leakage** — 0 eV on the lowest 42 (fitted) bands at Γ/R/M

Unit tests: character-table orthogonality and free-electron irreps agree with an independent
`characters.py` computation.

---

## 5. Scope and the one physical caveat

- **Points** Γ, R (O_h) and X, M (D_4h): full quantitative **sector loss**. All their
  double-group spinor irreps are ≥2-dim and time-reversal-safe.
- **Lines** Λ (Γ–R, C_3v), Σ (Γ–M, C_2v), T (M–R, C_4v): block-diagonalized, **labeled**, and
  now given a sector loss with **time-reversal co-representations merged** — Λ's two 1-dim
  spinor irreps are Kramers partners that merge into one 2-fold physical sector, so the model
  degeneracies match the (all-doublet) reference along the line. General k (little-group order
  < 4) falls back to the plain band-structure loss.
- **The k-independent |G|-sphere caveat — and the measured cutoff dependence.** The model's
  basis is a fixed |G|-sphere and the cutoff is on |G|, not |k+G|. A little-group operation maps
  k+G → R(k+G) (same length), so within a star **|k+G| is fixed but |G| varies** with direction.
  At Γ (k=0) |G|=|k+G|, so a star has a single |G| and is never cut → block-diagonalization is
  machine-exact and completeness is exactly 2·nbv, at **any** cutoff. At a **boundary** k (large
  |k|) a star spans a range of |G| of width ~2|k|, so stars near the surface are sliced; those
  incomplete stars are excluded (complete-star restriction). The excluded shell has ~fixed
  thickness set by |k|, so as maxKE grows it becomes a smaller fraction of the sphere **and moves
  to higher energy, away from the fitted bands** → the effect on the fitted spectrum ("boundary
  leakage") vanishes. Measured on the lowest 42 bands (CsPbBr₃, `inputs_cubic_irrep`):

  | k | leakage @ maxKE=8 | @ maxKE=10 | @ maxKE=14 |
  |---|---|---|---|
  | Γ | 0 eV | 0 eV | 0 eV |
  | R | 0.49 eV (28% of basis excluded) | 0.33 eV (25%) | 0 eV |
  | M | 0.31 eV (20% excluded) | — | 0 eV |

  So the sector loss at R/M is only trustworthy once the cutoff is large enough that its leakage
  → 0 (validated at **maxKE=14**); Γ is exact at any cutoff. **Verify the boundary-leakage test
  (#6) at your cutoff before enabling the loss at R/M**, or use `irrep_sector_points` to restrict
  the loss to the points that pass.

---

## 6. Files touched

```
NEW   utils/symmetry/{__init__,group,irreps,adapted_basis,interface,characters,
                      hot_path,loss,train_hooks,validate,qe_labels,report_qe_labels}.py
NEW   utils/symmetry/{README.md, tests/test_group.py, tests/test_irreps.py,
                      tests/test_qe_labels.py}
EDIT  utils/symmetry/train_hooks.py  (+ QE-anchored freeze via irrep_reference_labels,
                                      + qe_label_report, model-vs-QE compare at freeze)
EDIT  utils/ham.py       (+ buildHtot_cached)
EDIT  utils/read.py      (+ irrep_* config keys)
EDIT  utils/NN_train.py  (+ irrep loss term, per-epoch diagnostic, LOSS_TERM_NAMES)
```

The `irrep_reference_labels` key parses as a plain string (fallback path in `read.py`, like
`irrep_energy_zero`), so no parser change was needed.

---

## 7. First submitted run (`bromide/inputs_cubic_irrep`, maxKE=8)

A deliberately cheap first fit to exercise the whole pipeline. Because maxKE=8 leaks ~0.3–0.5 eV
at R/M (table in §5), the **sector loss is restricted to Γ** (exact at any cutoff); R is still
labeled every epoch by the diagnostic. Built by copying the validated `inputs_cubic_all` set and
setting:

```
maxKE               = 8.0        # input_0.par
cacheSO = 1 / cacheNL = 1        # required so the extra Gamma buildHtot is cheap
max_num_epochs      = 1000
irrep_loss = 1 / irrep_diagnostic = 1 / irrep_lambda = 1.0
irrep_energy_zero   = global     # avoids the (truncated) R block
irrep_sector_points = Gamma      # loss on Gamma only
```

Submit: `sbatch bromide/submit_irrep.sh` (1 CPU node, 8 h) →
`python main.py inputs_cubic_irrep/ results_cubic_irrep/ > run_cubic_irrep.dat`.

**Next step for the full loss:** raise maxKE until the boundary-leakage test (#6) hits ~0 at
R/M (validated at 14), then drop `irrep_sector_points` (or set it to `Gamma R M …`) to turn on
the boundary sectors.
