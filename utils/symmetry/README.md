# Symmetry-adapted (double-group) basis for the spinor pseudopotential Hamiltonian

Block-diagonalizes the SOC Hamiltonian by irrep at high-symmetry k so eigenvalues
come out **pre-labeled by irrep**, without ever computing eigenvectors in the
training hot path. For cubic Pm-3m CsPbX3 (Pb at origin), spinor states, double
group throughout.

## Modules (setup vs hot path are separate)

| module | when | what |
|---|---|---|
| `group.py` | setup | O_h (48 ops, integer crystal = Cartesian), SU(2) `D^{1/2}` + Wigner `D^{3/2}`, 96-element double group, little groups |
| `irreps.py` | setup | double-group character tables **generated** by the Burnside class-algebra method, orthogonality-validated, labeled Γ6/Γ7/Γ8± by structure (dim, parity, C4 character) not by strings |
| `adapted_basis.py` | setup | per-**star** sparse symmetry-adapted plane-wave-spinor basis `B_λ` (each column ≤96 nonzeros, `requires_grad=False`); complete/incomplete star detection |
| `interface.py` | setup | `build_sak_for_k(ham, kfrac)` — bridge to the DeePseudopot `Hamiltonian` |
| `characters.py` | setup/validate | group action `U(R)` + character-of-degenerate-subspace (labels QE/model eigenvectors) |
| `hot_path.py` | **hot** | `block_eigvals(H, sak) → {irrep: eigenvalues}` (sparse `B†HB`, eigvalsh per block, no eigenvectors, differentiable); `irrep_sequence` diagnostic |
| `loss.py` | setup+hot | `freeze_reference_labels` (energy-ordered, robust to accidental degeneracies) + differentiable `sector_loss_at_k` + ghost-detection asserts |
| `train_hooks.py` | both | `IrrepContext`: detect high-sym k, cache SAPW basis, freeze reference labels, `irrep_loss_term`, `diagnostic_lines` |
| `validate.py` | test | the validation suite, run against the **real** `buildHtot` pipeline |

## Enable in a fit (all opt-in; default OFF, zero overhead when off)

In `NN_config.par`:

```
irrep_loss = 1            # add the symmetry sector-wise loss at high-sym k
irrep_diagnostic = 1      # log the irrep sequence at Gamma/R every epoch
irrep_lambda = 1.0        # weight of the sector-loss term
irrep_degen_tol = 1e-4    # multiplet grouping tolerance (eV)
irrep_energy_zero = vbm_R # 'vbm_R' | 'global' | 'none'
```

The reference irrep labels are FROZEN once from the initial model (matching each
reference band to the model's per-band irrep in energy order) — so a later drift
into a wrong basin shows up as a per-sector state-count mismatch (`assert_sector_counts`)
or in the per-epoch diagnostic. **Cache SO/NL (`cacheSO`) for the extra
high-symmetry `buildHtot` calls to be cheap.**

## Scope & the one physical caveat

- Sector **loss** is wired for the high-symmetry POINTS Γ, R (O_h) and X, M (D_4h),
  whose double-group spinor irreps are all ≥2-dim and time-reversal-safe.
- LINES (Λ/Σ/T) are block-diagonalized and **labeled** for the diagnostic; Λ's
  (C_3v) two 1-dim spinor irreps are time-reversal-paired co-representations, so
  their sector loss needs co-rep merging — deferred.
- The model's basis is a **k-independent |G|-sphere**. At Γ every umklapp G0=0 so
  block-diagonalization is machine-exact and completeness is `2·nbv`. At boundary
  k a thin shell of stars maps outside the sphere and is excluded; block structure
  stays machine-exact on the complete-star subspace. At the production cutoff
  (maxKE=14) the lowest fitted bands live entirely in complete stars, so the block
  spectrum equals the full-H spectrum to machine precision (measured: 0 eV leakage
  at Γ/R/M).

## Run the tests

```
PYTHONPATH=<DeePseudopot> python utils/symmetry/tests/test_group.py
PYTHONPATH=<DeePseudopot> python utils/symmetry/tests/test_irreps.py
PYTHONPATH=<DeePseudopot> python -m utils.symmetry.validate <inputs_folder>   # exercises real buildHtot
```
