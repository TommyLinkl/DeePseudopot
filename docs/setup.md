# Environment and Input Bundle

## Runtime Environment
- **Python**: 3.9 or newer. Install dependencies with `pip install -r requirements.txt`.
- **CPU execution**: default; the driver pins to one thread (`torch.set_num_threads(1)`). Enable multiprocessing with `num_cores` in `NN_config.par`.
- **Optional tools**: `ffmpeg` for movie assembly, `mprof` when enabling memory profiling (`memory_flag = 1`).
- **Data prerequisites**: reference band structures, k-point paths, and initial Zunger parameters or q-space potentials for each system you plan to fit.

## Input Bundle Snapshot
Each run consumes a bundle of configuration and data files. At minimum provide:
- `NN_config.par` – global switches (architecture, optimizers, scheduler, Monte Carlo toggles).
- `system_X.par` and `input_X.par` – lattice geometry, atom list, band metadata, kinetic-energy cutoffs, plotting ranges.
- `kpoints_X.par`, `bandWeights_X.par`, `expBandStruct_X.par` – sampling path, loss weights, and reference band energies.
- `init_<atom>Params.par` or `init_qSpace_pot.par` – Zunger parameter sets or tabulated potentials for initialization.
- Optional `mcOpts*.par`, `<atom>ParamSteps.par`, `expCoupling_*.dat`, `expDefPot*.par` when running Monte Carlo or fitting auxiliary observables.

## Input Directory Layout
All files for a run reside in a single inputs directory. The table summarizes required and optional members (`X` indexes each system, starting from 0).

| File | Required | Contents / Role |
| --- | --- | --- |
| `NN_config.par` | yes | Global configuration (model shape, workflow mode, optimizer settings, diagnostics). |
| `system_X.par` | yes | Lattice scale, 3×3 primitive cell matrix, and fractional atomic coordinates. |
| `input_X.par` | yes | Metadata: `nBands`, `maxKE`, band indices (`idxVB`, `idxCB`, `idxGap`), plotting windows. |
| `kpoints_X.par` | yes | Fractional k-vectors and weights. Optional `kpoints_X_orderMatrix.par` fixes band ordering. |
| `bandWeights_X.par` | yes | Per-band weights applied in the loss (length must equal `nBands`). |
| `expBandStruct_X.par` | yes | Cumulative k-distance (column 0) plus target band energies. |
| `init_<atom>Params.par` | yes | Nine-parameter Zunger initialization vectors per unique atom label. |
| `init_qSpace_pot.par` | optional | Tabulated q-space potentials overriding analytic initialization. |
| `mcOpts*.par`, `<atom>ParamSteps.par` | optional | Monte Carlo iteration counts, perturbation magnitudes, or per-atom step sizes. |
| `expCoupling_*.dat`, `expDefPot*.par` | optional | Additional observables (optical couplings, defect level shifts) consumed when associated flags are set. |

Keep filenames consistent: atom ordering is inferred from `system_X.par` and reused to match `init_<atom>Params.par` and optional datasets. Once your bundle is organized, continue with the [Configuration Reference](configuration.md) to understand every switch in `NN_config.par`.
