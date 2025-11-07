# Environment and Input Bundle

## Runtime Environment
| Key | Type | Applies to | Required? | Notes |
| --- | --- | --- | --- | --- |
| Python | version | All users | Yes | Use Python 3.9+ and install dependencies via `pip install -r requirements.txt`. |
| `num_cores` | int | `NN_config.par` | Optional | Default CPU execution uses one thread; set `num_cores > 0` to enable multiprocessing. |
| `memory_flag` | int (0/1) | `NN_config.par` | Optional | When set to 1, the driver prints `mprof` instructions for memory profiling. Requires `mprof`. |
| `runtime_flag` | int (0/1) | `NN_config.par` | Optional | Enables additional runtime logging for diagnostics. |
| External tools | binaries | Optional workflows | Optional | Install `ffmpeg` for movie assembly; add `mprof` only if `memory_flag = 1`. |
| Data prerequisites | files | Each system | Yes | Provide reference band structures, k-point paths, and either Zunger parameters or tabulated q-space potentials per species. |

## Input Bundle Snapshot
| Key | Type | Applies to | Required? | Notes |
| --- | --- | --- | --- | --- |
| `NN_config.par` | file | Whole run | Yes | Holds global switches (architecture, optimizers, scheduler, Monte Carlo toggles, diagnostics). |
| `system_X.par` / `input_X.par` | files | Each system `X` | Yes | Capture lattice geometry, atom list, band metadata, kinetic-energy cutoffs, and plotting ranges. |
| `kpoints_X.par`, `bandWeights_X.par`, `expBandStruct_X.par` | files | Each system `X` | Yes | Provide sampling paths, loss weights, and reference band energies. |
| `init_<atom>Params.par` / `init_qSpace_pot.par` | files | Initialization | At least one path required | Supply Zunger parameters or tabulated q-space potentials per species. |
| `mcOpts*.par`, `<atom>ParamSteps.par`, `expCoupling_*.dat`, `expDefPot*.par` | files | Specialized workflows | Optional | Required only for Monte Carlo tuning or when fitting couplings/defect observables. |

## Input Directory Layout
All files for a run reside in a single inputs directory. The table summarizes required and optional members (`X` indexes each system, starting from 0).

| File | Type | Applies to | Required? | Notes |
| --- | --- | --- | --- | --- |
| `NN_config.par` | file | Whole run | Yes | Contains model shape, workflow mode, optimizer settings, and diagnostics. |
| `system_X.par` | file | System `X` | Yes | Provides lattice scale, 3×3 primitive cell matrix, and fractional atomic coordinates. |
| `input_X.par` | file | System `X` | Yes | Stores `nBands`, `maxKE`, band indices (`idxVB`, `idxCB`, `idxGap`), plotting windows. |
| `kpoints_X.par` | file | System `X` | Yes | Lists fractional k-vectors and weights; optional `kpoints_X_orderMatrix.par` fixes band ordering. |
| `bandWeights_X.par` | file | System `X` | Yes | Supplies per-band weights applied in the loss (length must equal `nBands`). |
| `expBandStruct_X.par` | file | System `X` | Yes | Contains cumulative k-distance (column 0) plus target band energies. |
| `init_<atom>Params.par` | file | Each species | Required unless alternative initialization provided | Nine-parameter Zunger vectors per atom species. |
| `init_qSpace_pot.par` | file | Initialization | Optional | Overrides analytic initialization with tabulated q-space potentials. |
| `mcOpts*.par`, `<atom>ParamSteps.par` | files | Monte Carlo runs | Optional | Control MC iteration counts, perturbation magnitudes, or per-atom step sizes. |
| `expCoupling_*.dat`, `expDefPot*.par` | files | Specialized datasets | Optional | Provide optical couplings or defect-level shifts when relevant flags are enabled. |

Keep filenames consistent: atom ordering is inferred from `system_X.par` and reused to match `init_<atom>Params.par` and optional datasets. Once organized, continue with the [Configuration Reference](configuration.md) to understand every switch in `NN_config.par`.
