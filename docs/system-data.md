# System and Data Files

This page documents all input files within the bundle for one or more systems (other than `NN_config.par`, which is detailed in [Global Calculations Settings](configuration.md)). 

Throughout, `X` denotes the system index (`0, 1, ...`) and `<atom>` refers to the chemical symbol in the order it appears inside `system_X.par`.

Conventions & Units: 

- Plain-text files accept comments after `#` unless otherwise noted.
- Keep atom ordering consistent across all files; the loader maps `<atom>`-specific inputs by position, not by name alone.
- All energies are in electron volts (eV) unless specified; lattice vectors `scale` is in Bohr. 

## 1. Geometry and Metadata
### `system_X.par`
Defines the cell and fractional atom positions. Minimal skeleton:
```
scale = 10.68309422

cell
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0

atoms
Ga 0.125 0.125 0.125
As -0.125 -0.125 -0.125
```

| Key | Description |
| --- | --- |
| `scale` | Scalar that multiplies the 3×3 cell matrix to yield **Bohr**-unit lattice vectors. |
| `cell` block | Three rows defining the primitive cell vectors in fractional coordinates before scaling. |
| `atoms` block | Block listing species followed by fractional coordinates. The order determines `<atom>` indices for every other file. |

### `input_X.par`

| Key | Type | Required? | Notes |
| --- | --- | --- | --- |
| `nBands` | int | Yes | Number of bands included in the reference band structure; must match `bandWeights_X.par` length. |
| `maxKE` | float | Yes | Plane-wave kinetic-energy cutoff in units of **Hartree** that determines the basis size. |
| `idxVB`, <br>`idxCB`, <br>`idxGap` | int | Optional | Zero-based band indices for valence/conduction edges and the band-gap definition used in plots. |
| `BS_plot_center`, <br>`BS_plot_CBVB_range`, <br>`BS_plot_CBVB_range_zoom` | float | Optional | Controls PDF y-limits for full and zoomed plots in separate panels. |
| `systemName` | string | Optional | Used in `*.POSCAR` headers and plot titles. |
| `fit_defPot` | bool | Optional | Enables deformation potential fitting and triggers `expDefPot_X.par` ingestion in the gradient-based workflows. |
| `relE_bIdx` | int | Optional | Supplies a relative band index across systems when the loss is defined with respect to relative energy differences. |

## 2. Spectral and Weighting Data
### `kpoints_X.par`
Lists reciprocal-space sampling for each system. 

Format: `k_x k_y k_z weight` per line, with **fractional coordinates** relative to the reciprocal lattice. Weights are used as supplied in the band-structure loss; the current loader does not renormalize them automatically. If you want averaging-style weights, normalize them manually yourself in the input file.

**Optional companion**: `kpoints_X_orderMatrix.par` (same base name) provides a fixed band-order matrix with shape `(nKpts × nBands)`. When present, DeepPseudopot uses it to track band connectivity, bypassing the automatic sorting heuristics.

### `bandWeights_X.par`
Single-column file with `nBands` entries. Values scale the per-band MSE contribution inside the training loss. Any mismatch between length and `nBands` results in an input error.

### `expBandStruct_X.par`
Reference band structures. Column 0 stores cumulative k-point distance (the plotting x-axis); remaining columns store band energies (one column per band). Header comments are allowed. **Units: eV**.

### `qpoints_X.par`
Structured identically to `kpoints_X.par` but applied to workflows that require explicit q-space sampling (e.g., coupling matrices on a dedicated grid). The loader looks for this file only when the active workflow calls `setQPointsAndWeights`.

### `expCoupling_*.dat`
Provides the electron-phonon coupling elements keyed by atom index. Files contain repeated blocks of the form:
```
Atom idx = <i>   atom = <symbol>   position = ...
vb-vb coupling elements. polarization of derivative = x
<values>
...
cb-cb coupling elements. polarization of derivative = z
<values>
```
`BulkSystem.setExpCouplings` parses each block and stores couplings as `(atom_idx, polarization, q_index, band_id)` tuples. Supply one file per system or polarization scenario and ensure that the band ordering matches `expBandStruct_X.par`.

### `expDefPot_X.par`
Sets deformation-potential targets as fixed-state transitions. The file is a table with seven columns:
`kidx_VB  bidx_VB  kidx_CB  bidx_CB  latConst_ratio  defPot_gap(eV)  weight`

The first four columns must be integers. `kidx_VB`/`bidx_VB` and `kidx_CB`/`bidx_CB` identify the two endpoint states of the transition. The two states may be at the same k-point or at different k-points. The final `weight` column is the explicit weighting for that target row; these targets are not reweighted by `kpoints_X.par`.

In the gradient-based workflows, these rows are handled consistently in serial, `separateKptGrad`, and multiprocessing runs. Each row is treated as a system-level observable rather than being attached to the current k-point loop index.

## 3. Initialization Files
### `init_<atom>Params.par`

Per-species coefficient vectors for the analytic Zunger form. The default ordering is nine parameters:

- `ppParams[0]` – `ppParams[3]`: Zunger-form local coefficients
- `ppParams[4]`: long-range parameter (only defined for $N-1$ species to enforce charge neutrality)
- `ppParams[5]`: spin–orbit coupling parameter
- `ppParams[6]`, `ppParams[7]`: nonlocal parameters
- `ppParams[8]`: strain-tensor parameter

Older datasets may only include the first 5 entries; remaining slots implicitly default to zero. Provide one file **per atom type** listed in `system_X.par`.

### `init_qSpace_pot.par`
Optional override for the initialization stage. The file contains one row per $q$ value, where $q$ is the magnitude of the reciprocal space vector $|\mathbf{G}|$:
```
q  V_atom0(q)  V_atom1(q)  ...
```
Each column after `q` follows the atom order defined in `system_X.par`. When present, this file replaces the analytic Zunger initialization curve.

### `init_PPmodel.pth` and `init_AdamState.pth`
PyTorch checkpoint files. Drop them into the inputs folder to restart from a previous NN (`init_PPmodel.pth`) and, optionally, resume the Adam optimizer state (`init_AdamState.pth`). When omitted, the code initializes weights using the constructor specified by `PPmodel`.

## 4. Monte Carlo and Specialized Controls
### `mcOpts*.par`
Text files to configure standalone Monte Carlo drivers (e.g., `main_mc_Zunger_*.py`). Supported keywords:

| Keyword | Example | Meaning |
| --- | --- | --- |
| `betas = <start> <stop>` | `betas = 100 1` | Defines the beta (inverse temperature) sweep. |
| `stepsAtTemp = <cooling> <heating>` | `stepsAtTemp = 5 5` | Number of MC steps spent at each beta on the down/up ramp. |
| `tempStepSizeMod = <down> <up>` | `tempStepSizeMod = 1.0 1.0` | Multiplier applied to the beta increments. |
| `totalIter = N` | `totalIter = 20` | Total number of MC iterations. |
| `fitDefPot`, <br>`fitCoupling`, <br>`fitEffMass`, <br>`optGaps` | `fitCoupling = True` | Boolean flags that enable auxiliary observables in the MC cost. |
| `defPotWeight`, <br>`writePerIter` | `defPotWeight = 0.5` | Optional floats that reweight deformation potentials or control logging cadence. The current Monte Carlo code still uses its older defPot interface rather than the row-based `expDefPot_X.par` transition table described above. |

### `<atom>ParamSteps.par`
When present, these files specify per-parameter perturbation magnitudes for Monte Carlo proposals. The expected layout is one float per NN/Zunger parameter (typically nine entries) matching the `<atom>` order defined in `system_X.par`. Omit these files to let the MC code infer default step sizes from `mc_percentage`.

### `mc_beta_schedule`
Optional annealing script. Provide two columns per line — `iteration_index  beta` — to override the constant `mc_beta` value. The runtime reads entries sequentially and wraps/holds the last value once the file is exhausted.
