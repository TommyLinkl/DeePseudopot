# Input Data Description

DeepPseudopot consumes a structured bundle of configuration files and reference data. This page provides an overview, while the detailed definitions live in:

- [Global Calculations Settings](configuration.md) — every key accepted in `NN_config.par`.
- [System-specific Data Files](system-data.md) — formats for all other input files.

## Quick Checklist
- [ ] Select a [workflow mode](workflows.md). Populate `NN_config.par` using the categories documented in [Global Calculations Settings](configuration.md).
- [ ] If you want reciprocal-space regularization, set the optional penalty keys in `NN_config.par` such as `penalize_starting`, `penalize_lambda`, `penalize_mag_threshold`, and `penalize_mag_lambda`.
- [ ] Prepare each `system_X` family (`system_X.par`, `input_X.par`, `kpoints_X.par`, `bandWeights_X.par`, `expBandStruct_X.par`) following [System-specific Data Files](system-data.md).
- [ ] Add initialization sources (`init_<atom>Params.par`, `init_qSpace_pot.par`, `init_PPmodel.pth`). 
- [ ] Stage optional Monte Carlo, electron-phonon coupling, or reciprocal-space inputs, etc. only when the associated flags in `NN_config.par` or `input_X.par` are enabled.

## Input Directory Layout Template
Organize each run so that all required inputs live under a single directory, referenced as `inputs/` when calling `python main.py <inputs> <results>`. A minimal tree looks like:

```text
inputs/
├── NN_config.par
├── system_0.par
├── input_0.par
├── kpoints_0.par
├── bandWeights_0.par
├── expBandStruct_0.par
├── init_GaParams.par
├── init_NParams.par
└── ... (additional systems or optional files)
```

For multi-system bundles, repeat the `system_X` family with incremented indices (`X = 0, 1, ...`). Keep atom ordering consistent across related files so that initialization parameters align with the lattice definitions.

## File Families at a Glance
| Family | Applies to | Required? | Detailed reference |
| --- | --- | --- | --- |
| `NN_config.par` | Whole run | Yes | Workflow mode, NN architecture, optimizer, diagnostics. |
| `system_X.par`, <br>`input_X.par` | Each system `X` | Yes | Cell geometry, atom order, basis convergence parameters, band plotting hints. |
| `kpoints_X.par`, <br>`bandWeights_X.par`, <br>`expBandStruct_X.par` | Each system `X` | Yes | Band structure $\mathbf{k}$-paths, per-band weights, and reference band structure energies. |
| `qpoints_X.par` | Workflows that require electron-phonon coupling | Optional | Defines phonon sampling  $\mathbf{q}$-points in reciprocal space. |
| `init_<atom>Params.par`, <br>`init_qSpace_pot.par`, <br>`init_PPmodel.pth`, <br>`init_AdamState.pth` | Initialization stage | Optional | Targets of model initialization, choose among analytic Zunger params, tabulated potentials, or pretrained checkpoints. |
| `mcOpts*.par`, <br>`<atom>ParamSteps.par`, <br>`mc_beta_schedule` | Monte Carlo workflows (`mc_bool = 1`) | Optional | Configure MC movement step sizes, temperature schedules, or standalone MC scripts. |
| `expCoupling_*.dat`, <br>`expDefPot*.par` | Training targets | Optional | Additional target observables that enter into the loss definition (electron-phonon coupling or deformation potentials). |

Once everything checks out, proceed to [Workflow Modes](workflows.md) to pick a training mode and follow [Installation & Quick Start](install.md) for launch commands.
