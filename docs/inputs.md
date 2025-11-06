# Input Data Description

DeepPseudopot consumes a structured bundle of configuration files and reference data. Use this page to understand how the pieces fit together and jump to detailed references when you need exact keyword definitions or file layouts.

## Quick Checklist
- [ ] Confirm your runtime prerequisites in [Environment & Input Bundle](setup.md#runtime-environment).
- [ ] Populate `NN_config.par` with the workflow toggles and optimization settings described in the [Configuration Reference](configuration.md).
- [ ] Prepare one or more system-specific datasets (`system_X.par`, `input_X.par`, etc.) following the formats in [System & Data Files](system-data.md).
- [ ] Stage optional Monte Carlo, coupling, or defect datasets when your workflow activates the associated flags.

## File Families at a Glance
| Component | Purpose | Where to learn more |
| --- | --- | --- |
| Global configuration (`NN_config.par`) | Selects workflow mode, neural-network shape, optimizers, diagnostics, and hardware controls. | [Configuration Reference](configuration.md) |
| System descriptors (`system_X.par`, `input_X.par`) | Define lattice geometry, atom ordering, band counts, plotting windows, and metadata for each system. | [System & Data Files](system-data.md#system_xpar-lattice-and-species) |
| Spectral references (`kpoints_X.par`, `bandWeights_X.par`, `expBandStruct_X.par`) | Provide reciprocal-space sampling, loss weights, and target band energies used for supervision. | [System & Data Files](system-data.md#spectral-data) |
| Initialization assets (`init_<atom>Params.par`, `init_qSpace_pot.par`, `init_PPmodel.pth`) | Seed the neural network with analytic Zunger parameters or pretrained weights. | [System & Data Files](system-data.md#initialization-data) |
| Optional Monte Carlo controls (`mcOpts*.par`, `<atom>ParamSteps.par`, `mc_beta_schedule`) | Tune perturbation magnitudes, acceptance temperature, and per-atom step sizes when `mc_bool = 1`. | [Configuration Reference](configuration.md#monte-carlo-keys) |
| Auxiliary observables (`expCoupling_*.dat`, `expDefPot*.par`, `qpoints_X.par`) | Supply additional targets for workflows that fit optical couplings, defect potentials, or q-space profiles. | [System & Data Files](system-data.md#optional-data-and-auxiliary-controls) |

## Directory Layout Template
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

## Putting It Together
Once the bundle is staged:
1. Review the [Workflow Modes](workflows.md) section to choose gradient training, Monte Carlo refinement, or initialization-only execution.
2. Double-check configuration toggles (e.g., `mc_bool`, `SObool`, `separateKptGrad`) so they match the files you prepared.
3. Launch the driver following the [Quick Start](install.md) instructions.

If the run emits input-related errors, consult the validation notes in [System & Data Files](system-data.md) and [Troubleshooting](troubleshooting.md) for common fixes.
