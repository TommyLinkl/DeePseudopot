# Workflow Modes

DeepPseudopot offers several workflows: 

- **Deterministic gradient training**: `max_num_epochs > 0`, `mc_bool = 0`. Provide optimizer and scheduler keys. Use `perturbEvery` for occasional random kicks without entering full Monte Carlo.
- **Monte Carlo refinement**: `mc_bool = 1`, `max_num_epochs = 0`. Tune `mc_iter`, `mc_percentage`, `mc_beta`, and per-atom step files to balance exploration vs. acceptance. Copy `mc_checkpoint.pth` into `init_PPmodel.pth` to continue later.
- **Initialization-only runs**: leave both `max_num_epochs` and `mc_bool` at zero. Useful when generating starting potentials or verifying input integrity.
- **Spin–orbit workflows**: enable `SObool = 1` and ensure the bundle contains spin–orbit Hamiltonian components. `cacheSO = 1` accelerates repeated diagonalizations; monitor memory usage accordingly.

After selecting a workflow, consult the [Output Data Description](outputs.md) for the artefacts you should expect, then use [Monitoring & Utilities](outputs-monitoring.md) to track progress and manage restarts.



## Execution Flow
DeepPseudopot's main driver (`main.py`) advances through the staged workflow below. Use each stage as a mental checkpoint when customizing experiments or debugging new observables.

### Stage 1: Configuration ingest
`read_NNConfigFile` parses global training inputs from `NN_config.par`, checking for mutually exclusive options such as Monte Carlo versus gradient training.

### Stage 2: System assembly
`setAllBulkSystems` loads structural and spectral data (`system_X.par`, `input_X.par`, `kpoints_X.par`, `expBandStruct_X.par`, etc.), constructs `BulkSystem` objects, and preserves atom ordering from `system_X.par`.

### Stage 3: Model instantiation
`setNN` builds the neural architecture selected via `PPmodel` and `hiddenLayers`.

### Stage 4: Initialization
`init_ZungerPP` seeds the network from checkpoints (`init_PPmodel.pth`, `init_qSpace_pot.par`) or fits analytic Zunger potentials contained in `init_<atom>Params.par`.

### Stage 5: Baseline evaluation
`evalBS_noGrad` archives the reference band structure predicted by the initialized potentials (`oldFunc_plotBS.pdf`).

### Stage 6: Optimization
Depending on `NN_config.par`, the driver either runs **gradient training** (`bandStruct_train_GPU`, `max_num_epochs > 0`, `mc_bool = 0`) with periodic diagnostic plots (`epoch_<N>_*`), or **Monte Carlo exploration** (`runMC_NN`, `mc_bool = 1`) that perturbs parameters according to `mc_percentage` and `mc_beta`, emitting `mc_checkpoint.pth` snapshots.

### Stage 7: Fourier transform & export
`FT_converge_and_write_pp` and `write_PP_qSpace` produce real- and reciprocal-space potentials (`*_pot.dat`, `*_qSpace_pot.dat`).

### Stage 8: Post-processing
`genMovie` stitches saved frames into videos when `ffmpeg` is available and removes temporary Monte Carlo imagery.

### Stage 9: Cleanup
Shared-memory segments and caches are released prior to exit.

Understanding this flow helps when tailoring custom runs or injecting new observables. When you are ready to assemble the necessary inputs, move on to [Environment & Inputs](setup.md).


## Extended Toolkit
| Script | Purpose |
| --- | --- |
| `charge_density_from_wfns.py` | Builds real-space charge densities from plane-wave eigenvectors calculated from DeepPseudopot.
| `convert_bgwBS.py` | Translates BerkeleyGW or Quantum ESPRESSO outputs into the DeepPseudopot bundle format.
| `convert_convCell_to_primCell.py` | Converts conventional cells into primitive cells during input preparation.
| `utils/cluster_pp.py` | Performs PCA/K-means analyses to cluster neural network pseudopotentials and assess coverage.
| `inflate_kpoints.py` | Densifies k-point paths for higher-resolution band structure calculations.
| `plot_BS_from_file.py`, `plot_SOC_NL_T_Vloc.py` | Plotting scripts for band structures and decomposed potential components.

