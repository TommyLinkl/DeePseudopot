# Monitoring & Utilities

For a complete catalogue of exported artefacts, visit the [Output Data Description](outputs.md). This page focuses on how to track progress while a run is active, recover from interruptions, and apply the developer tools bundled with the repository.

## Monitoring, Restart, and Checkpoints
| Key / Artefact | Type | Applies to | Required? | Notes |
| --- | --- | --- | --- | --- |
| Stdout/stderr logs | stream | All runs | Yes | Captures configuration echoes and warnings; redirect to `run.log` for long jobs. |
| `epoch_<N>_PPmodel.pth`, `epoch_<N>_plot*.{pdf,png}` | files | Gradient runs | Optional outputs | Copy the desired epoch checkpoint to `<inputs>/init_PPmodel.pth` (and `init_AdamState.pth` if keeping optimizer state) for restarts. |
| `mc_checkpoint.pth`, `best_pot.*`, `best_plotPP.*` | files | Monte Carlo runs | Optional outputs | Promote the chosen MC checkpoint into the next run by copying it to `init_PPmodel.pth`. |
| `separateKptGrad` | int (0/1) | Memory tuning | Optional | When set to 1, recomputes gradients per k-point to lower peak memory. |
| `checkpoint` | int (0/1) | Memory tuning | Optional | Saves memory by re-evaluating activations; trades compute for RAM. |

## Utilities and Tests
- `inflate_kpoints.py` – generates denser k-paths; pass input file paths as arguments.
- `plot_BS_from_file.py`, `plot_SOC_NL_T_Vloc.py` – post-process stored potentials and band structures.
- `utils/pp_func.py` – callable from notebooks for Fourier transforms and plotting helper functions.
- **Testing**: `pytest test_ham` validates Hamiltonian assembly and band fitting against stored references; `pytest test_parallel` checks shared-memory multiprocessing; `pytest test_memory` exercises profiling hooks. Run these after modifying core logic or introducing new options.

For additional context, cross-reference the project README for architectural motivation and high-level usage scenarios. If you encounter issues, jump to the [Troubleshooting Guide](troubleshooting.md).
