# Monitoring & Utilities

For a complete catalogue of exported artefacts, visit the [Output Data Description](outputs.md). This page focuses on how to track progress while a run is active, recover from interruptions, and apply the developer tools bundled with the repository.

## Monitoring, Restart, and Checkpoints
- **Logging**: stdout/stderr contain key configuration echoes and warnings (e.g., conflicting flags). Redirect to `run.log` for lengthy jobs.
- **Gradient restarts**: copy the desired `epoch_<N>_PPmodel.pth` to `<inputs>/init_PPmodel.pth`; optionally copy the associated optimizer state (`init_AdamState.pth`).
- **Monte Carlo restarts**: copy the latest `mc_checkpoint.pth` into the input directory as `init_PPmodel.pth`.
- **Memory management**: `separateKptGrad = 1` recomputes gradients per k-point (lower peak memory), while `checkpoint = 1` trades compute for memory by re-evaluating activations. Use `test_memory/` utilities to gauge headroom.

## Utilities and Tests
- `inflate_kpoints.py` – generates denser k-paths; pass input file paths as arguments.
- `plot_BS_from_file.py`, `plot_SOC_NL_T_Vloc.py` – post-process stored potentials and band structures.
- `utils/pp_func.py` – callable from notebooks for Fourier transforms and plotting helper functions.
- **Testing**: `pytest test_ham` validates Hamiltonian assembly and band fitting against stored references; `pytest test_parallel` checks shared-memory multiprocessing; `pytest test_memory` exercises profiling hooks. Run these after modifying core logic or introducing new options.

For additional context, cross-reference the project README for architectural motivation and high-level usage scenarios. If you encounter issues, jump to the [Troubleshooting Guide](troubleshooting.md).
