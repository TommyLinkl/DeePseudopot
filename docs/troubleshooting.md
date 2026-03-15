# Troubleshooting Guide

Use this page to monitor in-flight runs, restart cleanly after interruptions, and resolve the most common configuration or data issues. For a full catalogue of generated files, see the [Output Data Description](outputs.md).

## Monitor & Restart

| Artefact / Setting | Applies to | How to use |
| --- | --- | --- |
| Stdout / stderr logs | All runs | Tail logs in real time or redirect to `run.log` to capture warnings and configuration echoes. |
| `epoch_<N>_PPmodel.pth` | Gradient runs | Copy the desired epoch checkpoint to `<inputs>/init_PPmodel.pth` (and `init_AdamState.pth` if preserving optimizer state) to resume from that epoch. |
| `best_pot.*`, <br>`best_plotPP.*` | Monte Carlo | Promote the chosen file to `init_PPmodel.pth` and change the parameters to rerun for the next stage. |
| `separateKptGrad`, <br>`checkpoint` | Memory relief | Enable `separateKptGrad = 1` to process $\mathbf{k}$-points sequentially; add `checkpoint = 1` if memory pressure persists (expect slower runtimes). This changes only how the band loss is accumulated; defPot and coupling targets remain system-level terms in gradient runs. |

## Quick Fixes

| Symptom | Likely cause | Recommended fix |
| --- | --- | --- |
| Parser errors about missing keys | Required entries absent in `NN_config.par` | Ensure `PPmodel`, `hiddenLayers`, `nSystem`, and any mode-specific knobs (e.g., `max_num_epochs`, `mc_iter`) are present. |
| Band-count mismatch | `nBands` disagrees with `bandWeights_X.par` or `expBandStruct_X.par` | Regenerate inputs so `len(bandWeights) = nBands = columns(expBandStruct) - 1`. |
| Divergent or unstable loss | Step size too large or band weights skewed | Reduce `optimizer_lr`, confirm `scheduler_gamma < 1`, and inspect `bandWeights_X.par` for extreme values. |
| Memory exhaustion | Large `maxKE`, too many cores, no gradient splitting | Set `separateKptGrad = 1`, consider `checkpoint = 1`, reduce `maxKE`, or run with `num_cores = 0`. |

## Utilities & Self-Checks

- `inflate_kpoints.py` densifies $\mathbf{k}$-point paths for debugging convergence.
- `plot_BS_from_file.py`, `plot_SOC_NL_T_Vloc.py` visualize existing results without rerunning training.
- `utils/pp_func.py` exposes Fourier-transform helpers for notebooks.
