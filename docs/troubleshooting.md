# Troubleshooting Guide

| Issue | Key / Option | Applies to | Fix |
| --- | --- | --- | --- |
| Missing keys | `PPmodel`, `nSystem`, `hiddenLayers`, mode-specific flags | `NN_config.par` | Ensure all required keys are present; additional options become mandatory when activating features (e.g., set `max_num_epochs` when training). |
| Inconsistent band counts | `bandWeights_X.par`, `expBandStruct_X.par`, `nBands` | Input bundle | Verify the number of bands matches across files (`len(bandWeights) == nBands == columns in expBandStruct`). |
| Divergent training loss | `optimizer_lr`, `scheduler_gamma`, `bandWeights_X.par` | Gradient workflow | Lower `optimizer_lr`, adjust weights, and ensure `scheduler_gamma < 1` for adequate decay. |
| Movie export failures | `ffmpeg`, plotting options | Post-processing | Install `ffmpeg`, confirm PNG frames exist, or disable movie generation in `NN_config.par`. |
| SciPy integration errors | `scipy` version | Initialization / transforms | Upgrade to `scipy >= 1.7` to access `quad_vec`. |
| Memory exhaustion | `separateKptGrad`, `checkpoint`, `maxKE`, `num_cores` | Large systems | Enable `separateKptGrad = 1`, `checkpoint = 1`, reduce `maxKE`, or set `num_cores = 0` to limit multiprocessing. |

Need to recheck configuration choices after debugging? Revisit the [Configuration Reference](configuration.md) or [Workflow Modes](workflows.md) for deeper adjustments.
