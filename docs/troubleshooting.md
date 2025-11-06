# Troubleshooting Guide

- **Missing keys**: ensure `PPmodel`, `nSystem`, and `hiddenLayers` appear in `NN_config.par`. Additional keys become mandatory when certain modes are activated (e.g., `max_num_epochs > 0`).
- **Inconsistent band counts**: lengths of `bandWeights_X.par` and columns in `expBandStruct_X.par` must equal `nBands` in `input_X.par`.
- **Divergent training loss**: reduce `optimizer_lr`, adjust `bandWeights_X.par`, or tighten scheduler decay (`scheduler_gamma < 1`).
- **Movie export failures**: install `ffmpeg`, verify PNG frames exist, or disable movie generation by removing relevant options.
- **SciPy integration errors**: ensure `scipy >= 1.7` to access `quad_vec`.
- **Memory exhaustion**: combine `separateKptGrad`, `checkpoint`, or lower `maxKE`. Use `num_cores = 0` to disable multiprocessing on memory-constrained nodes.

Need to recheck configuration decisions after debugging? Revisit the [Configuration Reference](configuration.md) or [Workflow Modes](workflows.md) for deeper adjustments.
