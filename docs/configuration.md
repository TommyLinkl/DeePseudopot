# Global Configuration Settings (`NN_config.par`)

`NN_config.par` contains all the global runtime and calculation settings. 

The parser accepts `key = value` pairs, ignores blank lines, and strips anything after `#`. Booleans are written as `0`/`1`, and integer lists such as `hiddenLayers` are space-separated.

## Minimum Required Keys

| Key | Type | Purpose / Options |
| --- | --- | --- |
| `PPmodel` | string | Class name from `utils/nn_models.py`. <br>Common choices are: `Net_relu_xavier`, `Net_celu_HeInit_decayGaussian`, `Net_relu_xavier_BN_dropout_decayGaussian`, `Net_sigmoid_xavier_decayGaussian`, `Net_celu_RandInit_decayGaussian`, `ZeroFunction`. Each suffix encodes activation (relu/celu/sigmoid), initialization (xavier/HeInit/RandInit), normalization (`BN`), dropout, and gating (`decay`, `decayGaussian`). |
| `hiddenLayers` | list of int | Space-separated layer widths, e.g., `32 32`. <br>The code automatically prepends an 1D input layer ($\|\mathbf{G}\|$) and appends an output layer whose size equals the number of unique atoms inferred from `system_X.par`. |
| `nSystem` | int | Number of `system_X` datasets to ingest. Must match the number of indexed input bundles (`system_0.par`, `system_1.par`, ...). |

## Execution Controls & Diagnostic Flags
| Key | Type | Effect / Options |
| --- | --- | --- |
| `num_cores` | int | `0` disables multiprocessing (default). A positive value spawns that many worker processes for Hamiltonian assembly and caches SO/NL matrices in shared memory. <br>The value is clipped to the available CPU count. |
| `SHOWPLOTS` | bool (0/1) | `1` opens interactive Matplotlib windows for band/potential plots. Set `0` for cluster or batch runs to avoid GUI errors. |
| `separateKptGrad` | bool (0/1) | When `1`, separates the band-structure loss by each $\mathbf{k}$-point and accumulates gradients manually. Recommended for memory-heavy gradient runs. Slightly increases runtime but drastically shrinks peak memory usage. Auxiliary observables such as deformation potentials and coupling targets are still evaluated as global system-level loss terms, so they remain consistent with the non-separated workflow. |
| `checkpoint` | bool (0/1) | Enables PyTorch gradient checkpointing. Helpful in cases when memory limit is a major issue. Not generally recommended. Particularly, if both `checkpoint` and `separateKptGrad` are `1`, a warning is emitted because the run becomes slow. |
| `memory_flag` | bool (0/1) | Toggle for generating memory usage reports for debugging (run with `mprof` to generate the report and plots). |
| `runtime_flag` | bool (0/1) | Toggle for wall-clock timers and debugging print statements to stdout. |
| `printGrad` | bool (0/1) | Toggle for printing out per-layer gradient norms each epoch. Helps debug vanishing/exploding gradients or confirm that all atoms receive updates. |


## Spin-Orbit

See [Long-range (LR), spin-orbit coupling (SOC), nonlocal (NL), and strain terms](workflows.md#long-range-lr-spin-orbit-coupling-soc-nonlocal-nl-and-strain-terms) for additional context on how these channels enter the Hamiltonian.

| Key | Type | Effect / Options |
| --- | --- | --- |
| `SObool` | bool (0/1) | Toggle for the spin-orbit coupling matrices. Only enable when the bundle includes SO information in `init_<atom>Params.par` and the training data expects SOC splittings, as constructing the SOC Hamiltonian drastically increases runtime and memory. |
| `cacheSO` | bool (0/1) | Valid only when `SObool = 1`. Caches SO/NL matrices in shared memory for faster repeated access, but slightly memory hungry. `0` recomputes the matrices each epoch (slower but safer on limited RAM). <br>Strongly recommended to turn on when training with SO/NL parameters. |


## Model Architecture Details

For a broader discussion of the available NN architectures and analytic forms, see [Pseudopotential Model Selection](workflows.md#1-pseudopotential-model-selection).

| Key | Type | Effect / Options |
| --- | --- | --- |
| `PPmodel_decay_rate`, <br>`PPmodel_decay_center` | float | Used only by `*_decay` variants. `PPmodel_decay_rate` sets the steepness of the logistic tail; `PPmodel_decay_center` is the $\|\mathbf{G}\|$ value (in Bohr$^{-1}$) where decay begins. Increase `PPmodel_decay_rate` for sharper cutoffs. |
| `PPmodel_gaussian_std` | float | Used by `*_decayGaussian` variants. Controls the Gaussian envelope width in the reciprocal space. |
| `PPmodel_scale` | list of float | Exclusive to `Net_celu_HeInit_scale_decayGaussian`. Provide one scaling factor per atom species to bias certain channels (e.g., heavier atoms) during initialization. |
| `penalize_starting` | float | Lower bound (in Bohr$^{-1}$) of the reciprocal-space range where the penalty is applied. |
| `penalize_lambda` | float | Strength of the penalty that discourages deviations from the initialization curve beyond `penalize_starting`. Increase to keep the NN closer to the analytic prior. |
| `penalize_mag_threshold` | float | Global threshold applied independently to each atom-channel value of the reciprocal-space pseudopotential. The magnitude penalty acts on the excess `max(|v(G)| - penalize_mag_threshold, 0)` over the full sampled range `[0, 12]` Bohr$^{-1}$. |
| `penalize_mag_lambda` | float | Prefactor for the magnitude penalty. Set `<= 0` to disable this term entirely. |

> Tip: browse `utils/nn_models.py` to see the full catalog and the mathematical form of each `PPmodel`. The suffixes encode activation (`relu`, `celu`, `sigmoid`), initializer (`xavier`, `HeInit`, `RandInit`), and whether batch norm (`BN`), dropout, or decay gates are included.

> Numerical note: extremely large `penalize_mag_threshold` values are not useful in practice and can trigger numerical instability in the current implementation. The code emits a warning when `penalize_mag_threshold > 100`.

## Initialization Calculation Settings 

Additional guidance on these options appears in [Initialization Strategies](workflows.md#3-initialization-strategies).

These knobs control the initialization of the neural network pseudopotential's initialization against analytic potentials. Such initialization is run only when `init_Zunger_num_epochs > 0` and analytic `init_<atom>Params.par` vectors (or `init_qSpace_pot.par`) are present.

| Key | Type | Effect / Options |
| --- | --- | --- |
| `init_Zunger_num_epochs` | int | `0` (default) skips analytic pre-training. Any positive value trains against the Zunger (or tabulated) curves for that many epochs before fitting bands. |
| `init_Zunger_optimizer` | string | Optimizer for the initialization stage. <br>Supported: `adam` (default) or `sgd`. |
| `init_Zunger_optimizer_lr` | float | Learning rate during initialization. Required when `init_Zunger_num_epochs > 0`. |
| `init_Zunger_scheduler_gamma` | float | Exponential LR decay applied every `schedulerStep` init epochs. <br>Leave unset (or `=1.0`) to keep the LR constant. |
| `init_Zunger_plotEvery` | int | Frequency for saving `initZunger_plotPP.*` or `initZunger_epoch_*` outputs. Use large values (e.g., `500`) to minimize I/O, or `-1` to skip. |
| `init_Zunger_printGrad` | bool (0/1) | `1` prints gradient statistics during initialization, useful for diagnosing stagnant curves. |

> Order of initialization: <br>
Load `init_PPmodel.pth`/`init_AdamState.pth` if present, otherwise fit to `init_qSpace_pot.par`/`init_<atom>Params.par`, otherwise fall back to the weight initializer baked into `PPmodel` (Xavier/He/etc.).

## Gradient-Based Training Loop
Active only when `max_num_epochs > 0` and `mc_bool = 0`. See [Workflow Selection](workflows.md#2-workflow-selection) for guidance on when to use this mode versus Monte Carlo.

| Key | Type | Effect / Options |
| --- | --- | --- |
| `max_num_epochs` | int | Positive integer activates gradient training. |
| `optimizer` | string | PyTorch optimizer name. <br>Supported: `adam` (default), `sgd`, `asgd`, `lbfgs`, `adadelta`, `adagrad`, `adamw`, `sparseadam`, `adamax`, `nadam`, `radam`, `rmsprop`. |
| `optimizer_lr` | float | Initial learning rate for the chosen optimizer. |
| `sgd_momentum`, <br>`adam_beta1`, <br>`adam_beta2` | float | Optional overrides for the optimizer hyperparameters. Ignored unless the optimizer supports them. |
| `scheduler_gamma` | float | Multiplicative decay factor for the exponential LR scheduler (values <1 decay the LR). |
| `schedulerStep` | int | Number of epochs between scheduler updates. Combine with `scheduler_gamma` to control how quickly the LR decays. |
| `plotEvery` | int | Save `epoch_<N>_PPmodel.pth`, band plots, and cost snapshots every `plotEvery` epochs. |
| `patience` | int | Early-stopping patience measured in epochs. Defaults to `max_num_epochs + 1` (effectively disables early stopping). |
| `perturbEvery` | int | If >0, injects a random perturbation into NN weights every `perturbEvery` epochs. Set `-1` or omit to disable. |

> When switching optimizers mid-project, remove any stale `init_AdamState.pth` files; otherwise PyTorch will attempt to load incompatible states.

### Pre-adjustment Nudges (Optional)
| Key | Type | Effect / Options |
| --- | --- | --- |
| `pre_adjust_moves` | int | Number of single-parameter “pre-adjust” iterations to run before epoch 0. Each move optimizes the weight scalar with the largest gradient magnitude. <br>This is a handy feature when initial losses are huge as this method guarantees decrease of loss when learning rate is small. |
| `pre_adjust_stepSize` | float | Optional learning rate for the pre-adjust stage. Defaults to `optimizer_lr` if omitted. |

## Monte Carlo Mode
See [Workflow Selection](workflows.md#2-workflow-selection) for the decision chart comparing Monte Carlo refinement and gradient training.

Enable by setting `mc_bool = 1` while keeping `max_num_epochs = 0`. The Monte Carlo kernel honors the following keys:

| Key | Type | Effect / Options |
| --- | --- | --- |
| `mc_bool` | bool (0/1) | `1` activates Monte Carlo refinement. Requires `max_num_epochs = 0`. |
| `mc_iter` | int | Number of trial moves per Monte Carlo block. After each block the code logs cost traces and (optionally) writes plots. |
| `mc_perturb_mode` | int | Selects one of the perturbation kernels defined in `utils/NN_train.py::perturb_model` ($\varepsilon$ = `mc_percentage`): <br> **Mode 1** – Multiply every NN weight by a random factor in $[1-\varepsilon, 1+\varepsilon]$; SOC/NL parameters scale by $(1 \pm  \varepsilon/100)$. <br> **Mode 2** – Magnitude-aware perturbations: if the weight value has absolute value larger than 20, they are nudged back toward zero by at most $\varepsilon \cdot \|value\|$, near-zero values amplified by up to $10\cdot \varepsilon\cdot value$, SOC/NL scale by $(1 \pm  \varepsilon/1000)$. <br> **Mode 3** – Normalize each tensor, randomly shift 50% of entries by $N(0, \varepsilon)$ in normalized space, then denormalize (NN weights only). <br> **Mode 4** – Add absolute steps of $\pm \varepsilon$ to ~60% of NN weights and $\pm \varepsilon/1000$ to SOC/NL parameters. (Larger NN weights perturbations than SOC/NL parameters) <br> **Mode 5** – Add absolute steps of $\pm \varepsilon/10$ to ~60% of NN weights and $\pm \varepsilon$ to SOC/NL parameters. (Larger SOC/NL parameter perturbations than NN weights) <br> **Mode 6** – Keep NN weights fixed; perturb SOC parameters only by $\pm \varepsilon$. <br> **Mode 7** – Keep NN weights fixed; perturb NL parameters only by $\pm \varepsilon$. |
| `mc_percentage` | float | The \varepsilon magnitude passed to the selected perturbation mode. Increase to encourage larger exploration steps; decrease for conservative proposals. |
| `mc_beta` | float | Inverse temperature $\beta$ in the Monte Carlo Metropolis criterion. Larger $\beta$ makes the search greedier; smaller $\beta$ increases acceptance of uphill moves. |
| `mc_beta_schedule` | string | Optional path to a text file containing `iteration_index  beta` pairs. Overrides `mc_beta` to implement annealing/tempering schedules. |

The Monte Carlo workflow often read companion files described in [System & Data Files](system-data.md#4-monte-carlo-and-specialized-controls): `mcOpts*.par` (global MC settings), `<atom>ParamSteps.par` (per-atom step sizes), and `mc_beta_schedule` files.

## Eigenvalue Reordering
| Key | Type | Effect / Options |
| --- | --- | --- |
| `smooth_reorder` | bool (0/1) | Enables the smooth eigenvalue-matching heuristic. Use when frequent band crossings cause large loss spikes. |
| `eigvec_reorder` | bool (0/1) | Forces eigenvector-based band sorting (more robust but slower). |
