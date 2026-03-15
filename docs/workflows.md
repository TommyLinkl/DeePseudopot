# Workflow Modes

DeepPseudopot's utility can be expanded along three directions: 

(1) selecting the pseudopotential model family, 

(2) choosing an optimization workflow, and 

(3) deciding how to initialize that model. 

This page summarizes each knob so you can assemble the combination that matches your study. Please see our paper for theoretical details. 

## 1. Pseudopotential Model Selection

$$
\hat{H}=\hat{T}+\hat{V}_{\mathrm{loc}}+\hat{V}_{\mathrm{nl}}+\hat{V}_{\mathrm{soc}},
$$

$$
\hat{V}_{\mathrm{loc}}=\sum_{\alpha}^{\mathrm{N_{at}}}\hat{v}_{\mathrm{loc}}^{\alpha},\quad
\hat{V}_{\mathrm{nl}}=\sum_{\alpha}^{\mathrm{N_{at}}}\hat{v}_{\mathrm{nl}}^{\alpha},\quad
\hat{V}_{\mathrm{soc}}=\sum_{\alpha}^{\mathrm{N_{at}}}\hat{v}_{\mathrm{soc}}^{\alpha}.
$$

### Neural-network local potentials
DeepPseudopot's default use case is to learn a screened local pseudopotential modeled by a deep neural network, combined with a non-local (NL) angular momentum-dependent correction term and a spin orbit coupling (SOC) term supplied in the bundle. 

$$
v_{\mathrm{loc}}^{\alpha}\left(G\right) = \Bigl[\left(h^{H} \circ \cdots \circ h^{2} \circ h^{1} \right)\left(G\right)\Bigr]_{\alpha}, 
$$
where $h^{i}(x) = \sigma\bigl(W^{i} x + b^{i}\bigr)$ is the output of the $i$-th hidden layer, $\sigma$ is an activation function, and $W^{i}$ and $b^{i}$ are the weights and bias tensors.

The following `NN_config.par` keys govern the NN flavor, particularly `PPmodel`. The `PPmodel` catalog spans sigmoid, tanh, ReLU, and CELU activations with Xavier or He initialization, optional batch normalization, logistic decays, and Gaussian gating (`*_decayGaussian`). Gaussian-gated variants guarantee smooth decay at large $G$ in the reciprocal space; logistic decays (`*_decay`) offer a softer cutoff.

| Key | Type | Applies to | Required? | Notes |
| --- | --- | --- | --- | --- |
| `PPmodel` | string | All NN runs | Yes | Class name from `utils/nn_models.py` (e.g., `Net_relu_xavier`, `Net_celu_HeInit_decayGaussian`). Determines activation, normalization, dropout, and gating.
| `hiddenLayers` | list[int] | All NN runs | Yes | Number of neurons in each hidden layer, not including the input and output layers which are set automatically by the code. E.g., `64 64` for a fully connected MLP with `[1, 64, 64, 3]` for three elements.
| `PPmodel_decay_rate` | float | `Net_*_decay` variants | Only for decay models | Logistic decay rate in the reciprocal space used to attenuate the NN output at large $G$.
| `PPmodel_decay_center` | float | `Net_*_decay` variants | Only for decay models | Logistic inflection point (in units of Bohr$^{-1}$) specifying where attenuation begins.
| `PPmodel_gaussian_std` | float | `Net_*_decayGaussian` variants | Only for decayGaussian models | Standard deviation $\sigma$ in the Gaussian envelope $\exp[-q^2/(2\sigma^2)]$.
| `PPmodel_scale` | list[float] | `Net_celu_HeInit_scale_decayGaussian` | Only for scale variant | Per-species scaling factors applied after the Gaussian envelope; length must equal the number of atom types.

### Analytic local pseudopotential form ('Zunger form')

DeepPseudopot also retains the functionality to fit semi-empirical local potential forms first proposed by Wang and Zunger for III-V and II-VI semiconductors ([Phys. Rev. B **51**, 17398](https://doi.org/10.1103/PhysRevB.51.17398)):

$$
V_{\text{Zunger}}(G) = \frac{a_0\,(G^2 - a_1)}{a_2\,\exp(a_3 G^2) - 1}.
$$

Parameters $a_0 \dots a_3$ map to `ppParams[0:4]` within each `init_<atom>Params.par` input file. You can:

1. Use the analytic form as an initialization target for the NN local pseudopotential, which trains the NN to reproduce $V_{\text{Zunger}}(G)$ before band-structure fitting, or
2. Optimize the analytic potential directly by setting `PPmodel = ZeroFunction` (or a similar pass-through) so that only the Zunger coefficients are varied via Monte Carlo/gradient updates (more in [Workflow Selection](#2-workflow-selection)).

Relevant keys:

| Key | Type | Applies to | Required? | Notes |
| --- | --- | --- | --- | --- |
| `init_<atom>Params.par` | file | Each species | Required when using analytic form | Provides the nine-parameter vectors (local, long-range, SO, NL, strain) consumed by `init_ZungerPP` and `pot_func`. |
| `init_Zunger_num_epochs` | int | Initialization | Optional | When >0, trains the NN to match the analytic potential before optimization. |
| `ZeroFunction` | `PPmodel` choice | Analytic-only fits | Optional | Choose `PPmodel = ZeroFunction` (or another noop architecture) if you want to optimize just the analytic coefficients. |

### Long-range (LR), spin-orbit coupling (SOC), nonlocal (NL), and strain terms
Regardless of whether the local piece is NN-based or analytic-form-based, the remaining channels share the same parameter slots described in the README ("Order of Pseudopotential Parameters"). These channels can be toggled independently, letting you build hybrids in any combination with the local pseudopotential. 

| Key / Parameter | Type | Applies to | Required? | Notes |
| --- | --- | --- | --- | --- |
| `ppParams[4]` (long-range) | float per species (N-1 independent) | Polar materials | Optional | Enables the Gaussian-screened Fröhlich tail with attenuation `LRgamma`. Leave it as zero for non-polar systems. |
| `ppParams[5]` (spin–orbit) | float per species | When `SObool = 1` | Optional | Set `SObool = 1` to include SO; you may still zero `ppParams[5]` to keep only nonlocal terms active. |
| `ppParams[6-7]` (nonlocal) | float per species and non-local channel | When `SObool = 1` | Optional | Set the strength of the projectors used for the nonlocal part. |
| `ppParams[8]` (strain) | float per species | Strain workflows | Optional | Controls deformation-potential responses when fitting strain-dependent observables. |
| `SObool` | int (0/1) | Any run | Optional | Turns SO projectors on/off. With `SObool = 1` and `ppParams[5] = 0`, you effectively reuse the NL blocks while keeping SO inactive. |
| `cacheSO` | int (0/1) | Runs with heavy SO/NL reuse | Optional | Caches SO/NL matrices in shared memory. **This is highly recommended when training pseudopotentials with SOC terms but it also increases RAM usage.** |


## 2. Workflow Selection

DeepPseudopot supports four core execution modes:

### Gradient-based training 

Highly recommended when using a neural network local potential. 

- Set `max_num_epochs > 0` and `mc_bool = 0` to use this. 
- Provide an `optimizer` (`adam`, `sgd`, `adamw`, etc.), `optimizer_lr`, `scheduler_gamma`, and `schedulerStep`. 
- Optional `perturbEvery` injects random kicks to escape shallow local minima. 
- Expect periodic `epoch_<N>_*` checkpoints plus `plotBS`/`plotPP` plots.
- The total training objective can combine the band-structure loss with optional reciprocal-space penalties (`penalize_lambda`, `penalize_mag_lambda`) and, when enabled, deformation-potential or coupling terms.
- In gradient runs, deformation potentials are treated as global transition observables. Same-k and different-k rows from `expDefPot_X.par` are therefore handled consistently in serial, `separateKptGrad`, and multiprocessing modes.

| Key | Type | Required? | Notes |
| --- | --- | --- | --- |
| `optimizer` | string | Optional | Supported: `adam`, `sgd`, `lbfgs`, `rmsprop`, `adamw`; see function `init_optimizer` in `utils/init_NN_train.py` for the complete list. Default is `adam`. |
| `optimizer_lr` | float | Yes | Base learning rate for the chosen optimizer. |
| `scheduler_gamma` | float | Optional | Multiplicative decay factor (e.g., 0.95). |
| `schedulerStep` | int | Optional | Number of epochs between scheduler updates. |
| `perturbEvery` | int | Optional | Apply random parameter perturbation every `N` epochs; `-1` disables. |
   
### Monte Carlo fitting (parallel tempering)
Preferred for traditional Zunger, as well as for NL/SO parameters where the loss landscape is highly rugged and non-convex. 

- Set `mc_bool = 1` and `max_num_epochs = 0`. 
- Tune `mc_iter`, `mc_percentage`, `mc_beta`, and helper files (`mcOpts*.par`, `<atom>ParamSteps.par`, `mc_beta_schedule`) to balance exploration and acceptance. 
- Checkpoints are written in `mc_checkpoint.pth`/`best_pot.*` that can be used in future runs. 
- The Monte Carlo code still uses its legacy defPot path. If you need row-based same-k / different-k transition targets from `expDefPot_X.par`, use the gradient workflow.

| Key | Type | Required? | Notes |
| --- | --- | --- | --- |
| `mc_iter` | int | Yes | Number of trial moves per MC block. |
| `mc_percentage` | float | Yes | Fraction of parameters perturbed each move (0–1). |
| `mc_beta` | float | Yes | Inverse temperature controlling acceptance; higher values favor downhill moves. |
| `mcOpts*.par` | files | Optional | Provide the schedule, step sizes, and temperature settings for parallel tempering. |
| `<atom>ParamSteps.par` | files | Optional | Provide per-atom per-parameter step size modifications. |
| `mc_beta_schedule` | file | Optional | Enables simulated annealing/tempering across betas. |

### Initialization-only / validation sweeps

- Leave `max_num_epochs = 0` and `mc_bool = 0`. 
- The driver loads `init_PPmodel.pth` (or analytic parameters in `init_<atom>Params.par`), evaluates band structures/deformation potentials, and exits. 
- Use this mode to sanity-check bundles or export analytic potentials without training.

### Band-structure evaluation 

- Run `python eval_fullBand.py /path/to/inputs/ /path/to/results/` with any initialized pseudopotential to score new systems or dense $\mathbf{k}$-paths. 
- The script `eval_fullBand.py` is optimized for low memory use and parallel efficiency.

### Hybrid workflows

- Users can connect a chain of Zunger initialization, Monte Carlo fitting, and gradient-based training of NN local potentials by reusing the `final_pot*`, `epoch_<N>_*`, or `mc_checkpoint.pth` files as the initialization files for the next stage.

After picking a workflow, consult the [Output Data Description](outputs.md) for expected outputs and use [Troubleshooting Guide](troubleshooting.md) for restart procedures.


## 3. Initialization Strategies

Choosing the initial parameters to the neural network pseudopotential is very important and can save a lot of effort in training them. By default, the DeepPseudopot code goes throught the following options in order to provide a good initialization. The users can also pick whichever option that matches your need on hand

1. **Checkpoint reuse** — If `init_PPmodel.pth` (and optionally `init_AdamState.pth`) exists, the weights of the model (and optionally optimizer state) will be loaded directly. This is the fastest restart path and preserves optimizer momentum when desired.
2. **Fit to Zunger potentials** — When `init_Zunger_num_epochs > 0`, the NN initializes by training against `init_<atom>Params.par` targets before training against band structures and deformation potentials. This initialization training is controlled by `init_Zunger_optimizer`, `init_Zunger_optimizer_lr`, `init_Zunger_scheduler_gamma`, and `init_Zunger_plotEvery`.

    | Key | Type | Required? | Notes |
    | --- | --- | --- | --- |
    | `init_Zunger_optimizer` | string | Optional | Supported: `adam` (default) or `sgd`. |
    | `init_Zunger_optimizer_lr` | float | Required when init training enabled | Learning rate for the initialization phase. |
    | `init_Zunger_scheduler_gamma` | float | Optional | Multiplicative LR decay during initialization. |
    | `init_Zunger_plotEvery` | int | Optional | Interval (epochs) for saving `initZunger_plotPP.*` diagnostics. |

3. **Fit to tabulated $v(G)$** — Supplying `init_qSpace_pot.par` bypasses the analytic form; the NN will learn the provided reciprocal-space table during initialization. This option is well suited for initializing the semi-empirical pseudopotential from ab initio pseudopotential references or a known function. 
4. **He/Kaiming or Xavier initialization** — If `PPmodel` is one of the `Net_*HeInit*` or `Net_*xavier*` classes and `init_Zunger_num_epochs = 0`, the network relies solely on the underlying torch initializer. This is useful for cases when no analytic prior exists. 
5. **Randomized CELU variants** — Setting `PPmodel = Net_celu_RandInit*` samples weights from a custom normal distribution, allowing truly random weights for uncertainty estimation.

The selected initialization strategy is printed in the log files for traceability.

## Overall Execution Flow of `main.py`
DeepPseudopot's main driver (`main.py`) advances through the following stages. Please use them as checkpoints when customizing runs or debugging new observables:

1. **Configuration Ingest** – Parse `NN_config.par`, validate mutually exclusive toggles (Monte Carlo vs. gradient, SO flags, etc.), and record global settings. Load the structural and spectral inputs for each system (`system_X.par`, `input_X.par`, `kpoints_X.par`, `expBandStruct_X.par`, optional weights) and construct `BulkSystem` objects while preserving atom ordering. Refer to [Input Data Description](inputs.md) for the companion files expected alongside the config.
2. **Model Instantiation and Initialization** – Build the neural network architecture specified by `PPmodel`/`hiddenLayers` and prepare it for either analytic or NN-based optimization. Initialize the model from `init_PPmodel.pth`, `init_qSpace_pot.par`, or by fitting analytic Zunger parameters (`init_<atom>Params.par`). Output artefacts such as `initZunger_plotPP.*` are generated here.
3. **Baseline Evaluation** – Compute the band structure of the initialized potential and write `oldFunc_plotBS.pdf` / `initZunger_plotBS.pdf` for reference.
4. **Optimization Loop** – Depending on `NN_config.par`, run gradient-based training mode (`bandStruct_train_GPU`, `max_num_epochs > 0`, `mc_bool = 0`) or Monte Carlo mode (`runMC_NN`, `mc_bool = 1`), producing `epoch_<N>_*`, `training_cost.dat`, or `mc_checkpoint.pth` outputs.
5. **Fourier Transform & Export** – Outputs real- and reciprocal-space potentials via `FT_converge_and_write_pp` and `write_PP_qSpace` (`*_pot.dat`, `*_qSpace_pot.dat`).
6. **Post-processing & cleanup** – Generate optional animations with `genMovie`, remove temporary MC plots, and release shared-memory caches before exit.

## Extended Toolkit
| Script | Purpose |
| --- | --- |
| `charge_density_from_wfns.py` | Builds real-space charge densities from plane-wave eigenvectors calculated from DeepPseudopot.
| `convert_bgwBS.py` | Translates BerkeleyGW or Quantum ESPRESSO outputs into the DeepPseudopot input bundle format.
| `convert_convCell_to_primCell.py` | Converts conventional cells into primitive cells during input preparation.
| `utils/cluster_pp.py` | Performs PCA/K-means analyses to cluster neural network pseudopotentials and assess coverage.
| `plot_BS_from_file.py` | Plotting scripts for band structures.
| `plot_SOC_NL_T_Vloc.py` | Plotting scripts for decomposed Hamiltonian components.
