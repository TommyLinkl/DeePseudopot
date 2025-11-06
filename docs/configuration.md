# Configuration Reference (`NN_config.par`)

## File Format
Entries follow `key = value` with `#` introducing comments; whitespace is ignored.

## Core Keys
| Key | Type | Description |
| --- | --- | --- |
| `PPmodel` | string | Network architecture name defined in `utils/nn_models.py`.
| `hiddenLayers` | list of int | Layer widths; final entry equals the number of unique atom types.
| `nSystem` | int | Number of systems loaded from the bundle.
| `num_cores` | int | Multiprocessing workers (`0` disables parallel execution).
| `SHOWPLOTS` | bool (0/1) | Display interactive plots; disable on headless runs.
| `separateKptGrad`, `checkpoint` | bool | Memory/performance trade-offs. Enabling both triggers a warning.
| `SObool`, `cacheSO` | bool | Spin-orbit pathways and caching of SO/NL matrices.

## Gradient Training Keys
Active when `max_num_epochs > 0` and `mc_bool = 0`.

| Key | Type | Description |
| --- | --- | --- |
| `max_num_epochs` | int | Number of gradient-descent epochs.
| `optimizer` | string | `adam` (default) or `sgd`.
| `optimizer_lr` | float | Initial learning rate.
| `scheduler_gamma` | float | Exponential decay for the LR scheduler.
| `schedulerStep` | int | Scheduler step (epochs between decays).
| `patience` | int | Early-stopping patience; defaults to `max_num_epochs + 1` if absent.
| `plotEvery` | int | Plot/serialize frequency during training.
| `perturbEvery` | int | Apply random perturbation every `N` epochs (`-1` disables).

## Initialization Keys
Relevant when `init_Zunger_num_epochs > 0`.

| Key | Type | Description |
| --- | --- | --- |
| `init_Zunger_num_epochs` | int | Epochs to fit the NN to Zunger reference curves.
| `init_Zunger_optimizer_lr` | float | Learning rate for the initialization optimizer.
| `init_Zunger_scheduler_gamma` | float | Scheduler decay for initialization.
| `init_Zunger_plotEvery` | int | Interval for saving initialization plots and checkpoints.
| `init_Zunger_optimizer` | string | Optional (`adam` default, `sgd` alternative).
| `init_Zunger_printGrad` | bool | Dumps gradient diagnostics for debugging.

## Monte Carlo Keys
Set `mc_bool = 1` and leave `max_num_epochs = 0` to activate.

| Key | Type | Description |
| --- | --- | --- |
| `mc_bool` | bool | Enables Monte Carlo exploration.
| `mc_iter` | int | Iterations per Monte Carlo block.
| `mc_percentage` | float | Fraction of parameters perturbed in each proposal.
| `mc_beta` | float | Acceptance inverse temperature.
| `mc_perturb_mode` | int | Perturbation kernel (consult `utils/NN_train.py`).
| `mc_beta_schedule` | optional | Auxiliary files (`mcOpts_beta.par`) can vary temperature.

## Diagnostics and Profiling
Optional toggles:
- `memory_flag = 1` instructs the code to print `mprof` instructions.
- `runtime_flag = 1` enables additional runtime logging.
- `printGrad`, `smooth_reorder`, `eigvec_reorder` adjust debugging verbosity and band-order handling.

Pair these configuration choices with the file formats described in [System & Data Files](system-data.md) so your settings align with the inputs they control.
