# Long-range (LR) Parameter Gradient Plan

## Implemented
- Use a shared `nn.ParameterDict` to store all the atomisitic LR parameters. (`utils/read.py`) They are threaded through Hamiltonian construction so that local and nonlocal Hamiltonians pull those live LR tensors through the refreshed `pot_funcLR` function instead of static params[4]. Now the runtime `nn.Parameter` overrides the static tables without in-place mutation. (`utils/ham.py`)

- Optimizer owns a dedicated LR parameter group and the diagnostics now report LR gradient/value snapshots (`utils/init_NN_train.py`, `utils/NN_train.py`).

- Helper functions such as `get_lr_params` and `enforce_lr_constraint`, together with the deterministic gradient utilities, sit alongside the test module for the correctness of LR gradient implementation in `test_LR/` (`potential_lr_grad_check.py`, `test_LR/`).

- `trainLR=0` remains the default so legacy jobs stay fixed; LR tensors are only registered when explicitly requested (`utils/read.py`, `test_LR/inputs/input_0.par`). Downstream scripts (regression, Monte Carlo, etc.) keep working because they still receive `lr_params=None` unless the caller opts in.

- The test script `test_LR/test_lr_grad_check.py` now computes both finite-difference and autograd losses with the same weighting, and the serial branch performs a direct `weighted_loss` backward pass before reading gradients from `lr_params[...]`. The test also checks both analytic and NN variants with strict tolerances. 

## Tests
Run: `python -m test_LR/test_lr_grad_check.py`

## Todo
- Any pseudopotential plotting script must plot both the NN-local term alone and the NN-local + LR tail, in both real and reciprocal space. Also whenever we do Fourier Transform of the pseudopotential, vq shouldn't just be model(qGrid), it should be the NN + LR tail following pot_funcLR, if the parameter is non-zero. 
## Update 2025-09-22
- Added `test_LR/test_lr_training_grad_check.py` to exercise three-epoch LR training runs in both serial and parallel modes (with finite-difference cross-checks) for Zunger+LR and NN+LR cases.
- Extended pseudopotential plotting/FFT utilities to render NN-local and NN+LR tails in both q- and r-space, and ensured Fourier transforms use the combined potentials when LR coefficients are present.
- Threaded LR metadata (runtime parameters, static tables, LR gamma) through plotting and training helpers so diagnostics and exported files now reflect long-range tails alongside NN-local terms.
