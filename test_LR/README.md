Run these tests from the root directory of the repo,
using the command:

python -m test_LR.test_lr_grad_check

(note 1) the '.' rather than '/', 2) no .py at the end)

`test_lr_grad_check.py` compares finite-difference and autograd gradients for the LR parameter in both analytic and NN-local modes, using a small Hamiltonian setup, shared optimizer wiring, and per-mode tolerances. It still relies on the reference data and assumes the pooled branch mirrors production; incorrect cached matrices or mismatched k-point weights could slip through if they perturb model and finite-difference paths in the same way.