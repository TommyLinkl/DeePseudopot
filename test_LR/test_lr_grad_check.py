import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

from utils.ham import Hamiltonian
from utils.read import BulkSystem, read_PPparams, read_NNConfigFile
from utils.NN_train import calcEigValsAtK_wGrad_parallel, weighted_mse_energiesAtKpt, weighted_relative_mse_energiesAtKpt

def build_hamiltonian(inputs_dir: Path):
    system = BulkSystem()
    system.setSystem(f"{inputs_dir / 'system_0.par'}")
    system.setInputs(f"{inputs_dir / 'input_0.par'}")
    system.setKPointsAndWeights(f"{inputs_dir / 'kpoints_0.par'}")
    system.setBandWeights(f"{inputs_dir / 'bandWeights_0.par'}")
    system.setExpBS(f"{inputs_dir / 'expBandStruct_0.par'}")

    atom_pp_order = np.unique(system.atomTypes)
    train_lr = getattr(system, 'trainLR', False)
    pp_params, _, lr_params = read_PPparams(atom_pp_order, f"{inputs_dir}/init_", train_lr=train_lr)
    nn_config = read_NNConfigFile(f"{inputs_dir / 'NN_config.par'}")

    device = torch.device("cpu")
    ham = Hamiltonian(system, pp_params, atom_pp_order, device, NNConfig=nn_config, lr_params=lr_params)
    ham.NN_locbool = False
    return ham, system, lr_params, atom_pp_order


def weighted_loss(ham, system, requires_grad):
    total_loss = None
    for kidx in range(system.getNKpts()):
        eigvals = ham.calcEigValsAtK(kidx, cachedMats_info=None, requires_grad=requires_grad)
        if system.relE_bIdx != -1:
            loss = weighted_relative_mse_energiesAtKpt(eigvals, system, kidx, system.relE_bIdx)
        else:
            loss = weighted_mse_energiesAtKpt(eigvals, system, kidx)
        weighted_loss_k = loss * system.kptWeights[kidx]
        total_loss = weighted_loss_k if total_loss is None else total_loss + weighted_loss_k
    if total_loss is None:
        raise RuntimeError("No k-points available when computing weighted loss.")
    return total_loss


def serial_gradient(ham, system, lr_params, atom_label, use_nn_loc):
    ham.NNConfig['num_cores'] = 0
    ham.NN_locbool = use_nn_loc
    model = ham.get_NNmodel()
    optimizer = torch.optim.SGD([
        {'params': model.parameters(), 'lr': 0.0},
        {'params': lr_params.parameters(), 'lr': 0.0},
    ])

    optimizer.zero_grad()
    loss = weighted_loss(ham, system, requires_grad=True)
    loss.backward()

    grad_tensor = lr_params[atom_label].grad
    if grad_tensor is None:
        raise RuntimeError("Expected gradient for long-range parameter, but found None.")

    grad_value = grad_tensor.detach().clone()
    optimizer.zero_grad()
    return grad_value.item()


def finite_difference(ham, system, lr_params, atom_label, use_nn_loc, eps=1e-4):
    ham.NN_locbool = use_nn_loc
    param = lr_params[atom_label]
    with torch.no_grad():
        original = param.item()
        param.data += eps
        plus = weighted_loss(ham, system, requires_grad=False).item()
        param.data -= 2 * eps
        minus = weighted_loss(ham, system, requires_grad=False).item()
        param.data.fill_(original)
    return (plus - minus) / (2 * eps)


def parallel_gradient(ham, system, lr_params, atom_label, use_nn_loc):
    ham.NNConfig['num_cores'] = max(2, ham.NNConfig.get('num_cores', 0))
    if ham.NNConfig['num_cores'] < 2:
        raise AssertionError("Parallel gradient test expects at least two worker processes.")
    ham.NN_locbool = use_nn_loc
    model = ham.get_NNmodel()
    optimizer = torch.optim.SGD([
        {'params': model.parameters(), 'lr': 0.0},
        {'params': lr_params.parameters(), 'lr': 0.0},
    ])

    args = [
        (kidx, ham, system, optimizer, model, None, None)
        for kidx in range(system.getNKpts())
    ]
    with mp.Pool(ham.NNConfig['num_cores']) as pool:
        if getattr(pool, "_processes", 1) < 2:
            raise AssertionError("Multiprocessing pool failed to spawn multiple workers.")
        results = pool.starmap(calcEigValsAtK_wGrad_parallel, args)

    gradients_dict = {}
    for grad_dict, *_ in results:
        for key, value in grad_dict.items():
            if key not in gradients_dict:
                gradients_dict[key] = value.clone()
            else:
                gradients_dict[key] += value
    key = f"lr_params.{atom_label}"
    return gradients_dict[key].item()


def main():
    inputs_dir = Path(__file__).resolve().parent / "inputs"
    for use_nn_loc in (False, True):
        ham, system, lr_params, atom_pp_order = build_hamiltonian(inputs_dir)
        atom_label = str(atom_pp_order[0])
        model = nn.Linear(1, len(lr_params), bias=False).double()
        ham.set_NNmodel(model)
        ham.NN_locbool = use_nn_loc

        fd_grad = finite_difference(ham, system, lr_params, atom_label, use_nn_loc)
        serial_grad = serial_gradient(ham, system, lr_params, atom_label, use_nn_loc)
        parallel_grad = parallel_gradient(ham, system, lr_params, atom_label, use_nn_loc)

        mode_label = "NN local + LR tail" if use_nn_loc else "Zunger local potential + LR tail"
        max_diff = max(abs(serial_grad - fd_grad), abs(parallel_grad - fd_grad))
        if max_diff > 5e-6:
            raise AssertionError(
                f"Gradient mismatch ({mode_label}): max diff {max_diff:.3e}"
            )
        print(f"Long-range parameter gradient check. ")
        print(f"Mode: {mode_label}")
        print(f"Finite difference: {fd_grad:.6e}")
        print(f"Autograd (serial): {serial_grad:.6e}  |  diff = {abs(serial_grad - fd_grad):.3e}")
        print(f"Autograd (parallel accumulation): {parallel_grad:.6e}  |  diff = {abs(parallel_grad - fd_grad):.3e}")
        print("\n"*3)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
