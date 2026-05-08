import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

from typing import Tuple
import torch
import numpy as np
import torch.nn as nn

torch.set_default_dtype(torch.float64)

from utils.ham import Hamiltonian
from utils.read import BulkSystem, read_PPparams, read_NNConfigFile
from utils.NN_train import calcEigValsAtK_wGrad_parallel, weighted_mse_energiesAtKpt, weighted_relative_mse_energiesAtKpt


@dataclass
class TrainingState:
    ham: Hamiltonian
    system: BulkSystem
    lr_params: nn.ParameterDict
    optimizer: torch.optim.Optimizer
    model: nn.Module
    atom_label: str
    use_nn_loc: bool
    num_cores: int


@dataclass
class EpochResult:
    prev_value: float
    value: float
    grad: float
    finite_diff: float


def build_hamiltonian(inputs_dir: Path, num_cores: int) -> Tuple[Hamiltonian, BulkSystem, nn.ParameterDict, torch.device, list]:
    system = BulkSystem()
    system.setSystem(str(inputs_dir / 'system_0.par'))
    system.setInputs(str(inputs_dir / 'input_0.par'))
    system.setKPointsAndWeights(str(inputs_dir / 'kpoints_0.par'))
    system.setBandWeights(str(inputs_dir / 'bandWeights_0.par'))
    system.setExpBS(str(inputs_dir / 'expBandStruct_0.par'))

    atom_pp_order = np.unique(system.atomTypes)
    pp_params, _, lr_params = read_PPparams(atom_pp_order, f"{inputs_dir}/init_", train_lr=system.trainLR)
    nn_config = read_NNConfigFile(str(inputs_dir / 'NN_config.par'))
    nn_config['num_cores'] = num_cores

    device = torch.device('cpu')
    ham = Hamiltonian(system, pp_params, atom_pp_order, device, NNConfig=nn_config, lr_params=lr_params)
    ham.NNConfig['num_cores'] = num_cores
    return ham, system, lr_params, device, atom_pp_order


def weighted_loss(ham: Hamiltonian, system: BulkSystem, requires_grad: bool) -> torch.Tensor:
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
        raise RuntimeError('No k-points available when computing weighted loss.')
    return total_loss


def finite_difference(ham: Hamiltonian, system: BulkSystem, lr_params: nn.ParameterDict, atom_label: str, use_nn_loc: bool, eps: float = 1e-4) -> float:
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


def prepare_state(inputs_dir: Path, use_nn_loc: bool, num_cores: int) -> TrainingState:
    ham, system, lr_params, device, atom_pp_order = build_hamiltonian(inputs_dir, num_cores)
    model = nn.Linear(1, len(lr_params), bias=False).to(device).double()
    ham.set_NNmodel(model)
    ham.NN_locbool = use_nn_loc

    lr_value = 0.05
    if 'longRange_lr' in ham.NNConfig:
        lr_value = ham.NNConfig['longRange_lr']

    optimizer = torch.optim.SGD([
        {'params': model.parameters(), 'lr': 0.0},
        {'params': lr_params.parameters(), 'lr': lr_value},
    ])

    atom_label = str(atom_pp_order[0])
    return TrainingState(ham, system, lr_params, optimizer, model, atom_label, use_nn_loc, num_cores)


def run_serial_epoch(state: TrainingState) -> EpochResult:
    state.ham.NN_locbool = state.use_nn_loc
    state.ham.set_NNmodel(state.model)
    state.ham.NNConfig['num_cores'] = 0

    state.model.train()
    state.optimizer.zero_grad()

    lr_tensor = state.lr_params[state.atom_label]
    prev_value = lr_tensor.detach().clone().item()

    loss = weighted_loss(state.ham, state.system, requires_grad=True)
    loss.backward()

    if lr_tensor.grad is None:
        raise RuntimeError('Expected gradient for long-range parameter in serial epoch.')
    grad_value = lr_tensor.grad.detach().clone().item()

    fd_grad = finite_difference(state.ham, state.system, state.lr_params, state.atom_label, state.use_nn_loc)

    state.ham.enforce_lr_constraint()
    state.optimizer.step()

    lr_value = lr_tensor.detach().item()
    lr_rate = state.optimizer.param_groups[1]['lr']
    expected_update = prev_value - lr_rate * grad_value
    if abs(lr_value - expected_update) > 5e-8:
        raise AssertionError(
            f'Serial LR parameter update mismatch. expected {expected_update:.6e}, got {lr_value:.6e}'
        )

    return EpochResult(prev_value=prev_value, value=lr_value, grad=grad_value, finite_diff=fd_grad)


def run_parallel_epoch(state: TrainingState) -> EpochResult:
    if state.num_cores < 2:
        raise ValueError('Parallel epoch requires at least 2 cores.')

    state.ham.NN_locbool = state.use_nn_loc
    state.ham.set_NNmodel(state.model)
    state.ham.NNConfig['num_cores'] = state.num_cores

    state.model.train()
    state.optimizer.zero_grad()

    lr_tensor = state.lr_params[state.atom_label]
    prev_value = lr_tensor.detach().clone().item()

    args = [
        (kidx, state.ham, state.system, state.optimizer, state.model, None, None)
        for kidx in range(state.system.getNKpts())
    ]
    with mp.Pool(state.num_cores) as pool:
        results = pool.starmap(calcEigValsAtK_wGrad_parallel, args)

    combined_gradients = {}
    for grad_dict, *_ in results:
        for key, value in grad_dict.items():
            if key not in combined_gradients:
                combined_gradients[key] = value.detach().clone()
            else:
                combined_gradients[key] += value.detach().clone()

    state.optimizer.zero_grad()
    with torch.no_grad():
        for name, param in state.model.named_parameters():
            if name in combined_gradients:
                param.grad = combined_gradients[name].detach().clone()
            else:
                param.grad = None
        for lr_name, lr_param in state.lr_params.items():
            key = f'lr_params.{lr_name}'
            if key in combined_gradients:
                lr_param.grad = combined_gradients[key].detach().clone()
            else:
                lr_param.grad = None

    if lr_tensor.grad is None:
        raise RuntimeError('Expected gradient for long-range parameter in parallel epoch.')
    grad_value = lr_tensor.grad.detach().clone().item()

    fd_grad = finite_difference(state.ham, state.system, state.lr_params, state.atom_label, state.use_nn_loc)

    state.ham.enforce_lr_constraint()
    state.optimizer.step()

    lr_value = lr_tensor.detach().item()
    lr_rate = state.optimizer.param_groups[1]['lr']
    expected_update = prev_value - lr_rate * grad_value
    if abs(lr_value - expected_update) > 5e-8:
        raise AssertionError(
            f'Parallel LR parameter update mismatch. expected {expected_update:.6e}, got {lr_value:.6e}'
        )

    return EpochResult(prev_value=prev_value, value=lr_value, grad=grad_value, finite_diff=fd_grad)


def main():
    inputs_dir = Path(__file__).resolve().parent / 'inputs'
    modes = [
        (False, 'Zunger local + LR tail'),
        (True, 'NN local + LR tail'),
    ]

    for use_nn_loc, mode_label in modes:
        serial_state = prepare_state(inputs_dir, use_nn_loc, num_cores=0)
        parallel_state = prepare_state(inputs_dir, use_nn_loc, num_cores=2)

        print(f'Long-range training consistency check: {mode_label}')
        for epoch in range(3):
            serial_start = serial_state.lr_params[serial_state.atom_label].detach().item()
            parallel_start = parallel_state.lr_params[parallel_state.atom_label].detach().item()

            serial_epoch = run_serial_epoch(serial_state)
            parallel_epoch = run_parallel_epoch(parallel_state)

            if abs(serial_start - serial_epoch.prev_value) > 1e-12:
                raise AssertionError('Serial state did not carry LR parameter into epoch correctly.')
            if abs(parallel_start - parallel_epoch.prev_value) > 1e-12:
                raise AssertionError('Parallel state did not carry LR parameter into epoch correctly.')

            value_diff = abs(serial_epoch.value - parallel_epoch.value)
            grad_diff = abs(serial_epoch.grad - parallel_epoch.grad)
            if value_diff > 5e-6:
                raise AssertionError(
                    f'Epoch {epoch+1}: LR parameter mismatch ({mode_label}). diff={value_diff:.3e}'
                )
            if grad_diff > 5e-6:
                raise AssertionError(
                    f'Epoch {epoch+1}: LR gradient mismatch ({mode_label}). diff={grad_diff:.3e}'
                )

            if abs(serial_epoch.grad - serial_epoch.finite_diff) > 5e-5:
                raise AssertionError(
                    f'Epoch {epoch+1}: Serial gradient deviates from finite difference ({mode_label}). '
                    f'|serial-grad - fd| = {abs(serial_epoch.grad - serial_epoch.finite_diff):.3e}'
                )
            if abs(parallel_epoch.grad - parallel_epoch.finite_diff) > 5e-5:
                raise AssertionError(
                    f'Epoch {epoch+1}: Parallel gradient deviates from finite difference ({mode_label}). '
                    f'|parallel-grad - fd| = {abs(parallel_epoch.grad - parallel_epoch.finite_diff):.3e}'
                )

            serial_state_value = serial_state.lr_params[serial_state.atom_label].detach().item()
            parallel_state_value = parallel_state.lr_params[parallel_state.atom_label].detach().item()
            if abs(serial_state_value - serial_epoch.value) > 1e-12:
                raise AssertionError('Serial state LR parameter not retained after epoch.')
            if abs(parallel_state_value - parallel_epoch.value) > 1e-12:
                raise AssertionError('Parallel state LR parameter not retained after epoch.')

            print(
                f"  Epoch {epoch+1}: value={serial_epoch.value:.6e}, serial_grad={serial_epoch.grad:.6e}, "
                f"parallel_grad={parallel_epoch.grad:.6e}, fd={serial_epoch.finite_diff:.6e}"
            )
        print()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
