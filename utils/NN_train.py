import torch
import time, os
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import gc
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.rcParams['lines.markersize'] = 3
import copy
import random
import shutil
import os

torch.set_default_dtype(torch.float64)

from .constants import *
from .pp_func import plotPP, plotPP_spin, plotLSD, plot_training_validation_cost, plotBandStruct, plot_mc_cost, plotBandStruct_reorder
from .smooth_order import reorder_smoothness_deg2_tensors, reorder_kpt_smoothness_deg2_tensors
from .profiling import PROF
from .threads import pool_worker_init


# ---------------------------------------------------------------------------
# Gradient-based optimization of the spin-orbit / non-local prefactors.
#
# The SO and NL contributions to H are  sum_alpha (cached_projector_matrix) *
# PPparams[atom][idx], where the cached projector matrices are fixed constants
# (built once, detached) and only the scalar prefactor is variable. So the
# prefactors can be trained by gradient descent at essentially no extra cost:
# they ride the same eigvalsh backward already paid for by the local NN.
#
# All Hamiltonians share ONE PPparams dict (the same object is handed to every
# Hamiltonian in initAndCacheHams), so flipping requires_grad on each atom's
# parameter tensor once is enough for every system. We only want to move a
# subset of indices (default [5,6,7] = SOC, NL1, NL2), so after backward we zero
# the gradient on all other indices before stepping; because those indices carry
# exactly-zero grad from the very first step, Adam never accumulates momentum on
# them and they stay frozen.
# ---------------------------------------------------------------------------
def setup_nonlocal_grad(hams, atomPPOrder, NNConfig):
    """Enable gradient training of the SOC/NL prefactors in ham.PPparams.

    Returns a context dict {optimizer, scheduler, params, indices} or None if
    'nonlocal_grad' is not enabled in NNConfig.
    """
    if not NNConfig.get('nonlocal_grad', False):
        return None

    train_indices = NNConfig.get('nonlocal_grad_indices', [5, 6, 7])
    PPparams = hams[0].PPparams  # shared across all hams

    nl_params = {}
    for atom in dict.fromkeys(atomPPOrder):   # dedup, preserve order
        if atom not in PPparams:
            continue
        p = PPparams[atom]
        p.requires_grad_(True)
        nl_params[atom] = p

    if not nl_params:
        print("WARNING: nonlocal_grad is ON but no PPparams were found to train. Disabling.")
        return None

    lr = NNConfig.get('nonlocal_grad_lr', NNConfig['optimizer_lr'])
    nl_optimizer = torch.optim.Adam(list(nl_params.values()), lr=lr)
    nl_scheduler = ExponentialLR(nl_optimizer, gamma=NNConfig.get('nonlocal_grad_scheduler_gamma', NNConfig['scheduler_gamma']))

    print(f"\nNon-local/SOC gradient training ON. Optimizing PPparams indices {train_indices} "
          f"for atoms {[str(a) for a in nl_params]} at lr={lr}.")
    for atom, p in nl_params.items():
        vals = p.detach()
        print(f"    initial {atom} PPparams[{train_indices}] = {[round(float(vals[i]), 6) for i in train_indices]}")

    return {'optimizer': nl_optimizer, 'scheduler': nl_scheduler,
            'params': nl_params, 'indices': train_indices}


def _zero_nonlocal_grad(nl_ctx):
    if nl_ctx is not None:
        nl_ctx['optimizer'].zero_grad()


def _step_nonlocal_grad(nl_ctx):
    """Mask gradients to the trained indices, then take an optimizer step."""
    if nl_ctx is None:
        return
    train_indices = nl_ctx['indices']
    with torch.no_grad():
        for atom, p in nl_ctx['params'].items():
            if p.grad is None:
                continue
            keep = torch.zeros_like(p.grad)
            keep[train_indices] = p.grad[train_indices]
            p.grad.copy_(keep)
    nl_ctx['optimizer'].step()


def write_nonlocal_params(filename, nl_ctx, atom=None):
    """Dump the current full 9-entry PPparams.

    If `atom` is given, write ONLY that atom's params (one file per atom type);
    otherwise write every trained atom into the one file (legacy behavior, used
    for the aggregated final_nonlocalParams.dat dump).
    """
    if nl_ctx is None:
        return
    if atom is None:
        items = list(nl_ctx['params'].items())
    else:
        items = [(atom, nl_ctx['params'][atom])]
    with open(filename, 'w') as f:
        for _atom, p in items:
            vals = p.detach()
            for i in range(vals.shape[0]):
                f.write(f"{float(vals[i]):.8f}\n")


def print_and_inspect_gradients(model, filename=None, show=False):
    """
    Prints and/or saves the gradients of the model parameters.

    If 'filename' is provided and 'show' is True, it saves the gradients to the file.
    If 'filename' is None and 'show' is True, it prints the gradients.
    """
    if (filename is None) and show: 
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f'Parameter: {name}, Gradient shape: {param.grad.shape}')
                print(f'Gradient values:\n{param.grad}\n')
            else:
                print(f'Parameter: {name}, Gradient: None (no gradient computed)\n')
    elif (filename is not None) and show: 
        with open(filename, 'w') as f:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    f.write(f'Parameter: {name}, Gradient shape: {param.grad.shape}\n')
                    grad_str = np.array2string(param.grad.detach().cpu().numpy(), precision=5, suppress_small=True, max_line_width=999999, threshold=99*99)
                    f.write(f'Gradient values:\n{grad_str}\n\n')
                else:
                    f.write(f'Parameter: {name}, Gradient: None (no gradient computed)\n\n')    


def print_and_inspect_NNParams(model, filename=None, show=False): 
    """
    Prints and/or saves the values of the model parameters.

    If 'filename' is provided and 'show' is True, it saves the parameters to the file.
    If 'filename' is None and 'show' is True, it prints the parameters.
    """
    if (filename is None) and show: 
        for name, param in model.named_parameters():
            print(f'Parameter: {name}, Tensor shape: {param.shape}')
            print(f'Parameter values:\n{param}\n')
    elif (filename is not None) and show: 
        with open(filename, 'w') as f:
            for name, param in model.named_parameters():
                f.write(f'Parameter: {name}, Tensor shape: {param.shape}\n')
                tensor_str = np.array2string(param.detach().cpu().numpy(), precision=5, suppress_small=True, max_line_width=999999, threshold=99*99)
                f.write(f'Parameter values:\n{tensor_str}\n\n')


def write_PP_qSpace(writeFileName, model, atomPPOrder, qmax=40.0, nQGrid=4096):
    # q grid must match FT_converge_and_write_pp's choice grid (choiceQMax,
    # choiceNQGrid) so qSpace_pot.dat and final_pot_q_*.dat share one grid.
    qGrid = torch.linspace(0.0, qmax, int(nQGrid)).view(-1, 1)
    NN = model(qGrid)

    # write out
    with open(writeFileName, 'w') as file: 
        file.write("# q          ")
        for iAtom in range(len(atomPPOrder)): 
            file.write(f"v(q)_{atomPPOrder[iAtom]}          ")
        file.write("\n")

        for i in range(len(qGrid)):
            file.write(f"{qGrid[i,0]:.8f}          ")
            for iAtom in range(len(atomPPOrder)): 
                file.write(f"{NN[i,iAtom]:.8f}          ")
            file.write("\n")
    return

def write_PP_qSpace_spin(writeFileName, model, spinModel, atomPPOrder, qmax=40.0, nQGrid=4096):
    """
    Write the spin-resolved local pseudopotentials in q-space for a
    spin-polarized (tot_magnetization != 0) run. For each atom type we dump:
        V0(q)    : the spin-independent local potential (model)
        b(q)     : the learned spin/exchange field (spinModel)
        V_up(q)  = V0(q) + b(q)
        V_dn(q)  = V0(q) - b(q)
    The q grid matches write_PP_qSpace so columns line up across files.
    """
    qGrid = torch.linspace(0.0, qmax, int(nQGrid)).view(-1, 1)
    V0 = model(qGrid)
    b = spinModel(qGrid)
    Vup = V0 + b
    Vdn = V0 - b

    with open(writeFileName, 'w') as file:
        file.write("# q          ")
        for iAtom in range(len(atomPPOrder)):
            a = atomPPOrder[iAtom]
            file.write(f"V0(q)_{a}          b(q)_{a}          Vup(q)_{a}          Vdn(q)_{a}          ")
        file.write("\n")

        for i in range(len(qGrid)):
            file.write(f"{qGrid[i,0]:.8f}          ")
            for iAtom in range(len(atomPPOrder)):
                file.write(f"{V0[i,iAtom]:.8f}          {b[i,iAtom]:.8f}          "
                           f"{Vup[i,iAtom]:.8f}          {Vdn[i,iAtom]:.8f}          ")
            file.write("\n")
    return


def write_LSD_qSpace(writeFileName, LSDmodel, N_alpha):
    qGrid = torch.linspace(0.0, 30.0, 4096).view(-1, 1)
    N_alphas = N_alpha * torch.ones_like(qGrid)
    x_inputs = torch.cat((N_alphas, qGrid), dim=1)
    NN = LSDmodel(x_inputs)     

    output = np.concatenate(
        (qGrid.detach().numpy().reshape(-1,1), 
         NN.detach().numpy().reshape(-1,1)), axis=1)
    
    # write out
    np.savetxt(writeFileName, output, fmt="%8f", header=f"{N_alpha}")
    return


def get_max_gradient_param(model):
    """
    Returns the parameter that has the largest gradient, in terms of the 
    parameter tensor's name in the dictionary, the index within this tensor, 
    and the value of the gradient. 

    Later, one can access this parameter using: 
    dict(model.named_parameters())[max_grad_name].grad[max_grad_index]
    """
    gradients_populated = any(param.grad is not None for param in model.parameters())
    if not gradients_populated:
        raise ValueError("Gradients have not been populated. Ensure that a backward pass has been performed.")

    max_grad = None
    max_grad_index = None
    max_grad_name = None

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_abs_max_value = param.grad.abs().max().item()
            if max_grad is None or grad_abs_max_value > max_grad:
                max_grad = grad_abs_max_value
                max_grad_name = name
                max_grad_index = param.grad.abs().argmax().item()

    if max_grad_name is not None:
        param = dict(model.named_parameters())[max_grad_name]
        max_grad_value = param.grad.flatten()[max_grad_index].clone()
        max_grad_index = np.unravel_index(max_grad_index, param.grad.shape)
        # print(f"Values returned by the get_max_gradient_param function: {max_grad_name}, {max_grad_index}, {max_grad_value}")
        return max_grad_name, max_grad_index, max_grad_value
    else:
        return None, None, None


def judge_well_conditioned_grad(model, maxGradThreshold=50.0): 
    maxGrad = None
    minGrad = None
    for _, param in model.named_parameters():
        if param.grad is not None:
            if maxGrad is None or param.grad.abs().max().item() > maxGrad:
                maxGrad = param.grad.abs().max().item()
            if minGrad is None or param.grad.abs().min().item() > minGrad:
                minGrad = param.grad.abs().min().item()
    print(f"Max and min of absolute gradients = {maxGrad:.3f}, {minGrad:.3f}.   Are the gradients well-conditioned? {maxGrad<=maxGradThreshold}")
    return maxGrad, minGrad


def manual_GD_one_param(model, stepSize=None):
    """
    Make a manual gradient descent move on ONLY ONE parameter that has the largest
    absolute gradient value. This is designed to slowly yet surely optimize to the
    nearest local minimum on a multi-dimensional function space. The model is 
    changed in-place. 

    One can give an optional stepSize parameter. If not used, the manual 
    optimization steps (lr * grad) is hard-coded to be around 0.005
    """
    max_grad_name, max_grad_index, max_grad_value = get_max_gradient_param(model)
    
    if max_grad_name is None:
        raise ValueError("No maximum gradient found in the model. (Meaning that there were no gradients in the model).")

    # Zero all gradients, except for the one with maximum gradient
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.zero_()
        if name==max_grad_name: 
            param.grad[max_grad_index] = max_grad_value.item()

    # Set the learning rate, ensuring max_grad_value is used appropriately
    if stepSize is None:
        stepSize = 0.005
    learning_rate = stepSize * random.random() / abs(max_grad_value.item())

    # Perform the manual SGD step
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                param -= learning_rate * param.grad


def weighted_mse_bandStruct(bandStruct_hat, bulkSystem): 
    bandWeights = bulkSystem.bandWeights
    kptWeights = bulkSystem.kptWeights
    nkpt = bulkSystem.getNKpts()
    nBands = bulkSystem.nBands
    if (len(bandWeights)!=nBands) or (len(kptWeights)!=nkpt): 
        raise ValueError("bandWeights or kptWeights lengths aren't correct. ")
        
    newBandWeights = bandWeights.view(1, -1).expand(nkpt, -1)
    newKptWeights = kptWeights.view(-1, 1).expand(-1, nBands)
    
    MSE = torch.sum((bandStruct_hat-bulkSystem.expBandStruct)**2 * newBandWeights * newKptWeights)
    return MSE


def weighted_mse_energiesAtKpt(calcEnergiesAtKpt, bulkSystem, kidx): 
    bandWeights = bulkSystem.bandWeights
    nBands = bulkSystem.nBands
    if (len(calcEnergiesAtKpt)!=nBands): 
        raise ValueError("CalculatedEnergiesAtKpt is of different length as nBands. Can't calculated MSE.")

    MSE = torch.sum((calcEnergiesAtKpt-bulkSystem.expBandStruct[kidx])**2 * bandWeights)
    return MSE


def weighted_relative_mse_bandStruct(bandStruct_hat, bulkSystem, relE_bIdx): 
    # The relative energies are calculated with respect to the current kpoint, of the relE_bIdx: 
    # rel_refBS = refBS - refBS[kidx=curr, relE_bIdx]
    # rel_calcBS = calcBS - calcBS[kidx=curr, relE_bIdx]

    bandWeights = bulkSystem.bandWeights
    kptWeights = bulkSystem.kptWeights
    nkpt = bulkSystem.getNKpts()
    nBands = bulkSystem.nBands
    if (len(bandWeights)!=nBands) or (len(kptWeights)!=nkpt): 
        raise ValueError("bandWeights or kptWeights lengths aren't correct. ")
        
    newBandWeights = bandWeights.view(1, -1).expand(nkpt, -1)
    newKptWeights = kptWeights.view(-1, 1).expand(-1, nBands)
    
    rel_refBS = bulkSystem.expBandStruct - bulkSystem.expBandStruct[:, relE_bIdx].unsqueeze(1)
    rel_calcBS = bandStruct_hat - bandStruct_hat[:, relE_bIdx].unsqueeze(1)
    # rel_refBS = bulkSystem.expBandStruct - bulkSystem.expBandStruct[0, relE_bIdx]
    # rel_calcBS = bandStruct_hat - bandStruct_hat[0, relE_bIdx]
    MSE = torch.sum((rel_refBS - rel_calcBS)**2 * newBandWeights * newKptWeights)
    return MSE


def weighted_relative_mse_energiesAtKpt(calcEnergiesAtKpt, bulkSystem, kidx, relE_bIdx): 
    # Same definition as above. 
    # We subtract BS[kidx=curr, relE_bIdx]
    bandWeights = bulkSystem.bandWeights
    nBands = bulkSystem.nBands
    if (len(calcEnergiesAtKpt)!=nBands): 
        raise ValueError("CalculatedEnergiesAtKpt is of different length as nBands. Can't calculated MSE.")

    rel_refEAtKpt = bulkSystem.expBandStruct[kidx] - bulkSystem.expBandStruct[kidx, relE_bIdx]
    rel_calcEAtKpt = calcEnergiesAtKpt - calcEnergiesAtKpt[relE_bIdx]
    MSE = torch.sum((rel_refEAtKpt - rel_calcEAtKpt)**2 * bandWeights)
    return MSE


def penalty_loss(f_x, x, penalize_start=4.5, lambda_penalty=1.0, penalize=True):
    if not penalize:
        return torch.tensor(0.0)

    x_0 = penalize_start + 0.5  # Midpoint of ramp
    k = 10.0   # Sharpness of ramp (higher = steeper transition)

    # Compute the ramp function S(x)
    S_x = 1 / (1 + torch.exp(-k * (x - x_0)))

    # Ensure S_x is broadcastable to f_x
    if S_x.shape != f_x.shape:
        S_x = S_x.expand_as(f_x)  # Expand to match f_x shape if necessary

    # Compute penalty term
    penalty = lambda_penalty * torch.mean(S_x * torch.abs(f_x))

    return penalty

def mag_penalty_loss(f_x, x, f_x_max, lambda_penalty=1.0, penalize=True):
    if (not penalize) or (lambda_penalty <= 0):
        return torch.tensor(0.0)


    dq = x[1] - x[0]
   # 2. Pre-compute integration weights. Shape: [240, 1]

    integration_weights = (x ** 2) * dq * (1.0 / (2.0 * (np.pi ** 2)))


    # 3. Integrate across the Grid dimension (dim=0)
    # Resulting V_r0 shape will be [3] (one value per atom species)
    V_r0 = torch.sum(f_x * integration_weights, dim=0)

    # 4. Get the magnitude of the potential at r=0 for each species
    abs_V_r0 = torch.abs(V_r0)

    # 5. Calculate excess over the threshold for each species. Shape: [3]
    excess = torch.relu(abs_V_r0 - f_x_max)

    # 6. Take the mean of the penalties across the 3 species
    # This reduces it to a single scalar loss value for PyTorch optimizer
    mag_penalty = lambda_penalty * torch.mean(excess)



    return mag_penalty

def _penalty_term(model, NNConfig, nkpt, device):
    """Non-decay penalty on V(q). Zero tensor when not configured / no model."""
    if not (("penalize_starting" in NNConfig) and ("penalize_lambda" in NNConfig) and (model is not None)):
        return torch.tensor(0.0, dtype=torch.float64, device=device)
    q = torch.linspace(NNConfig["penalize_starting"], 12.0, 50, dtype=torch.float64, device=device).view(-1, 1)
    return penalty_loss(model(q), q, NNConfig["penalize_starting"], NNConfig["penalize_lambda"] * nkpt)


def _mag_penalty_term(model, spinModel, NNConfig, nkpt, device):
    """Magnitude penalty on V(q). Spin-aware: with a spinModel it penalizes the
    two spin channels V_up = model + spin and V_dn = model - spin and returns
    their average; without one it penalizes V = model. Zero when not configured."""
    if not (("penalize_mag_threshold" in NNConfig) and ("penalize_mag_lambda" in NNConfig)
            and (NNConfig["penalize_mag_lambda"] > 0) and (model is not None)):
        return torch.tensor(0.0, dtype=torch.float64, device=device)
    q = torch.linspace(0.0, 12.0, 240, dtype=torch.float64, device=device).view(-1, 1)
    thresh = NNConfig["penalize_mag_threshold"]
    lam = NNConfig["penalize_mag_lambda"] * nkpt
    if spinModel is not None:
        up = mag_penalty_loss(model(q) + spinModel(q), q, thresh, lam)
        dn = mag_penalty_loss(model(q) - spinModel(q), q, thresh, lam)
        return 0.5 * (up + dn)
    return mag_penalty_loss(model(q), q, thresh, lam)


def _defpot_term(ham, bulkSystem, cachedMats_info, requires_grad, device):
    """Deformation-potential MSE loss. Zero tensor when the system isn't fitting
    deformation potentials. DefPots are global transition observables, so this is
    NOT scaled by k-point weights or nkpt."""
    if not bulkSystem.fit_defPot:
        return torch.tensor(0.0, dtype=torch.float64, device=device)
    calcDefPots = ham.calcDefPots(cachedMats_info=cachedMats_info, requires_grad=requires_grad, verbosity=0)
    refDefPots = torch.tensor(bulkSystem.defPotInfo[:, 5], dtype=torch.float64, device=calcDefPots.device)
    defPotWeights = torch.tensor(bulkSystem.defPotInfo[:, 6], dtype=torch.float64, device=calcDefPots.device)
    loss = ((calcDefPots - refDefPots) ** 2 * defPotWeights).sum()
    print(f"Calculated defPots = {calcDefPots}, refDefPots = {refDefPots}, defPotLoss = {loss:.4f}")
    return loss


# ---------------------------------------------------------------------------
# Unified loss function -- the SINGLE source of truth.
#
# The training/validation loss is the sum of a per-k-point band-structure MSE
# and a set of GLOBAL (whole-system, k-INDEPENDENT) losses: a non-decay
# penalty, a spin-aware magnitude penalty, deformation potentials, e-ph
# couplings, and effective masses. EVERY path -- train_naive,
# trainIter_separateKptGrad (serial + multiprocessing) and the no-grad
# evaluation path evalBS_noGrad -- computes those global terms through the ONE
# function compute_global_system_losses() below, so the loss that trains the NN
# is identical to the loss that is reported and plotted. Previously each path
# recomputed these terms inline and inconsistently (couplings/eff-masses were
# silently dropped by separateKptGrad), so newly added loss terms did not
# actually update the NN parameters on every route.
#
# The band-structure term is the only per-k-point piece; the global terms are
# computed ONCE per system. Adding a new global loss term now means adding ONE
# _xxx_term helper, ONE key in compute_global_system_losses(), and ONE entry in
# LOSS_TERM_NAMES -- every path and the reporting then pick it up automatically.
# ---------------------------------------------------------------------------

# Canonical, ordered list of the loss components that are tracked, printed, and
# plotted. "bandStruct" is the per-k-point term the caller adds; the rest are
# the global terms returned by compute_global_system_losses().
LOSS_TERM_NAMES = ["bandStruct", "penalty", "mag_penalty", "defpot", "coupling", "effmass"]


def new_loss_components():
    """A fresh {term_name: 0.0} accumulator over LOSS_TERM_NAMES."""
    return {name: 0.0 for name in LOSS_TERM_NAMES}


def accumulate_loss_components(target, source):
    """Add the (tensor or float) values of `source` into the `target` float
    accumulator, detaching tensors. Keys not in LOSS_TERM_NAMES are ignored."""
    for name in LOSS_TERM_NAMES:
        if name in source:
            val = source[name]
            target[name] += float(val.detach()) if torch.is_tensor(val) else float(val)


def loss_components_total(loss_components):
    """Total loss = sum of all tracked components."""
    return sum(loss_components.get(name, 0.0) for name in LOSS_TERM_NAMES)


def bandStruct_loss(bandStruct_hat, bulkSystem):
    """Whole-band-structure MSE term (relative-to-reference-band if configured)."""
    if bulkSystem.relE_bIdx != -1:
        return weighted_relative_mse_bandStruct(bandStruct_hat, bulkSystem, bulkSystem.relE_bIdx)
    return weighted_mse_bandStruct(bandStruct_hat, bulkSystem)


def bandStruct_kpt_loss(calcEnergiesAtKpt, bulkSystem, kidx):
    """Single-k-point band-structure MSE term (sums over k to bandStruct_loss)."""
    if bulkSystem.relE_bIdx != -1:
        return weighted_relative_mse_energiesAtKpt(calcEnergiesAtKpt, bulkSystem, kidx, bulkSystem.relE_bIdx)
    return weighted_mse_energiesAtKpt(calcEnergiesAtKpt, bulkSystem, kidx)


def _coupling_term(ham, bulkSystem, device, requires_grad=True, coupling_debug=False):
    """e-ph coupling MSE loss. Zero tensor when the system isn't fitting couplings.
    Returns (loss, calcCouplings_dict); calcCouplings_dict is None when not fit.
    Scaled by nkpt so it balances against the k-point-summed band-structure term
    (this matches the historical inline definition exactly). Under requires_grad
    the differentiable calcCouplings() graph is kept alive (the LSD correction is
    trained by differentiating this term)."""
    if not bulkSystem.fit_eph:
        return torch.tensor(0.0, dtype=torch.float64, device=device), None
    loss = torch.tensor(0.0, dtype=torch.float64, device=device)
    grad_ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with grad_ctx:
        calcCouplings_dict = ham.calcCouplings()
        if coupling_debug:
            print(calcCouplings_dict)
        for atomidx in range(bulkSystem.getNAtoms()):
            for gamma in range(3):
                for qidx in range(bulkSystem.qpts.shape[0]):
                    for band in ["vb", "cb"]:
                        key = (atomidx, gamma, qidx, band)
                        if (key in calcCouplings_dict) and (key in bulkSystem.expCouplingBands):
                            cpl_weight = (bulkSystem.expCouplingWeights.get(key, 1.0)
                                          if bulkSystem.expCouplingWeights is not None else 1.0)
                            loss = loss + ((abs(calcCouplings_dict[key]) - abs(bulkSystem.expCouplingBands[key])) ** 2
                                           * bulkSystem.qptWeights[qidx] * cpl_weight) * bulkSystem.getNKpts()
                        else:
                            print(f"WARNING: The coupling key {key} is missing in either the calculated "
                                  f"or reference couplings. Skipping this entry in calculating the loss. ")
    return loss, calcCouplings_dict


def _effmass_term(ham, bulkSystem, cachedMats_info, requires_grad, device, bandStruct=None):
    """Effective-mass MSE loss. Zero tensor when the system isn't fitting eff. masses.
    Returns (loss, eff_masses). When a grad-carrying `bandStruct` is supplied
    (train_naive / eval already have the full band structure) it is reused;
    otherwise (separateKptGrad, whose per-k band graphs are freed) the two
    k-points calcEffMasses needs are re-evaluated here with a FRESH graph, so the
    term is differentiable in every path (including multiprocessing, where this
    runs in the parent)."""
    if not bulkSystem.fit_eff_masses:
        return torch.tensor(0.0, dtype=torch.float64, device=device), None
    if bandStruct is None:
        nBands = bulkSystem.nBands
        nkpt = bulkSystem.getNKpts()
        bandStruct = torch.zeros([nkpt, nBands], dtype=torch.float64, device=device)
        for kidx in (ham.idx_gap, ham.idx_gap - 1):
            bandStruct[kidx, :] = ham.calcEigValsAtK(kidx, cachedMats_info, requires_grad=requires_grad)
    eff_masses = ham.calcEffMasses(bandStruct)
    loss = bulkSystem.effMassWeight * ((eff_masses[0] - bulkSystem.expEffMasses[0]) ** 2
                                       + (eff_masses[1] - bulkSystem.expEffMasses[1]) ** 2)
    return loss, eff_masses


def compute_global_system_losses(model, spinModel, ham, bulkSystem, cachedMats_info=None,
                                  requires_grad=True, coupling_debug=False, bandStruct=None):
    """The single source of truth for the GLOBAL (whole-system, k-INDEPENDENT)
    loss terms of one system. Returns (loss_terms, extras) where:
      * loss_terms is a dict of grad-carrying scalar tensors keyed by the global
        entries of LOSS_TERM_NAMES (penalty, mag_penalty, defpot, coupling,
        effmass). The caller adds the per-k-point "bandStruct" term separately.
      * extras carries the by-products used for file output:
        {'calcCouplings': dict|None, 'eff_masses': tensor|None}.
    """
    device = next(model.parameters()).device if model is not None else bulkSystem.kpts.device
    nkpt = bulkSystem.getNKpts()

    coupling_loss, calcCouplings_dict = _coupling_term(
        ham, bulkSystem, device, requires_grad=requires_grad, coupling_debug=coupling_debug)
    effmass_loss, eff_masses = _effmass_term(
        ham, bulkSystem, cachedMats_info, requires_grad, device, bandStruct=bandStruct)

    loss_terms = {
        "penalty": _penalty_term(model, ham.NNConfig, nkpt, device),
        "mag_penalty": _mag_penalty_term(model, spinModel, ham.NNConfig, nkpt, device),
        "defpot": _defpot_term(ham, bulkSystem, cachedMats_info, requires_grad, device),
        "coupling": coupling_loss,
        "effmass": effmass_loss,
    }
    extras = {"calcCouplings": calcCouplings_dict, "eff_masses": eff_masses}
    return loss_terms, extras


def global_loss_sum(loss_terms):
    """Sum the global loss-term tensors into one scalar tensor ready for backward.
    Returns None if there is nothing to sum (all terms absent)."""
    total = None
    for name, val in loss_terms.items():
        total = val if total is None else total + val
    return total


def write_coupling_bands(filename, bulkSystem, calcCouplings_dict):
    """Write the vb-vb / cb-cb e-ph coupling matrix elements to a .dat file.
    No-op when there are no couplings (calcCouplings_dict is None)."""
    if calcCouplings_dict is None:
        return
    pol = {0: "x", 1: "y", 2: "z"}
    with open(filename, 'w') as fwrite:
        for atomidx in range(bulkSystem.getNAtoms()):
            print(f"Atom idx = {atomidx}   atom = {bulkSystem.atomTypes[atomidx]}   position = {bulkSystem.atomPos[atomidx]}", file=fwrite)
            for band in ["vb", "cb"]:
                print(f"{band}-{band} coupling elements. ", file=fwrite, end="")
                for gamma in range(3):
                    print(f"\npolarization of derivative = {pol[gamma]}", file=fwrite)
                    for qidx in range(bulkSystem.qpts.shape[0]):
                        key = (atomidx, gamma, qidx, band)
                        if key in calcCouplings_dict:
                            val = calcCouplings_dict[key]
                            val_item = val.item() if torch.is_tensor(val) else val
                            if abs(val_item) < 1e-9:
                                print("0   ", file=fwrite, end="")
                            else:
                                print(f"{val_item:.5e}   ", file=fwrite, end="")
                        else:
                            print("Not-fit   ", file=fwrite, end="")
                    print("\n", file=fwrite, end="")
                print("\n", file=fwrite, end="")
            print("\n\n", file=fwrite, end="")


# Header and formatter for the per-component cost .dat files. Columns are the
# total followed by each entry of LOSS_TERM_NAMES, so the files are self-
# describing and the breakdown plot / any downstream analysis can read them.
COST_FILE_HEADER = "# epoch    total    " + "    ".join(LOSS_TERM_NAMES) + "\n"


def format_cost_line(epoch, loss_components):
    cols = "    ".join(f"{loss_components.get(name, 0.0):.6e}" for name in LOSS_TERM_NAMES)
    return f"{epoch}    {loss_components_total(loss_components):.6e}    {cols}\n"


def plot_loss_breakdown(train_x, train_history, val_x=None, val_history=None, SHOWPLOTS=False):
    """Plot every non-zero loss component (and the total) vs epoch on a log scale,
    so a run shows which term dominates and how each is trending. train_history /
    val_history are lists of {term: value} dicts aligned with train_x / val_x."""
    colors = {"bandStruct": "tab:blue", "penalty": "tab:orange", "mag_penalty": "tab:green",
              "defpot": "tab:red", "coupling": "tab:purple", "effmass": "tab:brown"}
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    for name in LOSS_TERM_NAMES:
        series = [loss_components.get(name, 0.0) for loss_components in train_history]
        if any(abs(v) > 0 for v in series):
            axs.plot(train_x, series, "-", color=colors.get(name), label=f"train {name}")
            if (val_x is not None) and val_history:
                vseries = [loss_components.get(name, 0.0) for loss_components in val_history]
                axs.plot(val_x, vseries, ":", color=colors.get(name), alpha=0.7, label=f"val {name}")
    total_series = [loss_components_total(loss_components) for loss_components in train_history]
    axs.plot(train_x, total_series, "k-", linewidth=2, label="train total")
    if (val_x is not None) and val_history:
        vtotal = [loss_components_total(loss_components) for loss_components in val_history]
        axs.plot(val_x, vtotal, "k:", linewidth=2, label="val total")
    axs.set_yscale('log')
    axs.set(xlabel="Epochs", ylabel="Cost", title="Loss component breakdown")
    axs.legend(frameon=False, fontsize=7, ncol=2)
    axs.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    if SHOWPLOTS:
        plt.show()
    return fig

def evalBS_noGrad(model, BSplotFilename, runName, NNConfig, hams, systems, cachedMats_info=None, writeBS=False, LSDmodels=None, resultsFolder="", spinModel=None, loss_components_out=None):
    """No-grad evaluation of the band structures and the FULL loss. Uses the same
    single-source-of-truth loss as training (band-structure MSE + the global
    terms from compute_global_system_losses), so the returned scalar is directly
    comparable to the training loss. If `loss_components_out` (a dict) is passed,
    it is filled in place with the per-component breakdown over LOSS_TERM_NAMES."""
    if (model is not None):
        print(f"\t{runName}: Evaluating band structures using the NN-pp model. ")
        model.eval()
    else:
        print(f"\t{runName}: Evaluating band structures using the old Zunger function form. ")

    if LSDmodels:
        print(f"\t{runName}: Band structures will be corrected with LSD NN potential.")
        for key in LSDmodels:
            LSDmodels[key].eval()

    if spinModel is not None:
        print(f"\t{runName}: Spin-polarized local potential (up/down feel different potentials).")
        spinModel.eval()

    plot_bandStruct_list = []
    loss_components = new_loss_components()
    for iSys, sys in enumerate(systems):
        if (model is not None):
            hams[iSys].NN_locbool = True
            hams[iSys].set_NNmodel(model)
        else:
            hams[iSys].NN_locbool = False

        if (LSDmodels is not None):
            hams[iSys].set_LSDmodels(LSDmodels)

        if (spinModel is not None):
            hams[iSys].set_spinModel(spinModel)

        start_time = time.time()
        with torch.no_grad():
            evalBS = hams[iSys].calcBandStruct_noGrad(cachedMats_info)
        evalBS.detach_()
        end_time = time.time()
        if writeBS:
            if (not BSplotFilename.endswith('_plotBS.pdf')) and (not BSplotFilename.endswith('_plotBS.png')):
                raise ValueError("BSplotFilename must end with '_plotBS.pdf' or '_plotBS.png' to write BS.dat files. ")
            else:
                write_BS_filename = BSplotFilename.replace('_plotBS.pdf', f'_BS_sys{iSys}.dat')
            kptDistInputs_vertical = sys.kptDistInputs.view(-1, 1)
            write_tensor = torch.cat((kptDistInputs_vertical, evalBS), dim=1)
            np.savetxt(write_BS_filename, write_tensor, fmt='%.5f')
            if sys.relE_bIdx != -1:
                shutil.copy(write_BS_filename, write_BS_filename.replace(f'_BS_sys{iSys}.dat', f'_BS_sys{iSys}_trueE.dat'))
                write_tensor_shifted = torch.cat((kptDistInputs_vertical, evalBS - evalBS[:, sys.relE_bIdx].unsqueeze(1) + sys.expBandStruct[:, sys.relE_bIdx].unsqueeze(1)), dim=1)
                np.savetxt(BSplotFilename.replace('_plotBS.pdf', f'_BS_sys{iSys}_relative.dat'), write_tensor_shifted, fmt='%.5f')

        plot_bandStruct_list.append(sys.expBandStruct)
        plot_bandStruct_list.append(evalBS)
        # Same per-k-point band-structure MSE term the training paths use.
        loss_components["bandStruct"] += float(bandStruct_loss(evalBS, sys).detach())

        # Global (whole-system) losses through the SAME shared function used by
        # training. requires_grad=False -> no autograd graph is built. Pass the
        # already-computed evalBS so the eff-mass term is not re-diagonalized.
        with torch.no_grad():
            global_loss_terms, extras = compute_global_system_losses(
                model, spinModel, hams[iSys], sys, cachedMats_info=cachedMats_info,
                requires_grad=False, coupling_debug=True, bandStruct=evalBS)
        accumulate_loss_components(loss_components, global_loss_terms)

        # ----- file output side-effects (loss already tallied above) -----
        if sys.fit_eph:
            write_coupling_bands(os.path.join(resultsFolder, f"{runName}_couplingBands_{iSys}.dat"), sys, extras["calcCouplings"])
            write_coupling_bands(BSplotFilename.replace('_plotBS.pdf', f'_couplingBands_{iSys}.dat'), sys, extras["calcCouplings"])
            print(f"couplingMSE = {float(global_loss_terms['coupling'].detach()):.4g}")

        if sys.fit_eff_masses and extras["eff_masses"] is not None:
            eff_masses = extras["eff_masses"]
            print(f"Calculated effMasses = {eff_masses}, refEffMasses = {sys.expEffMasses}, effMass_Loss = {float(global_loss_terms['effmass'].detach()):.4f}")
            np.savetxt(BSplotFilename.replace('_plotBS.pdf', f'_effMasses_{iSys}.dat'),
                       eff_masses.detach().numpy() if torch.is_tensor(eff_masses) else np.asarray(eff_masses), fmt="%.2f")

        if sys.fit_defPot:
            calcDefPots = hams[iSys].calcDefPots(cachedMats_info=cachedMats_info, requires_grad=False, verbosity=0)
            np.savetxt(BSplotFilename.replace('_plotBS.pdf', f'_defPots_{iSys}.dat'),
                       calcDefPots.detach().numpy(), fmt="%.5f")

        print(f"\t{runName}: Finished evaluating {iSys}-th band structure with no gradient... "
              f"total = {loss_components_total(loss_components):.4f}. BS_MSE = {loss_components['bandStruct']:.4f}. "
              f"Penalty = {loss_components['penalty']:.4f}. mag_penalty = {loss_components['mag_penalty']:.4f}. defPot_MSE = {loss_components['defpot']:.4f}. effMass_MSE = {loss_components['effmass']:.4f}. "
              f"coupling_MSE = {loss_components['coupling']:.4g}.")

    if loss_components_out is not None:
        loss_components_out.clear()
        loss_components_out.update(loss_components)

    total = loss_components_total(loss_components)
    fig = plotBandStruct(systems, plot_bandStruct_list, NNConfig['SHOWPLOTS'])
    print(f"\t{runName}: Finished evaluating all band structures with no gradient... Elapsed time: {(end_time - start_time):.2f} seconds. "
          f"total = {total:.4f}. BS_MSE = {loss_components['bandStruct']:.4f}. Penalty = {loss_components['penalty']:.4f}. defPot_MSE = {loss_components['defpot']:.4f}.")
    fig.suptitle(f"{runName}: total = {total:.4f}. BS_MSE = {loss_components['bandStruct']:.4f}. Penalty = {loss_components['penalty']:.4f}. "
                 f"mag_penalty = {loss_components['mag_penalty']:.4f}. defPot_MSE = {loss_components['defpot']:.4f}. effMass_MSE = {loss_components['effmass']:.4f}. coupling_MSE = {loss_components['coupling']:.4g}.")
    fig.savefig(BSplotFilename)
    fig.savefig(BSplotFilename.replace('.pdf', '.png'))
    plt.close('all')
    torch.cuda.empty_cache()
    # Return a tensor (callers -- runMC_NN, bandStruct_train_GPU -- use .item(),
    # comparisons and np.sqrt on the result).
    return torch.tensor(total, dtype=torch.float64)


def calcEigValsAtK_wGrad_parallel(kidx, ham, bulkSystem, optimizer, model, cachedMats_info=None, prevBS=None, LSDmodels=None, LSDoptimizers=None, spinModel=None, spinOptimizer=None, collect_nl=False):
    """
    loop over kidx
    The rest of the arguments are "constants" / "constant functions" for a single kidx
    For performance, it is recommended that the ham in the argument doesn't have SOmat and NLmat initialized.

    collect_nl: when True, the SOC/NL prefactors in ham.PPparams are autograd
    leaves (set by setup_nonlocal_grad in the parent; the requires_grad flag is
    preserved through pickling). After backward we read ham.PPparams[atom].grad,
    weight it by this kpt's weight, and return it so the parent can accumulate
    and step the dedicated optimizer. The grad is the full 9-vector; the parent
    masks it to the trained indices.
    """
    # This runs in a spawned worker process with its own module-level PROF.
    # Configure + reset it so the timers reflect ONLY this kpt's work; the
    # snapshot is returned at the end and merged by the parent (so the
    # aggregated report includes the build+diagonalize time that happens here).
    PROF.configure(ham.NNConfig.get('runtime_flag', False), ham.NNConfig.get('memory_flag', False))
    PROF.reset()

    singleKptGradients = {}
    singleKptGradients_LSD = {}
    singleKptGradients_spin = {}
    singleKptGradients_nl = {}

    calcEnergies = ham.calcEigValsAtK(kidx, cachedMats_info, requires_grad=True)
    extrapolated_eigVal = calcEnergies.clone()
    if ham.NNConfig['smooth_reorder']: 
        col_ind, calcEnergies, extrapolated_eigVal = reorder_kpt_smoothness_deg2_tensors(calcEnergies, kidx, comparedBS=prevBS.detach() if prevBS is not None else None)

    # Only the per-k-point band-structure MSE term is computed here; the GLOBAL
    # losses are added once per system in the parent (trainIter_separateKptGrad).
    systemKptLoss = bandStruct_kpt_loss(calcEnergies, bulkSystem, kidx)

    optimizer.zero_grad()
    if LSDmodels:
        for key in LSDoptimizers:
            LSDoptimizers[key].zero_grad()
    if spinOptimizer is not None:
        spinOptimizer.zero_grad()
    with PROF.time("backward"):
        systemKptLoss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            if name not in singleKptGradients:
                singleKptGradients[name] = param.grad.detach().clone() * bulkSystem.kptWeights[kidx]
            else:
                singleKptGradients[name] += param.grad.detach().clone() * bulkSystem.kptWeights[kidx]
    trainLoss_systemKpt = systemKptLoss.detach().item() * bulkSystem.kptWeights[kidx]

    if spinModel is not None:
        for name, param in spinModel.named_parameters():
            if param.grad is not None:
                if name not in singleKptGradients_spin:
                    singleKptGradients_spin[name] = param.grad.detach().clone() * bulkSystem.kptWeights[kidx]
                else:
                    singleKptGradients_spin[name] += param.grad.detach().clone() * bulkSystem.kptWeights[kidx]

    if LSDmodels:
        for key in LSDmodels:
            singleKptGradients_LSD[key] = {}
            for name, param in LSDmodels[key].named_parameters():
                if param.grad is not None: 
                    if name not in singleKptGradients_LSD:
                        singleKptGradients_LSD[key][name] = param.grad.detach().clone() * bulkSystem.kptWeights[kidx]
                    else:
                        singleKptGradients_LSD[key][name] += param.grad.detach().clone() * bulkSystem.kptWeights[kidx]
    if collect_nl:
        for atom, p in ham.PPparams.items():
            if getattr(p, 'grad', None) is not None:
                singleKptGradients_nl[atom] = p.grad.detach().clone() * bulkSystem.kptWeights[kidx]

    del systemKptLoss
    gc.collect()

    calcEnergies = calcEnergies.detach()
    extrapolated_eigVal = extrapolated_eigVal.detach()
    return singleKptGradients, trainLoss_systemKpt, calcEnergies, extrapolated_eigVal, singleKptGradients_LSD, singleKptGradients_spin, singleKptGradients_nl, PROF.snapshot()


def trainIter_naive(model, systems, hams, NNConfig, optimizer, cachedMats_info=None, runtime_flag=False, preAdjustBool=False, preAdjustStepSize=None, resultsFolder=None, pre_epoch=0, epoch=0, verbosity=1, LSDmodels=None, LSDoptimizers=None, spinModel=None, spinOptimizer=None, nl_ctx=None):
    trainLoss = torch.tensor(0.0)
    loss_components = new_loss_components()   # per-component loss breakdown for reporting

    for iSys, sys in enumerate(systems):
        hams[iSys].NN_locbool = True
        hams[iSys].set_NNmodel(model)
        if LSDmodels:
            hams[iSys].set_LSDmodels(LSDmodels)
        if spinModel is not None:
            hams[iSys].set_spinModel(spinModel)

        NN_outputs = hams[iSys].calcBandStruct_withGrad(cachedMats_info)

        # reorder NN_outputs if the keyword is turned on
        if hams[iSys].NNConfig['smooth_reorder']:
            order_table, newBS, extrapolated_points = reorder_smoothness_deg2_tensors(NN_outputs)
            NN_outputs = newBS

            # Plot each individual band for debugging
            if verbosity>=1:
                for bandIdx in range(newBS.shape[1]):
                    fig, ax = plotBandStruct_reorder(newBS.detach().numpy(), bandIdx)
                    ax.plot(np.arange(len(newBS)), extrapolated_points[:, bandIdx].detach().numpy(), "gx:", alpha=0.8, markersize=4)
                    ax.set(ylim=(min(extrapolated_points[:, bandIdx])-0.1, max(extrapolated_points[:, bandIdx])+0.1))
                    # plot_highlight_kpt(ax, [0,3,6,13,19,26,34,40,50,60,65,70,79,90,100,108])
                    fig.savefig(f"{resultsFolder}epoch_{epoch+1}_newBand_{bandIdx}.png")
                    fig.savefig(f"{resultsFolder}epoch_{epoch+1}_newBand_{bandIdx}.pdf")
                    plt.close()

        # Band-structure MSE term.
        systemBSLoss = bandStruct_loss(NN_outputs, sys)

        # ALL global (whole-system) losses through the single source of truth:
        # penalty + spin-aware magnitude penalty + deformation potentials +
        # e-ph couplings + effective masses. The already-computed grad-carrying
        # NN_outputs is reused for the eff-mass term (no re-diagonalization).
        loss_terms, extras = compute_global_system_losses(
            model, spinModel, hams[iSys], sys, cachedMats_info, requires_grad=True, bandStruct=NN_outputs)

        trainLoss = trainLoss + systemBSLoss + global_loss_sum(loss_terms)

        # track the breakdown for reporting / plotting
        loss_components["bandStruct"] += float(systemBSLoss.detach())
        accumulate_loss_components(loss_components, loss_terms)

        # file output + progress printouts (loss already tallied above)
        if sys.fit_eph and extras["calcCouplings"] is not None:
            write_coupling_bands(os.path.join(resultsFolder, f"couplingBands_{iSys}.dat"), sys, extras["calcCouplings"])
            print(f"couplingMSE = {float(loss_terms['coupling'].detach()):.4g}")
        if sys.fit_eff_masses and extras["eff_masses"] is not None:
            print(f"Calculated effMasses = {extras['eff_masses']}, refEffMasses = {sys.expEffMasses}, "
                  f"effMass_Loss = {float(loss_terms['effmass'].detach()):.4f}")

    optimizer.zero_grad()
    if LSDmodels:
        for key in LSDoptimizers:
            LSDoptimizers[key].zero_grad()
    if spinOptimizer is not None:
        spinOptimizer.zero_grad()
    _zero_nonlocal_grad(nl_ctx)

    with PROF.time("backward"):
        trainLoss.backward()
    with PROF.time("optimizer_step"):
        if preAdjustBool:
            manual_GD_one_param(model, preAdjustStepSize)
            if LSDmodels:
                for key in LSDmodels:
                    manual_GD_one_param(LSDmodels[key], preAdjustStepSize)
            if spinModel is not None:
                manual_GD_one_param(spinModel, preAdjustStepSize)
        else:
            optimizer.step()
            if LSDmodels:
                for key in LSDoptimizers:
                    LSDoptimizers[key].step()
            if spinOptimizer is not None:
                spinOptimizer.step()
            # SOC/NL prefactors share the same backward as the local NN; just mask
            # to the trained indices and step their dedicated optimizer.
            _step_nonlocal_grad(nl_ctx)

    torch.cuda.empty_cache()
    return model, trainLoss, loss_components


def trainIter_separateKptGrad(model, systems, hams, NNConfig, optimizer, cachedMats_info=None, preAdjustBool=False, preAdjustStepSize=None, resultsFolder=None, pre_epoch=0, epoch=0, verbosity=1, prevBS=None, LSDmodels=None, LSDoptimizers=None, spinModel=None, spinOptimizer=None, nl_ctx=None):
    def merge_dicts(dicts):
        merged_dict = {}
        for d in dicts: # extracts dict from tuple of dicts
            for key in d: # loops over dict keys
                merged_dict[key] = merged_dict.get(key, 0) + d[key] # appends values to dict
        return merged_dict

    def merge_dicts_LSD(kpt_tuple):
        merged_dict = {}
        for kpt_dict in kpt_tuple: # extracts dict from tuple of dicts
            for key in kpt_dict: # loops over atomType keys
                merged_dict[key] = {}
                for nn_key in kpt_dict[key]:
                    merged_dict[key][nn_key] = merged_dict[key].get(nn_key, 0) + kpt_dict[key][nn_key] # appends values to dict
        return merged_dict
    
    trainLoss = 0.0
    loss_components = new_loss_components()   # per-component loss breakdown for reporting
    total_gradients = {}
    total_gradients_LSD = {}
    total_gradients_spin = {}
    # Manually accumulate the kpt-weighted SOC/NL prefactor gradients, mirroring
    # how the NN-model gradients are accumulated below. (The shared PPparams
    # leaves would otherwise accumulate an UN-weighted sum across kpts.) This is
    # used by both the serial (num_cores==0) and multiprocessing paths; in the
    # mp path the per-kpt grads are computed in spawned workers and returned.
    if nl_ctx is not None:
        nl_grad_accum = {atom: torch.zeros_like(p) for atom, p in nl_ctx['params'].items()}
    for iSys, sys in enumerate(systems):
        trainLoss_system = 0.0
        gradients_system = {}
        gradients_system_spin = {}
        hams[iSys].NN_locbool = True
        hams[iSys].set_NNmodel(model)
        if spinModel is not None:
            hams[iSys].set_spinModel(spinModel)

        if LSDmodels:
            gradients_system_LSD = {}
            for key in LSDmodels:
              gradients_system_LSD[key] = {}
              hams[iSys].set_LSDmodels(LSDmodels)

        if (NNConfig['num_cores']==0):   # No multiprocessing
            currBS = torch.zeros([sys.getNKpts(), sys.nBands])
            extrapolated_points = torch.zeros([sys.getNKpts(), sys.nBands])
            # Vloc is k-independent: build it ONCE per epoch and reuse the same
            # grad-carrying tensor for every k-point. Because this path does a
            # SEPARATE backward per k-point (to free each k's eigensolve graph),
            # the shared Vloc subgraph must survive across those backwards, so we
            # pass retain_graph=True to every per-k backward except the last.
            nkpts_sys = sys.getNKpts()
            precomp_Vloc = hams[iSys].buildVlocMat()
            for kidx in range(nkpts_sys):
                calcEnergies = hams[iSys].calcEigValsAtK(kidx, cachedMats_info, requires_grad=True, precomp_Vloc=precomp_Vloc)

                extrapolated_eigVal = calcEnergies.detach().clone()
                if NNConfig['smooth_reorder']: 
                    col_ind, calcEnergies, extrapolated_eigVal = reorder_kpt_smoothness_deg2_tensors(calcEnergies, kidx, comparedBS=prevBS.detach() if prevBS is not None else None)
                    extrapolated_points[kidx,:] = extrapolated_eigVal.detach().clone()

                systemKptLoss = bandStruct_kpt_loss(calcEnergies, sys, kidx)
                currBS[kidx,:] = calcEnergies.detach().clone()

                # NOTE: the GLOBAL losses (penalty, magnitude penalty, deformation
                # potentials, couplings, effective masses) are NOT added here. They
                # are computed once per system after this k-point loop through the
                # shared compute_global_system_losses(), so they apply identically
                # in the serial and multiprocessing paths (and match train_naive).

                optimizer.zero_grad()
                if LSDmodels:
                    for key in LSDmodels:
                        LSDoptimizers[key].zero_grad()
                if spinModel is not None:
                    spinOptimizer.zero_grad()
                _zero_nonlocal_grad(nl_ctx)
                with PROF.time("backward"):
                    # Keep the shared (k-independent) Vloc subgraph alive for the
                    # remaining k-points; free it on the last one. When fitting e-ph
                    # couplings, the coupling term (computed after this loop) reuses
                    # the per-k-point eigenVECTORS, so keep every k-point's graph
                    # alive until that final global backward frees them.
                    systemKptLoss.backward(retain_graph=(kidx < nkpts_sys - 1) or sys.fit_eph)

                loss_components["bandStruct"] += float(systemKptLoss.detach()) * float(sys.kptWeights[kidx])

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if name not in gradients_system:
                            gradients_system[name] = param.grad.detach().clone() * sys.kptWeights[kidx]
                        else:
                            gradients_system[name] += param.grad.detach().clone() * sys.kptWeights[kidx]

                if spinModel is not None:
                    for name, param in spinModel.named_parameters():
                        if param.grad is not None:
                            if name not in gradients_system_spin:
                                gradients_system_spin[name] = param.grad.detach().clone() * sys.kptWeights[kidx]
                            else:
                                gradients_system_spin[name] += param.grad.detach().clone() * sys.kptWeights[kidx]

                if LSDmodels:
                    for key in LSDmodels:
                        for name, param in LSDmodels[key].named_parameters():
                            if param.grad is not None:
                                if name not in gradients_system_LSD[key]:
                                    gradients_system_LSD[key][name] = param.grad.detach().clone() * sys.kptWeights[kidx]
                                else: 
                                    gradients_system_LSD[key][name] += param.grad.detach().clone() * sys.kptWeights[kidx]

                if nl_ctx is not None:
                    for atom, p in nl_ctx['params'].items():
                        if p.grad is not None:
                            nl_grad_accum[atom] += p.grad.detach().clone() * sys.kptWeights[kidx]

                trainLoss_system += systemKptLoss.detach().item() * sys.kptWeights[kidx]
                del systemKptLoss
                gc.collect()

        else: # multiprocessing
            if sys.fit_eph:
                raise NotImplementedError(
                    "e-ph coupling (fit_eph) is not supported with separateKptGrad + "
                    "multiprocessing (num_cores>0): the coupling loss needs the eigenvectors, "
                    "which are produced inside the worker processes and not returned. Use "
                    "train_naive (separateKptGrad=0), or run separateKptGrad serially "
                    "(num_cores=0), for coupling training.")
            optimizer.zero_grad()
            if LSDmodels:
                for key in LSDoptimizers:
                    LSDoptimizers[key].zero_grad()
            if spinOptimizer is not None:
                spinOptimizer.zero_grad()
            # Clear any stale PPparams grad (sets it to None) before pickling the
            # ham to workers, so each worker's backward starts from a clean slate.
            _zero_nonlocal_grad(nl_ctx)

            if (NNConfig['smooth_reorder']) and (prevBS is not None):
                print("WARNING. We are reordering the band structure according to smoothness using the previous iteration BS. ")
            prevBS = prevBS.detach() if prevBS is not None else None
            collect_nl = nl_ctx is not None
            args_list = [(kidx, hams[iSys], sys, optimizer, model, cachedMats_info, prevBS, LSDmodels, LSDoptimizers, spinModel, spinOptimizer, collect_nl) for kidx in range(sys.getNKpts())]

            # PyTorch autograd is not safe to use from forked workers.
            # Use an explicit spawn context for the per-k-point backward passes.
            # Each worker is pinned (via the initializer) to blas_threads_per_worker
            # linear-algebra threads, so the per-k-point eigensolve is multi-threaded
            # while num_cores*threads stays within the node budget (no oversubscription).
            blas_threads = NNConfig.get('blas_threads_per_worker', 1)
            ctx = mp.get_context("spawn")
            with ctx.Pool(NNConfig['num_cores'], initializer=pool_worker_init,
                          initargs=(blas_threads,)) as pool:
                results_systemKpt = pool.starmap(calcEigValsAtK_wGrad_parallel, args_list)
                gradients_systemKpt, trainLoss_systemKpt, eigValsList, extrapolated_eigValList, gradients_systemKpt_LSD, gradients_systemKpt_spin, gradients_systemKpt_nl, prof_snaps = zip(*results_systemKpt)
            # Fold each worker's per-kpt timing into the parent profiler so the
            # aggregated report includes the build+diagonalize work done in workers.
            for snap in prof_snaps:
                PROF.merge(snap)
            currBS = torch.stack(eigValsList).detach()
            extrapolated_points = torch.stack(extrapolated_eigValList).detach()

            gc.collect()
            gradients_system = merge_dicts(gradients_systemKpt)
            if spinModel is not None:
                gradients_system_spin = merge_dicts(gradients_systemKpt_spin)

            trainLoss_system = torch.sum(torch.tensor(trainLoss_systemKpt))
            loss_components["bandStruct"] += float(trainLoss_system)
            if LSDmodels:
                gradients_system_LSD = merge_dicts_LSD(gradients_systemKpt_LSD)

            # Accumulate the kpt-summed (already kpt-weighted) SOC/NL prefactor
            # grads from the workers into the running cross-system accumulator.
            if nl_ctx is not None:
                gradients_system_nl = merge_dicts(gradients_systemKpt_nl)
                for atom in nl_grad_accum:
                    if atom in gradients_system_nl:
                        nl_grad_accum[atom] = nl_grad_accum[atom] + gradients_system_nl[atom]

        # GLOBAL (whole-system, k-INDEPENDENT) losses -- non-decay penalty, spin-
        # aware magnitude penalty, deformation potentials, e-ph couplings, and
        # effective masses -- through the SAME shared compute_global_system_losses()
        # used by train_naive and evalBS_noGrad. Computed ONCE per system for BOTH
        # the serial and multiprocessing paths and backpropagated once, folding the
        # (un-kpt-weighted) grads into the manual accumulators alongside the per-kpt
        # band-structure grads. (Previously only penalty/mag/defpot were applied;
        # couplings and effective masses were silently dropped by this path.)
        global_terms, extras = compute_global_system_losses(
            model, spinModel, hams[iSys], sys, cachedMats_info, requires_grad=True)
        global_loss = global_loss_sum(global_terms)
        if (global_loss is not None) and global_loss.requires_grad:
            optimizer.zero_grad()
            if spinOptimizer is not None:
                spinOptimizer.zero_grad()
            if LSDmodels:
                for key in LSDoptimizers:
                    LSDoptimizers[key].zero_grad()
            _zero_nonlocal_grad(nl_ctx)
            global_loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients_system[name] = gradients_system.get(name, 0) + param.grad.detach().clone()
            if spinModel is not None:
                for name, param in spinModel.named_parameters():
                    if param.grad is not None:
                        gradients_system_spin[name] = gradients_system_spin.get(name, 0) + param.grad.detach().clone()
            if LSDmodels:
                for key in LSDmodels:
                    for name, param in LSDmodels[key].named_parameters():
                        if param.grad is not None:
                            gradients_system_LSD[key][name] = gradients_system_LSD[key].get(name, 0) + param.grad.detach().clone()
            if nl_ctx is not None:
                for atom, p in nl_ctx['params'].items():
                    if p.grad is not None:
                        nl_grad_accum[atom] = nl_grad_accum[atom] + p.grad.detach().clone()
        accumulate_loss_components(loss_components, global_terms)
        if global_loss is not None:
            trainLoss_system = trainLoss_system + float(global_loss.detach())

        # coupling / eff-mass file output + printouts (loss already tallied above)
        if sys.fit_eph and extras["calcCouplings"] is not None:
            write_coupling_bands(os.path.join(resultsFolder, f"couplingBands_{iSys}.dat"), sys, extras["calcCouplings"])
            print(f"couplingMSE = {float(global_terms['coupling'].detach()):.4g}")
        if sys.fit_eff_masses and extras["eff_masses"] is not None:
            print(f"Calculated effMasses = {extras['eff_masses']}, refEffMasses = {sys.expEffMasses}, "
                  f"effMass_Loss = {float(global_terms['effmass'].detach()):.4f}")

        total_gradients = merge_dicts([total_gradients, gradients_system])
        if spinModel is not None:
            total_gradients_spin = merge_dicts([total_gradients_spin, gradients_system_spin])
        if LSDmodels:
            total_gradients_LSD = merge_dicts_LSD([total_gradients_LSD, gradients_system_LSD])
        
        trainLoss += trainLoss_system

        # Plot each individual band for debugging
        if (NNConfig['smooth_reorder']) and (verbosity>=1): 
            for bandIdx in range(currBS.shape[1]):
                fig, ax = plotBandStruct_reorder(currBS.detach().numpy(), bandIdx)
                ax.plot(np.arange(len(currBS)), extrapolated_points[:, bandIdx].detach().numpy(), "gx:", alpha=0.8, markersize=4)
                ax.set(ylim=(min(extrapolated_points[:, bandIdx])-0.1, max(extrapolated_points[:, bandIdx])+0.1))
                # plot_highlight_kpt(ax, [0,3,6,13,19,26,34,40,50,60,65,70,79,90,100,108])
                fig.savefig(f"{resultsFolder}epoch_{epoch+1}_newBand_{bandIdx}.png")
                fig.savefig(f"{resultsFolder}epoch_{epoch+1}_newBand_{bandIdx}.pdf")
                plt.close()

    # Write the manually accumulated gradients and loss values back into the NN model
    optimizer.zero_grad()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in total_gradients:
                param.grad = total_gradients[name].detach().clone()

    if spinModel is not None:
        spinOptimizer.zero_grad()
        with torch.no_grad():
            for name, param in spinModel.named_parameters():
                if name in total_gradients_spin:
                    param.grad = total_gradients_spin[name].detach().clone()

    if LSDmodels:
        for key in LSDoptimizers:
            LSDoptimizers[key].zero_grad()
            with torch.no_grad():
                for name, param in LSDmodels[key].named_parameters():
                    if name in total_gradients_LSD[key]:
                        param.grad = total_gradients_LSD[key][name].detach().clone()

    with PROF.time("optimizer_step"):
        if preAdjustBool:
            if verbosity>1:
                print_and_inspect_gradients(model, f'{resultsFolder}preEpoch_{pre_epoch+1}_before_gradients.dat', show=True)
                print_and_inspect_NNParams(model, f'{resultsFolder}preEpoch_{pre_epoch+1}_before_params.dat', show=True)
            manual_GD_one_param(model, preAdjustStepSize)
            if LSDmodels:
                for key in LSDoptimizers:
                    manual_GD_one_param(LSDmodels[key], NNConfig['pre_adjust_LSD_step_size'])
            if spinModel is not None:
                manual_GD_one_param(spinModel, preAdjustStepSize)
        else:
            optimizer.step()
            if LSDmodels:
                for key in LSDoptimizers:
                    LSDoptimizers[key].step()
            if spinModel is not None:
                spinOptimizer.step()
            # Write the manually kpt-weighted SOC/NL prefactor gradients back onto the
            # shared PPparams leaves, then mask + step their optimizer.
            if nl_ctx is not None:
                nl_ctx['optimizer'].zero_grad()
                with torch.no_grad():
                    for atom, p in nl_ctx['params'].items():
                        p.grad = nl_grad_accum[atom].detach().clone()
                _step_nonlocal_grad(nl_ctx)

    torch.cuda.empty_cache()
    # print_and_inspect_gradients(model, show=NNConfig['printGrad'])

    return model, trainLoss, currBS, loss_components


def bandStruct_train_GPU(model, device, NNConfig, systems, hams, atomPPOrder, optimizer, scheduler, val_dataset, resultsFolder, cachedMats_info=None, LSDmodels=None, LSDoptimizers=None, LSDscheduler=None, LSDval_dataset=None, spinModel=None, spinOptimizer=None, spinScheduler=None):
    trainingCOST_x =[]
    training_COST = []
    validationCOST_x = []
    validation_COST =[]
    # Per-component loss breakdown histories, aligned with the *_COST_x lists, so
    # the final breakdown plot shows how each loss term (band structure, penalty,
    # mag_penalty, defpot, coupling, effmass) trends over training.
    training_comp_history = []
    validation_comp_history = []
    file_trainCost = open(f'{resultsFolder}final_training_cost.dat', "w")
    file_valCost = open(f'{resultsFolder}final_validation_cost.dat', "w")
    file_trainCost.write(COST_FILE_HEADER)
    file_valCost.write(COST_FILE_HEADER)
    file_trainCost.flush()
    file_valCost.flush()

    model.to(device)
    if LSDmodels:
        os.makedirs(f"{resultsFolder}LSD/", exist_ok=True)
        for key in LSDmodels:
            LSDmodels[key].to(device)
    if spinModel is not None:
        spinModel.to(device)

    # Enable gradient training of the SOC/NL prefactors (PPparams idx 5,6,7) if
    # requested. All hams share one PPparams dict, so this flips requires_grad
    # once for every system.
    nl_ctx = setup_nonlocal_grad(hams, atomPPOrder, NNConfig)
    if (nl_ctx is not None) and (NNConfig.get('perturbEvery', 0) > 0):
        raise NotImplementedError(
            "nonlocal_grad is incompatible with perturbEvery>0: perturb_model modifies "
            "PPparams in place, which is illegal on an autograd leaf. Set perturbEvery=0.")

    best_validation_loss = float('inf')
    no_improvement_count = 0
    prevBS = None

    pre_min_maxGrad = None
    pre_min_epoch = None
    pre_min_maxGrad_LSD = {atom: None for atom in set(atomPPOrder)}
    pre_min_epoch_LSD = {atom: None for atom in set(atomPPOrder)}
    # pre_adjustments. Optimizing only ONE PARAMETER at a time, which has the largest gradient
    if ('pre_adjust_moves' in NNConfig) and (NNConfig['pre_adjust_moves']>0): 
        for pre_epoch in range(NNConfig['pre_adjust_moves']):
            if ('pre_adjust_stepSize' in NNConfig): 
                pre_adjust_stepSize = NNConfig['pre_adjust_stepSize']
            else: 
                pre_adjust_stepSize = None

            model.train()
            if LSDmodels:
                for key in LSDmodels:
                    LSDmodels[key].train()
            if spinModel is not None:
                spinModel.train()

            if NNConfig['separateKptGrad']==0:
                model, trainLoss, trainComp = trainIter_naive(model, systems, hams, NNConfig, optimizer, cachedMats_info, NNConfig['runtime_flag'], preAdjustBool=True, preAdjustStepSize=pre_adjust_stepSize, resultsFolder=resultsFolder, pre_epoch=pre_epoch, LSDmodels=LSDmodels, LSDoptimizers=LSDoptimizers, spinModel=spinModel, spinOptimizer=spinOptimizer, nl_ctx=nl_ctx)
            else:
                model, trainLoss, prevBS, trainComp = trainIter_separateKptGrad(model, systems, hams, NNConfig, optimizer, cachedMats_info, preAdjustBool=True, preAdjustStepSize=pre_adjust_stepSize, resultsFolder=resultsFolder, pre_epoch=pre_epoch, prevBS=prevBS.detach() if prevBS is not None else None, spinModel=spinModel, spinOptimizer=spinOptimizer, nl_ctx=nl_ctx)

            pre_x = pre_epoch-NNConfig['pre_adjust_moves']-1
            file_trainCost.write(format_cost_line(pre_x, trainComp))
            file_trainCost.flush()
            trainingCOST_x.append(pre_x)
            training_COST.append(loss_components_total(trainComp))
            training_comp_history.append(trainComp)
            print(f"pre_adjust_moves [{pre_epoch+1}/{NNConfig['pre_adjust_moves']}], training cost: {loss_components_total(trainComp):.4f}")
            # print_and_inspect_gradients(model, f'{resultsFolder}preEpoch_{pre_epoch+1}_after_gradients.dat', show=True)
            # print_and_inspect_NNParams(model, f'{resultsFolder}preEpoch_{pre_epoch+1}_after_params.dat', show=True)

            model.eval()
            if LSDmodels:
                for key in LSDmodels:
                    LSDmodels[key].eval()
            if spinModel is not None:
                spinModel.eval()
            val_MSE = evalBS_noGrad(model, f'{resultsFolder}preEpoch_{pre_epoch+1}_plotBS.pdf', f'preEpoch_{pre_epoch+1}', NNConfig, hams, systems, cachedMats_info, writeBS=True, LSDmodels=LSDmodels, spinModel=spinModel)

            torch.save(model.state_dict(), f'{resultsFolder}preEpoch_{pre_epoch+1}_PPmodel.pth')
            if spinModel is not None:
                torch.save(spinModel.state_dict(), f'{resultsFolder}preEpoch_{pre_epoch+1}_spinModel.pth')
            if LSDmodels:
                for key in LSDmodels:
                    torch.save(LSDmodels[key].state_dict(), f'{resultsFolder}preEpoch_{pre_epoch+1}_LSDmodel_{key}.pth')
            else:
                print(f"WARNING: LSDmodels are NONE!")
            torch.cuda.empty_cache()

            maxGrad, _ = judge_well_conditioned_grad(model)
            if pre_min_maxGrad is None or maxGrad <= pre_min_maxGrad:
                print("This is the best pre-adjust epoch so far. ")
                pre_min_maxGrad = maxGrad
                pre_min_epoch = pre_epoch

            if LSDmodels:
                for key in LSDmodels:
                    maxGrad, _ = judge_well_conditioned_grad(LSDmodels[key])
                    if pre_min_maxGrad_LSD[key] is None or maxGrad <= pre_min_maxGrad_LSD[key]:
                        print(f"This is the best pre-adjust epoch for LSD[{key}] so far. ")
                        pre_min_maxGrad_LSD[key] = maxGrad
                        pre_min_epoch_LSD[key] = pre_epoch
            print()
        
        model.load_state_dict(torch.load(f'{resultsFolder}preEpoch_{pre_min_epoch+1}_PPmodel.pth'))
        print(f"We have re-loaded back to the preEpoch_{pre_min_epoch+1}, which gives the best-conditioned gradients. ")

        if spinModel is not None:
            spinModel.load_state_dict(torch.load(f'{resultsFolder}preEpoch_{pre_min_epoch+1}_spinModel.pth'))

        if LSDmodels:
            for key in LSDmodels:
                LSDmodels[key].load_state_dict(torch.load(f'{resultsFolder}preEpoch_{pre_min_epoch_LSD[key]+1}_LSDmodel_{key}.pth'))
                print(f"We have re-loaded LSD[{key}] back to the preEpoch_{pre_min_epoch_LSD[key]+1}, which gives the best-conditioned gradients. ")
        # Clean-up
        for pre_epoch in range(NNConfig['pre_adjust_moves']):
            if (pre_epoch%20!=0) and (pre_epoch!=pre_min_epoch): 
                os.remove(f'{resultsFolder}preEpoch_{pre_epoch+1}_BS_sys0.dat')
                os.remove(f'{resultsFolder}preEpoch_{pre_epoch+1}_PPmodel.pth')
                os.remove(f'{resultsFolder}preEpoch_{pre_epoch+1}_plotBS.pdf')
                os.remove(f'{resultsFolder}preEpoch_{pre_epoch+1}_plotBS.png')

    # Start the training-phase profile fresh (the pre-adjust moves / init eval
    # above accumulated into PROF; reset so the report below reflects training).
    PROF.reset()
    PROF.mem_checkpoint("train loop start")

    for epoch in range(NNConfig['max_num_epochs']):

        # train
        model.train()
        if LSDmodels:
            for key in LSDmodels:
                LSDmodels[key].train()
        if spinModel is not None:
            spinModel.train()
        if NNConfig['separateKptGrad']==0:
            model, trainLoss, trainComp = trainIter_naive(model, systems, hams, NNConfig, optimizer, cachedMats_info, NNConfig['runtime_flag'], resultsFolder=resultsFolder, epoch=epoch, LSDmodels=LSDmodels, LSDoptimizers=LSDoptimizers, spinModel=spinModel, spinOptimizer=spinOptimizer, nl_ctx=nl_ctx)
        else:
            model, trainLoss, prevBS, trainComp = trainIter_separateKptGrad(model, systems, hams, NNConfig, optimizer, cachedMats_info, resultsFolder=resultsFolder, epoch=epoch, prevBS=prevBS.detach() if prevBS is not None else None, LSDmodels=LSDmodels, LSDoptimizers=LSDoptimizers, spinModel=spinModel, spinOptimizer=spinOptimizer, nl_ctx=nl_ctx)
        file_trainCost.write(format_cost_line(epoch+1, trainComp))
        file_trainCost.flush()
        trainingCOST_x.append(epoch+1)
        training_COST.append(loss_components_total(trainComp))
        training_comp_history.append(trainComp)
        print(f"Epoch [{epoch+1}/{NNConfig['max_num_epochs']}], training cost (total incl. penalties): {loss_components_total(trainComp):.4f}  "
              + "  ".join(f"{n}={trainComp[n]:.3g}" for n in LOSS_TERM_NAMES if abs(trainComp[n]) > 0))
        PROF.mem_checkpoint(f"epoch {epoch+1}")
        # Periodic cumulative timing breakdown (build vs diagonalize) so a long
        # run shows where time is going without waiting for the final report.
        if (epoch + 1) % NNConfig['plotEvery'] == 0:
            PROF.report(f"RUNTIME PROFILE (cumulative through epoch {epoch+1})")
        if nl_ctx is not None:
            for atom, p in nl_ctx['params'].items():
                vals = p.detach()
                grad_vals = None if p.grad is None else [round(float(p.grad[i]), 6) for i in nl_ctx['indices']]
                print(f"    SOC/NL[{atom}] PPparams[{nl_ctx['indices']}] = "
                      f"{[round(float(vals[i]), 6) for i in nl_ctx['indices']]}  grad = {grad_vals}")
                # One file per atom type: previously this write sat OUTSIDE the
                # loop, so `atom` was whatever the last iteration left it as and a
                # single file (e.g. epoch_X_NParams.dat) held EVERY atom's params.
                write_nonlocal_params(f'{resultsFolder}epoch_{epoch+1}_{atom}Params.dat', nl_ctx, atom=atom)
        if (epoch<=9) or ((epoch + 1) % NNConfig['plotEvery'] == 0):
            print_and_inspect_gradients(model, f'{resultsFolder}epoch_{epoch+1}_gradients.dat', show=True)
            print_and_inspect_NNParams(model, f'{resultsFolder}epoch_{epoch+1}_params.dat', show=True)
            if LSDmodels:
                for key in LSDmodels:
                    print_and_inspect_gradients(LSDmodels[key], f'{resultsFolder}LSD/epoch_{epoch+1}_gradients_LSD_{key}.dat', show=True)
                    print_and_inspect_NNParams(LSDmodels[key], f'{resultsFolder}LSD/epoch_{epoch+1}_params_LSD_{key}.dat', show=True)

        judge_well_conditioned_grad(model)
        if LSDmodels:
            for key in LSDmodels:
                print(f"LSD ({key}):")
                judge_well_conditioned_grad(LSDmodels[key])

        # perturb the model
        if (NNConfig['perturbEvery']>0) and (epoch>0) and (epoch % NNConfig['perturbEvery']==0): 
            model, _, _ = perturb_model(model, hams, 0.10)
            print("WARNING: We have randomly perturbed all the params of the model by 10%. \n")

        # scheduler of learning rate
        if (epoch > 0) and (epoch % NNConfig['schedulerStep'] == 0):
            scheduler.step()
            if LSDmodels:
                LSDscheduler.step()
            if spinScheduler is not None:
                spinScheduler.step()
            if nl_ctx is not None:
                nl_ctx['scheduler'].step()

        # evaluation
        if (epoch + 1) % NNConfig['plotEvery'] == 0:
            model.eval()
            if LSDmodels:
                for key in LSDmodels:
                    LSDmodels[key].eval()
            if spinModel is not None:
                spinModel.eval()
            valComp = {}
            val_MSE = evalBS_noGrad(model, f'{resultsFolder}epoch_{epoch+1}_plotBS.pdf', f'epoch_{epoch+1}', NNConfig, hams, systems, cachedMats_info, writeBS=True, LSDmodels=LSDmodels, spinModel=spinModel, loss_components_out=valComp)
            validationCOST_x.append(epoch+1)
            validation_COST.append(val_MSE)
            validation_comp_history.append(valComp)
            print(f"Epoch [{epoch+1}/{NNConfig['max_num_epochs']}], validation cost (total incl. penalties): {loss_components_total(valComp):.4f}  "
                  + "  ".join(f"{n}={valComp[n]:.3g}" for n in LOSS_TERM_NAMES if abs(valComp[n]) > 0))
            file_valCost.write(format_cost_line(epoch+1, valComp))
            file_valCost.flush()
            
            model.cpu()
            fig = plotPP(atomPPOrder, val_dataset.q, val_dataset.q, val_dataset.vq_atoms, model(val_dataset.q), "ZungerForm", f"NN_{epoch+1}", ["-",":" ]*len(atomPPOrder), True, NNConfig['SHOWPLOTS']);
            fig.savefig(f'{resultsFolder}epoch_{epoch+1}_plotPP.pdf')
            fig.savefig(f'{resultsFolder}epoch_{epoch+1}_plotPP.png')
            if spinModel is not None:
                spinModel.cpu()
                fig_spin = plotPP_spin(atomPPOrder, val_dataset.q, model(val_dataset.q), spinModel(val_dataset.q), f"NN_{epoch+1}", NNConfig['SHOWPLOTS'])
                fig_spin.savefig(f'{resultsFolder}epoch_{epoch+1}_plotPP_spin.pdf')
                fig_spin.savefig(f'{resultsFolder}epoch_{epoch+1}_plotPP_spin.png')
                plt.close(fig_spin)
                spinModel.to(device)
            model.to(device)

            write_PP_qSpace(f'{resultsFolder}epoch_{epoch+1}_qSpace_pot.dat', model, atomPPOrder, qmax=NNConfig['qmax'], nQGrid=NNConfig['nQGrid'])
            if spinModel is not None:
                spinModel.cpu()
                write_PP_qSpace_spin(f'{resultsFolder}epoch_{epoch+1}_qSpace_pot_spin.dat', model, spinModel, atomPPOrder, qmax=NNConfig['qmax'], nQGrid=NNConfig['nQGrid'])
                spinModel.to(device)

            torch.save(model.state_dict(), f'{resultsFolder}epoch_{epoch+1}_PPmodel.pth')
            torch.save(optimizer.state_dict(), f'{resultsFolder}epoch_{epoch+1}_AdamState.pth')
            if spinModel is not None:
                torch.save(spinModel.state_dict(), f'{resultsFolder}epoch_{epoch+1}_spinModel.pth')
                torch.save(spinOptimizer.state_dict(), f'{resultsFolder}epoch_{epoch+1}_spin_AdamState.pth')
            torch.cuda.empty_cache()

            if LSDmodels:
                for key in LSDmodels:
                    LSDmodels[key].cpu()
                    # Plot all LSD potentials
                    for n_u in range(LSDval_dataset[key].n_unique):
                        n_q = LSDval_dataset[key].n_q_grid
                        q = LSDval_dataset[key].q[n_u*n_q:(n_u+1)*n_q].view(-1, 1)
                        vq_init = LSDval_dataset[key].vq_atoms[n_u*n_q:(n_u+1)*n_q].view(-1, 1)
                        N_alphas = LSDval_dataset[key].N_alphas[n_u*n_q:(n_u+1)*n_q].view(-1, 1)
                        x_inputs = torch.cat((N_alphas, q), dim=1)
                        fig = plotLSD(key, q, q, vq_init, LSDmodels[key](x_inputs), "InitialLSD", "OptLSD", ["-",":" ], True, NNConfig['SHOWPLOTS'])
                        fig.savefig(f'{resultsFolder}LSD/epoch_{epoch+1}_plotLSD_{key}_{n_u}.pdf')
                        LSDmodels[key].to(device)

                        write_LSD_qSpace(f'{resultsFolder}LSD/epoch_{epoch+1}_qSpace_pot_LSD_{key}_{n_u}.dat', LSDmodels[key], N_alphas[0])
                    
                    
                    print(f"Printing LSD model for {key} epoch {epoch}")
                    torch.save(LSDmodels[key].state_dict(), f'{resultsFolder}epoch_{epoch+1}_{key}_LSDmodel.pth')
                    torch.save(LSDoptimizers[key].state_dict(), f'{resultsFolder}epoch_{epoch+1}_{key}_LSD_AdamState.pth')
                    print_and_inspect_gradients(LSDmodels[key], f'{resultsFolder}LSD/epoch_{epoch+1}_gradients_LSD_{key}.dat', show=True)
                    print_and_inspect_NNParams(LSDmodels[key], f'{resultsFolder}LSD/epoch_{epoch+1}_params_LSD_{key}.dat', show=True)
        
        plt.close('all')
        torch.cuda.empty_cache()
    
    if LSDmodels:
        for key in LSDmodels:
            torch.save(LSDmodels[key].state_dict(), f'{resultsFolder}final_{key}_LSDmodel.pth')
            torch.save(LSDoptimizers[key].state_dict(), f'{resultsFolder}final_{key}_LSD_AdamState.pth')
    else:
        print(f"WARNING: LSDmodels is empty")

    if spinModel is not None:
        torch.save(spinModel.state_dict(), f'{resultsFolder}final_spinModel.pth')
        torch.save(spinOptimizer.state_dict(), f'{resultsFolder}final_spin_AdamState.pth')

    if nl_ctx is not None:
        write_nonlocal_params(f'{resultsFolder}final_nonlocalParams.dat', nl_ctx)
        print("\nFinal trained SOC/NL prefactors:")
        for atom, p in nl_ctx['params'].items():
            vals = p.detach()
            print(f"    {atom} PPparams[{nl_ctx['indices']}] = {[round(float(vals[i]), 6) for i in nl_ctx['indices']]}")

    fig_cost = plot_training_validation_cost(trainingCOST_x, training_COST, validation_cost_x=validationCOST_x, validation_cost=validation_COST, ylogBoolean=True, SHOWPLOTS=NNConfig['SHOWPLOTS']);
    fig_cost.savefig(resultsFolder + 'final_train_cost.pdf')

    # Per-component loss breakdown (band structure vs penalty vs defpot vs ...),
    # so it's clear which term dominates and how each trends over training.
    fig_break = plot_loss_breakdown(trainingCOST_x, training_comp_history,
                                    val_x=validationCOST_x, val_history=validation_comp_history,
                                    SHOWPLOTS=NNConfig['SHOWPLOTS'])
    fig_break.savefig(resultsFolder + 'final_train_cost_breakdown.pdf')
    fig_break.savefig(resultsFolder + 'final_train_cost_breakdown.png')
    plt.close('all')
    torch.cuda.empty_cache()

    # Final aggregated timing breakdown over the whole training run.
    PROF.report("FINAL RUNTIME PROFILE (whole training run)")
    PROF.mem_checkpoint("train loop end")
    return (training_COST, validation_COST)


def _perturb_nn_params(new_model, mode, percentage):
    """Apply the mode-dependent perturbation to an NN model's parameters in place.

    Mirrors the per-mode perturbation that perturb_model() applies to the
    local-potential `new_model`, so the spin-field model b(q) can be perturbed
    the same way in spin-polarized MC runs. Modes 6 and 7 don't perturb any NN
    parameters (they only touch SOC/NL ham params), so this is a no-op there.

    NOTE: b(q) is zero-initialized (final layer = 0). The multiplicative /
    normalization modes (1, 2, 3) leave zeros at zero and can never move b away
    from the unpolarized solution; use an additive mode (4, 5, or 8) to actually
    explore spin polarization.
    """
    if mode == 1:
        for param in new_model.parameters():
            perturbation = 1 + torch.rand_like(param) * (2 * percentage) - percentage
            param.data *= perturbation
    elif mode == 2:
        for param in new_model.parameters():
            perturbation = torch.zeros_like(param)
            with torch.no_grad():
                for idx in range(param.numel()):
                    value = param.view(-1)[idx]
                    if value > 20.0:
                        perturbation.view(-1)[idx] = -torch.rand(1) * percentage * value
                    elif value < -20.0:
                        perturbation.view(-1)[idx] = torch.rand(1) * percentage * value
                    elif -0.01 < value < 0.01:
                        random_sign = torch.randint(0, 2, (1,)) * 2 - 1
                        perturbation.view(-1)[idx] = random_sign * torch.rand(1) * 10 * percentage * value
                    else:
                        random_sign = torch.randint(0, 2, (1,)) * 2 - 1
                        perturbation.view(-1)[idx] = random_sign * torch.rand(1) * percentage * value
                param += perturbation
    elif mode == 3:
        original_params = {}
        for name, param in new_model.named_parameters():
            mean = param.data.mean()
            std = param.data.std()
            original_params[name] = (mean, std)
            param.data = (param.data - mean) / (std + 1e-8)
        with torch.no_grad():
            for name, param in new_model.named_parameters():
                num_params = param.data.numel()
                num_to_move = int(0.5 * num_params)
                indices = np.random.choice(num_params, num_to_move, replace=False)
                perturbations = torch.randn(num_params) * percentage
                param.data.view(-1)[indices] += perturbations[indices]
        for name, param in new_model.named_parameters():
            mean, std = original_params[name]
            param.data = param.data * std + mean
    elif mode == 4:
        for param in new_model.parameters():
            if (np.random.random() <= 0.6):
                random_sign = torch.randint(0, 2, param.shape, dtype=torch.float64) * 2 - 1
                param.data += percentage * random_sign
    elif mode == 5:
        for param in new_model.parameters():
            if (np.random.random() <= 0.6):
                random_sign = torch.randint(0, 2, param.shape, dtype=torch.float64) * 2 - 1
                param.data += percentage/10 * random_sign
    elif mode == 8:
        for param in new_model.parameters():
            if (np.random.random() <= 0.6):
                random_sign = torch.randint(0, 2, param.shape, dtype=torch.float64) * 2 - 1
                param.data += percentage/1 * random_sign


def perturb_model(model, hams, percentage=0.0, mode=1, spinModel=None):
    def check_atomPPOrder():
        atomPPOrder = getattr(hams[0], 'atomPPorder', None)  # This should be consistent across all hams
        if atomPPOrder is None:
            raise AttributeError("Expected the first ham to define `atomPPOrder`.")

        reference_order = tuple(atomPPOrder)
        for idx, ham in enumerate(hams[1:], start=1):
            ham_order = getattr(ham, 'atomPPorder', None)
            if ham_order is None:
                raise AttributeError(f"Hamiltonian at index {idx} does not have `atomPPOrder` defined.")
            if tuple(ham_order) != reference_order:
                raise ValueError(
                    "`atomPPOrder` must be consistent across all Hamiltonians. "
                    f"First Hamiltonian order={reference_order}, index {idx} order={tuple(ham_order)}.")
        return atomPPOrder

    atomPPOrder = check_atomPPOrder()

    # copy to new_model. Make changes on the new ones
    new_model = copy.deepcopy(model)
    # In spin-polarized runs, perturb the spin field b(q) alongside the local pot.
    new_spinModel = copy.deepcopy(spinModel) if spinModel is not None else None

    # Make a copy of the old ham_PPparams. Make changes in place on the hams.
    old_hams_PPparams = [copy.deepcopy(ham.PPparams) for ham in hams]

    # Perturb model on the new model, perturb the SOC and NL in place. 
    if mode == 1: 
        print(f"Perturbing the model by percentage: {percentage}")
        for param in new_model.parameters():
            perturbation = 1 + torch.rand_like(param) * (2 * percentage) - percentage
            param.data *= perturbation
            
        for atomType in atomPPOrder:
            # perturb SOC constant & NL constants
            for p in range(5, 8): # SOC and NL
                scale = (1 + np.random.random() * (2 * percentage/100) - percentage/100)
                for ham in hams: 
                    if atomType in ham.PPparams:
                        ham.PPparams[atomType][p] *= scale

    if mode == 2: 
        print(f"Perturbing the model by percentage: {percentage}")
        for param in new_model.parameters():
            perturbation = torch.zeros_like(param)
            
            with torch.no_grad():
                # Iterate over each element of the tensor
                for idx in range(param.numel()):
                    value = param.view(-1)[idx]  # Flatten the tensor to a 1D array for indexing

                    if value > 20.0:
                        perturbation.view(-1)[idx] = -torch.rand(1) * percentage * value
                    elif value < -20.0:
                        perturbation.view(-1)[idx] = torch.rand(1) * percentage * value
                    elif -0.01 < value < 0.01:
                        random_sign = torch.randint(0, 2, (1,)) * 2 - 1
                        perturbation.view(-1)[idx] = random_sign * torch.rand(1) * 10 * percentage * value
                    else:
                        random_sign = torch.randint(0, 2, (1,)) * 2 - 1
                        perturbation.view(-1)[idx] = random_sign * torch.rand(1) * percentage * value

                param += perturbation

        for atomType in atomPPOrder:
            # perturb SOC constant & NL constants
            for p in range(5, 8): # SOC and NL
                scale = (1 + np.random.random() * (2 * percentage/1000) - percentage/1000)
                for ham in hams: 
                    if atomType in ham.PPparams:
                        ham.PPparams[atomType][p] *= scale

    if mode == 3: 
        print(f"Perturbing the model by std after normalization: {percentage}")
        original_params = {}
        for name, param in new_model.named_parameters():
            mean = param.data.mean()
            std = param.data.std()
            original_params[name] = (mean, std)
            param.data = (param.data - mean) / (std + 1e-8)
        
        with torch.no_grad():
            for name, param in new_model.named_parameters():
                num_params = param.data.numel()
                num_to_move = int(0.5 * num_params)
                
                indices = np.random.choice(num_params, num_to_move, replace=False)

                perturbations = torch.randn(num_params) * percentage
                param.data.view(-1)[indices] += perturbations[indices]
        
        for name, param in new_model.named_parameters():
            mean, std = original_params[name]
            param.data = param.data * std + mean

    if mode == 4: 
        print(f"Perturbing the model by absolute steps: {percentage}. Perturbing the NL and SOC parameters by absolute steps: {percentage/1000}")
        for param in new_model.parameters():
            if (np.random.random() <= 0.6): 
                random_sign = torch.randint(0, 2, param.shape, dtype=torch.float64) * 2 - 1
                param.data += percentage * random_sign
  
        for atomType in atomPPOrder:
            # perturb SOC constant & NL constants
            for p in range(5, 8): # SOC and NL
                step = percentage/1000 * np.random.choice([-1, 1])
                if (np.random.random() <= 0.6): 
                    for ham in hams: 
                        if atomType in ham.PPparams:
                            ham.PPparams[atomType][p] += step

    if mode == 5: 
        print(f"Perturbing the model by absolute steps: {percentage/10}. Perturbing the NL and SOC parameters by absolute steps: {percentage}")
        for param in new_model.parameters():
            if (np.random.random() <= 0.6): 
                random_sign = torch.randint(0, 2, param.shape, dtype=torch.float64) * 2 - 1
                param.data += percentage/10 * random_sign

        for atomType in atomPPOrder:
            # perturb SOC constant & NL constants
            for p in range(5, 8): # SOC and NL
                step = percentage/1 * np.random.choice([-1, 1])
                if (np.random.random() <= 0.6): 
                    for ham in hams: 
                        if atomType in ham.PPparams:
                            ham.PPparams[atomType][p] += step

    if mode == 6: 
        print(f"Not perturbing the model. Perturbing the SOC parameter only by absolute steps: {percentage}")
        for atomType in atomPPOrder:
            for p in [5]: # SOC only
                step = percentage/1 * np.random.choice([-1, 1])
                if (np.random.random() <= 0.6): 
                    for ham in hams: 
                        if atomType in ham.PPparams:
                            ham.PPparams[atomType][p] += step

    if mode == 7: 
        print(f"Not perturbing the local model. Perturbing the NL parameter only by absolute steps: {percentage}")
        for atomType in atomPPOrder:
            for p in [6,7]: # NL only
                step = percentage/1 * np.random.choice([-1, 1])
                if (np.random.random() <= 0.6): 
                    for ham in hams: 
                        if atomType in ham.PPparams:
                            ham.PPparams[atomType][p] += step

    if mode == 8:
        print(f"Perturbing the model by absolute steps: {percentage}. Perturbing only the NL parameters by absolute steps: {percentage}")
        for param in new_model.parameters():
            if (np.random.random() <= 0.6):
                random_sign = torch.randint(0, 2, param.shape, dtype=torch.float64) * 2 - 1
                param.data += percentage/1 * random_sign

        for atomType in atomPPOrder:
            # perturb NL constants
            for p in [6, 7]: # NL
                step = percentage/1 * np.random.choice([-1, 1])
                if (np.random.random() <= 0.6):
                    for ham in hams:
                        if atomType in ham.PPparams:
                            ham.PPparams[atomType][p] += step

    # Perturb the spin field b(q) with the same per-mode rule as the local pot.
    # (The additive-mode caveat for b(q) is documented in _perturb_nn_params and
    # warned about once up front in runMC_NN.)
    if new_spinModel is not None:
        _perturb_nn_params(new_spinModel, mode, percentage)

    return new_model, new_spinModel, old_hams_PPparams


def runMC_NN(model, NNConfig, systems, hams, atomPPOrder, val_dataset, resultsFolder, cachedMats_info=None, spinModel=None):
    file_trainCost = open(f'{resultsFolder}final_mc_cost.dat', "w")
    file_trainCost.write("# iter      newLoss      accept?      bestLoss      currLoss\n")
    
    if spinModel is not None:
        print(f"Monte Carlo on a spin-polarized run (tot_magnetization != 0): "
              f"perturbing the spin field b(q) alongside the local potential.")
        mc_mode = NNConfig['mc_perturb_mode'] if 'mc_perturb_mode' in NNConfig else 1
        if mc_mode not in (4, 5, 8):
            print(f"WARNING: perturb mode {mc_mode} leaves the zero-initialized spin field b(q) "
                  f"at zero. Use an additive mode (4, 5, or 8) to explore spin polarization in MC.")

    bestModel = model
    bestSpinModel = spinModel
    bestLoss = evalBS_noGrad(bestModel, f'{resultsFolder}mc_iter_0_plotBS.pdf', f'mc_iter_0', NNConfig, hams, systems, cachedMats_info, resultsFolder=resultsFolder, spinModel=spinModel)
    print_and_inspect_NNParams(bestModel, f'{resultsFolder}best_params.dat', show=True)
    shutil.copy(f'{resultsFolder}mc_iter_0_plotBS.pdf', f'{resultsFolder}best_plotBS.pdf')
    currModel = model
    currSpinModel = spinModel
    currLoss = bestLoss
    trial_COST = [currLoss]
    accepted_COST = [currLoss]

    for iter in range(NNConfig['mc_iter']):
        print(f"\nIteration [{iter+1}/{NNConfig['mc_iter']}]: ")
        newModel, newSpinModel, old_PPparams = perturb_model(currModel, hams, percentage=NNConfig['mc_percentage'], mode=NNConfig['mc_perturb_mode'] if 'mc_perturb_mode' in NNConfig else 1, spinModel=currSpinModel)
        # writeBS=True dumps the band structures to mc_iter_{iter+1}_BS_sys{iSys}.dat
        # for this trial model. They are kept only when the step is accepted (see
        # below); rejected trials delete them to save storage.
        newLoss = evalBS_noGrad(newModel, f'{resultsFolder}mc_iter_{iter+1}_plotBS.pdf', f'mc_iter_{iter+1}', NNConfig, hams, systems, cachedMats_info, writeBS=True, resultsFolder=resultsFolder, spinModel=newSpinModel)
        print(f"newLoss={newLoss.item():.4f}. ")

        mc_rand = np.exp(-1 * NNConfig['mc_beta'] * (np.sqrt(newLoss) - np.sqrt(currLoss)))
        mc_accept_bool = mc_rand > np.random.uniform(low=0.0, high=1.0)

        if newLoss < bestLoss:   # accept
            bestLoss = newLoss
            bestModel = newModel
            bestSpinModel = newSpinModel
            currLoss = newLoss
            currModel = newModel
            currSpinModel = newSpinModel
            file_trainCost.write(f"{iter+1}    {newLoss.item():.4f}    {1}    {bestLoss.item():.4f}    {currLoss.item():.4f}\n")
            file_trainCost.flush()
            print(f"Accepted. currLoss={currLoss.item():.4f}")
            print_and_inspect_NNParams(newModel, f'{resultsFolder}best_params.dat', show=True)
            print_and_inspect_NNParams(newModel, f'{resultsFolder}final_params.dat', show=True)

            fig = plotPP(atomPPOrder, val_dataset.q, val_dataset.q, val_dataset.vq_atoms, currModel(val_dataset.q), "ZungerForm", f"mc_iter_{iter+1}", ["-",":" ]*len(atomPPOrder), True, NNConfig['SHOWPLOTS']);
            fig.savefig(f'{resultsFolder}mc_iter_{iter+1}_plotPP.pdf')
            fig.savefig(f'{resultsFolder}mc_iter_{iter+1}_plotPP.png')
            torch.save(currModel.state_dict(), f'{resultsFolder}mc_iter_{iter+1}_PPmodel.pth')
            write_PP_qSpace(f'{resultsFolder}final_qSpace_pot.dat', newModel, atomPPOrder, qmax=NNConfig['qmax'], nQGrid=NNConfig['nQGrid'])
            shutil.copy(f'{resultsFolder}final_qSpace_pot.dat', f'{resultsFolder}best_qSpace_pot.dat')

            # spin-polarized run: save the accepted spin field b(q) and the
            # up/down-resolved potentials for both best_ and final_.
            if newSpinModel is not None:
                write_PP_qSpace_spin(f'{resultsFolder}final_qSpace_pot_spin.dat', newModel, newSpinModel, atomPPOrder, qmax=NNConfig['qmax'], nQGrid=NNConfig['nQGrid'])
                shutil.copy(f'{resultsFolder}final_qSpace_pot_spin.dat', f'{resultsFolder}best_qSpace_pot_spin.dat')
                torch.save(newSpinModel.state_dict(), f'{resultsFolder}final_spinModel.pth')
                shutil.copy(f'{resultsFolder}final_spinModel.pth', f'{resultsFolder}best_spinModel.pth')
                fig_spin = plotPP_spin(atomPPOrder, val_dataset.q, newModel(val_dataset.q), newSpinModel(val_dataset.q), f"mc_iter_{iter+1}", NNConfig['SHOWPLOTS'])
                fig_spin.savefig(f'{resultsFolder}final_plotPP_spin.pdf')
                fig_spin.savefig(f'{resultsFolder}best_plotPP_spin.pdf')
                plt.close(fig_spin)

            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_PPmodel.pth', f'{resultsFolder}final_PPmodel.pth')
            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_plotPP.pdf', f'{resultsFolder}final_plotPP.pdf')
            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_plotBS.pdf', f'{resultsFolder}final_plotBS.pdf')
            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_PPmodel.pth', f'{resultsFolder}best_PPmodel.pth')
            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_plotPP.pdf', f'{resultsFolder}best_plotPP.pdf')
            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_plotBS.pdf', f'{resultsFolder}best_plotBS.pdf')

            # keep the accepted band structures; mirror them to best_/final_
            for iSys in range(len(systems)):
                for suffix in ['', '_trueE', '_relative']:
                    src = f'{resultsFolder}mc_iter_{iter+1}_BS_sys{iSys}{suffix}.dat'
                    if os.path.exists(src):
                        shutil.copy(src, f'{resultsFolder}final_BS_sys{iSys}{suffix}.dat')
                        shutil.copy(src, f'{resultsFolder}best_BS_sys{iSys}{suffix}.dat')

            # remove iteration files to save storage
            os.remove(f'{resultsFolder}mc_iter_{iter+1}_plotPP.pdf')
            os.remove(f'{resultsFolder}mc_iter_{iter+1}_plotBS.pdf')
            os.remove(f'{resultsFolder}mc_iter_{iter+1}_PPmodel.pth')

            for ham in hams: 
                for atomType in ham.PPparams:
                    f = open(f'{resultsFolder}mc_iter_{iter+1}_{atomType}Params.dat', "w")
                    for i in range(9): 
                        f.write(f"{ham.PPparams[atomType][i]:.8f}\n")
                    f.close()
                    shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_{atomType}Params.dat', f'{resultsFolder}final_{atomType}Params.dat')
                    shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_{atomType}Params.dat', f'{resultsFolder}best_{atomType}Params.dat')

        elif mc_accept_bool:   # new loss is higher, but we still accept.
            currLoss = newLoss
            currModel = newModel
            currSpinModel = newSpinModel
            file_trainCost.write(f"{iter+1}    {newLoss.item():.4f}    {1}    {bestLoss.item():.4f}    {currLoss.item():.4f}\n")
            file_trainCost.flush()
            print(f"Accepted. currLoss={currLoss.item():.4f}")
            print_and_inspect_NNParams(newModel, f'{resultsFolder}final_params.dat', show=True)

            fig = plotPP(atomPPOrder, val_dataset.q, val_dataset.q, val_dataset.vq_atoms, currModel(val_dataset.q), "ZungerForm", f"mc_iter_{iter+1}", ["-",":" ]*len(atomPPOrder), True, NNConfig['SHOWPLOTS']);
            fig.savefig(f'{resultsFolder}mc_iter_{iter+1}_plotPP.pdf')
            fig.savefig(f'{resultsFolder}mc_iter_{iter+1}_plotPP.png')
            torch.save(currModel.state_dict(), f'{resultsFolder}mc_iter_{iter+1}_PPmodel.pth')
            write_PP_qSpace(f'{resultsFolder}final_qSpace_pot.dat', newModel, atomPPOrder, qmax=NNConfig['qmax'], nQGrid=NNConfig['nQGrid'])

            # spin-polarized run: save the accepted spin field b(q) to final_.
            if newSpinModel is not None:
                write_PP_qSpace_spin(f'{resultsFolder}final_qSpace_pot_spin.dat', newModel, newSpinModel, atomPPOrder, qmax=NNConfig['qmax'], nQGrid=NNConfig['nQGrid'])
                torch.save(newSpinModel.state_dict(), f'{resultsFolder}final_spinModel.pth')
                fig_spin = plotPP_spin(atomPPOrder, val_dataset.q, newModel(val_dataset.q), newSpinModel(val_dataset.q), f"mc_iter_{iter+1}", NNConfig['SHOWPLOTS'])
                fig_spin.savefig(f'{resultsFolder}final_plotPP_spin.pdf')
                plt.close(fig_spin)

            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_PPmodel.pth', f'{resultsFolder}final_PPmodel.pth')
            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_plotPP.pdf', f'{resultsFolder}final_plotPP.pdf')
            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_plotBS.pdf', f'{resultsFolder}final_plotBS.pdf')

            # keep the accepted band structures; mirror them to final_
            for iSys in range(len(systems)):
                for suffix in ['', '_trueE', '_relative']:
                    src = f'{resultsFolder}mc_iter_{iter+1}_BS_sys{iSys}{suffix}.dat'
                    if os.path.exists(src):
                        shutil.copy(src, f'{resultsFolder}final_BS_sys{iSys}{suffix}.dat')

            for ham in hams:
                for atomType in ham.PPparams:
                    f = open(f'{resultsFolder}mc_iter_{iter+1}_{atomType}Params.dat', "w")
                    for i in range(9):
                        f.write(f"{ham.PPparams[atomType][i]:.8f}\n")
                    f.close()
                    shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_{atomType}Params.dat', f'{resultsFolder}final_{atomType}Params.dat')

        else:   # don't accept
            # currModel is never changed, as function perturb_model makes a copy of the model

            # But we need to revert the changes on the SOC and NL parameters
            for i, oldPPparam in enumerate(old_PPparams): 
                hams[i].PPparams = oldPPparam

            file_trainCost.write(f"{iter+1}    {newLoss.item():.4f}    {0}    {bestLoss.item():.4f}    {currLoss.item():.4f}\n")
            file_trainCost.flush()
            print(f"Not accepted. currLoss={currLoss.item():.4f}")
            
            fig = plotPP(atomPPOrder, val_dataset.q, val_dataset.q, val_dataset.vq_atoms, currModel(val_dataset.q), "ZungerForm", f"mc_iter_{iter+1}", ["-",":" ]*len(atomPPOrder), True, NNConfig['SHOWPLOTS']);
            # fig.savefig(f'{resultsFolder}mc_iter_{iter+1}_plotPP.pdf')
            os.remove(f'{resultsFolder}mc_iter_{iter+1}_plotBS.pdf')
            os.remove(f'{resultsFolder}mc_iter_{iter+1}_plotBS.png')

            # rejected trial: discard its band structures to save storage
            for iSys in range(len(systems)):
                for suffix in ['', '_trueE', '_relative']:
                    src = f'{resultsFolder}mc_iter_{iter+1}_BS_sys{iSys}{suffix}.dat'
                    if os.path.exists(src):
                        os.remove(src)
        
        trial_COST.append(newLoss.item())
        accepted_COST.append(currLoss.item())
    
        plt.close('all')
        torch.cuda.empty_cache()

    model = currModel
        
    fig_cost = plot_mc_cost(trial_COST, accepted_COST, False, NNConfig['SHOWPLOTS']);
    fig_cost.savefig(f'{resultsFolder}final_mc_cost.pdf')
    file_trainCost.close()
    return (trial_COST, accepted_COST, bestModel, currModel, bestSpinModel, currSpinModel)
