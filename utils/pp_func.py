import torch
import numpy as np
from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.rcParams['lines.markersize'] = 3
from .constants import * 

torch.set_default_dtype(torch.float64)

def pot_func(x, params): 
    pot = (params[0]*(x*x - params[1]) / (params[2] * torch.exp(params[3]*x*x) - 1.0))
    return pot


def long_range_correction(x, gamma, lr_coeff):
    """
    Long-range Coulomb tail added to the short-range potential.
    lr_coeff (long-range coefficient) is passed in as a live nn.Parameter. 
    """
    if not isinstance(lr_coeff, torch.Tensor):
        lr_coeff = torch.as_tensor(lr_coeff, dtype=x.dtype, device=x.device)
    elif lr_coeff.dtype != x.dtype or lr_coeff.device != x.device:
        lr_coeff = lr_coeff.to(dtype=x.dtype, device=x.device)

    correction = torch.zeros_like(x)
    mask = x > 1e-4
    if mask.any():
        correction[mask] = -lr_coeff * 4 * np.pi / (x[mask]**2) * torch.exp(-x[mask]**2 / (4 * gamma**2))
    return correction


def pot_funcLR(x, params, gamma, lr_scale=None):
    """
    Return Zunger local potential with optional long-range override.

    ``params[4]`` stores the static long-range coefficient from the init files.
    When LR training is enabled we pass a live ``nn.Parameter`` via
    ``lr_scale`` so gradients flow through that tensor without mutating the
    original parameter array.
    """
    pot = pot_func(x, params)
    lr_coeff = lr_scale if lr_scale is not None else params[4]
    return pot + long_range_correction(x, gamma, lr_coeff)


def add_long_range_to_model_output(q_grid, model_output, atomPPOrder, lr_params=None, pp_params=None, lr_gamma=0.2):
    """Add the LR tail to NN-local pseudopotentials evaluated on ``q_grid``."""
    if lr_gamma is None:
        return model_output.detach().clone() if isinstance(model_output, torch.Tensor) else torch.as_tensor(model_output, dtype=torch.float64)

    if isinstance(model_output, torch.Tensor):
        augmented = model_output.detach().clone()
        dtype = augmented.dtype
        device = augmented.device
    else:
        augmented = torch.as_tensor(model_output, dtype=torch.float64)
        dtype = augmented.dtype
        device = augmented.device

    q_tensor = torch.as_tensor(q_grid, dtype=dtype, device=device).view(-1)
    for idx, atom_label in enumerate(atomPPOrder):
        coeff_tensor = None
        if lr_params is not None and atom_label in lr_params:
            coeff_tensor = lr_params[atom_label].detach()
        elif pp_params is not None and atom_label in pp_params and len(pp_params[atom_label]) > 4:
            raw = pp_params[atom_label][4]
            coeff_tensor = raw.detach() if isinstance(raw, torch.Tensor) else torch.tensor(raw, dtype=dtype, device=device)

        if coeff_tensor is None:
            continue

        coeff_tensor = coeff_tensor.to(dtype=dtype, device=device)
        if abs(coeff_tensor.item()) < 1e-12:
            continue

        tail = long_range_correction(q_tensor, lr_gamma, coeff_tensor)
        augmented[:, idx] = augmented[:, idx] + tail

    return augmented
  

def realSpacePot(vq, qSpacePot, nRGrid, rmax=25): 
    # vq and qSpacePot are both 1D tensor of torch.Size([nQGrid]). vq is assumed to be equally spaced. 
    # rmax and nRGrid are both scalars
    dq = vq[1] - vq[0]
    
    # dr = 0.02*2*np.pi / (nGrid * dq)
    # vr = torch.linspace(0, (nGrid - 1) * dr, nGrid)
    vr = torch.linspace(0, rmax, nRGrid)
    rSpacePot = torch.zeros(nRGrid)
    
    for ir in range(nRGrid): 
        if ir==0: 
            prefactor = 4*np.pi*dq / (8*np.pi**3)
            rSpacePot[ir] = torch.sum(prefactor * vq**2 * qSpacePot)
        else: 
            prefactor = 4*np.pi*dq / (8*np.pi**3 * vr[ir])
            rSpacePot[ir] = torch.sum(prefactor * vq * torch.sin(vq * vr[ir]) * qSpacePot)

    return (vr.view(-1,1), rSpacePot.view(-1,1))


def plotBandStruct(bulkSystem_list, bandStruct_list, SHOWPLOTS): 
    # The input bandStruct_list is a list of tensors. They should be ordered as: 
    # ref_system1, predict_system1, ref_system2, predict_system2, ..., ref_systemN, predict_systemN
    systemNames = [x.systemName for x in bulkSystem_list]
    nSystem = len(systemNames)
    if (len(bandStruct_list)!=2*nSystem): 
        raise ValueError("The lengths of bandStruct_list do not match the expected values.")

    fig, axs = plt.subplots(nSystem, 2, figsize=(9, 4 * nSystem))
    axs_flat = axs.flatten()
    for iSystem in range(nSystem): 
        # plot ref
        numBands = len(bandStruct_list[2*iSystem][0])
        numKpts = len(bandStruct_list[2*iSystem])
        for i in range(numBands): 
            if bulkSystem_list[iSystem].bandWeights[i]!=0:
                axs_flat[2*iSystem+0].plot(np.arange(numKpts), bandStruct_list[2*iSystem][:, i].detach().numpy(), "bo", alpha=0.5, markersize=2)
                axs_flat[2*iSystem+1].plot(np.arange(numKpts), bandStruct_list[2*iSystem][:, i].detach().numpy(), "bo", alpha=0.5, markersize=2)
        axs_flat[2*iSystem+0].plot([], [], "bo", alpha=0.5, markersize=2, label='Reference')
                
        # plot prediction
        numBands = len(bandStruct_list[2*iSystem+1][0])
        numKpts = len(bandStruct_list[2*iSystem+1])
        for i in range(numBands): 
            if bulkSystem_list[iSystem].bandWeights[i]!=0:
                axs_flat[2*iSystem+0].plot(np.arange(numKpts), np.sort(bandStruct_list[2*iSystem+1].detach().numpy(), axis=1)[:, i], "r-", alpha=0.6)
                axs_flat[2*iSystem+1].plot(np.arange(numKpts), np.sort(bandStruct_list[2*iSystem+1].detach().numpy(), axis=1)[:, i], "r-", alpha=0.6)
        axs_flat[2*iSystem+0].plot([], [], "r-", alpha=0.6, label="NN prediction")
        axs_flat[2*iSystem+0].legend(frameon=False)
        # refEList = bandStruct_list[2*iSystem][bandStruct_list[2*iSystem] > -50]
        # refEmin = torch.min(refEList).item()
        # refEmax = torch.max(refEList).item()
        # predEList = bandStruct_list[2*iSystem+1][bandStruct_list[2*iSystem+1] > -50]
        # predEmin = torch.min(predEList).item()
        # predEmax = torch.max(predEList).item()
        # axs_flat[2*iSystem+0].set(ylim=(min(refEmin, predEmin)-0.5, max(refEmax, predEmax)+0.5))
        axs_flat[2*iSystem+0].set(ylim=(bulkSystem_list[iSystem].BS_plot_center-bulkSystem_list[iSystem].BS_plot_CBVB_range, bulkSystem_list[iSystem].BS_plot_center+bulkSystem_list[iSystem].BS_plot_CBVB_range))
        axs_flat[2*iSystem+1].set(ylim=(bulkSystem_list[iSystem].BS_plot_center-bulkSystem_list[iSystem].BS_plot_CBVB_range_zoom, bulkSystem_list[iSystem].BS_plot_center+bulkSystem_list[iSystem].BS_plot_CBVB_range_zoom), title=systemNames[iSystem])
        # axs_flat[2*iSystem+0].get_xaxis().set_ticks([0, 20, 40, 45, 60])
        # axs_flat[2*iSystem+0].get_xaxis().set_ticklabels(["L", r"$\Gamma$", "X", "K", r"$\Gamma$"])
        # axs_flat[2*iSystem+1].get_xaxis().set_ticks([0, 20, 40, 45, 60])
        # axs_flat[2*iSystem+1].get_xaxis().set_ticklabels(["L", r"$\Gamma$", "X", "K", r"$\Gamma$"])

    fig.tight_layout()
    if SHOWPLOTS: 
        plt.show()
    return fig


def plotBandStructFromFile(refFile, calcFile): 
    refBS = np.loadtxt(refFile)[:, 1:]
    calcBS = np.loadtxt(calcFile)[:, 1:]

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    # plot ref
    numBands = len(refBS[0])
    numKpts = len(refBS)
    for i in range(numBands): 
        if i==0: 
            axs[0].plot(np.arange(numKpts), refBS[:, i], "bo", alpha=0.5, markersize=2, label="Reference")
            axs[1].plot(np.arange(numKpts), refBS[:, i], "bo", alpha=0.5, markersize=2, label="Reference")
        else: 
            axs[0].plot(np.arange(numKpts), refBS[:, i], "bo", alpha=0.5, markersize=2)
            axs[1].plot(np.arange(numKpts), refBS[:, i], "bo", alpha=0.5, markersize=2)

    # plot prediction
    numBands = len(calcBS[0])
    numKpts = len(calcBS)
    for i in range(numBands): 
        if i==0: 
            axs[0].plot(np.arange(numKpts), calcBS[:, i], "r-", alpha=0.6, label="Calc")
            axs[1].plot(np.arange(numKpts), calcBS[:, i], "r-", alpha=0.6, label="Calc")
        else: 
            axs[0].plot(np.arange(numKpts), calcBS[:, i], "r-", alpha=0.6)
            axs[1].plot(np.arange(numKpts), calcBS[:, i], "r-", alpha=0.6)
    axs[0].legend(frameon=False)
    axs[0].set(ylim=(-3000, -1000))
    axs[1].set(ylim=(-9.5, -1.5))

    fig.tight_layout()
    return (fig, axs)


def plotBandStruct_reorder(newOrderBS, bandIdx): 
    fig, ax = plt.subplots(1, 1, figsize=(8,8))

    numBands = len(newOrderBS[0])
    numKpts = len(newOrderBS)
    for i in range(numBands): 
        if i==0: 
            ax.plot(np.arange(numKpts), newOrderBS[:, i], "bo-", alpha=0.1, markersize=2)
        else: 
            ax.plot(np.arange(numKpts), newOrderBS[:, i], "bo-", alpha=0.1, markersize=2)

    # plot new ordering
    numKpts = len(newOrderBS)
    ax.plot(np.arange(numKpts), newOrderBS[:, bandIdx], "ro-", alpha=0.8, markersize=2, label=f"band{bandIdx}")
    ax.legend()
    ax.set(ylim=(min(newOrderBS[:, bandIdx])-0.2, max(newOrderBS[:, bandIdx])+0.2))
    # ax.get_xaxis().set_ticks([0, 10, 20, 30, 40, 50, 60, 70, 79, 80, 90, 100, 108, 110, 120, 130, 140, 149])
    # ax.get_xaxis().set_ticklabels(["R", 10, 20, 30, 40, r"$\Gamma$", 60, 70, "X", 80, 90, 100, "M", 110, 120, 130, 140, r"$\Gamma$"])
    ax.grid(alpha=0.5)

    fig.tight_layout()
    return fig, ax


def plotPP(atomPPOrder, ref_q, pred_q, ref_vq_atoms, pred_vq_atoms, ref_labelName, pred_labelName, lineshape_array, boolPlotDiff, SHOWPLOTS, ref_component="local only", pred_component="local only", pred_lr_vq_atoms=None, pred_lr_component="NN_loc + LR tail", lr_params=None, pp_params=None, lr_gamma=0.2, lr_lineshape='--'):
    def _annotate(label: str, component: str) -> str:
        component = component.strip()
        component_suffix = component if component else "unspecified"
        return f"{label} [{component_suffix}]"

    def _as_tensor(data):
        return data if isinstance(data, torch.Tensor) else torch.as_tensor(data, dtype=torch.float64)

    def _lr_style(idx: int):
        if isinstance(lr_lineshape, (list, tuple)):
            return lr_lineshape[idx]
        return lr_lineshape

    ref_q_tensor = _as_tensor(ref_q).view(-1)
    pred_q_tensor = _as_tensor(pred_q).view(-1)
    ref_vq_tensor = _as_tensor(ref_vq_atoms)
    pred_vq_tensor = _as_tensor(pred_vq_atoms)

    if pred_lr_vq_atoms is None and (lr_params is not None or pp_params is not None):
        pred_lr_vq_atoms = add_long_range_to_model_output(pred_q_tensor, pred_vq_tensor, atomPPOrder, lr_params=lr_params, pp_params=pp_params, lr_gamma=lr_gamma)

    include_lr = pred_lr_vq_atoms is not None
    if include_lr:
        pred_lr_tensor = _as_tensor(pred_lr_vq_atoms)
        diff = (pred_lr_tensor - pred_vq_tensor).abs().max().item()
        if diff < 1e-12:
            include_lr = False
    else:
        pred_lr_tensor = None

    same_grid = ref_q_tensor.shape == pred_q_tensor.shape and torch.allclose(ref_q_tensor, pred_q_tensor)

    if boolPlotDiff and same_grid:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))

    q_ref_np = ref_q_tensor.detach().numpy()
    q_pred_np = pred_q_tensor.detach().numpy()

    for iAtom, atom_label in enumerate(atomPPOrder):
        ref_vq = ref_vq_tensor[:, iAtom].detach()
        pred_vq = pred_vq_tensor[:, iAtom].detach()
        ref_legend = _annotate(atom_label + " " + ref_labelName, ref_component)
        pred_legend = _annotate(atom_label + " " + pred_labelName, pred_component)

        axs[0].plot(q_ref_np, ref_vq.cpu().numpy(), lineshape_array[iAtom * 2], label=ref_legend)
        axs[0].plot(q_pred_np, pred_vq.cpu().numpy(), lineshape_array[iAtom * 2 + 1], label=pred_legend)

        if include_lr:
            pred_lr = pred_lr_tensor[:, iAtom].detach()
            axs[0].plot(q_pred_np, pred_lr.cpu().numpy(), _lr_style(iAtom), label=_annotate(atom_label + " " + pred_labelName, pred_lr_component))

        if boolPlotDiff and same_grid:
            diff_local = (pred_vq - ref_vq).cpu().numpy()
            axs[1].plot(q_pred_np, diff_local, lineshape_array[iAtom * 2 + 1], label=f"{atom_label} diff ({pred_component} - {ref_component})")
            if include_lr:
                diff_lr = (pred_lr - ref_vq).cpu().numpy()
                axs[1].plot(q_pred_np, diff_lr, _lr_style(iAtom), label=f"{atom_label} diff ({pred_lr_component} - {ref_component})")

        ref_vr, ref_rSpace = realSpacePot(ref_q_tensor, ref_vq, 3000)
        pred_vr, pred_rSpace = realSpacePot(pred_q_tensor, pred_vq, 3000)
        r_axis = 2 if boolPlotDiff and same_grid else 1
        axs[r_axis].plot(ref_vr.view(-1).detach().cpu().numpy(), ref_rSpace.view(-1).detach().cpu().numpy(), lineshape_array[iAtom * 2], label=ref_legend)
        axs[r_axis].plot(pred_vr.view(-1).detach().cpu().numpy(), pred_rSpace.view(-1).detach().cpu().numpy(), lineshape_array[iAtom * 2 + 1], label=pred_legend)

        if include_lr:
            pred_lr_vr, pred_lr_rSpace = realSpacePot(pred_q_tensor, pred_lr, 3000)
            axs[r_axis].plot(pred_lr_vr.view(-1).detach().cpu().numpy(), pred_lr_rSpace.view(-1).detach().cpu().numpy(), _lr_style(iAtom), label=_annotate(atom_label + " " + pred_labelName, pred_lr_component))

    axs[0].set(xlabel=r"$q$", ylabel=r"$v(q)$")
    axs[0].legend(frameon=False)

    if boolPlotDiff and same_grid:
        axs[1].set(xlabel=r"$q$", ylabel=r"$v_{NN}(q) - v_{func}(q)$")
        axs[1].legend(frameon=False)
        axs[2].set(xlabel=r"$r$", ylabel=r"$v(r)$")
        axs[2].legend(frameon=False)
    else:
        axs[1].set(xlabel=r"$r$", ylabel=r"$v(r)$")
        axs[1].legend(frameon=False)

    fig.tight_layout()
    if SHOWPLOTS:
        plt.show()
    return fig


def plot_training_validation_cost(training_cost_x, training_cost, validation_cost_x=None, validation_cost=None, ylogBoolean=True, SHOWPLOTS=False): 
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    
    # epochs = range(0, len(training_cost))
    axs.plot(training_cost_x, training_cost, "b-", label='Training Cost')     # np.array(epochs)+1

    if (validation_cost_x is not None) and (validation_cost is not None) and (len(validation_cost) != 0): 
        # evaluation_frequency = len(training_cost) // len(validation_cost)
        # evaluation_epochs = list(range(evaluation_frequency-1, len(training_cost), evaluation_frequency))
        # axs.plot(np.array(evaluation_epochs)+1, validation_cost, "r:", label='Validation Cost')
        axs.plot(validation_cost_x, validation_cost, "r:", label='Validation Cost')

    if ylogBoolean:
        axs.set_yscale('log')
    else:
        axs.set_yscale('linear')
    axs.set(xlabel="Epochs", ylabel="Cost", title="Training and Validation Costs")
    axs.legend(frameon=False)
    axs.grid(True)
    fig.tight_layout()
    if SHOWPLOTS:
        plt.show()
    return fig


def FT_converge_and_write_pp(atomPPOrder, qmax_array, nQGrid_array, nRGrid_array, model, val_dataset, xmin, xmax, ymin, ymax, choiceQMax, choiceNQGrid, choiceNRGrid, ppPlotFilePrefix, potRAtomFilePrefix, SHOWPLOTS, lr_params=None, pp_params=None, lr_gamma=0.2):
    cmap = plt.get_cmap('rainbow')
    figtot, axstot = plt.subplots(1, len(atomPPOrder), figsize=(9,4))
    
    # Ensure axstot is always iterable
    if len(atomPPOrder) == 1:
        axstot = [axstot]  # Wrap in a list
        
    combinations = list(product(qmax_array, nQGrid_array, nRGrid_array))
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(combinations)))
    for i, combo in enumerate(combinations):
        qmax, nQGrid, nRGrid = combo

        qGrid = torch.linspace(0.0, qmax, nQGrid).view(-1, 1)
        with torch.no_grad():
            nn_local = model(qGrid)
            nn_total = add_long_range_to_model_output(qGrid, nn_local, atomPPOrder, lr_params=lr_params, pp_params=pp_params, lr_gamma=lr_gamma)
        for iAtom in range(len(atomPPOrder)):
            (vr, rSpacePot) = realSpacePot(qGrid.view(-1), nn_total[:, iAtom].view(-1), nRGrid)
            if (qmax==choiceQMax) and (nQGrid==choiceNQGrid) and (nRGrid==choiceNRGrid): 
                axstot[iAtom].plot(vr.detach().numpy(), rSpacePot.detach().numpy(), "-", color=colors[i], label="My FT, 0<q<%d, nQGrid=%d, nRGrid=%d" % (qmax,nQGrid,nRGrid))
            else:
                axstot[iAtom].plot(vr.detach().numpy(), rSpacePot.detach().numpy(), "-", color=colors[i], label="0<q<%d, nQGrid=%d, nRGrid=%d" % (qmax,nQGrid,nRGrid))
    
    for iAtom in range(len(atomPPOrder)):
        axstot[iAtom].set(xlim=(xmin, xmax), ylim=(ymin, ymax), title=atomPPOrder[iAtom]+" PP", xlabel=r"$r$ (Bohr radius)", ylabel=r"$v(r)$")
    axstot[0].legend(frameon=False, fontsize=7)
    figtot.tight_layout()
    figtot.savefig(ppPlotFilePrefix+"converge.png") 
    if SHOWPLOTS: 
        plt.show()
    
    choiceQGrid = torch.linspace(0.0, choiceQMax, choiceNQGrid).view(-1, 1)
    with torch.no_grad():
        nn_local = model(choiceQGrid)
        nn_total = add_long_range_to_model_output(choiceQGrid, nn_local, atomPPOrder, lr_params=lr_params, pp_params=pp_params, lr_gamma=lr_gamma)
    fig = plotPP(
        atomPPOrder,
        val_dataset.q,
        choiceQGrid,
        val_dataset.vq_atoms,
        nn_local,
        "ZungerForm",
        "NN",
        ["-",":" ]*len(atomPPOrder),
        False,
        SHOWPLOTS,
        ref_component="analytic local (no LR tail)",
        pred_component="NN_loc (no LR tail)",
        pred_lr_vq_atoms=nn_total,
        lr_params=lr_params,
        pp_params=pp_params,
        lr_gamma=lr_gamma
    );
    fig.savefig(ppPlotFilePrefix+".png") 
    for iAtom in range(len(atomPPOrder)):
        (vr, rSpacePot) = realSpacePot(choiceQGrid.view(-1), nn_total[:, iAtom].view(-1), choiceNRGrid)
        pot = torch.cat((vr, rSpacePot), dim=1).detach().numpy()
        np.savetxt(potRAtomFilePrefix+"_"+atomPPOrder[iAtom]+".dat", pot, delimiter='    ', fmt='%e')
    if SHOWPLOTS: 
        plt.show()
    return


def plot_multiple_train_cost(*file_groups, labels=None, ylogBoolean=False, ymin=None, ymax=None, xlabel='nEpoch', ylabel='Training cost'):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    if labels is None:
        labels = [f'Group {i+1}' for i in range(len(file_groups))]
    
    for i, file_group in enumerate(file_groups):
        all_cost = np.zeros(0)
        for filename in file_group:
            data = np.loadtxt(filename)[:,1]
            all_cost = np.hstack([all_cost, data])
        ax.plot(all_cost, "-", alpha=0.7, label=labels[i])
    
    if ylogBoolean:
        ax.set_yscale('log')
    if (ymin is not None) or (ymax is not None):
        current_ylim = ax.get_ylim()
        new_ylim = (ymin if ymin is not None else current_ylim[0], ymax if ymax is not None else current_ylim[1])
        ax.set_ylim(new_ylim)

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_mc_cost(trial_cost, accepted_cost, ylogBoolean, SHOWPLOTS): 
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    iter = range(0, len(trial_cost))
    ax.plot(np.array(iter)+1, trial_cost, "b-", label='Trial Cost')
    ax.plot(np.array(iter)+1, accepted_cost, "r:", label='Accepted Cost')

    if ylogBoolean:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    ax.set(xlabel="Iterations", ylabel="Cost", title="Trial and Accepted Costs")
    ax.legend(frameon=False)
    ax.grid(True)
    fig.tight_layout()
    if SHOWPLOTS:
        plt.show()
    return fig
