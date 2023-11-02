import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.rcParams['lines.markersize'] = 3
import torch
from torch.utils.data import Dataset, DataLoader

from constants.constants import * 
torch.set_default_dtype(torch.float32)
torch.manual_seed(24)

def pot_func(x, params): 
    pot = (params[0]*(x*x - params[1]) / (params[2] * torch.exp(params[3]*x*x) - 1.0))
    return pot

# TODO: Add dr, vr as variables
def realSpacePot(vq, qSpacePot): 
    # each input is a 1D tensor of torch.Size([nGrid]). vq is also assumed to be equally spaced. 
    dq = vq[1] - vq[0]
    nGrid = vq.shape[0]
    
    dr = 0.02*2*np.pi / (nGrid * dq)
    vr = torch.linspace(0, (nGrid - 1) * dr, nGrid)
    rSpacePot = torch.zeros(nGrid)
    
    for ir in range(nGrid): 
        if ir==0: 
            prefactor = 4*np.pi*dq / (8*np.pi**3)
            rSpacePot[ir] = torch.sum(prefactor * vq**2 * qSpacePot)
        else: 
            prefactor = 4*np.pi*dq / (8*np.pi**3 * vr[ir])
            rSpacePot[ir] = torch.sum(prefactor * vq * torch.sin(vq * vr[ir]) * qSpacePot)

    return (vr.view(-1,1), rSpacePot.view(-1,1))


def plotBandStruct(nSystem, bandStruct_array, marker_array, label_array, SHOWPLOTS): 
    # The inputs bandStruct_array, marker_array, label_array are arrays of tensors. They should be ordered as: 
    # ref_system1, predict_system1, ref_system2, predict_system2, ..., ref_systemN, predict_systemN
    if nSystem == 1:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        for bandStructIndex in range(2): 
            numBands = len(bandStruct_array[bandStructIndex][0])
            numKpts = len(bandStruct_array[bandStructIndex])
            for i in range(numBands): 
                if i==0: 
                    axs[0].plot(np.arange(numKpts), bandStruct_array[bandStructIndex][:, i].detach().numpy(), marker_array[bandStructIndex], label=label_array[bandStructIndex])
                    axs[1].plot(np.arange(numKpts), bandStruct_array[bandStructIndex][:, i].detach().numpy(), marker_array[bandStructIndex], label=label_array[bandStructIndex])
                else: 
                    axs[0].plot(np.arange(numKpts), bandStruct_array[bandStructIndex][:, i].detach().numpy(), marker_array[bandStructIndex])
                    axs[1].plot(np.arange(numKpts), bandStruct_array[bandStructIndex][:, i].detach().numpy(), marker_array[bandStructIndex])
        axs[0].legend(frameon=False)
        axs[1].set(ylim=(-8, -2))
        axs[0].get_xaxis().set_ticks([0, 20, 40, 45, 60])
        axs[0].get_xaxis().set_ticklabels(["L", r"$\Gamma$", "X", "K", r"$\Gamma$"])
        axs[1].get_xaxis().set_ticks([0, 20, 40, 45, 60])
        axs[1].get_xaxis().set_ticklabels(["L", r"$\Gamma$", "X", "K", r"$\Gamma$"])
        
    else:
        fig, axs = plt.subplots(nSystem, 2, figsize=(9, 4 * nSystem))
        for iSystem in range(nSystem): 
            for bandStructIndex in range(iSystem, iSystem+2): 
                numBands = len(bandStruct_array[bandStructIndex][0])
                numKpts = len(bandStruct_array[bandStructIndex])
                for i in range(numBands): 
                    if i==0: 
                        axs[iSystem, 0].plot(np.arange(numKpts), bandStruct_array[bandStructIndex][:, i].detach().numpy(), marker_array[bandStructIndex], label=label_array[bandStructIndex])
                        axs[iSystem, 1].plot(np.arange(numKpts), bandStruct_array[bandStructIndex][:, i].detach().numpy(), marker_array[bandStructIndex], label=label_array[bandStructIndex])
                    else: 
                        axs[iSystem, 0].plot(np.arange(numKpts), bandStruct_array[bandStructIndex][:, i].detach().numpy(), marker_array[bandStructIndex])
                        axs[iSystem, 1].plot(np.arange(numKpts), bandStruct_array[bandStructIndex][:, i].detach().numpy(), marker_array[bandStructIndex])
            axs[iSystem, 0].legend(frameon=False)
            axs[iSystem, 1].set(ylim=(-8, -2))
            axs[iSystem, 0].get_xaxis().set_ticks([0, 20, 40, 45, 60])
            axs[iSystem, 0].get_xaxis().set_ticklabels(["L", r"$\Gamma$", "X", "K", r"$\Gamma$"])
            axs[iSystem, 1].get_xaxis().set_ticks([0, 20, 40, 45, 60])
            axs[iSystem, 1].get_xaxis().set_ticklabels(["L", r"$\Gamma$", "X", "K", r"$\Gamma$"])

    fig.tight_layout()
    if SHOWPLOTS: 
        plt.show()
    return fig


def plotPP(atomPPOrder, ref_q, pred_q, ref_vq_atoms, pred_vq_atoms, ref_labelName, pred_labelName, lineshape_array, boolPlotDiff, SHOWPLOTS):
    # ref_vq_atoms and pred_vq_atoms are 2D tensors. Each tensor contains the pseudopotential (either ref or pred)
    # for atoms in the order of atomPPOrder. 
    # ref_labelName and pred_labelName are strings. 
    # lineshape_array has twice the length of atomPPOrder, with: ref_atom1, pred_atom1, ref_atom2, pred_atom2, ... 
    if boolPlotDiff and torch.equal(ref_q, pred_q): 
        fig, axs = plt.subplots(1,3, figsize=(12,4))
        ref_q = ref_q.view(-1).detach().numpy()
        pred_q = pred_q.view(-1).detach().numpy()
        
        for iAtom in range(len(atomPPOrder)):
            ref_vq = ref_vq_atoms[:, iAtom].view(-1).detach().numpy()
            pred_vq = pred_vq_atoms[:, iAtom].view(-1).detach().numpy()
            axs[0].plot(ref_q, ref_vq, lineshape_array[iAtom*2], label=atomPPOrder[iAtom]+" "+ref_labelName)
            axs[0].plot(pred_q, pred_vq, lineshape_array[iAtom*2+1], label=atomPPOrder[iAtom]+" "+pred_labelName)
            axs[1].plot(ref_q, pred_vq - ref_vq, lineshape_array[iAtom*2], label=atomPPOrder[iAtom]+" diff (pred - ref)")
            (ref_vr, ref_rSpacePot) = realSpacePot(torch.tensor(ref_q), torch.tensor(ref_vq))
            (pred_vr, pred_rSpacePot) = realSpacePot(torch.tensor(pred_q), torch.tensor(pred_vq))
            axs[2].plot(ref_vr.view(-1).detach().numpy(), ref_rSpacePot.view(-1).detach().numpy(), lineshape_array[iAtom*2], label=atomPPOrder[iAtom]+" "+ref_labelName)
            axs[2].plot(pred_vr.view(-1).detach().numpy(), pred_rSpacePot.view(-1).detach().numpy(), lineshape_array[iAtom*2+1], label=atomPPOrder[iAtom]+" "+pred_labelName)
        axs[0].set(xlabel=r"$q$", ylabel=r"$v(q)$")
        axs[0].legend(frameon=False)
        axs[1].set(xlabel=r"$q$", ylabel=r"$v_{NN}(q) - v_{func}(q)$")
        axs[1].legend(frameon=False)
        axs[2].set(xlabel=r"$r$", ylabel=r"$v(r)$", xlim=(0,12))
        axs[2].legend(frameon=False)
    
    else:
        fig, axs = plt.subplots(1,2, figsize=(9,4))
        ref_q = ref_q.view(-1).detach().numpy()
        pred_q = pred_q.view(-1).detach().numpy()
        
        for iAtom in range(len(atomPPOrder)):
            ref_vq = ref_vq_atoms[:, iAtom].view(-1).detach().numpy()
            pred_vq = pred_vq_atoms[:, iAtom].view(-1).detach().numpy()
            axs[0].plot(ref_q, ref_vq, lineshape_array[iAtom*2], label=atomPPOrder[iAtom]+" "+ref_labelName)
            axs[0].plot(pred_q, pred_vq, lineshape_array[iAtom*2+1], label=atomPPOrder[iAtom]+" "+pred_labelName)
            (ref_vr, ref_rSpacePot) = realSpacePot(torch.tensor(ref_q), torch.tensor(ref_vq))
            (pred_vr, pred_rSpacePot) = realSpacePot(torch.tensor(pred_q), torch.tensor(pred_vq))
            axs[1].plot(ref_vr.view(-1).detach().numpy(), ref_rSpacePot.view(-1).detach().numpy(), lineshape_array[iAtom*2], label=atomPPOrder[iAtom]+" "+ref_labelName)
            axs[1].plot(pred_vr.view(-1).detach().numpy(), pred_rSpacePot.view(-1).detach().numpy(), lineshape_array[iAtom*2+1], label=atomPPOrder[iAtom]+" "+pred_labelName)
        axs[0].set(xlabel=r"$q$", ylabel=r"$v(q)$")
        axs[0].legend(frameon=False)
        axs[1].set(xlabel=r"$r$", ylabel=r"$v(r)$", xlim=(0,12))
        axs[1].legend(frameon=False)
        
    fig.tight_layout()
    if SHOWPLOTS: 
        plt.show()
    return fig


def plot_training_validation_cost(training_cost, validation_cost, ylogBoolean, SHOWPLOTS): 
    if len(training_cost) != len(validation_cost): 
        return
    else:
        epochs = range(1, len(training_cost) + 1)
    
    # Plot training and validation costs
    fig, axs = plt.subplots(1,1, figsize=(6,4))
    
    axs.plot(epochs, training_cost, "b-", label='Training Cost')
    axs.plot(epochs, validation_cost, "r:", label='Validation Cost')
    
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

# TODO: change
def FT_converge_and_write_pp(qmax_array, model, xmin, xmax, ymin, ymax, choiceQMax, choiceNQGrid, ppPlotFile, potCdFile, potSeFile):
    cmap = plt.get_cmap('rainbow')
    figtot, axstot = plt.subplots(1,2, figsize=(9,4))
    for i, color in enumerate(cmap(np.linspace(0, 1, len(qmax_array)))):
        plotqGrid = torch.linspace(0.0, qmax_array[i], NQGRID).view(-1, 1)
        NN = model(plotqGrid)
        (vr_Cd, rSpacePot_Cd) = realSpacePot(plotqGrid.view(-1), NN[:, 0].view(-1))
        (vr_Se, rSpacePot_Se) = realSpacePot(plotqGrid.view(-1), NN[:, 1].view(-1))
        if qmax_array[i]==30: 
            axstot[0].plot(vr_Cd.detach().numpy(), rSpacePot_Cd.detach().numpy(), "-", color=color, label="My FT, 0<q<%d" % qmax_array[i])
            axstot[1].plot(vr_Se.detach().numpy(), rSpacePot_Se.detach().numpy(), "-", color=color, label="My FT, 0<q<%d" % qmax_array[i])
        else:
            axstot[0].plot(vr_Cd.detach().numpy(), rSpacePot_Cd.detach().numpy(), "-", color=color, label="0<q<%d" % qmax_array[i])
            axstot[1].plot(vr_Se.detach().numpy(), rSpacePot_Se.detach().numpy(), "-", color=color, label="0<q<%d" % qmax_array[i])
    
    axstot[0].set(xlim=(xmin, xmax), ylim=(ymin, ymax), title="Cd PP", xlabel=r"$r$ (Bohr radius)", ylabel=r"$v(r)$")
    axstot[1].set(xlim=(xmin, xmax), ylim=(ymin, ymax), title="Se PP", xlabel=r"$r$ (Bohr radius)", ylabel=r"$v(r)$")
    axstot[0].legend()
    figtot.tight_layout()
    plt.show()
    
    choiceQGrid = torch.linspace(0.0, choiceQMax, choiceNQGrid).view(-1, 1)
    NN = model(choiceQGrid)
    fig = plotPP([val_dataset.q, choiceQGrid], [val_dataset.vq_Cd, NN[:, 0]], [val_dataset.vq_Se, NN[:, 1]], ["ZungerForm", f"NN_init_Zunger"], ["-", ":"], False);
    fig.savefig(ppPlotFile)
    (vr_Cd, rSpacePot_Cd) = realSpacePot(choiceQGrid.view(-1), NN[:, 0].view(-1))
    (vr_Se, rSpacePot_Se) = realSpacePot(choiceQGrid.view(-1), NN[:, 1].view(-1))
    potCd = torch.cat((vr_Cd, rSpacePot_Cd), dim=1).detach().numpy()
    potSe = torch.cat((vr_Se, rSpacePot_Se), dim=1).detach().numpy()
    np.savetxt(potCdFile, potCd, delimiter='    ', fmt='%e')
    np.savetxt(potSeFile, potCd, delimiter='    ', fmt='%e')
    return