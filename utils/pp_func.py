import torch
import numpy as np
from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.rcParams['lines.markersize'] = 3

from constants.constants import * 

def pot_func(x, params): 
    pot = (params[0]*(x*x - params[1]) / (params[2] * torch.exp(params[3]*x*x) - 1.0))
    return pot

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
            if i==0: 
                axs_flat[2*iSystem+0].plot(np.arange(numKpts), bandStruct_list[2*iSystem][:, i].detach().numpy(), "bo", label="Reference")
                axs_flat[2*iSystem+1].plot(np.arange(numKpts), bandStruct_list[2*iSystem][:, i].detach().numpy(), "bo", label="Reference")
            else: 
                axs_flat[2*iSystem+0].plot(np.arange(numKpts), bandStruct_list[2*iSystem][:, i].detach().numpy(), "bo")
                axs_flat[2*iSystem+1].plot(np.arange(numKpts), bandStruct_list[2*iSystem][:, i].detach().numpy(), "bo")
                
        # plot prediction
        numBands = len(bandStruct_list[2*iSystem+1][0])
        numKpts = len(bandStruct_list[2*iSystem+1])
        for i in range(numBands): 
            if i==0: 
                axs_flat[2*iSystem+0].plot(np.arange(numKpts), bandStruct_list[2*iSystem+1][:, i].detach().numpy(), "r-", label="NN prediction")
                axs_flat[2*iSystem+1].plot(np.arange(numKpts), bandStruct_list[2*iSystem+1][:, i].detach().numpy(), "r-", label="NN prediction")
            else: 
                axs_flat[2*iSystem+0].plot(np.arange(numKpts), bandStruct_list[2*iSystem+1][:, i].detach().numpy(), "r-")
                axs_flat[2*iSystem+1].plot(np.arange(numKpts), bandStruct_list[2*iSystem+1][:, i].detach().numpy(), "r-")
        axs_flat[2*iSystem+0].legend(frameon=False)
        refEList = bandStruct_list[2*iSystem][bandStruct_list[2*iSystem] > -50]
        refEmin = torch.min(refEList).item()
        refEmax = torch.max(refEList).item()
        predEList = bandStruct_list[2*iSystem+1][bandStruct_list[2*iSystem+1] > -50]
        predEmin = torch.min(predEList).item()
        predEmax = torch.max(predEList).item()
        axs_flat[2*iSystem+0].set(ylim=(min(refEmin, predEmin)-0.5, max(refEmax, predEmax)+0.5))
        axs_flat[2*iSystem+1].set(ylim=(bulkSystem_list[iSystem].BS_plot_center-4, bulkSystem_list[iSystem].BS_plot_center+5), title=systemNames[iSystem])
        # axs_flat[2*iSystem+0].get_xaxis().set_ticks([0, 20, 40, 45, 60])
        # axs_flat[2*iSystem+0].get_xaxis().set_ticklabels(["L", r"$\Gamma$", "X", "K", r"$\Gamma$"])
        # axs_flat[2*iSystem+1].get_xaxis().set_ticks([0, 20, 40, 45, 60])
        # axs_flat[2*iSystem+1].get_xaxis().set_ticklabels(["L", r"$\Gamma$", "X", "K", r"$\Gamma$"])

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
            (ref_vr, ref_rSpacePot) = realSpacePot(torch.tensor(ref_q), torch.tensor(ref_vq), 3000)
            (pred_vr, pred_rSpacePot) = realSpacePot(torch.tensor(pred_q), torch.tensor(pred_vq), 3000)
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
            (ref_vr, ref_rSpacePot) = realSpacePot(torch.tensor(ref_q), torch.tensor(ref_vq), 3000)
            (pred_vr, pred_rSpacePot) = realSpacePot(torch.tensor(pred_q), torch.tensor(pred_vq), 3000)
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
    epochs = range(0, len(training_cost))
    evaluation_frequency = len(training_cost) // len(validation_cost)
    evaluation_epochs = list(range(evaluation_frequency-1, len(training_cost), evaluation_frequency))
    
    # Plot training and validation costs
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))

    axs.plot(epochs, training_cost, "b-", label='Training Cost')
    axs.plot(evaluation_epochs, validation_cost, "r:", label='Validation Cost')

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

def FT_converge_and_write_pp(atomPPOrder, qmax_array, nQGrid_array, nRGrid_array, model, val_dataset, xmin, xmax, ymin, ymax, choiceQMax, choiceNQGrid, choiceNRGrid, ppPlotFilePrefix, potRAtomFilePrefix, SHOWPLOTS):
    cmap = plt.get_cmap('rainbow')
    figtot, axstot = plt.subplots(1, len(atomPPOrder), figsize=(9,4))
    
    combinations = list(product(qmax_array, nQGrid_array, nRGrid_array))
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(combinations)))
    for i, combo in enumerate(combinations):
        qmax, nQGrid, nRGrid = combo

        qGrid = torch.linspace(0.0, qmax, nQGrid).view(-1, 1)
        NN = model(qGrid)
        for iAtom in range(len(atomPPOrder)):
            (vr, rSpacePot) = realSpacePot(qGrid.view(-1), NN[:, iAtom].view(-1), nRGrid)
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
    NN = model(choiceQGrid)
    fig = plotPP(atomPPOrder, val_dataset.q, choiceQGrid, val_dataset.vq_atoms, NN, "ZungerForm", "NN", ["-",":" ]*len(atomPPOrder), False, SHOWPLOTS);
    fig.savefig(ppPlotFilePrefix+".png") 
    for iAtom in range(len(atomPPOrder)):
        (vr, rSpacePot) = realSpacePot(choiceQGrid.view(-1), NN[:, iAtom].view(-1), choiceNRGrid)
        pot = torch.cat((vr, rSpacePot), dim=1).detach().numpy()
        np.savetxt(potRAtomFilePrefix+"_"+atomPPOrder[iAtom]+".dat", pot, delimiter='    ', fmt='%e')
    if SHOWPLOTS: 
        plt.show()
    return