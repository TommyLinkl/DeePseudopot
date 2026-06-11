import torch
import numpy as np
from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib import rc
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['lines.markersize'] = 3
from .constants import * 

torch.set_default_dtype(torch.float64)

def qSpacePot_ft(r, V_r, q_magnitudes):
    """
    Compute V(q) = 4π/q ∫ V(r) sin(qr) r dr  for each |q|

    q_magnitudes: [NQGRID, 1] tensor of |q| values
    r_max: cutoff in real space (Bohr) — make sure V(r) -> 0 before here
    n_r: number of radial quadrature points
    """
    # Build a 1D radial grid (this is NOT paired with q points)
    dr = r[1] - r[0]

    # Evaluate the NN once on the r grid
    r = r.squeeze()
    V_r = V_r.squeeze()                            # [n_r]

    # Compute the transform for all q simultaneously
    q = q_magnitudes.squeeze()                     # [NQGRID]

    # Outer product: sin(qr) for all (q, r) pairs
    qr = torch.outer(q, r)                         # [NQGRID, n_r]
    sin_qr = torch.sin(qr)                         # [NQGRID, n_r]

    # Integrand: V(r) * sin(qr) * r, integrated over r
    integrand = V_r * r * sin_qr                   # [NQGRID, n_r]  (broadcasts)
    integral = torch.sum(integrand * dr, dim=-1)   # [NQGRID]

    # Handle q=0 separately via L'Hopital: V(q=0) = 4π ∫ V(r) r² dr
    V_q = 4 * torch.pi / q * integral

    # Fix q=0 if present
    q0_mask = q < 1e-10
    if q0_mask.any():
        V_q0 = 4 * torch.pi * torch.sum(V_r * r**2 * dr)
        V_q[q0_mask] = V_q0

    return V_q                                     # [NQGRID]

def pot_func(x, params):
    pot = (params[0]*(x*x - params[1]) / (params[2] * torch.exp(params[3]*x*x) - 1.0))
    return pot


def long_range_correction(x, gamma, lr_coeff):
    """Gaussian-screened Coulomb tail added to the short-range potential."""
    if not isinstance(lr_coeff, torch.Tensor):
        lr_coeff = torch.as_tensor(lr_coeff, dtype=x.dtype, device=x.device)
    elif lr_coeff.dtype != x.dtype or lr_coeff.device != x.device:
        lr_coeff = lr_coeff.to(dtype=x.dtype, device=x.device)

    correction = torch.zeros_like(x)
    mask = x > 1e-4
    if mask.any():
        correction[mask] = -lr_coeff * 4 * np.pi / (x[mask]**2) * torch.exp(-x[mask]**2 / (4 * gamma**2))
    return correction


def pot_funcLR(x, params, gamma):
    pot = params[0]*(x*x - params[1]) / (params[2] * torch.exp(params[3]*x*x) - 1.0)
    nzid = torch.nonzero(x > 1e-4, as_tuple=True) # x is batched, but want to avoid division by 0
    pot[nzid] -= params[4] * 4 * np.pi / (x[nzid]**2) * torch.exp(-1 * x[nzid]**2 / (4*gamma**2))
    return pot

def build_basisLSD(q, nBasis):
    """Construct and cache the radial basis for the local structure-dependent potentials"""

    # Default centers, sigmas
    centers = np.array([0.0] * nBasis)
    sig_i = 0.0
    sig_f = 1.0
    dsig = (sig_f - sig_i) / nBasis
    sigmas = np.linspace(sig_i + dsig, sig_f, nBasis)
    
    B = normalized_gaussians(q, centers, sigmas)
    B[B < 1e-16] = 0.0
    fig, ax = plt.subplots(figsize=(4,4))
    
    for m in range(nBasis):
      ax.scatter(q, B[:, m], linewidth=1.5, label=fr"m = {m}, $\sigma$ = {sigmas[m]:.2f}")
    ax.set_xlim(0.0, 4.0)
    ax.set_xlabel("q")
    ax.set_ylabel(r"$B_m(q)$")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"lsd_basis_m_{nBasis}.pdf", format='pdf', dpi=200)
    return B

def normalized_gaussians(r_grid, center, sigma):
    """
    Construct normalized radial Gaussians with cutoff.

    Parameters
    ----------
    r_grid : array, shape (nr,)
        Radial grid from 0..Rc (in Å or Bohr).
    centers : array-like, (M,)
        Centers rho_m of Gaussians.
    sigma : float
        Width parameter of Gaussians.
    Rc : float
        Cutoff radius.

    Returns
    -------
    g : (nr,) array
        One normalized basis function B_m(r).
    """
    # raw Gaussian
    g = np.exp(-0.5 * ((r_grid - center) / sigma)**2)
    norm = 1 / (sigma * np.sqrt(2*np.pi))
    # if norm < 1e-14:
    #     raise ValueError(f"Gaussian at rho={rho} too narrow or cutoff too strong")
    g = g / norm
    return g
    

def pot_funcLSD(q, coeffs, nBasis=3):
    """Return the local structure-dependent pseudopotential correction
    v_\alpha^{lsd} = sum_m c^\alpha_m(N_\alpha) * B^\alpha_m
    where 
    N_\alpha => atomistic descriptor (symmetry function like Behler-Parrinello)
    c^\alpha_m => coefficient determined by neural network
    B^\alpha_m => basis of radial functions in q-space
    """
    # Default centers, sigmas
    centers = np.array([0.0] * nBasis)
    sig_i = 0.0
    sig_f = 1.0
    dsig = (sig_f - sig_i) / nBasis
    sigmas = np.linspace(sig_i + dsig, sig_f, nBasis)

    # radial_basis = torch.tensor(build_basisLSD(q, nBasis), dtype=q.dtype, device=q.device)
    # print(f"Radial basis: \n{radial_basis[:,0]}")
    pot = torch.zeros_like(q)
    for m in range(nBasis):
        pot += coeffs[m] * normalized_gaussians(q, centers[m], sigmas[m])
    return pot

# Vectorized version of Fourier transform - Daniel C 3/12/26
def realSpacePot(vq, qSpacePot, nRGrid, rmax=25):
    dq = vq[1] - vq[0]
    vr = torch.linspace(0, rmax, nRGrid, device=vq.device)    # (nRGrid,)

    vq_ = vq.flatten()            # guarantee (nQGrid,)
    qp_ = qSpacePot.flatten()     # guarantee (nQGrid,)

    # (nRGrid-1, 1) * (1, nQGrid) -> (nRGrid-1, nQGrid)
    sin_term  = torch.sin(vr[1:, None] * vq_[None, :])
    prefactor = 4 * np.pi * dq / (8 * np.pi**3 * vr[1:])     # (nRGrid-1,)
    bulk      = prefactor * (sin_term * (vq_ * qp_)[None, :]).sum(dim=1)

    r0 = (4 * np.pi * dq / (8 * np.pi**3)) * (vq_**2 * qp_).sum()

    rSpacePot = torch.cat([r0.unsqueeze(0), bulk])

    return (vr.view(-1, 1), rSpacePot.view(-1, 1))

# def realSpacePot(vq, qSpacePot, nRGrid, rmax=25): 
#     # vq and qSpacePot are both 1D tensor of torch.Size([nQGrid]). vq is assumed to be equally spaced. 
#     # rmax and nRGrid are both scalars
#     dq = vq[1] - vq[0]
    
#     # dr = 0.02*2*np.pi / (nGrid * dq)
#     # vr = torch.linspace(0, (nGrid - 1) * dr, nGrid)
#     vr = torch.linspace(0, rmax, nRGrid)
#     rSpacePot = torch.zeros(nRGrid)
    
#     for ir in range(nRGrid): 
#         if ir==0: 
#             prefactor = 4*np.pi*dq / (8*np.pi**3)
#             rSpacePot[ir] = torch.sum(prefactor * vq**2 * qSpacePot)
#         else: 
#             prefactor = 4*np.pi*dq / (8*np.pi**3 * vr[ir])
#             rSpacePot[ir] = torch.sum(prefactor * vq * torch.sin(vq * vr[ir]) * qSpacePot)

#     return (vr.view(-1,1), rSpacePot.view(-1,1))


def plotBandStruct(bulkSystem_list, bandStruct_list, SHOWPLOTS, func_label="NN prediction"): 
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
        if numKpts != 1: 
            for i in range(numBands): 
                if bulkSystem_list[iSystem].bandWeights[i]!=0:
                    axs_flat[2*iSystem+0].plot(np.arange(numKpts), bandStruct_list[2*iSystem][:, i].detach().numpy(), "bo", alpha=0.5, markersize=2)
                    axs_flat[2*iSystem+1].plot(np.arange(numKpts), bandStruct_list[2*iSystem][:, i].detach().numpy(), "bo", alpha=0.5, markersize=2)
        else: 
            for i in range(numBands): 
                if bulkSystem_list[iSystem].bandWeights[i]!=0:
                    repeat_times = 3
                    axs_flat[2*iSystem+0].plot(np.arange(repeat_times), np.tile(bandStruct_list[2*iSystem][:, i].detach().numpy(), repeat_times), "bo", alpha=0.5, markersize=2)
                    axs_flat[2*iSystem+1].plot(np.arange(repeat_times), np.tile(bandStruct_list[2*iSystem][:, i].detach().numpy(), repeat_times), "bo", alpha=0.5, markersize=2)
        axs_flat[2*iSystem+0].plot([], [], "bo", alpha=0.5, markersize=2, label='Reference')
                
        # plot prediction
        numBands = len(bandStruct_list[2*iSystem+1][0])
        numKpts = len(bandStruct_list[2*iSystem+1])
        for i in range(numBands): 
            if bulkSystem_list[iSystem].bandWeights[i]!=0:
                axs_flat[2*iSystem+0].plot(np.arange(numKpts), np.sort(bandStruct_list[2*iSystem+1].detach().numpy(), axis=1)[:, i], "r-", alpha=0.6)
                axs_flat[2*iSystem+1].plot(np.arange(numKpts), np.sort(bandStruct_list[2*iSystem+1].detach().numpy(), axis=1)[:, i], "r-", alpha=0.6)
        axs_flat[2*iSystem+0].plot([], [], "r-", alpha=0.6, label=func_label)
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
    if numKpts == 1: 
        repeat_times = 3
        refBS = np.tile(refBS, (repeat_times, 1))
        numKpts = repeat_times
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
    if numKpts == 1: 
        repeat_times = 3
        calcBS = np.tile(calcBS, (repeat_times, 1))
        numKpts = repeat_times
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


def plotZungerPP(atomPPOrder, q_grid, v_qs, nRGrid, SHOWPLOTS, labelName="ZungerForm"):
    # ref_vq_atoms and pred_vq_atoms are 2D tensors. Each tensor contains the pseudopotential (either ref or pred)
    # for atoms in the order of atomPPOrder. 
    # ref_labelName and pred_labelName are strings. 
    # lineshape_array has twice the length of atomPPOrder, with: ref_atom1, pred_atom1, ref_atom2, pred_atom2, ... 
    
    # q_grid = q_grid.view(-1).detach().numpy()
    # v_qs = v_qs.view(-1).detach().numpy()

    fig, axs = plt.subplots(1,2, figsize=(9,4))
    
    for iAtom in range(len(atomPPOrder)):
        # Plot v(q)
        vq = v_qs[:, iAtom].clone().detach()
        axs[0].plot(q_grid, vq, label=atomPPOrder[iAtom]+" "+labelName)
        # Compute Fourier transform and plot v(r)
        (r_grid, v_r) = realSpacePot(torch.tensor(q_grid), torch.tensor(vq), nRGrid)
        axs[1].plot(r_grid.view(-1).detach().numpy(), v_r.view(-1).detach().numpy(), label=atomPPOrder[iAtom]+" "+labelName)

    axs[0].set(xlabel=r"$q$", ylabel=r"$v(q)$", xlim=(0,7))
    axs[0].legend(frameon=False)
    axs[1].set(xlabel=r"$r$", ylabel=r"$v(r)$", xlim=(0,8))
    axs[1].legend(frameon=False)
    
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
        axs[0].set(xlabel=r"$q$", ylabel=r"$v(q)$", xlim=(0,9))
        axs[0].legend(frameon=False)
        axs[1].set(xlabel=r"$q$", ylabel=r"$v_{NN}(q) - v_{func}(q)$", xlim=(0,9))
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
        axs[0].set(xlabel=r"$q$", ylabel=r"$v(q)$", xlim=(0,7))
        axs[0].legend(frameon=False)
        axs[1].set(xlabel=r"$r$", ylabel=r"$v(r)$", xlim=(0,8))
        axs[1].legend(frameon=False)
        
    fig.tight_layout()
    if SHOWPLOTS: 
        plt.show()
    return fig

def plotLSD(atom, ref_q, pred_q, ref_vq_atoms, pred_vq_atoms, ref_labelName, pred_labelName, lineshape_array, boolPlotDiff, SHOWPLOTS):
    # ref_vq_atoms and pred_vq_atoms are 2D tensors. Each tensor contains the pseudopotential (either ref or pred)
    # for atoms in the order of atomPPOrder. 
    # ref_labelName and pred_labelName are strings. 
    # lineshape_array has twice the length of atomPPOrder, with: ref_atom1, pred_atom1, ref_atom2, pred_atom2, ... 
    
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    ref_q = ref_q.view(-1).detach().numpy()
    pred_q = pred_q.view(-1).detach().numpy()
    
    ref_vq = ref_vq_atoms.view(-1).detach().numpy()
    pred_vq = pred_vq_atoms.view(-1).detach().numpy()
    axs[0].plot(ref_q, ref_vq, lineshape_array[0], label=atom+" "+ref_labelName)
    axs[0].plot(pred_q, pred_vq, lineshape_array[1], label=atom+" "+pred_labelName)
    axs[1].plot(ref_q, pred_vq - ref_vq, lineshape_array[0], label=atom+" diff (pred - ref)")
    (ref_vr, ref_rSpacePot) = realSpacePot(torch.tensor(ref_q), torch.tensor(ref_vq), 3000)
    (pred_vr, pred_rSpacePot) = realSpacePot(torch.tensor(pred_q), torch.tensor(pred_vq), 3000)
    axs[2].plot(ref_vr.view(-1).detach().numpy(), ref_rSpacePot.view(-1).detach().numpy(), lineshape_array[0], label=atom+" "+ref_labelName)
    axs[2].plot(pred_vr.view(-1).detach().numpy(), pred_rSpacePot.view(-1).detach().numpy(), lineshape_array[1], label=atom+" "+pred_labelName)
    axs[0].set(xlabel=r"$q$", ylabel=r"$v(q)$", xlim=(0,9))
    axs[0].legend(frameon=False)
    axs[1].set(xlabel=r"$q$", ylabel=r"$v_{NN}(q) - v_{func}(q)$", xlim=(0,9))
    axs[1].legend(frameon=False)
    axs[2].set(xlabel=r"$r$", ylabel=r"$v(r)$", xlim=(0,12))
    axs[2].legend(frameon=False)
  
    fig.tight_layout()
    if SHOWPLOTS: 
        plt.show()

    plt.close()
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


def FT_converge_and_write_pp(atomPPOrder, qmax_array, nQGrid_array, nRGrid_array, model, val_dataset, xmin, xmax, ymin, ymax, choiceQMax, choiceNQGrid, choiceNRGrid, ppPlotFilePrefix, potRAtomFilePrefix, SHOWPLOTS, PPparams=None, Rmax=25):
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
        NN = model(qGrid)
        for iAtom in range(len(atomPPOrder)):
            # Add long range term 
            # This is commented out because numerically FT this function is less accurate than using the analytic FT in post-processing
            # lr_coeff = PPparams[atomPPOrder[iAtom]][4]
            # lr_gamma = 0.2
            # lr_pot = long_range_correction(qGrid, lr_gamma, lr_coeff)
            qSpacePot = NN[:, iAtom].view(-1) # + lr_pot
            (vr, rSpacePot) = realSpacePot(qGrid.view(-1), qSpacePot, nRGrid, Rmax)
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
        # lr_coeff = PPparams[atomPPOrder[iAtom]][4]
        # print(f"Atom = {atomPPOrder[iAtom]} lr_coeff = {lr_coeff}")
        # lr_gamma = 0.2
        # lr_pot = long_range_correction(qGrid, lr_gamma, lr_coeff)
        qSpacePot = NN[:, iAtom].view(-1) # + lr_pot
        (vr, rSpacePot) = realSpacePot(choiceQGrid.view(-1), qSpacePot, choiceNRGrid, Rmax)
        pot = torch.cat((vr, rSpacePot), dim=1).detach().numpy()
        potq = torch.cat((choiceQGrid.view(-1,1), NN[:, iAtom].view(-1,1)), dim=1).detach().numpy()
        np.savetxt(potRAtomFilePrefix+"_"+atomPPOrder[iAtom]+".dat", pot, delimiter='    ', fmt='%e')
        np.savetxt(potRAtomFilePrefix+"_q_"+atomPPOrder[iAtom]+".dat", potq, delimiter='    ', fmt='%e')
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