import torch
import time, os
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt 

from utils.pp_func import plotBandStruct
from utils.read import setAllBulkSystems
from test_overlap import read_npz, overlap_2eigVec

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def rotate_curr_basis(u, v):
    norm = np.sqrt(u**2 + v**2)
    critical_point_pairs = [
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (u/norm, v/norm), (u/norm, -v/norm), 
        (-u/norm, v/norm), (-u/norm, -v/norm)
    ]

    def f(c, d):
        return c * u + d * v

    evaluations = [f(c, d) for c, d in critical_point_pairs]
    max_index = np.argmax(evaluations)
    max_c, max_d = critical_point_pairs[max_index]
    max_value = evaluations[max_index]

    return max_c, max_d, max_value


def init_overlap_matrix(nbands):
    S = np.zeros((nbands, nbands), dtype=np.float64)
    return S


def init_coeff_matrix(nbands):
    C = np.zeros((nbands, nbands), dtype=np.float64)
    return C


def overlapMatrix(kIdx1, kIdx2, results_dir, S, C): 
    evec_list_kIdx1 = read_npz(f"{results_dir}rotated_eigVec_k{kIdx1}.npz")
    evec_list_kIdx2 = read_npz(f"{results_dir}eigVec_k{kIdx2}.npz")
    if len(evec_list_kIdx1) != len(evec_list_kIdx2):
        raise ValueError("Error! kIdx1 and kIdx2 don't have the same number of eigenvectors!!!")
    else: 
        nbands = len(evec_list_kIdx1)

    for i in range(int(nbands/2)):
        for j in range(int(nbands/2)): 
            a_2i = evec_list_kIdx1[2*i]
            a_2i1 = evec_list_kIdx1[2*i+1]
            b_2j = evec_list_kIdx2[2*j]
            b_2j1 = evec_list_kIdx2[2*j+1]

            max_c, max_d, max_value = rotate_curr_basis(overlap_2eigVec(a_2i, b_2j), overlap_2eigVec(a_2i, b_2j1))

            # Update the overlap matrix
            S[2*i,2*j] = max_c * overlap_2eigVec(a_2i, b_2j) + max_d * overlap_2eigVec(a_2i, b_2j1)
            S[2*i,2*j+1] = -max_d * overlap_2eigVec(a_2i, b_2j) + max_c * overlap_2eigVec(a_2i, b_2j1)
            S[2*i+1,2*j] = max_c * overlap_2eigVec(a_2i1, b_2j) + max_d * overlap_2eigVec(a_2i1, b_2j1)
            S[2*i+1,2*j+1] = -max_d * overlap_2eigVec(a_2i1, b_2j) + max_c * overlap_2eigVec(a_2i1, b_2j1)

            # Update the coefficient matrix
            C[2*i,2*j] = max_c
            C[2*i,2*j+1] = max_d
            C[2*i+1,2*j] = max_c
            C[2*i+1,2*j+1] = max_d
            # print(f"The [{2*i}, {2*j}] entry of the overlap matrix S should be {max_value}. As it is now {S[2*i,2*j]}. Please verify: {np.allclose(max_value, S[2*i,2*j], atol=1e-4)}")


    np.savetxt(f"{results_dir}rotate_reorder_results/overlap_kIdx_{kIdx1}_{kIdx2}_CBQuad.dat", S[-6:,-6:], fmt='%.4f')
    np.savetxt(f"{results_dir}rotate_reorder_results/coeff_kIdx_{kIdx1}_{kIdx2}_CBQuad.dat", C[-6:,-6:], fmt='%.4f')
    plotOverlapMatrix(S, f"{results_dir}rotate_reorder_results/plotOverlap_kIdx_{kIdx1}_{kIdx2}.png")
    plotOverlapMatrix(C, f"{results_dir}rotate_reorder_results/plotCoeff_kIdx_{kIdx1}_{kIdx2}.png", cmap='bwr', vmin=-1, vmax=1)

    return S, C


def plotOverlapMatrix(S, plotFilename, cmap='Reds', vmin=0, vmax=1):
    plt.figure(figsize=(10, 10)) 
    plt.imshow(S, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()

    # Adding lighter dashed grid lines
    for x in np.arange(-0.5, 32, 1):
        plt.axhline(x, color='lightgrey', linestyle='--', linewidth=0.5)
        plt.axvline(x, color='lightgrey', linestyle='--', linewidth=0.5)

    # Overlaying solid lines for every second grid line
    for x in np.arange(0, 32, 2):
        plt.axhline(x - 0.5, color='black', linestyle='-', linewidth=2)
        plt.axvline(x - 0.5, color='black', linestyle='-', linewidth=2)

    # Highlighting the CB's, i.e., bottom right 6x6 quadrant
    highlight_start = 25.5
    highlight_end = 31.5
    plt.axhline(highlight_start, color='blue', linestyle='-', linewidth=3)
    plt.axhline(highlight_end, color='blue', linestyle='-', linewidth=3)
    plt.axvline(highlight_start, color='blue', linestyle='-', linewidth=3)
    plt.axvline(highlight_end, color='blue', linestyle='-', linewidth=3)

    # Setting ticks for grid lines and labeling every 4 grids
    tick_positions = np.arange(0, 33, 4)
    plt.xticks(tick_positions - 0.5, labels=np.arange(0, 33, 4))
    plt.yticks(tick_positions - 0.5, labels=np.arange(0, 33, 4))

    plt.savefig(plotFilename, bbox_inches='tight')
    plt.close()


def contractOverlapMatrix_deg2(S, mode='sum'): 
    if S.shape[0] % 2 != 0: 
        raise ValueError("Requesting to contract the overlap matrix by deg=2, but there are odd number of bands! Stop!")
    else:
        deg2_S = np.zeros((int(S.shape[0]/2), int(S.shape[0]/2)), dtype=np.float64)
    
    for i in range(0, S.shape[0], 2):
        for j in range(0, S.shape[1], 2):
            if mode=='sum': 
                deg2_S[int(i/2), int(j/2)] = S[i,j] + S[i+1,j+1] + S[i+1,j] + S[i,j+1]
            elif mode=='max': 
                deg2_S[int(i/2), int(j/2)] = max(S[i,j] + S[i+1,j+1], S[i+1,j] + S[i,j+1])
            elif mode=='totMax': 
                deg2_S[int(i/2), int(j/2)] = max(S[i,j], S[i+1,j+1], S[i+1,j], S[i,j+1])
            elif mode=='rotate_top_left': 
                deg2_S[int(i/2), int(j/2)] = S[i,j]
            else: 
                raise ValueError("We currently only accept deg2mode as 'sum', 'max' or 'totMax'. See the code for details. ")
            
    return deg2_S


def constrainOverlapMatrix(S, max_allowed_distance=999): 
    new_S = np.copy(S)
    constraints = np.zeros((new_S.shape[0], new_S.shape[0]))
    for i in range(new_S.shape[0]):
        for j in range(new_S.shape[0]):
            if abs(i - j) > max_allowed_distance:
                constraints[i, j] = 1
    new_S = new_S - 999999 * constraints
    return new_S


def reorder_bands_rotate(kIdx, nbands, results_dir, deg=2, max_distance=999, deg2mode='sum'): 
    print(f"\nWe are re-ordering the eigenvectors at kIdx = {kIdx}.")
    newOrder = np.arange(nbands)
    # read original eigVecs
    evec_list_currKIdx = read_npz(f"{results_dir}eigVec_k{kIdx}.npz")

    if kIdx==0: 
        np.savez(f"{results_dir}rotated_eigVec_k{kIdx}.npz", *evec_list_currKIdx)
        return newOrder

    S = init_overlap_matrix(nbands)
    C = init_coeff_matrix(nbands)
    S, C = overlapMatrix(kIdx-1, kIdx, results_dir, S, C)

    # Do the linear assignment problem
    S = constrainOverlapMatrix(S, max_distance)
    if deg==2:
        S = contractOverlapMatrix_deg2(S, deg2mode)

    _, col_ind = linear_sum_assignment(S, maximize=True)
    if deg==2: 
        col_ind_2deg = np.zeros(nbands, dtype=col_ind.dtype)
        col_ind_2deg[::2] = col_ind * 2
        col_ind_2deg[1::2] = col_ind * 2 + 1
        col_ind = col_ind_2deg
    print(f"New band order at kIdx: {col_ind}")

    # Use C, calculate the rotated eigVecs
    rotated_evec_list = []
    for i in range(int(nbands/2)): 
        c = C[2*i, col_ind[2*i]]
        d = C[2*i, col_ind[2*i]+1]
        rotated_evec_list.append(c * evec_list_currKIdx[col_ind[2*i]] + d * evec_list_currKIdx[col_ind[2*i+1]])
        rotated_evec_list.append(-d * evec_list_currKIdx[col_ind[2*i]] + c * evec_list_currKIdx[col_ind[2*i+1]])
    np.savez(f"{results_dir}rotated_eigVec_k{kIdx}.npz", *rotated_evec_list)

    return col_ind


def plotBandStruct_reorder(refGWBS, defaultBS, newOrderBS, bandIdx, zoomOut=False): 
    if zoomOut: 
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs_flat = axs.flatten()
    else: 
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        axs_flat = np.array([None, ax])

    # plot ref
    numBands = len(refGWBS[0])
    numKpts = len(refGWBS)
    for i in range(numBands): 
        if i==0: 
            if zoomOut:
                axs_flat[0].plot(np.arange(numKpts), refGWBS[:, i], "b-", alpha=0.1, markersize=2, label="Ref GW, default order")
            axs_flat[1].plot(np.arange(numKpts), refGWBS[:, i], "b-", alpha=0.1, markersize=2, label="Ref GW, default order")
        else: 
            if zoomOut:
                axs_flat[0].plot(np.arange(numKpts), refGWBS[:, i], "b-", alpha=0.1, markersize=2)
            axs_flat[1].plot(np.arange(numKpts), refGWBS[:, i], "b-", alpha=0.1, markersize=2)
            
    # plot default ordering
    numBands = len(defaultBS[0])
    numKpts = len(defaultBS)
    for i in range(numBands): 
        if i==0: 
            if zoomOut:
                axs_flat[0].plot(np.arange(numKpts), defaultBS[:, i], "ro-", alpha=0.1, markersize=2, label="Default order")
            axs_flat[1].plot(np.arange(numKpts), defaultBS[:, i], "ro-", alpha=0.1, markersize=2, label="Default order")
        else: 
            if zoomOut: 
                axs_flat[0].plot(np.arange(numKpts), defaultBS[:, i], "ro-", alpha=0.1, markersize=2)
            axs_flat[1].plot(np.arange(numKpts), defaultBS[:, i], "ro-", alpha=0.1, markersize=2)

    # plot new ordering
    numKpts = len(newOrderBS)
    if zoomOut:
        axs_flat[0].plot(np.arange(numKpts), newOrderBS[:, bandIdx], "ro-", alpha=0.8, markersize=2, label=f"New order, band{bandIdx}")
    axs_flat[1].plot(np.arange(numKpts), newOrderBS[:, bandIdx], "ro-", alpha=0.8, markersize=2, label=f"New order, band{bandIdx}")
    
    if zoomOut:
        axs_flat[0].legend(frameon=False)
        axs_flat[0].set(ylim=(-13, 1))
        axs_flat[0].grid(alpha=0.7)
    else: 
        axs_flat[1].legend(frameon=False)
    axs_flat[1].set(ylim=(min(newOrderBS[:, bandIdx])-0.5, max(newOrderBS[:, bandIdx])+0.5))
    axs_flat[1].grid(alpha=0.7)

    fig.tight_layout()
    return fig


########################## main ##########################
if __name__ == "__main__":
    reorderDeg = 2
    lookAhead = 1

    for testCase in [64, 128, 150]: # [16, 32, 64, 128, 150]:
        for deg2mode in ['rotate_top_left']: # ['sum', 'max', 'totMax', 'rotate_top_left']: 
            input_dir = f"CALCS/CsPbI3_test/inputs_{testCase}kpts/"
            results_dir = f"CALCS/CsPbI3_test/results_{testCase}kpts/"

            refGWBS = np.loadtxt(f"{input_dir}expBandStruct_0.par")
            currentBS = np.loadtxt(f"{results_dir}initZunger_BS_sys0.dat")
            print(f"Test Case of {testCase} kpoints. Algorithm assumes strict degeneracy of {reorderDeg}. \n")

            systems, atomPPOrder, nPseudopot, PPparams, totalParams, localPotParams = setAllBulkSystems(1, input_dir, results_dir)
            sys = systems[0]
            basis = sys.basis().numpy()
            print(f"The first 4 entries of the basis vectors: \n {basis[:4]} \n")
            kpts = sys.kpts.numpy()
            # print(kpts)
            kpts_diff = np.diff(kpts, axis=0)
            print(f"The maximum kpts_diff in any of the three dimensions: {np.max(kpts_diff)}\n")

            if lookAhead==1: 
                currKidx_order = reorder_bands_rotate(0, 32, results_dir, deg=reorderDeg, max_distance=8, deg2mode='sum')
                newBS = currentBS[0,1:]

                for kidx in range(1,testCase):
                    currKidx_order = reorder_bands_rotate(kidx, 32, results_dir, deg=reorderDeg, max_distance=8, deg2mode='sum')

                    newE = currentBS[kidx,1:][currKidx_order]
                    newBS = np.vstack((newBS, newE))
            elif lookAhead==2: 
                raise NotImplementedError("This is an experimental feature. Please stop! ")
                newBS = currentBS[0,1:]
                prev2Kidx_order = np.arange(32)
                prev1Kidx_order = np.arange(32)

                for kidx in range(1,testCase):
                    currKidx_order = reorder_bands_2steps(kidx, 32, prev2Kidx_order, prev1Kidx_order, results_dir, reorderDeg, max_distance=4, prevMultiplier=0.7, diag=diag)
                    newE = currentBS[kidx,1:][currKidx_order]
                    newBS = np.vstack((newBS, newE))
                    prev2Kidx_order = prev1Kidx_order
                    prev1Kidx_order = currKidx_order

            for bandIdx in range(32):
                fig = plotBandStruct_reorder(refGWBS[:,1:], currentBS[:,1:], newBS, bandIdx)
                fig.savefig(f"{results_dir}rotate_reorder_results/newBand_deg{reorderDeg}_{bandIdx}_{deg2mode}.pdf")
                plt.close()

