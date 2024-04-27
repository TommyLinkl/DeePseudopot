import torch
import time, os
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt 

from utils.pp_func import plotBandStruct
from utils.read import setAllBulkSystems

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def read_npz(filename):
    loaded_data = np.load(filename)
    array_names = loaded_data.files

    evec_list = []
    for array_name in array_names:
        evec_list.append(loaded_data[array_name])

    return evec_list


def overlap_2eigVec(a, b):
    a_conj = np.conj(a)
    overlap = np.dot(a_conj, b)
    return np.abs(overlap)


def overlapMatrix(kIdx1, kIdx2, deg=1, verbosity=0): 
    evec_list_kIdx1 = read_npz(f"CALCS/CsPbI3_test/results_{testCase}kpts/eigVec_k{kIdx1}.npz")
    evec_list_kIdx2 = read_npz(f"CALCS/CsPbI3_test/results_{testCase}kpts/eigVec_k{kIdx2}.npz")
    if len(evec_list_kIdx1) != len(evec_list_kIdx2):
        raise ValueError("Error! kIdx1 and kIdx2 don't have the same number of eigenvectors!!!")
    else: 
        nbands = len(evec_list_kIdx1)


    if deg==1: 
        M = np.zeros((nbands, nbands), dtype=np.float64)
        for i in range(nbands):
            for j in range(nbands): 
                M[i,j] = overlap_2eigVec(evec_list_kIdx1[i], evec_list_kIdx2[j])
    elif deg==2: 
        if nbands % 2 != 0: 
            raise ValueError("There are odd number of bands, for the degeneracy = 2 case! Stop!")
        M = np.zeros((int(nbands/2), int(nbands/2)), dtype=np.float64)
        for i in range(int(nbands/2)):
            for j in range(int(nbands/2)): 
                M[i,j] = max(overlap_2eigVec(evec_list_kIdx1[2*i], evec_list_kIdx2[2*j]) + overlap_2eigVec(evec_list_kIdx1[2*i+1], evec_list_kIdx2[2*j+1]), 
                             overlap_2eigVec(evec_list_kIdx1[2*i], evec_list_kIdx2[2*j+1]) + overlap_2eigVec(evec_list_kIdx1[2*i+1], evec_list_kIdx2[2*j]))
    else: 
        raise ValueError("Error! We currently only support calculating the overlap matrix for singly- or doubly-degenerate states!")

    if verbosity>0: 
        diff = M - np.eye(M.shape[0])
        if np.allclose(diff, 0, atol=1e-4):
            print("YES! The overlap matrix is effectively identity.")
        else:
            print("NO! The overlap matrix is not close to identity.")

    # np.set_printoptions(formatter={'float': lambda x: "{: .3f}".format(x)})
    # print(M)
    if deg==1:
        np.savetxt(f"CALCS/CsPbI3_test/results_{testCase}kpts/overlap_kIdx_{kIdx1}_{kIdx2}_CBQuad.dat", M[-6:,-6:], fmt='%.4f')

    return M


def reorder_bands(kIdx1, kIdx2, deg=1, max_distance=999, verbosity=0): 
    """
    Reorder bands based on a cost matrix.
    If deg == 1, costMatrix is a square matrix of size nbands * nbands. 
    Each row represents the bands at kIdx1, 
    each column represents the bands at kIdx2.

    if deg == 2, costMatrix is a square matrix of size (nbands/2) * (nbands/2). 
    Each row represents the degenerate subspace at kIdx1, 
    each column represents the degenerate subspace at kIdx2.
    
    Band order at kIdx1 is fixed. 
    Returns: Array containing the new column orders for kIdx2.
    """
    print(f"\nWe are working with kIdx1, kIdx2 = {kIdx1}, {kIdx2}.") if verbosity>0 else None
    costMatrix = overlapMatrix(kIdx1, kIdx2, deg, verbosity)
    if deg==1: 
        nbands = costMatrix.shape[0]
    elif deg==2: 
        nbands = costMatrix.shape[0]*2

    constraints = np.zeros((costMatrix.shape[0], costMatrix.shape[0]))
    for i in range(costMatrix.shape[0]):
        for j in range(costMatrix.shape[0]):
            if abs(i - j) > max_distance:
                constraints[i, j] = 1
    constrained_cost_matrix = costMatrix - 999999 * constraints
    row_ind, col_ind = linear_sum_assignment(-constrained_cost_matrix)
    # row_ind, col_ind = linear_sum_assignment(costMatrix, maximize=True)

    print(f"Max sum value of assignment problem = {costMatrix[row_ind, col_ind].sum()}. This value should be close to nBands = {nbands}.") if verbosity>0 else None
    if deg==2: 
        print(col_ind)
        col_ind_2deg = np.zeros(nbands, dtype=col_ind.dtype)
        col_ind_2deg[::2] = col_ind * 2
        col_ind_2deg[1::2] = col_ind * 2 + 1
        col_ind = col_ind_2deg
        print(col_ind)
    print(f"New band order at kIdx2: {col_ind}") if verbosity>0 else None
    return col_ind


def plotBandStruct_reorder(refGWBS, defaultBS, newOrderBS, bandIdx): 
    fig, axs = plt.subplots(1, 2, figsize=(9, 4 * 1))
    axs_flat = axs.flatten()

    # plot ref
    numBands = len(refGWBS[0])
    numKpts = len(refGWBS)
    for i in range(numBands): 
        if i==0: 
            axs_flat[0].plot(np.arange(numKpts), refGWBS[:, i], "b-", alpha=0.1, markersize=2, label="Ref GW")
            axs_flat[1].plot(np.arange(numKpts), refGWBS[:, i], "b-", alpha=0.1, markersize=2, label="Ref GW")
        else: 
            axs_flat[0].plot(np.arange(numKpts), refGWBS[:, i], "b-", alpha=0.1, markersize=2)
            axs_flat[1].plot(np.arange(numKpts), refGWBS[:, i], "b-", alpha=0.1, markersize=2)
            
    # plot default ordering
    numBands = len(defaultBS[0])
    numKpts = len(defaultBS)
    for i in range(numBands): 
        if i==0: 
            axs_flat[0].plot(np.arange(numKpts), defaultBS[:, i], "ro-", alpha=0.1, markersize=2, label="Default order")
            axs_flat[1].plot(np.arange(numKpts), defaultBS[:, i], "ro-", alpha=0.1, markersize=2, label="Default order")
        else: 
            axs_flat[0].plot(np.arange(numKpts), defaultBS[:, i], "ro-", alpha=0.1, markersize=2)
            axs_flat[1].plot(np.arange(numKpts), defaultBS[:, i], "ro-", alpha=0.1, markersize=2)

    # plot new ordering
    numKpts = len(newOrderBS)
    axs_flat[0].plot(np.arange(numKpts), newOrderBS[:, bandIdx], "ro-", alpha=0.8, markersize=2, label=f"New order, band{bandIdx}")
    axs_flat[1].plot(np.arange(numKpts), newOrderBS[:, bandIdx], "ro-", alpha=0.8, markersize=2, label=f"New order, band{bandIdx}")
            
    axs_flat[0].legend(frameon=False)
    axs_flat[0].set(ylim=(-13, 1))
    axs_flat[1].set(ylim=(min(newOrderBS[:, bandIdx])-0.5, max(newOrderBS[:, bandIdx])+0.5))

    fig.tight_layout()
    return fig



########################## main ##########################
testCase = 128    # 16 64 128 150 
reorderDeg = 1



for i in range(80, 100):
    constrained_cost_matrix = np.loadtxt(f"CALCS/CsPbI3_test/results_{testCase}kpts/overlap_kIdx_{i}_{i+1}_CBQuad.dat")
    row_ind, col_ind = linear_sum_assignment(-constrained_cost_matrix)
    # print(row_ind)
    print(col_ind)


'''
refGWBS = np.loadtxt(f"CALCS/CsPbI3_test/inputs_{testCase}kpts/expBandStruct_0.par")
currentBS = np.loadtxt(f"CALCS/CsPbI3_test/results_{testCase}kpts/epoch_1_BS_sys0.dat")
print(f"Test Case of {testCase} kpoints. Algorithm assumes strict degeneracy of {reorderDeg}. \n")

systems, atomPPOrder, nPseudopot, PPparams, totalParams, localPotParams = setAllBulkSystems(1, f"CALCS/CsPbI3_test/inputs_{testCase}kpts/", f"CALCS/CsPbI3_test/results_{testCase}kpts/")
sys = systems[0]

basis = sys.basis().numpy()
print(f"The first 4 entries of the basis vectors: \n {basis[:4]} \n")
kpts = sys.kpts.numpy()
# print(kpts)
kpts_diff = np.diff(kpts, axis=0)
print(f"The maximum kpts_diff in any of the three dimensions: {np.max(kpts_diff)}\n")

dict_orderSwap = {}
for i in range(testCase-1):
    j = i+1
    newOrder = reorder_bands(i, j, deg=reorderDeg, max_distance=4, verbosity=0)
    dict_orderSwap[f"{i}_to_{j}"] = newOrder

newBS = currentBS[0,1:]
currKidx_order = np.arange(32)
for kidx in range(1,testCase):
    oldE = currentBS[kidx,1:]
    currKidx_order = currKidx_order[dict_orderSwap[f"{kidx-1}_to_{kidx}"]]
    newE = currentBS[kidx,1:][currKidx_order]
    newBS = np.vstack((newBS, newE))
# print(newBS)

for bandIdx in range(32):
    fig = plotBandStruct_reorder(refGWBS[:,1:], currentBS[:,1:], newBS, bandIdx)
    fig.savefig(f"CALCS/CsPbI3_test/results_{testCase}kpts/newBand_deg{reorderDeg}_{bandIdx}.png")
    plt.close()

'''