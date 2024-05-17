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


def overlapMatrix(kIdx1, kIdx2): 
    evec_list_kIdx1 = read_npz(f"CALCS/CsPbI3_test/results_{testCase}kpts/eigVec_k{kIdx1}.npz")
    evec_list_kIdx2 = read_npz(f"CALCS/CsPbI3_test/results_{testCase}kpts/eigVec_k{kIdx2}.npz")
    if len(evec_list_kIdx1) != len(evec_list_kIdx2):
        raise ValueError("Error! kIdx1 and kIdx2 don't have the same number of eigenvectors!!!")
    else: 
        nbands = len(evec_list_kIdx1)

    M = np.zeros((nbands, nbands), dtype=np.float64)
    for i in range(nbands):
        for j in range(nbands): 
            M[i,j] = overlap_2eigVec(evec_list_kIdx1[i], evec_list_kIdx2[j])
    
    np.savetxt(f"CALCS/CsPbI3_test/results_{testCase}kpts/overlap_kIdx_{kIdx1}_{kIdx2}_CBQuad.dat", M[-6:,-6:], fmt='%.4f')

    return M


def contractM_deg2(M, diag=False): 
    if M.shape[0] % 2 != 0: 
            raise ValueError("There are odd number of bands, for the degeneracy = 2 case! Stop!")
    else:
        deg2_M = np.zeros((int(M.shape[0]/2), int(M.shape[0]/2)), dtype=np.float64)
    
    for i in range(0, M.shape[0], 2):
        for j in range(0, M.shape[1], 2):
            if diag: 
                deg2_M[int(i/2), int(j/2)] = max(M[i,j] + M[i+1,j+1], M[i+1,j] + M[i,j+1])
            else: 
                deg2_M[int(i/2), int(j/2)] = M[i,j] + M[i+1,j+1] + M[i+1,j] + M[i,j+1]
            
    return deg2_M


def constrainOverlapMatrix(M, max_allowed_distance=999): 
    new_M = np.copy(M)
    constraints = np.zeros((new_M.shape[0], new_M.shape[0]))
    for i in range(new_M.shape[0]):
        for j in range(new_M.shape[0]):
            if abs(i - j) > max_allowed_distance:
                constraints[i, j] = 1
    new_M = new_M - 999999 * constraints
    return new_M


def reorder_bands(kIdx, nbands, orderInd_prev, deg=1, max_distance=999, diag=True): 
    print(f"\nWe are re-ordering the eigenvectors at kIdx = {kIdx}.")
    newOrder = np.arange(nbands)
    if kIdx==0: 
        return newOrder
    
    costMatrix = overlapMatrix(kIdx-1, kIdx)[orderInd_prev]
    if deg==2: 
        costMatrix = contractM_deg2(costMatrix, diag)
    costMatrix = constrainOverlapMatrix(costMatrix, max_distance)

    _, col_ind = linear_sum_assignment(costMatrix, maximize=True)
    if deg==2: 
        col_ind_2deg = np.zeros(nbands, dtype=col_ind.dtype)
        col_ind_2deg[::2] = col_ind * 2
        col_ind_2deg[1::2] = col_ind * 2 + 1
        col_ind = col_ind_2deg
    print(f"New band order at kIdx: {col_ind}")
    return col_ind


def reorder_bands_2steps(kIdx, nbands, orderInd_prev2, orderInd_prev1, deg=1, max_distance=999, prevMultiplier=0.5, diag=True): 
    print(f"\nWe are re-ordering the eigenvectors at kIdx = {kIdx}.")
    newOrder = np.arange(nbands)
    if kIdx<=1: 
        return reorder_bands(kIdx, nbands, orderInd_prev1, deg, max_distance)

    if deg==1: 
        costMatrix = overlapMatrix(kIdx-1, kIdx)[orderInd_prev1] + prevMultiplier * overlapMatrix(kIdx-2, kIdx)[orderInd_prev2]
    elif deg==2: 
        costMatrix = contractM_deg2(overlapMatrix(kIdx-1, kIdx)[orderInd_prev1] , diag) + prevMultiplier * contractM_deg2(overlapMatrix(kIdx-2, kIdx)[orderInd_prev2] , diag)

    costMatrix = constrainOverlapMatrix(costMatrix, max_distance)
    _, col_ind = linear_sum_assignment(costMatrix, maximize=True)

    if deg==2: 
        # print(col_ind)
        col_ind_2deg = np.zeros(nbands, dtype=col_ind.dtype)
        col_ind_2deg[::2] = col_ind * 2
        col_ind_2deg[1::2] = col_ind * 2 + 1
        col_ind = col_ind_2deg
        # print(col_ind)
    print(f"New band order at kIdx: {col_ind}")
    return col_ind


def plotBandStruct_reorder(refGWBS, defaultBS, newOrderBS, bandIdx): 
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
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
    axs_flat[0].grid(alpha=0.7)
    axs_flat[1].grid(alpha=0.7)

    fig.tight_layout()
    return fig



########################## main ##########################
testCase = 64    # 16 32 64 128 150 
reorderDeg = 2
lookAhead = 2
diag = False

refGWBS = np.loadtxt(f"CALCS/CsPbI3_test/inputs_{testCase}kpts/expBandStruct_0.par")
currentBS = np.loadtxt(f"CALCS/CsPbI3_test/results_{testCase}kpts/initZunger_BS_sys0.dat")
print(f"Test Case of {testCase} kpoints. Algorithm assumes strict degeneracy of {reorderDeg}. \n")

systems, atomPPOrder, nPseudopot, PPparams, totalParams, localPotParams = setAllBulkSystems(1, f"CALCS/CsPbI3_test/inputs_{testCase}kpts/", f"CALCS/CsPbI3_test/results_{testCase}kpts/")
sys = systems[0]
basis = sys.basis().numpy()
print(f"The first 4 entries of the basis vectors: \n {basis[:4]} \n")
kpts = sys.kpts.numpy()
# print(kpts)
kpts_diff = np.diff(kpts, axis=0)
print(f"The maximum kpts_diff in any of the three dimensions: {np.max(kpts_diff)}\n")

if lookAhead==1: 
    newBS = currentBS[0,1:]
    prevKidx_order = np.arange(32)
    for kidx in range(1,testCase):
        currKidx_order = reorder_bands(kidx, 32, prevKidx_order, reorderDeg, max_distance=4, diag=diag)
        newE = currentBS[kidx,1:][currKidx_order]
        newBS = np.vstack((newBS, newE))
        prevKidx_order = currKidx_order
elif lookAhead==2:
    newBS = currentBS[0,1:]
    prev2Kidx_order = np.arange(32)
    prev1Kidx_order = np.arange(32)

    for kidx in range(1,testCase):
        currKidx_order = reorder_bands_2steps(kidx, 32, prev2Kidx_order, prev1Kidx_order, reorderDeg, max_distance=4, prevMultiplier=0.7, diag=diag)
        newE = currentBS[kidx,1:][currKidx_order]
        newBS = np.vstack((newBS, newE))
        prev2Kidx_order = prev1Kidx_order
        prev1Kidx_order = currKidx_order

for bandIdx in range(32):
    fig = plotBandStruct_reorder(refGWBS[:,1:], currentBS[:,1:], newBS, bandIdx)
    fig.savefig(f"CALCS/CsPbI3_test/results_{testCase}kpts/newBand_deg{reorderDeg}_{bandIdx}.png")
    plt.close()

