import torch
import os
import numpy as np
import matplotlib.pyplot as plt 

from utils.read import setAllBulkSystems
from test_overlap import reorder_bands, reorder_bands_2steps, plotBandStruct_reorder

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

########################## main ##########################
if __name__ == "__main__":
    reorderDeg = 1
    lookAhead = 1
    # deg2mode='max'   # 'sum', 'max', 'totMax'
    testCase = 64

    input_dir = f"CALCS/CsPbI3_randPertH/inputs_{testCase}kpts_pert/"
    results_dir = f"CALCS/CsPbI3_randPertH/results_{testCase}kpts_pert/"

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
        newBS = currentBS[0,1:]
        prevKidx_order = np.arange(32)
        for kidx in range(1,testCase):
            currKidx_order = reorder_bands(kidx, 32, prevKidx_order, results_dir, reorderDeg, max_distance=8)
            newE = currentBS[kidx,1:][currKidx_order]
            newBS = np.vstack((newBS, newE))
            prevKidx_order = currKidx_order
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
        fig.savefig(f"{results_dir}reorder_results/newBand_deg{reorderDeg}_{bandIdx}.pdf")
        plt.close()
