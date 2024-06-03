import torch
import time, os
import numpy as np
from scipy.optimize import linear_sum_assignment

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def extrapolate(y1, y2, y3, y4):
    first_derivative = y4 - y3
    second_derivative = y4 - 2 * y3 + y2
    third_derivative = (y4 - 2 * y3 + y2) - (y3 - 2 * y2 + y1)
    
    # more agressive extrapolation than Taylor expansion
    next_point = y4 + 1.3 * first_derivative + 0.5 * second_derivative + (1/6) * third_derivative
    
    return next_point


def reorder_smoothness_deg2_tensors(oldBS, max_band_switch=5):
    """
    One can obtain the newBS using one of the following: 
    for i in range(nkpts):
        newBS[i, :] = oldBS[i, orderTable[i, :]]

    OR

    newBS = oldBS[torch.arange(oldBS.size(0))[:, None], orderTable]

    newBS = oldBS[np.arange(oldBS.shape[0])[:, None], orderTable]

    newBS = [[oldBS[i][j] for j in orderTable[i]] for i in range(len(oldBS))]

    """
    numKpts, numBands = oldBS.shape
    order_table = torch.zeros((numKpts, numBands), dtype=torch.long)
    extrapolated_points = oldBS.clone()
    currBS = oldBS.clone()
    
    for i in range(4):
        order_table[i, :] = torch.arange(numBands, dtype=torch.long)

    for i in range(4, numKpts):
        predicted_points = extrapolate(currBS[i-4], currBS[i-3], currBS[i-2], currBS[i-1])
        extrapolated_points[i, :] = predicted_points
        costMatrix = torch.abs(predicted_points.unsqueeze(0) - oldBS[i].unsqueeze(1))
        costMatrix = costMatrix.T
        
        # Add restriction
        for r in range(numBands):
            for s in range(numBands):
                if abs(r - s) > max_band_switch:
                    costMatrix[r, s] = 999999

        # Two-fold degeneracy
        M = torch.zeros((numBands // 2, numBands // 2), dtype=torch.float64)
        for r in range(numBands // 2):
            for s in range(numBands // 2):
                M[r, s] = max(costMatrix[2*r, 2*s] + costMatrix[2*r+1, 2*s+1], 
                              costMatrix[2*r, 2*s+1] + costMatrix[2*r+1, 2*s])
        
        row_ind, col_ind = linear_sum_assignment(M.cpu().detach().numpy())
        col_ind = torch.tensor(col_ind, dtype=torch.long, device=oldBS.device)
        col_ind_2deg = torch.zeros(numBands, dtype=torch.long, device=oldBS.device)
        col_ind_2deg[::2] = col_ind * 2
        col_ind_2deg[1::2] = col_ind * 2 + 1
        col_ind = col_ind_2deg
        
        order_table[i, :] = col_ind
        currBS[i, :] = oldBS[i, col_ind]

    return order_table, currBS, extrapolated_points

