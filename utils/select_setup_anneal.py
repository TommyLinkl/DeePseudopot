import shutil
import os
import torch
import numpy as np
from .read import read_NNConfigFile, setNN
from .NN_train import write_PP_qSpace

def select_iteration(target_iter, mc_data, threshold=250.0): 
    closest_row = None  # Initialize closest_row as None
    min_diff = float('inf')
    
    # Iterate through the rows of mc_data
    for row in mc_data:
        iter_num = int(row[0])
        new_loss = row[1]
        accept = int(row[2])
        
        if accept == 1 and new_loss < threshold:
            diff = abs(iter_num - target_iter)
            
            # Find the closest iteration
            if diff < min_diff:
                min_diff = diff
                closest_row = row  # Update closest_row with a valid row
    
    if closest_row is None:
        # If no row met the condition, return None
        print("No suitable iteration found.")
        return None
    else:
        print(f"Testing choosing from uniform: (iteration, loss) = ({int(closest_row[0])}, {closest_row[1]})")
        return int(closest_row[0])
    

def mc_select_iterations(mc_data, num_selection=5, threshold=250.0): 
    total_iter = len(mc_data)
    target_iters = np.linspace(1, total_iter-2, num=num_selection+1, dtype=int)[1:]
    # print(target_iters)
    
    wanted_iters = []
    for target_iter in target_iters: 
        trial_iter = select_iteration(target_iter, mc_data, threshold)
        if trial_iter is not None: 
            wanted_iters.append(trial_iter)
    wanted_iters = set(wanted_iters)
    wanted_iters = sorted(list(wanted_iters))
    print(f"We have chosen the following iterations: {wanted_iters}")

    return wanted_iters


if __name__ == "__main__":
    DIR_PREFIX = "CALCS/CsPbBr3_gap/"
    DIR_LIST = []
    # for i in range(1, 121): 
    #         DIR_LIST.append(f"heat{i}")
    DIR_LIST = ['heat_1']
    # print(DIR_LIST)

    num = 2

    for DIR in DIR_LIST: 
        print(f"\n{DIR}")
        try: 
            mc_data = np.loadtxt(f"{DIR_PREFIX}{DIR}_results/final_mc_cost.dat")
            iters = mc_select_iterations(mc_data, num_selection=4, threshold=250.0)

            if (iters is None) or (len(iters)==0): 
                print("None of the trajectories satistfy the constraints. ")
            else:
                for iter in iters: 
                    print(f"anneal{num}_inputs/")
                    shutil.copytree(f"{DIR_PREFIX}{DIR}_inputs/", f"{DIR_PREFIX}anneal{num}_inputs/")
                    shutil.copy(f"{DIR_PREFIX}{DIR}_results/mc_iter_{iter}_CsParams.dat", f"{DIR_PREFIX}anneal{num}_inputs/init_CsParams.par")
                    shutil.copy(f"{DIR_PREFIX}{DIR}_results/mc_iter_{iter}_BrParams.dat", f"{DIR_PREFIX}anneal{num}_inputs/init_BrParams.par")
                    shutil.copy(f"{DIR_PREFIX}{DIR}_results/mc_iter_{iter}_PbParams.dat", f"{DIR_PREFIX}anneal{num}_inputs/init_PbParams.par")
                    shutil.copy(f"{DIR_PREFIX}{DIR}_results/mc_iter_{iter}_PPmodel.pth", f"{DIR_PREFIX}anneal{num}_inputs/init_PPmodel.pth")

                    with open(f"{DIR_PREFIX}anneal{num}_inputs/setup_init_cost.dat", 'w') as f: 
                        f.write(f"{mc_data[iter-1, 4]}")
                    
                    # initialize a calc, set up the NN model
                    inputsFolder = f"{DIR_PREFIX}anneal{num}_inputs/"
                    resultsFolder = f"{DIR_PREFIX}anneal{num}_inputs/"
                    NNConfig = read_NNConfigFile(f"{DIR_PREFIX}anneal{num}_inputs/NN_config.par")
                    atomPPOrder = ['Cs', 'I', 'Pb']   # ['Br', 'Cs', 'Pb']    # 
                    PPmodel = setNN(NNConfig, 3)

                    # read and load the .pth NN from MC or training (with Gaussian)
                    if os.path.exists(inputsFolder + 'init_PPmodel.pth'):
                        PPmodel.load_state_dict(torch.load(inputsFolder + 'init_PPmodel.pth'))
                    write_PP_qSpace(f'{resultsFolder}init_qSpace_pot.dat', PPmodel, atomPPOrder)

                    num += 1

        except FileNotFoundError as e:
            print(f"File is not found: {DIR_PREFIX}{DIR}_results/final_mc_cost.dat. SKIP. ")

    
    