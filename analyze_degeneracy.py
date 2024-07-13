import numpy as np

def analyze_degeneracy(BS_file1, BS_file2):
    arr1 = np.loadtxt(BS_file1)[:, 1:]
    arr2 = np.loadtxt(BS_file2)[:, 1:]
    n_cols = arr1.shape[1]
    
    max_deviation_arr1 = -1.0
    for i in range(0, n_cols, 2):
        max_dev1 = np.max(np.abs(arr1[:, i] - arr1[:, i+1]))
        max_deviation_arr1 = max(max_dev1, max_deviation_arr1)
    print(f"Is bandStruct1 doubly degenerate up to 1e-5 eV? {max_deviation_arr1<1e-5}")
    if max_deviation_arr1>=1e-5: 
        print(f"Max deviation in bandStruct1: {max_deviation_arr1:.5f} eV")

    mean_shift_arr2_max = -1.0
    mean_shift_arr2_min = 99999.0
    deg_split_arr2_max = -1.0
    deg_split_arr2_min = 99999.0
    
    for i in range(0, n_cols, 2):
        energy_mean_arr2 = (arr2[:, i] + arr2[:, i+1])/2
        energy_mean_arr1 = (arr1[:, i] + arr1[:, i+1])/2
        mean_shift = np.abs(energy_mean_arr2 - energy_mean_arr1)
        mean_shift_arr2_max = max(np.max(mean_shift), mean_shift_arr2_max)
        mean_shift_arr2_min = min(np.min(mean_shift), mean_shift_arr2_min)

        deg_split_arr2 = np.abs(arr2[:, i] - arr2[:, i+1])
        deg_split_arr2_max = max(np.max(deg_split_arr2), deg_split_arr2_max)
        deg_split_arr2_min = min(np.min(deg_split_arr2), deg_split_arr2_min)
    print(f"The largest shift of the average energy of pairs of degenerate bands: {mean_shift_arr2_max:.5f} eV")
    print(f"The smallest shift of the average energy of pairs of degenerate bands: {mean_shift_arr2_min:.5f} eV")
    print(f"The largest degeneracy splitting: {deg_split_arr2_max:.5f} eV")
    print(f"The smallest degeneracy splitting: {deg_split_arr2_min:.5f} eV")

    return 

results = analyze_degeneracy('CALCS/CsPbI3_randPertH/results_64kpts/initZunger_BS_sys0.dat', 'CALCS/CsPbI3_randPertH/results_64kpts_pert/initZunger_BS_sys0.dat')