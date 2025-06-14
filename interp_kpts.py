import numpy as np
import re

"""
This script interpolates the k-points from a QE `K_POINTS crystal_b` input 
to generate a discrete set of k-points for band structure calculations.

Both input and output k-points are in **crystal coordinates**, meaning 
they are in relative coordinates of the reciprocal lattice vectors.
"""

# QE_input_filename = "/global/homes/t/tommylin/BGW_CALCS/Si_lonsdaleite/1a-silicon/1b-mf-qe/4-bandstructure/bands.in" 
# save_kpts_filename = "/global/homes/t/tommylin/DeePseudopot/CALCS_Si/test_lonsdaleite_inputs/QE_kpoints.dat"

# QE_input_filename = "/global/homes/t/tommylin/DeePseudopot_DFT_CALCS/Si_I4mmm/si.2_bands.in" 
# save_kpts_filename = "CALCS_Si/test_I4mmm_inputs/QE_kpoints.dat"

QE_input_filename = "/global/homes/t/tommylin/DeePseudopot_DFT_CALCS/InGaP_alloy_8atoms/In50_Ga50_P_rand1/alloy.3_bands.in" 
save_kpts_filename = "CALCS_III_V_SOC/alloy_In50Ga50P_rand1_pre_calc_2.3_testSOC_inputs/QE_kpoints.dat"

from convert_bgwBS import k_path, read_qe_kpoints


if __name__=="__main__": 
    high_symmetry_kpoints, num_interpolated_points = read_qe_kpoints(QE_input_filename)
    # print(high_symmetry_kpoints)
    # print(num_interpolated_points)

    interpolated_k_path = k_path(high_symmetry_kpoints, num_interpolated_points)
    # print(interpolated_k_path)
    # print(len(interpolated_k_path))

    k_path_with_weights = np.hstack((interpolated_k_path, np.ones((interpolated_k_path.shape[0], 1))))
    np.savetxt(save_kpts_filename, k_path_with_weights, fmt="%.6f   %.6f   %.6f   %.1f")

