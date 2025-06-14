import numpy as np
import pandas as pd
from collections import defaultdict

from utils.pp_func import plotBandStructFromFile

def k_path(high_sym_points, num_kpoints):
    interpolated_kpoints = []
    for i in range(len(high_sym_points) - 1):
        start = high_sym_points[i]
        end = high_sym_points[i + 1]
        for j in range(num_kpoints[i]):  # Ensure left-inclusive, right-exclusive
            alpha = j / (num_kpoints[i])
            interpolated_kpoints.append((1 - alpha) * start + alpha * end)
    
    interpolated_kpoints.append(high_sym_points[-1])  # Add the final high-symmetry point
    return np.array(interpolated_kpoints)

def read_qe_kpoints(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    kpoints_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("K_POINTS"):
            kpoints_start = i
            break
    if kpoints_start is None:
        raise ValueError("K_POINTS block not found in QE input file.")

    kpoints_type = lines[kpoints_start].strip().split()
    if len(kpoints_type) < 2 or "crystal_b" not in kpoints_type[1]:
        raise ValueError("K_POINTS format is not `crystal_b`.")

    num_kpoints = int(lines[kpoints_start + 1].strip())
    high_symmetry_kpoints = []
    interpolation_counts = []

    for line in lines[kpoints_start + 2:kpoints_start + 2 + num_kpoints]:
        # ignore comments
        if "!" in line: 
            numbers = line.split("!")[0].split()
        else: 
            numbers = line.split()

        k_point = list(map(float, numbers[:3]))  # First three are kx, ky, kz
        high_symmetry_kpoints.append(k_point)
        interpolation_counts.append(int(float(numbers[3])))

    return np.array(high_symmetry_kpoints), interpolation_counts

def convert_BS_energies(bgw_filename, double_bands=True):
    bgw_bs = np.loadtxt(bgw_filename)
    # print(len(bgw_bs))
    
    # Get unique bands from the file
    unique_bands = np.unique(bgw_bs[:, 1].astype(int))
    
    # Extract k-point Cartesian coordinates and band energies
    k_cart = bgw_bs[bgw_bs[:, 1] == unique_bands[0]][:, [2, 3, 4]]  # First three columns (kx, ky, kz)
    band_energies = np.array([bgw_bs[bgw_bs[:, 1] == band][:, 6] for band in unique_bands]).T
    
    if double_bands:
        band_energies = np.repeat(band_energies, 2, axis=1)  # Repeat each column

    # Stack k-points with band energies
    stacked_data = np.hstack((k_cart, band_energies))
    
    # Compute cumulative Cartesian distance
    distances = np.zeros(len(k_cart))
    for i in range(1, len(k_cart)):
        distances[i] = distances[i - 1] + np.linalg.norm(k_cart[i] - k_cart[i - 1])
    
    # Replace first 3 columns with 1 column of cumulative distances
    stacked_data = np.column_stack([distances, band_energies])
    
    return stacked_data


if __name__=='__main__': 
    """
    QE_input_filename = "/global/homes/t/tommylin/BGW_CALCS/Si_defPot/Si_deformed_BS/1b-mf-qe/4-bandstructure/bands.in" 
    high_symmetry_kpoints, num_interpolated_points = read_qe_kpoints(QE_input_filename)

    interpolated_k_path = k_path(high_symmetry_kpoints, num_interpolated_points)

    weights = np.ones((interpolated_k_path.shape[0], 1))
    kpoints_with_weights = np.hstack((interpolated_k_path, weights))
    np.savetxt('CALCS_Si/defPot_tuning_1_inputs/kpoints_1_fromQE.par', kpoints_with_weights, fmt='%.12f   %.12f   %.12f   %.1f')

    fitting_bs = convert_BS_energies('/global/homes/t/tommylin/BGW_CALCS/Si_defPot/Si_deformed_BS/3-bgw-qe_final_converged/2-sigma/bandstructure/bandstructure.dat')
    np.savetxt('CALCS_Si/defPot_tuning_1_inputs/expBandStruct_1_fromBGW.par', fitting_bs, delimiter='  ', fmt='%.5f')

    fig, axs = plotBandStructFromFile('CALCS_Si/defPot_tuning_1_inputs/expBandStruct_1_fromBGW.par', 'CALCS_Si/defPot_tuning_1_inputs/expBandStruct_1_fromBGW.par')
    axs[0].set(ylim=(-10, 20))
    axs[1].set(ylim=(2, 10))
    fig.savefig('CALCS_Si/defPot_tuning_1_inputs/BS_from_BGW.pdf')
    """

    """
    fitting_bs = convert_BS_energies('/global/homes/t/tommylin/BGW_CALCS/Si/1a-silicon/2b-bgw-qe/2-sigma/effMass/bandstructure.dat')
    np.savetxt('CALCS_Si/eff_mass_inputs/effMass_BS_fromBGW.par', fitting_bs, delimiter='  ', fmt='%.9f')

    fig, axs = plotBandStructFromFile('CALCS_Si/eff_mass_inputs/effMass_BS_fromBGW.par', 'CALCS_Si/eff_mass_inputs/effMass_BS_fromBGW.par')
    axs[0].set(ylim=(-10, 20))
    axs[1].set(ylim=(2, 10))
    fig.savefig('CALCS_Si/eff_mass_inputs/effMass_BS_from_BGW.pdf')
    """
    
    """
    QE_input_filename = "/global/homes/t/tommylin/BGW_CALCS/Si_mp1095269/1a-silicon/1b-mf-qe/4-bandstructure/bands.in" 
    high_symmetry_kpoints, num_interpolated_points = read_qe_kpoints(QE_input_filename)

    interpolated_k_path = k_path(high_symmetry_kpoints, num_interpolated_points)

    weights = np.ones((interpolated_k_path.shape[0], 1))
    kpoints_with_weights = np.hstack((interpolated_k_path, weights))
    np.savetxt('CALCS_Si/test_Si24_inputs/kpoints_0_fromQE.par', kpoints_with_weights, fmt='%.12f   %.12f   %.12f   %.1f')
    """

    QE_input_filename = "/global/homes/t/tommylin/BGW_CALCS/Si_I4mmm/1a-silicon/1b-mf-qe/4-bandstructure/bands.in" 
    high_symmetry_kpoints, num_interpolated_points = read_qe_kpoints(QE_input_filename)

    interpolated_k_path = k_path(high_symmetry_kpoints, num_interpolated_points)

    weights = np.ones((interpolated_k_path.shape[0], 1))
    kpoints_with_weights = np.hstack((interpolated_k_path, weights))
    np.savetxt('CALCS_Si/test_I4mmm_inputs/kpoints_0_fromQE.par', kpoints_with_weights, fmt='%.12f   %.12f   %.12f   %.1f')

    fitting_bs = convert_BS_energies('/global/homes/t/tommylin/BGW_CALCS/Si_I4mmm/1a-silicon/2b-bgw-qe/2-sigma/bandstructure/bandstructure.dat')
    np.savetxt('CALCS_Si/test_I4mmm_inputs/expBandStruct_0_fromBGW.par', fitting_bs, delimiter='  ', fmt='%.5f')

    fig, axs = plotBandStructFromFile('CALCS_Si/test_I4mmm_inputs/expBandStruct_0_fromBGW.par', 'CALCS_Si/test_I4mmm_inputs/expBandStruct_0_fromBGW.par')
    axs[0].set(ylim=(-10, 20))
    axs[1].set(ylim=(2, 10))
    fig.savefig('CALCS_Si/test_I4mmm_inputs/BS_from_BGW.pdf')