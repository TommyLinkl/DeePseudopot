config = {
    'system_file': 'CALCS_GaP_totalRho/wavefunction_spinor_inputs/system_0.par',
    'input_file':  'CALCS_GaP_totalRho/wavefunction_spinor_inputs/input_0.par',
    'kpoints_file':'CALCS_GaP_totalRho/wavefunction_spinor_inputs/kpoints_0.par',
    'basis_mode': 'self',  
    'inputs_folder': 'CALCS_GaP_totalRho/wavefunction_spinor_inputs/',  
    # 'basis_file': 'CALCS_GaP_totalRho/wavefunction_spinor_results/basisStates_0.dat', 
    'eigvec_pattern': 'CALCS_GaP_totalRho/wavefunction_spinor_results/eigVec_k{idx}.npz',
    'nk': 47,
    'vbm_index': 7,
    'ngrid': [48,48,48],
    'spin_deg': 1.0,
    'spinor_components': 2,          # optional hint; detection is automatic
    'spinor_layout': 'blocked',      # 'blocked' (default) or 'interleaved'
    'output_cube': 'total_density_blocked.cube',
}
