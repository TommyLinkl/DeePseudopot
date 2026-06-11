import torch
import numpy as np
import os
import multiprocessing as mp

from .constants import *
from .nn_models import *
from .local_structure_correction import compute_Pb_descriptors, compute_Cs_descriptors, compute_halide_descriptors, retAllImageDist, cutoff_fc, cutoff_fc_prime

torch.set_default_dtype(torch.float64)

def read_NNConfigFile(filename):
    """
    This read function is able to skip empty lines, 
    able to ignore comments after # sign, 
    and not all keys are required. 
    """

    # Set default values for required keywords
    config = init_critical_NNconfig()
    config['init_Zunger_num_epochs'] = 0
    config['init_LSD_num_epochs'] = 0
    config['init_LSD_force_retrain'] = 0
    config['mc_bool'] = False
    config['max_num_epochs'] = 0
    # Pseudopotential output grids (shared by write_PP_qSpace and
    # FT_converge_and_write_pp so qSpace_pot.dat and final_pot_q match).
    config['Rmax']   = 30.0     # real-space cutoff (Bohr) for realSpacePot
    config['qmax']   = 40.0     # q-space grid extent (a.u.^-1)
    config['nQGrid'] = 4096     # number of q-space grid points
    config['nRGrid'] = 4096     # number of real-space grid points
    # LSD pre-training device control
    config['init_LSD_device'] = 'auto'          # 'auto' | 'cpu' | 'cuda' | 'cuda:N'
    config['init_LSD_parallel_atoms'] = False   # train atom-type nets 1-per-GPU
    config['init_LSD_dtype'] = 'float32'        # training precision; net restored to float64 after
    config['init_LSD_scheduler'] = 'cosine'     # 'cosine' (anneal lr->floor) | 'exponential'
    config['init_LSD_eta_min_frac'] = 1e-3      # cosine lr floor as fraction of init_LSD_optimizer_lr
    config['init_LSD_normalize'] = True         # standardize LSD net inputs (descriptors, q)
    config['descriptor_backend'] = 'handcrafted'  # 'handcrafted' | 'mace' (MACE-MP-0 invariants)
    config['pool_initSO'] = 0
    config['pool_initNL'] = 0
    config['force_retrain'] = 0

    with open(filename, 'r') as file:
        for line in file:
            stripped = line.split('#')[0].strip()   # drop inline AND full-line comments first
            if not stripped or '=' not in stripped: # skip blank and comment-only lines
                continue
            if '=' in stripped:
                key, value = stripped.split('=', 1) # split on first '=' only
                key = key.strip()
                value = value.strip()
                if key in ['SHOWPLOTS', 'separateKptGrad', 'checkpoint', 'SObool', 'cacheSO', 'memory_flag', 'runtime_flag', 'init_Zunger_printGrad', 'init_LSD_force_retrain', 'printGrad', 'mc_bool', 'smooth_reorder', 'eigvec_reorder', 'local_env_corr', 'init_LSD_parallel_atoms', 'init_LSD_normalize']:
                    config[key] = bool(int(value))
                elif key in ['nSystem', 'num_cores', 'num_threads', 'pool_initSO', 'pool_initNL', 'init_Zunger_num_epochs', 'init_Zunger_plotEvery', 'init_LSD_num_epochs', 'init_LSD_plot_every', 'init_LSD_scheduler_step', 'max_num_epochs', 'plotEvery', 'schedulerStep', 'patience', 'perturbEvery', 'mc_iter', 'pre_adjust_moves', 'mc_perturb_mode', 'nQGrid', 'nRGrid']:
                    config[key] = int(value)
                elif key in ['PPmodel_decay_rate', 'PPmodel_decay_center', 'PPmodel_gaussian_std', 'LSDmodel_decay_rate', 'LSDmodel_decay_center', 'LSDmodel_gaussian_std', 'LSDmodel_osc_alpha', 'init_Zunger_optimizer_lr', 'init_LSD_optimizer_lr', 'optimizer_lr', 'LSD_optimizer_lr', 'init_Zunger_scheduler_gamma', 'init_LSD_scheduler_gamma', 'scheduler_gamma', 'LSD_scheduler_gamma', 'sgd_momentum', 'adam_beta1', 'adam_beta2', 'mc_percentage', 'mc_beta', 'pre_adjust_stepSize', 'pre_adjust_LSD_step_size', 'penalize_starting', 'penalize_lambda', 'penalize_mag_threshold', 'penalize_mag_lambda', 'Rmax', 'qmax', 'init_LSD_eta_min_frac']:
                    config[key] = float(value)
                elif key in ['hiddenLayers', 'LSD_hiddenLayers', 'LSD_N_hiddenLayers']: 
                    config[key] = [int(x) for x in value.split()]
                elif key in ['PPmodel_scale']: 
                    config[key] = [float(x) for x in value.split()]
                else:
                    config[key] = value

    # Warning messages to address 1) input conflicts, 2) missing inputs, before running into errors. 
    print("All settings: ")

    if ('PPmodel' not in config) or ('nSystem' not in config) or ('hiddenLayers' not in config): 
        raise ValueError("One or more required parameters are missing: 'PPmodel', 'nSystem', 'hiddenLayers'.")
    
    if (config["checkpoint"]==1) and (config["separateKptGrad"]==1): 
        print("\tWARNING: Both checkpoint and separateKptGrad are turned on. \n")
    elif (config["checkpoint"]==1) and (config["separateKptGrad"]==0):
        print("\tWARNING: Using checkpointing! Please use this as a last resort, only for pseudopotential fitting where memory limit is a major issue. The code will run slower due to checkpointing. \n")
    elif (config["checkpoint"]==0) and (config["separateKptGrad"]==1): 
        print("\tUsing separateKptGrad. This can decrease the peak memory load during the fitting code.")

    if (config['num_cores']==0): 
        print("\tNot doing multiprocessing.")
    else:
        print(f"\tUsing num_cores = {config['num_cores']}, {config['num_cores'] * config['num_threads']} CPUs out of {mp.cpu_count()} total CPUs available.")
        print(f"\tEach (pool) uses {config['num_threads']} threads for lin. alg. multithreading. Beware of oversubscribing compute!")

    if config['memory_flag']: 
        print("\nWARNING: MEMORY_FLAG is ON. Please check to make sure that the script is run with:\n\tmprof run --output <mem_output_file> main.py <inputsFolder> <resultsFolder>\n\tmprof plot -o <mem_plot_file> <mem_output_file>\n")
    print("\nRUNTIME_FLAG is ON") if config['runtime_flag'] else None

    if ('penalize_mag_threshold' in config) and (config['penalize_mag_threshold'] > 100.0):
        print("\tWARNING: 'penalize_mag_threshold' is above 100. Extremely large values can cause numerical instability in the current mag_penalty implementation.\n")

    if config['init_Zunger_num_epochs']>0:
        if ('init_Zunger_plotEvery' not in config) or ('init_Zunger_optimizer_lr' not in config) or ('init_Zunger_scheduler_gamma' not in config): 
            raise ValueError("'init_Zunger_num_epochs'>0. But some required parameters for init_Zunger are missing.")

    if config['mc_bool']: 
        if ('mc_iter' not in config) or ('mc_percentage' not in config) or ('mc_beta' not in config): 
            raise ValueError("Input error: 'mc_iter', 'mc_percentage', and 'mc_beta' must be specified when 'mc_bool' is True.")

    if ('max_num_epochs' in config) and (config['max_num_epochs']>0): 
        if config['mc_bool']: 
            raise ValueError("Both doing Monte Carlo ('mc_bool') and doing NN training ('max_num_epochs'). This combination is invalid. Please use the 'perturbEvery' keyword if training + random perturbation is desired. ")
        config['mc_bool'] = False
        if ('plotEvery' not in config) or ('schedulerStep' not in config) or ('optimizer_lr' not in config) or ('scheduler_gamma' not in config): 
            raise ValueError("Missing required keys when 'max_num_epochs' > 0: 'plotEvery', 'schedulerStep', 'optimizer_lr', 'scheduler_gamma'")
        if ('patience' not in config): 
            config['patience'] = config['max_num_epochs']+1
        if ('perturbEvery' not in config): 
            config['perturbEvery'] = -1
            
    print()
    return config


def init_critical_NNconfig():
    config = {}
    config['runtime_flag'] = False
    config['memory_flag'] = False
    config['checkpoint'] = False
    config['num_cores'] = 0
    config['num_threads'] = 1
    config['SHOWPLOTS'] = False
    config['separateKptGrad'] = True
    config['SObool'] = False
    config['cacheSO'] = True
    config['local_env_corr'] = False

    config['smooth_reorder'] = False
    config['eigvec_reorder'] = False
    return config

def read_ParamSteps(atomPPOrder, paramStepsFilePath):
    paramSteps = {}
    anyFile = 0
    for atomType in atomPPOrder:
        file_path = f"{paramStepsFilePath}{atomType}ParamSteps.par"
        if os.path.isfile(file_path):
            anyFile += 1
            with open(file_path, 'r') as file:
                steps = [float(line.strip()) for line in file]
                assert len(steps) == 9 or len(steps) == 5
                paramSteps[atomType] = steps
    if anyFile == 0:
        paramSteps = None
    elif anyFile != len(atomPPOrder):
        raise ValueError("must supply a paramStep file for each atom type")
    
    return paramSteps

def read_PPparams(atomPPOrder, paramsFilePath): 
    PPparams = {}
    totalParams = torch.empty(0,9, dtype=torch.float64) # see the readme for definition of all 9 params.
                                   # They are not all used in this test. Only
                                   # params 0-3,5-7 are used (local pot, SOC,
                                   # and nonlocal, no long range or strain)
    for atomType in atomPPOrder:
        file_path = f"{paramsFilePath}{atomType}Params.par"
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                a = torch.tensor([float(line.strip()) for line in file], dtype=torch.float64)
            totalParams = torch.cat((totalParams, a.unsqueeze(0)), dim=0)
            PPparams[atomType] = a
        else:
            raise FileNotFoundError("Error: File " + file_path + " cannot be found. This atom cannot be initialized. ")
    return PPparams, totalParams


def read_LSDparams(atomPPOrder, nBasis, paramsFilePath): 
    # nBasis is the number of radial functions chosen for the local-structure dependent correction
    LSDparams = {}
    totalParams = torch.empty(0, nBasis, dtype=torch.float64)
    for atomType in atomPPOrder:
        file_path = f"{paramsFilePath}{atomType}LSDParams.par"
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                a = torch.tensor([float(line.strip()) for line in file], dtype=torch.float64)
            totalParams = torch.cat((totalParams, a.unsqueeze(0)), dim=0)
            LSDparams[atomType] = a
        else:
            raise FileNotFoundError("Error: File " + file_path + " cannot be found. This atom cannot be initialized. ")
    return LSDparams, totalParams

class BulkSystem:
    def __init__(self, scale=1.0, unitCellVectors_unscaled=None, atomTypes=None, atomPos_unscaled=None, kpts_recipLatVec=None, expBandStruct=None, nBands=16, maxKE=5, BS_plot_center=-5.0, BS_plot_CBVB_range=10.0, BS_plot_CBVB_range_zoom=5.0, systemName='No_Name'):
        if unitCellVectors_unscaled is None:
            unitCellVectors_unscaled = torch.zeros(3, 3)
        if atomTypes is None:
            atomTypes = np.array([])
        if atomPos_unscaled is None:
            atomPos_unscaled = torch.zeros(3)
        if kpts_recipLatVec is None:
            kpts_recipLatVec = torch.zeros(3)
        if expBandStruct is None:
            expBandStruct = torch.zeros(0)
            
        self.scale = scale
        self.unitCellVectors = unitCellVectors_unscaled * self.scale
        self.atomTypes = atomTypes
        self.atomPos = atomPos_unscaled @ self.unitCellVectors
        
        #self.kpts = kpts_recipLatVec @ self.getGVectors()
        self.expBandStruct = expBandStruct
        self.kptDistInputs = None
        self.nBands = nBands
        self.maxKE = maxKE
        self.expCouplingBands = None
        self.expCouplingWeights = None
        self.bandWeights = None
        self.expEffMasses = None 
        self.effMassWeights = None
        self.BS_plot_center = BS_plot_center
        self.BS_plot_CBVB_range = BS_plot_CBVB_range
        self.BS_plot_CBVB_range_zoom = BS_plot_CBVB_range_zoom
        self.systemName = systemName
        self.fit_defPot = False
        self.fit_eph = False
        self.fit_eff_masses = False
        self.relE_bIdx = -1

        self.G2 = None
        self.dG2_dR = None
        self.G4 = None
        self.dG4_dR = None
        
    def setInputs(self, inputFilename):
        attributes = {}
        with open(inputFilename, 'r') as file:
            for line in file:
                if '=' in line:
                    key, value = line.split('#')[0].strip().split('=')
                    key = key.strip()
                    value = value.strip()
                    if key in ['maxKE', 'BS_plot_center', 'BS_plot_CBVB_range', 'BS_plot_CBVB_range_zoom']:
                        attributes[key] = float(value)
                    elif key in ['nBands', 'idxVB', 'idxCB', 'idxGap', 'relE_bIdx']:            # nBands can be redundant
                        attributes[key] = int(float(value))
                    elif key in ['fit_defPot']: 
                        attributes[key] = bool(int(value))
                    elif key in ['fit_eph']:
                        attributes[key] = bool(int(value))
                    elif key in ['fit_eff_masses']:
                        attributes[key] = bool(int(value))
                    elif key in ['systemName']: 
                        attributes[key] = value
        vars(self).update(attributes)
        if "idxVB" in attributes:
            self.idx_vb = attributes["idxVB"]
        if "idxCB" in attributes:
            self.idx_cb = attributes["idxCB"]
        if "idxGap" in attributes:
            self.idx_gap = attributes["idxGap"]

        
    def setSystem(self, systemFilename):
        # scale, unitCellVectors_unscaled, atomTypes, atomPos
        scale = None
        cell = None
        atomTypes = []
        atomCoords = []
        with open(systemFilename, 'r') as file:
            section = None
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue  # Skip empty lines
                if parts[0] == 'scale':
                    scale = float(parts[2])
                elif parts[0] == 'cell':
                    section = 'cell'
                    cell = []
                    for _ in range(3):
                        cell_line = next(file).strip()
                        cell.append([float(x) for x in cell_line.split()])
                elif parts[0] == 'atoms':
                    section = 'atoms'
                    atomTypes = [] 
                    atomCoords = []
                elif section == 'atoms':
                    atomTypes.append([parts[0]])
                    atomCoords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
        self.scale = scale
        self.unitCellVectors = scale * torch.tensor(cell, dtype=torch.float64)
        self.atomTypes = np.array(atomTypes).flatten()
        self.atomPos = torch.tensor(atomCoords, dtype=torch.float64) @ self.unitCellVectors
        self.atomPos = self.atomPos.detach().requires_grad_(True)
        # self.systemName = ''.join(self.atomTypes)
        print("UnitCellVectors, scaled (in Bohr): ")
        print(self.unitCellVectors)
        print("atomTypes: ")
        print(self.atomTypes)
        print("AtomPos, scaled (in Bohr): ")
        print(self.atomPos)

    def set_LSDparams(self, atomPPOrder, inputsFolder):
        if self.nLSDBasis == None:
            return
        elif self.nLSDBasis:
            print("Reading in local structure dependent (LSD) corrections.\n")
            self.LSDparams, self.totalLSDParams = read_LSDparams(atomPPOrder, self.nLSDBasis, inputsFolder + "init_")
        return 
    
    def setKPointsAndWeights(self, kPointsFilename):
        with open(kPointsFilename, 'r') as file:
            data = np.loadtxt(file)
            data = np.atleast_2d(data)
            kpts = data[:, :3]
            kptWeights = data[:, 3]
            gVectors = self.getGVectors()
            
            self.kpts = torch.tensor(kpts, dtype=torch.float64) @ gVectors
            self.kptWeights = torch.tensor(kptWeights, dtype=torch.float64)
        
        # Define manual band ordering if the file "kpoints_0_orderMatrix.par" exist
        self.bandOrderMatrix = np.arange(self.nBands)[np.newaxis, :].repeat(self.getNKpts(), axis=0)
        bandOrderFilename = kPointsFilename.split(".")[0] + "_orderMatrix.par"
        if os.path.exists(bandOrderFilename):
            self.bandOrderMatrix = np.loadtxt(bandOrderFilename, dtype=int)
            print(f"NOTICE: We are reading and using the fixed order of bands from the file '{bandOrderFilename}'. ")
        else:
            print(f"Band order input file: xxx_orderMatrix.par not found... Not using manual band order input. ")
        # Ensure bandOrderMatrix is two-dimensional
        if self.bandOrderMatrix.ndim == 1:
            self.bandOrderMatrix = self.bandOrderMatrix[np.newaxis, :]

    

    def setQPointsAndWeights(self, qPointsFilename):
        with open(qPointsFilename, 'r') as file:
            data = np.loadtxt(file)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            print(f"data.shape = {data.shape}")
            qpts = data[:, :3]
            qptWeights = data[:, 3]
            gVectors = self.getGVectors()
            
            self.qpts = torch.tensor(qpts, dtype=torch.float64) @ gVectors
            self.qptWeights = torch.tensor(qptWeights, dtype=torch.float64)
        

    def setExpBS(self, expBSFilename):
        with open(expBSFilename, 'r') as file:
            fileContent = np.atleast_2d(np.loadtxt(file))
            self.expBandStruct = torch.tensor(fileContent[:, 1:], dtype=torch.float64)
            self.kptDistInputs = torch.tensor(fileContent[:, 0], dtype=torch.float64)

    def setExpEffMasses(self, expEffMassesFilename):
        m_eff_dat = np.loadtxt(expEffMassesFilename)
        self.expEffMasses = [m_eff_dat[0], m_eff_dat[1]]
        self.effMassWeight = m_eff_dat[2]

    def setBandWeights(self, bandWeightsFilename): 
        try:
            with open(bandWeightsFilename, 'r') as file:
                bandWeights = np.loadtxt(file)
                self.bandWeights = torch.tensor(bandWeights, dtype=torch.float64)
                if len(bandWeights) != self.nBands:
                    raise ValueError(f"Invalid number of bands in {bandWeightsFilename}, not equal to nBands input: {self.nBands}")
        except FileNotFoundError:
            print(f"File not found: {bandWeightsFilename}")
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")

    def setExpCouplings(self, expCplFilename):
        # Assume the units of the values are eV! See ham.calcCouplings comment for unit details. 
        print(f"Reading reference e-ph coupling data from file. Please make sure that your inputs are in the units of eV/Bohr. ")
        #with open(expCplFilename, 'r') as fread:
        #    self.expCouplingBands = torch.tensor(np.loadtxt(fread)[:, 1:], dtype=torch.float64)

        self.expCouplingBands = {}
        self.expCouplingWeights = {}
        n_qpts = self.qpts.shape[0] if hasattr(self, "qpts") and self.qpts is not None else None
        with open(expCplFilename, 'r') as fread:
            lines = fread.readlines()
            for lidx, line in enumerate(lines):
                if "Atom idx" in line:
                    sp = line.split()
                    atomidx = int(float(sp[3]))
                    begin_block = lidx
                elif "coupling elements" in line:
                    assert lidx == begin_block+1 or lidx == begin_block+8
                    sp = line.split()
                    bandid = sp[0].split("-")[0]
                    gamma = sp[-1]
                elif "polarization" in line:
                    assert lidx in [begin_block+3, begin_block+5, begin_block+10, begin_block+12]
                    gamma = line.split()[-1]

                else:
                    # numerical data read in this block

                    # first work out numerical value of gamma
                    if gamma == 'x' or gamma == 0:
                        gamma = 0
                    elif gamma == 'y' or gamma == 1:
                        gamma = 1
                    elif gamma == 'z' or gamma == 2:
                        gamma = 2
                    else:
                        raise ValueError("unexpected value of gamma")
                    
                    sp = line.split()
                    for qidx in range(len(sp)):
                        self.expCouplingBands[(atomidx, gamma, qidx, bandid)] = float(sp[qidx])
                        

    def setExpDefPot(self, expDefPotFilename, version='v2'):
        if version == 'v1':
            with open(expDefPotFilename, 'r') as fread:
                lines = fread.readlines()
                assert len(lines) == 2
                self.expDefPots = np.array([0.0, 0.0])
                self.expDefPots[0] = float(lines[0]) # VBM
                self.expDefPots[1] = float(lines[1]) # CBM

        elif version == 'v2':
            data = np.loadtxt(expDefPotFilename)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            assert data.shape[1] == 7, "Each row must have exactly 7 columns, corresponding to: kidx_VB(all 0-based index)    bidx_VB    kidx_CB    bidx_CB     latConst_ratio      defPot_gap(eV)    weight"
            assert np.all(data[:, :4] == data[:, :4].astype(int)), "First 4 columns must be integers: kidx_VB(all 0-based index)    bidx_VB    kidx_CB    bidx_CB     latConst_ratio      defPot_gap(eV)    weight"
            
            # Convert the first 4 columns to int to safely use them as indices later.
            data[:, :4] = data[:, :4].astype(int)
            
            self.defPotInfo = data
            # print(self.defPotInfo)


    def getCellVolume(self): 
        return float(torch.dot(self.unitCellVectors[0], torch.linalg.cross(self.unitCellVectors[1], self.unitCellVectors[2])))
    
    def getNAtoms(self):
        return len(self.atomTypes)
    
    def getNAtomTypes(self):
        # this could be generalized if we want the same element in different
        # chemical environments to have different potentials. This should
        # return the number of different potentials we have. This 
        # generalization could also be accomplished by using different labels
        # in the input files, e.g. "Cd1, Cd2".
        return len(np.unique(self.atomTypes))
    
    def getGVectors(self):
        cellVolume = self.getCellVolume()
        prefactor = 2 * np.pi / cellVolume
        # print(f'cellVolume = {cellVolume}')
        # print(f'prefactor = {prefactor}')
        gVector1 = prefactor * torch.linalg.cross(self.unitCellVectors[1], self.unitCellVectors[2])
        gVector2 = prefactor * torch.linalg.cross(self.unitCellVectors[2], self.unitCellVectors[0])
        gVector3 = prefactor * torch.linalg.cross(self.unitCellVectors[0], self.unitCellVectors[1])
        gVectors = torch.cat((gVector1.unsqueeze(0), gVector2.unsqueeze(0), gVector3.unsqueeze(0)), dim=0).to(torch.float64)
        return gVectors
    
    def getNKpts(self): 
        return self.kpts.shape[0]
    
    def getNQpts(self):
        return self.qpts.shape[0]

    def basis(self): 
        gVectors = self.getGVectors()
        minGMag = min(torch.norm(gVectors[0]), torch.norm(gVectors[1]), torch.norm(gVectors[2]))
        numMaxBasisVectors = int(np.sqrt(2*self.maxKE) / minGMag)
    
        k = torch.arange(-numMaxBasisVectors, numMaxBasisVectors+1, dtype=torch.float64).repeat((2*numMaxBasisVectors+1)**2)
        j = torch.arange(-numMaxBasisVectors, numMaxBasisVectors+1, dtype=torch.float64).repeat_interleave((2*numMaxBasisVectors+1)).repeat((2*numMaxBasisVectors+1))
        i = torch.arange(-numMaxBasisVectors, numMaxBasisVectors+1, dtype=torch.float64).repeat_interleave((2*numMaxBasisVectors+1)**2)
        allGrid = torch.vstack((i, j, k)).T
        # transform = gVectors.T
        allBasisSet = allGrid @ gVectors
    
        row_norms = torch.norm(allBasisSet, dim=1)
        condition = (HBAR*0.5*row_norms**2 / MASS < self.maxKE)
        indices = torch.where(condition)[0]
        basisSet = allBasisSet[indices]
        
        sorting_indices = torch.argsort(basisSet[:, 2], stable=True)
        basisSet = basisSet[sorting_indices]
        sorting_indices = torch.argsort(basisSet[:, 1], stable=True)
        basisSet = basisSet[sorting_indices]
        sorting_indices = torch.argsort(basisSet[:, 0], stable=True)
        basisSet = basisSet[sorting_indices]
        row_norms = torch.norm(basisSet, dim=1)
        sorting_indices = torch.argsort(row_norms[:], stable=True)
        sorted_basisSet = basisSet[sorting_indices]
        
        return sorted_basisSet
    
    def print_basisStates(self, basisStateFileName):
        sorted_basisSet = self.basis().numpy()
        norm_column = np.linalg.norm(sorted_basisSet, axis=1, keepdims=True)
        sorted_basisSet = np.hstack((sorted_basisSet, norm_column))
        
        first_column = np.arange(len(sorted_basisSet))[:, np.newaxis]
        sorted_basisSet = np.hstack((first_column, sorted_basisSet))

        np.savetxt(basisStateFileName, sorted_basisSet, fmt=['%d']+['%f']*(sorted_basisSet.shape[1]-1), delimiter='\t')
        return
    
    def compute_descriptors(self, backend='handcrafted'):
        """
        Compute per-atom structural descriptors for all atoms.
        Keeps atomPos as a torch tensor with requires_grad=True throughout
        so that gradients dNN/dR_mu can be computed via autograd.

        backend='handcrafted' (default): the analytic ORTHO_REF descriptors with
        a dN_dR autograd path (used by the band-structure stage).
        backend='mace': frozen MACE-MP-0 per-atom invariant descriptors (D=256);
        NO dN_dR (init_LSD pretraining only).

        Returns
        -------
        descriptors  : dict with keys 'Br'/'I', 'Pb', 'Cs'
                      Each value is an (N_species, n_descriptors) tensor with grad.
        atom_indices : dict with keys 'Br'/'I', 'Pb', 'Cs'
                      Each value is a (N_species,) integer tensor (no grad needed).
        """
        if str(backend).lower() == 'mace':
            from .mace_descriptors import mace_env_descriptors
            descriptors = mace_env_descriptors(self)
            self.env_descriptors = descriptors
            self.n_descr = {k: int(v.shape[1]) for k, v in descriptors.items()}
            atom_indices = {}
            for i, t in enumerate(self.atomTypes):
                atom_indices.setdefault(str(t), []).append(i)
            self.atom_indices = {k: torch.tensor(v, dtype=torch.long)
                                 for k, v in atom_indices.items()}
            self.dN_dR = None   # not provided for MACE; band-structure stage unsupported
            return descriptors, self.atom_indices

        atomPos   = self.atomPos   # already a torch tensor with requires_grad=True
        cell      = self.unitCellVectors
        nAtoms    = atomPos.shape[0]
        atomTypes = self.atomTypes
        material  = self.systemName

        # --- Precompute all image distances in torch ---
        # Build translation vectors T = n0*a + n1*b + n2*c
        ns = torch.arange(-2, 3, dtype=atomPos.dtype)
        n0, n1, n2    = torch.meshgrid(ns, ns, ns, indexing='ij')
        offsets        = torch.stack([n0.ravel(), n1.ravel(), n2.ravel()], dim=1)  # (M, 3)
        translations   = offsets @ cell                                             # (M, 3)

        # dR[i, j, m] = r_j + T_m - r_i,  shape (N, N, M, 3)
        dR_base = atomPos[None, :, :] - atomPos[:, None, :]                        # (N, N, 3)
        dR      = dR_base[:, :, None, :] + translations[None, None, :, :]         # (N, N, M, 3)
        dist    = torch.linalg.norm(dR, dim=-1)                                    # (N, N, M)

        # --- Self-interaction mask (integer indices, no grad needed) ---
        zero_image = torch.where((offsets == 0).all(dim=1))[0].item()

        self_mask = torch.zeros((nAtoms, nAtoms, dist.shape[2]), dtype=torch.bool)
        self_mask[torch.arange(nAtoms), torch.arange(nAtoms), zero_image] = True

        # --- Compute descriptors per species ---
        halide = [t for t in set(atomTypes) if t not in ('Cs', 'Pb')][0]

        descriptors  = {halide: [], 'Pb': [], 'Cs': []}
        atom_indices = {halide: [], 'Pb': [], 'Cs': []}

        for i, atype in enumerate(atomTypes):
            if atype == halide:
                d = compute_halide_descriptors (i, atomTypes, dist, dR, self_mask, material, halide)
                descriptors [halide].append(d)
                atom_indices[halide].append(i)
            elif atype == 'Pb':
                d = compute_Pb_descriptors(i, atomTypes, dist, dR, self_mask, material, halide)
                descriptors ['Pb'].append(d)
                atom_indices['Pb'].append(i)
            elif atype == 'Cs':
                d = compute_Cs_descriptors(i, atomTypes, dist, dR, self_mask, material, halide)
                descriptors ['Cs'].append(d)
                atom_indices['Cs'].append(i)
            
        # Stack into tensors — torch.stack preserves the computation graph
        descriptors  = {k: torch.stack(v)          for k, v in descriptors.items()  if len(v) > 0}
        atom_indices = {k: torch.tensor(v, dtype=torch.long) for k, v in atom_indices.items() if len(v) > 0}

        # for k, v in descriptors.items():
        #     print(f"{k}: {v.shape}  requires_grad={v.requires_grad}\n{v}")

        self.env_descriptors = descriptors
        self.atom_indices    = atom_indices
        self.n_descr         = {halide: 4, 'Pb': 5, 'Cs': 3}

        # Compute derivatives via autodifferentiation
        # dN_dR[k] shape: (N_species, n_descr, N_atoms, 3)
        # i.e. for species k, atom alpha, descriptor d: dN_dR[k][alpha, d, :, :]
        self.dN_dR = {}

        for k, v in descriptors.items():
            N_species, n_descr = v.shape
            dN_dR_k = torch.zeros(N_species, n_descr, nAtoms, 3, dtype=atomPos.dtype)

            for alpha in range(N_species):
                for d in range(n_descr):
                    grad = torch.autograd.grad(
                        outputs    = v[alpha, d],
                        inputs     = atomPos,
                        retain_graph = True,        # keep graph for next iteration
                        create_graph = False        # don't need higher order derivatives
                    )[0]                            # (N_atoms, 3)
                    dN_dR_k[alpha, d] = grad

            # print(f"{k} dN_dR: {dN_dR_k.shape}\n{dN_dR_k}")
            self.dN_dR[k] = dN_dR_k
        return descriptors, atom_indices
    
    # def compute_descriptors(self):
    #     """
    #     Compute per-atom structural descriptors for all atoms.

    #     Returns
    #     -------
    #     descriptors : dict with keys 'Br', 'I', 'Pb', 'Cs'
    #         Each value is an (N_species, n_descriptors) array.
    #     atom_indices : dict with keys 'Br', 'I', 'Pb', 'Cs'
    #         Each value is a list of atom indices for that species.
    #     """
    #     atomPos   = np.asarray(self.atomPos)
    #     cell      = np.asarray(self.unitCellVectors)
    #     nAtoms    = atomPos.shape[0]
    #     atomTypes = self.atomTypes
    #     material  = self.systemName

    #     # --- Precompute all image distances ---
    #     dR, dist = retAllImageDist(atomPos, cell, n_images=2)  # (N,N,M,3), (N,N,M)

    #     # --- Self-interaction mask ---
    #     ns = np.arange(-2, 3)
    #     n0, n1, n2 = np.meshgrid(ns, ns, ns, indexing='ij')
    #     offsets    = np.stack([n0.ravel(), n1.ravel(), n2.ravel()], axis=1)
    #     zero_image = np.where((offsets == 0).all(axis=1))[0][0]

    #     self_mask = np.zeros((nAtoms, nAtoms, dist.shape[2]), dtype=bool)
    #     self_mask[np.arange(nAtoms), np.arange(nAtoms), zero_image] = True

    #     # --- Compute descriptors per species ---
    #     descriptors  = {'Br': [], 'I': [], 'Pb': [], 'Cs': []}
    #     atom_indices = {'Br': [], 'I': [], 'Pb': [], 'Cs': []}

    #     halide = [t for t in set(atomTypes) if t not in ('Cs', 'Pb')][0]  # 'I', 'Br', or 'Cl'

    #     for i, atype in enumerate(atomTypes):
    #         if atype == halide:
    #             d = compute_halide_descriptors (i, atomTypes, dist, dR, self_mask, material, halide)
    #             descriptors [halide].append(d)
    #             atom_indices[halide].append(i)
    #         elif atype == 'Pb':
    #             d = compute_Pb_descriptors(i, atomTypes, dist, dR, self_mask, material, halide)
    #             descriptors ['Pb'].append(d)
    #             atom_indices['Pb'].append(i)
    #         elif atype == 'Cs':
    #             d = compute_Cs_descriptors (i, atomTypes, dist, dR, self_mask, material, halide)
    #             descriptors ['Cs'].append(d)
    #             atom_indices['Cs'].append(i)

    #     descriptors  = {k: torch.tensor(v) for k, v in descriptors.items()}
    #     atom_indices = {k: torch.tensor(v) for k, v in atom_indices.items()}

    #     for k, v in descriptors.items():
    #         print(f"{k}: {v.shape}\n{v}")

    #     self.env_descriptors = descriptors
    #     self.atom_indices = atom_indices
    #     self.n_descr = {"Cs": 3, "Pb": 5, "I": 2}
        
    #     return descriptors, atom_indices
    
    # def compute_G2_and_dG2_dR(self):
    #     """
    #     Computes:
    #       G2[alpha]                = scalar BP descriptor per atom (soft-normalized)
    #       dG2_dR[alpha, mu, gamma] = ∂G2_alpha / ∂R_{mu,gamma}

    #     All NumPy, analytic.
    #     """
    #     atomPos = np.asarray(self.atomPos)              # (N,3)
    #     cell = np.asarray(self.unitCellVectors)
    #     nAtoms = atomPos.shape[0]
    #     atomTypes = self.atomTypes
    #     material = self.systemName

    #     eta = 0.5
    #     Rs = computeEquilDist(atomTypes, material)
    #     # Per-species parameters
    #     Rc_map = {
    #         'CsPbI3':  {'Cs': 8.5, 'Pb': 7.0, 'I':  7.0},
    #         'CsPbBr3': {'Cs': 8.2, 'Pb': 6.5, 'Br': 6.5},
    #         'CsPbCl3': {'Cs': 7.8, 'Pb': 6.0, 'Cl': 6.0},
    #     }

    #     Z_map = {'Cs': 8, 'Pb': 6, 'I':  2, 'Br':  2, 'Cl':  2}
            

    #     Rc_per_atom = np.array([Rc_map[material][t] for t in atomTypes])   # (N,)
    #     Z_per_atom  = np.array([Z_map[t]  for t in atomTypes])    # (N,)

    #     # Minimum image distances (unchanged)
    #     dR, dist = retMinImageDist(atomPos, cell)                  # (N,N,3), (N,N)

    #     # Per-atom cutoff: fc[alpha, beta] uses Rc of alpha
    #     fc  = np.zeros((nAtoms, nAtoms))
    #     fcp = np.zeros((nAtoms, nAtoms))
    #     for alpha in range(nAtoms):
    #         fc [alpha] = cutoff_fc      (dist[alpha], Rc_per_atom[alpha])
    #         fcp[alpha] = cutoff_fc_prime(dist[alpha], Rc_per_atom[alpha])

    #     np.fill_diagonal(fc,  0.0)
    #     np.fill_diagonal(fcp, 0.0)
    #     np.fill_diagonal(Rs,  0.0)

    #     neighbor_mask = (fc > 0) & (dist > 1e-12)       # (N,N) bool
    #     N_neighbors = neighbor_mask.sum(axis=1)          # (N,)  int
    #     N_neighbors = np.maximum(N_neighbors, 1)         # guard against isolated atoms
    #     print(f"N_neighbors = {N_neighbors}")
    #     # Gaussian term
    #     exp_term = np.exp(-eta * (dist - Rs)**2)

    #     # G2
        
    #     S  = np.sum((exp_term - 1.0) * neighbor_mask, axis=1)               # (N,)
        
    #     G2 = S / Z_per_atom # - 1.0                                      # (N,)
        
    #     # gprime unchanged
    #     gprime = (exp_term - 1.0) * fcp - 2.0 * eta * (dist - Rs) * fc * exp_term

    #     # Unit vectors R_ab / |R_ab|
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         e_ab = np.zeros_like(dR)
    #         mask = dist > 1e-12
    #         e_ab[mask] = dR[mask] / dist[mask][:, None]

    #     # Accumulate Derivatives
    #     dG2_dR = np.zeros((nAtoms, nAtoms, 3), dtype=float)
    #     for alpha in range(nAtoms):
    #         for beta in range(nAtoms):
    #             if beta == alpha:
    #                 continue
    #             grad = gprime[alpha, beta] * e_ab[alpha, beta] / Z_per_atom[alpha]

    #             dG2_dR[alpha, alpha] += grad
    #             dG2_dR[alpha, beta]  -= grad

    #         assert np.allclose(np.sum(dG2_dR[alpha], axis=0), 0.0)

    #     self.G2     = torch.from_numpy(G2)
    #     self.dG2_dR = torch.from_numpy(dG2_dR)
    #     return

    def finite_difference_dG2(self, eps=1e-6):
        """
        Finite-difference check of dG2_dR.
        Central difference: (G2(R+eps) - G2(R-eps)) / (2 eps)
        """

        atomPos0 = np.asarray(self.atomPos, dtype=float)
        nAtoms = atomPos0.shape[0]

        # Reference G2
        self.atomPos = atomPos0.copy()
        self.compute_G2_and_dG2_dR()
        G2_ref = self.G2.copy()

        dG2_fd = np.zeros((nAtoms, nAtoms, 3), dtype=float)

        for mu in range(nAtoms):
            for gamma in range(3):
                # print(f"Positive eps dir {gamma}\n")
                # +eps displacement
                atomPos_p = atomPos0.copy()
                atomPos_p[mu, gamma] += eps
                self.atomPos = atomPos_p
                # print(f"atomPos_p = {atomPos_p}")
                self.compute_G2_and_dG2_dR()
                G2_p = self.G2.copy()

                # print(f"Negative eps dir {gamma}\n")
                # -eps displacement
                atomPos_m = atomPos0.copy()
                atomPos_m[mu, gamma] -= eps
                self.atomPos = atomPos_m
                # print(f"atomPos_m = {atomPos_m}")
                self.compute_G2_and_dG2_dR()
                G2_m = self.G2.copy()

                # Central difference
                dG2_fd[:, mu, gamma] = (G2_p - G2_m) / (2.0 * eps)
                

        # Restore original positions
        self.atomPos = atomPos0

        return dG2_fd

def setAllBulkSystems(nSystem, inputsFolder, resultsFolder, LSD_flag=False, descriptor_backend='handcrafted'):
    atomPPOrder = []
    systemsList = [BulkSystem() for _ in range(nSystem)]
    for iSys, sys in enumerate(systemsList):
        sys.setSystem(inputsFolder + "system_%d.par" % iSys)
        sys.setInputs(inputsFolder + "input_%d.par" % iSys)
        sys.setKPointsAndWeights(inputsFolder + "kpoints_%d.par" % iSys)
        if sys.fit_eph:
            sys.setQPointsAndWeights(inputsFolder + "qpoints_%d.par" % iSys)
            sys.setExpCouplings(inputsFolder + "expCoupling_%d.par" % iSys)
        sys.setExpBS(inputsFolder + "expBandStruct_%d.par" % iSys)
        sys.setBandWeights(inputsFolder + "bandWeights_%d.par" % iSys)
        sys.print_basisStates(resultsFolder + "basisStates_%d.dat" % iSys)
        if sys.fit_defPot: 
            sys.setExpDefPot_NEW(inputsFolder + "expDefPot_%d.par" % iSys)
        if sys.fit_eph: 
            sys.setExpCouplings(inputsFolder + "expCoupling_%d.par" % iSys)
            sys.setQPointsAndWeights(inputsFolder + "qpoints_%d.par" % iSys)
        if sys.fit_eff_masses: 
            sys.setExpEffMasses(inputsFolder + "expEffMasses_%d.par" % iSys)
        
        # Check that number of kpoints in expBS matches kpoints from kpoints.par
        if sys.kpts.shape[0] == sys.expBandStruct.shape[0]:
            pass
        else:
            print(f"Mismatch between number of kpts in expBS ({sys.expBandStruct.shape[0]}) and kpoints.par ({sys.kpts.shape[0]})!")
            exit(0)
        
        atomPPOrder.append(sys.atomTypes)
        # Write POSCAR files
        # Get unique atom types in order of first appearance
        unique_atom_types = []
        for atom in sys.atomTypes:
            if atom not in unique_atom_types:
                unique_atom_types.append(atom)

        # Count occurrences of each atom type
        atom_counts = {atype: 0 for atype in unique_atom_types}
        for atom in sys.atomTypes:
            atom_counts[atom] += 1

        # Group atomic positions by type
        sorted_positions = {atype: [] for atype in unique_atom_types}
        for iAtom, atomType in enumerate(sys.atomTypes):
            sorted_positions[atomType].append(sys.atomPos[iAtom])

        # Write POSCAR file
        with open(f"{resultsFolder}{iSys}.POSCAR", "w") as f: 
            f.write(f"{sys.systemName}\n1.0000\n")
            
            # Write unit cell vectors
            for line in range(3): 
                f.write(f"{sys.unitCellVectors[line,0] * AUTOAA}  {sys.unitCellVectors[line,1] * AUTOAA}  {sys.unitCellVectors[line,2] * AUTOAA}\n")
            
            # Write element types and counts
            f.write(" ".join(unique_atom_types) + "\n")
            f.write(" ".join(str(atom_counts[atype]) for atype in unique_atom_types) + "\n")
            
            # Specify coordinate mode (Cartesian or Direct)
            f.write("Cartesian\n")
            
            # Write atomic positions in the correct order
            for atype in unique_atom_types:
                for pos in sorted_positions[atype]:
                    f.write(f"{pos[0] * AUTOAA}  {pos[1] * AUTOAA}  {pos[2] * AUTOAA}\n")
            

    atomPPOrder = np.unique(np.concatenate(atomPPOrder))
    nPseudopot = len(atomPPOrder)
    print(f"There are {nPseudopot} atomic pseudopotentials. They are in the order of: {atomPPOrder}")
    
    PPparams, totalParams = read_PPparams(atomPPOrder, inputsFolder + "init_")
    localPotParams = totalParams[:,:4]

    for iSys, sys in enumerate(systemsList):
        # Read in local structure dependent potential coefficients
        if LSD_flag is True:
            #sys.compute_G2_and_dG2_dR()
            sys.compute_descriptors(backend=descriptor_backend)
            if str(descriptor_backend).lower() != 'mace':
                # Temporarily add to atomPos tensor to detect in-place modification
                sys.atomPos.register_hook(lambda grad: print(f"atomPos grad computed"))
                old_grad_fn = sys.atomPos.grad_fn
                old_version = sys.atomPos._version   # increments on every in-place op
                print(f"atomPos._version before = {old_version}")
            #print(f"\n\nComputing G4!!\n\n")
            #sys.compute_G4_and_dG4_dR()
        else:
            sys.G2 = None
            sys.dG2_dR = None
      
    return systemsList, atomPPOrder, nPseudopot, PPparams, totalParams, localPotParams

def setNN(config, nPseudopot, layers=None):
    if not layers:
        layers = [1] + config['hiddenLayers'] + [nPseudopot]

    if config['PPmodel'] in globals() and callable(globals()[config['PPmodel']]):
        if config['PPmodel'] in ['Net_relu_xavier_decay', 'Net_celu_HeInit_decay']: 
            PPmodel = globals()[config['PPmodel']](layers, decay_rate=config['PPmodel_decay_rate'], decay_center=config['PPmodel_decay_center'])
        elif config['PPmodel'] in ['Net_relu_xavier_decayGaussian', 'Net_relu_xavier_decayGaussian_LSD', 'Net_relu_xavier_BN_decayGaussian', 'Net_relu_xavier_BN_dropout_decayGaussian', 'Net_relu_HeInit_decayGaussian', 'Net_sigmoid_xavier_decayGaussian', 'Net_celu_HeInit_decayGaussian', 'Net_celu_RandInit_decayGaussian']: 
            PPmodel = globals()[config['PPmodel']](layers, gaussian_std=config['PPmodel_gaussian_std'])
        elif config['PPmodel'] in ['Net_celu_HeInit_scale_decayGaussian']: 
            PPmodel = globals()[config['PPmodel']](layers, gaussian_std=config['PPmodel_gaussian_std'], scale=torch.tensor(config['PPmodel_scale']))
        else: 
            PPmodel = globals()[config['PPmodel']](layers)
    else:
        raise ValueError(f"Function {config['PPmodel']} does not exist.")
    return PPmodel

def setNN_LSD(config, layers=None):
    if not layers:
        layers = [3] + config['LSD_hiddenLayers'] + [1]

    if config['LSDmodel'] in globals() and callable(globals()[config['LSDmodel']]):
        if config['LSDmodel'] in ['Net_relu_xavier_decay', 'Net_celu_HeInit_decay_LSD']: 
            LSDmodel = globals()[config['LSDmodel']](layers, decay_rate=config['LSDmodel_decay_rate'], decay_center=config['LSDmodel_decay_center'])
        elif config['LSDmodel'] in ['Net_relu_xavier_decayGaussian', 'Net_relu_xavier_decayGaussian_LSD', 'Net_relu_xavier_BN_decayGaussian', 'Net_relu_xavier_BN_dropout_decayGaussian', 'Net_relu_HeInit_decayGaussian', 'Net_sigmoid_xavier_decayGaussian', 'Net_celu_HeInit_decayGaussian', 'Net_celu_HeInit_decayGaussian_LSD', 'Net_celu_RandInit_decayGaussian']: 
            LSDmodel = globals()[config['LSDmodel']](layers, gaussian_std=config['LSDmodel_gaussian_std'])
        elif config['LSDmodel'] in ['Net_celu_HeInit_scale_decayGaussian']: 
            LSDmodel = globals()[config['LSDmodel']](layers, gaussian_std=config['LSDmodel_gaussian_std'], scale=torch.tensor(config['LSDmodel_scale']))
        elif config['LSDmodel'] in ['Net_osc_HeInit_decayGaussian_LSD']:
            LSDmodel = globals()[config['LSDmodel']](layers, gaussian_std=config['LSDmodel_gaussian_std'], alpha=config['LSDmodel_osc_alpha'])
        elif config['LSDmodel'] in ['Net_osc_HeInit_FiLM_LSD']:
            layers = [1] + config["LSD_hiddenLayers"] + [1]
            N_layers = [1] + config['LSD_N_hiddenLayers'] + [1]
            LSDmodel = globals()[config['LSDmodel']](layers, N_layers, gaussian_std=config['LSDmodel_gaussian_std'], alpha=config['LSDmodel_osc_alpha'])
        elif config['LSDmodel'] in ['Net_celu_HeInit_FiLM_LSD']:
            layers = [1] + config["LSD_hiddenLayers"] + [1]
            N_layers = [1] + config['LSD_N_hiddenLayers'] + [1]
            LSDmodel = globals()[config['LSDmodel']](layers, N_layers, gaussian_std=config['LSDmodel_gaussian_std'])        
        else: 
            LSDmodel = globals()[config['LSDmodel']](layers)
    else:
        raise ValueError(f"Function {config['LSDmodel']} does not exist.")
    return LSDmodel


def computeEquilDist(atomTypes, material):
    Rs_mat = []
    for atom1 in atomTypes:
        row = []
        for atom2 in atomTypes:
            equil_dist = retEquilDist(atom1, atom2, material)
            row.append(equil_dist)
        Rs_mat.append(row)

    return np.array(Rs_mat)
            

def retEquilDist(atom1, atom2, material):
    if ((atom1 == 'I') and (atom2 == 'I')):
        return 8.40493990176 # Bohr I-I distance in CsPbI3
    if ((atom1 == 'Br') and (atom2 == 'Br')):
        return 7.8437163206 # Bohr Br-Br distance in CsPbBr3
    if ((atom1 == 'Cl') and (atom2 == 'Cl')):
        return 7.48961492225 # Bohr Cl-Cl distance in CsPbCl3
    if ((atom1 == 'Pb') and (atom2 == 'I')) or ((atom2 == 'Pb') and (atom1 == 'I')):
        return 5.94319 # Bohr Pb-I distance
    if ((atom1 == 'Pb') and (atom2 == 'Br')) or ((atom2 == 'Pb') and (atom1 == 'Br')):
        return 5.546345 # Bohr Pb-Br distance
    if ((atom1 == 'Pb') and (atom2 == 'Cl')) or ((atom2 == 'Pb') and (atom1 == 'Cl')):
        return 5.2959575 # Bohr Pb-Cl distance
    if ((atom1 == 'I') and (atom2 == 'Cs')) or ((atom2 == 'I') and (atom1 == 'Cs')):
        return 8.40493990176 # Bohr I-Cs distance
    if ((atom1 == 'Br') and (atom2 == 'Cs')) or ((atom2 == 'Br') and (atom1 == 'Cs')):
        return 7.8437163206 # Bohr Br-Cs distance
    if ((atom1 == 'Cl') and (atom2 == 'Cs')) or ((atom2 == 'Cl') and (atom1 == 'Cs')):
        return 7.48961492225 # Bohr Cl-Cs distance
    if ((atom1 == 'Pb') and (atom2 == 'Cs')) or ((atom2 == 'Pb') and (atom1 == 'Cs')):
        if material == 'CsPbI3':
            return 10.293907039 # Bohr Pb-Cs distance in CsPbI3
        if material == 'CsPbBr3':
            return 9.60655133631 # Bohr Pb-Cs distance in CsPbI3
        if material == 'CsPbCl3':
            return 9.17286746472 # Bohr Pb-Cs distance in CsPbI3
    if ((atom1 == 'Pb') and (atom2 == 'Pb')):
        if material == 'CsPbI3':
            return 11.88638 # Bohr Pb-Pb distance in CsPbI3
        if material == 'CsPbBr3':
            return 11.09269 # Bohr Pb-Pb distance in CsPbI3
        if material == 'CsPbCl3':
            return 10.591915 # Bohr Pb-Pb distance in CsPbI3
    if ((atom1 == 'Cs') and (atom2 == 'Cs')):
        if material == 'CsPbI3':
            return 11.88638 # Bohr Cs-Cs distance in CsPbI3
        if material == 'CsPbBr3':
            return 11.09269 # Bohr Cs-Cs distance in CsPbI3
        if material == 'CsPbCl3':
            return 10.591915 # Bohr Cs-Cs distance in CsPbI3
    else:
        print(f"Undefined atom pair in retEquilDist {atom1} - {atom2}. read.py")
        exit()
