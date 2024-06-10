import torch
import numpy as np
import os
import multiprocessing as mp

from .constants import *
from .nn_models import *

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
    config['mc_bool'] = False
    config['max_num_epochs'] = 0

    with open(filename, 'r') as file:
        for line in file:
            if not line.strip():              # Skip empty lines
                continue
            if '=' in line:
                key, value = line.split('#')[0].strip().split('=')
                key = key.strip()
                value = value.strip()
                if key in ['SHOWPLOTS', 'separateKptGrad', 'checkpoint', 'SObool', 'memory_flag', 'runtime_flag', 'init_Zunger_printGrad', 'printGrad', 'mc_bool', 'smooth_reorder', 'eigvec_reorder']:
                    config[key] = bool(int(value))
                elif key in ['nSystem', 'num_cores', 'init_Zunger_num_epochs', 'init_Zunger_plotEvery', 'max_num_epochs', 'plotEvery', 'schedulerStep', 'patience', 'perturbEvery', 'mc_iter', 'pre_adjust_moves']:
                    config[key] = int(value)
                elif key in ['PPmodel_decay_rate', 'PPmodel_decay_center', 'PPmodel_gaussian_std', 'init_Zunger_optimizer_lr', 'optimizer_lr', 'init_Zunger_scheduler_gamma', 'scheduler_gamma', 'sgd_momentum', 'adam_beta1', 'adam_beta2', 'mc_percentage', 'mc_beta']:
                    config[key] = float(value)
                elif key in ['hiddenLayers']: 
                    config[key] = [int(x) for x in value.split()]
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
        print(f"\tUsing num_cores = {config['num_cores']} parallelization out of {mp.cpu_count()} total CPUs available.")

    if config['memory_flag']: 
        print("\nWARNING: MEMORY_FLAG is ON. Please check to make sure that the script is run with:\n\tmprof run --output <mem_output_file> main.py <inputsFolder> <resultsFolder>\n\tmprof plot -o <mem_plot_file> <mem_output_file>\n")
    print("\nRUNTIME_FLAG is ON") if config['runtime_flag'] else None

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
    config['SHOWPLOTS'] = False
    config['separateKptGrad'] = True
    config['SObool'] = False

    config['smooth_reorder'] = False
    config['eigvec_reorder'] = False
    return config


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
        self.bandWeights = None
        self.BS_plot_center = BS_plot_center
        self.BS_plot_CBVB_range = BS_plot_CBVB_range
        self.BS_plot_CBVB_range_zoom = BS_plot_CBVB_range_zoom
        self.systemName = systemName
        
        
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
                    elif key in ['nBands', 'idxVB', 'idxCB', 'idxGap']:            # nBands can be redundant
                        attributes[key] = int(float(value))
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
        # self.systemName = ''.join(self.atomTypes)
        
    
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
            print(f"The file '{bandOrderFilename}' does not exist. Not using manual band order input. ")

    

    def setQPointsAndWeights(self, qPointsFilename):
        with open(qPointsFilename, 'r') as file:
            data = np.loadtxt(file)
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
        #with open(expCplFilename, 'r') as fread:
        #    self.expCouplingBands = torch.tensor(np.loadtxt(fread)[:, 1:], dtype=torch.float64)

        self.expCouplingBands = {}
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


    def setExpDefPot(self, expDefPotFilename):
        with open(expDefPotFilename, 'r') as fread:
            lines = fread.readlines()
            assert len(lines) == 2
            self.expDefPots = np.array([0.0, 0.0])
            self.expDefPots[0] = float(lines[0]) # VBM
            self.expDefPots[1] = float(lines[1]) # CBM


    def getCellVolume(self): 
        return float(torch.dot(self.unitCellVectors[0], torch.cross(self.unitCellVectors[1], self.unitCellVectors[2])))
    
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
        gVector1 = prefactor * torch.cross(self.unitCellVectors[1], self.unitCellVectors[2])
        gVector2 = prefactor * torch.cross(self.unitCellVectors[2], self.unitCellVectors[0])
        gVector3 = prefactor * torch.cross(self.unitCellVectors[0], self.unitCellVectors[1])
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

def setAllBulkSystems(nSystem, inputsFolder, resultsFolder):
    atomPPOrder = []
    systemsList = [BulkSystem() for _ in range(nSystem)]
    for iSys, sys in enumerate(systemsList):
        sys.setSystem(inputsFolder + "system_%d.par" % iSys)
        sys.setInputs(inputsFolder + "input_%d.par" % iSys)
        sys.setKPointsAndWeights(inputsFolder + "kpoints_%d.par" % iSys)
        sys.setExpBS(inputsFolder + "expBandStruct_%d.par" % iSys)
        sys.setBandWeights(inputsFolder + "bandWeights_%d.par" % iSys)
        sys.print_basisStates(resultsFolder + "basisStates_%d.dat" % iSys)
        atomPPOrder.append(sys.atomTypes)
    atomPPOrder = np.unique(np.concatenate(atomPPOrder))
    nPseudopot = len(atomPPOrder)
    print(f"There are {nPseudopot} atomic pseudopotentials. They are in the order of: {atomPPOrder}")
    
    PPparams, totalParams = read_PPparams(atomPPOrder, inputsFolder + "init_")
    localPotParams = totalParams[:,:4]
    return systemsList, atomPPOrder, nPseudopot, PPparams, totalParams, localPotParams

def setNN(config, nPseudopot):
    layers = [1] + config['hiddenLayers'] + [nPseudopot]
    if config['PPmodel'] in globals() and callable(globals()[config['PPmodel']]):
        if config['PPmodel']=='Net_relu_xavier_decay': 
            PPmodel = globals()[config['PPmodel']](layers, decay_rate=config['PPmodel_decay_rate'], decay_center=config['PPmodel_decay_center'])
        elif config['PPmodel'] in ['Net_relu_xavier_decayGaussian', 'Net_relu_xavier_BN_decayGaussian', 'Net_relu_xavier_BN_dropout_decayGaussian', 'Net_relu_HeInit_decayGaussian', 'Net_sigmoid_xavier_decayGaussian', 'Net_celu_HeInit_decayGaussian', 'Net_celu_RandInit_decayGaussian']: 
            PPmodel = globals()[config['PPmodel']](layers, gaussian_std=config['PPmodel_gaussian_std'])
        else: 
            PPmodel = globals()[config['PPmodel']](layers)
    else:
        raise ValueError(f"Function {config['PPmodel']} does not exist.")
    return PPmodel

