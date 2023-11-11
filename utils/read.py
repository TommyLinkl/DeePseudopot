import numpy as np
import torch

from constants.constants import *

torch.set_default_dtype(torch.float32)
torch.manual_seed(24)

def read_NNConfigFile(filename):
    config = {}
    with open(filename, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=')
                key = key.strip()
                value = value.strip()
                if key == 'SHOWPLOTS':
                    config[key] = bool(int(value))
                elif key in ['nSystem', 'init_Zunger_num_epochs', 'init_Zunger_plotEvery', 'max_num_epochs', 'plotEvery', 'schedulerStep', 'patience']:
                    config[key] = int(value)
                elif key in ['init_Zunger_optimizer_lr', 'optimizer_lr', 'init_Zunger_scheduler_gamma', 'scheduler_gamma']:
                    config[key] = float(value)
                elif key in ['hiddenLayers']: 
                    config[key] = [int(x) for x in value.split()]
                else:
                    config[key] = value
    return config

class bulkSystem:
    def __init__(self, scale=1.0, unitCellVectors_unscaled=None, atomTypes=None, atomPos_unscaled=None, kpts_recipLatVec=None, expBandStruct=None, nBands=16, maxKE=5):
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
        
        self.kpts = kpts_recipLatVec @ self.getGVectors()
        self.expBandStruct = expBandStruct
        self.nBands = nBands
        self.maxKE = maxKE
        
    def setInputs(self, inputFilename):
        # nBands can be redundant
        maxKE = None
        nBands = None
        try:
            with open(inputFilename, 'r') as file:
                for line in file:
                    parts = line.strip().split('=')
                    if len(parts) == 2:
                        variable_name = parts[0].strip()
                        value = parts[1].strip()
                        if variable_name == 'maxKE':
                            maxKE = int(value)
                        elif variable_name == 'nBands':
                            nBands = int(value)
        except FileNotFoundError:
            print(f"File not found: {inputFilename}")
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")
        self.maxKE = maxKE
        self.nBands = nBands
        
    def setSystem(self, systemFilename):
        # scale, unitCellVectors_unscaled, atomTypes, atomPos
        scale = None
        cell = None
        atomTypes = []
        atomCoords = []
        try:
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
        except FileNotFoundError:
            print(f"File not found: {systemFilename}")
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")
            
        self.scale = scale
        self.unitCellVectors = scale * torch.tensor(cell)
        self.atomTypes = np.array(atomTypes).flatten()
        self.atomPos = torch.tensor(atomCoords) @ self.unitCellVectors
        self.systemName = ''.join(self.atomTypes)
    
    def setKPointsAndWeights(self, kPointsFilename):
        try:
            with open(kPointsFilename, 'r') as file:
                data = np.loadtxt(file)
                kpts = data[:, :3]
                kptWeights = data[:, 3]
                gVectors = self.getGVectors()
                
                self.kpts = torch.tensor(kpts, dtype=torch.float32) @ gVectors
                self.kptWeights = torch.tensor(kptWeights, dtype=torch.float32)
        except FileNotFoundError:
            print(f"File not found: {kPointsFilename}")
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")

    def setExpBS(self, expBSFilename):
        try:
            with open(expBSFilename, 'r') as file:
                self.expBandStruct = torch.tensor(np.loadtxt(file)[:, 1:], dtype=torch.float32)
        except FileNotFoundError:
            print(f"File not found: {expBSFilename}")
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")
            
    def setBandWeights(self, bandWeightsFilename): 
        try:
            with open(bandWeightsFilename, 'r') as file:
                bandWeights = np.loadtxt(file)
                self.bandWeights = torch.tensor(bandWeights, dtype=torch.float32)
                if len(bandWeights) != self.nBands:
                    raise ValueError(f"Invalid number of bands in {bandWeightsFilename}, not equal to nBands input: {self.nBands}")
        except FileNotFoundError:
            print(f"File not found: {kPointsFilename}")
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")

    def getCellVolume(self): 
        return torch.dot(self.unitCellVectors[0], torch.cross(self.unitCellVectors[1], self.unitCellVectors[2]))
    
    def getNAtoms(self):
        return len(self.atomTypes)
    
    def getGVectors(self):
        cellVolume = self.getCellVolume()
        prefactor = 2 * np.pi / cellVolume
        gVector1 = prefactor * torch.cross(self.unitCellVectors[1], self.unitCellVectors[2])
        gVector2 = prefactor * torch.cross(self.unitCellVectors[2], self.unitCellVectors[0])
        gVector3 = prefactor * torch.cross(self.unitCellVectors[0], self.unitCellVectors[1])
        gVectors = torch.cat((gVector1.unsqueeze(0), gVector2.unsqueeze(0), gVector3.unsqueeze(0)), dim=0).to(torch.float32)
        return gVectors
    
    def getNKpts(self): 
        return self.kpts.shape[0]
    
    def basis(self): 
        gVectors = self.getGVectors()
        minGMag = min(torch.norm(gVectors[0]), torch.norm(gVectors[1]), torch.norm(gVectors[2]))
        numMaxBasisVectors = int(np.sqrt(2*self.maxKE) / minGMag)
    
        k = torch.arange(-numMaxBasisVectors, numMaxBasisVectors+1, dtype=torch.float32).repeat((2*numMaxBasisVectors+1)**2)
        j = torch.arange(-numMaxBasisVectors, numMaxBasisVectors+1, dtype=torch.float32).repeat_interleave((2*numMaxBasisVectors+1)).repeat((2*numMaxBasisVectors+1))
        i = torch.arange(-numMaxBasisVectors, numMaxBasisVectors+1, dtype=torch.float32).repeat_interleave((2*numMaxBasisVectors+1)**2)
        allGrid = torch.vstack((i, j, k)).T
        transform = gVectors.T
        allBasisSet = allGrid @ transform
    
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
