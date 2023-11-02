import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.rcParams['lines.markersize'] = 3
import torch

from constants.constants import *
from utils.pp_func import pot_func, realSpacePot, plotBandStruct

def calcHamiltonianMatrix_GPU(NN_boolean, model, basisStates, atomPos, atomTypes, nAtoms, cellVolume, kVector, atomPPOrder, totalParams, device):
    model.to(device)
    basisStates = basisStates.to(device)
    kVector = kVector.to(device)
    atomPos = atomPos.to(device)
    n = basisStates.shape[0]
    HMatrix = torch.zeros((n, n), dtype=torch.complex128)
    HMatrix = HMatrix.to(device)
    
    # Kinetic energy
    for i in range(n): 
        HMatrix[i,i] += HBAR**2 / (2*MASS) * (torch.norm(basisStates[i] + kVector))**2
        
    # Local potential
    gDiff = torch.stack([basisStates] * (basisStates.shape[0]), dim=1) - basisStates.repeat(basisStates.shape[0], 1, 1)
    
    for k in range(nAtoms): 
        gDiffDotTau = torch.sum(gDiff * atomPos[k], axis=2)
        structFact = 1/cellVolume * (torch.cos(gDiffDotTau) + 1j*torch.sin(gDiffDotTau))

        thisAtomIndex = np.where(atomTypes[k]==atomPPOrder)[0]
        if len(thisAtomIndex)!=1: 
            raise ValueError("Type of atoms in PP. ")
        thisAtomIndex = thisAtomIndex[0]
        
        if NN_boolean: 
            atomFF = model(torch.norm(gDiff, dim=2).view(-1, 1))
            atomFF = atomFF[:, thisAtomIndex].view(n, n)
        else: 
            atomFF = pot_func(torch.norm(gDiff, dim=2), totalParams[thisAtomIndex])
        
        HMatrix += atomFF * structFact
    return HMatrix

def calcBandStruct_GPU(NN_boolean, model, bulkSystem, atomPPOrder, totalParams, device):
    nBands = bulkSystem.nBands
    kpts_coord = bulkSystem.kpts
    nkpt = bulkSystem.getNKpts()
    
    bandStruct = torch.zeros((nkpt, nBands))
    for kpt_index in range(nkpt): 
        HamiltonianMatrixAtKpt = calcHamiltonianMatrix_GPU(NN_boolean, model, bulkSystem.basis(), bulkSystem.atomPos, bulkSystem.atomTypes, bulkSystem.getNAtoms(), bulkSystem.getCellVolume(), bulkSystem.kpts[kpt_index], atomPPOrder, totalParams, device)
        # diagonalize the hamiltonian
        energies = torch.linalg.eigvalsh(HamiltonianMatrixAtKpt)
        
        energiesEV = energies * AUTOEV
        # 2-fold degeneracy due to spin
        final_energies = energiesEV.repeat_interleave(2)[:nBands]
    
        bandStruct[kpt_index] = final_energies

    return bandStruct