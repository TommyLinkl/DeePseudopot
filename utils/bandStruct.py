
import torch
import gc
from torch.utils.checkpoint import checkpoint
import numpy as np

from .constants import *
from .pp_func import pot_func

def calcHamiltonianMatrix_GPU(NN_boolean, model, basisStates, atomPos, atomTypes, nAtoms, cellVolume, kVector, atomPPOrder, totalParams, device):
    torch.set_default_dtype(torch.float64)
    '''
    This function is outdated and miss the implementation of 
    SOC and NL parts of the pseudopotential. Please use 
    functions for class Hamiltonian in ham.py instead. 
    '''
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
    
    def compute_atomFF():
        return model(torch.norm(gDiff, dim=2).view(-1, 1))

    for k in range(nAtoms): 
        gDiffDotTau = torch.sum(gDiff * atomPos[k], axis=2)
        structFact = 1/cellVolume * (torch.cos(gDiffDotTau) + 1j*torch.sin(gDiffDotTau))

        thisAtomIndex = np.where(atomTypes[k]==atomPPOrder)[0]
        if len(thisAtomIndex)!=1: 
            raise ValueError("Type of atoms in PP. ")
        thisAtomIndex = thisAtomIndex[0]
        
        if NN_boolean: 
            # atomFF = model(torch.norm(gDiff, dim=2).view(-1, 1))
            atomFF = checkpoint(compute_atomFF)
            atomFF = atomFF[:, thisAtomIndex].view(n, n)
        else: 
            atomFF = pot_func(torch.norm(gDiff, dim=2), totalParams[thisAtomIndex])
        
        HMatrix += atomFF * structFact
        del atomFF
        gc.collect()
        torch.cuda.empty_cache()
    return HMatrix

def calcBandStruct_GPU(NN_boolean, model, bulkSystem, atomPPOrder, totalParams, device):
    '''
    This function is outdated and miss the implementation of 
    SOC and NL parts of the pseudopotential. Please use 
    functions for class Hamiltonian in ham.py instead. 
    '''
    nBands = bulkSystem.nBands
    kpts_coord = bulkSystem.kpts
    nkpt = bulkSystem.getNKpts()
    
    bandStruct = torch.zeros((nkpt, nBands))
    for kpt_index in range(nkpt): 
        # print("\nConstructing H Matrix, before and after: ")
        HamiltonianMatrixAtKpt = calcHamiltonianMatrix_GPU(NN_boolean, model, bulkSystem.basis(), bulkSystem.atomPos, bulkSystem.atomTypes, bulkSystem.getNAtoms(), bulkSystem.getCellVolume(), bulkSystem.kpts[kpt_index], atomPPOrder, totalParams, device)

        # diagonalize the hamiltonian
        # print("\neigvalsh, before and after: ")
        energies = torch.linalg.eigvalsh(HamiltonianMatrixAtKpt)
        
        energiesEV = energies * AUTOEV
        # 2-fold degeneracy due to spin
        final_energies = energiesEV.repeat_interleave(2)[:nBands]
    
        bandStruct[kpt_index] = final_energies
        torch.cuda.empty_cache()

    return bandStruct