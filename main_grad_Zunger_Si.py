import numpy as np
np.set_printoptions(precision=10)
import torch
torch.set_printoptions(precision=14)
import pathlib
import os, sys
pwd = pathlib.Path(__file__).parent.resolve()

from utils.ham import Hamiltonian
from utils.read import BulkSystem
from utils.constants import AUTOEV
from utils.fit_mc import MonteCarloFit, read_mc_opts
from utils.pp_func import plotBandStruct

def cost_total(allparams, coupling=False):
    # this fn needs to take a torch array of all Zunger params (Cs,I,Pb concatenated)
    # then it rebuilds the PPparams dict to pass to ham1
    for idx, atomType in enumerate(atomPPorder):
        bid = int(idx * 9)
        eid = int((idx+1) * 9)
        PPparams[atomType] = allparams[bid:eid]

    ham1.set_PPparams(PPparams)
    bs = ham1.calcBandStruct()
    if coupling:
        cpl_dict = ham1.calcCouplings(atomgammaidxs=agids)

    cost = 0.0
    if optGaps:
        mse = 0.0
        ctr = 0
        for kidx in range(system1.kpts.shape[0]):
            tmp = 0.0
            for bidx in range(bs.shape[1] - 1):
                if abs(system1.expBandStruct[kidx, bidx+1]) > 1e-15 and abs(system1.expBandStruct[kidx, bidx]) > 1e-15:
                    ctr += 1
                    dgap = bs[kidx, bidx+1] - bs[kidx, bidx]
                    egap = system1.expBandStruct[kidx, bidx+1] - system1.expBandStruct[kidx, bidx]
                    tmp += (dgap - egap)**2 * (bndWeight[bidx] + bndWeight[bidx+1])/2
            mse += tmp * system1.kptWeights[kidx]
        cost += mse / ctr
    else:
        mse = 0.0
        ctr = 0
        for kidx in range(system1.kpts.shape[0]):
            tmp = 0.0
            for bidx in range(bs.shape[1]):
                if abs(system1.expBandStruct[kidx, bidx]) > 1e-15:
                    ctr += 1
                    tmp += (bs[kidx, bidx] - system1.expBandStruct[kidx,bidx])**2 * bndWeight[bidx]
            mse += tmp * system1.kptWeights[kidx]
        cost += mse / ctr

    # now coupling MSE
    if coupling:
        mse = 0
        ctr = 0
        for key, cpl in cpl_dict.items():
            # only compare the couplings that are computed, not necessarily all
            # reference data
            qidx = key[2]
            mse += (cpl - system1.expCouplingBands[key])**2 * system1.qptWeights[qidx]
            ctr += 1

        cost += mse / ctr

    return cost

def cost_memory(allparams, atomPPorder, PPparams, ham1, system1, coupling=False, optGaps=False):
    # this fn needs to take a torch array of all Zunger params (Cs,I,Pb concatenated)
    # then it rebuilds the PPparams dict to pass to ham1
    for idx, atomType in enumerate(atomPPorder):
        bid = int(idx * 9)
        eid = int((idx+1) * 9)
        PPparams[atomType] = allparams[bid:eid]

    ham1.set_PPparams(PPparams)

    grads = torch.zeros(len(allparams))
    totalCost = 0.0
    ctr = 0
    for kidx in range(system1.kpts.shape[0]):
        cost = 0.0
        H = ham1.buildHtot(kidx)
        energies = torch.linalg.eigvalsh(H)
        energies = energies.repeat_interleave(2)
        energiesEV = energies[:system1.nBands] * 27.2114 # convert from Hartree to eV
        if optGaps:
            tmp = 0.0
            for bidx in range(len(energiesEV) - 1):
                if abs(system1.expBandStruct[kidx, bidx+1]) > 1e-15 and abs(system1.expBandStruct[kidx, bidx]) > 1e-15:
                    ctr += 1
                    dgap = energiesEV[bidx+1] - energiesEV[bidx]
                    egap = system1.expBandStruct[kidx, bidx+1] - system1.expBandStruct[kidx, bidx]
                    tmp += (dgap - egap)**2 * (system1.bandWeights[bidx] + system1.bandWeights[bidx+1])/2
            cost += tmp * system1.kptWeights[kidx]
        else:
            tmp = 0.0
            for bidx in range(len(energiesEV)):
                if abs(system1.expBandStruct[kidx, bidx]) > 1e-15:
                    ctr += 1
                    tmp += (energiesEV[bidx] - system1.expBandStruct[kidx,bidx])**2 * system1.bandWeights[bidx]
            cost += tmp * system1.kptWeights[kidx]

        # accumulate gradients per kpt
        cost.backward()
        grads += allparams.grad
        allparams.grad = None
        totalCost += cost

    # now coupling MSE
    if coupling:
        raise NotImplementedError("not yet implemented in memory efficient scheme")
        cpl_dict = ham1.calcCouplings(atomgammaidxs=agids)
        mse = 0
        ctr = 0
        for key, cpl in cpl_dict.items():
            # only compare the couplings that are computed, not necessarily all
            # reference data
            qidx = key[2]
            mse += (cpl - system1.expCouplingBands[key])**2 * system1.qptWeights[qidx]
            ctr += 1

        cost += mse / ctr

    return totalCost / ctr, grads / ctr


def torch_optim(init_params, lr_list, atomPPorder, PPparams, ham1, system1, niter=100, coupling=False, optGaps=False):
    pp = init_params
    assert pp.requires_grad == True
    bestPP = None
    bestCost = 1e9
    for it in range(niter):
        # cost = cost_total(pp)
        # cost, grads = cost_memory(pp)
        cost, grads = cost_memory(pp, atomPPorder, PPparams, ham1, system1, coupling, optGaps)
        
        print(f"iter: {it}, cost= {cost}")
        print(pp, "\n")
        if cost < bestCost:
            bestCost = cost
            bestPP = pp.clone().detach()

        """
        if it % 2 == 0:
            # just update deriv w.r.t. a2 param for each atom, very small steps
            #pp.data[2] = pp.data[2] - 1e-5 * pp.grad[2]
            #pp.data[11] = pp.data[11] - 1e-5 * pp.grad[11]
            #pp.data[20] = pp.data[20] - 1e-5 * pp.grad[20]
            pp.data[2] = pp.data[2] - 5e-5 * grads[2]
            pp.data[11] = pp.data[11] - 5e-5 * grads[11]
            pp.data[20] = pp.data[20] - 5e-5 * grads[20]
        else:
            # just update deriv w.r.t. a3 param for each atom, very small steps
            pp.data[3] = pp.data[3] - 5e-5 * grads[3]
            pp.data[12] = pp.data[12] - 5e-5 * grads[12]
            pp.data[21] = pp.data[21] - 5e-5 * grads[21]
        """

        for i in range(9):
            pp.data[i] = pp.data[i] - lr_list[i] * grads[i]

        pp.grad = None

    return bestPP, bestCost


def main_grad_Zunger(inputsFolder = 'inputs/', resultsFolder = 'results/', nIterations=1000, optGaps=False, lr_list = [1e-4, 5e-5, 5e-5, 5e-5, 0, 0, 0, 0, 0]): 
    device = torch.device("cpu")
    os.makedirs(resultsFolder, exist_ok=True)

    # optimize the gaps between bands? (alternative is abs energies)
    # optGaps = False

    # Set learning rates for a0, a1, a2, a3, ...
    lr_list = [1e-4, 5e-5, 5e-5, 5e-5, 0, 0, 0, 0, 0]

    # read and set up system first system (no coupling)
    system1 = BulkSystem()
    system1.setSystem(f"{inputsFolder}system_0.par")
    system1.setInputs(f"{inputsFolder}input_0.par")
    system1.setKPointsAndWeights(f"{inputsFolder}kpoints_0.par")
    # system1.setQPointsAndWeights(f"{inputsFolder}qpoints_0.par")
    system1.setExpBS(f"{inputsFolder}expBandStruct_0.par")
    # system1.setExpCouplings(f"{inputsFolder}expCoupling_0.par")
    system1.setBandWeights(f"{inputsFolder}bandWeights_0.par")
    atomPPorder = np.unique(system1.atomTypes)
    print(f"atom order: {atomPPorder}")
    bndWeight = system1.bandWeights

    # set up zunger potential
    PPparams = {}
    totalParams = torch.empty(0,9)
    for atomType in atomPPorder:
        file_path = f"{inputsFolder}init_{atomType}Params.par"
        with open(file_path, 'r') as file:
            a = torch.tensor([float(line.strip()) for line in file])
        totalParams = torch.cat((totalParams, a.unsqueeze(0)), dim=0)
        PPparams[atomType] = a
    print(PPparams)

    # which couplings to compute
    agids = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (4,0), (4,1), (4,2)]

    ham1 = Hamiltonian(system1, PPparams, atomPPorder, device, iSystem=0, SObool=False, coupling=False)

    print("\nBeginning basic grad opt\n")
    print(f"Initial params: {PPparams}")
    # allparams = torch.concatenate((PPparams["Cs"], PPparams["I"], PPparams["Pb"]))
    allparams = PPparams["Si"]
    allparams.requires_grad = True
    print(f"initial cost, initial grad: {cost_memory(allparams, atomPPorder, PPparams, ham1, system1)}")

    xopt, costopt = torch_optim(allparams, lr_list, atomPPorder, PPparams, ham1, system1, niter=nIterations, coupling=False, optGaps=optGaps)
    print("\n\n\nDone with basic grad opt\n")
    print(f"Final Si params: {xopt[0:9]}")
    print(f"final cost fn: {costopt}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main_grad_Zunger.py <inputsFolder> <resultsFolder> ")
        sys.exit(1)

    inputsFolder = sys.argv[1]
    resultsFolder = sys.argv[2]
    main_grad_Zunger(inputsFolder, resultsFolder, nIterations=1000, lr_list = [1e-5, 1e-5, 1e-5, 1e-5, 0, 0, 0, 0, 0])

    # [1e-4, 5e-5, 5e-5, 5e-5, 0, 0, 0, 0, 0]
    # [1e-5, 1e-5, 1e-5, 1e-5, 0, 0, 0, 0, 0]
    


