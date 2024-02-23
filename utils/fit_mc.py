import numpy as np
import torch.linalg
import time
import os

from .constants import *

torch.set_default_dtype(torch.float64)

class MonteCarloFit:

    def __init__(self, ham,
                 writeDir,
                 betas=[100, 1],
                 stepsAtTemp=[5000, 500],
                 tempStepSizeMod=[1.0, 1.0],
                 paramSteps=None,
                 totalIter=100000,
                 fitDefPot=False,
                 fitCoupling=False,
                 fitEffMass=False,
                 optGaps=False,
                 defPotWeight=1.0,
                 couplingOpts=None):
        
        """
        This class runs Monte Carlo optimization on non-neural net 
        pseudopotential params. It is currently only written for a single 
        Hamiltonian (a single bandstructure), but generalization should be easy.

        It uses a quasi-parallel-tempering algorithm, where it switches
        between a high and low temperature to find different local minima and
        then explore them.

        writeDir specifies which directory we write output files in. If None,
        we don't write any intermediate files.
        betas is a list of TWO inverse temperatures to run the quasi-parallel-tempering
        monte carlo algorithm at.
        stepsAtTemp is a list of TWO integers specifying the maximum number
        of steps to take at each temperature before switching to the other
        temp. If you want to only run at a single temp, just set one of these to
        0.
        tempStepSizeMod is a list of TWO floats, which specify a global scalar
        that multiplies the random parameter steps when the temperature is
        at the respecitve "0" or "1" value.
        paramSteps allows to supply custom stepsizes for all PPparams. If input,
        it should be formatted as a dict exactly the same as Hamiltonian.PPparams
        is formatted. If not supplied, a default choice is used, see updatePPparams().
        If optGaps is True, the cost function we try to minimize uses the
        gaps between bands, rather than their absolute energies.
        If fitDefPot, then the MSE of the defpots will be multiplied by defWeight.
        couplingOpts should ba a kwarg dictionary that can be passed to
        ham.calcCouplings(**couplingOpts).
        """        
        
        self.ham = ham

        self.betas = betas
        self.stepsAtTemp = stepsAtTemp
        self.tempStepSizeMod = tempStepSizeMod
        self.paramSteps = paramSteps
        self.totalIter = totalIter

        self.fitDefPot = fitDefPot
        self.fitCoupling = fitCoupling
        self.fitEffMass = fitEffMass
        self.optGaps = optGaps
        self.defPotWeight = defPotWeight
        if couplingOpts is None:
            self.couplingOpts = {}
        else:
            self.couplingOpts = couplingOpts
        if fitEffMass is True:
            raise RuntimeError("We don't actually support this right now. should be simple to add in.")

        if writeDir is None:
            print("!! WARNING: you are running Monte Carlo but have not supplied a dir for output files!")
        self.writeDir = writeDir

        self.bestMSE = 0.0
        self.currentMSE = 0.0
        self.newMSE = 0.0
        self.firstIter = True

        self.kpts = ham.system.kpts
        self.kptWeights = ham.system.kptWeights
        self.qpts = ham.system.qpts
        self.qptWeights = ham.system.qptWeights

        self.expBS = ham.system.expBandStruct
        self.bndWeight = ham.system.bandWeights
        if self.bndWeight is None:
            self.bndWeight = torch.ones(ham.system.nBands)
        if fitCoupling:
            self.expCpl = ham.system.expCouplingBands
            #self.cplBndWeight = ham.system.couplingBandWeights
            self.idx_gap = ham.system.idxGap
            self.idx_vb = ham.system.idxVB
            self.idx_cb = ham.system.idxCB
        if fitDefPot:
            self.expDef = ham.system.expDefPots

        #self.run_mc(totalIter)



    def run_mc(self):

        # calculate initial band structure, ?defpot, ?coupling and 
        # write them to files
        bs = self.ham.calcBandStruct()
        self.writeBands(bs)
        if self.fitDefPot:
            defpots = self.calcDefPots(bs[self.idx_gap, self.idx_vb], bs[self.idx_gap, self.idx_cb])
        else:
            defpots = None
        if self.fitCoupling:
            cpl_dict = self.ham.calcCouplings(**self.couplingOpts)
            self.writeCoupling(cpl_dict)
        else:
            cpl_dict = None
        if self.fitEffMass:
            pass

        self.currentMSE = self.__evalCostFn(bs, defpots, cpl_dict)
        self.bestMSE = self.currentMSE
        
        self.writeIteration(1.0)

        nAccept = 0
        nIter = 0
        stepsAtTemp = 0
        tempIdx = 0
        bestAtTemp = 1e6
        sinceLastAccept = 0
        bestPP = None  # to store the globally optimal PPparams

        t0 = time.time()
        for _ in range(self.totalIter):
            sinceLastAccept += 1
            nIter += 1
            if sinceLastAccept > 10000:
                print("no accepted moves for a while...quitting")
                break

            # modify params step
            tmpPP = self.ham.get_PPparams()
            self.updatePPparams(tempIdx)

            # check cost fn
            bs = self.ham.calcBandStruct()
            if self.fitDefPot:
                defpots = self.calcDefPots(bs[self.idx_gap, self.idx_vb], bs[self.idx_gap, self.idx_cb])
            if self.fitCoupling:
                cpl_dict = self.ham.calcCouplings(**self.couplingOpts)

            self.newMSE = self.__evalCostFn(bs, defpots, cpl_dict)

            # check if we need to change temperature
            stepsAtTemp += 1
            if self.newMSE < bestAtTemp:
                bestAtTemp = self.newMSE
                #stepsAtTemp = 0 # this means stepsAtTemp is the number of steps WITHOUT IMPROVEMENT
            if tempIdx == 0 and stepsAtTemp >= self.stepsAtTemp[0]:
                tempIdx = 1
                stepsAtTemp = 0
                bestAtTemp = 1e6
            if tempIdx == 1 and stepsAtTemp >= self.stepsAtTemp[1]:
                tempIdx = 0
                stepsAtTemp = 0
                bestAtTemp = 0
            
            mc_rand = np.exp(-1*self.betas[tempIdx] * (np.sqrt(self.newMSE) - np.sqrt(self.currentMSE)))
            mc_bool = mc_rand > np.random.uniform(low=0.0, high=1.0)

            if self.newMSE < self.bestMSE or mc_bool:
                # update acceptance stats
                nAccept += 1
            
            # write iteration once per n iterations, before we revert the PPparams
            if nIter % 1 == 0: self.writeIteration(nAccept/nIter)

            # update PPparams?
            if self.newMSE < self.bestMSE:
                self.bestMSE = self.currentMSE = self.newMSE
                # in this case we want to retain updated PPparams
                bestPP = self.ham.get_PPparams()
                self.saveParams(bestPP)
                self.writeBands(bs, stub="/bestBandStruct_0.dat")
                sinceLastAccept = 0
                if self.fitCoupling:
                    self.writeCoupling(cpl_dict, stub="/bestCoupling_0.dat")
            elif mc_bool:
                # new MSE is higher, but we still accept.
                # keep self.ham.PPparams in the updated form
                self.currentMSE = self.newMSE
            else:
                # do not accept new params, revert to most recently accepted vals
                self.ham.set_PPparams(tmpPP)

        tf = time.time()
        print(f"\n\n\nDone fitting. Total iters = {nIter}. Total wall time = {tf-t0}")
        print(f"best MSE = {self.bestMSE}")
        print("Best PPparams (unformatted) = ")
        print(bestPP)
        print("\nAlso writing bestPPparams to files...")
        self.writeBestPPparams(bestPP)






    def __evalCostFn(self, bs, defpots, cpl_dict):
        """
        This just modularizes the calls to evaluate the total 
        cost function. You can play around with this to tailor the optimization.
        """
        if self.optGaps:
            cost = self.calcIndivMSEgaps(bs)
        else:
            cost = self.calcIndivMSE(bs)
        if self.ham.SObool:
            cost += self.calcNonLocalWeighting()
        if self.fitEffMass:
            #cost += self.calcEffMassMSE(bs)
            pass
        if self.fitDefPot:
            cost += self.calcDefPotMSE(defpots)
        if self.fitCoupling:
            cost += self.calcCouplingMSE(cpl_dict)
        
        return cost


    def calcIndivMSE(self, bs):
        # this can be sped up with array operations rather than for loops
        mse = 0.0
        ctr = 0
        for kidx in range(self.kpts.shape[0]):
            tmp = 0.0
            for bidx in range(bs.shape[1]):
                if abs(self.expBS[kidx, bidx]) > 1e-15:
                    ctr += 1
                    tmp += (bs[kidx, bidx] - self.expBS[kidx,bidx])**2 * self.bndWeight[bidx]
            mse += tmp * self.kptWeights[kidx]
        return mse / ctr 

    def calcIndivMSEgaps(self, bs):
        # this can be sped up with array operations rather than for loops
        mse = 0.0
        ctr = 0
        for kidx in range(self.kpts.shape[0]):
            tmp = 0.0
            for bidx in range(bs.shape[1] - 1):
                if abs(self.expBS[kidx, bidx+1]) > 1e-15 and abs(self.expBS[kidx, bidx]) > 1e-15:
                    ctr += 1
                    dgap = bs[kidx, bidx+1] - bs[kidx, bidx]
                    egap = self.expBS[kidx, bidx+1] - self.expBS[kidx, bidx]
                    tmp += (dgap - egap)**2 * (self.bndWeight[bidx] + self.bndWeight[bidx+1])/2
                    # ^^ should we normalize this by expected gap value?
                    # so if two bands are close in the ref data, deviations are
                    # measured on a relative scale. this is different than how the
                    # abs energies work, but it would make this function more sensistive
                    # and it would more equally weight the different bands.
                    # PROBLEMS WOULD HAPPEN FOR DEGENERATE BANDS!
            mse += tmp * self.kptWeights[kidx]
        return mse / ctr
    
    def calcNonLocalWeighting(self):
        """
        This incurs a simple penalty if the SOC factor is too large
        """
        weight = 0
        for atom, params in self.ham.PPparams.items(): 
            if params[5] > 8.0: weight += (params[5] - 8) * 10
        
        return weight
    
    def calcDefPotMSE(self, defpots):
        """
        expecting input to be [vbmDefPot, cbmDefPot]
        """
        return self.defPotWeight * ((defpots[0] - self.expDef[0])**2 + (defpots[1] - self.expDef[1])**2)
    
    def calcCouplingMSE(self, cpl_dict):
        mse = 0
        count = 0
        for key, cpl in cpl_dict.items():
            # only compare the couplings that are computed, not necessarily all
            # reference data
            qidx = key[2]
            mse += (cpl - self.expCpl[key])**2 * self.qptWeights[qidx]
            count += 1
        
        return mse / count


    def updatePPparams(self, tempIdx):
        """
        Randomly modify self.ham.PPparams. This is done IN-PLACE!!
        If supplied to the class constructor, custom step sizes 
        of each parameter will be used. Otherwise, the scale will be
        1/100 * parameter magnitude for each param, which is a decent
        paradigm.
        """
        steps = {}
        if self.paramSteps is not None:
            steps = self.paramSteps
        else:
            for atom, params in self.ham.PPparams.items():
                steps[atom] = params / 100.0

        for atom, params in self.ham.PPparams.items():
            for j in range(len(params)):
                self.ham.PPparams[atom][j] += np.random.uniform(low=-1.0,high=1.0) * steps[atom][j] * self.tempStepSizeMod[tempIdx]

        self.enforceParamConstraints()
        return

    
    def enforceParamConstraints(self):
        """
        This enforces that the sum of all the long-range potentials sum
        to 0 at q=0, which is rigorously required for a charge neutral system.
        The implementation for multiple band structures is tricky, and not
        yet resolved.

        It also forces the SOC constant to be positive, which is physical.
        """
        
        # to make multiple bandstructures work, we need to implement a global inspection of all the different bandstructure systems to see which have
        # common sets of atoms. Then figure out which atoms are simultaneously constrained across all bandstructures (i.e. same atom 
        # type(s) have to be constrained across all band structures).

        ctr = {}
        constrainLbl = None
        for i in range(self.ham.system.getNAtoms()):
            atom = self.ham.system.atomTypes[i]
            self.ham.PPparams[atom][5] = abs(self.ham.PPparams[atom][5]) # SOC must be positive
            # count the number of each atom type
            if atom in ctr:
                ctr[atom] += 1
            else:
                ctr[atom] = 1
                # find which atom to constrain for long-range sum.
                # Ignore atoms with LR param = 0
                if abs(self.ham.PPparams[atom][4]) > 1e-10:
                    constrainLbl = atom
            
        if constrainLbl is None:
            # there are no long-range potentials, nothing more to do
            return
        sumLR = 0.0
        for i in range(self.ham.system.getNAtoms()):
            # add LR param for all atoms that are not constrained
            atom = self.ham.system.atomTypes[i]
            if atom != constrainLbl:
                sumLR += self.ham.PPparams[atom][4] 

        cparam = -1 * sumLR / ctr[constrainLbl]  # this makes the total sum to 0
        self.ham.PPparams[constrainLbl][4] = cparam

        return
            

    def calcDefPots(self, vbm_reg, cbm_reg, verbosity=2):
        """
        This is a simple function to compute the deformation potential by
        finite difference. It takes the NON-deformed vbm energy and cbm energy
        in units of EV as arguments to avoid redundant computation.
        """
        defscale = 1.01 # 1% expansion is not converged, but matches DFT defpot literature
        Hdef = self.ham.buildHtot_def(scale=defscale, verbosity=verbosity)
        evals = torch.linalg.eigvalsh(Hdef) * AUTOEV

        diffVBM = vbm_reg - evals[self.idx_vb]
        diffCBM = cbm_reg - evals[self.idx_cb]
        vol = self.ham.system.getCellVolume()
        defpotVBM = diffVBM / (vol - defscale**3 * vol) * 0.5 * (vol + defscale**3*vol)
        defpotCBM = diffCBM / (vol - defscale**3 * vol) * 0.5 * (vol + defscale**3*vol)

        if verbosity >= 1:
            print(f"VB deformation potential = {defpotVBM}")
            print(f"CB deformation potential = {defpotCBM}")

        return [defpotVBM, defpotCBM]
    

    def writeBands(self, bs, stub="/bandStruct_0.dat"):
        with open(self.writeDir + stub, 'w') as fwrite:
            pathlength = 0.0
            for i in range(bs.shape[0]):
                if i > 0:
                    diff = torch.sqrt(torch.sum((self.kpts[i] - self.kpts[i-1])**2))
                    pathlength += diff
                print(f"{pathlength:.4f}  ", file=fwrite, end="")
                for j in range(bs.shape[1]):
                    print(f"{bs[i,j]:.8f} ", file=fwrite, end="")
                print("\n", file=fwrite, end="")


    def writeCoupling(self, cpl_dict, stub="/couplingBands_0.dat"):
        """
        This function takes the couplings as they are output from ham.calcCouplings()
        and writes them in a labelled data file. 
        """
        with open(self.writeDir + stub, 'w') as fwrite:
            for atomidx in range(self.ham.system.getNAtoms()):
                print(f"Atom idx = {atomidx}   atom = {self.ham.system.atomTypes[atomidx]}   position = {self.ham.system.atomPos[atomidx]}", file=fwrite)

                for band in ["vb", "cb"]:
                    print(f"{band}-{band} coupling elements. ", file=fwrite, end="")
                    for gamma in range(3):
                        if gamma == 0:
                            print("polarization of derivative = x", file=fwrite)
                        elif gamma == 1:
                            print("polarization of derivative = y", file=fwrite)
                        else:
                            print("polarization of derivative = z", file=fwrite)
                        
                        for qidx in range(self.qpts.shape[0]):
                            if (atomidx, gamma, qidx, band) in cpl_dict:
                                print(f"{cpl_dict[(atomidx, gamma, qidx, band)]:.6e}   ", file=fwrite, end="")
                            else:
                                print("Not-fit   ", file=fwrite, end="")
                        print("\n", file=fwrite, end="")
                    print("\n", file=fwrite, end="")
                print("\n\n", file=fwrite, end="")


    def writeIteration(self, perAccept, stub="/iterations.dat"):
        if self.firstIter is True:
            if os.path.isfile(self.writeDir + stub):
                # don't append to an existing file, remove existing file
                os.remove(self.writeDir + stub)

        with open(self.writeDir + stub, 'a') as fwrite:
            if self.firstIter is True:
                print("newMSE \t curMSE \t bstMSE \t %Accpt \t ", file=fwrite, end="")
                for atom, params in self.ham.PPparams.items():
                    for j in range(len(params)):
                        print(f"{atom}_a{j}  \t", file=fwrite, end="")
                print("\n",file=fwrite, end="")

            print(f"{self.newMSE:.6g}\t {self.currentMSE:.6g}\t {self.bestMSE:.6g}\t {perAccept:.5g}\t ", file=fwrite, end="")
            for atom, params in self.ham.PPparams.items():
                for j in range(len(params)):
                    print(f"{params[j]:.14f}\t", file=fwrite, end="")
            print("\n", file=fwrite, end="")
        
        if self.firstIter:
            self.firstIter = False


    def saveParams(self, params, stub="bestParams.pt"):
        for atom, lst in params.items():
            torch.save(lst, self.writeDir + f"/{atom}" + stub)


    def writeBestPPparams(self, param_dict):
        for atom, params in param_dict.items():
            with open(self.writeDir + f"/best_{atom}Params.dat", 'w') as fwrite:
                for j in range(len(params)):
                    print(f"{params[j]:.14f}", file=fwrite)
                



def read_mc_opts(filename):
    """
    Helper function to read monte carlo options from a file
    and return them as a dict, which can be passed to the
    MonteCarloFit constructor with **kwargs.
    """
    mc_opts = {}
    with open(filename, 'r') as fread:
        lines = fread.readlines()
        for line in lines:
            if " = " not in line:
                raise RuntimeError("each line must contain an equals sign and spaces between every distinct word/symbol/number")
            if "," in line:
                raise RuntimeError("don't put commas in between numbers")
            
            if "betas" in line:
                sp = line.split()
                mc_opts[sp[0]] = [float(sp[2]), float(sp[3])]
            elif "stepsAtTemp" in line:
                sp = line.split()
                mc_opts[sp[0]] = [int(float(sp[2])), int(float(sp[3]))]
            elif "tempStepSizeMod" in line:
                sp = line.split()
                mc_opts[sp[0]] = [float(sp[2]), float(sp[3])]
            elif "paramSteps" in line:
                print("!WARNING! paramSteps should not be sepcified in MC input file")
            elif "totalIter" in line:
                sp = line.split()
                mc_opts[sp[0]] = int(float(sp[2]))
            elif "fitDefPot" in line:
                sp = line.split()
                mc_opts[sp[0]] = (sp[2] == "True" or sp[2] == "true")
            elif "fitCoupling" in line:
                sp = line.split()
                mc_opts[sp[0]] = (sp[2] == "True" or sp[2] == "true") 
            elif "fitEffMass" in line:
                sp = line.split()
                mc_opts[sp[0]] = (sp[2] == "True" or sp[2] == "true")
            elif "optGaps" in line:
                sp = line.split()
                mc_opts[sp[0]] = (sp[2] == "True" or sp[2] == "true")
            elif "defPotWeight" in line:
                sp = line.split()
                mc_opts[sp[0]] = float(sp[2])
            else:
                sp = line.split()
                raise ValueError(f"unexpected montecarlo input keyword: {sp[0]}")
            
    return mc_opts