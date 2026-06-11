import numpy as np
import torch.linalg
import time
import os, sys
import matplotlib.pyplot as plt

from .constants import *
from .pp_func import plotPP, plot_training_validation_cost, plotBandStruct, plot_mc_cost, plotBandStruct_reorder

torch.set_default_dtype(torch.float64)

class MonteCarloFit:

    def __init__(self, hams, 
                 writeDir,
                 nSystems=1,
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
                 couplingOpts=None, 
                 writePerIter=100):
        
        """
        This class runs Monte Carlo optimization on non-neural net 
        pseudopotential params. It is currently only written for a single 
        Hamiltonian (a single bandstructure), but generalization should be easy.
        Actually, as of 10/01/2025, support for simultaneous multi-bandstructure fitting
        is enabled - Daniel C

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

        self.hams = hams
        self.nSystems = nSystems

        kpts = []
        kptWeights = []
        qpts = []
        qptWeights = []
        expBS = []
        bandWeights = []
        expCpl = []
        idx_gap = []
        idx_vb = []
        idx_cb = []
        expDef = []
        expEffMasses = []
        effMassWeights = []

        for iSys in range(nSystems):
            kpts.append(hams[iSys].system.kpts)
            kptWeights.append(hams[iSys].system.kptWeights)

            expBS.append(hams[iSys].system.expBandStruct)
            bandWeights.append(hams[iSys].system.bandWeights)

            if bandWeights is None:
                bandWeights = torch.ones(hams[iSys].system.nBands)
            if fitDefPot:
                expDef.append(hams[iSys].system.expDefPots)
            if fitCoupling:
                expCpl.append(hams[iSys].system.expCouplingBands)
                qpts.append(hams[iSys].system.qpts)
                qptWeights.append(hams[iSys].system.qptWeights)
            if fitEffMass:
                expEffMasses.append(self.hams[iSys].system.expEffMasses)
                effMassWeights.append(self.hams[iSys].system.effMassWeight)
            if fitCoupling or fitEffMass:
                idx_gap.append(hams[iSys].system.idxGap)
                idx_vb.append(hams[iSys].system.idxVB)
                idx_cb.append(hams[iSys].system.idxCB)
        
        self.kpts = kpts
        self.kptWeights = kptWeights

        self.expBS = expBS
        self.bndWeight = bandWeights
        if fitEffMass is True:
            self.expEffMasses = expEffMasses 
            self.effMassWeights = effMassWeights

        if fitCoupling:
            self.expCpl = expCpl
            self.qpts = qpts
            self.qptWeights = qptWeights
            #self.cplBndWeight = ham.system.couplingBandWeights
        if fitDefPot:
            self.expDef = expDef

        if fitEffMass or fitCoupling:
            self.idx_gap = idx_gap
            self.idx_vb = idx_vb
            self.idx_cb = idx_cb
            
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
        

        if writeDir is None:
            print("!! WARNING: you are running Monte Carlo but have not supplied a dir for output files!")
        self.writeDir = writeDir
        self.writePerIter = writePerIter

        self.bestMSE = 0.0
        self.currentMSE = 0.0
        self.newMSE = 0.0
        self.firstIter = True

        return


    def run_mc(self, cachedMats_info=None):
        currentMSE = 0
        nAccept = 0
        nIter = 0
        stepsAtTemp = 0
        tempIdx = 0
        bestAtTemp = 1e6
        sinceLastAccept = 0
        bestPP = None  # to store the globally optimal PPparams
        
        # Initialize for each system
        for iSys in range(self.nSystems):
            # calculate initial band structure, ?defpot, ?coupling and 
            # write them to files
            bs = self.hams[iSys].calcBandStruct(cachedMats_info=cachedMats_info)
            self.writeBands(bs, iSys)
            if self.fitDefPot:
                defpots = self.calcDefPots(bs[self.idx_gap[iSys], self.idx_vb[iSys]], bs[self.idx_gap[iSys], self.idx_cb[iSys]])
            else:
                defpots = None
            if self.fitCoupling:
                cpl_dict = self.hams[iSys].calcCouplings(**self.couplingOpts)
                self.writeCoupling(cpl_dict, iSys)
            else:
                cpl_dict = None
            if self.fitEffMass:
                eff_masses = self.calcEffMasses(bs, iSys)
                self.writeEffMasses(eff_masses, iSys)
            else:
                eff_masses = None

            currentMSE += self.__evalCostFn(bs, iSys, defpots, cpl_dict, eff_masses)
        
        self.currentMSE = currentMSE
        self.bestMSE = self.currentMSE
        
        self.writeIteration(1.0)

        # Start Monte Carlo iterations
        t0 = time.time()
        for _ in range(self.totalIter):
            sinceLastAccept += 1
            nIter += 1
            if sinceLastAccept > 10000:
                print("no accepted moves for a while...quitting")
                break

            # modify params step
            # This assumes that all systems are trying to fit the same PP params
            tmpPP = self.hams[0].get_PPparams()
            self.updatePPparams(tempIdx)
            

            # Compute cost function
            newMSE = 0
            bs = []
            for iSys in range(self.nSystems):
                # check cost fn
                bs.append(self.hams[iSys].calcBandStruct(cachedMats_info=cachedMats_info))
                if self.fitDefPot:
                    defpots = self.calcDefPots(bs[iSys][self.idx_gap[iSys], self.idx_vb[iSys]], bs[iSys][self.idx_gap[iSys], self.idx_cb[iSys]])
                if self.fitCoupling:
                    cpl_dict = self.hams[iSys].calcCouplings(**self.couplingOpts)
                if self.fitEffMass:
                    eff_masses = self.calcEffMasses(bs[iSys], iSys)

                newMSE += self.__evalCostFn(bs[iSys], iSys, defpots, cpl_dict, eff_masses)

            self.newMSE = newMSE
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
            if nIter % 1 == 0:
                self.writeIteration(nAccept/nIter, nIter=nIter)

            # update PPparams?
            if self.newMSE < self.bestMSE:
                self.bestMSE = self.currentMSE = self.newMSE
                # in this case we want to retain updated PPparams
                bestPP = self.hams[0].get_PPparams()
                self.saveParams(bestPP)
                self.writeIterPPparams(bestPP, nIter)
                

                for iSys in range(self.nSystems):
                    self.writeBands(bs[iSys], iSys, stub=f"/nIter_{nIter}_BS")
                    self.writeBands(bs[iSys], iSys, stub=f"/bestBandStruct_{nIter}")
                    
                    if self.fitCoupling:
                        self.writeCoupling(cpl_dict, iSys, stub=f"/bestCoupling_{nIter}")
                    if self.fitEffMass:
                        self.writeEffMasses(eff_masses, iSys, stub=f"bestEffMasses_{nIter}")
                    # Plot this bestBandstructure
                    fig = plotBandStruct([self.hams[iSys].system], [self.expBS[iSys], bs[iSys]], True, "Zunger prediction")
                    fig.suptitle(f"mc_BS_MSE = {self.newMSE:.4f}. total_BS_MSE = {self.newMSE * self.hams[iSys].system.nBands * self.hams[iSys].system.getNKpts():.4f}. ")
                    fig.savefig(f'{self.writeDir}/bestBandstruct_{nIter}_{iSys}.pdf')
                    plt.close('all')
                sinceLastAccept = 0
            elif mc_bool:
                # new MSE is higher, but we still accept.
                # keep self.ham.PPparams in the updated form
                self.currentMSE = self.newMSE
            else:
                # do not accept new params, revert to most recently accepted vals
                # This assumes that the multiple systems are trying to fit the same PP parameters!
                self.hams[0].set_PPparams(tmpPP)
                

            if (nIter == 0) or (nIter%self.writePerIter==0):
                print(f"Finishing iteration {nIter}...")
                self.writeIterPPparams(tmpPP, nIter)
                for iSys in range(self.nSystems):
                    self.writeBands(bs[iSys], iSys, stub=f"/nIter_{nIter}_BS")
                    
                    if self.fitCoupling:
                        self.writeCoupling(cpl_dict, iSys, stub=f"/nIter_{nIter}_Coupling")
                    if self.fitEffMass:
                        self.writeEffMasses(eff_masses, iSys, stub=f"/nIter_{nIter}_effMasses")

                    fig = plotBandStruct([self.hams[iSys].system], [self.expBS[iSys], bs[iSys]], True)
                    fig.suptitle(f"mc_BS_MSE = {self.newMSE:.4f}. total_BS_MSE = {self.newMSE * self.hams[iSys].system.nBands * self.hams[iSys].system.getNKpts():.4f}. ")
                    fig.savefig(f'{self.writeDir}/nIter_{nIter}_plotBS_{iSys}.pdf')
                    fig.savefig(f'{self.writeDir}/nIter_{nIter}_plotBS_{iSys}.png')
                    plt.close('all')

        tf = time.time()
        print(f"\n\n\nDone fitting. Total iters = {nIter}. Total wall time = {tf-t0}")
        print(f"best MSE = {self.bestMSE}")
        print("Best PPparams (unformatted) = ")
        print(bestPP)
        
        print("\nAlso writing bestPPparams to files...")
        try:
          self.writeBestPPparams(bestPP)
        except AttributeError:
            print("\nUnable to write bestPP params because no better solution found...")
            print("\nWriting current PP params...")
            bestPP = tmpPP

        return bestPP






    def __evalCostFn(self, bs, iSys, defpots, cpl_dict, eff_masses):
        """
        This just modularizes the calls to evaluate the total 
        cost function. You can play around with this to tailor the optimization.
        """
        if self.optGaps:
            cost = self.calcIndivMSEgaps(bs, iSys)
        else:
            cost = self.calcIndivMSE(bs, iSys)
        if self.hams[iSys].SObool:
            cost += self.calcNonLocalWeighting(iSys)
        if self.fitEffMass:
            cost += self.calcEffMassMSE(eff_masses, iSys)
        if self.fitDefPot:
            cost += self.calcDefPotMSE(defpots)
        if self.fitCoupling:
            cost += self.calcCouplingMSE(cpl_dict, iSys)
        
        return cost

    def calcEffMasses(self, bs, iSys):
        '''Calculate the vbm and cbm effective masses assuming parabolic bands.
        This REQUIRES that idxGap, idxVB, and idxCB are set. The neighboring point at Gamma - dk
        used to compute the derivative should be at idxGap - 1 in expBandstructure.par.
        Returns a list eff_masses: [vb_eff_mass, cb_eff_mass]'''

        eff_masses = [None, None]

        # -----------------------------
        # Constants
        # -----------------------------
        hbar = 1.054571817e-34       # J·s
        eV_to_J = 1.602176634e-19     # J / eV
        m_e = 9.1093837015e-31        # kg
        bohr_to_ang = 0.529177

        # -----------------------------
        # Compute |Δk| in m^-1; fractional k already scaled by reciprocal lat vecs
        # -----------------------------
        
        kpt0 = 1e10 / bohr_to_ang * self.hams[iSys].system.kpts[self.idx_gap[iSys]]     # a.u.^-1
        kpt1 = 1e10 / bohr_to_ang * self.hams[iSys].system.kpts[self.idx_gap[iSys] - 1] # a.u.^-1
        
        dk = kpt1 - kpt0
        dk_mag = np.linalg.norm(dk)
        
        # -----------------------------
        # Extract energies at band extrema in eV
        # -----------------------------
        vb0 = bs[self.idx_gap[iSys], self.idx_vb[iSys]]
        vb1 = bs[self.idx_gap[iSys] - 1, self.idx_vb[iSys]]

        cb0 = bs[self.idx_gap[iSys], self.idx_cb[iSys]]
        cb1 = bs[self.idx_gap[iSys] - 1, self.idx_cb[iSys]]

        # -----------------------------
        # Compute second derivative
        # -----------------------------
        # E1 - E0 = 1/2 E'' (|Δk|)^2  => E'' = 2 ΔE / (|Δk|)^2
        dE_vb = (vb1 - vb0) * eV_to_J  # convert to J
        E_dp_vb = 2 * dE_vb / (dk_mag ** 2)
        
        dE_cb = (cb1 - cb0) * eV_to_J  # convert to J
        E_dp_cb = 2 * dE_cb / (dk_mag ** 2)

        # -----------------------------
        # Effective mass
        # -----------------------------
        m_eff_vb = -1 * float(hbar**2 / E_dp_vb / m_e)
        m_eff_cb = float(hbar**2 / E_dp_cb / m_e)

        eff_masses[0] = round(m_eff_vb, 2)
        eff_masses[1] = round(m_eff_cb, 2)
        
        return eff_masses

    def calcIndivMSE(self, bs, iSys):
        # this can be sped up with array operations rather than for loops
        mse = 0.0
        ctr = 0
        for kidx in range(self.kpts[iSys].shape[0]):
            tmp = 0.0
            for bidx in range(bs.shape[1]):
                if abs(self.expBS[iSys][kidx, bidx]) > 1e-15:
                    ctr += 1
                    tmp += (bs[kidx, bidx] - self.expBS[iSys][kidx,bidx])**2 * self.bndWeight[iSys][bidx]
            mse += tmp * self.kptWeights[iSys][kidx]
        
        return mse / ctr 
        # return mse

    def calcIndivMSEgaps(self, bs, iSys):
        # this can be sped up with array operations rather than for loops
        mse = 0.0
        ctr = 0
        for kidx in range(self.kpts[iSys].shape[0]):
            tmp = 0.0
            for bidx in range(bs.shape[1] - 1):
                if abs(self.expBS[iSys][kidx, bidx+1]) > 1e-15 and abs(self.expBS[iSys][kidx, bidx]) > 1e-15:
                    ctr += 1
                    dgap = bs[kidx, bidx+1] - bs[kidx, bidx]
                    egap = self.expBS[iSys][kidx, bidx+1] - self.expBS[iSys][kidx, bidx]
                    tmp += (dgap - egap)**2 * (self.bndWeight[iSys][bidx] + self.bndWeight[iSys][bidx+1])/2
                    # ^^ should we normalize this by expected gap value?
                    # so if two bands are close in the ref data, deviations are
                    # measured on a relative scale. this is different than how the
                    # abs energies work, but it would make this function more sensistive
                    # and it would more equally weight the different bands.
                    # PROBLEMS WOULD HAPPEN FOR DEGENERATE BANDS!
            mse += tmp * self.kptWeights[iSys][kidx]
        return mse / ctr
    
    def calcNonLocalWeighting(self, iSys):
        """
        This incurs a simple penalty if the SOC factor is too large
        """
        weight = 0
        for atom, params in self.hams[iSys].PPparams.items(): 
            if params[5] > 8.0: weight += (params[5] - 8)**2 * 5
        
        return weight
    
    def calcDefPotMSE(self, defpots):
        """
        expecting input to be [vbmDefPot, cbmDefPot]
        """
        return self.defPotWeight * ((defpots[0] - self.expDef[0])**2 + (defpots[1] - self.expDef[1])**2)
    
    def calcEffMassMSE(self, eff_masses, iSys):
        """
        expecting input to be [vbmEffMass, cbmEffMass]
        """
        return self.effMassWeights[iSys] * ((eff_masses[0] - self.expEffMasses[iSys][0])**2 + (eff_masses[1] - self.expEffMasses[iSys][1])**2)
    
    def calcCouplingMSE(self, cpl_dict, iSys):
        mse = 0
        count = 0
        for key, cpl in cpl_dict.items():
            # only compare the couplings that are computed, not necessarily all
            # reference data
            qidx = key[2]
            mse += (cpl - self.expCpl[iSys][key])**2 * self.qptWeights[iSys][qidx]
            count += 1
        
        return mse / count


    def updatePPparams(self, tempIdx):
        """
        Randomly modify ALL self.hams.PPparams to the same value. This is done IN-PLACE!!
        If supplied to the class constructor, custom step sizes 
        of each parameter will be used. Otherwise, the scale will be
        1/100 * parameter magnitude for each param, which is a decent
        paradigm.
        """
        steps = {}
        if self.paramSteps is not None:
            steps = self.paramSteps
        else:
            for atom, params in self.hams[0].PPparams.items():
                steps[atom] = params / 100.0
                steps[atom][3] = np.log(params[3] + 1) / 100.0 # vary the exponential parameter logarithmically slowly

        for atom, params in self.hams[0].PPparams.items():
            for j in range(len(params)):
                self.hams[0].PPparams[atom][j] += np.random.uniform(low=-1.0,high=1.0) * steps[atom][j] * self.tempStepSizeMod[tempIdx]
        
        # Set ALL systems to have the same PP params
        for iSys in range(1, self.nSystems):
            self.hams[iSys].PPparams = self.hams[0].PPparams

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
        for i in range(self.hams[0].system.getNAtoms()):
            atom = self.hams[0].system.atomTypes[i]
            self.hams[0].PPparams[atom][5] = abs(self.hams[0].PPparams[atom][5]) # SOC must be positive
            # count the number of each atom type
            if atom in ctr:
                ctr[atom] += 1
            else:
                ctr[atom] = 1
                # find which atom to constrain for long-range sum.
                # Ignore atoms with LR param = 0
                if abs(self.hams[0].PPparams[atom][4]) > 1e-10:
                    constrainLbl = atom
            
        if constrainLbl is None:
            # there are no long-range potentials, nothing more to do
            return
        sumLR = 0.0
        for i in range(self.hams[0].system.getNAtoms()):
            # add LR param for all atoms that are not constrained
            atom = self.hams[0].system.atomTypes[i]
            if atom != constrainLbl:
                sumLR += self.hams[0].PPparams[atom][4] 

        cparam = -1 * sumLR / ctr[constrainLbl]  # this makes the total sum to 0
        self.hams[0].PPparams[constrainLbl][4] = cparam

        # Set ALL systems to have the same PP params
        for iSys in range(1, self.nSystems):
            self.hams[iSys].PPparams = self.hams[0].PPparams

        return
            

    def calcDefPots(self, vbm_reg, cbm_reg, verbosity=2):
        """
        This is a simple function to compute the deformation potential by
        finite difference. It takes the NON-deformed vbm energy and cbm energy
        in units of EV as arguments to avoid redundant computation.
        """
        # ************************************************************************
        # THIS FUNCTION HAS NOT YET BEEN MADE COMPATIBLE WITH MULTIPLE SYSTEMS
        # ************************************************************************

        defscale = 1.01 # 1% expansion is not converged, but matches DFT defpot literature
        Hdef = self.hams[0].buildHtot_def(scale=defscale, verbosity=verbosity)
        evals = torch.linalg.eigvalsh(Hdef) * AUTOEV

        diffVBM = vbm_reg - evals[self.idx_vb[0]]
        diffCBM = cbm_reg - evals[self.idx_cb[0]]
        vol = self.hams[0].system.getCellVolume()
        defpotVBM = diffVBM / (vol - defscale**3 * vol) * 0.5 * (vol + defscale**3*vol)
        defpotCBM = diffCBM / (vol - defscale**3 * vol) * 0.5 * (vol + defscale**3*vol)

        if verbosity >= 1:
            print(f"VB deformation potential = {defpotVBM}")
            print(f"CB deformation potential = {defpotCBM}")

        return [defpotVBM, defpotCBM]
    

    def writeBands(self, bs, iSys, stub="/bandStruct"):
        with open(self.writeDir + stub + f"_{iSys}.dat", 'w') as fwrite:
            pathlength = 0.0
            for i in range(bs.shape[0]):
                if i > 0:
                    diff = torch.sqrt(torch.sum((self.kpts[iSys][i] - self.kpts[iSys][i-1])**2))
                    pathlength += diff
                print(f"{pathlength:.4f}  ", file=fwrite, end="")
                for j in range(bs.shape[1]):
                    print(f"{bs[i,j]:.8f} ", file=fwrite, end="")
                print("\n", file=fwrite, end="")

    def writeEffMasses(self, eff_masses, iSys, stub="/effMasses"):
        """
        This function takes the eff_masses from calcEffMasses()
        and prints them in a labelled data file. 
        """
        filename = self.writeDir + stub + f"_{iSys}.dat"
        with open(filename, 'w') as fwrite:
            print(f"vb = {eff_masses[0]}\ncb = {eff_masses[1]}", file=fwrite, end="")
        return


    def writeCoupling(self, cpl_dict, iSys, stub="/couplingBands_"):
        """
        This function takes the couplings as they are output from ham.calcCouplings()
        and writes them in a labelled data file. 
        """
        with open(self.writeDir + stub + f"_{iSys}.dat", 'w') as fwrite:
            for atomidx in range(self.hams[iSys].system.getNAtoms()):
                print(f"Atom idx = {atomidx}   atom = {self.hams[iSys].system.atomTypes[atomidx]}   position = {self.hams[iSys].system.atomPos[atomidx]}", file=fwrite)

                for band in ["vb", "cb"]:
                    print(f"{band}-{band} coupling elements. ", file=fwrite, end="")
                    for gamma in range(3):
                        if gamma == 0:
                            print("polarization of derivative = x", file=fwrite)
                        elif gamma == 1:
                            print("polarization of derivative = y", file=fwrite)
                        else:
                            print("polarization of derivative = z", file=fwrite)
                        
                        for qidx in range(self.qpts[iSys].shape[0]):
                            if (atomidx, gamma, qidx, band) in cpl_dict:
                                print(f"{cpl_dict[(atomidx, gamma, qidx, band)]:.6e}   ", file=fwrite, end="")
                            else:
                                print("Not-fit   ", file=fwrite, end="")
                        print("\n", file=fwrite, end="")
                    print("\n", file=fwrite, end="")
                print("\n\n", file=fwrite, end="")


    def writeIteration(self, perAccept, stub="/iterations.dat", nIter=None):
        if self.firstIter is True:
            if os.path.isfile(self.writeDir + stub):
                # don't append to an existing file, remove existing file
                os.remove(self.writeDir + stub)

        with open(self.writeDir + stub, 'a') as fwrite:
            if self.firstIter is True:
                print("newMSE \t curMSE \t bstMSE \t %Accpt \t ", file=fwrite, end="")
                for atom, params in self.hams[0].PPparams.items():
                    for j in range(len(params)):
                        print(f"{atom}_a{j}  \t", file=fwrite, end="")
                print("\n",file=fwrite, end="")
            
            nIter_str = f"{nIter} " if nIter != None else ''
            print(f"{nIter_str}{self.newMSE:.6g}\t {self.currentMSE:.6g}\t {self.bestMSE:.6g}\t {perAccept:.5g}\t ", file=fwrite, end="")
            for atom, params in self.hams[0].PPparams.items():
                for j in range(len(params)):
                    print(f"{params[j]:.14f}\t", file=fwrite, end="")
            print("\n", file=fwrite, end="")
        
        if self.firstIter:
            self.firstIter = False


    def saveParams(self, params, stub="bestParams.pt"):
        for atom, lst in params.items():
            torch.save(lst, self.writeDir + f"/{atom}" + stub)

    def writeIterPPparams(self, param_dict, nIter):
        for atom, params in param_dict.items():
            with open(self.writeDir + f"/nIter_{nIter}_{atom}_PPparams.dat", "w") as fwrite:
                for j in range(len(params)):
                    print(f"{params[j]:.14f}", file=fwrite)
        
    def writeBestPPparams(self, param_dict):
        for atom, params in param_dict.items():
            with open(self.writeDir + f"/best_{atom}Params.dat", 'w') as fwrite:
                for j in range(len(params)):
                    print(f"{params[j]:.14f}", file=fwrite)
                

def initMonteCarlo(hams, writeDir, nSystems, mc_opts, betas=[100, 1], stepsAtTemp=[5000, 500], tempStepSizeMod=[1.0, 1.0],
      paramSteps=None, totalIter=100000, fitDefPot=False, fitCoupling=False, fitEffMass=False, optGaps=False, defPotWeight=1.0,
      couplingOpts=None, writePerIter=100):
    
    optimizers = []
    for iSys in nSystems:
        optimizer = MonteCarloFit(hams[0], writeDir, paramSteps=paramSteps, **mc_opts)
        optimizers.append(optimizer)

    return optimizers
    

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
            elif "writePerIter" in line: 
                sp = line.split()
                mc_opts[sp[0]] = float(sp[2])
            else:
                sp = line.split()
                raise ValueError(f"unexpected montecarlo input keyword: {sp[0]}")
            
    return mc_opts
