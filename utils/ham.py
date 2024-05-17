import sys, os
import torch
import numpy as np
from scipy.special import erf
from scipy.integrate import quad, quadrature, quad_vec
from scipy.optimize import linear_sum_assignment
import time
import copy
import gc
from torch.utils.checkpoint import checkpoint
import multiprocessing as mp
from multiprocessing import Process, Queue, Pool, shared_memory
import gc

from .constants import *
from .pp_func import pot_func, pot_funcLR
from .read import init_critical_NNconfig

torch.set_default_dtype(torch.float64)

class Hamiltonian:
    def __init__(
        self,
        system,
        PPparams,
        atomPPorder,
        device, 
        NNConfig = None, 
        iSystem = 0, 
        SObool = False,
        cacheSO = True,
        NN_locbool = False,
        model = None,
        coupling = False
    ):
        """
        The Hamiltonian is initialized by passing it an initialized and
        populated BulkSystem class, which contains all the relevant 
        information about the basis, atoms, etc. 
        PPparams should be formatted as a dict of lists, where
        PPparams[atomkey] = [params], and atomkey is the string symbol of the atom.
        "atomPPorder" is an ordered list of the unique atoms in the system. If 
        using a NN model for local potential, it is important that this arg is
        consistent with the construction of the NN.
        "device" should be specified using torch for cpu vs gpu.
        "iSystem" is the global (static) index of the system that gives this 
        hamiltonian instance. 
        "coupling" should be True if you want to also fit e-ph coupling matrix
        elements. 
        The other kwargs are specified for using a NN, currently only for 
        the local potential.
        """

        self.basis = system.basis() # check if this is done the same as Daniel
        self.PPparams = PPparams
        self.atomPPorder = atomPPorder
        self.system = system
        self.device = device
        if NNConfig == None:
            self.NNConfig = init_critical_NNconfig()
            print("\n~Warning: you didn't supply an NNConfig dict...")
            print("Setting default values for parallelization, checkpointing, and timing\n")
        else:
            self.NNConfig = NNConfig
        self.iSystem = iSystem
        self.SObool = SObool
        self.cacheSO = cacheSO
        self.NN_locbool = NN_locbool
        self.model = model
        self.coupling = coupling   # fit the e-ph couplings? boolean
        self.physicalBandOrdering = system.physicalBandOrdering

        self.LRgamma = 0.2   # erf attenuation parameter for long-range 
                             # component of potential. This is a good value

        # if spin orbit, do a bunch of caching to speed up the inner loop 
        # of the optimization. This uses more memory (storing natom * nkpt
        # matrices of size 4*nbasis^2) in exchange for avoiding loops
        # over the basis within the optimization inner loop.
        self.SOmats = None
        self.NLmats = None
        self.SOmats_def = None
        self.NLmats_def = None
        if SObool and cacheSO:
            self.SOmats = self.initSOmat_fast()
            self.SOmats_def = None
            # check if nonlocal potentials are included, if so, cache them
            self.checknl = False
            for alpha in range(system.getNAtomTypes()):
                if abs(self.PPparams[self.system.atomTypes[alpha]][6]) > 1e-8:
                    self.checknl = True
                    break
                elif abs(self.PPparams[self.system.atomTypes[alpha]][7]) > 1e-8:
                    self.checknl = True
                    break
            if self.checknl:
                self.NLmats = self.initNLmat_fast()
                self.NLmats_def = None
       
        elif (SObool) and (not cacheSO) and (NNConfig['num_cores']==0):
            print("WARNING: Calculation requires SObool, but we are not cache-ing the SOmats and NLmats. Without multiprocessing parallelization. This is not recommended. ")

        
        if self.coupling:
            nkpt = self.system.getNKpts()
            #nbv = self.basis.shape[0]
            #if SObool: nbv *= 2
            #self.vb_vecs = torch.zeros([nkpt, nbv, 1], dtype=torch.complex128)
            #self.cb_vecs = torch.zeros([nkpt, nbv, 1], dtype=torch.complex128)
            self.vb_vecs = {k : [] for k in range(nkpt)}
            self.cb_vecs = {k : [] for k in range(nkpt)}

            if not isinstance(self.system.idxVB, int):
                raise ValueError("need to specify vb, cb indices for coupling")
            elif not isinstance(self.system.idxCB, int):
                raise ValueError("need to specify vb, cb indices for coupling")
            elif not isinstance(self.system.idxGap, int):
                raise ValueError("need to specify kpt index of bandgap for coupling")
            else:
                self.idx_vb = self.system.idxVB
                self.idx_cb = self.system.idxCB
                self.idx_gap = self.system.idxGap

            if SObool:
                self.SOmats_couple, self.NLmats_couple = self.initCouplingMats()

        # For storing the shared eigenvector information and data
        # self.eVec_info = {}
        # self.shm_eVec = {}
        self.eVec_info = mp.Manager().dict()
        # self.shm_eVec = mp.Manager().dict()

        # send things to gpu, if enabled ??
        # Or is it better to send some things at the last minute before diagonalization?
        if model is not None:
            model.to(device)
        

    def buildHtot(self, kidx, preComp_SOmats_kidx=None, preComp_NLmats_kidx=None, requires_grad=True):
        """
        Build the total Hamiltonian for a given kpt, specified by its kidx. 
        preComp_SOmats_kidx and preComp_NLmats_kidx are the pre-computed
        SO and NL matrices (actual matrices) at the certain kidx
        """
        nbv = self.basis.shape[0]
        if self.SObool:
            Htot = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)
        else:
            Htot = torch.zeros([nbv, nbv], dtype=torch.complex128)
        
        # kinetic energy
        if self.SObool: top = 2*nbv
        else: top = nbv
        for i in range(top):
            Htot[i,i] = HBAR**2 / (2*MASS) * torch.norm(self.basis[i%nbv] + self.system.kpts[kidx])**2

        # local potential
        start_time = time.time() if self.NNConfig['runtime_flag'] else None
        Htot = self.buildVlocMat(addMat=Htot)
        if not requires_grad: 
            Htot = Htot.detach()
        end_time = time.time() if self.NNConfig['runtime_flag'] else None
        print(f"Building VlocMat, elapsed time: {(end_time - start_time):.2f} seconds") if self.NNConfig['runtime_flag'] else None

        if self.SObool:
            start_time = time.time() if self.NNConfig['runtime_flag'] else None
            Htot = self.buildSOmat(kidx, preComp_SOmats_kidx, addMat=Htot)
            end_time = time.time() if self.NNConfig['runtime_flag'] else None
            print(f"Building SOmat, elapsed time: {(end_time - start_time):.2f} seconds") if self.NNConfig['runtime_flag'] else None

            start_time = time.time() if self.NNConfig['runtime_flag'] else None
            Htot = self.buildNLmat(kidx, preComp_NLmats_kidx, addMat=Htot)
            end_time = time.time() if self.NNConfig['runtime_flag'] else None
            print(f"Building NLmat, elapsed time: {(end_time - start_time):.2f} seconds") if self.NNConfig['runtime_flag'] else None

        if self.device.type == "cuda":
            # !!! is this sufficient to match previous performance?
            # This limits data movement to gpu (good), but obviously
            # performs construction of H on cpu (at least the first time?), 
            # which might be slower.
            Htot.to(self.device)

        if not requires_grad: 
            Htot = Htot.detach()
        return Htot
    

    def buildHtot_def(self, scale=1.0001, verbosity=2):
        """
        Build the total Hamiltonian in the deformed basis, for ONLY the
        bandgap kpt. This is used for the "classic" method of
        computing the deformation potential. The deformed unit cell is scaled
        by "scale". IMPORTANT: this function assumes that you only want to
        construct the deformed Hamiltonian at a SINGLE kpoint - the kpoint 
        corresponding to the bandgap.
        """
        """
        This function currently doesn't account for the shared_memory SOmats and NLmats. 
        It might mess things up. 
        """
        if verbosity >= 2:
            print("***************************")
            print("You are computing deformation potentials by directly changing")
            print("the volume of the material. To be precise, computing a")
            print("quantity that can be correctly compared to the DFT literature,")
            print("or experiments, requires very careful consideration of the")
            print("g_i - g_j = 0 point in the potentials. These considerations")
            print("are not made here. Consult the DFT literature, e.g.")
            print("PRB 73 245206 (2006) and its references.")
            print("***************************")

        kidx = self.idx_gap

        self.defscale = self.system.scale * scale
        # modify the relevent quantities, then modify them back after diagonalizing
        self.basis *= (self.system.scale / self.defscale)
        self.system.kpts *= (self.system.scale / self.defscale)
        self.system.unitCellVectors *= (self.defscale / self.system.scale)
        self.system.atomPos *= (self.defscale / self.system.scale)



        nbv = self.basis.shape[0]
        if self.SObool:
            Htot = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)
        else:
            Htot = torch.zeros([nbv, nbv], dtype=torch.complex128)
        
        # kinetic energy
        if self.SObool: top = 2*nbv
        else: top = nbv
        for i in range(top):
            Htot[i,i] = HBAR**2 / (2*MASS) * torch.norm(self.basis[i%nbv] + self.system.kpts[kidx])**2

        # local potential
        Htot = self.buildVlocMat(addMat=Htot)

        if self.SObool:
            store_SOmats = self.SOmats
            if self.checknl:
                store_NLmats = self.NLmats
            # only compute the SO integrals for the kpt corresponding to the gap (assuming direct gap).
            # check if we cached them from the first call...
            if self.SOmats_def is not None:
                self.SOmats = self.SOmats_def
            else:
                self.SOmats_def = self.initSOmat_fast(defbool=True, idxGap=kidx)
                self.SOmats = self.SOmats_def
            if self.NLmats_def is not None and self.checknl:
                self.NLmats = self.NLmats_def
            elif self.checknl:
                self.NLmats_def = self.initNLmat_fast(defbool=True, idxGap=kidx)
                self.NLmats = self.NLmats_def

            # the below calls are kidx=0 because they index into the SOmats and NLmats
            # arrays, for which there is only a single kpoint. There are no calls
            # self.system.kpts[kidx] in these functions, so it does not cause any
            # issues.
            Htot = self.buildSOmat(0, addMat=Htot)
            if self.checknl:
                Htot = self.buildNLmat(0, addMat=Htot)

        

        # now return everything to its non-deformed values
        self.basis *= (self.defscale / self.system.scale)
        self.system.kpts *= (self.defscale / self.system.scale)
        self.system.unitCellVectors *= (self.system.scale / self.defscale)
        self.system.atomPos *= (self.system.scale / self.defscale)
        if self.SObool:
            self.SOmats = store_SOmats
            if self.checknl:
                self.NLmats = store_NLmats

        return Htot

    
    def buildVlocMat(self, addMat=None):
        """
        Computes the local potential, either using the algebraic form
        or the NN form.
        V_{i,j} = <G_i|V|G_j> = \sum_k [e^{+i(G_i-G_j)\cdot\tau_k} * v(|G_i-G_j|) / (V_cell)].
        "addMat" can be set to be a partially constructed Hamiltonian matrix, to
        which the local potential can be added. Might help save slightly on memory. 
        """
        nbv = self.basis.shape[0]
        gdiff = torch.stack([self.basis] * nbv, dim=1 ) - self.basis.repeat(nbv,1,1)

        def compute_atomFF():
            return self.model(torch.norm(gdiff, dim=2).view(-1,1))
    
        if addMat is not None:
            if self.SObool:
                assert addMat.shape[0] == 2*nbv
                assert addMat.shape[1] == 2*nbv
            Vmat = addMat
        else:
            if self.SObool:
                Vmat = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)
            else:
                Vmat = torch.zeros([nbv, nbv])

        for alpha in range(self.system.getNAtoms()):
            gdiffDotTau = torch.sum(gdiff * self.system.atomPos[alpha], axis=2)
            sfact_re = 1/self.system.getCellVolume() * torch.cos(gdiffDotTau)
            sfact_im = 1/self.system.getCellVolume() * torch.sin(gdiffDotTau)

            thisAtomIndex = np.where(self.system.atomTypes[alpha]==self.atomPPorder)[0]
            if len(thisAtomIndex)!=1: 
                raise ValueError("Type of atoms in PP. ")
            thisAtomIndex = thisAtomIndex[0]

            if self.NN_locbool:
                # atomFF = self.model(torch.norm(gdiff, dim=2).view(-1,1))
                if self.NNConfig['checkpoint']==0: 
                    atomFF = self.model(torch.norm(gdiff, dim=2).view(-1,1))
                elif self.NNConfig['checkpoint']==1: 
                    atomFF = checkpoint(compute_atomFF, use_reentrant=False)
                atomFF = atomFF[:, thisAtomIndex].view(nbv, nbv)
            else:
                # atomFF = pot_func(torch.norm(gdiff, dim=2), self.PPparams[self.system.atomTypes[alpha]])
                atomFF = pot_funcLR(torch.norm(gdiff, dim=2), self.PPparams[self.system.atomTypes[alpha]], self.LRgamma)

            if self.SObool:
                # local potential has delta function on spin --> block diagonal
                Vmat[:nbv, :nbv] = Vmat[:nbv, :nbv] + atomFF * torch.complex(sfact_re, sfact_im)
                Vmat[nbv:, nbv:] = Vmat[nbv:, nbv:] + atomFF * torch.complex(sfact_re, sfact_im)
            else:
                #sfact = torch.complex(sfact_re, sfact_im)
                #print(sfact.dtype)
                #print((sfact*atomFF)[:8, :8])
                Vmat = Vmat + atomFF * torch.complex(sfact_re, sfact_im)

        return Vmat


    def initSOmat(self, SOwidth=0.7, defbool=False, idxGap=None):
        """
        Calculates the SO integral Vso(K,K') = integral from 0 t0 infinity of
        dr*r^2*j1(Kr)*exp^(-(r/0.7)^2)*j1(K'r) where j1 is the 1st bessel function,
        K = kpoint + basisVector and exp^(-(r/0.7)^2) is the  spin-orbit potential
        excluding the variable "a" parameter. Then builds the SO matrix components
        corresponding to every atom type at each kpoint. WARNING: might consume
        significant memory. You are storing natom * nkpt complex matrices of dimension
        (2*nbasis) x (2*nbasis). Format of output is SOmats[kidx, atomidx] = SOmatrix
        """
        nbv = self.basis.shape[0]
        # set integral dr ~ 0.0089 Bohr at 25 Hartree energy cutoff
        #dr = 2*np.pi / (100 * torch.norm(self.basis[-1]))
        # set radial cutoff ~ 4.2488 Bohr; V(rcut) = 1e-16 for default SOwidth
        rcut = np.sqrt(SOwidth**2 * 16 * np.log(10.0))
        #ncut = int(rcut/dr)
        
        if defbool:
            nkp = 1  # to allow for deformation calcs at a single kpoint
            if idxGap is None:
                raise RuntimeError("need to specify kpt idx of gap in deformed calc")
        else:
            nkp = self.system.getNKpts()
        
        SOmats = np.empty([nkp, self.system.getNAtoms()], dtype=object)
        for id1 in range(nkp):
            for id2 in range(self.system.getNAtoms()):
                SOmats[id1,id2] = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)

        # this can be parallelized over kpoints, but it's not critical since
        # this is only done once during initialization
        for kidx in range(nkp):
            print(f"initializing SO: kpt {kidx+1}/{nkp}")
            # i = g
            for i in range(nbv):
                # j = g'
                for j in range(nbv):
                    if defbool:
                        gikp = self.basis[i] + self.system.kpts[idxGap]
                        gjkp = self.basis[j] + self.system.kpts[idxGap]
                    else:
                        gikp = self.basis[i] + self.system.kpts[kidx]
                        gjkp = self.basis[j] + self.system.kpts[kidx]
                    #gdiff = self.basis[j] - self.basis[i]
                    gdiff = self.basis[i] - self.basis[j]

                    isum = 0.0
                    inm = torch.norm(gikp)
                    jnm = torch.norm(gjkp)

                    if inm < 1e-10 or jnm < 1e-10:
                        # V_SO = 0 if either of these are 0
                        continue

                    #for gp in range(1,ncut):
                    #    r = gp * dr
                    #    isum += (r**2 * dr * self._bessel1(inm*r, 1/(inm*r + 1e-10)) *
                    #            torch.exp(-(r/SOwidth)**2) *
                    #            self._bessel1(jnm*r, 1/(jnm*r + 1e-10)) )
                    isum = self._soIntegral(inm, jnm, rcut, SOwidth)

                    prefactor = 12.0 * np.pi / (inm * jnm)
                    gcross = torch.cross(gikp, gjkp)
                    for alpha in range(self.system.getNAtoms()):
                        if not defbool:
                            gdiffDotTau = torch.dot(gdiff, self.system.atomPos[alpha])
                            sfact_re = 1 / self.system.getCellVolume() * torch.cos(gdiffDotTau)
                            sfact_im = 1 / self.system.getCellVolume() * torch.sin(gdiffDotTau)
                        else:
                            gdiffDotTau = torch.dot(gdiff, self.system.atomPosDef[alpha])
                            sfact_re = 1 / self.system.getCellVolumeDef() * torch.cos(gdiffDotTau)
                            sfact_im = 1 / self.system.getCellVolumeDef() * torch.sin(gdiffDotTau)

                        # build SO matrix
                        # up up
                        # -i * gcp dot S_up,up is pure imag: -i/2 * (gcp.z)
                        real_part = prefactor * isum * 0.5 * gcross[2] * sfact_im
                        im_part = prefactor * isum * -0.5 * gcross[2] * sfact_re
                        SOmats[kidx,alpha][i,j] = torch.complex(real_part, im_part)

                        # dn dn
                        # -i * gcp dot S_dn,dn is pure imag: i/2 * (gcp.z)
                        real_part = prefactor * isum * -0.5 * gcross[2] * sfact_im
                        im_part = prefactor * isum * 0.5 * gcross[2] * sfact_re
                        SOmats[kidx,alpha][i+nbv, j+nbv] = torch.complex(real_part, im_part)

                        # up dn
                        # -i * gcp dot S_up,dn is: -i/2 * (gcp.x) - 1/2 * (gcp.y)
                        real_part = prefactor * isum * (0.5 * gcross[0] * sfact_im -0.5 * gcross[1] * sfact_re)
                        im_part = prefactor * isum * (-0.5 * gcross[0] * sfact_re -0.5 * gcross[1] * sfact_im)
                        SOmats[kidx,alpha][i, j+nbv] = torch.complex(real_part, im_part)

                        # dn up
                        # -i * gcp dot S_dn,up is: -i/2 * (gcp.x) + 1/2 * (gcp.y)
                        real_part = prefactor * isum * (0.5 * gcross[0] * sfact_im + 0.5 * gcross[1] * sfact_re)
                        im_part = prefactor * isum * (-0.5 * gcross[0] * sfact_re + 0.5 * gcross[1] * sfact_im)
                        SOmats[kidx,alpha][i+nbv, j] = torch.complex(real_part, im_part)

        return SOmats
    

    def initSOmat_fast(self, SOwidth=0.7, defbool=False, idxGap=None):
        """
        Calculates the SO integral Vso(K,K') = integral from 0 t0 infinity of
        dr*r^2*j1(Kr)*exp^(-(r/0.7)^2)*j1(K'r) where j1 is the 1st bessel function,
        K = kpoint + basisVector and exp^(-(r/0.7)^2) is the  spin-orbit potential
        excluding the variable "a" parameter. Then builds the SO matrix components
        corresponding to every atom type at each kpoint. WARNING: might consume
        significant memory. You are storing natom * nkpt complex matrices of dimension
        (2*nbasis) x (2*nbasis). Format of output is SOmats[kidx, atomidx] = SOmatrix.
        THIS OUTPUTS NUMPY ndarrays, not torch tensors!

        This function is a little bit of a messy mixture of numpy ndarray and
        torch tensors, which are not super compatible. For now, I think it has
        to be like this because we need numpy/scipy functions for vectorization, 
        but the default self.system objects such as the basis/kpts are natively in 
        torch datatypes. Be careful if editing, because torch tensors and ndarrays 
        can behave differently in subtle ways (i.e. make sure you really understand the code).
        """
        nbv = self.basis.shape[0]
        # set integral dr ~ 0.0089 Bohr at 25 Hartree energy cutoff
        #dr = 2*np.pi / (100 * torch.norm(self.basis[-1]))
        # set radial cutoff ~ 4.2488 Bohr; V(rcut) = 1e-16 for default SOwidth
        rcut = np.sqrt(SOwidth**2 * 16 * np.log(10.0))
        #ncut = int(rcut/dr)
        
        if defbool:
            nkp = 1  # to allow for deformation calcs at a single kpoint
            if idxGap is None:
                raise RuntimeError("need to specify kpt idx of gap in deformed calc")
        else:
            nkp = self.system.getNKpts()
        
        # this can be parallelized over kpoints, but it's not critical since
        # this is only done once during initialization
        SOmats_4d = np.zeros((nkp, self.system.getNAtoms(), 2*nbv, 2*nbv), dtype=np.complex128)
        for kidx in range(nkp):
            self.initSOmat_fast_oneKpt(kidx, SOmats_4d[kidx], SOwidth, defbool, idxGap)
            gc.collect()

        return SOmats_4d


    def initSOmat_fast_oneKpt(self, kidx, SOmats_oneKpt_toFill, SOwidth=0.7, defbool=False, idxGap=None):
        """
        Calculates the SO integral Vso(K,K') = integral from 0 t0 infinity of
        dr*r^2*j1(Kr)*exp^(-(r/0.7)^2)*j1(K'r) where j1 is the 1st bessel function,
        K = kpoint + basisVector and exp^(-(r/0.7)^2) is the  spin-orbit potential
        excluding the variable "a" parameter. 
        
        Then builds the SO matrix components corresponding to every atom type at 
        only one kpoint as indexed by kidx. Storing natom complex matrices of dimension
        (2*nbasis) x (2*nbasis). Format of output is SOmats_oneKpt[atomidx] = SOmatrix.

        This function is a little bit of a messy mixture of numpy ndarray and
        torch tensors, which are not super compatible. For now, I think it has
        to be like this because we need numpy/scipy functions for vectorization, 
        but the default self.system objects such as the basis/kpts are natively in 
        torch datatypes. Be careful if editing, because torch tensors and ndarrays 
        can behave differently in subtle ways (i.e. make sure you really understand the code).
        """
        nbv = self.basis.shape[0]
        # set integral dr ~ 0.0089 Bohr at 25 Hartree energy cutoff
        #dr = 2*np.pi / (100 * torch.norm(self.basis[-1]))
        # set radial cutoff ~ 4.2488 Bohr; V(rcut) = 1e-16 for default SOwidth
        rcut = np.sqrt(SOwidth**2 * 16 * np.log(10.0))
        #ncut = int(rcut/dr)
        
        if defbool:
            nkp = 1  # to allow for deformation calcs at a single kpoint
            if idxGap is None:
                raise RuntimeError("need to specify kpt idx of gap in deformed calc")
        else:
            nkp = self.system.getNKpts()

        print(f"initializing SO: kpt {kidx+1}/{nkp}")
        sys.stdout.flush()
        if defbool:
            gikp = self.basis + torch.stack([self.system.kpts[idxGap]] * nbv, dim=0)
            gjkp = self.basis + torch.stack([self.system.kpts[idxGap]] * nbv, dim=0)
        else:
            gikp = self.basis + torch.stack([self.system.kpts[kidx]] * nbv, dim=0)
            gjkp = self.basis + torch.stack([self.system.kpts[kidx]] * nbv, dim=0)
        gdiff = torch.stack([self.basis]*nbv, dim=1) - self.basis.repeat(nbv, 1, 1)

        gikp = gikp.numpy(force=True)
        gjkp = gjkp.numpy(force=True)
        inm = np.linalg.norm(gikp, axis=1)
        jnm = np.linalg.norm(gjkp, axis=1)


        isum = self._soIntegral_vect(inm, jnm, rcut, SOwidth)
        #isum = self._soIntegral_dan(inm, jnm, SOwidth) # for testing, use the prev line for real calcs

        #prefactor = 12.0 * np.pi / (inm[:, np.newaxis] * jnm)
        prefactor = np.zeros([nbv,nbv], dtype=float)
        denom = inm[:, np.newaxis] * jnm
        ids = np.nonzero(denom)
        prefactor[ids] = 12.0 * np.pi / denom[ids]

        gcross = np.cross(np.stack([gikp]*nbv, axis=1), 
                            np.stack([gjkp]*nbv, axis=0), axisa=-1, axisb=-1, axisc=-1)

        for alpha in range(self.system.getNAtoms()):
            gdiffDotTau = gdiff * self.system.atomPos[alpha]
            gdiffDotTau = np.sum(gdiffDotTau.numpy(force=True), axis=2)
            sfact_re = 1 / self.system.getCellVolume() * np.cos(gdiffDotTau)
            sfact_im = 1 / self.system.getCellVolume() * np.sin(gdiffDotTau)

            # build SO matrix
            # up up
            # -i * gcp dot S_up,up is pure imag: -i/2 * (gcp.z)
            real_part = prefactor * isum * 0.5 * gcross[:,:, 2] * sfact_im
            im_part = prefactor * isum * -0.5 * gcross[:,:, 2] * sfact_re
            SOmats_oneKpt_toFill[alpha, :nbv, :nbv] = real_part + 1j * im_part

            # dn dn
            # -i * gcp dot S_dn,dn is pure imag: i/2 * (gcp.z)
            real_part = prefactor * isum * -0.5 * gcross[:,:, 2] * sfact_im
            im_part = prefactor * isum * 0.5 * gcross[:,:, 2] * sfact_re
            SOmats_oneKpt_toFill[alpha, nbv:, nbv:] = real_part + 1j * im_part

            # up dn
            # -i * gcp dot S_up,dn is: -i/2 * (gcp.x) - 1/2 * (gcp.y)
            real_part = prefactor * isum * (0.5 * gcross[:,:, 0] * sfact_im -0.5 * gcross[:,:, 1] * sfact_re)
            im_part = prefactor * isum * (-0.5 * gcross[:,:, 0] * sfact_re -0.5 * gcross[:,:, 1] * sfact_im)
            SOmats_oneKpt_toFill[alpha, :nbv, nbv:] = real_part + 1j * im_part

            # dn up
            # -i * gcp dot S_dn,up is: -i/2 * (gcp.x) + 1/2 * (gcp.y)
            real_part = prefactor * isum * (0.5 * gcross[:,:, 0] * sfact_im + 0.5 * gcross[:,:, 1] * sfact_re)
            im_part = prefactor * isum * (-0.5 * gcross[:,:, 0] * sfact_re + 0.5 * gcross[:,:, 1] * sfact_im)
            SOmats_oneKpt_toFill[alpha, nbv:, :nbv] = real_part + 1j * im_part
        return


    def initNLmat(self, width1=1.0, width2=1.0, shift=1.5, defbool=False, idxGap=None):
        """
        Calculates the nonlocal integrals V_{l=1}(K,K') = 
        integral from 0 to infinity of
        dr*r^2*j1(Kr)* [exp^(-(r/width1)^2)] *j1(K'r) and
        dr*r^2*j1(Kr)* [exp^(-((r-shift)/width2)^2)] *j1(K'r)
        where j1 is the 1st bessel function.
        Then builds the Nonlocal matrix components
        corresponding to every atom type at each kpoint for each integral. 
        WARNING: might consume
        significant memory. You are storing natom * nkpt * 2 complex matrices of dimension
        (2*nbasis) x (2*nbasis). Format of output is SOmats[kidx, atomidx,{0,1}] = NLmatrix{0,1}
        """
        nbv = self.basis.shape[0]
        # set integral dr ~ 0.0089 Bohr at 25 Hartree energy cutoff
        dr = 2*np.pi / (100 * torch.norm(self.basis[-1]))
        # set radial cutoff ~ 4.2488 Bohr; V(rcut) = 1e-16 for default SOwidth
        rcut = np.sqrt(width1*width2 * 16 * np.log(10.0))
        ncut = int(rcut/dr)
        
        if defbool:
            nkp = 1  # to allow for deformation calcs at a single kpoint
            if idxGap is None:
                raise RuntimeError("need to specify kpt idx of gap in deformed calc")
        else:
            nkp = self.system.getNKpts()
        
        NLmats = np.empty([nkp, self.system.getNAtoms(), 2], dtype=object)
        for id1 in range(nkp):
            for id2 in range(self.system.getNAtoms()):
                for id3 in [0,1]:
                    NLmats[id1,id2,id3] = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)

        # this can be parallelized over kpoints, but it's not critical since
        # this is only done once during initialization
        for kidx in range(nkp):
            print(f"initializing NL pots: kpt {kidx+1}/{nkp}")
            # i = g
            for i in range(nbv):
                # j = g'
                for j in range(nbv):
                    if defbool:
                        gikp = self.basis[i] + self.system.kpts[idxGap]
                        gjkp = self.basis[j] + self.system.kpts[idxGap]
                    else:
                        gikp = self.basis[i] + self.system.kpts[kidx]
                        gjkp = self.basis[j] + self.system.kpts[kidx]
                    #gdiff = self.basis[j] - self.basis[i]
                    gdiff = self.basis[i] - self.basis[j]

                    isum1 = 0.0
                    isum2 = 0.0
                    inm = torch.norm(gikp)
                    jnm = torch.norm(gjkp)

                    if inm < 1e-10 or jnm < 1e-10:
                        # V_SO = 0 if either of these are 0
                        continue

                    for gp in range(1,ncut):
                        r = gp * dr
                        isum1 += (r**2 * dr * self._bessel1(inm*r, 1/(inm*r + 1e-10)) *
                                torch.exp(-(r/width1)**2) *
                                self._bessel1(jnm*r, 1/(jnm*r + 1e-10)) )
                        isum2 += (r**2 * dr * self._bessel1(inm*r, 1/(inm*r + 1e-10)) *
                                  torch.exp(-((r-shift)/width2)**2) *
                                  self._bessel1(jnm*r, 1/(jnm*r + 1e-10))  )

                    prefactor = 12.0 * np.pi / (inm * jnm)
                    gdot = torch.dot(gikp, gjkp)

                    for alpha in range(self.system.getNAtoms()):
                        if not defbool:
                            gdiffDotTau = torch.dot(gdiff, self.system.atomPos[alpha])
                            sfact_re = 1 / self.system.getCellVolume() * torch.cos(gdiffDotTau)
                            sfact_im = 1 / self.system.getCellVolume() * torch.sin(gdiffDotTau)
                        else:
                            gdiffDotTau = torch.dot(gdiff, self.system.atomPosDef[alpha])
                            sfact_re = 1 / self.system.getCellVolumeDef() * torch.cos(gdiffDotTau)
                            sfact_im = 1 / self.system.getCellVolumeDef() * torch.sin(gdiffDotTau)
                    
                        # This potential is block diagonal on spin
                        # up up, 1st integral
                        real_part = prefactor * isum1 * gdot * sfact_re
                        im_part = prefactor * isum1 * gdot * sfact_im
                        NLmats[kidx,alpha,0][i,j] = torch.complex(real_part, im_part)
                        # 2nd integral
                        real_part = prefactor * isum2 * gdot * sfact_re
                        im_part = prefactor * isum2 * gdot * sfact_im
                        NLmats[kidx,alpha,1][i,j] = torch.complex(real_part, im_part)

                        # dn dn, 1st integral
                        real_part = prefactor * isum1 * gdot * sfact_re
                        im_part = prefactor * isum1 * gdot * sfact_im
                        NLmats[kidx,alpha,0][i+nbv, j+nbv] = torch.complex(real_part, im_part)
                        # 2nd integral
                        real_part = prefactor * isum2 * gdot * sfact_re
                        im_part = prefactor * isum2 * gdot * sfact_im
                        NLmats[kidx,alpha,1][i+nbv, j+nbv] = torch.complex(real_part, im_part)

        return NLmats


    def initNLmat_fast(self, width1=1.0, width2=1.0, shift=1.5, defbool=False, idxGap=None):
        """
        Calculates the nonlocal integrals V_{l=1}(K,K') = 
        integral from 0 to infinity of
        dr*r^2*j1(Kr)* [exp^(-(r/width1)^2)] *j1(K'r) and
        dr*r^2*j1(Kr)* [exp^(-((r-shift)/width2)^2)] *j1(K'r)
        where j1 is the 1st bessel function.
        Then builds the Nonlocal matrix components
        corresponding to every atom type at each kpoint for each integral. 
        WARNING: might consume
        significant memory. You are storing natom * nkpt * 2 complex matrices of dimension
        (2*nbasis) x (2*nbasis). Format of output is SOmats[kidx, atomidx,{0,1}] = NLmatrix{0,1}.
        THIS OUTPUTS NUMPY ndarrays, not torch tensors!

        This function is a little bit of a messy mixture of numpy ndarray and
        torch tensors, which are not super compatible. For now, I think it has
        to be like this because we need numpy/scipy functions for stable integration, 
        but the default self.system objects such as the basis/kpts are natively in 
        torch datatypes. Be careful if editing, because torch tensors and ndarray can behave
        differently in subtle ways (i.e. make sure you really understand the code).
        """
        nbv = self.basis.shape[0]
        # set integral dr ~ 0.0089 Bohr at 25 Hartree energy cutoff
        #dr = 2*np.pi / (100 * torch.norm(self.basis[-1]))
        # set radial cutoff ~ 4.2488 Bohr; V(rcut) = 1e-16 for default SOwidth
        rcut = np.sqrt(width1*width2 * 16 * np.log(10.0))
        #ncut = int(rcut/dr)
        
        if defbool:
            nkp = 1  # to allow for deformation calcs at a single kpoint
            if idxGap is None:
                raise RuntimeError("need to specify kpt idx of gap in deformed calc")
        else:
            nkp = self.system.getNKpts()
        
        # this can be parallelized over kpoints, but it's not critical since
        # this is only done once during initialization
        NLmats_5d = np.zeros((nkp, self.system.getNAtoms(), 2, 2*nbv, 2*nbv), dtype=np.complex128)
        for kidx in range(nkp):
            self.initNLmat_fast_oneKpt(kidx, NLmats_5d[kidx], width1, width2, shift, defbool, idxGap)
            gc.collect()

        return NLmats_5d
    

    def initNLmat_fast_oneKpt(self, kidx, NLmats_oneKpt_toFill, width1=1.0, width2=1.0, shift=1.5, defbool=False, idxGap=None):
        """
        Calculates the nonlocal integrals V_{l=1}(K,K') = 
        integral from 0 to infinity of
        dr*r^2*j1(Kr)* [exp^(-(r/width1)^2)] *j1(K'r) and
        dr*r^2*j1(Kr)* [exp^(-((r-shift)/width2)^2)] *j1(K'r)
        where j1 is the 1st bessel function.
        Then builds the Nonlocal matrix components
        corresponding to every atom type at each kpoint for each integral. 
        
        WARNING: might consume significant memory. You are storing 
        natom * 2 complex matrices of dimension
        (2*nbasis) x (2*nbasis). Format of output is SOmats[atomidx,{0,1}] = NLmatrix{0,1}.
        THIS OUTPUTS NUMPY ndarrays, not torch tensors!

        This function is a little bit of a messy mixture of numpy ndarray and
        torch tensors, which are not super compatible. For now, I think it has
        to be like this because we need numpy/scipy functions for stable integration, 
        but the default self.system objects such as the basis/kpts are natively in 
        torch datatypes. Be careful if editing, because torch tensors and ndarray can behave
        differently in subtle ways (i.e. make sure you really understand the code).
        """
        nbv = self.basis.shape[0]
        # set integral dr ~ 0.0089 Bohr at 25 Hartree energy cutoff
        #dr = 2*np.pi / (100 * torch.norm(self.basis[-1]))
        # set radial cutoff ~ 4.2488 Bohr; V(rcut) = 1e-16 for default SOwidth
        rcut = np.sqrt(width1*width2 * 16 * np.log(10.0))
        #ncut = int(rcut/dr)
        
        if defbool:
            nkp = 1  # to allow for deformation calcs at a single kpoint
            if idxGap is None:
                raise RuntimeError("need to specify kpt idx of gap in deformed calc")
        else:
            nkp = self.system.getNKpts()
        
        print(f"initializing NL pots: kpt {kidx+1}/{nkp}")
        sys.stdout.flush()
        if defbool:
            gikp = self.basis + torch.stack([self.system.kpts[idxGap]] * nbv, dim=0)
            gjkp = self.basis + torch.stack([self.system.kpts[idxGap]] * nbv, dim=0)
        else:
            gikp = self.basis + torch.stack([self.system.kpts[kidx]] * nbv, dim=0)
            gjkp = self.basis + torch.stack([self.system.kpts[kidx]] * nbv, dim=0)
        gdiff = torch.stack([self.basis]*nbv, dim=1) - self.basis.repeat(nbv, 1, 1)

        gikp = gikp.numpy(force=True)
        gjkp = gjkp.numpy(force=True)
        inm = np.linalg.norm(gikp, axis=1)
        jnm = np.linalg.norm(gjkp, axis=1)

        t1 = time.time()
        isum1 = self._soIntegral_vect(inm, jnm, rcut, width1)
        #isum1 = self._soIntegral_dan(inm, jnm, width1)  # for testing only
        t2 = time.time()
        # print(f"time int1: {t2-t1}")
        isum2 = self._nlIntegral_vect(inm, jnm, rcut, width2, shift)
        #isum2 = self._nlIntegral_dan(inm, jnm, width2, shift)  # for testing
        t3 = time.time()
        # print(f"time int2: {t3-t2}")

        #gdot = torch.dot(gikp, gjkp)
        # this tensordot call is like mat[i,j] = sum_k gikp[i,k] * gjkp[j,k]
        gdot = np.tensordot(gikp, gjkp, axes=[[1],[1]])      
        #prefactor = 12.0 * np.pi / (inm[:, np.newaxis] * jnm)
        prefactor = np.zeros([nbv,nbv], dtype=float)
        denom = inm[:, np.newaxis] * jnm
        ids = np.nonzero(denom)
        prefactor[ids] = 12.0 * np.pi / denom[ids]

        for alpha in range(self.system.getNAtoms()):
            gdiffDotTau = gdiff * self.system.atomPos[alpha]
            gdiffDotTau = np.sum(gdiffDotTau.numpy(force=True), axis=2)
            sfact_re = 1 / self.system.getCellVolume() * np.cos(gdiffDotTau)
            sfact_im = 1 / self.system.getCellVolume() * np.sin(gdiffDotTau)
            
        
            # This potential is block diagonal on spin
            # up up, 1st integral
            real_part = prefactor * isum1 * gdot * sfact_re
            im_part = prefactor * isum1 * gdot * sfact_im
            NLmats_oneKpt_toFill[alpha,0, :nbv, :nbv] = real_part + 1j* im_part
            # 2nd integral
            real_part = prefactor * isum2 * gdot * sfact_re
            im_part = prefactor * isum2 * gdot * sfact_im
            NLmats_oneKpt_toFill[alpha,1, :nbv, :nbv] = real_part + 1j * im_part

            # dn dn, 1st integral
            real_part = prefactor * isum1 * gdot * sfact_re
            im_part = prefactor * isum1 * gdot * sfact_im
            NLmats_oneKpt_toFill[alpha,0, nbv:, nbv:] = real_part + 1j * im_part
            # 2nd integral
            real_part = prefactor * isum2 * gdot * sfact_re
            im_part = prefactor * isum2 * gdot * sfact_im
            NLmats_oneKpt_toFill[alpha,1, nbv:, nbv:] = real_part + 1j * im_part
        return
    
    
    def buildSOmat(self, kidx, preComp_SOmats_kidx=None, addMat=None):
        """
        Build the final SO mat for a given kpoint (specified by its kidx).
        Using the cached SOmats at the kidx (preComp_SOmats_kidx, the 
        actual matrices), this function just multiplies by the 
        current values of the PPparams, and then sums over all atoms.
        "addMat" can be set to be a partially constructed Hamiltonian matrix, to
        which the local potential can be added. Might help save slightly on memory.
        """
        if preComp_SOmats_kidx is None: 
            if self.NNConfig['num_cores'] != 0:
                print("WARNING: Didn't find precomputed SOmats stored in shared memory. This buildSOmat could drastically slow down multiprocessing parallelization.")
            if self.SOmats is None: 
                print("DANGEROUS!!! WARNING. Attempting to build the SOmat, but 1) no precomputed SOmats are stored in shared memory, 2) no cached SOmatrices in the ham class. \nCalculating the SOmats for each kpt on the fly. ")
                SOmats_kidx = np.zeros((self.system.getNAtoms(), 2*self.basis.shape[0], 2*self.basis.shape[0]), dtype=np.complex128)
                self.initSOmat_fast_oneKpt(kidx, SOmats_kidx)
            else: 
                SOmats_kidx = self.SOmats[kidx]
        else: 
            SOmats_kidx = preComp_SOmats_kidx

        nbv = self.basis.shape[0]
        if addMat is not None:
            assert addMat.shape[0] == 2*nbv
            assert addMat.shape[1] == 2*nbv
            SOmatf = addMat
        else:
            SOmatf = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)
        
        for alpha in range(self.system.getNAtoms()):
            if isinstance(SOmats_kidx[alpha], torch.Tensor):
                tmp = SOmats_kidx[alpha]
            else:
                tmp = torch.tensor(SOmats_kidx[alpha])

            SOmatf = SOmatf + tmp * self.PPparams[self.system.atomTypes[alpha]][5]

        return SOmatf
    

    def buildNLmat(self, kidx, preComp_NLmats_kidx=None, addMat=None):
        """
        Build the final nonlocal mat for a given kpoint (specified by its kidx).
        Using the cached NLmats at this kidx (preComp_NLmats_kidx, the actual
        matrices), this function just multiplies by the 
        current values of the PPparams, and then sums over all atoms.
        "addMat" can be set to be a partially constructed Hamiltonian matrix, to
        which the local potential can be added. Might help save slightly on memory.
        """
        if preComp_NLmats_kidx is None: 
            if self.NNConfig['num_cores'] != 0:
                print("WARNING: Didn't find precomputed NLmats stored in shared memory. This buildNLmat could drastically slow down multiprocessing parallelization.")
            if self.NLmats is None: 
                print("DANGEROUS!!! WARNING. Attempting to build the NLmat, but 1) no precomputed NLmats are stored in shared memory, 2) no cached NL matrices in the ham class. \nCalculating the NLmats on the fly. ")
                NLmats_kidx = np.zeros((self.system.getNAtoms(), 2, 2*self.basis.shape[0], 2*self.basis.shape[0]), dtype=np.complex128)
                self.initNLmat_fast_oneKpt(kidx, NLmats_kidx)
            else: 
                NLmats_kidx = self.NLmats[kidx]
        else: 
            NLmats_kidx = preComp_NLmats_kidx
        
        nbv = self.basis.shape[0]
        if addMat is not None:
            assert addMat.shape[0] == 2*nbv
            assert addMat.shape[1] == 2*nbv
            NLmatf = addMat
        else:
            NLmatf = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)
        
        for alpha in range(self.system.getNAtoms()):
            if isinstance(NLmats_kidx[alpha,0], torch.Tensor):
                tmp1 = NLmats_kidx[alpha,0]
            else:
                tmp1 = torch.tensor(NLmats_kidx[alpha,0])
            if isinstance(NLmats_kidx[alpha,1], torch.Tensor):
                tmp2 = NLmats_kidx[alpha,1]
            else:
                tmp2 = torch.tensor(NLmats_kidx[alpha,1])

            NLmatf = (NLmatf + tmp1 * self.PPparams[self.system.atomTypes[alpha]][6]
                             + tmp2 * self.PPparams[self.system.atomTypes[alpha]][7] )

        return NLmatf


    def calcEigValsAtK(self, kidx, cachedMats_info=None, requires_grad=True, verbosity=0, parallelization=True, writeEVecsToFile=False, writeEVecsFolderName=""):
        '''
        This function builds the Htot at a certain kpoint that is given as the input, 
        digonalizes the Htot, and obtains the eigenvalues at this kpoint. 

        By default, it also calculates the eigenvectors in order to re-order the eigenvalues. 
        '''

        nbands = self.system.nBands

        if (cachedMats_info is None) and (self.SObool==False):    # proceed as normal. Won't even go into buildSO or buildNL. Need to pass None into buildSO and buildNL
            preComp_SOmats_kidx = None
            preComp_NLmats_kidx = None
        elif (cachedMats_info is None) and (self.SObool==True):   # no cached matrices in the shared memory
            preComp_SOmats_kidx = None
            preComp_NLmats_kidx = None     # functions buildSOmat and buildNLmat will handle these cases
        elif (cachedMats_info is not None): 
            # Load SOmats and NLmats from shared memory
            start_time = time.time() if self.NNConfig['runtime_flag'] else None
            shm_SOmats = shared_memory.SharedMemory(name=f"SOmats_{self.iSystem}_{kidx}")
            preComp_SOmats_kidx = np.ndarray(cachedMats_info[f"SO_{self.iSystem}_{kidx}"]['shape'], dtype=cachedMats_info[f"SO_{self.iSystem}_{kidx}"]['dtype'], buffer=shm_SOmats.buf)
            shm_NLmats = shared_memory.SharedMemory(name=f"NLmats_{self.iSystem}_{kidx}")
            preComp_NLmats_kidx = np.ndarray(cachedMats_info[f"NL_{self.iSystem}_{kidx}"]['shape'], dtype=cachedMats_info[f"NL_{self.iSystem}_{kidx}"]['dtype'], buffer=shm_NLmats.buf)
            end_time = time.time() if self.NNConfig['runtime_flag'] else None
            print(f"Loading shared memory, elapsed time: {(end_time - start_time):.2f} seconds") if self.NNConfig['runtime_flag'] else None
        else: 
            raise ValueError("Error in calcEigValsAtK, in loading the cached SO and NL mats. ")

        start_time = time.time() if self.NNConfig['runtime_flag'] else None
        H = self.buildHtot(kidx, preComp_SOmats_kidx, preComp_NLmats_kidx, requires_grad)
        if not requires_grad: 
            H = H.detach()
        end_time = time.time() if self.NNConfig['runtime_flag'] else None
        print(f"Building Htot, elapsed time: {(end_time - start_time):.2f} seconds") if self.NNConfig['runtime_flag'] else None

        start_time = time.time() if self.NNConfig['runtime_flag'] else None
        if (not self.coupling) and (not self.physicalBandOrdering):     # Fastest. Don't need any eigenvector information. 
            energies = torch.linalg.eigvalsh(H)
            energiesEV = energies * AUTOEV

            if not self.SObool:
                # 2-fold degeneracy for spin. Not sure why this is necessary, but
                # it is included in Tommy's code...
                energiesEV = energiesEV.repeat_interleave(2)
                # dont need to interleave eigenvecs (if stored) since we only
                # store the vb and cb anyways.
            energiesEV = energiesEV[:nbands]

            # reorder the energies according to the manual input in self.system.bandOrderMatrix
            energiesEV = energiesEV[self.system.bandOrderMatrix[kidx, :]]


        elif (self.coupling) and (not self.physicalBandOrdering):       # NOTE!!! This is not made compatible with the physicalBandOrdering parameter.  
            # this will be slower than necessary, since torch seems to only support
            # full diagonalization including all eigenvectors. 
            # If computing couplings, it would be faster to
            # implement a custom torch diagonalization wrapper
            # that uses scipy under the hood to allow for better partial
            # diagonalization algorithms (e.g. the ?heevr driver).
            ens, vecs = torch.linalg.eigh(H)
            energiesEV = ens * AUTOEV

            if not self.SObool:
                # 2-fold degeneracy for spin. Not sure why this is necessary, but
                # it is included in Tommy's code...
                energiesEV = energiesEV.repeat_interleave(2)
                # dont need to interleave eigenvecs (if stored) since we only
                # store the vb and cb anyways.
            energiesEV = energiesEV[:nbands]

            self.vb_vecs[kidx].append(vecs[:, self.idx_vb])
            self.cb_vecs[kidx].append(vecs[:, self.idx_cb])
            # NOTE!!! that using the eigenvectors with torch autodiff can result in non-uniqueness
            # an instability if there are degenerate eigenvalues. 

            # To avoid gauge phase-dependent values of the coupling when we
            # have degenerate electronic states, we collect all degenerate bands,
            # to compute their couplings and THEN average the couplings. This is
            # different than doing an average over degenerate eigenvectors first, 
            # which is wrong (results will depend on arbitrary phase in degenerate subspace).
            ctr = 1
            for idx in range(self.idx_vb-1, 0, -1):
                if abs(ens[self.idx_vb] - ens[idx]) < 1e-5:
                    # this describes a degenerate state as begin within .01 meV (adopted from EPW source)
                    self.vb_vecs[kidx].append(vecs[:, idx])
                    ctr += 1
                else:
                    break

            if ctr == 1 and self.SObool and verbosity >= 2:
                print(f"\nWARNING: spin-orbit calc but vb spin states are not degenerate to 1e-10, kidx={kidx}\n")
            if verbosity >= 3:
                print(f"kidx={kidx}, vb_vec[0:5]= {self.vb_vecs[kidx, :5]}")

            ctr = 1
            for idx in range(self.idx_cb+1, self.system.nBands):
                if abs(ens[self.idx_cb] - ens[idx]) < 1e-5:
                    # this describes a degenerate state as begin within .01 meV (adopted from EPW source)
                    self.cb_vecs[kidx].append(vecs[:, idx])
                    ctr += 1
                else:
                    break

            if ctr == 1 and self.SObool and verbosity >= 2:
                print(f"\nWARNING: spin-orbit calc but cb spin states are not degenerate to 1e-10, kidx={kidx}\n")
            if verbosity >= 3:
                print(f"kidx={kidx}, cb_vec[0:5]= {self.cb_vecs[kidx, :5]}")
        elif (not self.coupling) and (self.physicalBandOrdering): 
            ens, evecs = torch.linalg.eigh(H)
            energiesEV = ens * AUTOEV
            evecs = evecs.detach().numpy()

            if not self.SObool:
                # 2-fold degeneracy for spin. Not sure why this is necessary, but
                # it is included in Tommy's code...
                energiesEV = energiesEV.repeat_interleave(2)
                # dont need to interleave eigenvecs (if stored) since we only
                # store the vb and cb anyways.
            energiesEV = energiesEV[:nbands]

            write_evec_list = []
            for bandIdx in range(nbands):  
                eVecKey = f"currIter_eVec_{kidx}_{bandIdx}"
                eVecValue = {'dtype': evecs[:, bandIdx].dtype, 'shape': evecs[:, bandIdx].shape}
                if eVecKey in self.eVec_info:    # Check if the shm object already exists
                    shm_obj = shared_memory.SharedMemory(name=eVecKey)
                    shm_obj.close()
                    shm_obj.unlink()
                    del shm_obj
                    del self.eVec_info[eVecKey]
                self.eVec_info[eVecKey] = eVecValue

                # Move eVecs to shared memory
                shm = shared_memory.SharedMemory(create=True, size=evecs[:, bandIdx].nbytes, name=eVecKey)
                tmp_arr = np.copy(np.ndarray(evecs[:, bandIdx].shape, dtype=evecs[:, bandIdx].dtype, buffer=shm.buf))  # Create a NumPy array backed by shared memory
                tmp_arr[:] = evecs[:, bandIdx]
                
                # print(f"Eigenvector {bandIdx}: ")
                # print(evecs[:, bandIdx])
                write_evec_list.append(evecs[:, bandIdx])
            if writeEVecsToFile: 
                np.savez(f"{writeEVecsFolderName}eigVec_k{kidx}.npz", *write_evec_list)

            newOrder = self.reorder_bands(kidx, parallelization)
            energiesEV = energiesEV[newOrder]
        else: 
            raise NotImplementedError("Currently function calcEigValsAtK doesn't support both self.physicalBandOrdering and self.coupling. ")

        end_time = time.time() if self.NNConfig['runtime_flag'] else None
        print(f"Diagonalization (in some cases includes eigvecs), elapsed time: {(end_time - start_time):.2f} seconds") if self.NNConfig['runtime_flag'] else None

        '''
        # Testing with random matrix
        start_time = time.time() if self.NNConfig['runtime_flag'] else None
        test_H = torch.randn(2000, 2000, dtype=torch.complex128)
        eigenvalues = torch.linalg.eigvalsh(test_H)
        end_time = time.time() if self.NNConfig['runtime_flag'] else None
        total_time = end_time - start_time
        print(f"Generating and diagonalizing a random 2000x2000 matrix. Time: {total_time:.2f} seconds") if self.NNConfig['runtime_flag'] else None
        '''
        
        print(f"On this subprocess, we are working with kIdx {kidx}. The current self.eVec_info has length {len(self.eVec_info)} and looks like the following. We should see the length increase although multiprocessing. ")
        if not requires_grad: 
            energiesEV = energiesEV.detach()
        return energiesEV


    def overlapMatrix(self, shm_key_kIdx1, shm_key_kIdx2, verbosity=0): 
        """
        shm_key_kIdx1 and 2 would look like the following: 
        f"currIter_eVec_{kidx}"
        f"prevIter_eVec_{kidx}" 
        
        We will need to concatenate f"_{bandIdx}" to its end to use it. 
        """
        nbands = self.system.nBands
        evec_list_kIdx1 = []
        evec_list_kIdx2 = []
        print(f"Loading eigenvectors from shm with names: {shm_key_kIdx1}_xxx(bands) and {shm_key_kIdx2}_xxx(bands). ") if verbosity>0 else None
        for bandIdx in range(nbands): 
            # Load eigenvectors from shm objects, assuming they all exist
            callName = shm_key_kIdx1 + f"_{bandIdx}"
            try:
                shm_obj = shared_memory.SharedMemory(name=callName)
                tmp_eVec = np.copy(np.ndarray(self.eVec_info[callName]['shape'], dtype=self.eVec_info[callName]['dtype'], buffer=shm_obj.buf))
                evec_list_kIdx1.append(tmp_eVec)
            except FileNotFoundError:
                print(f"Shared memory object {callName} does not exist.")
                continue

            callName = shm_key_kIdx2 + f"_{bandIdx}"
            try:
                shm_obj = shared_memory.SharedMemory(name=callName)
                tmp_eVec = np.copy(np.ndarray(self.eVec_info[callName]['shape'], dtype=self.eVec_info[callName]['dtype'], buffer=shm_obj.buf))
                evec_list_kIdx2.append(tmp_eVec)
            except FileNotFoundError:
                print(f"Shared memory object {callName} does not exist.")
                continue
        
        M = np.zeros((nbands, nbands), dtype=np.float64)
        for i in range(nbands):
            for j in range(nbands): 
                M[i,j] = overlap_2eigVec(evec_list_kIdx1[i], evec_list_kIdx2[j])
       
        if verbosity>1: 
            diff = M - np.eye(M.shape[0])
            if np.allclose(diff, 0, atol=1e-4):
                print("YES! The overlap matrix is effectively identity.")
            else:
                print("NO! The overlap matrix is not close to identity.")
        if verbosity>2: 
            np.set_printoptions(formatter={'float': lambda x: "{: .3f}".format(x)})
            print(M)
        # np.savetxt(f"CALCS/CsPbI3_test/results/overlap_{kIdx1}_{kIdx2}.dat", M, fmt='%.4f')

        return M


    def reorder_bands(self, kIdx, parallelization=True, verbosity=0): 
        """
        Reorder bands at kIdx based on cost matrices.
        If each kpoint is diagonalized in parallel, then we don't have information about
        the overlap matrix between the current k-point and others. And thus, we rely on 
        overlap matrix calculated against the previous iteration. 

        If no parallelization is implemented, we can access all previous k-points since 
        they are calculated and diagonalized in order. 

        In either case, the band order at the kIdx is computed by solving the linear sum 
        assignment problem between the following chain of k-points:
        a) prevIter_eVec_{0}, ..., prevIter_eVec_{kIdx-1}, currIter_eVec_{kIdx}
        b) currIter_eVec_{0}, ..., currIter_eVec_{kIdx-1}, currIter_eVec_{kIdx}
        """
        print(f"We are re-ordering the bands at kIdx {kIdx}. ")
        nbands = self.system.nBands
        newOrder = np.arange(nbands)
        if kIdx == 0: 
            return newOrder

        if parallelization: 
            keys_to_check = [f"prevIter_eVec_{k}_{b}" for k in range(kIdx) for b in range(nbands)] + [f"currIter_eVec_{kIdx}_{b}" for b in range(nbands)]
            if not all(key in self.eVec_info for key in keys_to_check):
                print("Some eigenvectors are missing in the shared memory INFO DICT. I can't do band re-ordering. ")
                return newOrder
            '''
            if not all(key in self.shm_eVec for key in keys_to_check):
                print("Some eigenvectors are missing in the shared memory SHM DICT. I can't do band re-ordering. ")
                return newOrder
            '''

            for k in range(kIdx-1): 
                costMatrix = self.overlapMatrix(f"prevIter_eVec_{k}", f"prevIter_eVec_{k+1}", verbosity)
                _, col_ind = linear_sum_assignment(costMatrix, maximize=True)
                newOrder = newOrder[col_ind]
            
            costMatrix = self.overlapMatrix(f"prevIter_eVec_{kIdx-1}", f"currIter_eVec_{kIdx}", verbosity)
            _, col_ind = linear_sum_assignment(costMatrix, maximize=True)
            newOrder = newOrder[col_ind]

            print(f"New band order at kIdx: {newOrder}") if verbosity>0 else None
            return newOrder
        else: 
            keys_to_check = [f"currIter_eVec_{k}_{b}" for k in range(kIdx+1) for b in range(nbands)]
            if not all(key in self.eVec_info for key in keys_to_check):
                print("Some eigenvectors are missing in the shared memory INFO DICT. I can't do band re-ordering. ")
                return newOrder
            '''
            if not all(key in self.shm_eVec for key in keys_to_check):
                print("Some eigenvectors are missing in the shared memory SHM DICT. I can't do band re-ordering. ")
                return newOrder
            '''

            for k in range(kIdx): 
                costMatrix = self.overlapMatrix(f"currIter_eVec_{k}", f"currIter_eVec_{k+1}", verbosity)
                _, col_ind = linear_sum_assignment(costMatrix, maximize=True)
                newOrder = newOrder[col_ind]

            print(f"New band order at kIdx: {newOrder}") if verbosity>0 else None
            return newOrder


    def _copy_currIter_to_prevIter_shm(self): 
        """
        This routine deals with the shared memory objects of eigenvectors. 
        It copies copy "currIter..." shared memory objects to "prevIter..."
        In both parallel and non-parallel cases, this function should: 
        - NOT be called unless all calcEigValsAtK have been completed for all kpoints
        - be called at the end of every band structure calculation. 
        """
        ########## NOTE: I need to deal with situations where the dictionary entry is None!
        nbands = self.system.nBands
        nkpt = self.system.getNKpts()
        for kidx in range(nkpt):
            for bandIdx in range(nbands):  
                curr_eVecKey = f"currIter_eVec_{kidx}_{bandIdx}"
                prev_eVecKey = f"prevIter_eVec_{kidx}_{bandIdx}"
                if curr_eVecKey not in self.eVec_info:
                    print(f"Skipping copying {curr_eVecKey} to {prev_eVecKey} because it's not found in eVec_info. ")
                    continue
                if prev_eVecKey in self.eVec_info:    # Check if the shm object already exists
                    shm_obj = shared_memory.SharedMemory(name=prev_eVecKey)
                    shm_obj.close()
                    shm_obj.unlink()
                    del shm_obj
                    del self.eVec_info[prev_eVecKey]
                self.eVec_info[prev_eVecKey] = self.eVec_info[curr_eVecKey]
                
                # Read from curr_key shm object
                shm_obj = shared_memory.SharedMemory(name=curr_eVecKey)
                tmp_eVec = np.copy(np.ndarray(self.eVec_info[curr_eVecKey]['shape'], dtype=self.eVec_info[curr_eVecKey]['dtype'], buffer=shm_obj.buf))

                # Create and copy to prev_key shm object
                shm = shared_memory.SharedMemory(create=True, size=tmp_eVec.nbytes, name=prev_eVecKey)
                tmp_arr = np.copy(np.ndarray(tmp_eVec.shape, dtype=tmp_eVec.dtype, buffer=shm.buf))  # Create a NumPy array backed by shared memory
                tmp_arr[:] = tmp_eVec
        return




    def calcBandStruct(self, grad=False, cachedMats_info=None): 
        if grad: 
            return self.calcBandStruct_withGrad(cachedMats_info)
        else: 
            return self.calcBandStruct_noGrad(cachedMats_info)


    def calcBandStruct_withGrad(self, cachedMats_info=None):
        '''
        Multiprocessing is not implemented.
        '''
        
        nbands = self.system.nBands
        nkpt = self.system.getNKpts()
        bandStruct = torch.zeros([nkpt, nbands])
        for kidx in range(nkpt):
            eigValsAtK = self.calcEigValsAtK(kidx, cachedMats_info, requires_grad=True, parallelization=False)
            bandStruct[kidx,:] = eigValsAtK
        self._copy_currIter_to_prevIter_shm()
                    
        return bandStruct


    def calcBandStruct_noGrad(self, cachedMats_info=None):
        """
        Multiprocessing is implemented. However, the returned bandStruct doesn't have gradients.
        """
        
        nbands = self.system.nBands
        nkpt = self.system.getNKpts()

        bandStruct = torch.zeros([nkpt, nbands], requires_grad=False)
        if (self.NNConfig['num_cores']==0):     # No multiprocessing
            for kidx in range(nkpt):
                eigValsAtK = self.calcEigValsAtK(kidx, cachedMats_info, requires_grad=False, parallelization=False)
                
                # !!! FOR TESTING ONLY: 
                # eigValsAtK = self.calcEigValsAtK(kidx, cachedMats_info, requires_grad=False, parallelization=False, writeEVecsToFile=True, writeEVecsFolderName="CALCS/CsPbI3_test/results_32kpts/")

                bandStruct[kidx,:] = eigValsAtK
            self._copy_currIter_to_prevIter_shm()
        else:       # multiprocessing
            torch.set_num_threads(1)
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            # print(f"The size of cachedMats_info is: {sys.getsizeof(cachedMats_info)/1024} KB")
            
            # args_list = [(kidx, cachedMats_info, False) for kidx in range(nkpt)]
            args_list = [(kidx, cachedMats_info, False, 0, True) for kidx in range(nkpt)]     # requires_grad=False, verbosity=0, parallelization=True
            with mp.Pool(self.NNConfig['num_cores']) as pool:
                eigValsList = pool.starmap(self.calcEigValsAtK, args_list)
            bandStruct = torch.stack(eigValsList)
            self._copy_currIter_to_prevIter_shm()

        return bandStruct


    def initCouplingMats(self, SOwidth=0.7, NLwidth=1.0, NLshift=1.5):
        """
        This function is for caching the SOC and NL derivative potentials.
        It doesn't do the local potential at all, just builds
        the SO and NL matrices in the basis <G_i | dV | G_j + q>, where q is
        the phonon wavevector. For further explanation, see buildCouplingMat().
        In general, we can't reuse the computations from initSOmat() or
        initNLmat() because the j basis can be shifted by an arbitrary amount q. 
        These caluclation will be performed at the kidx of the bandgap kpoint 
        (see buildCouplingMat()).
        """

        kidx = self.idx_gap
        nbv = self.basis.shape[0]
        # set radial cutoff ~ 4.2488 Bohr; V(rcut) = 1e-16 for default SOwidth
        rcut_so = np.sqrt(SOwidth**2 * 16 * np.log(10.0))
        rcut_nl = np.sqrt(NLwidth**2 * 16 * np.log(10.0))

        nqp = self.system.getNQpts()

        # BEWARE! these might use a lot of memory!
        # for example, a 4000 x 4000 numpy array with dtype complex128
        # uses approx 244 MB of RAM. We are initializing
        # 3* (Natom * 3 * nqp) of these matrices. If there are a lot
        # of atoms or a lot of qpoints, this will use a considerable
        # amount of RAM. If we really need to, we can only store the 
        # upper triangles since the matrices are hermitian. The NL mats
        # also have 0 off diagonal BLOCKS. Not doing any of this yet.
        SOmats = np.empty([nqp, self.system.getNAtoms(), 3], dtype=object)
        for id1 in range(nqp):
            for id2 in range(self.system.getNAtoms()):
                for id3 in range(3):
                    SOmats[id1,id2,id3] = np.zeros([2*nbv, 2*nbv], dtype=np.complex128)

        NLmats = np.empty([nqp, self.system.getNAtoms(), 3, 2], dtype=object)
        for id1 in range(nqp):
            for id2 in range(self.system.getNAtoms()):
                for id3 in range(3):
                    for id4 in range(2):
                        NLmats[id1,id2,id3, id4] = np.zeros([2*nbv, 2*nbv], dtype=np.complex128)

        for qidx in range(nqp):
            print(f"initializing coupling SO + NL: qpt {qidx+1}/{nqp}")
            sys.stdout.flush()

            gjPlusQ = self.basis + self.system.qpts[qidx]
            gjqPlusK = gjPlusQ + self.system.kpts[kidx]
            giPlusK = self.basis + self.system.kpts[kidx]
            gqDiff = torch.stack([self.basis] * nbv, dim=1 ) - gjPlusQ.repeat(nbv,1,1)  # G_i - (G_j + q)

            giPlusK = giPlusK.numpy(force=True)
            gjqPlusK = gjqPlusK.numpy(force=True)
            inm = np.linalg.norm(giPlusK, axis=1)
            jnm = np.linalg.norm(gjqPlusK, axis=1)

            isum = self._soIntegral_vect(inm, jnm, rcut_so, SOwidth)
            isum2 = self._soIntegral_vect(inm, jnm, rcut_nl, NLwidth)
            isum3 = self._nlIntegral_vect(inm, jnm, rcut_nl, NLwidth, NLshift)

            # this is the normal SOC prefactor (no derivs)
            SOprefactor = np.zeros([nbv,nbv], dtype=float)
            denom = inm[:, np.newaxis] * jnm
            ids = np.nonzero(denom)
            SOprefactor[ids] = 12 * np.pi / denom[ids]  # this DOES NOT include the factor of -i in front of the entire V_SO

            gcross = np.cross(np.stack([giPlusK]*nbv, axis=1),
                                np.stack([gjqPlusK]*nbv, axis=0), axisa=-1, axisb=-1, axisc=-1)

            gdot = np.tensordot(giPlusK, gjqPlusK, axes=[[1],[1]])

            for alpha in range(self.system.getNAtoms()):
                gqDiffDotTau = gqDiff * self.system.atomPos[alpha]
                gqDiffDotTau = np.sum(gqDiffDotTau.numpy(force=True), axis=2)
                structFact = (1.0 / self.system.getCellVolume()) * (np.cos(gqDiffDotTau) + 1j * np.sin(gqDiffDotTau))

                for gamma in range(3):
                    # Now add derivative of SOC potential and nonlocal potential.
                    # First consider the SOC potential: it is composed of 4 "parts":
                    # the first includes the prefactor and the cross product, we will call this c(k+G_i, k+G_j)
                    # the second is the integral over r from 0 to infinity, we will call this f(|r-tau_{alpha}|, |k+G_i|, |k+G_j|)
                    # the third is the structure factor, which we will call g(|G_i - G_j|, tau_{alpha})
                    # the fourth is the spin operator S_{sigma, sigma'}.
                    # We can thus write V_SOC = c(k+G_i, k+G_j) * \sum_{alpha} [f|r-tau_{alpha}|, |k+G_i|, |k+G_j|) * g(|G_i - G_j|, tau_{alpha})]  DOT S_{sigma,sigma'}
                    # Now we want <k+G_i|  dV / d tau_{alpha, gamma, q}  |k+G_j+q>
                    # = c(k+G_i, k+G_j+q) * f(|r-tau_{alpha}|, |k+G_i|, |k+G_j+q|) * dg(|G_i - (G_j+q)|, tau_{alpha}) / d tau_{alpha,gamma,q}   DOT S
                    # + c(k+G_i, k+G_j+q) * df(|r-tau_{alpha}|, |k+G_i|, |k+G_j+q|) / d tau_{gamma,alpha,q} * g(|G_i - (G_j+q)|, tau_{alpha})   DOT S
                    # --> The second term goes to 0 for any integral over r that converges. Consider df/dtau = df/d(r-tau) * d(r-tau)/dtau.
                    # We have df/d(r-tau) = d/d(r-tau) integral 0 to infty d(r-tau) of some function. This is like considering
                    # d/dx \integral_0^infty dx f(x). As long as the integral converges, the resulting expression is a constant (or, in the case
                    # of a multi-variable function, it contains no dependence on x), and thus the derivative is 0.
                    # This means that the deriv of the SOC potential is very similar to the deriv of the local potential:
                    # <k+G_i| dV_{i,j} / dtau_{alpha,gamma,q} |k+G_j+q> =  c(k+G_i, k+G_j+q) * f(|r-tau_{alpha}|, |k+G_i|, |k+G_j+q|) * 
                    #                                                               +i(G_i - (G_j+q))_{gamma} * g(|G_i - (G_j+q)|, tau_{alpha})   DOT S

                    derivFact = 1j * gqDiff[:,:, gamma]
                    derivFact = derivFact.numpy()  # send this from torch type to ndarray

                    # build SOC matrix
                    # up up
                    # gcp dot S_up,up is: 1/2 * (gcp.z)
                    common = -1j * SOprefactor * derivFact * isum * structFact
                    SOmats[qidx, alpha, gamma][:nbv, :nbv] = common * 0.5 * gcross[:,:,2]

                    # dn dn
                    # gcp dot S_dn,dn is: -1/2 * (gcp.z)
                    SOmats[qidx, alpha, gamma][nbv:, nbv:] = common * -0.5 * gcross[:,:,2]

                    # up dn
                    # gcp dot S_up,dn is: 1/2 * (gcp.x) - i/2 * (gcp.y)
                    SOmats[qidx,alpha,gamma][:nbv, nbv:] = common * 0.5 * (gcross[:,:,0] - 1j*gcross[:,:,1])

                    # dn up
                    # gcp dot S_dn,up is: 1/2 * (gcp.x) + i/2 * (gcp.y)
                    SOmats[qidx, alpha, gamma][nbv:, :nbv] = common * 0.5 * (gcross[:,:,0] + 1j*gcross[:,:,1])


                    # build NL matrix. It has the same deriv factor as SOC part.
                    # this potential is block diagonal on spin.
                    # It doesn't have the global factor of -i in front, like SOC does.
                    # up up, 1st integral
                    common = SOprefactor * derivFact * structFact * gdot
                    NLmats[qidx, alpha, gamma, 0][:nbv, :nbv] = isum2 * common
                    # 2nd integral
                    NLmats[qidx, alpha, gamma, 1][:nbv, :nbv] = isum3 * common

                    # dn dn
                    NLmats[qidx, alpha, gamma, 0][nbv:, nbv:] = isum2 * common
                    NLmats[qidx, alpha, gamma, 1][nbv:, nbv:] = isum3 * common


        return SOmats, NLmats


    def buildCouplingMats(self, qidx, atomgammaidxs=None):
        """
        The derivative of the potential (local or not) is a matrix of the same 
        size as the Hamiltonian (2*nbv x 2*nbv, in SOC case).
        This is for a given k-point, phonon wavevector (q-point), atom, and polarization direction (x,y,z).
        The k-point (electronic) will be assumed to be fixed at the bandgap kpoint.
        The q-point is the phonon wavevector, there is a different derivative (different matrix) for each q 
        like there is a different electronic Hamiltonian for each k. The qidx also need be
        specified as an arg.
        The derivative of the potential is with respect to the position of a given nucleus (atom) in the unit
        cell, along a specific direction (x,y,z)
        Like the calcHamiltonianMatrix function, this will only calculate the matrices for a single, 
        given q-vector. As a default behavior, this function will return all natom*3 derivatives for that 
        q-vector in a dict with keys that are tuples (atomidx, gamma). If you only want the derivs for a
        subset of atoms/gammas, you can specify which you want to compute using the "atomgammaidxs"
        kwarg, which should be a list of tuples like [(atomidx1, gamma1), (atomidx2, gamma2), ...]. 
        """

        nbv = self.basis.shape[0]
        natom = self.system.getNAtoms()

        ret_dict = {}

        if atomgammaidxs is None:
            atomgammaidxs = [(a, g) for a in range(natom) for g in range(3)]

        # local potential: dV_{i,j} / d tau_{alpha, gamma, q} = <G_i |dV_{alpha} / d tau_{alpha,gamma,q}|G_j + q> = 
        # +i*(G_{i,gamma} - (G_{j,gamma} + q_{gamma})) * [e^{+i(G_i-(G_j+q))\cdot\tau_{alpha}} * v_{alpha}(|G_i - (G_j + q)|) / (V_cell)]
        # i,j labels the plane wave basis. alpha labels the atom identity. gamma labels the (x,y,z) component of a vector,
        # and q is the phonon wave vector.
        # !! WHAT ABOUT STRAIN TERM?? -- not implementing it here for now, its deriv is a bit complicated for a generic
        # unit cell geometry. It also depends on our definition of cell volume: does it depend
        # on atomic positions, or only lattice vectors? This is a choice...?

        gjPlusQ = self.basis + self.system.qpts[qidx]
        gqDiff = torch.stack([self.basis] * nbv, dim=1 ) - gjPlusQ.repeat(nbv,1,1)  # G_i - (G_j + q)

        for alpha, gamma in atomgammaidxs:
            if self.SObool:
                dV = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)
            else:
                dV = torch.zeros([nbv, nbv], dtype=torch.complex128)

            # this prefactor comes from the derivative of the structure factor
            if gamma == 0:
                # x
                prefactor = 1j * gqDiff[:,:,0]
            elif gamma == 1:
                # y
                prefactor = 1j * gqDiff[:,:,1]
            else:
                # z
                prefactor = 1j * gqDiff[:,:,2]
            # test
            #prefactor = torch.ones_like(prefactor)

            gqDiffDotTau = torch.sum(gqDiff * self.system.atomPos[alpha], axis=2)
            structFact = (1.0 / self.system.getCellVolume()) * (torch.cos(gqDiffDotTau) + 1j * torch.sin(gqDiffDotTau))

            thisAtomIndex = np.where(self.system.atomTypes[alpha]==self.atomPPorder)[0]
            if len(thisAtomIndex)!=1: 
                raise ValueError("Type of atoms in PP. ")
            thisAtomIndex = thisAtomIndex[0]

            if self.NN_locbool:
                atomFF = self.model(torch.norm(gqDiff, dim=2).view(-1,1))
                atomFF = atomFF[:, thisAtomIndex].view(nbv, nbv)
            else:
                #atomFF = pot_func(torch.norm(gqDiff, dim=2), self.PPparams[self.system.atomTypes[alpha]])
                atomFF = pot_funcLR(torch.norm(gqDiff, dim=2), self.PPparams[self.system.atomTypes[alpha]], self.LRgamma)

            dV[:nbv, :nbv] = prefactor * structFact * atomFF

            if self.SObool:
                # local potential has delta function on spin --> block diagonal
                dV[nbv:, nbv:] = prefactor * structFact * atomFF

                # SOC part
                if isinstance(self.SOmats_couple[qidx, alpha, gamma], torch.Tensor):
                    tmp = self.SOmats_couple[qidx, alpha, gamma]
                else:
                    tmp = torch.tensor(self.SOmats_couple[qidx, alpha, gamma])

                dV = dV + tmp * self.PPparams[self.system.atomTypes[alpha]][5]
            

                # NL part
                if isinstance(self.NLmats_couple[qidx,alpha,gamma,0], torch.Tensor):
                    tmp1 = self.NLmats_couple[qidx,alpha,gamma,0]
                else:
                    tmp1 = torch.tensor(self.NLmats_couple[qidx,alpha,gamma,0])
                if isinstance(self.NLmats_couple[qidx,alpha,gamma,1], torch.Tensor):
                    tmp2 = self.NLmats_couple[qidx,alpha,gamma,1]
                else:
                    tmp2 = torch.tensor(self.NLmats_couple[qidx,alpha,gamma,1])

                dV = (dV + tmp1 * self.PPparams[self.system.atomTypes[alpha]][6]
                                + tmp2 * self.PPparams[self.system.atomTypes[alpha]][7] )
                
            ret_dict[(alpha,gamma)] = dV
                
        return ret_dict


    def calcCouplings(self, qlist=None, atomgammaidxs=None, symm_equiv=None):
        """
        All we do here is call buildCouplingMats(), check the we have the
        correct eigenstates (from bandstructure calculation) to compute the
        desired matrix elements, then compute the expectation values.

        This return a dictionary with keys that are tuples: 
        (atomidx, gamma, qidx, 'vb'/'cb')
        and values are just floats (the coupling value). The couplings are in
        units of eV.

        qlist is a list of qidx integers corresponding to the phonon q-points
        for which we want to evaluate the coupling. The default behavior
        is to compute the coupling for all qpoint supplied in the
        input files. A few notes about this:
        - the qpoint and kpoint grids supplied have to be commensurate, so
        that for every q vec we have a k' vector so that k_{bg} + q = k', where
        k_{bg} is the kpoint vector of the bandgap.
        - the couplings are always evaluated at the bandgap kpoint, according
        to the above expression.
        - the coupling are computed for valence-valence band scattering (coupling) 
        and conduction-conduction band scattering (coupling). I.e. there is 
        no valence-conduction band scattering or other bands.

        As a default behavior, this function will return all natom*3 couplings 
        for each q-vector in a dict with keys that are tuples (atomidx, gamma). 
        If you only want the derivs for a subset of atoms/directions (gammas), 
        you can specify which you want to compute using the 
        "atomgammaidxs" kwarg, which should be a list of tuples like 
        [(atomidx1, gamma1), (atomidx2, gamma2), ...]. 

        The coupling can be a complex number, but its magnitude is a 
        gauge-invariant quantity, which is invariant to sign conventions
        in the code. This function therefore returns the 
        magnitude of the number, averged over degenerate band spaces and,
        optionally, over symmetry equivalent derivative directions (x,y,z).
        IMPORTANT NOTE: when there are exactly degenerate bands in the VB or
        CB space (i.e. energy difference less than 1e-15), the x, y, and z
        derivs can be subject to an arbitrary unitary rotation. If you know that
        some of these derivs should be the same due to the spherical symmetry
        of the atomic potentials and the unit cell geometry, you can recover the
        correct values by averaging over the symmetry equivalent directions. The
        User needs to specify this for each atom in a dict "symm_equiv" which
        has keys corresponding to the atom idxs, and values are tuples
        corresponding to the directions to be averaged e.g. ('x','y','z').
        You can see an example in test_ham/test_couple.py.
        I don't think there will be any cases when you need to average over
        multiple different atoms, since they all should have different symmetry
        operations..?
        """

        if qlist is None:
            qlist = list(range(self.system.getNQpts()))
        
        k_bg = self.system.kpts[self.idx_gap]
        ret_dict = {}
        equiv_arr = torch.ones([3,3]) # use this to check for matching kpoint (up to periodic boundary conditions)
        equiv_arr[0,:] *= 0.0
        equiv_arr[1,:] *= 2*np.pi / self.system.scale
        equiv_arr[2,:] *= -2*np.pi / self.system.scale

        for qid in qlist:
            needKidx = None
            #qvec = self.system.qpts[qid]
            kp = k_bg + self.system.qpts[qid]
            for kid in range(self.system.getNKpts()):
                if torch.any(torch.all(torch.isclose(kp - self.system.kpts[kid], equiv_arr), dim=1)):
                    # this complicated looking statement is true when the vector "kp"
                    # differs from a kpt vector by equiv_arr[0,:], equiv_arr[1,:], or equiv_arr[2,:]
                    needKidx = kid
                    break
            if needKidx is None:
                raise ValueError("kpt and qpt grids are not commensurate: k_{bg} + q != k'")

            dV_dict = self.buildCouplingMats(qid, atomgammaidxs=atomgammaidxs)

            # check if we need to avg over symmetry equivalent deriv directions
            symm_equiv_compat = {}
            avg_couple = {}
            if symm_equiv is not None:
                print("\nWARNING: This feature is no longer necessary for atomic derivs.")
                print("Degeneracy of electronic bands is now handled automatically.")
                print("This feature should only be necessary for explicit phonons.\n")
                for key in symm_equiv:
                    avg_couple[(key, 'cb')] = torch.zeros([1,], dtype=torch.complex128)
                    avg_couple[(key, 'vb')] = torch.zeros([1,], dtype=torch.complex128)
                    tmp = symm_equiv[key]
                    symm_equiv_compat[key] = []
                    for i in range(len(tmp)):
                        if tmp[i] == 'x' or tmp[i] == 'X':
                            symm_equiv_compat[key].append(0)
                        elif tmp[i] == 'y' or tmp[i] == 'Y':
                            symm_equiv_compat[key].append(1)
                        else:
                            assert tmp[i] == 'z' or tmp[i] == 'Z'
                            symm_equiv_compat[key].append(2) 

                for key in dV_dict:
                    if key[0] in symm_equiv:
                        if key[1] in symm_equiv_compat[key[0]]:
                            n_right = len(self.cb_vecs[needKidx])
                            n_left = len(self.cb_vecs[self.idx_gap])
                            if n_right > 1:
                                right_vecs = torch.stack(self.cb_vecs[needKidx], dim=-1)
                            else:
                                right_vecs = self.cb_vecs[needKidx][0].view(-1,1)
                            if n_left > 1:
                                left_vecs = torch.stack(self.cb_vecs[self.idx_gap], dim=0)
                            else:
                                left_vecs = self.cb_vecs[self.idx_gap][0].view(1,-1)
                            tmp = torch.matmul(dV_dict[key], right_vecs)   # batched multiplication of all degenerate bands
                            tmp = torch.matmul(torch.conj(left_vecs), tmp) # n_right * n_left dot products in the elements of a matrix
                            mag = torch.sum(torch.sqrt(tmp.conj() * tmp)).real
                            avg_couple[(key[0], 'cb')] += mag / (len(symm_equiv[key[0]]) * n_right * n_left)

                            n_right = len(self.vb_vecs[needKidx])
                            n_left = len(self.vb_vecs[self.idx_gap])
                            if n_right > 1:
                                right_vecs = torch.stack(self.vb_vecs[needKidx], dim=-1)
                            else:
                                right_vecs = self.vb_vecs[needKidx][0].view(-1,1)
                            if n_left > 1:
                                left_vecs = torch.stack(self.vb_vecs[self.idx_gap], dim=0)
                            else:
                                left_vecs = self.vb_vecs[self.idx_gap][0].view(1,-1)
                            tmp2 = torch.matmul(dV_dict[key], right_vecs) # batched multiplication of all degenerate bands
                            tmp2 = torch.matmul(torch.conj(left_vecs), tmp2) # n_right * n_left dot products in the elements of a matrix
                            mag2 = torch.sum(torch.sqrt(tmp2.conj() * tmp2)).real
                            avg_couple[(key[0], 'vb')] += mag2 / (len(symm_equiv[key[0]]) * n_right * n_left)

            # build ret_dict 
            for key in dV_dict:
                if key[0] in symm_equiv_compat:
                    if key[1] in symm_equiv_compat[key[0]]:
                        #avg_cb = avg_couple[(key[0], 'cb')]
                        #avg_vb = avg_couple[(key[0], 'vb')]
                        #ret_dict[key+(qid,'cb')] = torch.sqrt(avg_cb.conj() * avg_cb).real * AUTOEV
                        #ret_dict[key+(qid,'vb')] = torch.sqrt(avg_vb.conj() * avg_vb).real * AUTOEV
                        ret_dict[key + (qid,'cb')] = avg_couple[(key[0], 'cb')]
                        ret_dict[key + (qid,'vb')] = avg_couple[(key[0], 'vb')]

                else:
                    n_right = len(self.cb_vecs[needKidx])
                    n_left = len(self.cb_vecs[self.idx_gap])
                    if n_right > 1:
                        right_vecs = torch.stack(self.cb_vecs[needKidx], dim=-1)
                    else:
                        right_vecs = self.cb_vecs[needKidx][0].view(-1,1)
                    if n_left > 1:
                        left_vecs = torch.stack(self.cb_vecs[self.idx_gap], dim=0)
                    else:
                        left_vecs = self.cb_vecs[self.idx_gap][0].view(1,-1)
                    cpl = torch.matmul(dV_dict[key], right_vecs) # batched multiplication of all degenerate bands
                    cpl = torch.matmul(torch.conj(left_vecs), cpl) # n_right * n_left dot products in the elements of a matrix
                    cpl_mag = torch.sum(torch.sqrt(cpl.conj() * cpl)).real
                    ret_dict[key + (qid,'cb')] = (cpl_mag / (n_right * n_left)) * AUTOEV # average coupling from degenerate subspace

                    n_right = len(self.vb_vecs[needKidx])
                    n_left = len(self.vb_vecs[self.idx_gap])
                    if n_right > 1:
                        right_vecs = torch.stack(self.vb_vecs[needKidx], dim=-1)
                    else:
                        right_vecs = self.vb_vecs[needKidx][0].view(-1,1)
                    if n_left > 1:
                        left_vecs = torch.stack(self.vb_vecs[self.idx_gap], dim=0)
                    else:
                        left_vecs = self.vb_vecs[self.idx_gap][0].view(1,-1)
                    cpl = torch.matmul(dV_dict[key], right_vecs) # batched multiplication of all degenerate bands
                    cpl = torch.matmul(torch.conj(left_vecs), cpl) # n_right * n_left dot products in the elements of a matrix
                    cpl_mag = torch.sum(torch.sqrt(cpl.conj() * cpl)).real
                    ret_dict[key + (qid,'vb')] = (cpl_mag / (n_right * n_left)) * AUTOEV

        return ret_dict

    
    def _bessel1(self, x, x1):
        # sin(x)/(x^2) - cos(x)/x = sin(x) * x1^2 - cos(x) * x1
        return np.sin(x) * x1**2 - np.cos(x) * x1

    
    def _bessel1_exact(self, x):
        ids = np.nonzero(x)
        ret = np.zeros_like(x)
        ret[ids] = np.sin(x[ids]) / (x[ids]**2) - np.cos(x[ids]) / x[ids]
        return ret

    
    def _soIntegral(self, k, kp, rcut, width):
        """
        integral from 0 to rcut of
        dr*r^2*j1(Kr)*exp^(-(r/width)^2)*j1(K'r) where j1 is the 1st bessel function,
        K = |kpoint + basisVector|.
        This assumes k, kp, rcut, and width are all scalars
        """

        # s1 = 2 * (torch.exp( 2j * rcut * k) - 1) * torch.exp(-rcut*(1j * (k-kp) + rcut/(width**2)))
        # s1 += -2 * (torch.exp( 2j * rcut * k) - 1) * torch.exp(-rcut*(1j * (k+kp) + rcut/(width**2)))
        # s1 *= 1.0/rcut

        # s2 = torch.exp(-0.25*(k-kp)**2 * width**2) * np.sqrt(np.pi) * (k*kp*width**2 - 2)
        # #s2 *= torch.erf(rcut/width - 0.5j * (k-kp) * width)
        # #s3 = torch.exp(-0.25*(k-kp)**2 * width**2) * np.sqrt(np.pi) * (k*kp*width**2 - 2)
        # #s3 *= torch.erf(rcut/width + 0.5j * (k-kp) * width)
        # s3 = s2 * torch.erf(rcut/width + 0.5j * (k-kp) * width)
        # s2 *= torch.erf(rcut/width - 0.5j * (k-kp) * width)

        # s4 = torch.exp(-0.25*(k+kp)**2 * width**2) * np.sqrt(np.pi) * (k*kp*width**2 + 2)
        # s5 = s4 * torch.erf(rcut/width + 0.5j * (k+kp) * width)
        # s4 *= torch.erf(rcut/width - 0.5j * (k+kp) * width)

        s1 = 2 * (np.exp( 2j * rcut * k) - 1) * np.exp(-rcut*(1j * (k-kp) + rcut/(width**2)))
        s1 += -2 * (np.exp( 2j * rcut * k) - 1) * np.exp(-rcut*(1j * (k+kp) + rcut/(width**2)))
        s1 *= 1.0/rcut

        s2 = np.exp(-0.25*(k-kp)**2 * width**2) * np.sqrt(np.pi) * (k*kp*width**2 - 2)
        s3 = s2 * erf(rcut/width + 0.5j * (k-kp) * width)
        s2 = s2 * erf(rcut/width - 0.5j * (k-kp) * width)

        s4 = np.exp(-0.25*(k+kp)**2 * width**2) * np.sqrt(np.pi) * (k*kp*width**2 + 2)
        s5 = s4 * erf(rcut/width + 0.5j * (k+kp) * width)
        s4 = s4 * erf(rcut/width - 0.5j * (k+kp) * width)

        ret = 1/(8*k**2 * kp**2) * (s1 + (1/width)*(s2+s3+s4+s5))
        assert abs(np.imag(ret)) < 1e-10
        return np.real(ret)
        
    
    def _soIntegral_vect(self, k, kp, rcut, width):
        """
        Computes the same quantity as soIntegral(), but vectorized.
        Assumes k and kp are NUMPY vectors of length nbv, returns a matrix of
        integrals mat[idx_k, idx_kp] for every k,kp combination.
        """
        #k = np.array(k)
        #kp = np.array(kp)
        k_p_kp = k[:, np.newaxis] + kp
        k_m_kp = k[:, np.newaxis] - kp
        k_x_kp = k[:, np.newaxis] * kp
        s1kvec = 2 * (np.exp( 2j * rcut * k) - 1) 
        s1 = s1kvec[:,np.newaxis] * np.exp(-rcut*(1j * (k_m_kp) + rcut/(width**2)))
        s1 += -1.0 * s1kvec[:,np.newaxis] * np.exp(-rcut*(1j * (k_p_kp) + rcut/(width**2)))
        s1 *= 1.0/rcut

        s2 = np.exp(-0.25*(k_m_kp)**2 * width**2) * np.sqrt(np.pi) * (k_x_kp * width**2 - 2)
        s3 = s2 * erf(rcut/width + 0.5j * (k_m_kp) * width)
        s2 = s2 * erf(rcut/width - 0.5j * (k_m_kp) * width)

        s4 = np.exp(-0.25*(k_p_kp)**2 * width**2) * np.sqrt(np.pi) * (k_x_kp * width**2 + 2)
        s5 = s4 * erf(rcut/width + 0.5j * (k_p_kp) * width)
        s4 = s4 * erf(rcut/width - 0.5j * (k_p_kp) * width)

        denom = 8 * k[:, np.newaxis]**2 * kp**2
        ids = np.nonzero(denom)
        ret = np.zeros([len(k), len(kp)], dtype=np.complex128)
        ret[ids] = 1/denom[ids] * (s1 + (1/width)*(s2+s3+s4+s5))[ids]
        #ret = 1/(8 * k[:,np.newaxis]**2 * kp**2) * (s1 + (1/width)*(s2+s3+s4+s5))
        assert np.all(np.abs(np.imag(ret)) < 1e-10)
        return np.real(ret)


    def _soIntegral_dan(self, k, kp, width):
        """
        SO integral exactly as daniel's c code computes it,
        vectorized over k,kp (so assuming k,kp are vectors of
        length nbv). This is useful for testing. The 
        _soIntegral_vect() routine is faster and more robust.
        This method will get systematically worse as maxKE gets larger.
        """
        # set integral dr ~ 0.0089 Bohr at 25 Hartree energy cutoff
        dr = 2*np.pi / (100 * np.linalg.norm(self.basis[-1]))
        # set radial cutoff ~ 4.2488 Bohr; V(rcut) = 1e-16 for default SOwidth
        rcut = np.sqrt(width**2 * 16 * np.log(10.0))
        ncut = int(rcut/dr)
        sum = np.zeros([len(k), len(k)], dtype=float)
        for gp in range(1,ncut):
            r = dr * gp
            kv = self._bessel1(k*r, 1/(k*r + 1e-10))
            kpv = self._bessel1(kp*r, 1/(kp*r + 1e-10))
            scal = r**2 * np.exp(-(r/width)**2) * dr
            sum += (kv[:, np.newaxis] * kpv) * scal
        
        return sum

    
    def _nlIntegral_vect(self, k, kp, rcut, width, shift):
        """
        Calculates the nonlocal integral V_{l=1}(K,K') = 
        integral from 0 to rcut of
        dr*r^2*j1(Kr)* [exp^(-((r-shift)/width)^2)] *j1(K'r)
        where j1 is the 1st bessel function.

        This integral does not seem to have a closed form for
        arbitrary shift parameter, so it is evaluated using vectorized
        numerical integration, converged to a relative error of ~10^-5.
        """
        def integrand(r):
            scal = r**2 * np.exp(-((r-shift)/width)**2)
            kv = self._bessel1_exact(k*r)
            kpv = self._bessel1_exact(kp*r)
            return ((kv[:,np.newaxis] * kpv) * scal).reshape(-1)
        
        ret, err = quad_vec(integrand, 1e-10, shift+rcut, epsabs=1e-20, epsrel=1e-5, quadrature="gk21")
        ret = ret.reshape(len(k), len(kp))
        # print(f"int2 est. maxerr: {np.amax(err)}")
        return ret


    def _nlIntegral_dan(self, k, kp, width, shift):
        """
        NL integral exactly as daniel's c code computes it,
        vectorized over k,kp (so assuming k,kp are vectors of
        length nbv). This is useful for testing. The 
        _nlIntegral_vect() routine is much more robust. It's
        pretty clear that daniel's routine is not well converged for
        arbitrary k,kp,width,shift.
        """
        # set integral dr ~ 0.0089 Bohr at 25 Hartree energy cutoff
        dr = 2*np.pi / (100 * np.linalg.norm(self.basis[-1]))
        # set radial cutoff ~ 4.2488 Bohr; 
        rcut = np.sqrt(width**2 * 16 * np.log(10.0))
        ncut = int(rcut/dr)
        sum = np.zeros([len(k), len(k)], dtype=float)
        for gp in range(1,ncut):
            r = gp * dr
            scal = r**2 * np.exp(-((r-shift)/width)**2)
            kv = self._bessel1(k*r, 1/(k*r + 1e-10))
            kpv = self._bessel1(kp*r, 1/(kp*r + 1e-10))
            sum += (kv[:,np.newaxis] * kpv) * scal * dr

        return sum


    def get_NNmodel(self):
        """
        Use this for getting the current NN model.
        Useful if fitting multiple materials at once.
        """
        return self.model
    

    def set_NNmodel(self, newmodel):
        """
        Use this to set the current NN model.
        Useful if fitting multiple materials at once.
        """
        self.model = newmodel


    def get_PPparams(self):
        return copy.deepcopy(self.PPparams)
    

    def set_PPparams(self, newparams):
        """
        Set new values for the algebraic PP "a" params.
        This is useful when performing optimization of the algebraic
        parts of the PP.
        """
        self.PPparams = newparams


def overlap_2eigVec(a, b):
    a_conj = np.conj(a)
    overlap = np.dot(a_conj, b)
    return np.abs(overlap)


def initAndCacheHams(systemsList, NNConfig, PPparams, atomPPOrder, device):
    """
    Initialize the ham class for each BulkSystem. 
    dummy_ham is used to initialize and store the cached SOmats and NLmats in dict cachedMats. 
    As I initialize dummy_ham, immediately load them into share memory
    Use a dict "cachedMats_info" to store dtype and shape
    Then remove dummy_ham, and any intermediate variables
    """
    print("\nInitializing the ham class for each BulkSystem. Cache-ing the SOmats, NLmats, and putting them into shared memeory. ")
    hams = []
    cachedMats_info = None
    shm_dict_SO = None
    shm_dict_NL = None
    for iSys, sys in enumerate(systemsList):
        start_time = time.time()

        # Here I separate: 
        # 1. SObool = False --> Just initialize ham. No storage / moving is needed. 
        # 2. SObool = True, no parallel --> Initialize ham with cache. No storage / moving is needed.
        # 3. SObool = True, yes parallel --> Do the complicated storage / moving. 
        if not NNConfig['SObool']: 
            ham = Hamiltonian(sys, PPparams, atomPPOrder, device, NNConfig=NNConfig, iSystem=iSys, SObool=NNConfig['SObool'])
        elif (NNConfig['SObool']) and (NNConfig['num_cores']==0): 
            ham = Hamiltonian(sys, PPparams, atomPPOrder, device, NNConfig=NNConfig, iSystem=iSys, SObool=NNConfig['SObool'])
        else: 
            cachedMats_info = {}
            shm_dict_SO = {}
            shm_dict_NL = {}
            ham = Hamiltonian(sys, PPparams, atomPPOrder, device, NNConfig=NNConfig, iSystem=iSys, SObool=True, cacheSO=False)
            dummy_ham = Hamiltonian(sys, PPparams, atomPPOrder, device, NNConfig=NNConfig, iSystem=iSys, SObool=NNConfig['SObool'])

            if dummy_ham.SOmats is not None: 
                # reshape dummy_ham.SOmats has shape (nkpt)*(nAtoms)*(2*nbasis) x (2*nbasis)
                dummy_ham.SOmats
                for kidx in range(sys.getNKpts()):
                    SOkey = f"SO_{iSys}_{kidx}"
                    SOvalue = {'dtype': dummy_ham.SOmats[kidx].dtype,
                        'shape': dummy_ham.SOmats[kidx].shape,
                    }
                    cachedMats_info[SOkey] = SOvalue

                    # Move the SOmats to shared memory
                    shm_dict_SO[f"shm_SO_{iSys}_{kidx}"] = shared_memory.SharedMemory(create=True, size=dummy_ham.SOmats[kidx].nbytes, name=f"SOmats_{iSys}_{kidx}")
                    tmp_arr = np.ndarray(cachedMats_info[f"SO_{iSys}_{kidx}"]['shape'], dtype=cachedMats_info[f"SO_{iSys}_{kidx}"]['dtype'], buffer=shm_dict_SO[f"shm_SO_{iSys}_{kidx}"].buf)  # Create a NumPy array backed by shared memory
                    tmp_arr[:] = dummy_ham.SOmats[kidx][:]   # Copy the cached SOmat into shared memory

            if dummy_ham.NLmats is not None: 
                # reshape dummy_ham.NLmats has shape (nkpt)*(nAtoms)*(2)*(2*nbasis) x (2*nbasis)
                dummy_ham.NLmats
                for kidx in range(sys.getNKpts()):
                    NLkey = f"NL_{iSys}_{kidx}"
                    NLvalue = {'dtype': dummy_ham.NLmats[kidx].dtype,
                        'shape': dummy_ham.NLmats[kidx].shape,
                    }
                    cachedMats_info[NLkey] = NLvalue

                    # Move the NLmats to shared memory
                    shm_dict_NL[f"shm_NL_{iSys}_{kidx}"] = shared_memory.SharedMemory(create=True, size=dummy_ham.NLmats[kidx].nbytes, name=f"NLmats_{iSys}_{kidx}")
                    tmp_arr = np.ndarray(cachedMats_info[f"NL_{iSys}_{kidx}"]['shape'], dtype=cachedMats_info[f"NL_{iSys}_{kidx}"]['dtype'], buffer=shm_dict_NL[f"shm_NL_{iSys}_{kidx}"].buf) 
                    tmp_arr[:] = dummy_ham.NLmats[kidx][:] 

            del dummy_ham
            gc.collect()
            print("Finished putting the cached SO and NLmats into shared memory ...")
        hams.append(ham)
        end_time = time.time()
        print(f"Elapsed time: {(end_time - start_time):.2f} seconds\n")
    return hams, cachedMats_info, shm_dict_SO, shm_dict_NL
