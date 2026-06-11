import sys, os
import torch
import numpy as np
from scipy.special import erf
from scipy.integrate import quad_vec   # quad, quadrature, 
import time
import copy
import gc
from torch.utils.checkpoint import checkpoint
import multiprocessing as mp
from multiprocessing import Process, Queue, Pool, shared_memory
import gc

from .constants import *
from .pp_func import pot_func, pot_funcLR, long_range_correction, qSpacePot_ft
from .read import init_critical_NNconfig, setNN
from utils.local_structure_correction import calcLocalSymmDescriptor

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
        coupling = False,
        LSDmodels = None
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
        self.fit_eff_masses = system.fit_eff_masses

        self.LRgamma = 0.2   # erf attenuation parameter for long-range 
                             # component of potential. This is a good value

        # if spin orbit, do a bunch of caching to speed up the inner loop 
        # of the optimization. This uses more memory (storing natom * nkpt
        # matrices of size 4*nbasis^2) in exchange for avoiding loops
        # over the basis within the optimization inner loop.
        self.SOmats = None
        self.NLmats = None
        self.SOmats_def = {}
        self.NLmats_def = {}
        if SObool and cacheSO:
            print("Caching SO mats.", flush=True)
            sys.stdout.flush()
            self.SOmats = self.initSOmat_fast()
            self.SOmats_def = {}
            # check if nonlocal potentials are included, if so, cache them
            self.checknl = False
            for atom in self.atomPPorder:
                if abs(self.PPparams[atom][6]) > 1e-8:
                    self.checknl = True
                    break
                elif abs(self.PPparams[atom][7]) > 1e-8:
                    self.checknl = True
                    break
            if self.checknl:
                print("Caching NL mats.", flush=True)
                sys.stdout.flush()
                self.NLmats = self.initNLmat_fast()
                self.NLmats_def = {}
       
        elif (SObool) and (not cacheSO) and (NNConfig['num_cores']==0):
            print("WARNING: Calculation requires SObool, but we are not cache-ing the SOmats and NLmats. Without multiprocessing parallelization. This is not recommended. ")

        
        self.checknl = False
        for atom in self.atomPPorder:
            if abs(self.PPparams[atom][6]) > 1e-8:
                self.checknl = True
                break
            elif abs(self.PPparams[atom][7]) > 1e-8:
                self.checknl = True
                break
        
        if self.coupling or self.fit_eff_masses:
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
                if not SObool:
                    print("NOTE: SOC is off. idxVB and idxCB are zero-indexed band indices without 2x interleaving for spin. Please double check to ensure your inputs of idxVB and idxCB correspond to your intended bands. ")

            if SObool:
                self.SOmats_couple, self.NLmats_couple = self.initCouplingMats()

        if self.NNConfig['local_env_corr']:
            # Compute the Behler-Parrinello atomic descriptors (local symmetry descriptors)
            self.LSDmodels = LSDmodels
        else:
            self.LSDmodels = None

        # send things to gpu, if enabled ??
        # Or is it better to send some things at the last minute before diagonalization?
        if model is not None:
            model.to(device)
        

    def _deformed_cache_key(self, kidx, scale):
        return (int(kidx), float(scale))


    def _get_deformed_cached_mats(self, kidx, scale):
        cache_key = self._deformed_cache_key(kidx, scale)

        if cache_key not in self.SOmats_def:
            self.SOmats_def[cache_key] = self.initSOmat_fast(defbool=True, idxGap=kidx)

        so_mats = self.SOmats_def[cache_key]
        nl_mats = None
        if self.checknl:
            if cache_key not in self.NLmats_def:
                self.NLmats_def[cache_key] = self.initNLmat_fast(defbool=True, idxGap=kidx)
            nl_mats = self.NLmats_def[cache_key]

        return so_mats, nl_mats


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
        print(f"Building VlocMat, elapsed time: {(end_time - start_time):.2f} seconds", flush=True) if self.NNConfig['runtime_flag'] else None
        
        if self.SObool:
            start_time = time.time() if self.NNConfig['runtime_flag'] else None
            Htot = self.buildSOmat(kidx, preComp_SOmats_kidx, addMat=Htot)
            end_time = time.time() if self.NNConfig['runtime_flag'] else None
            print(f"Building SOmat, elapsed time: {(end_time - start_time):.2f} seconds", flush=True) if self.NNConfig['runtime_flag'] else None

            if self.checknl: 
                start_time = time.time() if self.NNConfig['runtime_flag'] else None
                Htot = self.buildNLmat(kidx, preComp_NLmats_kidx, addMat=Htot)
                end_time = time.time() if self.NNConfig['runtime_flag'] else None
                print(f"Building NLmat, elapsed time: {(end_time - start_time):.2f} seconds", flush=True) if self.NNConfig['runtime_flag'] else None

        if self.device.type == "cuda":
            # !!! is this sufficient to match previous performance?
            # This limits data movement to gpu (good), but obviously
            # performs construction of H on cpu (at least the first time?), 
            # which might be slower.
            Htot.to(self.device)
        
        if not requires_grad: 
            Htot = Htot.detach()

        sys.stdout.flush()
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
            self.SOmats, self.NLmats = self._get_deformed_cached_mats(kidx, scale)

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


    def buildHtot_def_NEW(self, kidx, scale=1.01, verbosity=2, requires_grad=True):
        """
        Just like the function above, but with the added flexibility of 
        calculating at various k-points. 
        """
        if verbosity >= 3:
            print("***************************")
            print("You are computing deformation potentials by directly changing")
            print("the volume of the material. To be precise, computing a")
            print("quantity that can be correctly compared to the DFT literature,")
            print("or experiments, requires very careful consideration of the")
            print("g_i - g_j = 0 point in the potentials. These considerations")
            print("are not made here. Consult the DFT literature, e.g.")
            print("PRB 73 245206 (2006) and its references.")
            print("***************************")

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
            self.SOmats, self.NLmats = self._get_deformed_cached_mats(kidx, scale)

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

        if not requires_grad: 
            Htot = Htot.detach()
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
        q = torch.norm(gdiff, dim=2).view(-1,1)


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
            atomType = self.system.atomTypes[alpha]
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
                lr_coeff = self.PPparams[atomType][4]
                atomFF = atomFF + long_range_correction(torch.norm(gdiff, dim=2), self.LRgamma, lr_coeff)
            else:
                # atomFF = pot_func(torch.norm(gdiff, dim=2), self.PPparams[atom])
                atomFF = pot_funcLR(torch.norm(gdiff, dim=2), self.PPparams[atomType], self.LRgamma)

            if self.NNConfig["local_env_corr"]:
                descriptors = self.system.env_descriptors[atomType]
                indx_alpha = torch.where(self.system.atom_indices[atomType] == alpha)[0].squeeze(0)
                N_alpha = descriptors[indx_alpha, :]
                
                N_alphas = N_alpha.repeat(q.shape[0], 1)
                
                x_input = torch.cat([N_alphas, q], dim=1)
                # x_ref_input = torch.cat([zeros, q], dim=1)

                # rSpaceLSD = self.LSDmodels[atomType](x_input)
                # print(f"shape of vr = {vr.shape}, rSpaceLSD = {rSpaceLSD.shape}")
                # qSpaceLSD = spherical_ft(vr, rSpaceLSD, q)
                # atomFF += qSpaceLSD.view(nbv, nbv)
                # print(f"Added q LSD")
                atomFF += self.LSDmodels[atomType](x_input).view(nbv, nbv)
                
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
            print(f"\tinitializing SO: kpt {kidx+1}/{nkp}")
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
    
    def _wrap_initSOmat(self, args):
        nbv = self.basis.shape[0]
        kidx, SOwidth, defbool, idxGap = args
        # Allocate a local matrix for this k-point
        mat = np.zeros((self.system.getNAtoms(), 2*nbv, 2*nbv), dtype=np.complex128)
        self.initSOmat_fast_oneKpt(kidx, mat, SOwidth, defbool, idxGap)
        gc.collect()
        return (kidx, mat)

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
        
        if (self.NNConfig["num_cores"] == 0) or (self.NNConfig["pool_initSO"] == 0):
            # serial path
            SOmats_4d = np.zeros((nkp, self.system.getNAtoms(), 2*nbv, 2*nbv), dtype=np.complex128)
            for kidx in range(nkp):
                self.initSOmat_fast_oneKpt(kidx, SOmats_4d[kidx], SOwidth, defbool, idxGap)
                gc.collect()
            
        else:
            print(f"Initializing with {self.NNConfig['num_cores']} pools\n")
            args_list = [(kidx, SOwidth, defbool, idxGap) for kidx in range(nkp)]
            with mp.Pool(self.NNConfig['num_cores']) as pool:
                results = pool.map(self._wrap_initSOmat, args_list)

            # collect into big array
            SOmats_4d = np.zeros((nkp, self.system.getNAtoms(), 2*nbv, 2*nbv), dtype=np.complex128)
            for kidx, mat in results:
                SOmats_4d[kidx] = mat

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

        print(f"\tinitializing SO: kpt {kidx+1}/{nkp}", flush=True)
        sys.stdout.flush()

        if defbool:
            gikp = self.basis + torch.stack([self.system.kpts[idxGap]] * nbv, dim=0)
            gjkp = self.basis + torch.stack([self.system.kpts[idxGap]] * nbv, dim=0)
        else:
            gikp = self.basis + torch.stack([self.system.kpts[kidx]] * nbv, dim=0)
            gjkp = self.basis + torch.stack([self.system.kpts[kidx]] * nbv, dim=0)
        
        gdiff = torch.stack([self.basis]*nbv, dim=1) - self.basis.repeat(nbv, 1, 1)
        #gdiff = self.basis.unsqueeze(0) - self.basis.unsqueeze(1)
        #basis = self.basis.to("cuda")
        #gdiff = basis.unsqueeze(0) - basis.unsqueeze(1)
        #gdiff = self.basis[:, None, :] - self.basis[None, :, :]
        
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
            print(f"\tinitializing NL pots: kpt {kidx+1}/{nkp}", flush=True)
            sys.stdout.flush()
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


    def _wrap_initNLmat(self, args):
        nbv = self.basis.shape[0]
        kidx, width1, width2, shift, defbool, idxGap = args
        # Allocate a local matrix for this k-point
        mat = np.zeros((self.system.getNAtoms(), 2, 2*nbv, 2*nbv), dtype=np.complex128)
        self.initNLmat_fast_oneKpt(kidx, mat, width1, width2, shift, defbool, idxGap)
        gc.collect()
        return (kidx, mat)

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
        if (self.NNConfig["num_cores"] == 0) or (self.NNConfig["pool_initNL"] == 0):
          NLmats_5d = np.zeros((nkp, self.system.getNAtoms(), 2, 2*nbv, 2*nbv), dtype=np.complex128)
          for kidx in range(nkp):
              self.initNLmat_fast_oneKpt(kidx, NLmats_5d[kidx], width1, width2, shift, defbool, idxGap)
              gc.collect()
        else:
            print(f"Initializing with {self.NNConfig['num_cores']} pools\n")
            args_list = [(kidx, width1, width2, shift, defbool, idxGap) for kidx in range(nkp)]
            with mp.Pool(self.NNConfig['num_cores']) as pool:
                results = pool.map(self._wrap_initNLmat, args_list)

            # collect into big array
            NLmats_5d = np.zeros((nkp, self.system.getNAtoms(), 2, 2*nbv, 2*nbv), dtype=np.complex128)
            for kidx, mat in results:
                NLmats_5d[kidx] = mat

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
        
        print(f"\tinitializing NL pots: kpt {kidx+1}/{nkp}", flush=True)
        sys.stdout.flush()

        if defbool:
            gikp = self.basis + torch.stack([self.system.kpts[idxGap]] * nbv, dim=0)
            gjkp = self.basis + torch.stack([self.system.kpts[idxGap]] * nbv, dim=0)
        else:
            gikp = self.basis + torch.stack([self.system.kpts[kidx]] * nbv, dim=0)
            gjkp = self.basis + torch.stack([self.system.kpts[kidx]] * nbv, dim=0)
        gdiff = torch.stack([self.basis]*nbv, dim=1) - self.basis.repeat(nbv, 1, 1)
        #gdiff = self.basis.unsqueeze(0) - self.basis.unsqueeze(1)

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
                print("WARNING. THIS WILL BE SLOW. Attempting to build the SOmat, but 1) no precomputed SOmats are stored in shared memory, 2) no cached SOmatrices in the ham class. \nCalculating the SOmats for each kpt on the fly. ")
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
                print("WARNING. THIS WILL BE SLOW. Attempting to build the NLmat, but 1) no precomputed NLmats are stored in shared memory, 2) no cached NL matrices in the ham class. \nCalculating the NLmats on the fly. ")
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


    def calcEigValsAtK(self, kidx, cachedMats_info=None, requires_grad=True, verbosity=0, def_H=False, def_scale=0.01):
        '''
        This function builds the Htot at a certain kpoint that is given as the input, 
        digonalizes the Htot, and obtains the eigenvalues at this kpoint. 
        '''

        nbands = self.system.nBands
        eigVals = torch.zeros(nbands)

        if (cachedMats_info is None) and (self.SObool==False):    # proceed as normal. Won't even go into buildSO or buildNL. Need to pass None into buildSO and buildNL
            preComp_SOmats_kidx = None
            preComp_NLmats_kidx = None
        elif (cachedMats_info is None) and (self.SObool==True):   # no cached matrices in the shared memory
            preComp_SOmats_kidx = None
            preComp_NLmats_kidx = None     # functions buildSOmat and buildNLmat will handle these cases
        elif (cachedMats_info is not None): 
            start_time = time.time() if self.NNConfig['runtime_flag'] else None
            shm_SOmats = shared_memory.SharedMemory(name=f"SOmats_{self.iSystem}_{kidx}")
            preComp_SOmats_kidx = np.ndarray(cachedMats_info[f"SO_{self.iSystem}_{kidx}"]['shape'], dtype=cachedMats_info[f"SO_{self.iSystem}_{kidx}"]['dtype'], buffer=shm_SOmats.buf)
            if self.checknl:
                shm_NLmats = shared_memory.SharedMemory(name=f"NLmats_{self.iSystem}_{kidx}")
                preComp_NLmats_kidx = np.ndarray(cachedMats_info[f"NL_{self.iSystem}_{kidx}"]['shape'], dtype=cachedMats_info[f"NL_{self.iSystem}_{kidx}"]['dtype'], buffer=shm_NLmats.buf)
            else: 
                preComp_NLmats_kidx = None
            end_time = time.time() if self.NNConfig['runtime_flag'] else None
            print(f"Loading shared memory, elapsed time: {(end_time - start_time):.2f} seconds") if self.NNConfig['runtime_flag'] else None
        else: 
            raise ValueError("Error in calcEigValsAtK. ")

        start_time = time.time() if self.NNConfig['runtime_flag'] else None
        if not def_H: 
            H = self.buildHtot(kidx, preComp_SOmats_kidx, preComp_NLmats_kidx, requires_grad)
        else: 
            H = self.buildHtot_def_NEW(kidx, scale=def_scale, requires_grad=requires_grad)

        if not requires_grad: 
            H = H.detach()
        end_time = time.time() if self.NNConfig['runtime_flag'] else None
        print(f"Building Htot, elapsed time: {(end_time - start_time):.2f} seconds") if self.NNConfig['runtime_flag'] else None

        start_time = time.time() if self.NNConfig['runtime_flag'] else None
        if not self.coupling:
            energies = torch.linalg.eigvalsh(H)
            energiesEV = energies * AUTOEV

            # reorder the energies according to the manual input in self.system.bandOrderMatrix
            energiesEV = energiesEV[self.system.bandOrderMatrix[kidx, :]]

        else:
            # this will be slower than necessary, since torch seems to only support
            # full diagonalization including all eigenvectors. 
            # If computing couplings, it would be faster to
            # implement a custom torch diagonalization wrapper
            # that uses scipy under the hood to allow for better partial
            # diagonalization algorithms (e.g. the ?heevr driver).

            """
            WARNING: This else clause hasn't been made compatible with band ordering!!! 
            """
            ens, vecs = torch.linalg.eigh(H)
            energiesEV = ens * AUTOEV
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
                if abs(ens[self.idx_vb] - ens[idx]) < 1e-5 / AUTOEV:
                    # this describes a degenerate state as begin within .01 meV (adopted from EPW source)
                    self.vb_vecs[kidx].append(vecs[:, idx])
                    ctr += 1
                else:
                    break

            if ctr == 1 and self.SObool and verbosity >= 2:
                print(f"\nWARNING: spin-orbit calc but vb spin states are not degenerate to 1e-5 eV, kidx={kidx}\n")
            if verbosity >= 3:
                print(f"kidx={kidx}, vb_vec[0:5]= {self.vb_vecs[kidx, :5]}")

            ctr = 1
            for idx in range(self.idx_cb+1, self.system.nBands):
                if abs(ens[self.idx_cb] - ens[idx]) < 1e-5 / AUTOEV:
                    # this describes a degenerate state as begin within .01 meV (adopted from EPW source)
                    self.cb_vecs[kidx].append(vecs[:, idx])
                    ctr += 1
                else:
                    break

            if ctr == 1 and self.SObool and verbosity >= 2:
                print(f"\nWARNING: spin-orbit calc but cb spin states are not degenerate to 1e-5 eV, kidx={kidx}\n")
            if verbosity >= 3:
                print(f"kidx={kidx}, cb_vec[0:5]= {self.cb_vecs[kidx, :5]}")

        if not self.SObool:
            # 2-fold degeneracy for spin. Not sure why this is necessary, but
            # it is included in Tommy's code...
            energiesEV = energiesEV.repeat_interleave(2)
            # dont need to interleave eigenvecs (if stored) since we only
            # store the vb and cb anyways.
        eigVals[:] = energiesEV[:nbands]
        end_time = time.time() if self.NNConfig['runtime_flag'] else None
        print(f"eigvalsh and storing energies, elapsed time: {(end_time - start_time):.2f} seconds") if self.NNConfig['runtime_flag'] else None

        '''
        # Testing with random matrix
        start_time = time.time() if self.NNConfig['runtime_flag'] else None
        test_H = torch.randn(2000, 2000, dtype=torch.complex128)
        eigenvalues = torch.linalg.eigvalsh(test_H)
        end_time = time.time() if self.NNConfig['runtime_flag'] else None
        total_time = end_time - start_time
        print(f"Generating and diagonalizing a random 2000x2000 matrix. Time: {total_time:.2f} seconds") if self.NNConfig['runtime_flag'] else None
        '''
        
        if requires_grad: 
            return eigVals
        else: 
            return eigVals.detach()


    def calcBandStruct(self, grad=False, cachedMats_info=None): 
        if grad: 
            return self.calcBandStruct_withGrad(cachedMats_info)
        else: 
            return self.calcBandStruct_noGrad(cachedMats_info)


    def calcBandStruct_withGrad(self, cachedMats_info=None):
        '''
        Multiprocessing is not implemented due to the requirement to keep gradients.
        '''
        
        nbands = self.system.nBands
        nkpt = self.system.getNKpts()
        bandStruct = torch.zeros([nkpt, nbands])
        for kidx in range(nkpt):
            eigValsAtK = self.calcEigValsAtK(kidx, cachedMats_info, requires_grad=True)
            bandStruct[kidx,:] = eigValsAtK
        
        return bandStruct


    def calcBandStruct_noGrad(self, cachedMats_info=None):
        """
        Multiprocessing is implemented. However, the returned bandStruct doesn't have gradients.
        """
        nbands = self.system.nBands
        nkpt = self.system.getNKpts()

        bandStruct = torch.zeros([nkpt, nbands], requires_grad=False)
        if (self.NNConfig['num_cores']==0): 
            # No multiprocessing
            for kidx in range(nkpt):
                eigValsAtK = self.calcEigValsAtK(kidx, cachedMats_info, requires_grad=False)
                bandStruct[kidx,:] = eigValsAtK
        else: # multiprocessing
            # print(f"The size of cachedMats_info is: {sys.getsizeof(cachedMats_info)/1024} KB")
            args_list = [(kidx, cachedMats_info, False) for kidx in range(nkpt)]
            with mp.Pool(self.NNConfig['num_cores']) as pool:
                eigValsList = pool.starmap(self.calcEigValsAtK, args_list)
            bandStruct = torch.stack(eigValsList)
        return bandStruct


    def calcDefPots(self, cachedMats_info=None, requires_grad=True, verbosity=2): 
        defpot_tensors = []
        
        for defPot_entry in self.system.defPotInfo: 
            kidx_VB = int(defPot_entry[0])
            kidx_CB = int(defPot_entry[2])
            def_scale = defPot_entry[4]

            eigValsAtVB = self.calcEigValsAtK(kidx_VB, cachedMats_info, requires_grad=requires_grad, verbosity=verbosity)
            eigValsAtVB_def = self.calcEigValsAtK(kidx_VB, cachedMats_info, requires_grad=requires_grad, def_H=True, def_scale=def_scale, verbosity=verbosity)
            eigValsAtCB = self.calcEigValsAtK(kidx_CB, cachedMats_info, requires_grad=requires_grad, verbosity=verbosity)
            eigValsAtCB_def = self.calcEigValsAtK(kidx_CB, cachedMats_info, requires_grad=requires_grad, def_H=True, def_scale=def_scale, verbosity=verbosity)

            gap_org = eigValsAtCB[int(defPot_entry[3])] - eigValsAtVB[int(defPot_entry[1])]
            gap_def = eigValsAtCB_def[int(defPot_entry[3])] - eigValsAtVB_def[int(defPot_entry[1])]
            defpot = (gap_org - gap_def) / 2 * (1+def_scale**3) / (1-def_scale**3)
            defpot_tensors.append(defpot)

        return torch.stack(defpot_tensors)# the same number of defpots

    def calcEffMasses(self, bs):
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
        
        kpt0 = 1e10 / bohr_to_ang * self.system.kpts[self.idx_gap]     # a.u.^-1
        kpt1 = 1e10 / bohr_to_ang * self.system.kpts[self.idx_gap - 1] # a.u.^-1
        
        dk = kpt1 - kpt0
        dk_mag = np.linalg.norm(dk)
        
        # -----------------------------
        # Extract energies at band extrema in eV
        # -----------------------------
        vb0 = bs[self.idx_gap, self.idx_vb]
        vb1 = bs[self.idx_gap - 1, self.idx_vb]

        cb0 = bs[self.idx_gap, self.idx_cb]
        cb1 = bs[self.idx_gap - 1, self.idx_cb]

        # -----------------------------
        # Compute second derivative
        # -----------------------------
        # E1 - E0 = 1/2 E'' (|Δk|)^2  => E'' = 2 ΔE / (|Δk|)^2
        # dE_vb = (vb0 - vb1) * eV_to_J  # convert to J
        # E_dp_vb = 2 * dE_vb / (dk_mag ** 2)
        dE_vb = - (vb1 - 2 * vb0 + vb1) * eV_to_J  # convert to J
        E_dp_vb = dE_vb / (dk_mag ** 2)
        
        # dE_cb = (cb1 - cb0) * eV_to_J  # convert to J
        # E_dp_cb = 2 * dE_cb / (dk_mag ** 2)
        dE_cb = (cb1 - 2 * cb0 + cb1) * eV_to_J  # convert to J
        E_dp_cb = dE_cb / (dk_mag ** 2)

        # -----------------------------
        # Effective mass
        # -----------------------------
        m_eff_vb = hbar**2 / E_dp_vb / m_e
        m_eff_cb = hbar**2 / E_dp_cb / m_e

        eff_masses[0] = m_eff_vb
        eff_masses[1] = m_eff_cb

        return eff_masses
    
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
            print(f"\tinitializing coupling SO + NL: qpt {qidx+1}/{nqp}")
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
        q = torch.norm(gqDiff, dim=2).view(-1,1)

        if self.NNConfig["local_env_corr"]:
            # Precompute the necessary chain rule elements for DeltaV derivative coupling
            dv_lsd_dR_all = []
            structFactBeta = []
            for beta in range(self.system.getNAtoms()):
                gqDiffDotBeta = torch.sum(gqDiff * self.system.atomPos[beta], axis=2)
                tmpStructFact = (1.0 / self.system.getCellVolume()) * (torch.cos(gqDiffDotBeta) + 1j * torch.sin(gqDiffDotBeta))
                structFactBeta.append(tmpStructFact)

                LSD_atomType = self.system.atomTypes[beta]
                indx_beta    = torch.where(self.system.atom_indices[LSD_atomType] == beta)[0].squeeze(0).item()
                N_beta       = self.system.env_descriptors[LSD_atomType][indx_beta].unsqueeze(0)  # (1, n_descr)

                dv_lsd_dR = self.compute_dv_lsd_dR(LSD_atomType, N_beta, q, self.system.atomPos)
                dv_lsd_dR_all.append(dv_lsd_dR)

        for alpha, gamma in atomgammaidxs:
            atomType = self.system.atomTypes[alpha]

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
                atomFF = self.model(q)
                atomFF = atomFF[:, thisAtomIndex].view(nbv, nbv)
                lr_coeff = self.PPparams[atomType][4]
                atomFF = atomFF + long_range_correction(torch.norm(gqDiff, dim=2), self.LRgamma, lr_coeff)
            else:
                #atomFF = pot_func(torch.norm(gqDiff, dim=2), self.PPparams[self.system.atomTypes[alpha]])
                atomFF = pot_funcLR(torch.norm(gqDiff, dim=2), self.PPparams[self.system.atomTypes[alpha]], self.LRgamma)

            # Multiply by structFact before LSD terms to avoid double counting during chain rule
            atomFF.to(torch.complex128)
            atomFF = structFact * atomFF

            atomFF_LSD = torch.zeros_like(atomFF)
            if self.NNConfig["local_env_corr"]:
                descriptors = self.system.env_descriptors[atomType]
                indx_alpha = torch.where(self.system.atom_indices[atomType] == alpha)[0].squeeze(0)
                N_alpha = descriptors[indx_alpha, :]
                
                N_alphas = N_alpha.repeat(q.shape[0], 1)
                
                x_input = torch.cat([N_alphas, q], dim=1)
                
                delta_v_alpha = self.LSDmodels[atomType](x_input).view(nbv, nbv)

                atomFF_LSD += structFact * delta_v_alpha

                # --- Chain rule term ∂v/∂N * ∂N/∂R ---
                for beta in range(self.system.getNAtoms()):
                    atomFF_LSD += structFactBeta[beta] * dv_lsd_dR_all[beta][gamma]
                    # Now we loop over all atoms... beta? Sorry, this notation is SUPER confusing.
                    # In the mathematical documentation, we represent the local potential
                    # V_loc(r) = \sum_\alpha v_\alpha(r). Alpha is an arbitrary atom index.
                    # When we take a derivative, we take the derivative with respect to 
                    # # a specific atom, \mu.
                    # dV^loc(r)/dR_\mu = \sum_\alpha dv_\alpha(r)/dR_\mu. 
                    # This derivative is only nonzero if \alpha = \mu, so we got used to writing
                    # dV^loc(r)/dR_\alpha = dv_\alpha(r)/dR_\alpha. 
                    # This is kind of sloppy notation. We should have written 
                    # dV_loc(r)/dR_\mu = dv_\mu(r)/dR_\mu. 
                    # Now we're getting kicked for it. In truth, the index "alpha" in this loop 
                    # should be called "mu" because it is indexing the derivative atom R_\mu!
                    # It never mattered before because we only ever needed one index anyway.
                    # However, for the LSD potential
                    # dV^lsd/dR_\mu = \sum_\alpha dv^lsd_\alpha(r)/dR_\mu
                    # is NOT, I repeat, NOT just dV^lsd/dR_\mu = dv^lsd_\mu(r)/dR_\mu !
                    # The LSD potential is pairwise, not independent, so there are contributions from
                    # atoms other than \mu to its derivative. This "mu" vs. "alpha" distinction becomes important.
                    # For legacy reasons, I will not change the above loop variable to "mu", even though
                    # it is indexing over the derivative variable. I will leave it as alpha.
                    # I will call the true "alpha" term "beta" because alpha was already taken)
                    # i.e. \mu -> \alpha and \alpha -> \beta
                    # Basically, the code can be understood by thinking about the derivative as
                    # dV^lsd/dR_\alpha = \sum_\beta dv^lsd_\beta(r)/dR_\alpha
                    # Now, for the LSD derivative, we need the chain rule term. In our new notation 
                    # dv^lsd_\beta/dN_\beta \cdot dN_\beta/dR_\alpha.
                    
                    # This line is implementing the lookup for element
                    # dN_\alpha/dR_{\mu\gamma}
                    # But in out weird legacy indexing where \mu -> \alpha and \alpha -> \beta
                    # dN_\beta/dR_{\alpha\gamma}
                    # dN_dR = self.system.dG2_dR[beta, alpha, gamma]
                    # if abs(dN_dR) < 1e-14:
                    #     continue # skip atoms with zero contribution
                    
                    # Compute form factor term
                    atomFF_LSD += structFactBeta[beta] * dv_lsd_dR_all[beta][gamma]

            dV[:nbv, :nbv] = prefactor * (atomFF + atomFF_LSD)

            if self.SObool:
                # local potential has delta function on spin --> block diagonal
                dV[nbv:, nbv:] = prefactor * (atomFF + atomFF_LSD)

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

    def compute_dV_dn(self, atomType, N_alpha, qvals):
        """
        Computes ∂v_lsd(q, N) / ∂N for each q point and each descriptor.

        Parameters
        ----------
        atomType : str
        N_alpha  : (1, n_descr) tensor — descriptor vector for this atom
        qvals    : (nbv*nbv, 1) tensor — q grid

        Returns
        -------
        dV_dn : (nbv*nbv, n_descr) tensor — gradient of network output w.r.t each descriptor
        """
        q = qvals.clone().detach().requires_grad_(True)   # (nbv*nbv, 1) — no grad needed on q
        N = N_alpha.detach().requires_grad_(True)         # (1, n_descr) — leaf, grad w.r.t. this
        N_rep = N.expand(q.shape[0], -1)                  # (nbv*nbv, n_descr)
        print(f"N.requires_grad   = {N.requires_grad} {N.grad_fn}")
        print(f"N_rep.requires_grad = {N_rep.requires_grad} {N.grad_fn}")
        x_input = torch.cat([N_rep, q], dim=1)            # (nbv*nbv, n_descr + 1)
        print(f"x_input.requires_grad = {x_input.requires_grad} {x_input.grad_fn}")
        v = self.LSDmodels[atomType](x_input)             # (nbv*nbv, 1)
        print(f"v.requires_grad   = {v.requires_grad} {v.grad_fn}")
        dV_dn = torch.autograd.grad(
            outputs      = v,                        # scalar
            inputs       = N,
            grad_outputs = torch.ones_like(v),
            create_graph = False,
            retain_graph = False
        )[0]                                               # (1, n_descr)

        return dV_dn                            # (n_descr,)

    def calcCouplings(self, qlist=None, atomgammaidxs=None, symm_equiv=None):
        """
        All we do here is call buildCouplingMats(), check the we have the
        correct eigenstates (from bandstructure calculation) to compute the
        desired matrix elements, then compute the expectation values.

        This return a dictionary with keys that are tuples: 
        (atomidx, gamma, qidx, 'vb'/'cb')
        and values are just floats (the coupling value). The couplings are in
        units of eV/Bohr.

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
                            # print(f"cb degeneracy info: {n_right} right, {n_left} left")
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
                            avg_couple[(key[0], 'cb')] += torch.sqrt(mag / (n_right * n_left)) / len(symm_equiv[key[0]])

                            n_right = len(self.vb_vecs[needKidx])
                            n_left = len(self.vb_vecs[self.idx_gap])
                            # print(f"vb degeneracy info: {n_right} right, {n_left} left")
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
                            avg_couple[(key[0], 'vb')] += torch.sqrt(mag2 / (n_right * n_left)) / len(symm_equiv[key[0]])

            # build ret_dict 
            for key in dV_dict:
                if key[0] in symm_equiv_compat:
                    if key[1] in symm_equiv_compat[key[0]]:
                        #avg_cb = avg_couple[(key[0], 'cb')]
                        #avg_vb = avg_couple[(key[0], 'vb')]
                        #ret_dict[key+(qid,'cb')] = torch.sqrt(avg_cb.conj() * avg_cb).real * AUTOEV
                        #ret_dict[key+(qid,'vb')] = torch.sqrt(avg_vb.conj() * avg_vb).real * AUTOEV
                        ret_dict[key + (qid,'cb')] = avg_couple[(key[0], 'cb')] * AUTOEV
                        ret_dict[key + (qid,'vb')] = avg_couple[(key[0], 'vb')] * AUTOEV

                else:
                    n_right = len(self.cb_vecs[needKidx])
                    n_left = len(self.cb_vecs[self.idx_gap])
                    if n_right > 1:
                        right_vecs = torch.stack(self.cb_vecs[needKidx], dim=-1)
                    else:
                        ###############
                        # There are issues with self.cb_vecs when multiprocessing is turned on. 
                        # The root cause should be somewhere in calcEigValsAtK() function, 
                        # in the case of multiprocessing. self.cb_vecs and self.vb_vecs are not 
                        # properly gathered back to the main process.
                        ###############
                        right_vecs = self.cb_vecs[needKidx][0].view(-1,1)
                    if n_left > 1:
                        left_vecs = torch.stack(self.cb_vecs[self.idx_gap], dim=0)
                    else:
                        left_vecs = self.cb_vecs[self.idx_gap][0].view(1,-1)
                    cpl = torch.matmul(dV_dict[key], right_vecs) # batched multiplication of all degenerate bands
                    cpl = torch.matmul(torch.conj(left_vecs), cpl) # n_right * n_left dot products in the elements of a matrix
                    cpl_mag = torch.sum(cpl.conj() * cpl).real
                    ret_dict[key + (qid,'cb')] = torch.sqrt((cpl_mag / (n_right * n_left))) * AUTOEV # average coupling from degenerate subspace

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
                    cpl_mag = torch.sum(cpl.conj() * cpl).real
                    ret_dict[key + (qid,'vb')] = torch.sqrt((cpl_mag / (n_right * n_left))) * AUTOEV

        return ret_dict

    def calcCouplings_diag_fd(
        self,
        delta=1e-6,
        degen_tol_ev=1e-5,
        debug=False,
        select_atomidx=None,
        select_gamma=None,
        base_vals=None,
    ):
        """
        Compute diagonal e-ph couplings using one-sided finite differences at Gamma (q=0).

        Evaluates band-edge energy derivatives with respect to atomic 
        displacements by constructing one displaced system per atom
        and direction: +delta in Cartesian coordinates (x, y, z). The
        displacement is applied to the scaled atomic positions in
        system.atomPos (Bohr). For each displaced system, it computes the
        eigenvalues at the bandgap k-point (idxGap) and forms the finite difference:
            dE/dR = (E_plus - E_base) / delta
            coupling = sqrt( sum((dE/dR)^2) / (d * d) )  over a dim-d degenerate
              subspace

        The couplings are returned for the band indices specified in the input
        files (idxVB/idxCB) at the Gamma q-point and the bandgap k-point
        (idxGap). Energies are converted to eV.
        This routine returns coupling magnitudes in eV/Bohr.
        
        The original system is not modified; each displacement is applied to a
        deep-copied BulkSystem and evaluated with a temporary Hamiltonian.

        Optional controls:
          - select_atomidx: iterable of atom indices (or a single int) to include.
          - select_gamma: iterable of Cartesian directions (0,1,2) (or a single int).
          - base_vals: precomputed eigenvalues at Gamma (Hartree) to reuse.
        """
        if not isinstance(self.system.idxVB, int):
            raise ValueError("need to specify vb index for diagonal coupling")
        if not isinstance(self.system.idxCB, int):
            raise ValueError("need to specify cb index for diagonal coupling")

        def eigvals_no_order(ham, kidx, requires_grad=True):
            H = ham.buildHtot(kidx, requires_grad=requires_grad)
            vals = torch.linalg.eigvalsh(H)
            return vals[:ham.system.nBands]

        def collect_degen_indices(vals, start_idx, direction, tol_ha):
            ref = vals[start_idx]
            idxs = [start_idx]
            idx = start_idx + direction
            while 0 <= idx < len(vals):
                if torch.abs(vals[idx] - ref) <= tol_ha:
                    idxs.append(idx)
                    idx += direction
                else:
                    break
            return sorted(idxs)

        zero_vec = torch.zeros(3, dtype=self.system.kpts.dtype)
        if self.system.getNQpts() != 1 or not torch.allclose(self.system.qpts[0], zero_vec, atol=1e-12):
            raise ValueError("calcCouplings_diag_fd requires q-point list to be only Gamma")

        kidx_gap = getattr(self, "idx_gap", None)
        if kidx_gap is None:
            kidx_gap = getattr(self.system, "idxGap", None)
        if not isinstance(kidx_gap, int):
            raise ValueError("calcCouplings_diag_fd requires a valid idxGap for the bandgap k-point")
        if not (0 <= kidx_gap < self.system.getNKpts()):
            raise ValueError("calcCouplings_diag_fd requires idxGap to be within the k-point list")


        print(f"Bandgap kidx = {kidx_gap}")
        qidx_gamma = 0

        if base_vals is None:
            base_vals = eigvals_no_order(self, kidx_gap, requires_grad=True)
        else:
            base_vals = torch.as_tensor(
                base_vals,
                dtype=self.system.kpts.dtype,
                device=self.system.kpts.device,
            )
        degen_tol_ha = degen_tol_ev / AUTOEV
        
        # Note, user's inputs of idxVB/idxCB shouldn't include the artificial 
        # 2x interleaving of eigenenergies when SOC is off. 
        vb_degen = collect_degen_indices(base_vals, self.system.idxVB, -1, degen_tol_ha)
        cb_degen = collect_degen_indices(base_vals, self.system.idxCB, 1, degen_tol_ha)
        unit_scale = AUTOEV  # report energies/couplings in eV and (eV/Bohr)^2
        unit_label = "eV"
        base_vals_out = base_vals * unit_scale
        
        if debug:
            print("\n[calcCouplings_diag_fd] Debug info")
            print("Coupling units: eV/Bohr")
            print(f"delta (Bohr): {delta}, gap kidx: {kidx_gap}, Gamma qidx: {qidx_gamma}")
            print(f"Inputs of idxVB: {self.system.idxVB}, idxCB: {self.system.idxCB}")
            if not self.SObool: 
                print(f"True idxVB (without 2x interleaving): {int((self.system.idxVB-1)/2)}, idxCB: {int(self.system.idxCB/2)}")
            print(f"VB degenerate indices: {vb_degen}. Energies ({unit_label}): " + ", ".join([f"{base_vals_out[i].item():.5e}" for i in vb_degen]))
            print(f"CB degenerate indices: {cb_degen}. Energies ({unit_label}): " + ", ".join([f"{base_vals_out[i].item():.5e}" for i in cb_degen]))
            print(f"Gap k-point (Bohr^-1): {self.system.kpts[kidx_gap]}")
            print("Atom positions (scaled, Bohr):")
            print(self.system.atomPos)

        if select_atomidx is None:
            atom_indices = list(range(self.system.getNAtoms()))
        elif isinstance(select_atomidx, int):
            atom_indices = [select_atomidx]
        else:
            atom_indices = list(select_atomidx)

        if select_gamma is None:
            gamma_indices = [0, 1, 2]
        elif isinstance(select_gamma, int):
            gamma_indices = [select_gamma]
        else:
            gamma_indices = list(select_gamma)

        ret_dict = {}
        for atomidx in atom_indices:
            for gamma in gamma_indices:
                if debug:
                    print(f"\natomidx={atomidx}, gamma={gamma}")
                system_plus = copy.copy(self.system)
                system_plus.atomPos = self.system.atomPos.clone()
                system_plus.atomPos[atomidx, gamma] = system_plus.atomPos[atomidx, gamma] + delta
                if debug:
                    print("Displaced atom position +delta (Bohr): " + f"{system_plus.atomPos[atomidx]}")

                ham_plus = Hamiltonian(
                    system_plus,
                    self.PPparams,
                    self.atomPPorder,
                    self.device,
                    NNConfig=self.NNConfig,
                    iSystem=self.iSystem,
                    SObool=self.SObool,
                    cacheSO=self.cacheSO,
                    NN_locbool=self.NN_locbool,
                    model=self.model,
                    coupling=False,
                    LSDmodels=self.LSDmodels
                )

                vals_plus = eigvals_no_order(ham_plus, kidx_gap, requires_grad=True) * unit_scale

                vb_diff = (vals_plus[vb_degen] - base_vals_out[vb_degen]) / delta
                cb_diff = (vals_plus[cb_degen] - base_vals_out[cb_degen]) / delta
                vb_cpl = torch.sqrt(torch.sum(vb_diff * vb_diff) / (len(vb_degen) * len(vb_degen)))
                cb_cpl = torch.sqrt(torch.sum(cb_diff * cb_diff) / (len(cb_degen) * len(cb_degen)))
                if debug:
                    print(f"VB energies +delta ({unit_label}): " + ", ".join([f"{vals_plus[i].item():.5e}" for i in vb_degen]))
                    print(f"VB energies base ({unit_label}): " + ", ".join([f"{base_vals_out[i].item():.5e}" for i in vb_degen]))
                    print(f"CB energies +delta ({unit_label}): " + ", ".join([f"{vals_plus[i].item():.5e}" for i in cb_degen]))
                    print(f"CB energies base ({unit_label}): " + ", ".join([f"{base_vals_out[i].item():.5e}" for i in cb_degen]))
                    print(f"VB fd ({unit_label}/Bohr): {vb_cpl.item():.5e}")
                    print(f"CB fd ({unit_label}/Bohr): {cb_cpl.item():.5e}")

                ret_dict[(atomidx, gamma, qidx_gamma, 'vb')] = vb_cpl
                ret_dict[(atomidx, gamma, qidx_gamma, 'cb')] = cb_cpl

        return ret_dict

    def compute_dv_lsd_dR(self, atomType, N_alpha, qvals, atomPos):
        """
        Computes ∂v_lsd(q, N_alpha) / ∂N_alpha
        N_alpha: scalar (float)
        qvals: (nbv*nbv, 1) tensor
        Returns (nbv, nbv) tensor
        """
        
        q = qvals.clone().detach().requires_grad_(True)
        N = N_alpha.repeat(q.shape[0], 1)
        print(f"q {q.shape}")
        print(f"N {N.shape}")
        x_input = torch.cat([N, q], dim=1)
        v = self.LSDmodels[atomType](x_input)
        
        dv_dR = torch.autograd.grad(
            outputs=v,
            inputs=atomPos,
            grad_outputs=torch.ones_like(v),
            create_graph=True
        )[0]
        print(f"dv_dR {dv_dR.shape}\n{dv_dR}")
        nbv = self.basis.shape[0]
        return dv_dR.view(nbv, nbv)
    
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
        SO integral exactly as daniel weinberg's c code computes it,
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
        NL integral exactly as daniel weinberg's c code computes it,
        vectorized over k,kp (so assuming k,kp are vectors of
        length nbv). This is useful for testing. The 
        _nlIntegral_vect() routine is much more robust. It's
        pretty clear that daniel weinberg's routine is not well converged for
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

    def set_LSDmodels(self, newmodels):
        """
        Use this to set the LSD models for each atom type
        """
        self.LSDmodels = {k: v for k, v in newmodels.items()}

    def get_PPparams(self):
        return copy.deepcopy(self.PPparams)
    
    def get_LSDparams(self):
        return copy.deepcopy(self.system.LSDparams)

    def set_PPparams(self, newparams):
        """
        Set new values for the algebraic PP "a" params.
        This is useful when performing optimization of the algebraic
        parts of the PP.
        """
        self.PPparams = newparams

    def set_LSDparams(self, newparams):
        """
        Set new values for the algebraic PP "a" params.
        This is useful when performing optimization of the algebraic
        parts of the PP.
        """
        self.system.LSDparams = newparams


def initAndCacheHams(systemsList, NNConfig, PPparams, atomPPOrder, device, model=None, LSDmodels=None):
    """
    Initialize the ham class for each BulkSystem. 
    dummy_ham is used to initialize and store the cached SOmats and NLmats in dict cachedMats. 
    As I initialize dummy_ham, immediately load them into share memory
    Use a dict "cachedMats_info" to store dtype and shape
    Then remove dummy_ham, and any intermediate variables
    """
    print("\nInitializing the ham class for each BulkSystem. Cache-ing the SOmats, NLmats, and putting them into shared memeory. ")
    hams = []
    cachedMats_info = {}
    shm_dict_SO = {}
    shm_dict_NL = {}
    for iSys, sys in enumerate(systemsList):
        start_time = time.time()

        # Here I separate: 
        # 1. SObool = False --> Just initialize ham. No storage / moving is needed. 
        # 2. SObool = True, no parallel --> Initialize ham with cache. No storage / moving is needed.
        # 3. SObool = True, yes parallel --> Do the complicated storage / moving. 
        if not NNConfig['SObool']: 
            ham = Hamiltonian(sys, PPparams, atomPPOrder, device, NNConfig=NNConfig, iSystem=iSys, SObool=NNConfig['SObool'], cacheSO=NNConfig['cacheSO'], LSDmodels=LSDmodels, coupling=sys.fit_eph)
            cachedMats_info = None
            shm_dict_SO = None
            shm_dict_NL = None
        elif (NNConfig['SObool']) and (NNConfig['num_cores']==0):
            print(f"num_cores set to {NNConfig['num_cores']}. Initializing Hamiltonian without caching SO mats.") 
            ham = Hamiltonian(sys, PPparams, atomPPOrder, device, NNConfig=NNConfig, iSystem=iSys, SObool=NNConfig['SObool'], cacheSO=NNConfig['cacheSO'], LSDmodels=LSDmodels, coupling=sys.fit_eph)
            cachedMats_info = None
            shm_dict_SO = None
            shm_dict_NL = None
        elif (NNConfig['SObool']) and (NNConfig['cacheSO']==0):
            print(f"cacheSO set to {NNConfig['cacheSO']}. Initializing Hamiltonian without caching SO mats.") 
            ham = Hamiltonian(sys, PPparams, atomPPOrder, device, NNConfig=NNConfig, iSystem=iSys, SObool=NNConfig['SObool'], cacheSO=False, LSDmodels=LSDmodels, coupling=sys.fit_eph)
            cachedMats_info = None
            shm_dict_SO = None
            shm_dict_NL = None
        else:
            ham = Hamiltonian(sys, PPparams, atomPPOrder, device, NNConfig=NNConfig, iSystem=iSys, SObool=True, cacheSO=False, LSDmodels=LSDmodels, coupling=sys.fit_eph)
            dummy_ham = Hamiltonian(sys, PPparams, atomPPOrder, device, NNConfig=NNConfig, iSystem=iSys, SObool=NNConfig['SObool'], LSDmodels=LSDmodels, coupling=sys.fit_eph)

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


def set_LSDModels(ham, LSDmodels):
    ham.set_LSDModels(LSDmodels)
    