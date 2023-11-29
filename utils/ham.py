import numpy as np
import torch
from scipy.special import erf
from scipy.integrate import quad, quadrature, quad_vec
import time
import sys

from constants.constants import *
from utils.pp_func import pot_func


class Hamiltonian:
    def __init__(
        self,
        system,
        PPparams,
        atomPPorder,
        device,
        SObool = False,
        cacheSO = True,
        NN_locbool = False,
        model = None,
        coupling = False
    ):
        """
        The Hamiltonian is initialized by passing it an initialized and
        populated bulkSystem class, which contains all the relevant 
        information about the basis, atoms, etc. 
        PPparams should be formatted as a dict of lists, where
        PPparams[atomkey] = [params], and atomkey is the string symbol of the atom.
        "atomPPorder" is an ordered list of the unique atoms in the system. If 
        using a NN model for local potential, it is important that this arg is
        consistent with the construction of the NN.
        "device" should be specified using torch for cpu vs gpu.
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
        self.SObool = SObool
        self.cacheSO = cacheSO
        self.NN_locbool = NN_locbool
        self.model = model
        self.coupling = coupling   # fit the e-ph couplings? boolean


        # if spin orbit, do a bunch of caching to speed up the inner loop 
        # of the optimization. This uses more memory (storing natom * nkpt
        # matrices of size 4*nbasis^2) in exchange for avoiding loops
        # over the basis within the optimization inner loop.
        if SObool and cacheSO:
            #self.SOmats = self.initSOmat()
            self.SOmats = self.initSOmat_fast()
            # check if nonlocal potentials are included, if so, cache them
            checknl = False
            for alpha in range(system.getNAtomTypes()):
                if abs(self.PPparams[self.system.atomTypes[alpha]][6]) > 1e-8:
                    checknl = True
                    break
                elif abs(self.PPparams[self.system.atomTypes[alpha]][7]) > 1e-8:
                    checknl = True
                    break
            if checknl:
                #self.NLmats = self.initNLmat()
                self.NLmats = self.initNLmat_fast()
        elif SObool and not cacheSO:
            raise NotImplementedError("currently we only support caching the SO matrices")
        
        if self.coupling:
            nkpt = self.system.getNKpt()
            nbv = self.system.basis.shape[0]
            if SObool: nbv *= 2
            self.vb_vecs = torch.zeros([nkpt, nbv], dtype=torch.complex128)
            self.cb_vecs = torch.zeros([nkpt, nbv], dtype=torch.complex128)

            if not isinstance(self.system.idx_bv, int):
                raise ValueError("need to specify vb, cb indices for coupling")
            elif not isinstance(self.system.idx_cb, int):
                raise ValueError("need to specify vb, cb indices for coupling")
            else:
                self.idx_vb = self.system.idx_vb
                self.idx_cb = self.system.idx_cb

        # send things to gpu, if enabled ??
        # Or is it better to send some things at the last minute before diagonalization?
        if model is not None:
            model.to(device)
        


    def buildHtot(self, kidx, defbool=False):
        """
        Build the total Hamiltonian for a given kpt, specified by its kidx.
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
        Htot = self.buildVlocMat(defbool=defbool, addMat=Htot)

        if self.SObool:
            Htot = self.buildSOmat(kidx, addMat=Htot)
            Htot = self.buildNLmat(kidx, addMat=Htot)

        if self.device.type == "cuda":
            # !!! is this sufficient to match previous performance?
            # This limits data movement to gpu (good), but obviously
            # performs construction of H on cpu (at least the first time?), 
            # which might be slower.
            Htot.to(self.device)

        return Htot

    
    def buildVlocMat(self, defbool=False, addMat=None):
        """
        Computes the local potential, either using the algebraic form
        or the NN form.
        V_{i,j} = <G_i|V|G_j> = \sum_k [e^{-i(G_i-G_j)\cdot\tau_k} * v(|G_i-G_j|) / (V_cell)].
        "addMat" can be set to be a partially constructed Hamiltonian matrix, to
        which the local potential can be added. Might help save slightly on memory. 
        """
        nbv = self.basis.shape[0]
        gdiff = torch.stack([self.basis] * nbv, dim=1 ) - self.basis.repeat(nbv,1,1)

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
            if not defbool:
                gdiffDotTau = torch.sum(gdiff * self.system.atomPos[alpha], axis=2)
                sfact_re = 1/self.system.getCellVolume() * torch.cos(gdiffDotTau)
                sfact_im = 1/self.system.getCellVolume() * torch.sin(gdiffDotTau)
            else:
                # technically, I think gdiff should be different for deformed calc,
                # but this was not implemented in Dipti's code...?
                # the basis is technically rescaled in the deformed basis, so the
                # differences are also rescaled...?
                gdiffDotTau = torch.sum(gdiff * self.system.atomPosDef[alpha], axis=2)
                sfact_re = 1/self.system.getCellVolumeDef() * torch.cos(gdiffDotTau)
                sfact_im = 1/self.system.getCellVolumeDef() * torch.sin(gdiffDotTau)

            thisAtomIndex = np.where(self.system.atomTypes[alpha]==self.atomPPorder)[0]
            if len(thisAtomIndex)!=1: 
                raise ValueError("Type of atoms in PP. ")
            thisAtomIndex = thisAtomIndex[0]

            if self.NN_locbool:
                atomFF = self.model(torch.norm(gdiff, dim=2).view(-1,1))
                atomFF = atomFF[:, thisAtomIndex].view(nbv, nbv)
            else:
                atomFF = pot_func(torch.norm(gdiff, dim=2), self.PPparams[self.system.atomTypes[alpha]])

            if self.SObool:
                # local potential has delta function on spin --> block diagonal
                Vmat[:nbv, :nbv] = Vmat[:nbv, :nbv] + atomFF * torch.complex(sfact_re, sfact_im)
                Vmat[nbv:, nbv:] = Vmat[nbv:, nbv:] + atomFF * torch.complex(sfact_re, sfact_im)
            else:
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
                    gdiff = self.basis[j] - self.basis[i]

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
        
        SOmats = np.empty([nkp, self.system.getNAtoms()], dtype=object)
        for id1 in range(nkp):
            for id2 in range(self.system.getNAtoms()):
                SOmats[id1,id2] = np.zeros([2*nbv, 2*nbv], dtype=np.complex128)

        # this can be parallelized over kpoints, but it's not critical since
        # this is only done once during initialization
        for kidx in range(nkp):
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
                if not defbool:
                    gdiffDotTau = gdiff * self.system.atomPos[alpha]
                    gdiffDotTau = np.sum(gdiffDotTau.numpy(force=True), axis=2)
                    sfact_re = 1 / self.system.getCellVolume() * np.cos(gdiffDotTau)
                    sfact_im = 1 / self.system.getCellVolume() * np.sin(gdiffDotTau)
                else:
                    gdiffDotTau = gdiff * self.system.atomPosDef[alpha]
                    gdiffDotTau = np.sum(gdiffDotTau.numpy(force=True), axis=2)
                    sfact_re = 1 / self.system.getCellVolumeDef() * np.cos(gdiffDotTau)
                    sfact_im = 1 / self.system.getCellVolumeDef() * np.sin(gdiffDotTau)

                # build SO matrix
                # up up
                # -i * gcp dot S_up,up is pure imag: -i/2 * (gcp.z)
                real_part = prefactor * isum * 0.5 * gcross[:,:, 2] * sfact_im
                im_part = prefactor * isum * -0.5 * gcross[:,:, 2] * sfact_re
                SOmats[kidx,alpha][:nbv, :nbv] = real_part + 1j * im_part

                # dn dn
                # -i * gcp dot S_dn,dn is pure imag: i/2 * (gcp.z)
                real_part = prefactor * isum * -0.5 * gcross[:,:, 2] * sfact_im
                im_part = prefactor * isum * 0.5 * gcross[:,:, 2] * sfact_re
                SOmats[kidx,alpha][nbv:, nbv:] = real_part + 1j * im_part

                # up dn
                # -i * gcp dot S_up,dn is: -i/2 * (gcp.x) - 1/2 * (gcp.y)
                real_part = prefactor * isum * (0.5 * gcross[:,:, 0] * sfact_im -0.5 * gcross[:,:, 1] * sfact_re)
                im_part = prefactor * isum * (-0.5 * gcross[:,:, 0] * sfact_re -0.5 * gcross[:,:, 1] * sfact_im)
                SOmats[kidx,alpha][:nbv, nbv:] = real_part + 1j * im_part

                # dn up
                # -i * gcp dot S_dn,up is: -i/2 * (gcp.x) + 1/2 * (gcp.y)
                real_part = prefactor * isum * (0.5 * gcross[:,:, 0] * sfact_im + 0.5 * gcross[:,:, 1] * sfact_re)
                im_part = prefactor * isum * (-0.5 * gcross[:,:, 0] * sfact_re + 0.5 * gcross[:,:, 1] * sfact_im)
                SOmats[kidx,alpha][nbv:, :nbv] = real_part + 1j * im_part

        return SOmats


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
                    gdiff = self.basis[j] - self.basis[i]

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
        
        NLmats = np.empty([nkp, self.system.getNAtoms(), 2], dtype=object)
        for id1 in range(nkp):
            for id2 in range(self.system.getNAtoms()):
                for id3 in [0,1]:
                    NLmats[id1,id2,id3] = np.zeros([2*nbv, 2*nbv], dtype=np.complex128)

        # this can be parallelized over kpoints, but it's not critical since
        # this is only done once during initialization
        for kidx in range(nkp):
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
            print(f"time int1: {t2-t1}")
            isum2 = self._nlIntegral_vect(inm, jnm, rcut, width2, shift)
            #isum2 = self._nlIntegral_dan(inm, jnm, width2, shift)  # for testing
            t3 = time.time()
            print(f"time int2: {t3-t2}")

            #gdot = torch.dot(gikp, gjkp)
            # this tensordot call is like mat[i,j] = sum_k gikp[i,k] * gjkp[j,k]
            gdot = np.tensordot(gikp, gjkp, axes=[[1],[1]])      
            #prefactor = 12.0 * np.pi / (inm[:, np.newaxis] * jnm)
            prefactor = np.zeros([nbv,nbv], dtype=float)
            denom = inm[:, np.newaxis] * jnm
            ids = np.nonzero(denom)
            prefactor[ids] = 12.0 * np.pi / denom[ids]

            for alpha in range(self.system.getNAtoms()):
                if not defbool:
                    gdiffDotTau = gdiff * self.system.atomPos[alpha]
                    gdiffDotTau = np.sum(gdiffDotTau.numpy(force=True), axis=2)
                    sfact_re = 1 / self.system.getCellVolume() * np.cos(gdiffDotTau)
                    sfact_im = 1 / self.system.getCellVolume() * np.sin(gdiffDotTau)
                else:
                    gdiffDotTau = gdiff * self.system.atomPosDef[alpha]
                    gdiffDotTau = np.sum(gdiffDotTau.numpy(force=True), axis=2)
                    sfact_re = 1 / self.system.getCellVolumeDef() * np.cos(gdiffDotTau)
                    sfact_im = 1 / self.system.getCellVolumeDef() * np.sin(gdiffDotTau)

            
                # This potential is block diagonal on spin
                # up up, 1st integral
                real_part = prefactor * isum1 * gdot * sfact_re
                im_part = prefactor * isum1 * gdot * sfact_im
                NLmats[kidx,alpha,0][:nbv, :nbv] = real_part + 1j* im_part
                # 2nd integral
                real_part = prefactor * isum2 * gdot * sfact_re
                im_part = prefactor * isum2 * gdot * sfact_im
                NLmats[kidx,alpha,1][:nbv, :nbv] = real_part + 1j * im_part

                # dn dn, 1st integral
                real_part = prefactor * isum1 * gdot * sfact_re
                im_part = prefactor * isum1 * gdot * sfact_im
                NLmats[kidx,alpha,0][nbv:, nbv:] = real_part + 1j * im_part
                # 2nd integral
                real_part = prefactor * isum2 * gdot * sfact_re
                im_part = prefactor * isum2 * gdot * sfact_im
                NLmats[kidx,alpha,1][nbv:, nbv:] = real_part + 1j * im_part

        return NLmats
    
    
    def buildSOmat(self, kidx, addMat=None):
        """
        Build the final SO mat for a given kpoint (specified by its kidx).
        Using the cached SOmats, this function just multiplies by the 
        current values of the PPparams, and then sums over all atoms.
        "addMat" can be set to be a partially constructed Hamiltonian matrix, to
        which the local potential can be added. Might help save slightly on memory.
        """
        nbv = self.basis.shape[0]
        if addMat is not None:
            assert addMat.shape[0] == 2*nbv
            assert addMat.shape[1] == 2*nbv
            SOmatf = addMat
        else:
            SOmatf = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)
        
        for alpha in range(self.system.getNAtoms()):
            if isinstance(self.SOmats[kidx,alpha], torch.Tensor):
                tmp = self.SOmats[kidx,alpha]
            else:
                tmp = torch.tensor(self.SOmats[kidx,alpha])

            SOmatf = SOmatf + tmp * self.PPparams[self.system.atomTypes[alpha]][5]
        
        return SOmatf
    

    def buildNLmat(self, kidx, addMat=None):
        """
        Build the final nonlocal mat for a given kpoint (specified by its kidx).
        Using the cached NLmats, this function just multiplies by the 
        current values of the PPparams, and then sums over all atoms.
        "addMat" can be set to be a partially constructed Hamiltonian matrix, to
        which the local potential can be added. Might help save slightly on memory.
        """
        nbv = self.basis.shape[0]
        if addMat is not None:
            assert addMat.shape[0] == 2*nbv
            assert addMat.shape[1] == 2*nbv
            NLmatf = addMat
        else:
            NLmatf = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)
        
        for alpha in range(self.system.getNAtoms()):
            if isinstance(self.NLmats[kidx,alpha,0], torch.Tensor):
                tmp1 = self.NLmats[kidx,alpha,0]
            else:
                tmp1 = torch.tensor(self.NLmats[kidx,alpha,0])
            if isinstance(self.NLmats[kidx,alpha,1], torch.Tensor):
                tmp2 = self.NLmats[kidx,alpha,1]
            else:
                tmp2 = torch.tensor(self.NLmats[kidx,alpha,1])

            NLmatf = (NLmatf + tmp1 * self.PPparams[self.system.atomTypes[alpha]][6]
                             + tmp2 * self.PPparams[self.system.atomTypes[alpha]][7] )
        
        return NLmatf


    def calcBandStruct(self):
        nbands = self.system.nBands
        nkpt = self.system.getNKpts()

        bandStruct = torch.zeros([nkpt, nbands])

        # this loop should be parallelized for good performance.
        # can be done with shared memory by simply using the multiprocessing module
        for kidx in range(nkpt):
            H = self.buildHtot(kidx)

            if not self.coupling:
                energies = torch.linalg.eigvalsh(H)
                energiesEV = energies * AUTOEV
            else:
                # this will be slow, since torch seems to only support
                # full diagonalization including all eigenvectors. 
                # If computing couplings, it might
                # make sense to implement a custom torch diagonalization wrapper
                # that uses scipy under the hood to allow for better partial
                # diagonalization algorithms.
                ens, vecs = torch.linalg.eigh(H)
                energiesEV = ens * AUTOEV
                self.vb_vecs[kidx, :] = vecs[:, self.idx_vb]
                self.cb_vecs[kidx, :] = vecs[:, self.idx_cb]


            if not self.SObool:
                # 2-fold degeneracy for spin. Not sure why this is necessary, but
                # it is included in Tommy's code...
                energiesEV = energiesEV.repeat_interleave(2)
                # dont need to interleave eigenvecs (if stored) since we only
                # store the vb and cb anyways.

            bandStruct[kidx,:] = energiesEV[:nbands]

        return bandStruct


    def calcCouplings(self):
        raise NotImplementedError()

    
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
        print(f"int2 est. maxerr: {np.amax(err)}")
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
        return self.PPparams
    
    def set_PPparams(self, newparams):
        """
        Set new values for the algebraic PP "a" params.
        This is useful when performing optimization of the algebraic
        parts of the PP.
        """
        self.PPparams = newparams

