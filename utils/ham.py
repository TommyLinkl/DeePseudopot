import numpy as np
import torch
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
        "device" should be specified using torch types for cpu vs gpu.
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
            self.SOmats = self.initSOmat()
        if SObool and not cacheSO:
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
            #Htot = self.buildNLmat(kidx, addmat=Htot)

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
        V_{i,j} = <G_i|V|G_j> = \sum_k [e^{-i(G_i-G_j)\cdot\tau_k} * v(|G_i-G_j|) / (V_cell)]
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
        dr = 2*np.pi / (100 * torch.norm(self.basis[-1]))
        # set radial cutoff ~ 4.2488 Bohr; V(rcut) = 1e-16 for default SOwidth
        rcut = np.sqrt(SOwidth**2 * 16 * np.log(10.0))
        ncut = int(rcut/dr)
        eps = dr / 10.0
        if defbool:
            nkp = 1  # to allow for deformation calcs at a single kpoint
            if idxGap is None:
                raise RuntimeError("need to specify kpt idx of gap in deformed calc")
        else:
            nkp = self.system.getNKpts()
        
        SOmats = np.array([nkp, self.system.getNAtoms()], dtype=object)
        for id1 in range(nkp):
            for id2 in range(self.system.getNAtoms()):
                SOmats[id1,id2] = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)

        # this can be parallelized over kpoints, but it's not critical since
        # this is only done once during initialization
        for kidx in range(nkp):
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

                    for gp in range(1,ncut):
                        r = gp * dr
                        isum += (r**2 * dr * self._bessel1(inm*r, 1/(inm*r + 1e-10)) *
                                torch.exp(-(r/SOwidth)**2) *
                                self._bessel1(jnm*r, 1/(jnm*r + 1e-10)) )

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


    def buildSOmat(self, kidx, addMat=None):
        """
        Build the final SO mat for a given kpoint (specified by its kidx).
        Using the cached SOmats, this functions just multiplies by the 
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
            SOmatf = SOmatf + self.SOmats[kidx,alpha] * self.PPparams[self.system.atomTypes[alpha]][5]
        
        return SOmatf


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
                # make sense to implement a custom torch diagonalization routine
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
        return torch.sin(x) * x1**2 - torch.cos(x) * x1

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

