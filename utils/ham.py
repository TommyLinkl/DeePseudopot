import sys, os
import uuid
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
from concurrent.futures import ThreadPoolExecutor
import gc

from .constants import *
from .pp_func import pot_func, pot_funcLR, long_range_correction, qSpacePot_ft
from .read import init_critical_NNconfig, setNN
from .profiling import PROF
from .threads import available_cpus, init_thread_count, pool_worker_init
from .partial_eig import partial_eigvalsh
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
        NLbool = None,
        cacheSO = True,
        NN_locbool = False,
        model = None,
        coupling = False,
        LSDmodels = None,
        spinModel = None
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
        "SObool" enables the spin-orbit potential. "NLbool" independently enables
        the non-local (l=1 projector) potential; the non-local potential is
        block-diagonal in spin (and identical in both spin blocks), so it can be
        evaluated with or without the spin-orbit term and does NOT by itself
        require a spinor (2*nbv) Hamiltonian. The spinor representation is turned
        on only by spin-orbit coupling (SObool) or finite total magnetization
        (tot_magnetization != 0); an NL-only Hamiltonian is built on the smaller
        nbv x nbv block. If "NLbool" is left as None it defaults to "SObool",
        which reproduces the legacy behavior in which the non-local potential was
        only ever built when spin-orbit coupling was enabled.
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
        # Non-local potential switch, decoupled from spin-orbit. Defaults to
        # SObool when not given, preserving the legacy behavior where the
        # non-local potential rode along with the spin-orbit potential.
        self.NLbool = SObool if NLbool is None else NLbool
        self.cacheSO = cacheSO
        self.NN_locbool = NN_locbool
        self.model = model
        self.coupling = coupling   # fit the e-ph couplings? boolean
        self.fit_eff_masses = system.fit_eff_masses

        # Spin-polarized (spin-unrestricted) local potential. When
        # tot_magnetization != 0, the up and down spin channels feel different
        # learned local potentials: V_up = V0 + b, V_down = V0 - b, where V0 is
        # self.model and b is self.spinModel (a learned spin/exchange field).
        # `spinor` is the unified flag that controls the 2*nbv matrix sizing: it is
        # True whenever the two spin sectors must explicitly coexist in a single
        # Hamiltonian, i.e. whenever spin-orbit coupling (SObool) mixes the sectors,
        # or spin polarization (magBool) gives them different local potentials. It
        # is NOT turned on by the non-local potential: the NL potential is block-
        # diagonal in spin AND identical in both spin blocks, so an NL-only
        # Hamiltonian is built on the smaller nbv x nbv block, which already holds
        # every distinct spatial band exactly once. Its eigenvalues are returned
        # directly (no artificial spin-degeneracy doubling), so the spectrum matches
        # reference band structures that list each band once. Building NL without
        # spinors halves the matrix dimension (and ~4x the eigensolve cost) relative
        # to the redundant 2*nbv block-diagonal form.
        # SOC-specific physics stays gated on self.SObool; non-local physics is
        # gated on self.NLbool. The matrices that NL contributes to are sized to
        # match the Hamiltonian: 2*nbv when self.spinor, nbv otherwise.
        self.tot_magnetization = self.NNConfig.get('tot_magnetization', 0.0)
        self.magBool = (self.tot_magnetization != 0)
        self.spinor = self.SObool or self.magBool
        self.spinModel = spinModel
        if self.tot_magnetization == 0:
            # Spin polarization is OFF. The spectrum is NOT spin-doubled: each
            # distinct spatial band is returned ONCE ([e0,e1,e2,...]), so idxVB/idxCB
            # and the reference band structure must index the un-doubled spectrum.
            # With tot_magnetization != 0 the spectrum is spin-resolved and doubled
            # ([up0,dn0,up1,dn1,...]); mixing the two conventions shifts every band
            # index and "messes up" the band energies. Set tot_magnetization != 0
            # (e.g. 1) to run spin-polarized.
            print(f"WARNING (iSystem={iSystem}): tot_magnetization = 0 -> spin "
                  "polarization OFF. Bands are NOT spin-doubled; idxVB/idxCB and the "
                  "reference bands must index the un-doubled spectrum. Set "
                  "tot_magnetization != 0 to enable the spin-polarized (doubled) spectrum.")

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

        # Per-job tag embedded in the POSIX shared-memory segment names so that
        # segments leaked by a crashed job never collide with this job's (the
        # /dev/shm namespace is global per user). Populated by initAndCacheHams
        # with f"{os.getpid()}_{uuid.uuid4().hex}"; the worker-side reattach code
        # in calcEigValsAtK reconstructs the exact same names from this tag.
        self.shm_tag = None

        # The SO/NL matrices are cached grouped by atom TYPE (default) rather than
        # per atom. Atoms of the same type share the same PPparams prefactor, so
        # summing their (constant) projector matrices into one slot per type is
        # exactly equivalent to summing them in buildSOmat/buildNLmat (associativity
        # of the Hamiltonian sum), but stores nTypes matrices instead of nAtoms. For
        # supercells with many same-type atoms (e.g. graphene) this is a large RAM
        # saving in the SO/NL cache + shared memory. The per-atom loops that fill
        # these matrices still use each atom's true position (the structure factor
        # in init{SO,NL}mat_fast_oneKpt), so position dependence is fully preserved;
        # only the storage is collapsed by type.
        #
        # NOTE: this exactness relies on every atom in a group carrying IDENTICAL
        # PPparams prefactors (indices 5=SOC, 6=NL1, 7=NL2). If site-resolved
        # prefactors are ever needed, set 'group_by_type'=False to fall back to the
        # legacy one-group-per-atom storage (this is also how test_low_mem verifies
        # the grouped result is bit-for-bit identical to the per-atom result).
        #
        # "matGroupTypes[g]" is the atom type whose prefactor multiplies group g's
        # matrix; "atomToGroup[alpha]" maps an atom to the matrix slot it fills.
        self.group_by_type = bool(self.NNConfig.get('group_by_type', True))
        atomTypesList = list(self.system.atomTypes)
        if self.group_by_type:
            self.matGroupTypes = list(dict.fromkeys(atomTypesList))  # distinct types, first-appearance order
            self.atomToGroup = [self.matGroupTypes.index(t) for t in atomTypesList]
        else:
            self.matGroupTypes = atomTypesList                       # one group per atom (legacy)
            self.atomToGroup = list(range(len(atomTypesList)))
        self.nMatGroups = len(self.matGroupTypes)

        # Disk-backed cache (low_mem): when on, the per-k-point SO/NL matrices are
        # written to .npy files under './<mat_cache_dir>/' (labeled by the per-job
        # shm tag) and loaded one k-point at a time on demand, instead of being held
        # resident in POSIX shared memory for the whole run. Peak cache RAM drops
        # from the full k-point stack to a single k-point's matrices per worker, at
        # the cost of a disk read per k-point. The flag is named 'low_mem' for
        # backward compatibility; it now toggles this disk spill (type-grouping
        # above is unconditional).
        self.disk_cache = bool(self.NNConfig.get('low_mem', False))
        # Evaluate the fallback LAZILY: a plain dict default would eagerly index
        # self.NNConfig["resultsFolder"] and KeyError when it is absent (e.g. a
        # Hamiltonian built with the minimal default NNConfig and no results dir).
        self.mat_cache_dir = self.NNConfig.get('mat_cache_dir') or f'{self.NNConfig.get("resultsFolder", "")}mat_cache'

        # Partial eigensolver: only the lowest ~nBands eigenvalues of the
        # (2*nbv)-dim Hamiltonian are needed for the band-structure loss. When
        # enabled, the eigenvalue-only diag path uses a direct LAPACK subset
        # driver (utils.partial_eig.partial_eigvalsh) with a degeneracy-safe
        # custom autograd backward, instead of the full torch.linalg.eigvalsh.
        # Big win when nBands << 2*nbv. The coupling path (which needs
        # eigenVECTORS) ignores this flag and keeps using torch.linalg.eigh.
        self.partial_eig = bool(self.NNConfig.get('partial_eig', False))
        self.partial_eig_driver = self.NNConfig.get('partial_eig_driver', 'evr')

        # Detect whether any atom actually carries non-local potential
        # coefficients (PPparams indices 6 and 7). This guards the (otherwise
        # wasteful) construction of NL matrices that would be identically zero.
        self.checknl = False
        for atom in self.atomPPorder:
            if abs(self.PPparams[atom][6]) > 1e-8:
                self.checknl = True
                break
            elif abs(self.PPparams[atom][7]) > 1e-8:
                self.checknl = True
                break

        # The SO and NL potentials are cached independently: the SO matrices are
        # built whenever SObool is on, and the NL matrices whenever NLbool is on
        # (and there are actually non-local coefficients to include). This lets
        # the non-local potential be evaluated with or without spin-orbit.
        if SObool and cacheSO:
            print("Caching SO mats.", flush=True)
            sys.stdout.flush()
            self.SOmats = self.initSOmat_fast()
            self.SOmats_def = {}
        elif SObool and (not cacheSO) and (NNConfig['num_cores']==0):
            print("WARNING: Calculation requires SObool, but we are not cache-ing the SOmats. Without multiprocessing parallelization. This is not recommended. ")

        if self.NLbool and self.checknl and cacheSO:
            print("Caching NL mats.", flush=True)
            sys.stdout.flush()
            self.NLmats = self.initNLmat_fast()
            self.NLmats_def = {}
        elif self.NLbool and self.checknl and (not cacheSO) and (NNConfig['num_cores']==0):
            print("WARNING: Calculation requires NLbool, but we are not cache-ing the NLmats. Without multiprocessing parallelization. This is not recommended. ")


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
                if not self.spinor:
                    print("NOTE: spinor sector is off (no SOC, no magnetization; the non-local potential does NOT require spinors). idxVB and idxCB are zero-indexed band indices into the distinct (un-doubled) spectrum, without any 2x interleaving for spin. Please double check to ensure your inputs of idxVB and idxCB correspond to your intended bands. ")

            # The coupling SO and NL derivative matrices are needed whenever
            # spin-orbit OR the non-local potential contributes to the coupling.
            if self.SObool or (self.NLbool and self.checknl):
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

        # Build the deformed SO matrices only when spin-orbit is active, and the
        # deformed NL matrices only when the non-local potential is active. This
        # mirrors the independent SObool / NLbool gating of the undeformed path.
        so_mats = None
        if self.SObool:
            if cache_key not in self.SOmats_def:
                self.SOmats_def[cache_key] = self.initSOmat_fast(defbool=True, idxGap=kidx)
            so_mats = self.SOmats_def[cache_key]

        nl_mats = None
        if self.NLbool and self.checknl:
            if cache_key not in self.NLmats_def:
                self.NLmats_def[cache_key] = self.initNLmat_fast(defbool=True, idxGap=kidx)
            nl_mats = self.NLmats_def[cache_key]

        return so_mats, nl_mats


    def _init_parallel_mode(self, pool_flag, nkp):
        """
        Decide how the SO/NL matrix initialization parallelizes over k-points.

        Returns one of:
          "serial"  - single-threaded loop (num_cores==0 or only one k-point).
          "thread"  - OpenMP-style shared-memory threads (default when
                      num_cores>0). The per-k-point integrals are heavy
                      numpy/scipy that release the GIL, so threads parallelize
                      with no per-worker copy of the Hamiltonian and no pickling.
          "process" - legacy mp.Pool over k-points. Only used when the user
                      explicitly disables init_threads AND sets the
                      pool_initSO/pool_initNL flag. Higher memory (pickles the
                      whole ham to each worker and copies results back).
        """
        num_cores = self.NNConfig.get("num_cores", 0)
        if nkp <= 1:
            return "serial"
        # Thread mode is independent of num_cores: SO/NL init is a one-shot phase
        # that owns the whole node before any k-point worker pool exists, so it
        # threads over k-points (GIL-released numpy/scipy) even for a serial
        # (num_cores==0) training run. See utils.threads.init_thread_count.
        if self.NNConfig.get("init_threads", True):
            return "thread"
        if num_cores > 0 and pool_flag != 0:
            return "process"
        return "serial"


    def buildHtot(self, kidx, preComp_SOmats_kidx=None, preComp_NLmats_kidx=None, requires_grad=True, precomp_Vloc=None):
        """
        Build the total Hamiltonian for a given kpt, specified by its kidx.
        preComp_SOmats_kidx and preComp_NLmats_kidx are the pre-computed
        SO and NL matrices (actual matrices) at the certain kidx

        precomp_Vloc: the local-potential matrix built ONCE for this epoch and
        reused across all k-points. Vloc is k-independent (it depends only on the
        G-vector differences and atom positions, not on k), so when this is
        supplied we just add it to the kinetic term instead of rebuilding it per
        k. The SAME grad-carrying tensor is shared by every k, so a downstream
        backward correctly accumulates the model gradient through the shared
        subgraph (see calcBandStruct_withGrad / trainIter_separateKptGrad).
        """
        nbv = self.basis.shape[0]
        # kinetic energy (spin-diagonal: identical in both spin blocks).
        # Vectorized over the basis to avoid a Python loop over (2*)nbv per kpt
        # in the training inner loop.
        with PROF.time("Htot_kinetic"):
            if self.spinor:
                Htot = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)
            else:
                Htot = torch.zeros([nbv, nbv], dtype=torch.complex128)
            kvec = self.basis + self.system.kpts[kidx]
            kin = (HBAR**2 / (2*MASS)) * (kvec * kvec).sum(dim=1)   # real, length nbv
            if self.spinor:
                kin = kin.repeat(2)                                 # up block then dn block
            diag_idx = torch.arange(kin.shape[0])
            Htot[diag_idx, diag_idx] = kin.to(Htot.dtype)

        # local potential
        with PROF.time("Htot_Vloc"):
            if precomp_Vloc is not None:
                # Reuse the epoch's k-independent Vloc; just add it onto kinetic.
                Htot = Htot + precomp_Vloc
            else:
                Htot = self.buildVlocMat(addMat=Htot)
            if not requires_grad:
                Htot = Htot.detach()

        if self.SObool:
            with PROF.time("Htot_SO"):
                Htot = self.buildSOmat(kidx, preComp_SOmats_kidx, addMat=Htot)

        # The non-local potential is added independently of spin-orbit, gated on
        # NLbool (and the presence of non-local coefficients).
        if self.NLbool and self.checknl:
            with PROF.time("Htot_NL"):
                Htot = self.buildNLmat(kidx, preComp_NLmats_kidx, addMat=Htot)

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
        # Deform the relevant quantities, then restore them. Use OUT-OF-PLACE
        # reassignment (not in-place *=): self.system.atomPos is a leaf that
        # requires grad, and an in-place op on such a leaf raises. Originals are
        # stashed and restored in the finally block.
        store_basis = self.basis
        store_kpts = self.system.kpts
        store_cell = self.system.unitCellVectors
        store_atomPos = self.system.atomPos
        need_def_mats = self.SObool or (self.NLbool and self.checknl)
        store_SOmats, store_NLmats = self.SOmats, self.NLmats
        try:
            self.basis = self.basis * (self.system.scale / self.defscale)
            self.system.kpts = self.system.kpts * (self.system.scale / self.defscale)
            self.system.unitCellVectors = self.system.unitCellVectors * (self.defscale / self.system.scale)
            self.system.atomPos = self.system.atomPos * (self.defscale / self.system.scale)

            nbv = self.basis.shape[0]
            if self.spinor:
                Htot = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)
            else:
                Htot = torch.zeros([nbv, nbv], dtype=torch.complex128)

            # kinetic energy (spin-diagonal: identical in both spin blocks).
            # Vectorized over the basis to avoid a Python loop over (2*)nbv.
            kvec = self.basis + self.system.kpts[kidx]
            kin = (HBAR**2 / (2*MASS)) * (kvec * kvec).sum(dim=1)   # real, length nbv
            if self.spinor:
                kin = kin.repeat(2)                                 # up block then dn block
            diag_idx = torch.arange(kin.shape[0])
            Htot[diag_idx, diag_idx] = kin.to(Htot.dtype)

            # local potential (carries the spin split when magnetic)
            Htot = self.buildVlocMat(addMat=Htot)

            # The SO and NL terms are deformed/added independently. We need the
            # deformed cached matrices whenever either spin-orbit or the non-local
            # potential is active.
            if need_def_mats:
                self.SOmats, self.NLmats = self._get_deformed_cached_mats(kidx, scale)
                # the below calls are kidx=0 because they index into the SOmats and
                # NLmats arrays, for which there is only a single kpoint. There are no
                # calls self.system.kpts[kidx] in these functions, so it does not
                # cause any issues.
                if self.SObool:
                    Htot = self.buildSOmat(0, addMat=Htot)
                if self.NLbool and self.checknl:
                    Htot = self.buildNLmat(0, addMat=Htot)
        finally:
            # Restore the non-deformed values (even if building H raised).
            self.basis = store_basis
            self.system.kpts = store_kpts
            self.system.unitCellVectors = store_cell
            self.system.atomPos = store_atomPos
            if need_def_mats:
                self.SOmats = store_SOmats
                self.NLmats = store_NLmats

        return Htot


    def buildHtot_def_NEW(self, kidx, scale=1.01, verbosity=2, requires_grad=True):
        """
        Just like the function above, but with the added flexibility of
        calculating at various k-points. The local potential (buildVlocMat) carries
        the spin split when tot_magnetization != 0, so this builds the correct
        spin-polarized deformed Hamiltonian as well.
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
        # Deform the relevant quantities, then restore them after building H.
        # Use OUT-OF-PLACE reassignment (not in-place *=): self.system.atomPos is a
        # leaf that requires grad, and an in-place op on such a leaf raises. The
        # originals are stashed and restored in the finally block below.
        store_basis = self.basis
        store_kpts = self.system.kpts
        store_cell = self.system.unitCellVectors
        store_atomPos = self.system.atomPos
        need_def_mats = self.SObool or (self.NLbool and self.checknl)
        store_SOmats, store_NLmats = self.SOmats, self.NLmats
        try:
            self.basis = self.basis * (self.system.scale / self.defscale)
            self.system.kpts = self.system.kpts * (self.system.scale / self.defscale)
            self.system.unitCellVectors = self.system.unitCellVectors * (self.defscale / self.system.scale)
            self.system.atomPos = self.system.atomPos * (self.defscale / self.system.scale)

            nbv = self.basis.shape[0]
            if self.spinor:
                Htot = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)
            else:
                Htot = torch.zeros([nbv, nbv], dtype=torch.complex128)

            # kinetic energy (spin-diagonal: identical in both spin blocks).
            # Vectorized over the basis to avoid a Python loop over (2*)nbv.
            kvec = self.basis + self.system.kpts[kidx]
            kin = (HBAR**2 / (2*MASS)) * (kvec * kvec).sum(dim=1)   # real, length nbv
            if self.spinor:
                kin = kin.repeat(2)                                 # up block then dn block
            diag_idx = torch.arange(kin.shape[0])
            Htot[diag_idx, diag_idx] = kin.to(Htot.dtype)

            # local potential (carries the spin split when magnetic)
            Htot = self.buildVlocMat(addMat=Htot)

            # The SO and NL terms are deformed/added independently. We need the
            # deformed cached matrices whenever either spin-orbit or the non-local
            # potential is active.
            if need_def_mats:
                self.SOmats, self.NLmats = self._get_deformed_cached_mats(kidx, scale)
                # the below calls are kidx=0 because they index into the SOmats and
                # NLmats arrays, for which there is only a single kpoint. There are no
                # calls self.system.kpts[kidx] in these functions, so it does not
                # cause any issues.
                if self.SObool:
                    Htot = self.buildSOmat(0, addMat=Htot)
                if self.NLbool and self.checknl:
                    Htot = self.buildNLmat(0, addMat=Htot)
        finally:
            # Restore the non-deformed values (even if building H raised).
            self.basis = store_basis
            self.system.kpts = store_kpts
            self.system.unitCellVectors = store_cell
            self.system.atomPos = store_atomPos
            if need_def_mats:
                self.SOmats = store_SOmats
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
            return self.model(q)

        def compute_b():
            # spin/exchange field b(q); same form-factor shape as compute_atomFF
            return self.spinModel(q)

        if addMat is not None:
            if self.spinor:
                assert addMat.shape[0] == 2*nbv
                assert addMat.shape[1] == 2*nbv
            Vmat = addMat
        else:
            if self.spinor:
                Vmat = torch.zeros([2*nbv, 2*nbv], dtype=torch.complex128)
            else:
                Vmat = torch.zeros([nbv, nbv])

        # The NN form factors depend only on q = |G_i - G_j|, not on the atom
        # (the model output carries one column per atom *type*). Evaluate the
        # model(s) ONCE here rather than once per atom in the loop below. This
        # avoids rebuilding NAtoms identical copies of the autograd subgraph,
        # which would otherwise make the compute-graph memory scale linearly
        # with the number of atoms. Per-atom code just slices the right column.
        if self.NN_locbool:
            if self.NNConfig['checkpoint'] == 0:
                atomFF_full = self.model(q)
            elif self.NNConfig['checkpoint'] == 1:
                atomFF_full = checkpoint(compute_atomFF, use_reentrant=False)
        if self.magBool:
            if self.NNConfig['checkpoint'] == 0:
                bff_full = self.spinModel(q)
            elif self.NNConfig['checkpoint'] == 1:
                bff_full = checkpoint(compute_b, use_reentrant=False)

        # The local-potential matrix element is
        #     V_{ij} = sum_alpha atomFF_{type(alpha)}(|G_i-G_j|) * sfact_alpha,
        #     sfact_alpha = exp(+i (G_i-G_j).tau_alpha) / V_cell.
        # The structure factor sfact_alpha is built from atom positions and
        # G-vectors only -- it carries NO autograd graph -- while atomFF carries
        # the (expensive) model graph. When atomFF depends only on the atom TYPE
        # (the usual case), the sum over atoms of one type factorizes:
        #     sum_{alpha in t} atomFF_t * sfact_alpha = atomFF_t * (sum_alpha sfact_alpha).
        # So we accumulate the structure factors per type FIRST (grad-free) and do
        # a single grad-tracked multiply per TYPE. The retained backward graph is
        # then O(N_types) instead of O(N_atoms), which is what made memory grow
        # ~linearly per atom. This is exact (the H sum is associative and same-type
        # atoms share atomFF), so it reproduces the per-atom result to round-off.
        #
        # CONDITION: the factorization holds only when atomFF is type-shared. The
        # local-environment (LSD) correction breaks that -- it adds a term that
        # depends on each atom's descriptor N_alpha -- so when local_env_corr is on
        # we fall back to the exact per-atom loop (unchanged behavior).
        gdiff_norm = torch.norm(gdiff, dim=2)
        invV = 1.0 / self.system.getCellVolume()

        def type_index(atomType):
            idx = np.where(atomType == self.atomPPorder)[0]
            if len(idx) != 1:
                raise ValueError("Type of atoms in PP. ")
            return idx[0]

        def base_atomFF(atomType, thisAtomIndex):
            # Spin-independent local form factor V0(|G_i-G_j|) for this atom type,
            # including the long-range correction. Depends only on the type, so it
            # is computed once per type (NN path slices the precomputed column).
            if self.NN_locbool:
                atomFF = atomFF_full[:, thisAtomIndex].view(nbv, nbv)
                lr_coeff = self.PPparams[atomType][4]
                atomFF = atomFF + long_range_correction(gdiff_norm, self.LRgamma, lr_coeff)
            else:
                atomFF = pot_funcLR(gdiff_norm, self.PPparams[atomType], self.LRgamma)
            return atomFF

        def add_contribution(atomFF, sfact, thisAtomIndex):
            # Apply the spin/exchange split (when magnetic) and add atomFF * sfact
            # into Vmat. `sfact` is complex (nbv,nbv): a per-type summed structure
            # factor on the grouped path, or a single-atom one on the LSD path.
            nonlocal Vmat
            if self.magBool:
                # Slice the per-type column from spinModel output computed once above.
                bff = bff_full[:, thisAtomIndex].view(nbv, nbv)
                atomFF_up = atomFF + bff
                atomFF_dn = atomFF - bff
            else:
                atomFF_up = atomFF
                atomFF_dn = atomFF

            if self.spinor:
                # local potential is spin-diagonal --> block diagonal; the up and
                # down blocks carry V_up and V_down respectively.
                Vmat[:nbv, :nbv] = Vmat[:nbv, :nbv] + atomFF_up * sfact
                Vmat[nbv:, nbv:] = Vmat[nbv:, nbv:] + atomFF_dn * sfact
            else:
                Vmat = Vmat + atomFF * sfact

        if self.NNConfig["local_env_corr"]:
            # Per-atom path: the LSD correction depends on each atom's local
            # environment (N_alpha), so atomFF is NOT shared across same-type atoms
            # and the structure-factor sum cannot be collapsed by type.
            for alpha in range(self.system.getNAtoms()):
                atomType = self.system.atomTypes[alpha]
                thisAtomIndex = type_index(atomType)
                gdiffDotTau = torch.sum(gdiff * self.system.atomPos[alpha], axis=2)
                sfact = torch.complex(invV * torch.cos(gdiffDotTau), invV * torch.sin(gdiffDotTau))

                atomFF = base_atomFF(atomType, thisAtomIndex)

                descriptors = self.system.env_descriptors[atomType]
                indx_alpha = torch.where(self.system.atom_indices[atomType] == alpha)[0].squeeze(0)
                N_alpha = descriptors[indx_alpha, :]
                N_alphas = N_alpha.repeat(q.shape[0], 1)
                x_input = torch.cat([N_alphas, q], dim=1)
                atomFF = atomFF + self.LSDmodels[atomType](x_input).view(nbv, nbv)

                add_contribution(atomFF, sfact, thisAtomIndex)
        else:
            # Per-type path: accumulate the (grad-free) structure factors over all
            # atoms of each type, then do one grad-tracked multiply per type.
            type_sfact_re = {}
            type_sfact_im = {}
            for alpha in range(self.system.getNAtoms()):
                atomType = self.system.atomTypes[alpha]
                gdiffDotTau = torch.sum(gdiff * self.system.atomPos[alpha], axis=2)
                s_re = invV * torch.cos(gdiffDotTau)
                s_im = invV * torch.sin(gdiffDotTau)
                if atomType in type_sfact_re:
                    type_sfact_re[atomType] = type_sfact_re[atomType] + s_re
                    type_sfact_im[atomType] = type_sfact_im[atomType] + s_im
                else:
                    type_sfact_re[atomType] = s_re
                    type_sfact_im[atomType] = s_im

            for atomType in type_sfact_re:
                thisAtomIndex = type_index(atomType)
                sfact = torch.complex(type_sfact_re[atomType], type_sfact_im[atomType])
                atomFF = base_atomFF(atomType, thisAtomIndex)
                add_contribution(atomFF, sfact, thisAtomIndex)

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
        mat = np.zeros((self.nMatGroups, 2*nbv, 2*nbv), dtype=np.complex128)
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
        
        # Single result buffer that every k-point writes into directly. This is
        # the only large allocation: workers fill disjoint k-point slices in
        # place, so we never hold a second copy of the matrices.
        SOmats_4d = np.zeros((nkp, self.nMatGroups, 2*nbv, 2*nbv), dtype=np.complex128)

        mode = self._init_parallel_mode(self.NNConfig.get("pool_initSO", 0), nkp)
        with PROF.time("init_SO"):
            if mode == "thread":
                # OpenMP-style shared-memory parallelism over k-points. The SO
                # integral is heavy numpy/scipy (erf, exp, tensordot) that releases
                # the GIL, so threads give real parallelism with no per-worker copy
                # of the Hamiltonian and no IPC/pickling overhead.
                n_workers = min(init_thread_count(self.NNConfig.get("num_threads", 0)), nkp)
                print(f"Initializing SO mats with {n_workers} shared-memory threads\n", flush=True)
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    list(ex.map(
                        lambda kidx: self.initSOmat_fast_oneKpt(kidx, SOmats_4d[kidx], SOwidth, defbool, idxGap),
                        range(nkp)))
            elif mode == "process":
                print(f"Initializing SO mats with {self.NNConfig['num_cores']} processes\n", flush=True)
                args_list = [(kidx, SOwidth, defbool, idxGap) for kidx in range(nkp)]
                with mp.Pool(self.NNConfig['num_cores']) as pool:
                    results = pool.map(self._wrap_initSOmat, args_list)
                for kidx, mat in results:
                    SOmats_4d[kidx] = mat
            else:
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
            SOmats_oneKpt_toFill[self.atomToGroup[alpha], :nbv, :nbv] += real_part + 1j * im_part

            # dn dn
            # -i * gcp dot S_dn,dn is pure imag: i/2 * (gcp.z)
            real_part = prefactor * isum * -0.5 * gcross[:,:, 2] * sfact_im
            im_part = prefactor * isum * 0.5 * gcross[:,:, 2] * sfact_re
            SOmats_oneKpt_toFill[self.atomToGroup[alpha], nbv:, nbv:] += real_part + 1j * im_part

            # up dn
            # -i * gcp dot S_up,dn is: -i/2 * (gcp.x) - 1/2 * (gcp.y)
            real_part = prefactor * isum * (0.5 * gcross[:,:, 0] * sfact_im -0.5 * gcross[:,:, 1] * sfact_re)
            im_part = prefactor * isum * (-0.5 * gcross[:,:, 0] * sfact_re -0.5 * gcross[:,:, 1] * sfact_im)
            SOmats_oneKpt_toFill[self.atomToGroup[alpha], :nbv, nbv:] += real_part + 1j * im_part

            # dn up
            # -i * gcp dot S_dn,up is: -i/2 * (gcp.x) + 1/2 * (gcp.y)
            real_part = prefactor * isum * (0.5 * gcross[:,:, 0] * sfact_im + 0.5 * gcross[:,:, 1] * sfact_re)
            im_part = prefactor * isum * (-0.5 * gcross[:,:, 0] * sfact_re + 0.5 * gcross[:,:, 1] * sfact_im)
            SOmats_oneKpt_toFill[self.atomToGroup[alpha], nbv:, :nbv] += real_part + 1j * im_part
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
        
        # NL matrices match the Hamiltonian dimension: 2*nbv with spinors, nbv
        # otherwise (block-diagonal, identical in both spin blocks).
        ndim = 2*nbv if self.spinor else nbv
        NLmats = np.empty([nkp, self.system.getNAtoms(), 2], dtype=object)
        for id1 in range(nkp):
            for id2 in range(self.system.getNAtoms()):
                for id3 in [0,1]:
                    NLmats[id1,id2,id3] = torch.zeros([ndim, ndim], dtype=torch.complex128)

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
                    
                        # This potential is block diagonal on spin AND identical
                        # in both spin blocks. Fill the dn-dn block only when the
                        # Hamiltonian is a spinor (2*nbv); otherwise the single
                        # up-up block is all that is needed.
                        # up up, 1st integral
                        real_part = prefactor * isum1 * gdot * sfact_re
                        im_part = prefactor * isum1 * gdot * sfact_im
                        NLmats[kidx,alpha,0][i,j] = torch.complex(real_part, im_part)
                        # 2nd integral
                        real_part = prefactor * isum2 * gdot * sfact_re
                        im_part = prefactor * isum2 * gdot * sfact_im
                        NLmats[kidx,alpha,1][i,j] = torch.complex(real_part, im_part)

                        if self.spinor:
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
        # Allocate a local matrix for this k-point. Dimension matches the
        # Hamiltonian: 2*nbv with spinors, nbv otherwise (NL is block-diagonal
        # and identical in both spin blocks, so the single block suffices).
        ndim = 2*nbv if self.spinor else nbv
        mat = np.zeros((self.nMatGroups, 2, ndim, ndim), dtype=np.complex128)
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
        
        # NL matrices are sized to match the Hamiltonian: 2*nbv with spinors,
        # nbv otherwise (the NL potential is block-diagonal and identical across
        # spin blocks, so the single block is sufficient when spinors are off).
        ndim = 2*nbv if self.spinor else nbv

        # Single result buffer that every k-point writes into directly. This is
        # the only large allocation: workers fill disjoint k-point slices in
        # place, so we never hold a second copy of the matrices.
        NLmats_5d = np.zeros((nkp, self.nMatGroups, 2, ndim, ndim), dtype=np.complex128)

        mode = self._init_parallel_mode(self.NNConfig.get("pool_initNL", 0), nkp)
        with PROF.time("init_NL"):
            if mode == "thread":
                # OpenMP-style shared-memory parallelism over k-points. The non-local
                # integral is dominated by scipy quad_vec / numpy work that releases
                # the GIL, so threads give real parallelism with no per-worker copy of
                # the Hamiltonian and no IPC/pickling overhead (low memory).
                n_workers = min(init_thread_count(self.NNConfig.get("num_threads", 0)), nkp)
                print(f"Initializing NL mats with {n_workers} shared-memory threads\n", flush=True)
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    list(ex.map(
                        lambda kidx: self.initNLmat_fast_oneKpt(kidx, NLmats_5d[kidx], width1, width2, shift, defbool, idxGap),
                        range(nkp)))
            elif mode == "process":
                print(f"Initializing NL mats with {self.NNConfig['num_cores']} processes\n", flush=True)
                args_list = [(kidx, width1, width2, shift, defbool, idxGap) for kidx in range(nkp)]
                with mp.Pool(self.NNConfig['num_cores']) as pool:
                    results = pool.map(self._wrap_initNLmat, args_list)
                for kidx, mat in results:
                    NLmats_5d[kidx] = mat
            else:
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
            
        
            # This potential is block diagonal on spin AND identical in both spin
            # blocks. The destination array is sized 2*nbv when self.spinor (fill
            # both the up-up and dn-dn blocks) and nbv otherwise (fill the single
            # block; its eigenvalues are the distinct spatial bands, returned
            # directly without spin doubling in calcEigValsAtK).
            # up up, 1st integral
            real_part = prefactor * isum1 * gdot * sfact_re
            im_part = prefactor * isum1 * gdot * sfact_im
            NLmats_oneKpt_toFill[self.atomToGroup[alpha],0, :nbv, :nbv] += real_part + 1j* im_part
            # 2nd integral
            real_part = prefactor * isum2 * gdot * sfact_re
            im_part = prefactor * isum2 * gdot * sfact_im
            NLmats_oneKpt_toFill[self.atomToGroup[alpha],1, :nbv, :nbv] += real_part + 1j * im_part

            if self.spinor:
                # dn dn, 1st integral
                real_part = prefactor * isum1 * gdot * sfact_re
                im_part = prefactor * isum1 * gdot * sfact_im
                NLmats_oneKpt_toFill[self.atomToGroup[alpha],0, nbv:, nbv:] += real_part + 1j * im_part
                # 2nd integral
                real_part = prefactor * isum2 * gdot * sfact_re
                im_part = prefactor * isum2 * gdot * sfact_im
                NLmats_oneKpt_toFill[self.atomToGroup[alpha],1, nbv:, nbv:] += real_part + 1j * im_part
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
                SOmats_kidx = np.zeros((self.nMatGroups, 2*self.basis.shape[0], 2*self.basis.shape[0]), dtype=np.complex128)
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
        
        for g in range(self.nMatGroups):
            if isinstance(SOmats_kidx[g], torch.Tensor):
                tmp = SOmats_kidx[g]
            else:
                tmp = torch.tensor(SOmats_kidx[g])

            SOmatf = SOmatf + tmp * self.PPparams[self.matGroupTypes[g]][5]

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
                ndim = 2*self.basis.shape[0] if self.spinor else self.basis.shape[0]
                NLmats_kidx = np.zeros((self.nMatGroups, 2, ndim, ndim), dtype=np.complex128)
                self.initNLmat_fast_oneKpt(kidx, NLmats_kidx)
            else:
                NLmats_kidx = self.NLmats[kidx]
        else:
            NLmats_kidx = preComp_NLmats_kidx

        nbv = self.basis.shape[0]
        # The NL matrix matches the Hamiltonian dimension: 2*nbv with spinors,
        # nbv otherwise.
        ndim = 2*nbv if self.spinor else nbv
        if addMat is not None:
            assert addMat.shape[0] == ndim
            assert addMat.shape[1] == ndim
            NLmatf = addMat
        else:
            NLmatf = torch.zeros([ndim, ndim], dtype=torch.complex128)
        
        for g in range(self.nMatGroups):
            if isinstance(NLmats_kidx[g,0], torch.Tensor):
                tmp1 = NLmats_kidx[g,0]
            else:
                tmp1 = torch.tensor(NLmats_kidx[g,0])
            if isinstance(NLmats_kidx[g,1], torch.Tensor):
                tmp2 = NLmats_kidx[g,1]
            else:
                tmp2 = torch.tensor(NLmats_kidx[g,1])

            NLmatf = (NLmatf + tmp1 * self.PPparams[self.matGroupTypes[g]][6]
                             + tmp2 * self.PPparams[self.matGroupTypes[g]][7] )

        return NLmatf


    def _mat_cache_path(self, kind, kidx):
        """
        Filesystem path for a disk-cached SO/NL matrix file. 'kind' is "SOmats" or
        "NLmats". Files are labeled by the per-job shm tag (so concurrent jobs in
        the same working directory never collide) and live under ./<mat_cache_dir>/.
        Mirrors the in-memory shared-memory segment names exactly, with a .npy
        suffix. Used by both the build side (initAndCacheHams) and the per-k-point
        load in calcEigValsAtK.
        """
        return os.path.join(self.mat_cache_dir,
                            f"{kind}_{self.shm_tag}_{self.iSystem}_{kidx}.npy")

    def buildHtot_cached(self, kidx, cachedMats_info=None, requires_grad=True, precomp_Vloc=None):
        """Return the full dense Htot at k-index `kidx`, loading the SO/NL matrices
        from the shared-memory / disk cache exactly as calcEigValsAtK does. Provided
        for the symmetry-block path (utils.symmetry), which needs the DENSE H (not
        just eigenvalues) to form B^dag H B. Isolated from calcEigValsAtK so the
        production eigenvalue path is untouched; the shm-load block below mirrors
        calcEigValsAtK's loader and must be kept in sync with it."""
        if cachedMats_info is None:
            preComp_SOmats_kidx = None
            preComp_NLmats_kidx = None
        else:
            if self.SObool:
                if self.disk_cache:
                    preComp_SOmats_kidx = np.load(self._mat_cache_path("SOmats", kidx))
                else:
                    shm_SOmats = shared_memory.SharedMemory(name=f"SOmats_{self.shm_tag}_{self.iSystem}_{kidx}")
                    preComp_SOmats_kidx = np.ndarray(cachedMats_info[f"SO_{self.iSystem}_{kidx}"]['shape'], dtype=cachedMats_info[f"SO_{self.iSystem}_{kidx}"]['dtype'], buffer=shm_SOmats.buf)
            else:
                preComp_SOmats_kidx = None
            if self.NLbool and self.checknl:
                if self.disk_cache:
                    preComp_NLmats_kidx = np.load(self._mat_cache_path("NLmats", kidx))
                else:
                    shm_NLmats = shared_memory.SharedMemory(name=f"NLmats_{self.shm_tag}_{self.iSystem}_{kidx}")
                    preComp_NLmats_kidx = np.ndarray(cachedMats_info[f"NL_{self.iSystem}_{kidx}"]['shape'], dtype=cachedMats_info[f"NL_{self.iSystem}_{kidx}"]['dtype'], buffer=shm_NLmats.buf)
            else:
                preComp_NLmats_kidx = None
        H = self.buildHtot(kidx, preComp_SOmats_kidx, preComp_NLmats_kidx, requires_grad, precomp_Vloc=precomp_Vloc)
        return H if requires_grad else H.detach()

    def calcEigValsAtK(self, kidx, cachedMats_info=None, requires_grad=True, verbosity=0, def_H=False, def_scale=0.01, precomp_Vloc=None):
        '''
        This function builds the Htot at a certain kpoint that is given as the input, 
        digonalizes the Htot, and obtains the eigenvalues at this kpoint. 
        '''

        nbands = self.system.nBands
        eigVals = torch.zeros(nbands)

        if (cachedMats_info is None):
            # No cached matrices in shared memory. buildSOmat / buildNLmat handle
            # the SObool==False / NLbool==False cases by simply not being called.
            preComp_SOmats_kidx = None
            preComp_NLmats_kidx = None
        elif (cachedMats_info is not None):
            # The SO and NL matrices live in shared memory independently: load the
            # SO matrices only when spin-orbit is active, and the NL matrices only
            # when the non-local potential is active.
            with PROF.time("shm_load"):
                if self.SObool:
                    if self.disk_cache:
                        # Load just this k-point's SO matrices from disk (one .npy
                        # per k-point); they leave RAM again when this scope exits.
                        preComp_SOmats_kidx = np.load(self._mat_cache_path("SOmats", kidx))
                    else:
                        shm_SOmats = shared_memory.SharedMemory(name=f"SOmats_{self.shm_tag}_{self.iSystem}_{kidx}")
                        preComp_SOmats_kidx = np.ndarray(cachedMats_info[f"SO_{self.iSystem}_{kidx}"]['shape'], dtype=cachedMats_info[f"SO_{self.iSystem}_{kidx}"]['dtype'], buffer=shm_SOmats.buf)
                else:
                    preComp_SOmats_kidx = None
                if self.NLbool and self.checknl:
                    if self.disk_cache:
                        preComp_NLmats_kidx = np.load(self._mat_cache_path("NLmats", kidx))
                    else:
                        shm_NLmats = shared_memory.SharedMemory(name=f"NLmats_{self.shm_tag}_{self.iSystem}_{kidx}")
                        preComp_NLmats_kidx = np.ndarray(cachedMats_info[f"NL_{self.iSystem}_{kidx}"]['shape'], dtype=cachedMats_info[f"NL_{self.iSystem}_{kidx}"]['dtype'], buffer=shm_NLmats.buf)
                else:
                    preComp_NLmats_kidx = None
        else:
            raise ValueError("Error in calcEigValsAtK. ")

        # buildHtot is profiled internally (Htot_kinetic/Vloc/SO/NL); no wrapper
        # timer here so the leaf sections partition the inner loop cleanly.
        if not def_H:
            H = self.buildHtot(kidx, preComp_SOmats_kidx, preComp_NLmats_kidx, requires_grad, precomp_Vloc=precomp_Vloc)
        else:
            H = self.buildHtot_def_NEW(kidx, scale=def_scale, requires_grad=requires_grad)

        if not requires_grad:
            H = H.detach()

        if not self.coupling:
            # Number of lowest eigenvalues actually needed: every band index that
            # the reorder below (bandOrderMatrix) can reference, at least nBands.
            # For the default bandOrderMatrix = arange(nBands) this is just nBands.
            # The partial eigensolver computes only this many; the full eigvalsh
            # path ignores it.
            need = max(nbands, int(self.system.bandOrderMatrix[kidx].max()) + 1)
            if self.magBool and not self.SObool:
                # Spin-polarized, no SOC: H is block-diagonal in spin, with the
                # up block H[:nbv,:nbv] carrying V_up = V0 + b and the down block
                # H[nbv:,nbv:] carrying V_down = V0 - b (there is no SO/NL term to
                # couple the blocks). Diagonalize each spin channel SEPARATELY so
                # that each channel's eigenvalues are sorted ascending WITHIN that
                # channel. A single eigvalsh on the full 2*nbv matrix would merge
                # and globally sort both channels together, which swaps band
                # identity wherever an up band crosses a down band (the bug where
                # BS_up[N] ends up equal to BS_down[N-1]).
                nbv = self.basis.shape[0]
                with PROF.time("diag"):
                    if self.partial_eig:
                        # Interleaving [up0,dn0,up1,dn1,...] and keeping the lowest
                        # `need` entries needs the lowest ceil(need/2) from each
                        # spin block.
                        kblk = min((need + 1) // 2, nbv)
                        e_up = partial_eigvalsh(H[:nbv, :nbv], kblk, driver=self.partial_eig_driver) * AUTOEV
                        e_dn = partial_eigvalsh(H[nbv:, nbv:], kblk, driver=self.partial_eig_driver) * AUTOEV
                    else:
                        e_up = torch.linalg.eigvalsh(H[:nbv, :nbv]) * AUTOEV
                        e_dn = torch.linalg.eigvalsh(H[nbv:, nbv:]) * AUTOEV
                # Interleave the channels: [up0, dn0, up1, dn1, ...]. Even output
                # columns are spin-up, odd columns are spin-down. This is the
                # spin-RESOLVED convention (same as the SObool path): each spatial
                # band appears once per spin channel. At the start of training
                # b(q)=0, so e_up==e_dn and the spectrum is exactly the unpolarized
                # spectrum with every band doubled, [e0,e0,e1,e1,...]. (The
                # non-spin-polarized path returns each band ONCE, [e0,e1,e2,...];
                # to compare the two, spin-double the unpolarized spectrum.)
                energiesEV = torch.stack([e_up, e_dn], dim=1).reshape(-1)
            else:
                with PROF.time("diag"):
                    if self.partial_eig:
                        k = min(need, H.shape[0])
                        energies = partial_eigvalsh(H, k, driver=self.partial_eig_driver)
                    else:
                        energies = torch.linalg.eigvalsh(H)
                energiesEV = energies * AUTOEV

            # reorder the energies according to the manual input in self.system.bandOrderMatrix
            with PROF.time("band_reorder"):
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
            with PROF.time("diag"):
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

        # No artificial spin-degeneracy doubling of the spectrum. When spinors are
        # off, the local + non-local potential is identical in both spin channels,
        # so the nbv x nbv Hamiltonian already contains every DISTINCT spatial band
        # exactly once. We return those bands directly (the lowest nBands, selected
        # by bandOrderMatrix), so that nBands counts distinct bands and the output
        # matches reference band structures that list each band once (NOT spin-
        # doubled). When spinors are on (SObool or magBool) the 2*nbv eigensolve
        # already yields the full spin-resolved spectrum. In neither case do we pad.
        #
        # (Legacy behavior applied energiesEV.repeat_interleave(2) on the non-spinor
        # path, padding the spectrum to a doubly-degenerate [e0,e0,e1,e1,...] form.
        # That has been removed: it assumed spin-doubled reference data, whereas the
        # reference band structures here list each band once.)
        eigVals[:] = energiesEV[:nbands]

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
        # Vloc is k-INDEPENDENT: build it ONCE (one model forward over the q-grid +
        # one structure-factor pass) and reuse the SAME grad-carrying tensor for
        # every k-point. The caller does a SINGLE backward over the whole band
        # structure, so autograd correctly accumulates the model gradient through
        # this shared subgraph. Previously buildVlocMat ran once per k-point.
        precomp_Vloc = self.buildVlocMat()
        for kidx in range(nkpt):
            eigValsAtK = self.calcEigValsAtK(kidx, cachedMats_info, requires_grad=True, precomp_Vloc=precomp_Vloc)
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
            # Pin each worker to its share of the node's lin.alg threads so the
            # eigensolve is multi-threaded without num_cores*threads exceeding
            # the budget (see utils.threads). Use a spawn context (not the default
            # fork): the parent has a live OpenMP/MKL threadpool, and forking that
            # state deadlocks. The training path spawns for the same reason.
            blas_threads = self.NNConfig.get('blas_threads_per_worker', 1)
            ctx = mp.get_context("spawn")
            with ctx.Pool(self.NNConfig['num_cores'], initializer=pool_worker_init,
                          initargs=(blas_threads,)) as pool:
                eigValsList = pool.starmap(self.calcEigValsAtK, args_list)
            bandStruct = torch.stack(eigValsList)
        return bandStruct


    def calcDefPots(self, cachedMats_info=None, requires_grad=True, verbosity=2):
        """Compute the deformation potential for every entry in the target file.

        Consumes the table loaded by System.setExpDefPot (version='v2'), stored on
        self.system as:
            defPotInfo : (nEntries, 7) array, columns
                         [kidx_VB, bidx_VB, kidx_CB, bidx_CB, latConst_ratio,
                          defPot_gap, weight]   (band/k indices 0-based)
            defPotSpin : (nEntries, 2) int array [spin_VB, spin_CB] (0=up, 1=down),
                         or None when the input file had only 7 columns.

        For each entry, the CB-VB gap is evaluated at the relaxed cell and at a cell
        scaled by latConst_ratio, and the deformation potential is
            (gap_org - gap_def)/2 * (1 + s^3)/(1 - s^3),  s = latConst_ratio.

        Spin handling: in a spin-polarized run without SOC (magBool and not SObool)
        the eigenvalues from calcEigValsAtK come back interleaved as
        [up0, dn0, up1, dn1, ...], so band n of spin s lives at 2*n + s. The requested
        band index is remapped to that layout using defPotSpin (defaulting to spin-up
        when defPotSpin is None). In a spin-unpolarized run, or with SOC (where the
        spinor spectrum already resolves spin), the band indices are used verbatim and
        the spin columns have no effect.

        Returns a 1-D tensor of deformation potentials, one per input row.
        """
        defpot_tensors = []
        defPotSpin = getattr(self.system, 'defPotSpin', None)  # per-entry [spin_VB, spin_CB] or None

        for iEntry, defPot_entry in enumerate(self.system.defPotInfo):
            kidx_VB = int(defPot_entry[0])
            kidx_CB = int(defPot_entry[2])
            bidx_VB = int(defPot_entry[1])
            bidx_CB = int(defPot_entry[3])
            def_scale = defPot_entry[4]

            # Spin-polarized (no SOC) eigenvalues are returned interleaved as
            # [up0, dn0, up1, dn1, ...] by calcEigValsAtK, so band n of spin s
            # lives at 2*n + s (s: 0=up, 1=down). Map the (band, spin) request onto
            # that list. Without SOC-off spin polarization the indices are used as-is.
            if self.magBool and not self.SObool:
                sVB = int(defPotSpin[iEntry][0]) if defPotSpin is not None else 0
                sCB = int(defPotSpin[iEntry][1]) if defPotSpin is not None else 0
                bidx_VB = 2 * bidx_VB + sVB
                bidx_CB = 2 * bidx_CB + sCB

            eigValsAtVB = self.calcEigValsAtK(kidx_VB, cachedMats_info, requires_grad=requires_grad, verbosity=verbosity)
            eigValsAtVB_def = self.calcEigValsAtK(kidx_VB, cachedMats_info, requires_grad=requires_grad, def_H=True, def_scale=def_scale, verbosity=verbosity)
            eigValsAtCB = self.calcEigValsAtK(kidx_CB, cachedMats_info, requires_grad=requires_grad, verbosity=verbosity)
            eigValsAtCB_def = self.calcEigValsAtK(kidx_CB, cachedMats_info, requires_grad=requires_grad, def_H=True, def_scale=def_scale, verbosity=verbosity)

            gap_org = eigValsAtCB[bidx_CB] - eigValsAtVB[bidx_VB]
            gap_def = eigValsAtCB_def[bidx_CB] - eigValsAtVB_def[bidx_VB]
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

        # The NL coupling derivative matrices match the Hamiltonian dimension:
        # 2*nbv with spinors, nbv otherwise (block-diagonal, identical in both
        # spin blocks). The SO matrices above are only consumed when self.SObool,
        # which implies spinors, so they stay 2*nbv.
        ndim_nl = 2*nbv if self.spinor else nbv
        NLmats = np.empty([nqp, self.system.getNAtoms(), 3, 2], dtype=object)
        for id1 in range(nqp):
            for id2 in range(self.system.getNAtoms()):
                for id3 in range(3):
                    for id4 in range(2):
                        NLmats[id1,id2,id3, id4] = np.zeros([ndim_nl, ndim_nl], dtype=np.complex128)

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
                    # this potential is block diagonal on spin AND identical in
                    # both spin blocks. It doesn't have the global factor of -i in
                    # front, like SOC does.
                    # up up, 1st integral
                    common = SOprefactor * derivFact * structFact * gdot
                    NLmats[qidx, alpha, gamma, 0][:nbv, :nbv] = isum2 * common
                    # 2nd integral
                    NLmats[qidx, alpha, gamma, 1][:nbv, :nbv] = isum3 * common

                    if self.spinor:
                        # dn dn (only present when the Hamiltonian is a spinor)
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

            if self.spinor:
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

            if self.spinor:
                # local potential has delta function on spin --> block diagonal.
                # The down block carries the same local form factor as the up
                # block (no spin splitting in the coupling derivative here).
                dV[nbv:, nbv:] = prefactor * (atomFF + atomFF_LSD)

            if self.SObool:
                # SOC part
                if isinstance(self.SOmats_couple[qidx, alpha, gamma], torch.Tensor):
                    tmp = self.SOmats_couple[qidx, alpha, gamma]
                else:
                    tmp = torch.tensor(self.SOmats_couple[qidx, alpha, gamma])

                dV = dV + tmp * self.PPparams[self.system.atomTypes[alpha]][5]

            if self.NLbool and self.checknl:
                # NL part (independent of spin-orbit)
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
            # idxVB/idxCB index the raw (eigvalsh-sorted) eigenvalues directly.
            # There is no artificial 2x spin-interleaving to undo: the spectrum is
            # never doubled (spinors off -> distinct bands; spinors on -> the 2*nbv
            # eigensolve gives the spin-resolved bands directly).
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
                    NLbool=self.NLbool,
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

    def set_spinModel(self, newmodel):
        """
        Use this to set the current spin-field model (the learned b(q) that
        splits the up/down local potentials when tot_magnetization != 0).
        """
        self.spinModel = newmodel

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


def _parse_shm_pid(name):
    """
    Parse the owning PID out of a tagged shared-memory segment name of the form
    SOmats_{pid}_{uuidhex}_{iSys}_{kidx} (or the NLmats_ analogue). Returns the
    PID as an int, or None if the name carries no recognizable per-job tag, e.g.
    a legacy untagged name SOmats_{iSys}_{kidx}. The tag is recognized by its
    32-character uuid4 hex field, which distinguishes it from the small integer
    indices of the legacy scheme.
    """
    for prefix in ("SOmats_", "NLmats_"):
        if name.startswith(prefix):
            parts = name[len(prefix):].split("_")
            # tagged layout: pid, 32-char uuid hex, iSys, kidx
            if (len(parts) >= 4 and len(parts[1]) == 32
                    and all(c in "0123456789abcdef" for c in parts[1])):
                try:
                    return int(parts[0])
                except ValueError:
                    return None
            return None
    return None


def _pid_is_running(pid):
    """Return True if a process with this PID currently exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but is owned by someone else (shouldn't happen for
        # our own segments); treat it as alive so we never disturb it.
        return True
    return True


def sweep_stale_shared_memory(max_age_hours=6.0):
    """
    Remove orphaned SOmats_/NLmats_ POSIX shared-memory segments left behind by
    crashed jobs. Only the current user's segments are touched. For tagged names
    the owning PID is parsed out of the name and the segment is removed only when
    that PID is no longer running, so segments belonging to other live jobs are
    never disturbed. Legacy untagged names (which carry no PID) fall back to an
    age-based cutoff of max_age_hours.
    """
    shm_dir = "/dev/shm"
    if not os.path.isdir(shm_dir):
        return
    try:
        my_uid = os.getuid()
    except AttributeError:
        # Non-POSIX platform; nothing to sweep.
        return

    now = time.time()
    max_age_sec = max_age_hours * 3600.0
    removed = 0
    for entry in os.listdir(shm_dir):
        if not (entry.startswith("SOmats_") or entry.startswith("NLmats_")):
            continue
        path = os.path.join(shm_dir, entry)
        try:
            st = os.stat(path)
        except FileNotFoundError:
            continue
        # Only touch this user's own segments.
        if st.st_uid != my_uid:
            continue

        pid = _parse_shm_pid(entry)
        if pid is not None:
            # Tagged name: remove only if the owning job is gone.
            stale = not _pid_is_running(pid)
        else:
            # Legacy untagged name: fall back to an age-based cutoff.
            stale = (now - st.st_mtime) > max_age_sec

        if not stale:
            continue
        try:
            shm = shared_memory.SharedMemory(name=entry)
            shm.close()
            shm.unlink()
            removed += 1
        except FileNotFoundError:
            pass
        except Exception:
            # Best-effort: fall back to removing the backing file directly.
            try:
                os.unlink(path)
                removed += 1
            except OSError:
                pass

    if removed:
        print(f"sweep_stale_shared_memory: removed {removed} orphaned SO/NL shared-memory segment(s).")


def sweep_stale_mat_cache(cache_dir):
    """
    Remove orphaned disk-cache .npy files (SOmats_/NLmats_ written by the low_mem
    disk cache) left behind by crashed jobs. The owning PID is parsed out of the
    tagged filename and the file is removed only when that PID is no longer
    running, so files belonging to other live jobs sharing this directory are
    never disturbed. Matches the per-job-tag policy of sweep_stale_shared_memory.
    """
    if not os.path.isdir(cache_dir):
        return
    removed = 0
    for entry in os.listdir(cache_dir):
        if not (entry.startswith("SOmats_") or entry.startswith("NLmats_")) or not entry.endswith(".npy"):
            continue
        pid = _parse_shm_pid(entry)            # also works on the .npy-suffixed name
        if pid is None or _pid_is_running(pid):
            continue                            # untagged or owned by a live job; leave it
        try:
            os.unlink(os.path.join(cache_dir, entry))
            removed += 1
        except OSError:
            pass
    if removed:
        print(f"sweep_stale_mat_cache: removed {removed} orphaned SO/NL disk-cache file(s) from ./{cache_dir}/.")


def initAndCacheHams(systemsList, NNConfig, PPparams, atomPPOrder, device, model=None, LSDmodels=None, spinModel=None):
    """
    Initialize the ham class for each BulkSystem. 
    dummy_ham is used to initialize and store the cached SOmats and NLmats in dict cachedMats. 
    As I initialize dummy_ham, immediately load them into share memory
    Use a dict "cachedMats_info" to store dtype and shape
    Then remove dummy_ham, and any intermediate variables
    """
    print("\nInitializing the ham class for each BulkSystem. Cache-ing the SOmats, NLmats, and putting them into shared memeory. ")

    # Clean up shared-memory segments orphaned by crashed jobs before we create
    # our own. Only this user's segments are touched; tagged segments owned by a
    # still-running job are left alone (see sweep_stale_shared_memory).
    sweep_stale_shared_memory()
    # Same dead-PID sweep for the low_mem disk cache, so .npy files from crashed
    # jobs in this working directory don't accumulate run after run.
    if bool(NNConfig.get('low_mem', False)):
        sweep_stale_mat_cache(NNConfig.get('mat_cache_dir', 'mat_cache'))

    # Unique per-job tag embedded in every shared-memory segment name so that
    # segments leaked by a crashed job can never collide with this job's (the
    # /dev/shm namespace is global per user). The PID prefix lets a later sweep
    # tell whether the owning job is still alive.
    shm_tag = f"{os.getpid()}_{uuid.uuid4().hex}"

    hams = []
    cachedMats_info = {}
    shm_dict_SO = {}
    shm_dict_NL = {}
    for iSys, sys in enumerate(systemsList):
        start_time = time.time()

        # The SO and NL potentials are both cached as precomputed matrices, and
        # caching/sharing is needed whenever EITHER spin-orbit (SObool) or the
        # non-local potential (NLbool) is active. NLbool defaults to SObool when
        # absent, preserving the legacy behavior.
        SObool = NNConfig['SObool']
        NLbool = NNConfig.get('NLbool', NNConfig['SObool'])
        cacheNeeded = SObool or NLbool

        # Here I separate:
        # 1. Neither SO nor NL --> Just initialize ham. No storage / moving is needed.
        # 2. SO and/or NL, no parallel --> Initialize ham with cache. No storage / moving is needed.
        # 3. SO and/or NL, yes parallel --> Do the complicated storage / moving.
        if not cacheNeeded:
            ham = Hamiltonian(sys, PPparams, atomPPOrder, device, NNConfig=NNConfig, iSystem=iSys, SObool=SObool, NLbool=NLbool, cacheSO=NNConfig['cacheSO'], LSDmodels=LSDmodels, spinModel=spinModel, coupling=sys.fit_eph)
            cachedMats_info = None
            shm_dict_SO = None
            shm_dict_NL = None
        elif cacheNeeded and (NNConfig['num_cores']==0):
            print(f"num_cores set to {NNConfig['num_cores']}. Initializing Hamiltonian by caching SO/NL mats in the ham class (no shared memory).")
            ham = Hamiltonian(sys, PPparams, atomPPOrder, device, NNConfig=NNConfig, iSystem=iSys, SObool=SObool, NLbool=NLbool, cacheSO=NNConfig['cacheSO'], LSDmodels=LSDmodels, spinModel=spinModel, coupling=sys.fit_eph)
            cachedMats_info = None
            shm_dict_SO = None
            shm_dict_NL = None
        elif cacheNeeded and (NNConfig['cacheSO']==0):
            print(f"cacheSO set to {NNConfig['cacheSO']}. Initializing Hamiltonian without caching SO/NL mats.")
            ham = Hamiltonian(sys, PPparams, atomPPOrder, device, NNConfig=NNConfig, iSystem=iSys, SObool=SObool, NLbool=NLbool, cacheSO=False, LSDmodels=LSDmodels, spinModel=spinModel, coupling=sys.fit_eph)
            cachedMats_info = None
            shm_dict_SO = None
            shm_dict_NL = None
        else:
            # Parallel path. Build each k-point's SO/NL matrices DIRECTLY into
            # its shared-memory segment, never materializing the full per-k-point
            # stack in a private array first. This is the minimal-memory layout:
            # the only copy of the cache lives in shared memory (the previous
            # implementation built a complete dummy_ham.SOmats/NLmats stack and
            # then copied it into shared memory, transiently doubling RAM).
            ham = Hamiltonian(sys, PPparams, atomPPOrder, device, NNConfig=NNConfig, iSystem=iSys, SObool=SObool, NLbool=NLbool, cacheSO=False, LSDmodels=LSDmodels, spinModel=spinModel, coupling=sys.fit_eph)
            # Stamp the per-job tag onto the worker-facing ham so its reattach
            # code in calcEigValsAtK reconstructs the exact segment names below.
            ham.shm_tag = shm_tag

            nbv = ham.basis.shape[0]
            nkpt = sys.getNKpts()
            need_SO = ham.SObool
            need_NL = ham.NLbool and ham.checknl
            ndim_NL = 2 * nbv if ham.spinor else nbv
            cdt = np.dtype(np.complex128)
            itemsize = cdt.itemsize

            num_cores = NNConfig['num_cores']
            # SO/NL init runs before any k-point worker pool exists, so it owns
            # the node and threads over k-points regardless of num_cores (the
            # integral kernels release the GIL). init_nworkers is the full-node
            # thread budget for this one-shot phase.
            use_threads = bool(NNConfig.get('init_threads', True)) and (nkpt > 1)
            init_nworkers = init_thread_count(NNConfig.get('num_threads', 0))

            # When disk_cache (low_mem) is on, each k-point's matrices are written
            # to ./<mat_cache_dir>/ as a .npy file (labeled by the job's shm tag)
            # and loaded one k-point at a time in calcEigValsAtK, instead of being
            # held resident in shared memory for the whole run.
            disk_cache = ham.disk_cache
            if disk_cache:
                os.makedirs(ham.mat_cache_dir, exist_ok=True)

            def _alloc_shm(name, shape):
                nbytes = int(np.prod(shape)) * itemsize
                shm = shared_memory.SharedMemory(create=True, size=nbytes, name=name)
                arr = np.ndarray(shape, dtype=cdt, buffer=shm.buf)
                arr[:] = 0
                return shm, arr

            def _build_cache(kind, cachedPrefix, shape, init_fn, shm_dict, shmKeyPrefix):
                # Register shape/dtype so calcEigValsAtK takes the cached-matrix
                # branch (cachedMats_info is its "a cache exists" discriminator).
                for kidx in range(nkpt):
                    cachedMats_info[f"{cachedPrefix}_{iSys}_{kidx}"] = {'dtype': cdt, 'shape': shape}
                if disk_cache:
                    # Build into a private array, write it out, then drop it: only
                    # one k-point's matrices (per worker) are ever resident.
                    def work(kidx):
                        arr = np.zeros(shape, dtype=cdt)
                        init_fn(kidx, arr)
                        np.save(ham._mat_cache_path(kind, kidx), arr)
                        del arr
                    dest = "disk (./%s/)" % ham.mat_cache_dir
                else:
                    # Build directly into the shared-memory segments (no private
                    # copy); shm_dict keeps the segments alive for the run.
                    views = {}
                    for kidx in range(nkpt):
                        shm, arr = _alloc_shm(f"{kind}_{shm_tag}_{iSys}_{kidx}", shape)
                        shm_dict[f"{shmKeyPrefix}_{iSys}_{kidx}"] = shm
                        views[kidx] = arr
                    def work(kidx):
                        init_fn(kidx, views[kidx])
                    dest = "shared memory"
                if use_threads:
                    n_workers = min(init_nworkers, nkpt)
                    print(f"Building {kind} into {dest} with {n_workers} threads\n", flush=True)
                    with ThreadPoolExecutor(max_workers=n_workers) as ex:
                        list(ex.map(work, range(nkpt)))
                else:
                    for kidx in range(nkpt):
                        work(kidx)

            if need_SO:
                so_shape = (ham.nMatGroups, 2 * nbv, 2 * nbv)
                _build_cache("SOmats", "SO", so_shape, ham.initSOmat_fast_oneKpt, shm_dict_SO, "shm_SO")

            if need_NL:
                nl_shape = (ham.nMatGroups, 2, ndim_NL, ndim_NL)
                _build_cache("NLmats", "NL", nl_shape, ham.initNLmat_fast_oneKpt, shm_dict_NL, "shm_NL")

            gc.collect()
            where = f"disk (./{ham.mat_cache_dir}/)" if ham.disk_cache else "shared memory"
            print(f"Finished building the cached SO and NLmats into {where} ...")
        hams.append(ham)
        end_time = time.time()
        print(f"Elapsed time: {(end_time - start_time):.2f} seconds\n")
    return hams, cachedMats_info, shm_dict_SO, shm_dict_NL


def set_LSDModels(ham, LSDmodels):
    ham.set_LSDModels(LSDmodels)
    