import numpy as np
import torch
from constants.constants import *
from utils.pp_func import pot_func, realSpacePot, plotBandStruct


class Hamiltonian:
    def __init__(
        self,
        system,
        device,
        SObool = False,
        NN_locbool = False,
        model = None,
    ):
        # The Hamiltonian is initialized by passing it an initialized and
        # populated bulkSystem class, which contains all the relevant 
        # information about the basis, atoms, etc. "device" should be
        # specified using torch for cpu vs gpu. The other kwargs are
        # specified for using a NN, currently only for the local potential.
       
        self.basis = system.basis() # check if this is done the same as Daniel
        self.kpoints = system.kpts
        self.system = system
        #self.atomPos = system.atomPos
        #self.atomTypes = system.atomTypes

        # if spin orbit, do a bunch of caching to speed up the inner loop 
        # of the optimization. This uses more memory (storing multiple
        # matrices of size 4*nbasis^2) in exchange for avoiding loops
        # over the basis within the optimization inner loop.
        if SObool:
            self.SOmats = np.array(len(self.system.atomTypes), dtype=object)
            for i in range(len(self.system.atomTypes)):
            SOmats = calc

        # send things to gpu, if enabled
        if model in not None:
            model.to(device)
        

    def get_NNmodel(self):
        return
    def set_NNmodel(self):
        return

