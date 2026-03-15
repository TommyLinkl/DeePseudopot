import pathlib
import sys

import numpy as np
import torch


PWD = pathlib.Path(__file__).parent.resolve()
REPO_ROOT = PWD.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.ham import Hamiltonian
from utils.read import init_critical_NNconfig


class DummySystem:
    def __init__(self):
        self.scale = 1.0
        self.unitCellVectors = torch.eye(3, dtype=torch.float64)
        self.atomPos = torch.zeros((1, 3), dtype=torch.float64)
        self.atomTypes = np.array(["X"])
        self.kpts = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float64)
        self.nBands = 2
        self.bandOrderMatrix = np.array([[0, 1], [0, 1]], dtype=int)
        self.defPotInfo = None

    def basis(self):
        return torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)

    def getNAtomTypes(self):
        return 1

    def getNAtoms(self):
        return 1

    def getNKpts(self):
        return self.kpts.shape[0]


class DummyHamiltonian(Hamiltonian):
    def initSOmat_fast(self, SOwidth=0.7, defbool=False, idxGap=None):
        marker = 10.0 if not defbool else 100.0 + float(idxGap)
        return marker

    def initNLmat_fast(self, width1=1.0, width2=1.0, shift=1.5, defbool=False, idxGap=None):
        marker = 20.0 if not defbool else 200.0 + float(idxGap)
        return marker

    def buildVlocMat(self, addMat=None):
        return addMat

    def buildSOmat(self, kidx, preComp_SOmats_kidx=None, addMat=None):
        marker = self.SOmats if preComp_SOmats_kidx is None else preComp_SOmats_kidx
        addMat = addMat.clone()
        addMat[0, 0] += marker
        return addMat

    def buildNLmat(self, kidx, preComp_NLmats_kidx=None, addMat=None):
        marker = self.NLmats if preComp_NLmats_kidx is None else preComp_NLmats_kidx
        addMat = addMat.clone()
        addMat[1, 1] += marker
        return addMat


def make_hamiltonian():
    system = DummySystem()
    pp_params = {"X": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float64)}
    nn_config = init_critical_NNconfig()
    ham = DummyHamiltonian(system, pp_params, np.array(["X"]), torch.device("cpu"), NNConfig=nn_config, iSystem=0, SObool=True)
    return system, ham


def calc_defpots(rows):
    system, ham = make_hamiltonian()
    system.defPotInfo = np.array(rows, dtype=np.float64)
    return ham.calcDefPots(requires_grad=False, verbosity=0).detach().cpu().numpy()


def main():
    same_k_row = [0, 0, 0, 1, 1.0001, 0.0, 1.0]
    diff_k_row = [0, 0, 1, 1, 1.0001, 0.0, 1.0]

    same_only = calc_defpots([same_k_row])
    diff_only = calc_defpots([diff_k_row])
    combined = calc_defpots([same_k_row, diff_k_row])
    combined_reversed = calc_defpots([diff_k_row, same_k_row])

    np.testing.assert_allclose(combined, np.array([same_only[0], diff_only[0]]), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(combined_reversed, np.array([diff_only[0], same_only[0]]), rtol=0.0, atol=1e-12)

    # The different-k row must stay correct regardless of whether a same-k row ran before it.
    np.testing.assert_allclose(combined[1], diff_only[0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(combined_reversed[0], diff_only[0], rtol=0.0, atol=1e-12)

    print("test_defpot_multi_k_cache: PASS")


if __name__ == "__main__":
    main()
