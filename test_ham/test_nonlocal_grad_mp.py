"""
Test: SOC/NL prefactor gradients are IDENTICAL between the serial (num_cores=0)
and multiprocessing (num_cores>0) separateKptGrad paths.

The mp path runs each k-point's backward in a spawned worker
(calcEigValsAtK_wGrad_parallel). The worker reads ham.PPparams[atom].grad after
backward (requires_grad is preserved through pickling), kpt-weights it, and
returns it; the parent merges across workers and steps the dedicated optimizer.
This test runs the REAL trainIter_separateKptGrad both ways from an identical
starting state and checks the resulting masked PPparams gradients match.

NOTE: the executable code lives under `if __name__ == "__main__":` because the
multiprocessing 'spawn' start method re-imports this module in every worker.

Run from the repo root:
    python -m test_ham.test_nonlocal_grad_mp
"""
import copy
import pathlib
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from utils.ham import Hamiltonian
from utils.read import BulkSystem, read_PPparams, read_NNConfigFile, setNN
from utils.NN_train import setup_nonlocal_grad, trainIter_separateKptGrad


def main():
    device = torch.device("cpu")
    pwd = pathlib.Path(__file__).parent.resolve()

    # ---- system (small basis for speed) ------------------------------------
    system = BulkSystem()
    system.setSystem(f"{pwd}/inputs/soc/system_0.par")
    system.setInputs(f"{pwd}/inputs/soc/input_0.par")
    system.maxKE = 3.0
    system.setKPointsAndWeights(f"{pwd}/inputs/soc/kpoints_0.par")
    system.setExpBS(f"{pwd}/inputs/soc/bandStruct_0.dat")
    system.bandWeights = torch.ones(system.nBands, dtype=torch.float64)
    atomPPorder = np.unique(system.atomTypes)

    PPparams, totalParams = read_PPparams(atomPPorder, f"{pwd}/inputs/soc/")

    NNConfig = read_NNConfigFile(f"{pwd}/inputs/NN_config.par")
    NNConfig['nonlocal_grad'] = True
    NNConfig['nonlocal_grad_indices'] = [5, 6, 7]
    NNConfig['optimizer_lr'] = 5e-3
    NNConfig['nonlocal_grad_lr'] = 5e-3
    NNConfig['runtime_flag'] = False
    NNConfig['smooth_reorder'] = False
    NNConfig['PPmodel_gaussian_std'] = 1.0  # needed by Net_celu_HeInit_decayGaussian

    # ---- NN local-potential model ------------------------------------------
    torch.manual_seed(0)
    model = setNN(NNConfig, len(atomPPorder))
    model_init_state = copy.deepcopy(model.state_dict())

    # ---- Hamiltonian (cacheSO=False -> on-the-fly, no shared-memory cache) --
    ham = Hamiltonian(system, PPparams, atomPPorder, device, NNConfig=NNConfig,
                      iSystem=0, SObool=True, cacheSO=False, NN_locbool=True, model=model)

    nl_ctx = setup_nonlocal_grad([ham], atomPPorder, NNConfig)
    assert nl_ctx is not None

    pp_snapshot = {atom: p.detach().clone() for atom, p in nl_ctx['params'].items()}

    def restore_state():
        model.load_state_dict(copy.deepcopy(model_init_state))
        with torch.no_grad():
            for atom, p in nl_ctx['params'].items():
                p.copy_(pp_snapshot[atom])
                p.grad = None

    def run(num_cores):
        restore_state()
        NNConfig['num_cores'] = num_cores
        optimizer = torch.optim.Adam(model.parameters(), lr=NNConfig['optimizer_lr'])
        nl_ctx['optimizer'] = torch.optim.Adam(list(nl_ctx['params'].values()),
                                               lr=NNConfig['nonlocal_grad_lr'])
        trainIter_separateKptGrad(model, [system], [ham], NNConfig, optimizer,
                                  cachedMats_info=None, resultsFolder=f"{pwd}/",
                                  epoch=0, prevBS=None, nl_ctx=nl_ctx)
        return {atom: nl_ctx['params'][atom].grad.detach().clone()
                for atom in nl_ctx['params']}

    print("Running SERIAL (num_cores=0) ...")
    g_serial = run(0)
    print("Running MULTIPROCESSING (num_cores=2) ...")
    g_mp = run(2)

    print("\nComparing masked PPparams gradients (serial vs mp):")
    all_ok = True
    for atom in nl_ctx['params']:
        gs, gm = g_serial[atom], g_mp[atom]
        for i in nl_ctx['indices']:
            diff = abs(float(gs[i]) - float(gm[i]))
            rel = diff / (abs(float(gs[i])) + 1e-12)
            ok = (rel < 1e-9) or (diff < 1e-12)
            all_ok = all_ok and ok
            print(f"  {atom}[{i}]: serial={float(gs[i]):+.8e}  mp={float(gm[i]):+.8e}  "
                  f"abs_diff={diff:.2e}  [{'OK' if ok else 'FAIL'}]")

    print("\n=========================================")
    print(f"serial-vs-mp NL gradient match: {'PASS' if all_ok else 'FAIL'}")
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    print("=========================================")


if __name__ == "__main__":
    main()
