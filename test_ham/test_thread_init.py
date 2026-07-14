"""
Test: the OpenMP-style shared-memory THREADED initialization of the SO/NL
matrices is bit-for-bit identical to the serial initialization, and the
direct-to-shared-memory cache path (initAndCacheHams) yields an identical band
structure to the in-class cached path.

The executable code lives under `if __name__ == "__main__":` because the
multiprocessing 'spawn'/'forkserver' start method (used by calcBandStruct_noGrad
when num_cores>0) re-imports this module in every worker.

Run from the repo root:
    python -m test_ham.test_thread_init
"""
import pathlib
import sys
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from utils.ham import Hamiltonian, initAndCacheHams
from utils.read import BulkSystem, read_PPparams, read_NNConfigFile

device = torch.device("cpu")
pwd = pathlib.Path(__file__).parent.resolve()


def build_system():
    system = BulkSystem()
    system.setSystem(f"{pwd}/inputs/soc/system_0.par")
    system.setInputs(f"{pwd}/inputs/soc/input_0.par")
    system.maxKE = 3.0
    system.setKPointsAndWeights(f"{pwd}/inputs/soc/kpoints_0.par")
    system.setExpBS(f"{pwd}/inputs/soc/bandStruct_0.dat")
    system.fit_eph = False
    return system


def make_ham(num_cores, init_threads, low_mem=False):
    system = build_system()
    atomPPorder = np.unique(system.atomTypes)
    PPparams, _ = read_PPparams(atomPPorder, f"{pwd}/inputs/soc/")
    NNConfig = read_NNConfigFile(f"{pwd}/inputs/NN_config.par", f"{pwd}/")
    NNConfig['num_cores'] = num_cores
    NNConfig['init_threads'] = init_threads
    NNConfig['low_mem'] = low_mem
    ham = Hamiltonian(system, PPparams, atomPPorder, device, NNConfig=NNConfig,
                      iSystem=0, SObool=True, cacheSO=True)
    return ham


def main():
    all_ok = True

    # ---- 1) serial vs threaded init: SO and NL matrices identical -----------
    ham_serial = make_ham(num_cores=0, init_threads=True)
    ham_thread = make_ham(num_cores=4, init_threads=True)

    so_diff = np.abs(ham_serial.SOmats - ham_thread.SOmats).max()
    nl_diff = np.abs(ham_serial.NLmats - ham_thread.NLmats).max()
    so_ok = so_diff == 0.0
    nl_ok = nl_diff == 0.0
    all_ok = all_ok and so_ok and nl_ok
    print(f"[init] serial vs threaded  SO max|d| = {so_diff:.2e}  [{'OK' if so_ok else 'FAIL'}]")
    print(f"[init] serial vs threaded  NL max|d| = {nl_diff:.2e}  [{'OK' if nl_ok else 'FAIL'}]")

    # threaded init must also match for low_mem grouping
    ham_serial_lm = make_ham(num_cores=0, init_threads=True, low_mem=True)
    ham_thread_lm = make_ham(num_cores=4, init_threads=True, low_mem=True)
    so_diff_lm = np.abs(ham_serial_lm.SOmats - ham_thread_lm.SOmats).max()
    nl_diff_lm = np.abs(ham_serial_lm.NLmats - ham_thread_lm.NLmats).max()
    lm_ok = (so_diff_lm == 0.0) and (nl_diff_lm == 0.0)
    all_ok = all_ok and lm_ok
    print(f"[init,low_mem] serial vs threaded  SO/NL max|d| = {max(so_diff_lm, nl_diff_lm):.2e}  [{'OK' if lm_ok else 'FAIL'}]")

    # ---- 2) band structure from serial cache vs threaded init --------------
    bs_serial = ham_serial.calcBandStruct().detach()
    bs_thread = ham_thread.calcBandStruct().detach()
    bs_ok = torch.allclose(bs_serial, bs_thread, atol=0, rtol=0)
    all_ok = all_ok and bs_ok
    print(f"[BS] serial-init vs threaded-init  max|d| = {(bs_serial - bs_thread).abs().max():.2e}  [{'OK' if bs_ok else 'FAIL'}]")

    # ---- 3) direct-to-shared-memory cache path (initAndCacheHams) -----------
    ref_bs = ham_serial.calcBandStruct().detach()

    system = build_system()
    atomPPorder = np.unique(system.atomTypes)
    PPparams, _ = read_PPparams(atomPPorder, f"{pwd}/inputs/soc/")
    NNConfig = read_NNConfigFile(f"{pwd}/inputs/NN_config.par", f"{pwd}/")
    NNConfig['num_cores'] = 4
    NNConfig['init_threads'] = True
    NNConfig['cacheSO'] = True

    hams, cachedMats_info, shm_SO, shm_NL = initAndCacheHams(
        [system], NNConfig, PPparams, atomPPorder, device)
    try:
        bs_shm = hams[0].calcBandStruct_noGrad(cachedMats_info).detach()
        shm_ok = torch.allclose(ref_bs, bs_shm, atol=0, rtol=0)
        all_ok = all_ok and shm_ok
        print(f"[shm] direct-to-shm BS vs reference  max|d| = {(ref_bs - bs_shm).abs().max():.2e}  [{'OK' if shm_ok else 'FAIL'}]")
    finally:
        if shm_SO is not None:
            for shm in shm_SO.values():
                shm.close(); shm.unlink()
        if shm_NL is not None:
            for shm in shm_NL.values():
                shm.close(); shm.unlink()

    print()
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
