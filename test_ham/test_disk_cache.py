"""
Test: the low_mem on-disk SO/NL cache produces band structures IDENTICAL to the
in-shared-memory cache. With low_mem on, initAndCacheHams writes each k-point's
SO/NL matrices to .npy files under ./<mat_cache_dir>/ (labeled by the job tag)
and calcEigValsAtK loads them one k-point at a time, instead of holding the whole
stack resident in POSIX shared memory.

Exercises the parallel caching path (num_cores>0), so it also confirms the
per-job-tag file naming and the np.save/np.load round-trip across worker procs.

Run from the repo root:
    python -m test_ham.test_disk_cache
"""
import os
import glob
import pathlib
import shutil
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from utils.ham import initAndCacheHams
from utils.read import BulkSystem, read_PPparams, read_NNConfigFile

device = torch.device("cpu")
pwd = pathlib.Path(__file__).parent.resolve()


def build_system():
    system = BulkSystem()
    system.setSystem(f"{pwd}/inputs/soc/system_0.par")
    system.setInputs(f"{pwd}/inputs/soc/input_0.par")
    system.maxKE = 3.0   # small basis for speed
    system.setKPointsAndWeights(f"{pwd}/inputs/soc/kpoints_0.par")
    system.setExpBS(f"{pwd}/inputs/soc/bandStruct_0.dat")
    return system


def run(low_mem, mat_cache_dir):
    system = build_system()
    atomPPorder = np.unique(system.atomTypes)
    PPparams, _ = read_PPparams(atomPPorder, f"{pwd}/inputs/soc/")
    NNConfig = read_NNConfigFile(f"{pwd}/inputs/NN_config.par")
    NNConfig['SObool'] = True
    NNConfig['NLbool'] = True
    NNConfig['cacheSO'] = True
    NNConfig['num_cores'] = 2          # exercise the parallel caching path
    NNConfig['low_mem'] = low_mem
    NNConfig['mat_cache_dir'] = mat_cache_dir

    hams, cachedMats_info, shm_dict_SO, shm_dict_NL = initAndCacheHams(
        [system], NNConfig, PPparams, atomPPorder, device)
    ham = hams[0]
    # Read the cache back serially in the main process (still exercises the disk
    # np.load path in calcEigValsAtK) -- avoids a second mp.Pool just to diagonalize.
    nkpt = system.getNKpts()
    bs = torch.stack([ham.calcEigValsAtK(k, cachedMats_info, requires_grad=False).detach()
                      for k in range(nkpt)])

    # Release shared memory (no-op / empty dicts on the disk path).
    for d in (shm_dict_SO, shm_dict_NL):
        if d is not None:
            for shm in d.values():
                shm.close()
                shm.unlink()
    return bs


cache_dir = str(pwd / "_disk_cache_test")
shutil.rmtree(cache_dir, ignore_errors=True)

all_ok = True

bs_shm = run(low_mem=False, mat_cache_dir=cache_dir)
bs_disk = run(low_mem=True,  mat_cache_dir=cache_dir)

# The disk path must have actually written tagged .npy files.
files = sorted(glob.glob(os.path.join(cache_dir, "*.npy")))
n_so = len([f for f in files if os.path.basename(f).startswith("SOmats_")])
n_nl = len([f for f in files if os.path.basename(f).startswith("NLmats_")])
nkpt = build_system().getNKpts()
files_ok = (n_so == nkpt) and (n_nl == nkpt)
all_ok = all_ok and files_ok
print(f"disk-cache files written: {n_so} SO + {n_nl} NL .npy (expected {nkpt} each)  [{'OK' if files_ok else 'FAIL'}]")

max_diff = (bs_shm - bs_disk).abs().max().item()
bs_ok = torch.allclose(bs_shm, bs_disk, atol=1e-12, rtol=0)
all_ok = all_ok and bs_ok
print(f"band-structure shm vs disk max|Δ| = {max_diff:.2e}  [{'OK' if bs_ok else 'FAIL'}]")

shutil.rmtree(cache_dir, ignore_errors=True)

print("\n=========================================")
print(f"disk cache == shared memory: {'PASS' if all_ok else 'FAIL'}")
print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
print("=========================================")
