#!/usr/bin/env python3
"""
diagnose_lsd_fit.py -- why is the LSD fit not as good as you want?

It separates the three hypotheses:
  (A) descriptors not expressive enough  -> irreducible error FLOOR is high
  (B) model too small / training too short -> model error >> floor
  (C) targets themselves noisy (non-unique base-PP fits) -> also shows as a floor

Core idea: the LSD net learns a FUNCTION  descriptors N -> Delta v(q).  If two
environments have (nearly) the same N but different Delta v(q), a function cannot
fit both -- the best it can do is output their average, incurring an irreducible
error.  We estimate that floor with a leave-one-out nearest-neighbour predictor
in descriptor space and compare it to the model's actual RMSE (sqrt of final
training cost).  Verdict:
    floor ~ model RMSE      -> bottleneck is descriptors/targets (A or C)
    floor << model RMSE     -> bottleneck is model capacity / optimization (B)

Run:  python diagnose_lsd_fit.py [inputs_dir] [results_dir]
      (defaults: inputs_lsd_train/  results_lsd_train/)
"""
import os, sys, glob, re
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from utils.read import BulkSystem
from utils.init_LSD_train import env_keep_mask

INPUTS  = sys.argv[1] if len(sys.argv) > 1 else "inputs_lsd_train/"
RESULTS = sys.argv[2] if len(sys.argv) > 2 else "results_lsd_train/"
SYSTEM_NAME = "CsPbI3"
ELEMENTS = ["Cs", "I", "Pb"]
RTOL = 0.05                       # must match init_LSD_PP's dedup


def _read_backend():
    cfg = os.path.join(INPUTS, "NN_config.par")
    if os.path.exists(cfg):
        for line in open(cfg):
            s = line.split('#')[0].strip()
            if s.startswith('descriptor_backend'):
                return s.split('=', 1)[1].strip()
    return 'handcrafted'

BACKEND = _read_backend()


def gather(el):
    """Return D [n_env, n_descr] (kept rows, training order) and T [n_env, n_q]."""
    sys_files = sorted(glob.glob(os.path.join(INPUTS, "system_*.par")),
                       key=lambda p: int(re.search(r"system_(\d+)", p).group(1)))
    D, T, q_ref = [], [], None
    for iSys, sp in enumerate(sys_files):
        bs = BulkSystem(); bs.systemName = SYSTEM_NAME
        bs.setSystem(sp); bs.compute_descriptors(backend=BACKEND)
        desc = bs.env_descriptors[el]
        keep = env_keep_mask(desc, BACKEND, rtol=RTOL)
        kept = [r for r in range(keep.shape[0]) if bool(keep[r])]
        for i, r in enumerate(kept):
            fp = os.path.join(INPUTS, f"pot_q_{el}_diff_{iSys}_{i}.par")
            d = np.loadtxt(fp); q_ref = d[:, 0]
            D.append(desc[r].detach().numpy())
            T.append(d[:, 1])
    return np.array(D), np.array(T), q_ref


def main():
    import traceback
    for el in ELEMENTS:
        try:
            diagnose_one(el)
        except Exception:
            print(f"\n=== {el}: FAILED ===")
            traceback.print_exc()


def diagnose_one(el):
        D, T, q = gather(el)
        n_env, n_descr = D.shape
        lowq = q <= 8.0

        # z-normalize descriptors so the NN distance is scale-free; flag dead dims
        std = D.std(axis=0)
        active = std > 1e-6
        Dz = np.zeros_like(D)
        Dz[:, active] = (D[:, active] - D[:, active].mean(0)) / std[active]

        # leave-one-out 1-NN in descriptor space
        # floor: best single-valued model outputs the neighbour-average -> err ~ dist/2
        sq_floor_full, sq_floor_low, nn_dist = [], [], []
        for i in range(n_env):
            d2 = ((Dz - Dz[i])**2).sum(1); d2[i] = np.inf
            j = int(np.argmin(d2))
            nn_dist.append(np.sqrt(d2[j]))
            diff = T[i] - T[j]
            sq_floor_full.append(0.25 * (diff**2).mean())          # midpoint model
            sq_floor_low.append(0.25 * (diff[lowq]**2).mean())
        floor_full = np.sqrt(np.mean(sq_floor_full))
        floor_low  = np.sqrt(np.mean(sq_floor_low))

        # model's actual error from the training cost log (mean MSE over full grid)
        cost = np.loadtxt(os.path.join(RESULTS, f"init_{el}_train_cost.dat"))
        model_rmse = float(np.sqrt(cost[-1, 1]))

        # collisions: descriptor-near pairs with large target difference
        collisions = sum(1 for i in range(n_env)
                         for j in range(i+1, n_env)
                         if np.all(np.abs(D[i]-D[j]) <= RTOL*(np.abs(D[i])+1e-9))
                         and np.sqrt(((T[i]-T[j])[lowq]**2).mean()) > 0.1)

        # near-dead dims: vary <10% of the most-varying dim -> carry little info
        near_dead = [d for d in range(n_descr) if std[d] < 0.1 * std.max()]
        n_thresh = max(1, n_env // 30)            # tolerate a couple of noisy targets

        # Verdict keys on COLLISIONS (the robust signal). The NN floor is only a
        # loose UPPER bound on irreducible error: when neighbours are well
        # separated (large median NN dist), a flexible net rightly beats it, so
        # "model > floor" with ~0 collisions means UNDERFITTING, not weak
        # descriptors.
        if collisions > n_thresh:
            verdict = (f"DESCRIPTOR-LIMITED: {collisions} collisions "
                       f"(near-identical descriptors, different Δv) -> add/improve descriptors")
        else:
            note = f"  [aside: near-dead dim(s) {near_dead} add little]" if near_dead else ""
            verdict = ("descriptors SUFFICIENT (~0 collisions, single-valued map); "
                       "residual error is MODEL CAPACITY / TRAINING or target noise "
                       "-> bigger net / more epochs" + note)

        print(f"\n=== {el}  | {n_env} envs | {n_descr} descriptors ===")
        print(f"  descriptor std per dim : {np.array2string(std, precision=3)}")
        if near_dead:
            print(f"  !! near-dead descriptor dims (std < 0.1*max): {near_dead}")
        print(f"  median NN distance (z-space)        : {np.median(nn_dist):.3f}")
        print(f"  NN-floor rmse (LOOSE upper bound, full grid): {floor_full:.4e}")
        print(f"  NN-floor rmse (LOOSE upper bound, q<=8)     : {floor_low:.4e}")
        print(f"  MODEL rmse (sqrt final train cost)  : {model_rmse:.4e}")
        print(f"  descriptor 'collisions' (near N, |Δv(q<8)|rms>0.1): {collisions}")
        print(f"  => {verdict}")


if __name__ == "__main__":
    main()
