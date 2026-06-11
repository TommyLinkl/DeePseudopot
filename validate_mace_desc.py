#!/usr/bin/env python3
"""
validate_mace_desc.py -- Phase 1 check: do MACE descriptors separate the
environments (resolve the collisions) the hand-crafted ones could not?

For every system it builds the BulkSystem geometry (inputs_lsd_train/system_*.par),
computes MACE per-atom descriptors, and pairs each atom with its target
difference potential (final_pot_q of the distorted atom minus the cubic
reference). Then, per element, it reports descriptor dimensionality, nearest-
neighbour separation in standardized descriptor space, and a collision count
(near-identical descriptors with different Δv) -- the same idea as
diagnose_lsd_fit.py but on the raw per-atom MACE features (no dedup).

Run:  python validate_mace_desc.py
"""
import os, glob, re
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from utils.read import BulkSystem
from utils.mace_descriptors import mace_env_descriptors

INPUTS   = "inputs_lsd_train"
CUBIC    = "results_cubic_q-0.95_iter1_all"
DISTORTED = sorted(glob.glob("results_0[0-9][0-9]_g_1"))
ELEMENTS = ["Cs", "I", "Pb"]
MODEL    = "medium"


def main():
    cubic_ref = {el: np.loadtxt(f"{CUBIC}/final_pot_q_{el}.dat")[:, 1] for el in ELEMENTS}

    D = {el: [] for el in ELEMENTS}     # descriptors
    T = {el: [] for el in ELEMENTS}     # target diff potentials

    # systems 1..N are the distorted fits (system_0 is the cubic reference, diff=0)
    for iSys, ddir in enumerate(DISTORTED, start=1):
        bs = BulkSystem(); bs.systemName = "CsPbI3"
        bs.setSystem(f"{INPUTS}/system_{iSys}.par")
        env = mace_env_descriptors(bs, model=MODEL)
        print(f"system {iSys} ({ddir}) descriptors: "
              + ", ".join(f"{el}:{tuple(env[el].shape)}" for el in ELEMENTS))
        for el in ELEMENTS:
            for k in range(env[el].shape[0]):
                fp = f"{ddir}/final_pot_q_{el}{k}.dat"
                v = np.loadtxt(fp)[:, 1]
                D[el].append(env[el][k].numpy())
                T[el].append(v - cubic_ref[el])

    q = np.loadtxt(f"{CUBIC}/final_pot_q_{ELEMENTS[0]}.dat")[:, 0]
    lowq = q <= 8.0
    print("\n" + "=" * 70)
    for el in ELEMENTS:
        Dm = np.array(D[el]); Tm = np.array(T[el])
        n, dim = Dm.shape
        std = Dm.std(0)
        active = std > 1e-8
        Dz = np.zeros_like(Dm)
        Dz[:, active] = (Dm[:, active] - Dm[:, active].mean(0)) / std[active]

        nn_dist, coll = [], 0
        for i in range(n):
            d2 = ((Dz - Dz[i]) ** 2).sum(1); d2[i] = np.inf
            j = int(np.argmin(d2)); dnn = np.sqrt(d2[j])
            nn_dist.append(dnn)
        med = np.median(nn_dist)
        # collision: nearest neighbour very close (<10% of median sep) yet different Δv
        for i in range(n):
            d2 = ((Dz - Dz[i]) ** 2).sum(1); d2[i] = np.inf
            j = int(np.argmin(d2))
            if np.sqrt(d2[j]) < 0.1 * med and \
               np.sqrt(((Tm[i] - Tm[j])[lowq] ** 2).mean()) > 0.1:
                coll += 1

        print(f"\n=== {el}  | {n} atoms | D={dim} ===")
        print(f"  active dims (std>1e-8)        : {int(active.sum())}/{dim}")
        print(f"  median NN distance (z-space)  : {med:.3f}")
        print(f"  descriptor 'collisions'       : {coll}")
        print(f"  => {'MACE separates the environments (collisions ~0)' if coll == 0 else 'still some collisions'}")
    print("=" * 70)
    print("If collisions ~0 everywhere, MACE descriptors are expressive enough; "
          "proceed to wire in the descriptor_backend flag (Phase 2).")


if __name__ == "__main__":
    main()
