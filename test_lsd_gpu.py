#!/usr/bin/env python3
"""
test_lsd_gpu.py -- smoke test for GPU + threaded multi-GPU LSD pre-training.

It builds tiny synthetic datasets (no fitting data needed) for three fake
atom types and exercises the real init_LSD_train_GPU on every visible GPU:

  1. sequential   : all atoms trained one after another on a single device
  2. parallel     : atoms trained concurrently, atom k on cuda:(k % ngpu)
                    (exactly the threading init_LSD_PP uses)

It checks that (a) each net is actually on its assigned device, (b) the loss
decreases, and (c) sequential and parallel give the same answer (deterministic
seeds) -- i.e. the threading/device placement doesn't corrupt anything. It also
prints wall-clock for both so you can see when parallelism pays off.

Run on a GPU node, e.g.:
    CUDA_VISIBLE_DEVICES=0,1,2 python test_lsd_gpu.py
    python test_lsd_gpu.py --n_unique 64 --n_q 4096 --hidden 40 20 --epochs 2000
    python test_lsd_gpu.py --hidden 512 512 --n_unique 256   # stress one A100
"""

import os, time, shutil, tempfile, argparse, threading
import torch

torch.set_default_dtype(torch.float64)

from utils.nn_models import Net_celu_HeInit_decayGaussian_LSD
from utils.init_LSD_train import init_LSD_data, init_LSD_train_GPU
from utils.init_NN_train import init_Zunger_weighted_mse

# Three fake atom types with the real per-element descriptor widths (Cs=3,I=2,Pb=5).
ATOMS = [("Cs", 3), ("I", 2), ("Pb", 5)]


def build_datasets(n_descr, n_unique, n_q, seed):
    """Synthetic (descriptor, q) -> v_ref dataset with the init_LSD_data layout."""
    g = torch.Generator().manual_seed(seed)
    descr = torch.rand(n_unique, n_descr, generator=g)              # [n_unique, n_descr]
    # mimic real env_descriptors, which require grad (graph back to atomPos) -- the
    # training code must detach these or backward() reuses a freed graph.
    descr.requires_grad_(True)
    q = torch.linspace(0.0, 30.0, n_q)                             # [n_q]

    N_alphas = descr.repeat_interleave(n_q, dim=0)                  # [n_unique*n_q, n_descr]
    q_all = q.repeat(n_unique)                                      # [n_unique*n_q]
    # smooth target that depends on both the descriptor and q (something to learn)
    amp = descr.sum(dim=1).repeat_interleave(n_q)                   # [n_unique*n_q]
    v_all = 0.1 * amp * torch.exp(-q_all**2 / 8.0) * torch.cos(q_all)

    train = init_LSD_data(N_alphas, q_all, v_all, n_unique, n_q, train=True)
    val   = init_LSD_data(N_alphas, q_all, v_all, n_unique, n_q, train=False)
    return train, val


def make_model(n_descr, hidden, seed):
    torch.manual_seed(seed)                                         # deterministic init
    layers = [n_descr + 1] + list(hidden) + [1]
    model = Net_celu_HeInit_decayGaussian_LSD(layers, gaussian_std=2.0)
    model.N_ref = torch.zeros(1, n_descr)                           # reference environment
    return model


def cost_endpoints(path):
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]
    first = float(lines[0].split()[1])
    last  = float(lines[-1].split()[1])
    return first, last


def run_atom(atom, n_descr, device, NNConfig, resultsFolder, results, seed):
    """Train one atom; record device/loss/time into the shared results dict."""
    train, val = build_datasets(n_descr, NNConfig['_n_unique'], NNConfig['_n_q'], seed)
    model = make_model(n_descr, NNConfig['_hidden'], seed)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)

    t0 = time.time()
    init_LSD_train_GPU(model, device, train, val, init_Zunger_weighted_mse, opt, sch,
                       NNConfig, atom, resultsFolder, make_plots=False)
    dt = time.time() - t0

    p0 = next(model.parameters())
    l0, l1 = cost_endpoints(os.path.join(resultsFolder, f"init_{atom}_train_cost.dat"))
    results[atom] = dict(req=str(device), got=str(p0.device), dtype=str(p0.dtype), t=dt,
                         l0=l0, l1=l1, nrows=train.len,
                         nparams=sum(p.numel() for p in model.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_unique", type=int, default=64, help="unique environments per atom")
    ap.add_argument("--n_q", type=int, default=4096, help="q-grid points per environment")
    ap.add_argument("--hidden", type=int, nargs="+", default=[40, 20], help="hidden layer widths")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--dtype", default="float32", help="training precision: float32 | float64")
    args = ap.parse_args()

    ngpu = torch.cuda.device_count()
    print(f"visible CUDA devices: {ngpu}  "
          f"({[torch.cuda.get_device_name(i) for i in range(ngpu)]})")
    print(f"scale: n_unique={args.n_unique}  n_q={args.n_q}  hidden={args.hidden}  "
          f"epochs={args.epochs}  -> {args.n_unique*args.n_q:,} rows/atom")

    NNConfig = {
        'init_LSD_num_epochs': args.epochs,
        'init_LSD_plot_every': args.epochs + 1,   # only the epoch-0 checkpoint is written
        'init_LSD_scheduler_step': max(1, args.epochs // 4),
        'SHOWPLOTS': 0,
        'init_LSD_dtype': args.dtype,
        '_n_unique': args.n_unique, '_n_q': args.n_q, '_hidden': args.hidden,
    }

    tmp = tempfile.mkdtemp(prefix="lsd_gpu_test_")
    os.makedirs(os.path.join(tmp, "LSD"), exist_ok=True)
    rf = tmp + "/"
    seeds = {atom: 1000 + i for i, (atom, _) in enumerate(ATOMS)}

    # warm up each device so CUDA-init cost isn't charged to the timed runs
    for i in range(ngpu):
        torch.zeros(1, device=f"cuda:{i}")

    base_dev = torch.device("cuda:0" if ngpu >= 1 else "cpu")

    # ---- sequential ---------------------------------------------------------
    seq = {}
    t0 = time.time()
    for atom, nd in ATOMS:
        run_atom(atom, nd, base_dev, NNConfig, rf, seq, seeds[atom])
    seq_t = time.time() - t0

    # ---- parallel (threaded, one atom per GPU) ------------------------------
    par = {}
    par_t = None
    if ngpu > 1:
        threads = []
        t0 = time.time()
        for k, (atom, nd) in enumerate(ATOMS):
            dev = torch.device(f"cuda:{k % ngpu}")
            th = threading.Thread(target=run_atom,
                                  args=(atom, nd, dev, NNConfig, rf, par, seeds[atom]),
                                  name=f"LSD-{atom}")
            th.start(); threads.append(th)
        for th in threads:
            th.join()
        par_t = time.time() - t0

    # ---- report -------------------------------------------------------------
    print(f"\n--- sequential (single device, train dtype={args.dtype}) ---")
    for atom, _ in ATOMS:
        r = seq[atom]
        print(f"  {atom:3s} dev={r['got']:8s} rows={r['nrows']:>9,} params={r['nparams']:>6} "
              f"loss {r['l0']:.3e}->{r['l1']:.3e}  {r['t']:.2f}s")
    print(f"  total: {seq_t:.2f}s")

    ok = all(seq[a]['l1'] < seq[a]['l0'] for a, _ in ATOMS)
    restored = all(seq[a]['dtype'] == 'torch.float64' for a, _ in ATOMS)
    print(f"\nloss decreased for all atoms: {ok}")
    print(f"net restored to float64 after training: {restored}")

    if par_t is not None:
        print("\n--- parallel (one atom per GPU) ---")
        for atom, _ in ATOMS:
            r = par[atom]
            print(f"  {atom:3s} req={r['req']:8s} got={r['got']:8s} "
                  f"loss {r['l0']:.3e}->{r['l1']:.3e}  {r['t']:.2f}s")
        print(f"  total: {par_t:.2f}s   speedup vs sequential: {seq_t/par_t:.2f}x")
        placed = all(par[a]['req'] == par[a]['got'] for a, _ in ATOMS)
        maxdiff = max(abs(par[a]['l1'] - seq[a]['l1']) for a, _ in ATOMS)
        print(f"each net on its assigned device: {placed}")
        print(f"max |parallel - sequential| final loss: {maxdiff:.2e} "
              f"(should be ~0; deterministic seeds)")
    else:
        print("\n(parallel test skipped: needs >1 visible CUDA device)")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
