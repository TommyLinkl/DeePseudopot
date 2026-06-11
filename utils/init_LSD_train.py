import os, time, threading
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
import re

from .constants import *
from .pp_func import pot_funcLSD, plotLSD, plot_training_validation_cost, qSpacePot_ft
from .NN_train import print_and_inspect_gradients, print_and_inspect_NNParams
from .init_NN_train import init_Zunger_weighted_mse

torch.set_default_dtype(torch.float64)

# matplotlib (pyplot) is not thread-safe; serialize all plotting so the LSD
# correction nets can be trained on separate GPUs in parallel threads.
_PLOT_LOCK = threading.Lock()

# Max environments plotted per atom type in the LSD init diagnostics. The MACE
# backend keeps every atom as its own environment, so n_unique can be hundreds;
# plotting them all (each a figure + a 4096-pt FT) dominates the end-of-training
# time. Only the LSD pretraining plots are capped by this.
_MAX_PLOT_ENVS = 10


def _select_lsd_device(NNConfig):
    """Pick the device for LSD pre-training, independent of the band-structure
    (CPU) path. 'auto' -> first CUDA device if present, else CPU. An explicit
    'cuda', 'cuda:N' or 'cpu' in NN_config is honored."""
    pref = str(NNConfig.get('init_LSD_device', 'auto')).lower()
    if pref.startswith('cpu'):
        return torch.device('cpu')
    if not torch.cuda.is_available():
        if pref not in ('auto', 'cpu'):
            print(f"WARNING: init_LSD_device='{pref}' but CUDA is unavailable; using CPU.")
        return torch.device('cpu')
    return torch.device('cuda') if pref == 'auto' else torch.device(pref)


def _torch_dtype(name):
    """Map an init_LSD_dtype string to a torch dtype (default float32)."""
    return {'float32': torch.float32, 'fp32': torch.float32, 'float': torch.float32,
            'float64': torch.float64, 'fp64': torch.float64, 'double': torch.float64
            }.get(str(name).lower(), torch.float32)

class init_LSD_data(Dataset):
    def __init__(self, N_alphas, q, v_ref, n_unique, n_q, train=True):
        """
        Custom dataset for neural network pseudopotential training.

        Args:
            N_alphas: shape [n_unique * n_q_grid, 2]  — columns are [G2, G4]
            q:        shape [n_unique * n_q_grid]
            v_ref:    shape [n_unique * n_q_grid]
        """
        self.n_q_grid = n_q
        self.n_unique = n_unique

        # N_alphas is already [n_samples, 2]; keep as-is
        self.N_alphas  = N_alphas                           # [n_unique * n_q_grid, 2]
        self.q         = q.reshape(-1, 1)                   # [n_unique * n_q_grid, 1]
        self.vq_atoms  = v_ref.reshape(-1, 1)               # [n_unique * n_q_grid, 1]

        # inputs: [G2, G4, q]  →  shape [n_unique * n_q_grid, 3]
        self.inputs = torch.cat((self.N_alphas, self.q), dim=1)

        if train:
            rand_indices   = torch.randperm(self.inputs.shape[0])
            self.inputs    = self.inputs[rand_indices]
            self.vq_atoms  = self.vq_atoms[rand_indices]
            # Up-weight the physically-important q in [3,8] shoulder: the high-q
            # tail is forced ~0 by the Gaussian decay, so uniform MSE spends most
            # of the loss budget on trivially-zero points. Read q from the
            # (shuffled) inputs' last column so the weights stay aligned with the
            # shuffled vq_atoms (self.q is NOT shuffled).
            q_col  = self.inputs[:, -1:]
            mask   = (q_col >= 3.0) & (q_col <= 8.0)
            self.w = torch.where(mask, torch.tensor(4.0), torch.tensor(1.0))
        else:
            # val cost stays unweighted -> an honest, comparable MSE diagnostic.
            self.w = torch.ones_like(self.vq_atoms)

        self.len = self.inputs.shape[0]


    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        inputs = self.inputs[idx] # [3, 1]
        v_ref = self.vq_atoms[idx] # shape [1]
        w = self.w[idx] # shape [1]
        
        return inputs, v_ref, w 



def _plot_lsd_epoch(model, atom, epoch, num_epochs, Xv, Yv, val_loss, n_q, n_unique,
                    resultsFolder, SHOWPLOTS):
    """Diagnostic plots for one checkpoint. Guarded by _PLOT_LOCK because
    matplotlib's pyplot state is global (so this is safe under parallel atoms)."""
    with torch.no_grad():
        model.eval()
        pred = model(Xv)
    # cast to float64 on the host: training may be float32, but the FT helper and
    # plot routines mix in float64 constants.
    plot_q   = Xv[:, -1].detach().to('cpu', torch.float64)
    plot_tgt = Yv.detach().to('cpu', torch.float64)
    plot_prd = pred.detach().to('cpu', torch.float64)
    print(f"[{atom}] epoch [{epoch+1}/{num_epochs}], validation loss: {val_loss:.4g}")
    with _PLOT_LOCK:
        # Cap environments plotted per atom type: with the MACE backend every atom
        # is its own environment, so n_unique can be huge. Plot .pdf only (no .png)
        # to keep plotting fast.
        for n_u in range(min(n_unique, _MAX_PLOT_ENVS)):
            sl = slice(n_u * n_q, (n_u + 1) * n_q)
            fig = plotLSD(atom, plot_q[sl], plot_q[sl], plot_tgt[sl], plot_prd[sl],
                          "LSDcorr", f"NN_{epoch+1}", ["-", ":"], True, SHOWPLOTS)
            fig.savefig(f"{resultsFolder}LSD/initLSD_{atom}_epoch_{epoch+1}_plotPP_{n_u}.pdf")
            q_mags = torch.linspace(0.0, 80.0, 4096)
            ft_pot = qSpacePot_ft(plot_q[sl], plot_prd[sl], q_mags)
            outdata = torch.cat([q_mags.view(-1, 1), ft_pot.view(-1, 1)], dim=1).numpy()
            np.savetxt(f"{resultsFolder}LSD/initLSD_{atom}_epoch_{epoch+1}_plotPP_{n_u}.dat", outdata)
            plt.close(fig)


def _plot_lsd_results(model, atom, val_ds, training_cost, validation_cost,
                      resultsFolder, NNConfig):
    """Final LSD-fit diagnostics, run on the MAIN thread after training completes
    (so it is safe even when atoms were trained in parallel on separate GPUs).
    Writes, per unique environment, the fitted vs target correction (plotPP) and
    its real-space FT, plus the train/validation cost curve. The model is expected
    to be on CPU in float64 at this point (restored by init_LSD_train_GPU)."""
    with torch.no_grad():
        model.eval()
        Xv   = val_ds.inputs.detach().to('cpu', torch.float64)
        pred = model.to('cpu')(Xv)
    q   = Xv[:, -1]
    tgt = val_ds.vq_atoms.detach().to('cpu', torch.float64).view(-1, 1)
    n_q, n_unique = val_ds.n_q_grid, val_ds.n_unique

    # cap environments per atom type (see _MAX_PLOT_ENVS) and write .pdf only
    for n_u in range(min(n_unique, _MAX_PLOT_ENVS)):
        sl = slice(n_u * n_q, (n_u + 1) * n_q)
        fig = plotLSD(atom, q[sl], q[sl], tgt[sl], pred[sl], "LSDcorr", "final",
                      ["-", ":"], True, NNConfig['SHOWPLOTS'])
        fig.savefig(f"{resultsFolder}LSD/initLSD_{atom}_final_plotPP_{n_u}.pdf")
        q_mags = torch.linspace(0.0, 80.0, 4096)
        ft_pot = qSpacePot_ft(q[sl], pred[sl], q_mags)
        outdata = torch.cat([q_mags.view(-1, 1), ft_pot.view(-1, 1)], dim=1).numpy()
        np.savetxt(f"{resultsFolder}LSD/initLSD_{atom}_final_plotPP_{n_u}.dat", outdata)
        plt.close(fig)

    ep = list(range(len(training_cost)))
    fig_cost = plot_training_validation_cost(ep, training_cost, ep, validation_cost,
                                             ylogBoolean=False, SHOWPLOTS=NNConfig['SHOWPLOTS'])
    fig_cost.savefig(resultsFolder + f'init_{atom}_LSD_train_cost.pdf')
    plt.close(fig_cost)


def init_LSD_train_GPU(model, device, train_ds, val_ds, criterion, optimizer, scheduler,
                       NNConfig, atom, resultsFolder, make_plots=True):
    """Full-batch GPU training of one LSD correction net.

    Data piping: training is full-batch, so the entire (inputs, target, weight)
    tensors are moved to `device` ONCE and kept resident. There is no per-epoch
    host->device copy and no DataLoader/collation overhead. The hot loop runs
    entirely on `device` with set_to_none gradients and NO torch.cuda.empty_cache()
    (empty_cache forces a full device sync + allocator flush and was the dominant
    per-epoch stall in the old loop). cuda is emptied once, after training.

    Precision: training runs in `init_LSD_dtype` (default float32 -- ~2x A100
    throughput, TF32-eligible). The net is cast back to float64 before returning so
    the (float64) band-structure stage is unaffected.
    """
    train_dtype = _torch_dtype(NNConfig.get('init_LSD_dtype', 'float32'))
    if device.type == 'cuda':
        torch.cuda.set_device(device)   # make this (possibly threaded) context use `device`
    # Keep an exact float64 copy of N_ref. The float32 training cast below rounds it
    # (~1e-8); the descriptors fed at inference/plot time are full float64, so a
    # rounded N_ref no longer bit-matches the reference atom's descriptor and the
    # NN(x)-NN(x_ref) cancellation at the reference geometry stops being exactly 0.
    # Restored after casting back to float64 (below).
    n_ref_f64 = (model.N_ref.detach().to('cpu', torch.float64).clone()
                 if getattr(model, 'N_ref', None) is not None else None)
    model.to(device=device, dtype=train_dtype)   # casts params + buffers (N_ref, gaussian_std)

    # --- move the full dataset to the device a single time ------------------
    pin = (device.type == 'cuda')
    def _dev(t):
        # detach() is essential: the dataset inputs carry an autograd graph back to
        # atomPos (descriptors require grad). The tensors are resident and reused
        # every epoch, so without detaching, the first backward() frees that shared
        # graph and the next epoch errors ("backward through the graph a second
        # time"). LSD pretraining treats descriptors/q as constant inputs.
        t = t.detach().contiguous().to(dtype=train_dtype)   # cast on host: halves H2D bytes for fp32
        if pin:
            t = t.pin_memory()
        return t.to(device, non_blocking=pin)
    X,  Y,  W  = _dev(train_ds.inputs), _dev(train_ds.vq_atoms), _dev(train_ds.w)
    Xv, Yv, Wv = _dev(val_ds.inputs),   _dev(val_ds.vq_atoms),   _dev(val_ds.w)
    n_q, n_unique = train_ds.n_q_grid, train_ds.n_unique

    # input standardization: set the model's normalization buffers from the
    # training inputs so the sub-network sees ~unit-scale descriptors and q
    # (fixes the conditioning that made the net plateau). Buffers persist with
    # the model -> the same transform is applied at inference (ham). Toggle with
    # init_LSD_normalize (identity buffers => off, behaves like the raw-input net).
    if bool(NNConfig.get('init_LSD_normalize', True)):
        with torch.no_grad():
            std = X.std(dim=0)
            model.in_mean = X.mean(dim=0)
            if str(NNConfig.get('descriptor_backend', 'handcrafted')).lower() == 'mace':
                # MACE: per-dim standardization. Descriptor dims are tiny
                # (~1e-3..1e-2) vs q's huge std, so a single global floor would
                # leave them ~unnormalized -> normalize each by its own std. But
                # flooring only at 1e-8 amplifies near-dead dims (std << max) to
                # unit variance, injecting their jitter as full-scale noise. Floor
                # the descriptor dims at a fraction of the largest *descriptor* std
                # (q, the last dim, keeps its own std).
                in_std = std.clone()
                desc_std = std[:-1]
                floor = torch.clamp(0.1 * desc_std.max(), min=1e-8)
                in_std[:-1] = torch.clamp(desc_std, min=floor)
                model.in_std = in_std
            else:
                # handcrafted: floor std at a fraction of the largest so a
                # near-constant descriptor isn't amplified to unit variance.
                floor = torch.clamp(0.1 * std.max(), min=1e-8)
                model.in_std = torch.clamp(std, min=floor)

    num_epochs = NNConfig['init_LSD_num_epochs']
    plot_every = NNConfig['init_LSD_plot_every']
    sched_step = NNConfig['init_LSD_scheduler_step']

    training_cost_x, training_cost = [], []
    validation_cost_x, validation_cost = [], []
    trainCost_file = open(f"{resultsFolder}init_{atom}_train_cost.dat", "w")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(X), Y, W)
        loss.backward()
        optimizer.step()

        # cosine anneals every epoch; the legacy exponential steps every sched_step
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif epoch > 0 and epoch % sched_step == 0:
            scheduler.step()

        with torch.no_grad():
            model.eval()
            val_loss = criterion(model(Xv), Yv, Wv)

        # one tiny host sync per epoch (negligible for these small nets), needed
        # for the live cost log; kept out of the GPU compute path otherwise.
        train_c, val_c = loss.item(), val_loss.item()
        training_cost_x.append(epoch);   training_cost.append(train_c)
        validation_cost_x.append(epoch); validation_cost.append(val_c)
        trainCost_file.write(f"{epoch} {train_c:.6g}\n")

        if (epoch == 0) or ((epoch + 1) % plot_every == 0):
            print_and_inspect_NNParams(model, filename=f'{resultsFolder}LSD/initLSD_{atom}_epoch_{epoch+1}_params.dat', show=True)
            torch.save(model.state_dict(), f'{resultsFolder}LSD/initLSD_{atom}_epoch_{epoch+1}_{atom}_PPmodel.pth')
            trainCost_file.flush()
            if make_plots:
                _plot_lsd_epoch(model, atom, epoch, num_epochs, Xv, Yv, val_c,
                                n_q, n_unique, resultsFolder, NNConfig['SHOWPLOTS'])

    trainCost_file.close()

    # NOTE: final fit + cost plots are produced by _plot_lsd_results on the MAIN
    # thread after training (see init_LSD_PP), so they work under parallel atoms
    # too. Per-epoch convergence frames above stay gated by make_plots.

    # restore float64 so the band-structure stage (and on-disk model) stay double
    model.to(dtype=torch.float64)
    # restore the exact float64 N_ref (see above) so the reference-geometry
    # prediction cancels to exactly 0 at inference/plot time.
    if n_ref_f64 is not None:
        model.N_ref = n_ref_f64.to(next(model.parameters()).device)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return (training_cost, validation_cost)

def unique_within_tolerance(t, rtol=0.01):
    """
    Return indices of unique rows where rows within rtol of each other are merged.
    No sorting — order is preserved so G2 and G4 stay aligned.
    Returns a boolean mask over the input rows.
    """
    n = t.shape[0]
    keep = torch.ones(n, dtype=torch.bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            # Two rows are duplicates if both G2 and G4 are within rtol
            close = torch.all(
                torch.abs(t[i] - t[j]) <= rtol * torch.abs(t[i] + 1e-10)
            )
            if close:
                keep[j] = False
    return keep   # boolean mask, same length as input


def env_keep_mask(descriptors, backend, rtol=0.05):
    """Which environments to keep as 'unique'. Hand-crafted descriptors are
    deduplicated within a tolerance; MACE descriptors are high-dimensional and
    every atom is its own environment, so keep them all (the relative-tolerance
    dedup is ill-defined on 256-d embeddings)."""
    if str(backend).lower() == 'mace':
        return torch.ones(descriptors.shape[0], dtype=torch.bool)
    return unique_within_tolerance(descriptors, rtol=rtol)


def load_ref_potentials(inputsFolder, atom, iSys, n_unique):
    """
    Load all pot_q_{atom}_diff_{iSys}_*.par files and return stacked q and v_ref tensors.
    Files are matched by index to N_alphas[0], N_alphas[1], etc.
    """
    q_list, v_list = [], []
    for i in range(n_unique):
        fpath = f"{inputsFolder}pot_q_{atom}_diff_{iSys}_{i}.par"
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Expected {n_unique} reference files for atom '{atom}', system {iSys}, "
                f"but could not find: {fpath}"
            )
        data = torch.tensor(np.loadtxt(fpath))  # shape [NQGRID, 2]
        q_list.append(data[:, 0])
        v_list.append(data[:, 1])

    return q_list, v_list  # each a list of n_unique tensors of shape [NQGRID]


def init_LSD_PP(inputsFolder, LSDmodels, systems, atomPPOrder, NNConfig, resultsFolder, force_retrain=False):
    LSD_PPFunc_train = {}
    LSD_PPFunc_val   = {}

    for atom in atomPPOrder:
        q_all       = []
        v_ref_all   = []
        N_alphas_all = []
        n_unique_tot = 0

        for iSys, system in enumerate(systems):
            # Build per-atom descriptor matrix
            descriptors = system.env_descriptors[atom]    # (n_atoms_of_type, n_descr_of_type)
            
            # Unique environments (dedup for hand-crafted; all atoms for MACE)
            backend     = NNConfig.get('descriptor_backend', 'handcrafted')
            keep_mask   = env_keep_mask(descriptors, backend, rtol=0.05)
            N_alphas    = descriptors[keep_mask]        # (n_unique, n_descr)
            n_unique    = N_alphas.shape[0]

            # Load one reference file per unique descriptor pair
            q_list, v_list = load_ref_potentials(inputsFolder, atom, iSys, n_unique)
            n_q = q_list[0].shape[0]

            # Set reference values
            if iSys == 0:
                # Reference environment descriptor for this atom type, taken from the
                # cubic reference system (iSys==0). Every atom of a given type in the
                # cubic cell is symmetry-equivalent, so their (rotation-invariant)
                # descriptors are identical to round-off -> just take the first row as
                # THE reference; no averaging needed. Shape [1, n_descr]. detach() so
                # it never carries a descriptor autograd graph onto the GPU as a buffer.
                LSDmodels[atom].N_ref = N_alphas[0:1].detach().clone()
                # if atom == "Cs":
                #     LSDmodels[atom].N_ref = torch.tensor([0.0, 0.0, 0.0]).view(1, -1)
                # if atom == "Pb":
                #     LSDmodels[atom].N_ref = torch.tensor([0.978034, 0.0, 0.0, 0.0, 0.0]).view(1, -1)
                # if atom == "I":
                #     LSDmodels[atom].N_ref = torch.tensor([0.0, 0.0]).view(1, -1)
                # if atom == "Br":
                #     LSDmodels[atom].N_ref = torch.tensor([0.0, 0.0]).view(1, -1)
                    
            # Pair each unique descriptor row with its q and v_ref
            for row, q, v_ref in zip(N_alphas, q_list, v_list):
                # row: (2,) → repeat n_q times → (n_q, 2)
                row_repeated = row.unsqueeze(0).expand(n_q, -1)
                N_alphas_all.append(row_repeated)
                q_all.append(q)
                v_ref_all.append(v_ref)
            
            n_unique_tot += n_unique

        # Concatenate across all systems
        q_all        = torch.cat(q_all,        dim=0)   # (total_samples,)
        v_ref_all    = torch.cat(v_ref_all,    dim=0)   # (total_samples,)
        N_alphas_all = torch.cat(N_alphas_all, dim=0)   # (total_samples, 2)
        
        print(f"atom = {atom}, N_alphas_all unique rows:")
        for row in torch.unique(N_alphas_all, dim=0):
            print("    ".join(f"{i:.6f} " for i in row))

        LSD_PPFunc_train[atom] = init_LSD_data(N_alphas_all, q_all, v_ref_all, n_unique_tot, n_q, train=True)
        LSD_PPFunc_val[atom]   = init_LSD_data(N_alphas_all, q_all, v_ref_all, n_unique_tot, n_q, train=False)
        
    n_atoms_found = 0
    atoms_to_train = [atom for atom in atomPPOrder]
    for atom in atomPPOrder:
        if os.path.exists(inputsFolder + f"init_{atom}_LSDmodel.pth"):
            print(f"\n{'#' * 40}\nInitializing the LSD NN with file {inputsFolder}init_{atom}_LSDmodel.pth.")
            LSDmodels[atom].load_state_dict(torch.load(inputsFolder + f"init_{atom}_LSDmodel.pth"), strict=False)
            # LSDmodels[atom].N_ref = LSDmodels[atom].N_ref.view(-1, systems[0].n_descr[atom])
            n_atoms_found += 1
            atoms_to_train.remove(atom)
    if (n_atoms_found == len(atomPPOrder)):
        print(f"\nAll LSDmodels found in input directory!")
        if (force_retrain == False):
            return LSDmodels, LSD_PPFunc_val
        else:
            atoms_to_train = [atom for atom in atomPPOrder]
            print(f"\ninit_LSD_force_retrain turned on. Retraining all LSD models to refine fit.")
    else:
        print(f"\nOnly {n_atoms_found}/{len(atomPPOrder)} LSDmodels found for atoms in the system. Retraining {atoms_to_train} models.")
        

    if ('init_LSD_num_epochs' not in NNConfig) or (NNConfig['init_LSD_num_epochs']==0): 
        print("\nWARNING: Not initializing the LSD NN corrections. Could lead to slow convergence of LSD algorithm. \n")
        return LSDmodels, LSD_PPFunc_val

    print(f"\n{'#' * 40}\nInitializing the LSD NNs by training to the pseudopotential differences. ")

    # Device selection. LSD pre-training runs on GPU (if available) independently
    # of the band-structure path, which stays on CPU. With several GPUs
    # and init_LSD_parallel_atoms on, the atom-type nets train concurrently, one
    # per GPU (each is independent: its own data, model, optimizer, output files).
    lsd_device = _select_lsd_device(NNConfig)
    n_cuda     = torch.cuda.device_count() if torch.cuda.is_available() else 0
    parallel   = (bool(NNConfig.get('init_LSD_parallel_atoms', False))
                  and lsd_device.type == 'cuda' and n_cuda > 1)
    print(f"LSD init device = {lsd_device} | visible CUDA devices = {n_cuda} | "
          f"parallel_atoms = {parallel}")

    cost_hist = {}   # atom -> (training_cost, validation_cost), for the final plots

    def _train_one(atom, dev):
        print(f"Fitting atom type {atom} on {dev}")
        model = LSDmodels[atom]
        model.to(dev)                   # move params/buffers (incl. N_ref) before optimizer
        model.eval()
        criterion = init_Zunger_weighted_mse
        lr0 = NNConfig['init_LSD_optimizer_lr']
        optimizer = torch.optim.Adam(model.parameters(), lr=lr0)
        if str(NNConfig.get('init_LSD_scheduler', 'cosine')).lower() == 'cosine':
            # smooth decay lr0 -> lr0*eta_min_frac over training; settles the
            # plateau the old "0.9 every 1000 epochs" exponential left high.
            eta_min = lr0 * float(NNConfig.get('init_LSD_eta_min_frac', 1e-3))
            scheduler = CosineAnnealingLR(optimizer, T_max=NNConfig['init_LSD_num_epochs'],
                                          eta_min=eta_min)
        else:
            scheduler = ExponentialLR(optimizer, gamma=NNConfig['init_LSD_scheduler_gamma'])

        t0 = time.time()
        tc, vc = init_LSD_train_GPU(model, dev, LSD_PPFunc_train[atom], LSD_PPFunc_val[atom],
                                    criterion, optimizer, scheduler, NNConfig, atom, resultsFolder,
                                    make_plots=not parallel)
        cost_hist[atom] = (tc, vc)
        print(f"[{atom}] init elapsed: {time.time() - t0:.2f} s")

        # band-structure stage runs on CPU -> move the trained net back.
        model.to('cpu')
        torch.save(model.state_dict(), resultsFolder + f"init_{atom}_LSDmodel.pth")
        LSD_PPFunc_val[atom] = init_LSD_data(
            LSD_PPFunc_val[atom].N_alphas, LSD_PPFunc_val[atom].q,
            LSD_PPFunc_val[atom].vq_atoms, LSD_PPFunc_val[atom].n_unique,
            LSD_PPFunc_val[atom].n_q_grid, train=False)

    if parallel:
        threads = []
        for k, atom in enumerate(atoms_to_train):
            dev = torch.device(f"cuda:{k % n_cuda}")
            th = threading.Thread(target=_train_one, args=(atom, dev), name=f"LSD-{atom}")
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        print("NOTE: per-epoch convergence frames are skipped in parallel mode "
              "(matplotlib is not thread-safe); final fit plots are made below.")
    else:
        for atom in atoms_to_train:
            _train_one(atom, lsd_device)

    # Final LSD-fit plots on the MAIN thread -> safe in both sequential and
    # parallel modes. Models are on CPU/float64 here.
    print("Plotting final LSD fits.")
    for atom in atoms_to_train:
        tc, vc = cost_hist[atom]
        _plot_lsd_results(LSDmodels[atom], atom, LSD_PPFunc_val[atom], tc, vc,
                          resultsFolder, NNConfig)

    print("Done with LSD NN initialization.")
    return LSDmodels, LSD_PPFunc_val



