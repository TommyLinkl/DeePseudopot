import os, time
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import matplotlib.pyplot as plt
import re

from .constants import *
from .pp_func import pot_funcLSD, plotLSD, plot_training_validation_cost
from .NN_train import print_and_inspect_gradients, print_and_inspect_NNParams
from .init_NN_train import init_Zunger_weighted_mse

torch.set_default_dtype(torch.float64)

class init_LSD_data(Dataset):
    def __init__(self, N_alphas, q, v_ref, n_unique, n_q, train=True):
        """
        Custom dataset for neural network pseudopotential training.

        Args:
            N_file (str): Path to numpy file with N_alpha values, shape [n_samples].
            v_ref_file (str): Path to numpy file with v_ref(q) values, shape [n_samples, n_q].
            transform (callable, optional): Optional transform to apply to N_alpha or v_ref.
        """
        self.n_q_grid = n_q
        self.n_unique = n_unique
        self.N_alphas = N_alphas.reshape(-1, 1) # [n_unique * n_q_grid, 1]
        self.q = q.reshape(-1, 1) # [n_unique * n_q_grid, 1]
        self.vq_atoms = v_ref.reshape(-1, 1) # [n_unique * n_q_grid, 1]
        
        self.inputs = torch.cat((self.N_alphas, self.q), dim=1) # [n_unique * n_q_grid, 2]
        
        if train == True:
            # Generate a random permutation of indices
            rand_indices = torch.randperm(self.inputs.shape[0])   # shape [n_unique * n_q_grid]
            self.inputs = self.inputs[rand_indices]
            self.vq_atoms = self.vq_atoms[rand_indices]
            mask = (self.q >= 0.0) & (self.q <= 8.0)
            self.w = torch.where(mask, torch.tensor(10.0), torch.tensor(1.0))
            self.w = torch.ones_like(self.vq_atoms)
        if train == False:
            self.w = torch.ones_like(self.vq_atoms)

        self.w = torch.ones_like(self.vq_atoms)
        self.len = self.inputs.shape[0]


    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        inputs = self.inputs[idx] # [2, 1]
        v_ref = self.vq_atoms[idx] # shape [1]
        w = self.w[idx] # shape [1]
        
        return inputs, v_ref, w 



def init_LSD_train_GPU(model, device, train_loader, val_loader, criterion, optimizer, scheduler, NNConfig, atom, resultsFolder):
    training_cost_x=[]
    training_cost=[]
    validation_cost_x=[]
    validation_cost=[]
    model.to(device)
    lambda_ = 0.01
    trainCost_file = open(f"{resultsFolder}/init_{atom}_train_cost.dat", "w")
    
    for epoch in range(NNConfig['init_LSD_num_epochs']):
        train_cost = 0
        train_bc_cost = 0
        val_cost = 0
        for inputs, vq_atoms, w in train_loader:
            model.train()
            optimizer.zero_grad()
            inputs = inputs.to(device) 
            vq_atoms = vq_atoms.to(device)
            w = w.to(device)
            vq_pred = model(inputs)
            loss = criterion(vq_pred, vq_atoms, w)
            
            # BC loss - penalizes f(0, q) being nonzero
            # x_ref = torch.zeros_like(inputs)
            # x_ref[:, 1] = inputs[:, 1].clone()
            # bc_loss = (model.neural_network(x_ref) ** 2).mean()
            
            # loss += lambda_ * bc_loss
            train_cost += loss.item()
            loss.backward()
            optimizer.step()

        training_cost_x.append(epoch)
        training_cost.append(train_cost)
        trainCost_file.write(f"{epoch} {train_cost:.6g}\n")
        if (epoch==0) or ((epoch + 1) % NNConfig['init_LSD_plot_every'] == 0):
            #print_and_inspect_gradients(model, filename=f'{resultsFolder}initZunger_epoch_{epoch+1}_gradients.dat', show=True)
            print_and_inspect_NNParams(model, filename=f'{resultsFolder}LSD/initLSD_{atom}_epoch_{epoch+1}_params.dat', show=True)
            torch.save(model.state_dict(), f'{resultsFolder}LSD/initLSD_{atom}_epoch_{epoch+1}_{atom}_PPmodel.pth')
            torch.cuda.empty_cache()
            trainCost_file.flush()
        if epoch > 0 and epoch % NNConfig['init_LSD_scheduler_step'] == 0:
            scheduler.step()

        for inputs, vq_atoms, w in val_loader:
            model.eval()
            inputs = inputs.to(device) 
            vq_atoms = vq_atoms.to(device)
            w = w.to(device)
            
            vq_pred = model(inputs)
            
            loss = criterion(vq_pred, vq_atoms, w)
            val_cost += loss.item()
        if (epoch + 1) % NNConfig['init_LSD_plot_every'] == 0:
            plot_q = inputs[:, 1].cpu()
            plot_vq_atoms = vq_atoms.cpu()
            plot_pred_outputs = vq_pred.cpu()
            print(f"Epoch [{epoch+1}/{NNConfig['init_LSD_num_epochs']}], Validation Loss: {loss.item():.4f}")
            n_q = train_loader.dataset.n_q_grid
            for n_u in range(train_loader.dataset.n_unique):
                fig = plotLSD(atom, plot_q[n_u*n_q:(n_u+1)*n_q], plot_q[n_u*n_q:(n_u+1)*n_q], plot_vq_atoms[n_u*n_q:(n_u+1)*n_q], plot_pred_outputs[n_u*n_q:(n_u+1)*n_q], "LSDcorr", f"NN_{epoch+1}", ["-",":" ], True, NNConfig['SHOWPLOTS'])
                fig.savefig(f"{resultsFolder}LSD/initLSD_{atom}_epoch_{epoch+1}_plotPP_{n_u}.pdf")
                fig.savefig(f"{resultsFolder}LSD/initLSD_{atom}_epoch_{epoch+1}_plotPP_{n_u}.png")
            plt.close()
        validation_cost_x.append(epoch)
        validation_cost.append(val_cost)
        torch.cuda.empty_cache()

    fig_cost = plot_training_validation_cost(training_cost_x, training_cost, validation_cost_x, validation_cost, ylogBoolean=False, SHOWPLOTS=NNConfig['SHOWPLOTS'])
    fig_cost.savefig(resultsFolder + f'init_{atom}_LSD_train_cost.pdf')
    torch.cuda.empty_cache()
    trainCost_file.close()
    return (training_cost, validation_cost)


def init_LSD_PP(inputsFolder, LSDmodels, systems, atomPPOrder, NNConfig, device, resultsFolder, force_retrain=False):
    """
    Initializes the neural network for local structure dependent pseudopotential corrections by either
    1. getting the NN parameters from {inputsFolder}init_LSD_PPmodel.pth
    OR
    2. trainining to the existing pseudopotential correction data. 
    """
    LSD_PPFunc_train = {}
    LSD_PPFunc_val = {}

    for atom in atomPPOrder:
        q_all = []
        v_ref_all = []
        N_alphas_all = []
        n_unique_tot = 0
        for iSys, system in enumerate(systems):
            # Match atom type
            atypeIdx = np.where(atom == system.atomTypes)[0]

            # Load reference potential
            v_ref_atom = torch.tensor(np.loadtxt(f"{inputsFolder}pot_q_{atom}_diff_{iSys}.par"))  # shape [NQGRID, 2]
            q = v_ref_atom[:, 0]
            v_ref = v_ref_atom[:, 1]

            # Zero out first system for reference
            if iSys == 0:
                v_ref = torch.zeros_like(v_ref)

            # Get symmetry descriptors, N_alphas, for this atom
            N_alphas = torch.unique(system.localSymmDescr['G2'][atypeIdx])
            N_alphas = torch.unique(system.G2[atypeIdx])
            
            # Reshape data to include all relevant input->output pairs
            n_q = q.shape[0]
            n_unique = N_alphas.shape[0]
            N_alphas = N_alphas.repeat_interleave(n_q)
            q = q.repeat(n_unique)
            v_ref = v_ref.repeat(n_unique)

            # Collect per-system tensors
            q_all.append(q)
            v_ref_all.append(v_ref)
            N_alphas_all.append(N_alphas)
            n_unique_tot += n_unique

        # Concatenate all systems’ data (idiomatic PyTorch)
        q_all = torch.cat(q_all, dim=0)
        v_ref_all = torch.cat(v_ref_all, dim=0)
        N_alphas_all = torch.cat(N_alphas_all, dim=0)
        print(f"atom = {atom} N_alphas_all = \n")
        for elem in torch.unique(N_alphas_all):
            print(f"{elem:.8f}")
        # Create datasets
        LSD_PPFunc_train[atom] = init_LSD_data(N_alphas_all, q_all, v_ref_all, n_unique_tot, n_q, train=True)
        LSD_PPFunc_val[atom] = init_LSD_data(N_alphas_all, q_all, v_ref_all, n_unique_tot, n_q, train=False)

    n_atoms_found = 0
    atoms_to_train = [atom for atom in atomPPOrder]
    for atom in atomPPOrder:
        if os.path.exists(inputsFolder + f"init_{atom}_LSDmodel.pth"):
            print(f"\n{'#' * 40}\nInitializing the LSD NN with file {inputsFolder}init_{atom}_LSDmodel.pth.")
            LSDmodels[atom].load_state_dict(torch.load(inputsFolder + f"init_{atom}_LSDmodel.pth"))
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
    atoms_to_train = ["Pb", "I", "Cs"]
    for atom in atoms_to_train:
        print(f"Fitting atom type {atom}\n")
        LSDmodels[atom].cpu()
        LSDmodels[atom].eval()
        # NN_init = LSDmodels[atom](LSD_PPFunc_val[atom].N_alphas)
        # plotPP([atom], LSD_PPFunc_val[atom].q, LSD_PPFunc_val[atom].q, LSD_PPFunc_val[atom].vq_atoms, NN_init, "LSD_corr", "NN_init", ["-",":" ], False, NNConfig['SHOWPLOTS'])

        init_LSD_criterion = init_Zunger_weighted_mse

        init_LSD_optimizer = torch.optim.Adam(LSDmodels[atom].parameters(), lr=NNConfig['init_LSD_optimizer_lr'])
        init_LSD_scheduler = ExponentialLR(init_LSD_optimizer, gamma=NNConfig['init_LSD_scheduler_gamma'])
        # trainloader = DataLoader(dataset = ZungerPPFunc_train, batch_size = int(ZungerPPFunc_train.len/4),shuffle=True)
        trainloader = DataLoader(dataset = LSD_PPFunc_train[atom], batch_size=LSD_PPFunc_train[atom].len, shuffle=False)
        validationloader = DataLoader(dataset = LSD_PPFunc_val[atom], batch_size=LSD_PPFunc_val[atom].len, shuffle=False)

        start_time = time.time()
        (training_cost, validation_cost) = init_LSD_train_GPU(LSDmodels[atom], device, trainloader, validationloader, init_LSD_criterion, init_LSD_optimizer, init_LSD_scheduler, NNConfig, atom, resultsFolder)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Initialization elapsed time: %.2f seconds" % elapsed_time)

        torch.save(LSDmodels[atom].state_dict(), resultsFolder + f"init_{atom}_LSDmodel.pth")

        print("Done with NN initialization to the latest function form.")

        LSD_PPFunc_val[atom] = init_LSD_data(LSD_PPFunc_val[atom].N_alphas, LSD_PPFunc_val[atom].q, LSD_PPFunc_val[atom].vq_atoms, LSD_PPFunc_val[atom].n_unique, LSD_PPFunc_val[atom].n_q_grid, train=False)

    return LSDmodels, LSD_PPFunc_val



