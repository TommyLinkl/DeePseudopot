import os, time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

from .constants import *
from .pp_func import pot_func, plotPP, plot_training_validation_cost
from .NN_train import print_and_inspect_gradients, print_and_inspect_NNParams

torch.set_default_dtype(torch.float64)

class init_Zunger_data(Dataset):
    def __init__(self, atomPPOrder, totalParams, train=True, trainDataSize=4000, valDataSize=NQGRID):
        if train==True:
            self.q = (torch.rand(trainDataSize, 1) * 10.0).view(-1,1)
            self.q[0:10, 0] = 0.0
            
            self.vq_atoms = torch.empty(trainDataSize, 0)
            for iAtom in range(len(atomPPOrder)): 
                vq = pot_func(self.q, totalParams[iAtom])
                self.vq_atoms = torch.cat((self.vq_atoms, vq), dim=1)
            mask = (self.q >= 0) & (self.q <= 1)
            w = torch.where(mask, torch.tensor(10.0), torch.tensor(1.0))
            self.w = torch.cat([w] * len(atomPPOrder), dim=1)
        elif train==False:
            self.q = torch.linspace(0.0, 10.0, valDataSize).view(-1,1)
            self.vq_atoms = torch.empty(valDataSize, 0)
            for iAtom in range(len(atomPPOrder)): 
                vq = pot_func(self.q, totalParams[iAtom])
                self.vq_atoms = torch.cat((self.vq_atoms, vq), dim=1)
            self.w = torch.ones_like(self.vq_atoms)
        self.len = self.q.shape[0]


    def __getitem__(self,index):
        return self.q[index],self.vq_atoms[index],self.w[index]
    

    def __len__(self):
        return self.len


class load_qSpace_data(Dataset):
    def __init__(self, atomPPOrder, filename):
        qSpace_file = np.loadtxt(filename)
        with open(filename, 'r') as file:
            comment_line = file.readline().strip()

        print(f"Please make sure the order of the atoms are the same in the following: {atomPPOrder} vs. {comment_line.split()}")


        self.q = torch.from_numpy(qSpace_file[:, 0]).view(-1,1)
        self.vq_atoms = torch.empty(len(qSpace_file), 0)
        for iAtom in range(len(atomPPOrder)): 
            vq = torch.from_numpy(qSpace_file[:, iAtom+1]).view(-1,1)
            self.vq_atoms = torch.cat((self.vq_atoms, vq), dim=1)
        self.w = torch.ones_like(self.vq_atoms)

        self.len = self.q.shape[0]


    def __getitem__(self,index):
        return self.q[index],self.vq_atoms[index],self.w[index]
    

    def __len__(self):
        return self.len


def init_Zunger_weighted_mse(yhat,y,weight):
    return torch.mean(weight*(yhat-y)**2)


def init_Zunger_train_GPU(model, device, train_loader, val_loader, criterion, optimizer, scheduler, NNConfig, atomPPOrder, resultsFolder):
    training_cost_x=[]
    training_cost=[]
    validation_cost_x=[]
    validation_cost=[]
    model.to(device)
    for epoch in range(NNConfig['init_Zunger_num_epochs']):
        train_cost = 0
        val_cost = 0
        for q, vq_atoms, w in train_loader:
            model.train()
            q = q.to(device) 
            vq_atoms = vq_atoms.to(device)
            w = w.to(device)
            
            outputs = model(q)
            loss = criterion(outputs, vq_atoms, w)
            train_cost += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        training_cost_x.append(epoch)
        training_cost.append(train_cost)
        if (epoch==0) or ((epoch + 1) % NNConfig['init_Zunger_plotEvery'] == 0):
            # print_and_inspect_gradients(model, filename=f'{resultsFolder}initZunger_epoch_{epoch+1}_gradients.dat', show=True)
            print_and_inspect_NNParams(model, filename=f'{resultsFolder}initZunger_epoch_{epoch+1}_params.dat', show=True)
            torch.save(model.state_dict(), f'{resultsFolder}initZunger_epoch_{epoch+1}_PPmodel.pth')
            torch.cuda.empty_cache()
        if epoch > 0 and epoch % NNConfig['init_Zunger_plotEvery'] == 0:
            scheduler.step()

        for q, vq_atoms, w in val_loader:
            model.eval()
            q = q.to(device) 
            vq_atoms = vq_atoms.to(device)
            w = w.to(device)
            
            pred_outputs = model(q)
            loss = criterion(pred_outputs, vq_atoms, w)
            val_cost += loss.item()
            if (epoch + 1) % NNConfig['init_Zunger_plotEvery'] == 0:
                plot_q = q.cpu()
                plot_vq_atoms = vq_atoms.cpu()
                plot_pred_outputs = pred_outputs.cpu()
                print(f"Epoch [{epoch+1}/{NNConfig['init_Zunger_num_epochs']}], Validation Loss: {loss.item():.4f}")
                fig = plotPP(
                    atomPPOrder,
                    plot_q,
                    plot_q,
                    plot_vq_atoms,
                    plot_pred_outputs,
                    "ZungerForm",
                    f"NN_{epoch+1}",
                    ["-",":" ]*len(atomPPOrder),
                    True,
                    NNConfig['SHOWPLOTS'],
                    ref_component="analytic local (no LR tail)",
                    pred_component="NN_loc (no LR tail)"
                )
                fig.savefig(f"{resultsFolder}initZunger_epoch_{epoch}_plotPP.pdf")
                fig.savefig(f"{resultsFolder}initZunger_epoch_{epoch}_plotPP.png")
        validation_cost_x.append(epoch)
        validation_cost.append(val_cost)
        torch.cuda.empty_cache()
    fig_cost = plot_training_validation_cost(training_cost_x, training_cost, validation_cost_x, validation_cost, ylogBoolean=False, SHOWPLOTS=NNConfig['SHOWPLOTS'])
    fig_cost.savefig(resultsFolder + 'init_train_cost.pdf')
    torch.cuda.empty_cache()
    return (training_cost, validation_cost)


def init_ZungerPP(inputsFolder, PPmodel, atomPPOrder, localPotParams, nPseudopot, NNConfig, device, resultsFolder, force_retrain=False, lr_params=None, pp_params=None, lr_gamma=0.2):
    """
    Initializes the neural network pseudopotentials by either
    1. getting the NN parameters from {inputsFolder}init_PPmodel.pth
    OR
    2. trainining to the existing Zunger function form pseudopotential. 
    """
    ZungerPPFunc_train = init_Zunger_data(atomPPOrder, localPotParams, train=True)
    ZungerPPFunc_val = init_Zunger_data(atomPPOrder, localPotParams, train=False)
    if os.path.exists(inputsFolder + 'init_qSpace_pot.par'):
        print("Reading in init_qSpace_pot.par for initialization. Not using the Zunger function for this purpose. ")
        del ZungerPPFunc_train, ZungerPPFunc_val
        ZungerPPFunc_train = load_qSpace_data(atomPPOrder, f"{inputsFolder}init_qSpace_pot.par")
        ZungerPPFunc_val = load_qSpace_data(atomPPOrder, f"{inputsFolder}init_qSpace_pot.par")


    if os.path.exists(inputsFolder + 'init_PPmodel.pth'):
        print(f"\n{'#' * 40}\nInitializing the NN with file {inputsFolder}init_PPmodel.pth.")
        PPmodel.load_state_dict(torch.load(inputsFolder + 'init_PPmodel.pth'))
        if not force_retrain: 
            print(f"Done with NN initialization to the file {inputsFolder}init_PPmodel.pth.")
            return PPmodel, ZungerPPFunc_val
    
    if ('init_Zunger_num_epochs' not in NNConfig) or (NNConfig['init_Zunger_num_epochs']==0): 
        print("\nWARNING: Not initializing the NN to the Zunger function form. Could lead to slow convergence. \n")
        return PPmodel, ZungerPPFunc_val

    print(f"\n{'#' * 40}\nInitializing the NN by training to the Zunger function form of pseudopotentials. ")
    PPmodel.cpu()
    PPmodel.eval()
    NN_init = PPmodel(ZungerPPFunc_val.q)
    plotPP(
        atomPPOrder,
        ZungerPPFunc_val.q,
        ZungerPPFunc_val.q,
        ZungerPPFunc_val.vq_atoms,
        NN_init,
        "ZungerForm",
        "NN_init",
        ["-",":" ]*nPseudopot,
        False,
        NNConfig['SHOWPLOTS'],
        ref_component="analytic local (no LR tail)",
        pred_component="NN_loc (no LR tail)",
        lr_params=lr_params,
        pp_params=pp_params,
        lr_gamma=lr_gamma
    )

    init_Zunger_criterion = init_Zunger_weighted_mse

    if ('init_Zunger_optimizer' not in NNConfig) or (NNConfig['init_Zunger_optimizer']=='adam'): 
        init_Zunger_optimizer = torch.optim.Adam(PPmodel.parameters(), lr=NNConfig['init_Zunger_optimizer_lr'])
    elif NNConfig['init_Zunger_optimizer']=='sgd': 
        init_Zunger_optimizer = torch.optim.SGD(PPmodel.parameters(), lr=NNConfig['init_Zunger_optimizer_lr'])
    else: 
        raise ValueError("We only support 'adam' and 'sgd' for the init_Zunger fitting. ")

    init_Zunger_optimizer = torch.optim.Adam(PPmodel.parameters(), lr=NNConfig['init_Zunger_optimizer_lr'])
    init_Zunger_scheduler = ExponentialLR(init_Zunger_optimizer, gamma=NNConfig['init_Zunger_scheduler_gamma'])
    # trainloader = DataLoader(dataset = ZungerPPFunc_train, batch_size = int(ZungerPPFunc_train.len/4),shuffle=True)
    trainloader = DataLoader(dataset = ZungerPPFunc_train, batch_size = ZungerPPFunc_train.len, shuffle=False)
    validationloader = DataLoader(dataset = ZungerPPFunc_val, batch_size =ZungerPPFunc_val.len, shuffle=False)

    start_time = time.time()
    (training_cost, validation_cost) = init_Zunger_train_GPU(PPmodel, device, trainloader, validationloader, init_Zunger_criterion, init_Zunger_optimizer, init_Zunger_scheduler, NNConfig, atomPPOrder, resultsFolder)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Initialization elapsed time: %.2f seconds" % elapsed_time)

    torch.save(PPmodel.state_dict(), resultsFolder + 'initZunger_PPmodel.pth')

    print("Done with NN initialization to the latest function form.")

    ZungerPPFunc_val = init_Zunger_data(atomPPOrder, localPotParams, train=False)
    return PPmodel, ZungerPPFunc_val


def init_optimizer(inputsFolder, model, NNConfig, lr_params=None):
    base_lr = NNConfig['optimizer_lr']
    lr_param_list = []
    if lr_params is not None:
        lr_param_list = list(lr_params.parameters())

    param_groups = [{'params': list(model.parameters()), 'lr': base_lr}]
    if lr_param_list:
        long_range_lr = NNConfig.get('longRange_lr', base_lr / 10.0)
        param_groups.append({'params': lr_param_list, 'lr': long_range_lr})
    else:
        long_range_lr = None

    optimizer_name = NNConfig.get('optimizer', 'adam')
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=base_lr)
        if os.path.exists(inputsFolder + 'init_AdamState.pth'):
            print(f"Reading in the stored Adam optimizer 1st and 2nd momentum from {inputsFolder}init_AdamState.pth to initialize the optimizer.")
            try:
                optimizer.load_state_dict(torch.load(inputsFolder + 'init_AdamState.pth'))
            except ValueError:
                print("Stored Adam state is incompatible with current parameter groups. Skipping state load.")
            print(f"We are also re-setting the learning rates as specified in the NN_config.par file. ")
        if 'adam_beta1' in NNConfig:
            optimizer.beta1 = NNConfig['adam_beta1']
            print(f"Setting the Adam beta1 as {NNConfig['adam_beta1']}. ")
        if 'adam_beta2' in NNConfig:
            optimizer.beta2 = NNConfig['adam_beta2']
            print(f"Setting the Adam beta2 as {NNConfig['adam_beta2']}. ")
    elif optimizer_name == 'sgd':
        momentum = NNConfig.get('sgd_momentum', 0.0)
        optimizer = torch.optim.SGD(param_groups, lr=base_lr, momentum=momentum)
        if 'sgd_momentum' in NNConfig:
            print(f"Setting the SGD momentum as {NNConfig['sgd_momentum']}. ")
    elif optimizer_name == 'asgd':
        optimizer = torch.optim.ASGD(param_groups, lr=base_lr)
    elif optimizer_name == 'lbfgs':
        optimizer = torch.optim.LBFGS(param_groups, lr=base_lr)
    elif optimizer_name == 'adadelta':
        optimizer = torch.optim.Adadelta(param_groups, lr=base_lr)
    elif optimizer_name == 'adagrad':
        optimizer = torch.optim.Adagrad(param_groups, lr=base_lr)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr)
    elif optimizer_name == 'sparseadam':
        optimizer = torch.optim.SparseAdam(param_groups, lr=base_lr)
    elif optimizer_name == 'adamax':
        optimizer = torch.optim.Adamax(param_groups, lr=base_lr)
    elif optimizer_name == 'nadam':
        optimizer = torch.optim.NAdam(param_groups, lr=base_lr)
    elif optimizer_name == 'radam':
        optimizer = torch.optim.RAdam(param_groups, lr=base_lr)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_groups, lr=base_lr)
    else:
        raise ValueError("We don't support the optimizer you provided in the band structure fitting. ")

    if os.path.exists(inputsFolder + 'init_AdamState.pth') and optimizer_name == 'adam':
        # Ensure updated learning rates even if state was loaded successfully
        for idx, param_group in enumerate(optimizer.param_groups):
            if idx == 0:
                param_group['lr'] = base_lr
            elif long_range_lr is not None:
                param_group['lr'] = long_range_lr
    else:
        if long_range_lr is not None:
            optimizer.param_groups[0]['lr'] = base_lr
            optimizer.param_groups[1]['lr'] = long_range_lr

    return optimizer
