import os, time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from .constants import *
from .pp_func import pot_func, plotPP, plot_training_validation_cost


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


def init_Zunger_weighted_mse(yhat,y,weight):
    return torch.mean(weight*(yhat-y)**2)


def init_Zunger_train_GPU(model, device, train_loader, val_loader, criterion, optimizer, scheduler, scheduler_step, epochs, plot_every, atomPPOrder, SHOWPLOTS, resultsFolder):
    training_cost=[]
    validation_cost=[]
    model.to(device)
    for epoch in range(epochs):
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
        training_cost.append(train_cost)
        if epoch > 0 and epoch % scheduler_step == 0:
            scheduler.step()

        for q, vq_atoms, w in val_loader:
            model.eval()
            q = q.to(device) 
            vq_atoms = vq_atoms.to(device)
            w = w.to(device)
            
            pred_outputs = model(q)
            loss = criterion(pred_outputs, vq_atoms, w)
            val_cost += loss.item()
            if (epoch + 1) % plot_every == 0:
                plot_q = q.cpu()
                plot_vq_atoms = vq_atoms.cpu()
                plot_pred_outputs = pred_outputs.cpu()
                print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {loss.item():.4f}')
                plotPP(atomPPOrder, plot_q, plot_q, plot_vq_atoms, plot_pred_outputs, "ZungerForm", f"NN_{epoch+1}", ["-",":" ]*len(atomPPOrder), True, SHOWPLOTS)
        validation_cost.append(val_cost)
        torch.cuda.empty_cache()
    fig_cost = plot_training_validation_cost(training_cost, validation_cost, True, SHOWPLOTS)
    fig_cost.savefig(resultsFolder + 'init_train_cost.png')
    torch.cuda.empty_cache()
    return (training_cost, validation_cost)


def init_ZungerPP(inputsFolder, PPmodel, atomPPOrder, localPotParams, nPseudopot, NNConfig, device, resultsFolder):
    """
    Initializes the neural network pseudopotentials by 
    1. getting the NN parameters from {inputsFolder}init_PPmodel.pth
    OR
    2. trainining to the existing Zunger function form pseudopotential. 
    """
    ZungerPPFunc_train = init_Zunger_data(atomPPOrder, localPotParams, train=True)
    ZungerPPFunc_val = init_Zunger_data(atomPPOrder, localPotParams, train=False)

    if os.path.exists(inputsFolder + 'init_PPmodel.pth'):
        print(f"\n{'#' * 40}\nInitializing the NN with file {inputsFolder}init_PPmodel.pth.")
        PPmodel.load_state_dict(torch.load(inputsFolder + 'init_PPmodel.pth'))
        print(f"Done with NN initialization to the file {inputsFolder}init_PPmodel.pth.")
    else:
        print(f"\n{'#' * 40}\nInitializing the NN by training to the Zunger function form of pseudopotentials. ")
        PPmodel.cpu()
        PPmodel.eval()
        NN_init = PPmodel(ZungerPPFunc_val.q)
        plotPP(atomPPOrder, ZungerPPFunc_val.q, ZungerPPFunc_val.q, ZungerPPFunc_val.vq_atoms, NN_init, "ZungerForm", "NN_init", ["-",":" ]*nPseudopot, False, NNConfig['SHOWPLOTS'])

        init_Zunger_criterion = init_Zunger_weighted_mse
        init_Zunger_optimizer = torch.optim.Adam(PPmodel.parameters(), lr=NNConfig['init_Zunger_optimizer_lr'])
        init_Zunger_scheduler = ExponentialLR(init_Zunger_optimizer, gamma=NNConfig['init_Zunger_scheduler_gamma'])
        trainloader = DataLoader(dataset = ZungerPPFunc_train, batch_size = int(ZungerPPFunc_train.len/4),shuffle=True)
        validationloader = DataLoader(dataset = ZungerPPFunc_val, batch_size =ZungerPPFunc_val.len, shuffle=False)

        start_time = time.time()
        init_Zunger_train_GPU(PPmodel, device, trainloader, validationloader, init_Zunger_criterion, init_Zunger_optimizer, init_Zunger_scheduler, 20, NNConfig['init_Zunger_num_epochs'], NNConfig['init_Zunger_plotEvery'], atomPPOrder, NNConfig['SHOWPLOTS'], resultsFolder)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Initialization elapsed time: %.2f seconds" % elapsed_time)

        torch.save(PPmodel.state_dict(), resultsFolder + 'initZunger_PPmodel.pth')

        print("Done with NN initialization to the latest function form.")

    return PPmodel, ZungerPPFunc_val

