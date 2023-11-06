import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.rcParams['lines.markersize'] = 3
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import ExponentialLR

from utils.pp_func import *
torch.set_default_dtype(torch.float32)
torch.manual_seed(24)

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


# Initialize the NN parameters to fit the current Zunger form
def init_Zunger_weighted_mse(yhat,y,weight):
    return torch.mean(weight*(yhat-y)**2)
#criterion=nn.MSELoss()

def init_Zunger_train_GPU(model, device, train_loader, val_loader, criterion, optimizer, scheduler, scheduler_step, epochs, plot_every, atomPPOrder, SHOWPLOTS):
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
    fig_cost = plot_training_validation_cost(training_cost, validation_cost, True, SHOWPLOTS)
    fig_cost.savefig('results/init_train_cost.png')
    return (training_cost, validation_cost)