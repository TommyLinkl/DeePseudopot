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
from utils.bandStruct import calcHamiltonianMatrix_GPU, calcBandStruct_GPU
torch.set_default_dtype(torch.float32)
torch.manual_seed(24)

def weighted_mse_bandStruct(bandStruct_hat, bulkSystem): 
    bandWeights = bulkSystem.bandWeights
    kptWeights = bulkSystem.kptWeights
    nkpt = bulkSystem.getNKpts()
    nBands = bulkSystem.nBands
    if (len(bandWeights)!=nBands) or (len(kptWeights)!=nkpt): 
        raise ValueError("bandWeights or kptWeights lengths aren't correct. ")
        
    newBandWeights = bandWeights.view(1, -1).expand(nkpt, -1)
    newKptWeights = kptWeights.view(-1, 1).expand(-1, nBands)
    
    MSE = torch.sum((bandStruct_hat-bulkSystem.expBandStruct)**2 * newBandWeights * newKptWeights)
    return MSE

def BandStruct_train_GPU(model, device, bulkSystem_list, atomPPOrder, totalParams, criterion_singleSystem, optimizer, scheduler, scheduler_step, max_epochs, plot_every, patience_epochs, val_dataset, SHOWPLOTS):
    training_COST=[]
    validation_COST=[]
    file_trainCost = open('results/final_training_cost.dat', "w")
    file_valCost = open('results/final_validation_cost.dat', "w")
    model.to(device)
    best_validation_loss = float('inf')
    no_improvement_count = 0
    
    for epoch in range(max_epochs):
        # train
        model.train()
        loss = torch.tensor(0.0)
        for iSystem in range(len(bulkSystem_list)):
            NN_outputs = calcBandStruct_GPU(True, model, bulkSystem_list[iSystem], atomPPOrder, totalParams, device)
            systemLoss = criterion_singleSystem(NN_outputs, bulkSystem_list[iSystem])
            loss += systemLoss
        training_COST.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        file_trainCost.write(f"{epoch+1}  {loss.item()}\n")
        
        if epoch > 0 and epoch % scheduler_step == 0:
            scheduler.step()
            
        # evaluation
        if (epoch + 1) % plot_every == 0:
            model.eval()
            plot_bandStruct_list = []
            val_loss = torch.tensor(0.0)
            for iSystem in range(len(bulkSystem_list)):
                NN_outputs = calcBandStruct_GPU(True, model, bulkSystem_list[iSystem], atomPPOrder, totalParams, device)
                systemLoss = criterion_singleSystem(NN_outputs, bulkSystem_list[iSystem])
                val_loss += systemLoss
                
                plot_bandStruct_list.append(bulkSystem_list[iSystem].expBandStruct)
                NN_bandStruct = NN_outputs.cpu()
                plot_bandStruct_list.append(NN_bandStruct)
            validation_COST.append(val_loss.item())
            print(f'Epoch [{epoch+1}/{max_epochs}], training cost: {loss.item():.4f}, validation cost: {val_loss.item():.4f}')
            file_valCost.write(f"{epoch+1}  {val_loss.item()}\n")
            
            model.cpu()
            fig = plotPP(atomPPOrder, val_dataset.q, val_dataset.q, val_dataset.vq_atoms, model(val_dataset.q), "ZungerForm", f"NN_{epoch+1}", ["-",":" ]*len(atomPPOrder), True, SHOWPLOTS);
            fig.savefig('results/epoch_%d_plotPP.png' % epoch)
            model.to(device)
            
            fig = plotBandStruct([x.systemName for x in bulkSystem_list], plot_bandStruct_list, SHOWPLOTS)
            fig.savefig('results/epoch_%d_plotBS.png' % epoch)
            torch.save(model.state_dict(), 'results/epoch_%d_PPmodel.pth' % epoch)
        
        '''
        # Dynamic stopping: Stop training if no improvement (or less than 1e-4 in the loss) for 'patience' epochs
        # Should be for validation_loss
        if loss.item() < best_validation_loss - 1e-4:
            best_validation_loss = loss.item()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= patience_epochs:
            print("Early stopping at Epoch %d due to lack of improvement." % epoch)
            break
        '''
        plt.close('all')
    fig_cost = plot_training_validation_cost(training_COST, validation_COST, True, SHOWPLOTS);
    fig_cost.savefig('results/final_train_cost.png')

    return (training_COST, validation_COST)
