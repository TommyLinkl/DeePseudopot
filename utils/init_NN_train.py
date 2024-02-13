import os, time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.rcParams['lines.markersize'] = 3

from constants.constants import * 
from utils.pp_func import pot_func, plotBandStruct, plotPP, plot_training_validation_cost, FT_converge_and_write_pp
from utils.NN_train import weighted_mse_bandStruct

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


def calcOldFuncBS(systems, hams, NNConfig, cachedMats_info, resultsFolder):
    """
    Calculate band structures with the old Zunger function form 
    using the parameters given in PPparams
    """
    oldFunc_plot_bandStruct_list = []
    oldFunc_totalMSE = 0
    for iSys, sys in enumerate(systems):
        start_time = time.time()
        oldFunc_bandStruct = hams[iSys].calcBandStruct_noGrad(NNConfig, iSys, cachedMats_info if cachedMats_info is not None else None)
        oldFunc_bandStruct.detach_()
        end_time = time.time()
        print(f"Old Zunger BS: Finished calculating {iSys}-th band structure in the Zunger function form ... Elapsed time: {(end_time - start_time):.2f} seconds")
        oldFunc_plot_bandStruct_list.append(sys.expBandStruct)
        oldFunc_plot_bandStruct_list.append(oldFunc_bandStruct)
        oldFunc_totalMSE += weighted_mse_bandStruct(oldFunc_bandStruct, sys)
    fig = plotBandStruct(systems, oldFunc_plot_bandStruct_list, NNConfig['SHOWPLOTS'])
    print("oldFunc_totalMSE = %e " % oldFunc_totalMSE)
    fig.suptitle("oldFunc_totalMSE = %e " % oldFunc_totalMSE)
    fig.savefig(resultsFolder + 'oldFunc_plotBS.png')
    plt.close('all')

    return oldFunc_totalMSE


def evalBS_convergePP(PPmodel, qmax, nQGrid, nRGrid):
    print("Evaluate the band structures and converge the pseudopotentials for the initialized NN. ")

    print("Plotting and write pseudopotentials in the real and reciprocal space.")
    torch.cuda.empty_cache()
    PPmodel.eval()
    FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, ZungerPPFunc_val, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, f'{resultsFolder}initZunger_plotPP', f'{resultsFolder}initZunger_pot', NNConfig['SHOWPLOTS'])

    print("\nEvaluating band structures using the initialized pseudopotentials. ")
    plot_bandStruct_list = []
    init_totalMSE = 0
    for iSystem in range(nSystem): 
        hams[iSystem].NN_locbool = True
        hams[iSystem].set_NNmodel(PPmodel)
        start_time = time.time()
        init_bandStruct = hams[iSystem].calcBandStruct_noGrad(NNConfig, iSystem, cachedMats_info if cachedMats_info is not None else None)
        init_bandStruct.detach_()
        end_time = time.time()
        print(f"Finished calculating {iSystem}-th band structure in the initialized NN form... Elapsed time: {(end_time - start_time):.2f} seconds")
        plot_bandStruct_list.append(systems[iSystem].expBandStruct)
        plot_bandStruct_list.append(init_bandStruct)
        init_totalMSE += weighted_mse_bandStruct(init_bandStruct, systems[iSystem])
    fig = plotBandStruct(systems, plot_bandStruct_list, NNConfig['SHOWPLOTS'])
    print("The total bandStruct MSE = %e " % init_totalMSE)
    fig.suptitle("The total bandStruct MSE = %e " % init_totalMSE)
    fig.savefig(resultsFolder + 'initZunger_plotBS.png')
    plt.close('all')
    torch.cuda.empty_cache()
    return