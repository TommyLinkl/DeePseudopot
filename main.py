import os
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import ExponentialLR

from constants.constants import *
from utils.nn_models import Net_relu_xavier, Net_relu_xavier_decay1, Net_relu_xavier_decay2
from utils.read import bulkSystem
from utils.pp_func import pot_func, realSpacePot, plotBandStruct, plotPP, plot_training_validation_cost, FT_converge_and_write_pp
from utils.bandStruct import calcHamiltonianMatrix_GPU, calcBandStruct_GPU
from utils.init_NN_train import init_Zunger_data, init_Zunger_weighted_mse, init_Zunger_train_GPU
from utils.NN_train import weighted_mse_bandStruct, BandStruct_train_GPU

torch.set_default_dtype(torch.float32)
torch.manual_seed(24)
SHOWPLOTS = True

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available.\n")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.\n")
    
############## main ##############
nSystem = 1

# Read and set up systems
print("############################################\nReading and setting up the bulkSystems. ")
atomPPOrder = []
systems = [bulkSystem() for _ in range(nSystem)]
for iSys in range(nSystem): 
    systems[iSys].setSystem("inputs/system_%d.par" % iSys)
    systems[iSys].setInputs("inputs/input_%d.par" % iSys)
    systems[iSys].setKPoints("inputs/kpoints_%d.par" % iSys)
    systems[iSys].setExpBS("inputs/expBandStruct_%d.par" % iSys)
    
    bandWeights=torch.ones(systems[iSys].nBands)
    bandWeights[4:8] = 5.0
    systems[iSys].setBandWeights(bandWeights)
    
    kptWeights=torch.ones(systems[iSys].getNKpts())
    kptWeights[[0, 20, 40]] = 5.0
    systems[iSys].setKptWeights(kptWeights)
    
    atomPPOrder.append(systems[iSys].atomTypes)
    
# Count how many atomTypes there are
atomPPOrder = np.unique(atomPPOrder)
nPseudopot = len(atomPPOrder)
print("There are %d atomic pseudopotentials. They are in the order of: " % nPseudopot)
print(atomPPOrder)


# Set up NN accordingly
# PPmodel = Net_relu_xavier([1, 20, 20, 20, 2])
# PPmodel = Net_relu_xavier_decay1([1, 20, 20, 20, 2], 1.5, 6)
PPmodel = Net_relu_xavier_decay2([1, 20, 20, 20, nPseudopot])

# Set up datasets accordingly
totalParams = torch.empty(0, 5)
for atomType in atomPPOrder: 
    file_path = 'inputs/init_'+atomType+'Params.par'
    if os.path.isfile(file_path):
        print(atomType + " is being initialized to the function form as stored in " + file_path)
        with open(file_path, 'r') as file:
            a = torch.tensor([float(line.strip()) for line in file])
        totalParams = torch.cat((totalParams, a.unsqueeze(0)), dim=0)
    else:
        print("File " + file_path + " cannot be found. This atom will not be initialized. OR IT WILL BE INITIALIZED TO BE 0. ")
        # BUT WE NEED TO KEEP GRADIENT
print(totalParams)


############# Initialize the NN #############
print("\n############################################\nInitializing the NN by fitting to the latest function form of pseudopotentials. ")
train_dataset = init_Zunger_data(atomPPOrder, totalParams, True)
val_dataset = init_Zunger_data(atomPPOrder, totalParams, False)

PPmodel.eval()
NN_init = PPmodel(val_dataset.q)
plotPP(atomPPOrder, val_dataset.q, val_dataset.q, val_dataset.vq_atoms, NN_init, "ZungerForm", "NN_init", ["-", ":", "-", ":"], False, SHOWPLOTS)

init_Zunger_criterion = init_Zunger_weighted_mse
init_Zunger_optimizer = torch.optim.Adam(PPmodel.parameters(), lr=0.1)
init_Zunger_scheduler = ExponentialLR(init_Zunger_optimizer, gamma=0.90)
trainloader = DataLoader(dataset = train_dataset, batch_size = int(train_dataset.len/4),shuffle=True)
validationloader = DataLoader(dataset = val_dataset, batch_size =val_dataset.len, shuffle=False)
init_Zunger_num_epochs = 1000
plotEvery = 500

start_time = time.time()
init_Zunger_train_GPU(PPmodel, device, trainloader, validationloader, init_Zunger_criterion, init_Zunger_optimizer, init_Zunger_scheduler, 20, init_Zunger_num_epochs, plotEvery, atomPPOrder, SHOWPLOTS)
end_time = time.time()
elapsed_time = end_time - start_time
print("GPU initialization: elapsed time: %.2f seconds" % elapsed_time)

os.makedirs('results', exist_ok=True)
torch.save(PPmodel.state_dict(), 'results/initZunger_PPmodel.pth')

print("\nDone with NN initialization to the latest function form. \nPlotting and write pseudopotentials in the real and reciprocal space.")
PPmodel.eval()
PPmodel.cpu()

qmax = np.array([10.0, 20.0, 30.0])
nQGrid = np.array([2048, 4096])
nRGrid = np.array([2048, 4096])
FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, val_dataset, 0.0, 8.0, -2.0, 0.5, 20.0, 2048, 2048, 'results/initZunger_plotPP', 'results/initZunger_pot', SHOWPLOTS)

print("\nEvaluating band structures using the initialized pseudopotentials. ")
plot_bandStruct_list = []
init_totalMSE = 0
for iSystem in range(nSystem): 
    init_bandStruct = calcBandStruct_GPU(True, PPmodel, systems[iSystem], atomPPOrder, totalParams, device)
    plot_bandStruct_list.append(systems[iSystem].expBandStruct)
    plot_bandStruct_list.append(init_bandStruct)
    init_totalMSE += weighted_mse_bandStruct(init_bandStruct, systems[iSystem])
    
fig = plotBandStruct(nSystem, plot_bandStruct_list, ["bo", "r-"], ["Reference zbCdSe", "NN_init"], SHOWPLOTS)
print("After fitting the NN to the latest function forms, we can reproduce satisfactory band structures. ")
print("The total bandStruct MSE = %e " % init_totalMSE)
fig.savefig('results/initZunger_plotBS.png')


############# Fit NN to band structures ############# 
print("\n############################################\nStart training of the NN to fit to band structures. ")

# PPmodel = nn_models.Net_relu_xavier_decay2([1, 20, 20, 20, 2])
# PPmodel.load_state_dict(torch.load('results/initZunger_PPmodel.pth'))

criterion = weighted_mse_bandStruct
optimizer = torch.optim.Adam(PPmodel.parameters(), lr=0.005)
scheduler = ExponentialLR(optimizer, gamma=0.90)

max_num_epochs = 200
plotEvery = 50
schedulerStep = 20
patience = 50

start_time = time.time()
LOSS = BandStruct_train_GPU(PPmodel, device, systems, atomPPOrder, totalParams, criterion, optimizer, scheduler, schedulerStep, max_num_epochs, plotEvery, patience, val_dataset, SHOWPLOTS)
end_time = time.time()
elapsed_time = end_time - start_time
print("GPU training: elapsed time: %.2f seconds" % elapsed_time)


############# Writing the trained NN PP ############# 
print("\n############################################\nWriting the NN pseudopotentials")
# PPmodel = Net_relu_xavier_decay2([1, 20, 20, 20, 2])
# PPmodel.load_state_dict(torch.load('results/epoch_199_PPmodel.pth')) # , map_location=torch.device('cpu')))
PPmodel.eval()
PPmodel.cpu()

qmax = np.array([10.0, 20.0, 30.0])
nQGrid = np.array([2048, 4096])
nRGrid = np.array([2048, 4096])
FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, val_dataset, 0.0, 8.0, -2.0, 0.5, 20.0, 2048, 2048, 'results/final_plotPP', 'results/final_pot', SHOWPLOTS)