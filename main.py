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
from utils.read import bulkSystem, read_NNConfigFile
from utils.pp_func import pot_func, realSpacePot, plotBandStruct, plotPP, plot_training_validation_cost, FT_converge_and_write_pp
from utils.bandStruct import calcHamiltonianMatrix_GPU, calcBandStruct_GPU
from utils.init_NN_train import init_Zunger_data, init_Zunger_weighted_mse, init_Zunger_train_GPU
from utils.NN_train import weighted_mse_bandStruct, BandStruct_train_GPU

torch.set_default_dtype(torch.float64)
torch.manual_seed(24)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available.\n")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.\n")


############## main ##############

NNConfig = read_NNConfigFile('inputs/NN_config.par')
SHOWPLOTS = NNConfig['SHOWPLOTS']
nSystem = NNConfig['nSystem']
hiddenLayers = NNConfig['hiddenLayers']
init_Zunger_optimizer_lr = NNConfig['init_Zunger_optimizer_lr']
init_Zunger_scheduler_gamma = NNConfig['init_Zunger_scheduler_gamma']
init_Zunger_num_epochs = NNConfig['init_Zunger_num_epochs']
init_Zunger_plotEvery = NNConfig['init_Zunger_plotEvery']
optimizer_lr = NNConfig['optimizer_lr']
scheduler_gamma = NNConfig['scheduler_gamma']
max_num_epochs = NNConfig['max_num_epochs']
plotEvery = NNConfig['plotEvery']
schedulerStep = NNConfig['schedulerStep']
patience = NNConfig['patience']

# Read and set up systems
print("############################################\nReading and setting up the bulkSystems. ")
atomPPOrder = []
systems = [bulkSystem() for _ in range(nSystem)]
for iSys in range(nSystem): 
    systems[iSys].setSystem("inputs/system_%d.par" % iSys)
    systems[iSys].setInputs("inputs/input_%d.par" % iSys)
    systems[iSys].setKPointsAndWeights("inputs/kpoints_%d.par" % iSys)
    systems[iSys].setExpBS("inputs/expBandStruct_%d.par" % iSys)
    systems[iSys].setBandWeights("inputs/bandWeights_%d.par" % iSys)
    atomPPOrder.append(systems[iSys].atomTypes)

# Count how many atomTypes there are
atomPPOrder = np.unique(np.concatenate(atomPPOrder))
nPseudopot = len(atomPPOrder)
print("There are %d atomic pseudopotentials. They are in the order of: " % nPseudopot)
print(atomPPOrder)
allSystemNames = [x.systemName for x in systems]

# Set up NN accordingly
layers = [1] + hiddenLayers + [nPseudopot]
# PPmodel = Net_relu_xavier([1, 20, 20, 20, 2])
# PPmodel = Net_relu_xavier_decay1([1, 20, 20, 20, 2], 1.5, 6)
PPmodel = Net_relu_xavier_decay2(layers)

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

oldFunc_plot_bandStruct_list = []
oldFunc_totalMSE = 0
for iSystem in range(nSystem): 
    oldFunc_bandStruct = calcBandStruct_GPU(False, PPmodel, systems[iSystem], atomPPOrder, totalParams, device)
    oldFunc_plot_bandStruct_list.append(systems[iSystem].expBandStruct)
    oldFunc_plot_bandStruct_list.append(oldFunc_bandStruct)
    oldFunc_totalMSE += weighted_mse_bandStruct(oldFunc_bandStruct, systems[iSystem])
fig = plotBandStruct(allSystemNames, oldFunc_plot_bandStruct_list, SHOWPLOTS)
fig.suptitle("The total bandStruct MSE = %e " % oldFunc_totalMSE)
fig.savefig('results/oldFunc_plotBS.png')
plt.close('all')

############# Initialize the NN #############
train_dataset = init_Zunger_data(atomPPOrder, totalParams, True)
val_dataset = init_Zunger_data(atomPPOrder, totalParams, False)

if os.path.exists('inputs/init_PPmodel.pth'):
    print("\n############################################\nInitializing the NN with file inputs/init_PPmodel.pth.")
    PPmodel.load_state_dict(torch.load('inputs/init_PPmodel.pth'))
    print("\nDone with NN initialization to the file inputs/init_PPmodel.pth.")
else:
    print("\n############################################\nInitializing the NN by fitting to the latest function form of pseudopotentials. ")
    PPmodel.cpu()
    PPmodel.eval()
    NN_init = PPmodel(val_dataset.q)
    plotPP(atomPPOrder, val_dataset.q, val_dataset.q, val_dataset.vq_atoms, NN_init, "ZungerForm", "NN_init", ["-",":" ]*nPseudopot, False, SHOWPLOTS)
    
    init_Zunger_criterion = init_Zunger_weighted_mse
    init_Zunger_optimizer = torch.optim.Adam(PPmodel.parameters(), lr=init_Zunger_optimizer_lr)
    init_Zunger_scheduler = ExponentialLR(init_Zunger_optimizer, gamma=init_Zunger_scheduler_gamma)
    trainloader = DataLoader(dataset = train_dataset, batch_size = int(train_dataset.len/4),shuffle=True)
    validationloader = DataLoader(dataset = val_dataset, batch_size =val_dataset.len, shuffle=False)
    
    start_time = time.time()
    init_Zunger_train_GPU(PPmodel, device, trainloader, validationloader, init_Zunger_criterion, init_Zunger_optimizer, init_Zunger_scheduler, 20, init_Zunger_num_epochs, init_Zunger_plotEvery, atomPPOrder, SHOWPLOTS)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("GPU initialization: elapsed time: %.2f seconds" % elapsed_time)
    
    os.makedirs('results', exist_ok=True)
    torch.save(PPmodel.state_dict(), 'results/initZunger_PPmodel.pth')

    print("\nDone with NN initialization to the latest function form.")

print("\nPlotting and write pseudopotentials in the real and reciprocal space.")
torch.cuda.empty_cache()
PPmodel.eval()
PPmodel.cpu()

qmax = np.array([10.0, 20.0, 30.0])
nQGrid = np.array([2048, 4096])
nRGrid = np.array([2048, 4096])
FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, val_dataset, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, 'results/initZunger_plotPP', 'results/initZunger_pot', SHOWPLOTS)

print("\nEvaluating band structures using the initialized pseudopotentials. ")
plot_bandStruct_list = []
init_totalMSE = 0
for iSystem in range(nSystem): 
    init_bandStruct = calcBandStruct_GPU(True, PPmodel, systems[iSystem], atomPPOrder, totalParams, device)
    plot_bandStruct_list.append(systems[iSystem].expBandStruct)
    plot_bandStruct_list.append(init_bandStruct)
    init_totalMSE += weighted_mse_bandStruct(init_bandStruct, systems[iSystem])
fig = plotBandStruct(allSystemNames, plot_bandStruct_list, SHOWPLOTS)
print("After fitting the NN to the latest function forms, we can reproduce satisfactory band structures. ")
print("The total bandStruct MSE = %e " % init_totalMSE)
fig.suptitle("The total bandStruct MSE = %e " % init_totalMSE)
fig.savefig('results/initZunger_plotBS.png')
plt.close('all')
torch.cuda.empty_cache()

############# Fit NN to band structures ############# 
print("\n############################################\nStart training of the NN to fit to band structures. ")
# layers = [1] + hiddenLayers + [nPseudopot]
# PPmodel = Net_relu_xavier_decay2([layers])
# PPmodel.load_state_dict(torch.load('results/initZunger_PPmodel.pth'))

criterion = weighted_mse_bandStruct
optimizer = torch.optim.Adam(PPmodel.parameters(), lr=optimizer_lr)
scheduler = ExponentialLR(optimizer, gamma=scheduler_gamma)

start_time = time.time()
(training_cost, validation_cost) = BandStruct_train_GPU(PPmodel, device, systems, atomPPOrder, totalParams, criterion, optimizer, scheduler, schedulerStep, max_num_epochs, plotEvery, patience, val_dataset, SHOWPLOTS)
end_time = time.time()
elapsed_time = end_time - start_time
print("GPU training: elapsed time: %.2f seconds" % elapsed_time)
torch.cuda.empty_cache()

############# Writing the trained NN PP ############# 
print("\n############################################\nWriting the NN pseudopotentials")
# layers = [1] + hiddenLayers + [nPseudopot]
# PPmodel = Net_relu_xavier_decay2([layers])
# PPmodel.load_state_dict(torch.load('results/epoch_199_PPmodel.pth')) # , map_location=torch.device('cpu')))
PPmodel.eval()
PPmodel.cpu()

qmax = np.array([10.0, 20.0, 30.0])
nQGrid = np.array([2048, 4096])
nRGrid = np.array([2048, 4096])
FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, val_dataset, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, 'results/final_plotPP', 'results/final_pot', SHOWPLOTS)