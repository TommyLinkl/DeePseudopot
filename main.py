import os
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt

from constants.constants import *
from utils.nn_models import *
from utils.read import BulkSystem, read_NNConfigFile, read_PPparams
from utils.pp_func import plotPP, FT_converge_and_write_pp, plotBandStruct
from utils.init_NN_train import init_Zunger_data, init_Zunger_weighted_mse, init_Zunger_train_GPU
from utils.NN_train import weighted_mse_bandStruct, weighted_mse_energiesAtKpt, bandStruct_train_GPU
from utils.ham import Hamiltonian
from utils.memory import print_memory_usage, plot_memory_usage, set_debug_memory_flag

torch.set_default_dtype(torch.float32)
torch.manual_seed(24)

'''
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available.\n")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.\n")
'''
device = torch.device("cpu")
memory_usage_data = []
set_debug_memory_flag(False)

############## main ##############
inputsFolder = 'inputs/'
resultsFolder = 'results/'
os.makedirs(resultsFolder, exist_ok=True)

NNConfig = read_NNConfigFile(inputsFolder + 'NN_config.par')
nSystem = NNConfig['nSystem']

# Read and set up systems
print("############################################\nReading and setting up the BulkSystems. ")
atomPPOrder = []
systems = [BulkSystem() for _ in range(nSystem)]
for iSys in range(nSystem): 
    systems[iSys].setSystem(inputsFolder + "system_%d.par" % iSys)
    systems[iSys].setInputs(inputsFolder + "input_%d.par" % iSys)
    systems[iSys].setKPointsAndWeights(inputsFolder + "kpoints_%d.par" % iSys)
    systems[iSys].setExpBS(inputsFolder + "expBandStruct_%d.par" % iSys)
    systems[iSys].setBandWeights(inputsFolder + "bandWeights_%d.par" % iSys)
    atomPPOrder.append(systems[iSys].atomTypes)

# Calculate atomPPOrder. Read in initial PPparams. Set up NN accordingly
atomPPOrder = np.unique(np.concatenate(atomPPOrder))
nPseudopot = len(atomPPOrder)
print(f"There are {nPseudopot} atomic pseudopotentials. They are in the order of: {atomPPOrder}")
allSystemNames = [x.systemName for x in systems]
PPparams, totalParams = read_PPparams(atomPPOrder, inputsFolder + "init_")
localPotParams = totalParams[:,:4]
layers = [1] + NNConfig['hiddenLayers'] + [nPseudopot]
# PPmodel = Net_relu_xavier_decay2(layers)
if NNConfig['PPmodel'] in globals() and callable(globals()[NNConfig['PPmodel']]):
    if NNConfig['PPmodel']=='Net_relu_xavier_decay': 
        PPmodel = globals()[NNConfig['PPmodel']](layers, decay_rate=NNConfig['PPmodel_decay_rate'], decay_center=NNConfig['PPmodel_decay_center'])
    elif NNConfig['PPmodel']=='Net_relu_xavier_decayGaussian': 
        PPmodel = globals()[NNConfig['PPmodel']](layers, gaussian_std=NNConfig['PPmodel_gaussian_std'])
    else: 
        PPmodel = globals()[NNConfig['PPmodel']](layers)
else:
    raise ValueError(f"Function {NNConfig['PPmodel']} does not exist.")
print_memory_usage()

# Initialize the ham class for each BulkSystem
print("Initializing the ham class for each BulkSystem. ")
hams = []
for iSys in range(nSystem): 
    start_time = time.time()
    ham = Hamiltonian(systems[iSys], PPparams, atomPPOrder, device, NNConfig, SObool=NNConfig['SObool'])
    hams.append(ham)
    end_time = time.time()
    print(f"Finished initializing {iSys}-th Hamiltonian Class... Elapsed time: {(end_time - start_time):.2f} seconds")
print_memory_usage()

oldFunc_plot_bandStruct_list = []
oldFunc_totalMSE = 0
for iSystem in range(nSystem): 
    start_time = time.time()
    oldFunc_bandStruct = hams[iSystem].calcBandStruct()  
    # oldFunc_bandStruct = calcBandStruct_GPU(False, PPmodel, systems[iSystem], atomPPOrder, localPotParams, device) 
    end_time = time.time()
    print(f"Finished calculating {iSystem}-th band structure... Elapsed time: {(end_time - start_time):.2f} seconds")
    oldFunc_plot_bandStruct_list.append(systems[iSystem].expBandStruct)
    oldFunc_plot_bandStruct_list.append(oldFunc_bandStruct)
    oldFunc_totalMSE += weighted_mse_bandStruct(oldFunc_bandStruct, systems[iSystem])
fig = plotBandStruct(allSystemNames, oldFunc_plot_bandStruct_list, NNConfig['SHOWPLOTS'])
fig.suptitle("The total bandStruct MSE = %e " % oldFunc_totalMSE)
fig.savefig(resultsFolder + 'oldFunc_plotBS.png')
plt.close('all')
print_memory_usage()

############# Initialize the NN to the local pot function form #############
train_dataset = init_Zunger_data(atomPPOrder, localPotParams, train=True)
val_dataset = init_Zunger_data(atomPPOrder, localPotParams, train=False)

if os.path.exists(inputsFolder + 'init_PPmodel.pth'):
    print(f"\n############################################\nInitializing the NN with file {inputsFolder}init_PPmodel.pth.")
    PPmodel.load_state_dict(torch.load(inputsFolder + 'init_PPmodel.pth'))
    print(f"\nDone with NN initialization to the file {inputsFolder}init_PPmodel.pth.")
else:
    print("\n############################################\nInitializing the NN by fitting to the latest function form of pseudopotentials. ")
    PPmodel.cpu()
    PPmodel.eval()
    NN_init = PPmodel(val_dataset.q)
    plotPP(atomPPOrder, val_dataset.q, val_dataset.q, val_dataset.vq_atoms, NN_init, "ZungerForm", "NN_init", ["-",":" ]*nPseudopot, False, NNConfig['SHOWPLOTS'])
    
    init_Zunger_criterion = init_Zunger_weighted_mse
    init_Zunger_optimizer = torch.optim.Adam(PPmodel.parameters(), lr=NNConfig['init_Zunger_optimizer_lr'])
    init_Zunger_scheduler = ExponentialLR(init_Zunger_optimizer, gamma=NNConfig['init_Zunger_scheduler_gamma'])
    trainloader = DataLoader(dataset = train_dataset, batch_size = int(train_dataset.len/4),shuffle=True)
    validationloader = DataLoader(dataset = val_dataset, batch_size =val_dataset.len, shuffle=False)
    
    start_time = time.time()
    init_Zunger_train_GPU(PPmodel, device, trainloader, validationloader, init_Zunger_criterion, init_Zunger_optimizer, init_Zunger_scheduler, 20, NNConfig['init_Zunger_num_epochs'], NNConfig['init_Zunger_plotEvery'], atomPPOrder, NNConfig['SHOWPLOTS'], resultsFolder)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("GPU initialization: elapsed time: %.2f seconds" % elapsed_time)
    
    torch.save(PPmodel.state_dict(), resultsFolder + 'initZunger_PPmodel.pth')

    print("\nDone with NN initialization to the latest function form.")
print_memory_usage()

print("\nPlotting and write pseudopotentials in the real and reciprocal space.")
torch.cuda.empty_cache()
PPmodel.eval()
PPmodel.cpu()
qmax = np.array([10.0, 20.0, 30.0])
nQGrid = np.array([2048, 4096])
nRGrid = np.array([2048, 4096])
FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, val_dataset, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, resultsFolder + 'initZunger_plotPP', resultsFolder + 'initZunger_pot', NNConfig['SHOWPLOTS'])
print_memory_usage()

print("\nEvaluating band structures using the initialized pseudopotentials. ")
plot_bandStruct_list = []
init_totalMSE = 0
for iSystem in range(nSystem): 
    hams[iSystem].NN_locbool = True
    hams[iSystem].set_NNmodel(PPmodel)
    start_time = time.time()
    init_bandStruct = hams[iSystem].calcBandStruct()
    # init_bandStruct = calcBandStruct_GPU(True, PPmodel, systems[iSystem], atomPPOrder, localPotParams, device)
    end_time = time.time()
    print(f"Finished calculating {iSystem}-th band structure... Elapsed time: {(end_time - start_time):.2f} seconds")
    plot_bandStruct_list.append(systems[iSystem].expBandStruct)
    plot_bandStruct_list.append(init_bandStruct)
    init_totalMSE += weighted_mse_bandStruct(init_bandStruct, systems[iSystem])
fig = plotBandStruct(allSystemNames, plot_bandStruct_list, NNConfig['SHOWPLOTS'])
print("After fitting the NN to the latest function forms, we can reproduce satisfactory band structures. ")
print("The total bandStruct MSE = %e " % init_totalMSE)
fig.suptitle("The total bandStruct MSE = %e " % init_totalMSE)
fig.savefig(resultsFolder + 'initZunger_plotBS.png')
plt.close('all')
torch.cuda.empty_cache()
print_memory_usage()

############# Fit NN to band structures ############# 
print("\n############################################\nStart training of the NN to fit to band structures. ")

criterion_singleSystem = weighted_mse_bandStruct
criterion_singleKpt = weighted_mse_energiesAtKpt
optimizer = torch.optim.Adam(PPmodel.parameters(), lr=NNConfig['optimizer_lr'])
scheduler = ExponentialLR(optimizer, gamma=NNConfig['scheduler_gamma'])

start_time = time.time()
(training_cost, validation_cost) = bandStruct_train_GPU(PPmodel, device, NNConfig, systems, hams, atomPPOrder, localPotParams, criterion_singleSystem, criterion_singleKpt, optimizer, scheduler, val_dataset, resultsFolder)
end_time = time.time()
elapsed_time = end_time - start_time
print("GPU training: elapsed time: %.2f seconds" % elapsed_time)
torch.cuda.empty_cache()

############# Writing the trained NN PP ############# 
print("\n############################################\nWriting the NN pseudopotentials")
PPmodel.eval()
PPmodel.cpu()
qmax = np.array([10.0, 20.0, 30.0])
nQGrid = np.array([2048, 4096])
nRGrid = np.array([2048, 4096])
FT_converge_and_write_pp(atomPPOrder, qmax, nQGrid, nRGrid, PPmodel, val_dataset, 0.0, 8.0, -2.0, 1.0, 20.0, 2048, 2048, resultsFolder + 'final_plotPP', resultsFolder + 'final_pot', NNConfig['SHOWPLOTS'])
plot_memory_usage(resultsFolder)