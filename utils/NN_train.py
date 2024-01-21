import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.rcParams['lines.markersize'] = 3
import torch
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import ExponentialLR
import gc
import multiprocessing as mp
from functools import partial

from utils.pp_func import *
from utils.memory import print_memory_usage
torch.set_default_dtype(torch.float32)
torch.manual_seed(24)

def print_and_inspect_gradients(model): 
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f'Parameter: {name}, Gradient shape: {param.grad.shape}')
            print(f'Gradient values:\n{param.grad}\n')
        else:
            print(f'Parameter: {name}, Gradient: None (no gradient computed)\n')

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

def weighted_mse_energiesAtKpt(calcEnergiesAtKpt, bulkSystem, kidx): 
    bandWeights = bulkSystem.bandWeights
    nBands = bulkSystem.nBands
    if (len(calcEnergiesAtKpt)!=nBands): 
        raise ValueError("CalculatedEnergiesAtKpt is of different length as nBands. Can't calculated MSE.")

    MSE = torch.sum((calcEnergiesAtKpt-bulkSystem.expBandStruct[kidx])**2 * bandWeights)
    return MSE

def calcGradSingleKpt_parallel(kidx, ham, iSystem, bulkSystem, criterion_singleKpt, optimizer, model, cachedMats_info):
    # loop over kidx
    # The rest of the arguments are "constants" / "constant functions" for a single kidx
    # For performance, it is recommended that the ham in the argument doesn't have SOmat and NLmat initialized. 
    singleKptGradients = {}
    calcEnergies = ham.calcEigValsAtK(kidx, iSystem, cachedMats_info, requires_grad=True)

    systemKptLoss = criterion_singleKpt(calcEnergies, bulkSystem, kidx)
    start_time = time.time()
    optimizer.zero_grad()
    systemKptLoss.backward()
    end_time = time.time()
    print(f"loss_backward + optimizer.step, elapsed time: {(end_time - start_time):.2f} seconds")
    for name, param in model.named_parameters():
        if param.grad is not None:
            if name not in singleKptGradients:
                singleKptGradients[name] = param.grad.detach().clone() * bulkSystem.kptWeights[kidx]
            else: 
                singleKptGradients[name] += param.grad.detach().clone() * bulkSystem.kptWeights[kidx]
    trainLoss_systemKpt = systemKptLoss.detach().item() * bulkSystem.kptWeights[kidx]
    del systemKptLoss
    gc.collect()
    return singleKptGradients, trainLoss_systemKpt

def bandStruct_train_GPU(model, device, NNConfig, bulkSystem_list, ham_list, atomPPOrder, totalParams, criterion_singleSystem, criterion_singleKpt, optimizer, scheduler, val_dataset, resultsFolder, cachedMats_info):
    training_COST=[]
    validation_COST=[]
    file_trainCost = open(resultsFolder + 'final_training_cost.dat', "w")
    file_valCost = open(resultsFolder + 'final_validation_cost.dat', "w")
    model.to(device)
    best_validation_loss = float('inf')
    no_improvement_count = 0
    
    for epoch in range(NNConfig['max_num_epochs']):
        # train
        print_memory_usage()
        model.train()
        if NNConfig['separateKptGrad']==0: 
            loss = torch.tensor(0.0)
            for iSystem in range(len(bulkSystem_list)):
                ham_list[iSystem].NN_locbool = True
                ham_list[iSystem].set_NNmodel(model)
                print_memory_usage()
                NN_outputs = ham_list[iSystem].calcBandStruct_withGrad(iSystem, cachedMats_info)
                # NN_outputs = calcBandStruct_GPU(True, model, bulkSystem_list[iSystem], atomPPOrder, totalParams, device)
                print_memory_usage()
                systemLoss = criterion_singleSystem(NN_outputs, bulkSystem_list[iSystem])
                # print_and_inspect_gradients(model)
                loss += systemLoss
            print_memory_usage()
            training_COST.append(loss.item())
            start_time = time.time()
            optimizer.zero_grad()
            print_memory_usage()
            loss.backward()
            # print_and_inspect_gradients(model)
            print_memory_usage()
            optimizer.step()
            end_time = time.time()
            print(f"loss_backward + optimizer.step, elapsed time: {(end_time - start_time):.2f} seconds")
            print_memory_usage()
            file_trainCost.write(f"{epoch+1}  {loss.item()}\n")
            torch.cuda.empty_cache()
            print_memory_usage()
            print(f"Epoch [{epoch+1}/{NNConfig['max_num_epochs']}], training cost: {loss.item():.4f}")
        else: 
            trainLoss = 0.0
            total_gradients = {}
            for iSystem in range(len(bulkSystem_list)):
                ham_list[iSystem].NN_locbool = True
                ham_list[iSystem].set_NNmodel(model)
                print_memory_usage()

                if ('num_cores' not in NNConfig): # or (NNConfig['num_cores']==1): 
                    # No multiprocessing
                    for kidx in range(bulkSystem_list[iSystem].getNKpts()): 
                        calcEnergies = ham_list[iSystem].calcEigValsAtK(kidx, iSystem, cachedMats_info, requires_grad=True)
                        systemKptLoss = criterion_singleKpt(calcEnergies, bulkSystem_list[iSystem], kidx)
                        start_time = time.time()
                        optimizer.zero_grad()
                        systemKptLoss.backward()
                        end_time = time.time()
                        print(f"loss_backward + optimizer.step, elapsed time: {(end_time - start_time):.2f} seconds")
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                if name not in total_gradients:
                                    total_gradients[name] = param.grad.detach().clone() * bulkSystem_list   [iSystem].kptWeights[kidx]
                                else: 
                                    total_gradients[name] += param.grad.detach().clone() * bulkSystem_list  [iSystem].kptWeights[kidx]
                        trainLoss += systemKptLoss.detach().item() * bulkSystem_list[iSystem].kptWeights[kidx]
                        del systemKptLoss
                        gc.collect()
                else: # multiprocessing
                    def merge_dicts(dicts):
                        merged_dict = {}
                        for d in dicts:
                            for key in d:
                                merged_dict[key] = merged_dict.get(key, 0) + d[key]
                        return merged_dict
                    print(f"Total num_cores available = {mp.cpu_count()}. We are using num_cores = {NNConfig['num_cores']}.")
                    pool = mp.Pool(NNConfig['num_cores'])
                    results_systemKpt = pool.map(partial(calcGradSingleKpt_parallel, 
                                                         ham=ham_list[iSystem], 
                                                         iSystem=iSystem, 
                                                         bulkSystem=bulkSystem_list[iSystem], 
                                                         criterion_singleKpt=criterion_singleKpt, 
                                                         optimizer=optimizer, 
                                                         model=model, 
                                                         cachedMats_info=cachedMats_info), 
                                                         range(bulkSystem_list[iSystem].getNKpts()))

                    gradients_systemKpt, trainLoss_systemKpt = zip(*results_systemKpt)
                    gc.collect()
                    pool.close()
                    pool.join()
                    total_gradients = merge_dicts(gradients_systemKpt)
                    trainLoss = torch.sum(torch.tensor(trainLoss_systemKpt))

            optimizer.zero_grad()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in total_gradients:
                        param.grad = total_gradients[name].detach().clone()
            print_memory_usage()
            optimizer.step()
            file_trainCost.write(f"{epoch+1}  {trainLoss}\n")
            training_COST.append(trainLoss)
            torch.cuda.empty_cache()
            print_memory_usage()
            print(f"Epoch [{epoch+1}/{NNConfig['max_num_epochs']}], training cost: {trainLoss:.4f}")
            # print_and_inspect_gradients(model)

        if epoch > 0 and epoch % NNConfig['schedulerStep'] == 0:
            scheduler.step()

        # evaluation
        print_memory_usage()
        if (epoch + 1) % NNConfig['plotEvery'] == 0:
            model.eval()
            plot_bandStruct_list = []
            val_loss = torch.tensor(0.0)
            for iSystem in range(len(bulkSystem_list)):
                ham_list[iSystem].set_NNmodel(model)
                with torch.no_grad():
                    NN_outputs = ham_list[iSystem].calcBandStruct_noGrad(NNConfig, iSystem, cachedMats_info)
                # NN_outputs = calcBandStruct_GPU(True, model, bulkSystem_list[iSystem], atomPPOrder, totalParams, device)
                systemLoss = criterion_singleSystem(NN_outputs, bulkSystem_list[iSystem])
                val_loss += systemLoss.item()
                
                plot_bandStruct_list.append(bulkSystem_list[iSystem].expBandStruct)
                NN_bandStruct = NN_outputs.cpu()
                plot_bandStruct_list.append(NN_bandStruct)
            validation_COST.append(val_loss.item())
            print(f"Epoch [{epoch+1}/{NNConfig['max_num_epochs']}], validation cost: {val_loss.item():.4f}")
            file_valCost.write(f"{epoch+1}  {val_loss.item()}\n")
            
            model.cpu()
            fig = plotPP(atomPPOrder, val_dataset.q, val_dataset.q, val_dataset.vq_atoms, model(val_dataset.q), "ZungerForm", f"NN_{epoch+1}", ["-",":" ]*len(atomPPOrder), True, NNConfig['SHOWPLOTS']);
            fig.savefig(resultsFolder + 'epoch_%d_plotPP.png' % epoch)
            model.to(device)
            
            fig = plotBandStruct(bulkSystem_list, plot_bandStruct_list, NNConfig['SHOWPLOTS'])
            fig.savefig(resultsFolder + 'epoch_%d_plotBS.png' % epoch)
            torch.save(model.state_dict(), resultsFolder + 'epoch_%d_PPmodel.pth' % epoch)
            torch.cuda.empty_cache()
        
        '''
        # Dynamic stopping: Stop training if no improvement (or less than 1e-4 in the loss) for 'patience' epochs
        # Should be for validation_loss
        if loss.item() < best_validation_loss - 1e-4:
            best_validation_loss = loss.item()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= NNConfig['patience']:
            print("Early stopping at Epoch %d due to lack of improvement." % epoch)
            break
        '''

        print_memory_usage()
        plt.close('all')
        torch.cuda.empty_cache()
    fig_cost = plot_training_validation_cost(training_COST, validation_COST, True, NNConfig['SHOWPLOTS']);
    fig_cost.savefig(resultsFolder + 'final_train_cost.png')
    torch.cuda.empty_cache()
    return (training_COST, validation_COST)