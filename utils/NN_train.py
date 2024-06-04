import torch
import time, os
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import gc
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.rcParams['lines.markersize'] = 3
import copy

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from .constants import *
from .pp_func import plotPP, plot_training_validation_cost, plotBandStruct, plot_mc_cost, plotBandStruct_reorder
from .smooth_order import reorder_smoothness_deg2_tensors

def print_and_inspect_gradients(model, filename=None, show=False): 
    if (filename is None) and show: 
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f'Parameter: {name}, Gradient shape: {param.grad.shape}')
                print(f'Gradient values:\n{param.grad}\n')
            else:
                print(f'Parameter: {name}, Gradient: None (no gradient computed)\n')
    elif (filename is not None) and show: 
        with open(filename, 'w') as f:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    f.write(f'Parameter: {name}, Gradient shape: {param.grad.shape}\n')
                    grad_str = np.array2string(param.grad.numpy(), precision=5, suppress_small=True, max_line_width=999999, threshold=99*99)
                    f.write(f'Gradient values:\n{grad_str}\n\n')
                else:
                    f.write(f'Parameter: {name}, Gradient: None (no gradient computed)\n\n')    


def print_and_inspect_NNParams(model, filename=None, show=False): 
    if (filename is None) and show: 
        for name, param in model.named_parameters():
            print(f'Parameter: {name}, Tensor shape: {param.shape}')
            print(f'Parameter values:\n{param}\n')
    elif (filename is not None) and show: 
        with open(filename, 'w') as f:
            for name, param in model.named_parameters():
                f.write(f'Parameter: {name}, Tensor shape: {param.shape}\n')
                tensor_str = np.array2string(param.detach().numpy(), precision=5, suppress_small=True, max_line_width=999999, threshold=99*99)
                f.write(f'Gradient values:\n{tensor_str}\n\n')


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


def evalBS_noGrad(model, BSplotFilename, runName, NNConfig, hams, systems, cachedMats_info=None, writeBS=False): 
    if (model is not None): 
        print(f"{runName}: Evaluating band structures using the NN-pp model. ")
        model.eval()
    else:
        print(f"{runName}: Evaluating band structures using the old Zunger function form. ")
    
    plot_bandStruct_list = []
    totalMSE = 0
    for iSys, sys in enumerate(systems):
        if (model is not None): 
            hams[iSys].NN_locbool = True
            hams[iSys].set_NNmodel(model)
        else: 
            hams[iSys].NN_locbool = False

        start_time = time.time()
        with torch.no_grad():
            evalBS = hams[iSys].calcBandStruct_noGrad(cachedMats_info)
        evalBS.detach_()
        end_time = time.time()
        print(f"{runName}: Finished evaluating {iSys}-th band structure with no gradient... Elapsed time: {(end_time - start_time):.2f} seconds")
        if writeBS: 
            if not BSplotFilename.endswith('_plotBS.png'):
                raise ValueError("BSplotFilename must end with '_plotBS.png' to write BS.dat files. ")
            else:
                write_BS_filename = BSplotFilename.replace('_plotBS.png', f'_BS_sys{iSys}.dat')
            kptDistInputs_vertical = sys.kptDistInputs.view(-1, 1)
            write_tensor = torch.cat((kptDistInputs_vertical, evalBS), dim=1)
            np.savetxt(write_BS_filename, write_tensor, fmt='%.5f')
            print(f"{runName}: Wrote BS to file {write_BS_filename}. ")
        plot_bandStruct_list.append(sys.expBandStruct)
        plot_bandStruct_list.append(evalBS)
        totalMSE += weighted_mse_bandStruct(evalBS, sys)
    fig = plotBandStruct(systems, plot_bandStruct_list, NNConfig['SHOWPLOTS'])
    print(f"{runName}: totalMSE = {totalMSE:f}")
    fig.suptitle(f"{runName}: totalMSE = {totalMSE:f}")
    fig.savefig(BSplotFilename)
    plt.close('all')
    torch.cuda.empty_cache()
    return totalMSE


def calcEigValsAtK_wGrad_parallel(kidx, ham, bulkSystem, criterion_singleKpt, optimizer, model, cachedMats_info=None):
    """
    loop over kidx
    The rest of the arguments are "constants" / "constant functions" for a single kidx
    For performance, it is recommended that the ham in the argument doesn't have SOmat and NLmat initialized. 
    """
    singleKptGradients = {}
    calcEnergies = ham.calcEigValsAtK(kidx, cachedMats_info, requires_grad=True)

    systemKptLoss = criterion_singleKpt(calcEnergies, bulkSystem, kidx)
    start_time = time.time() if ham.NNConfig['runtime_flag'] else None
    optimizer.zero_grad()
    systemKptLoss.backward()
    end_time = time.time() if ham.NNConfig['runtime_flag'] else None
    print(f"loss_backward, elapsed time: {(end_time - start_time):.2f} seconds") if ham.NNConfig['runtime_flag'] else None
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


def calcEigValsAtK_wGrad_parallel_smoothReorder(kidx, ham, bulkSystem, optimizer, model, cachedMats_info=None):
    nbands = bulkSystem.nBands
    lin_grad_kpt = [None for _ in range(nbands)]
    quad_grad_kpt = [None for _ in range(nbands)]

    calcEnergies = ham.calcEigValsAtK(kidx, cachedMats_info, requires_grad=True)
    energies_detach = calcEnergies.detach()

    start_time = time.time() if ham.NNConfig['runtime_flag'] else None
    for bandIdx, bandEne in enumerate(calcEnergies):
        # optimizer.zero_grad()
        grads = torch.autograd.grad(bandEne, model.parameters(), retain_graph=True)

        tmp_grad = {}
        for i, (name, param) in enumerate(model.named_parameters()):
            if grads[i] is not None:
                tmp_grad[name] = tmp_grad.get(name, 0) + grads[i].detach().clone()
        lin_grad_kpt[bandIdx] = tmp_grad

        tmp_grad = {}
        for i, (name, param) in enumerate(model.named_parameters()):
            if grads[i] is not None:
                tmp_grad[name] = tmp_grad.get(name, 0) + 2 * bandEne.detach().clone() * grads[i].detach().clone()
        quad_grad_kpt[kidx][bandIdx] = tmp_grad

        bandEne.detach()
        del bandEne
        torch.cuda.empty_cache()
        gc.collect()

    end_time = time.time() if ham.NNConfig['runtime_flag'] else None
    print(f"Backprop on individual energies + storing all the gradients, elapsed time: {(end_time - start_time):.2f} seconds") if ham.NNConfig['runtime_flag'] else None
        
    return energies_detach, lin_grad_kpt, quad_grad_kpt


def trainIter_naive(model, systems, hams, criterion_singleSystem, optimizer, cachedMats_info=None, runtime_flag=False):
    trainLoss = torch.tensor(0.0)
    for iSys, sys in enumerate(systems):
        hams[iSys].NN_locbool = True
        hams[iSys].set_NNmodel(model)

        NN_outputs = hams[iSys].calcBandStruct_withGrad(cachedMats_info)
        
        systemLoss = criterion_singleSystem(NN_outputs, sys)
        # print_and_inspect_gradients(model, show=True)
        trainLoss += systemLoss

    start_time = time.time() if runtime_flag else None
    optimizer.zero_grad()
    trainLoss.backward()
    # print_and_inspect_gradients(model, show=True)
    optimizer.step()
    end_time = time.time() if runtime_flag else None
    print(f"loss_backward + optimizer.step, elapsed time: {(end_time - start_time):.2f} seconds") if runtime_flag else None

    torch.cuda.empty_cache()
    return model, trainLoss


def trainIter_separateKptGrad(model, systems, hams, NNConfig, criterion_singleKpt, optimizer, cachedMats_info=None): 
    def merge_dicts(dicts):
        merged_dict = {}
        for d in dicts:
            for key in d:
                merged_dict[key] = merged_dict.get(key, 0) + d[key]
        return merged_dict
    
    trainLoss = 0.0
    total_gradients = {}
    for iSys, sys in enumerate(systems):
        trainLoss_system = 0.0
        gradients_system = {}
        hams[iSys].NN_locbool = True
        hams[iSys].set_NNmodel(model)

        if (NNConfig['num_cores']==0):   # No multiprocessing
            for kidx in range(sys.getNKpts()): 
                calcEnergies = hams[iSys].calcEigValsAtK(kidx, cachedMats_info, requires_grad=True)
                systemKptLoss = criterion_singleKpt(calcEnergies, sys, kidx)

                start_time = time.time() if NNConfig['runtime_flag'] else None
                optimizer.zero_grad()
                systemKptLoss.backward()
                end_time = time.time() if NNConfig['runtime_flag'] else None
                print(f"loss_backward, elapsed time: {(end_time - start_time):.2f} seconds") if NNConfig['runtime_flag'] else None

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if name not in gradients_system:
                            gradients_system[name] = param.grad.detach().clone() * sys.kptWeights[kidx]
                        else: 
                            gradients_system[name] += param.grad.detach().clone() * sys.kptWeights[kidx]
                trainLoss_system += systemKptLoss.detach().item() * sys.kptWeights[kidx]
                del systemKptLoss
                gc.collect()

        else: # multiprocessing
            optimizer.zero_grad()
            args_list = [(kidx, hams[iSys], sys, criterion_singleKpt, optimizer, model, cachedMats_info) for kidx in range(sys.getNKpts())]
            with mp.Pool(NNConfig['num_cores']) as pool:
                results_systemKpt = pool.starmap(calcEigValsAtK_wGrad_parallel, args_list)
                gradients_systemKpt, trainLoss_systemKpt = zip(*results_systemKpt)
            gc.collect()
            gradients_system = merge_dicts(gradients_systemKpt)
            trainLoss_system = torch.sum(torch.tensor(trainLoss_systemKpt))
        
        total_gradients = merge_dicts([total_gradients, gradients_system])
        trainLoss += trainLoss_system

    # Write the manually accumulated gradients and loss values back into the NN model
    optimizer.zero_grad()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in total_gradients:
                param.grad = total_gradients[name].detach().clone()

    start_time = time.time() if NNConfig['runtime_flag'] else None
    optimizer.step()
    end_time = time.time() if NNConfig['runtime_flag'] else None
    print(f"optimizer step, elapsed time: {(end_time - start_time):.2f} seconds") if NNConfig['runtime_flag'] else None

    torch.cuda.empty_cache()
    # print_and_inspect_gradients(model, show=NNConfig['printGrad'])

    return model, trainLoss


def trainIter_naive_smoothReorder(model, systems, hams, criterion_singleSystem, optimizer, cachedMats_info=None, runtime_flag=False):
    trainLoss = torch.tensor(0.0)
    for iSys, sys in enumerate(systems):
        hams[iSys].NN_locbool = True
        hams[iSys].set_NNmodel(model)

        NN_outputs = hams[iSys].calcBandStruct_withGrad(cachedMats_info)

        # reorder NN_outputs
        order_table, newBS, extrapolated_points = reorder_smoothness_deg2_tensors(NN_outputs)
        systemLoss = criterion_singleSystem(newBS, sys)
        trainLoss += systemLoss

    start_time = time.time() if runtime_flag else None
    optimizer.zero_grad()
    trainLoss.backward()
    optimizer.step()
    end_time = time.time() if runtime_flag else None
    print(f"loss_backward + optimizer.step, elapsed time: {(end_time - start_time):.2f} seconds") if runtime_flag else None

    torch.cuda.empty_cache()
    return model, trainLoss, order_table, newBS, extrapolated_points


def trainIter_separateKptGrad_smoothReorder(model, systems, hams, NNConfig, optimizer, cachedMats_info=None): 
    def merge_dicts(dicts):
        merged_dict = {}
        for d in dicts:
            for key in d:
                merged_dict[key] = merged_dict.get(key, 0) + d[key]
        return merged_dict
    
    trainLoss = 0.0
    total_gradients = {}
    for iSys, sys in enumerate(systems):
        hams[iSys].NN_locbool = True
        hams[iSys].set_NNmodel(model)
        nbands = sys.nBands
        nkpt = sys.getNKpts()

        # A list of lists of dictionaries. I will copy out, detach, and store all the parameter gradients. 
        lin_grad = [[None] * nbands for _ in range(nkpt)]
        quad_grad = [[None] * nbands for _ in range(nkpt)]
        calcBS_nograd = torch.zeros([nkpt, nbands], requires_grad=False)
        if (NNConfig['num_cores']==0):   # No multiprocessing
            for kidx in range(nkpt): 
                calcEnergies = hams[iSys].calcEigValsAtK(kidx, cachedMats_info, requires_grad=True)
                calcBS_nograd[kidx,:] = calcEnergies.detach()

                start_time = time.time()
                grads = torch.autograd.grad(torch.sum(calcEnergies ** 2), model.parameters(), retain_graph=True)
                end_time = time.time()
                print(f"Test, backprop on the MSE of all energies, elapsed time: {(end_time - start_time):.2f} seconds\n")

                for bandIdx, bandEne in enumerate(calcEnergies):
                    print(f"kIdx = {kidx}, bandIdx = {bandIdx}")
                    start_time = time.time() # if NNConfig['runtime_flag'] else None
                    # optimizer.zero_grad()
                    grads = torch.autograd.grad(bandEne, model.parameters(), retain_graph=True)
                    end_time = time.time()
                    print(f"Backprop on this energy, elapsed time: {(end_time - start_time):.2f} seconds")
                    
                    start_time = time.time()
                    tmp_grad = {}
                    for i, (name, param) in enumerate(model.named_parameters()):
                        if grads[i] is not None:
                            tmp_grad[name] = tmp_grad.get(name, 0) + grads[i].detach().clone()
                    lin_grad[kidx][bandIdx] = tmp_grad

                    tmp_grad = {}
                    for i, (name, param) in enumerate(model.named_parameters()):
                        if grads[i] is not None:
                            tmp_grad[name] = tmp_grad.get(name, 0) + 2 * bandEne.detach().clone() * grads[i].detach().clone()
                    quad_grad[kidx][bandIdx] = tmp_grad

                    bandEne.detach()
                    del bandEne
                    torch.cuda.empty_cache()
                    gc.collect()

                    end_time = time.time() # if NNConfig['runtime_flag'] else None
                    print(f"Storing all the gradients, elapsed time: {(end_time - start_time):.2f} seconds")
                    # print(f"Backprop on this energy + storing all the gradients, elapsed time: {(end_time - start_time):.2f} seconds") #  if NNConfig['runtime_flag'] else None

        else: # multiprocessing
            optimizer.zero_grad()
            args_list = [(kidx, hams[iSys], sys, optimizer, model, cachedMats_info) for kidx in range(sys.getNKpts())]
            with mp.Pool(NNConfig['num_cores']) as pool:
                results = pool.starmap(calcEigValsAtK_wGrad_parallel_smoothReorder, args_list)

                energies_detach_list, lin_grad, quad_grad = zip(*results)
                calcBS_nograd = torch.stack(energies_detach_list, dim=1)
            gc.collect()

        # Reorder the bands according to smoothness
        order_table, newBS, extrapolated_points = reorder_smoothness_deg2_tensors(calcBS_nograd)

        # Reorder the list of lists of dictionaries using order_table
        calcBS_nograd_reorder = calcBS_nograd[torch.arange(calcBS_nograd.size(0))[:, None], order_table]
        lin_grad_reorder = [[lin_grad[i][j] for j in order_table[i]] for i in range(len(lin_grad))]
        quad_grad_reorder = [[quad_grad[i][j] for j in order_table[i]] for i in range(len(quad_grad))]

        # Calculate the actual loss and gradients according to smoothness
        trainLoss_system = 0.0
        gradients_system = {key: 0.0 for key in lin_grad_reorder[0][0]}
        # Add weights to kpts and bands. bandWeights follow reference order! 
        bandWeights = sys.bandWeights
        kptWeights = sys.kptWeights
        if (len(bandWeights)!=nbands) or (len(kptWeights)!=nkpt): 
            raise ValueError("bandWeights or kptWeights lengths aren't correct. ")
        newBandWeights = bandWeights.view(1, -1).expand(nkpt, -1)
        newKptWeights = kptWeights.view(-1, 1).expand(-1, nbands)

        trainLoss_system = torch.sum((calcBS_nograd_reorder - sys.expBandStruct)**2 * newBandWeights * newKptWeights)

        for kIdx in range(nkpt):
            for bandIdx in range(nbands):
                for key in gradients_system:
                    gradients_system[key] += (quad_grad_reorder[kIdx][bandIdx][key]
                                             - 2*sys.expBandStruct[kIdx, bandIdx] * lin_grad_reorder[kIdx][bandIdx][key])* newBandWeights * newKptWeights

        
        total_gradients = merge_dicts([total_gradients, gradients_system])
        trainLoss += trainLoss_system

    # Write the manually accumulated gradients and loss values back into the NN model
    optimizer.zero_grad()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in total_gradients:
                param.grad = total_gradients[name].detach().clone()

    start_time = time.time() if NNConfig['runtime_flag'] else None
    optimizer.step()
    end_time = time.time() if NNConfig['runtime_flag'] else None
    print(f"optimizer step, elapsed time: {(end_time - start_time):.2f} seconds") if NNConfig['runtime_flag'] else None
    torch.cuda.empty_cache()

    return model, trainLoss


def bandStruct_train_GPU(model, device, NNConfig, systems, hams, atomPPOrder, criterion_singleSystem, criterion_singleKpt, optimizer, scheduler, val_dataset, resultsFolder, cachedMats_info=None):
    training_COST=[]
    validation_COST=[]
    file_trainCost = open(f'{resultsFolder}final_training_cost.dat', "w")
    file_valCost = open(f'{resultsFolder}final_validation_cost.dat', "w")
    model.to(device)
    best_validation_loss = float('inf')
    no_improvement_count = 0
    
    for epoch in range(NNConfig['max_num_epochs']):
        
        # train
        model.train()
        if NNConfig['separateKptGrad']==0: 
            if NNConfig['smooth_reorder']: 
                model, trainLoss, order_table, newBS, extrapolated_points = trainIter_naive_smoothReorder(model, systems, hams, criterion_singleSystem, optimizer, cachedMats_info, NNConfig['runtime_flag'])
                for bandIdx in range(newBS.shape[1]):
                    fig, ax = plotBandStruct_reorder(newBS.detach().numpy(), bandIdx)
                    ax.plot(np.arange(len(newBS)), extrapolated_points[:, bandIdx].detach().numpy(), "gx:", alpha=0.8, markersize=4)
                    ax.set(ylim=(min(extrapolated_points[:, bandIdx])-0.1, max(extrapolated_points[:, bandIdx])+0.1))
                    # plot_highlight_kpt(ax, [0,3,6,13,19,26,34,40,50,60,65,70,79,90,100,108])
                    fig.savefig(f"{resultsFolder}epoch_{epoch+1}_newBand_{bandIdx}.png")
                    fig.savefig(f"{resultsFolder}epoch_{epoch+1}_newBand_{bandIdx}.pdf")
                    plt.close()

            else: 
                model, trainLoss = trainIter_naive(model, systems, hams, criterion_singleSystem, optimizer, cachedMats_info, NNConfig['runtime_flag'])

        else:
            if NNConfig['smooth_reorder']: 
                model, trainLoss = trainIter_separateKptGrad_smoothReorder(model, systems, hams, NNConfig, optimizer, cachedMats_info)
            else: 
                model, trainLoss = trainIter_separateKptGrad(model, systems, hams, NNConfig, criterion_singleKpt, optimizer, cachedMats_info)

        training_COST.append(trainLoss.item())
        file_trainCost.write(f"{epoch+1}  {trainLoss.item()}\n")
        print(f"Epoch [{epoch+1}/{NNConfig['max_num_epochs']}], training cost: {trainLoss.item():.4f}")
        if (epoch<=9) or ((epoch + 1) % NNConfig['plotEvery'] == 0):
            print_and_inspect_gradients(model, f'{resultsFolder}epoch_{epoch+1}_gradients.dat', show=True)
            print_and_inspect_NNParams(model, f'{resultsFolder}epoch_{epoch+1}_params.dat', show=True)

        # perturb the model
        if (NNConfig['perturbEvery']>0) and (epoch>0) and (epoch % NNConfig['perturbEvery']==0): 
            perturb_model(model, 0.10)
            print("WARNING: We have randomly perturbed all the params of the model by 10%. \n")

        # scheduler of learning rate
        if (epoch > 0) and (epoch % NNConfig['schedulerStep'] == 0):
            scheduler.step()

        # evaluation
        if (epoch + 1) % NNConfig['plotEvery'] == 0:
            model.eval()
            val_MSE = evalBS_noGrad(model, f'{resultsFolder}epoch_{epoch+1}_plotBS.png', f'epoch_{epoch+1}', NNConfig, hams, systems, cachedMats_info, writeBS=True)
            
            validation_COST.append(val_MSE)
            print(f"Epoch [{epoch+1}/{NNConfig['max_num_epochs']}], validation cost: {val_MSE:.4f}")
            file_valCost.write(f"{epoch+1}  {val_MSE}\n")
            
            model.cpu()
            fig = plotPP(atomPPOrder, val_dataset.q, val_dataset.q, val_dataset.vq_atoms, model(val_dataset.q), "ZungerForm", f"NN_{epoch+1}", ["-",":" ]*len(atomPPOrder), True, NNConfig['SHOWPLOTS']);
            fig.savefig(f'{resultsFolder}epoch_{epoch+1}_plotPP.png')
            model.to(device)

            torch.save(model.state_dict(), f'{resultsFolder}epoch_{epoch+1}_PPmodel.pth')
            torch.save(optimizer.state_dict(), f'{resultsFolder}epoch_{epoch+1}_AdamState.pth')
            torch.cuda.empty_cache()
        '''
        # Dynamic stopping: Stop training if no improvement for 'patience' epochs
        if val_MSE < best_validation_loss - 1e-4:
            best_validation_loss = val_MSE
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >= NNConfig['patience']:
            print(f"Early stopping at Epoch {epoch} due to lack of improvement.")
            break
        '''
        plt.close('all')
        torch.cuda.empty_cache()
    fig_cost = plot_training_validation_cost(training_COST, validation_COST, True, NNConfig['SHOWPLOTS']);
    fig_cost.savefig(resultsFolder + 'final_train_cost.png')
    torch.cuda.empty_cache()
    return (training_COST, validation_COST)


def perturb_model(model, percentage=0.0): 
    print(f"Perturbing the model by percentage: {percentage}")
    for param in model.parameters():
        perturbation = 1 + torch.rand_like(param) * (2 * percentage) - percentage
        param.data *= perturbation
    return model


def runMC_NN(model, NNConfig, systems, hams, atomPPOrder, val_dataset, resultsFolder, cachedMats_info=None):
    file_trainCost = open(f'{resultsFolder}MC_training_cost.dat', "w")
    file_trainCost.write("# iter      newLoss      accept?      currLoss\n")
    
    bestModel = model
    bestLoss = evalBS_noGrad(bestModel, f'{resultsFolder}mc_iter_0_plotBS.png', f'mc_iter_0', NNConfig, hams, systems, cachedMats_info)
    currModel = model
    currLoss = bestLoss
    trial_COST = [currLoss]
    accepted_COST = [currLoss]

    for iter in range(NNConfig['mc_iter']):
        print(f"\nIteration [{iter+1}/{NNConfig['mc_iter']}]: ")
        newModel = perturb_model(currModel, percentage=NNConfig['mc_percentage'])
        newLoss = evalBS_noGrad(newModel, f'{resultsFolder}mc_iter_{iter+1}_plotBS.png', f'mc_iter_{iter+1}', NNConfig, hams, systems, cachedMats_info)
        print(f"newLoss={newLoss.item():.4f}. ")

        mc_rand = np.exp(-1 * NNConfig['mc_beta'] * (np.sqrt(newLoss) - np.sqrt(currLoss)))
        mc_accept_bool = mc_rand > np.random.uniform(low=0.0, high=1.0)

        if newLoss < bestLoss:   # accept
            bestLoss = newLoss
            bestModel = newModel
            currLoss = newLoss
            currModel = newModel
            file_trainCost.write(f"{iter+1}    {newLoss.item()}    {1}    {currLoss.item()}\n")
            print(f"Accepted. currLoss={currLoss.item():.4f}")
            print_and_inspect_NNParams(newModel, f'{resultsFolder}mc_iter_{iter+1}_params.dat', show=True)

            fig = plotPP(atomPPOrder, val_dataset.q, val_dataset.q, val_dataset.vq_atoms, currModel(val_dataset.q), "ZungerForm", f"mc_iter_{iter+1}", ["-",":" ]*len(atomPPOrder), True, NNConfig['SHOWPLOTS']);
            fig.savefig(f'{resultsFolder}mc_iter_{iter+1}_plotPP.png')
            torch.save(currModel.state_dict(), f'{resultsFolder}mc_iter_{iter+1}_PPmodel.pth')
        elif mc_accept_bool:   # new loss is higher, but we still accept.
            currLoss = newLoss
            currModel = newModel
            file_trainCost.write(f"{iter+1}    {newLoss.item()}    {1}    {currLoss.item()}\n")
            print(f"Accepted. currLoss={currLoss.item():.4f}")

            fig = plotPP(atomPPOrder, val_dataset.q, val_dataset.q, val_dataset.vq_atoms, currModel(val_dataset.q), "ZungerForm", f"mc_iter_{iter+1}", ["-",":" ]*len(atomPPOrder), True, NNConfig['SHOWPLOTS']);
            fig.savefig(f'{resultsFolder}mc_iter_{iter+1}_plotPP.png')
            torch.save(currModel.state_dict(), f'{resultsFolder}mc_iter_{iter+1}_PPmodel.pth')
        else:   # don't accept
            file_trainCost.write(f"{iter+1}    {newLoss.item()}    {0}    {currLoss.item()}\n")
            print(f"Not accepted. currLoss={currLoss.item():.4f}")
        
        trial_COST.append(newLoss.item())
        accepted_COST.append(currLoss.item())
    
        plt.close('all')
        torch.cuda.empty_cache()

    model = currModel
        
    fig_cost = plot_mc_cost(trial_COST, accepted_COST, True, NNConfig['SHOWPLOTS']);
    fig_cost.savefig(resultsFolder + 'final_mc_cost.png')
    file_trainCost.close()
    return (trial_COST, accepted_COST)