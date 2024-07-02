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
import random
import shutil

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from .constants import *
from .pp_func import plotPP, plot_training_validation_cost, plotBandStruct, plot_mc_cost, plotBandStruct_reorder
from .smooth_order import reorder_smoothness_deg2_tensors, reorder_kpt_smoothness_deg2_tensors

def print_and_inspect_gradients(model, filename=None, show=False): 
    """
    Prints and/or saves the gradients of the model parameters.

    If 'filename' is provided and 'show' is True, it saves the gradients to the file.
    If 'filename' is None and 'show' is True, it prints the gradients.
    """
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
    """
    Prints and/or saves the values of the model parameters.

    If 'filename' is provided and 'show' is True, it saves the parameters to the file.
    If 'filename' is None and 'show' is True, it prints the parameters.
    """
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


def get_max_gradient_param(model):
    """
    Returns the parameter that has the largest gradient, in terms of the 
    parameter tensor's name in the dictionary, the index within this tensor, 
    and the value of the gradient. 

    Later, one can access this parameter using: 
    dict(model.named_parameters())[max_grad_name].grad[max_grad_index]
    """
    gradients_populated = any(param.grad is not None for param in model.parameters())
    if not gradients_populated:
        raise ValueError("Gradients have not been populated. Ensure that a backward pass has been performed.")

    max_grad = None
    max_grad_index = None
    max_grad_name = None

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_abs_max_value = param.grad.abs().max().item()
            if max_grad is None or grad_abs_max_value > max_grad:
                max_grad = grad_abs_max_value
                max_grad_name = name
                max_grad_index = param.grad.abs().argmax().item()

    if max_grad_name is not None:
        param = dict(model.named_parameters())[max_grad_name]
        max_grad_value = param.grad.flatten()[max_grad_index].clone()
        max_grad_index = np.unravel_index(max_grad_index, param.grad.shape)
        # print(f"Values returned by the get_max_gradient_param function: {max_grad_name}, {max_grad_index}, {max_grad_value}")
        return max_grad_name, max_grad_index, max_grad_value
    else:
        return None, None, None


def judge_well_conditioned_grad(model, maxGradThreshold=50.0): 
    maxGrad = None
    minGrad = None
    for _, param in model.named_parameters():
        if param.grad is not None:
            if maxGrad is None or param.grad.abs().max().item() > maxGrad:
                maxGrad = param.grad.abs().max().item()
            if minGrad is None or param.grad.abs().min().item() > minGrad:
                minGrad = param.grad.abs().min().item()
    print(f"Max and min of absolute gradients = {maxGrad:.3f}, {minGrad:.3f}.   Are the gradients well-conditioned? {maxGrad<=maxGradThreshold}")
    return maxGrad, minGrad


def manual_GD_one_param(model, stepSize=None):
    """
    Make a manual gradient descent move on ONLY ONE parameter that has the largest
    absolute gradient value. This is designed to slowly yet surely optimize to the
    nearest local minimum on a multi-dimensional function space. The model is 
    changed in-place. 

    One can give an optional stepSize parameter. If not used, the manual 
    optimization steps (lr * grad) is hard-coded to be around 0.005
    """
    max_grad_name, max_grad_index, max_grad_value = get_max_gradient_param(model)
    
    if max_grad_name is None:
        raise ValueError("No maximum gradient found in the model. (Meaning that there were no gradients in the model).")

    # Zero all gradients, except for the one with maximum gradient
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.zero_()
        if name==max_grad_name: 
            param.grad[max_grad_index] = max_grad_value.item()

    # Set the learning rate, ensuring max_grad_value is used appropriately
    if stepSize is None:
        stepSize = 0.005
    learning_rate = stepSize * random.random() / abs(max_grad_value.item())

    # Perform the manual SGD step
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                param -= learning_rate * param.grad


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
        print(f"\t{runName}: Evaluating band structures using the NN-pp model. ")
        model.eval()
    else:
        print(f"\t{runName}: Evaluating band structures using the old Zunger function form. ")
    
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
        if writeBS: 
            if not BSplotFilename.endswith('_plotBS.png'):
                raise ValueError("BSplotFilename must end with '_plotBS.png' to write BS.dat files. ")
            else:
                write_BS_filename = BSplotFilename.replace('_plotBS.png', f'_BS_sys{iSys}.dat')
            kptDistInputs_vertical = sys.kptDistInputs.view(-1, 1)
            write_tensor = torch.cat((kptDistInputs_vertical, evalBS), dim=1)
            np.savetxt(write_BS_filename, write_tensor, fmt='%.5f')
            # print(f"\t{runName}: Wrote BS to file {write_BS_filename}. ")
        plot_bandStruct_list.append(sys.expBandStruct)
        plot_bandStruct_list.append(evalBS)
        totalMSE += weighted_mse_bandStruct(evalBS, sys)
    fig = plotBandStruct(systems, plot_bandStruct_list, NNConfig['SHOWPLOTS'])
    print(f"\t{runName}: Finished evaluating {iSys}-th band structure with no gradient... Elapsed time: {(end_time - start_time):.2f} seconds. TotalMSE = {totalMSE:f}")
    fig.suptitle(f"{runName}: totalMSE = {totalMSE:f}")
    fig.savefig(BSplotFilename)
    plt.close('all')
    torch.cuda.empty_cache()
    return totalMSE


def calcEigValsAtK_wGrad_parallel(kidx, ham, bulkSystem, criterion_singleKpt, optimizer, model, cachedMats_info=None, prevBS=None, verbosity=0):
    """
    loop over kidx
    The rest of the arguments are "constants" / "constant functions" for a single kidx
    For performance, it is recommended that the ham in the argument doesn't have SOmat and NLmat initialized. 
    """
    singleKptGradients = {}

    calcEnergies = ham.calcEigValsAtK(kidx, cachedMats_info, requires_grad=True)
    extrapolated_eigVal = calcEnergies.clone()
    if ham.NNConfig['smooth_reorder']: 
        col_ind, calcEnergies, extrapolated_eigVal = reorder_kpt_smoothness_deg2_tensors(calcEnergies, kidx, comparedBS=prevBS.detach() if prevBS is not None else None)

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

    calcEnergies = calcEnergies.detach()
    extrapolated_eigVal = extrapolated_eigVal.detach()
    return singleKptGradients, trainLoss_systemKpt, calcEnergies, extrapolated_eigVal


def trainIter_naive(model, systems, hams, criterion_singleSystem, optimizer, cachedMats_info=None, runtime_flag=False, preAdjustBool=False, preAdjustStepSize=None, resultsFolder=None, pre_epoch=0, epoch=0, verbosity=1):
    trainLoss = torch.tensor(0.0)
    for iSys, sys in enumerate(systems):
        hams[iSys].NN_locbool = True
        hams[iSys].set_NNmodel(model)

        NN_outputs = hams[iSys].calcBandStruct_withGrad(cachedMats_info)

        # reorder NN_outputs if the keyword is turned on
        if hams[iSys].NNConfig['smooth_reorder']: 
            order_table, newBS, extrapolated_points = reorder_smoothness_deg2_tensors(NN_outputs)
            NN_outputs = newBS

            # Plot each individual band for debugging
            if verbosity>=1: 
                for bandIdx in range(newBS.shape[1]):
                    fig, ax = plotBandStruct_reorder(newBS.detach().numpy(), bandIdx)
                    ax.plot(np.arange(len(newBS)), extrapolated_points[:, bandIdx].detach().numpy(), "gx:", alpha=0.8, markersize=4)
                    ax.set(ylim=(min(extrapolated_points[:, bandIdx])-0.1, max(extrapolated_points[:, bandIdx])+0.1))
                    # plot_highlight_kpt(ax, [0,3,6,13,19,26,34,40,50,60,65,70,79,90,100,108])
                    fig.savefig(f"{resultsFolder}epoch_{epoch+1}_newBand_{bandIdx}.png")
                    fig.savefig(f"{resultsFolder}epoch_{epoch+1}_newBand_{bandIdx}.pdf")
                    plt.close()
        
        systemLoss = criterion_singleSystem(NN_outputs, sys)
        trainLoss += systemLoss

    start_time = time.time() if runtime_flag else None
    optimizer.zero_grad()
    trainLoss.backward()
    if preAdjustBool: 
        manual_GD_one_param(model, preAdjustStepSize)
    else:
        optimizer.step()
    end_time = time.time() if runtime_flag else None
    print(f"loss_backward + optimizer.step, elapsed time: {(end_time - start_time):.2f} seconds") if runtime_flag else None

    torch.cuda.empty_cache()
    return model, trainLoss


def trainIter_separateKptGrad(model, systems, hams, NNConfig, criterion_singleKpt, optimizer, cachedMats_info=None, preAdjustBool=False, preAdjustStepSize=None, resultsFolder=None, pre_epoch=0, epoch=0, verbosity=1, prevBS=None): 
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
            currBS = torch.zeros([sys.getNKpts(), sys.nBands])
            extrapolated_points = torch.zeros([sys.getNKpts(), sys.nBands])
            for kidx in range(sys.getNKpts()): 
                calcEnergies = hams[iSys].calcEigValsAtK(kidx, cachedMats_info, requires_grad=True)

                extrapolated_eigVal = calcEnergies.detach().clone()
                if NNConfig['smooth_reorder']: 
                    col_ind, calcEnergies, extrapolated_eigVal = reorder_kpt_smoothness_deg2_tensors(calcEnergies, kidx, comparedBS=prevBS.detach() if prevBS is not None else None)
                    extrapolated_points[kidx,:] = extrapolated_eigVal.detach().clone()

                systemKptLoss = criterion_singleKpt(calcEnergies, sys, kidx)
                currBS[kidx,:] = calcEnergies.detach().clone()

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
                
            if (NNConfig['smooth_reorder']) and (prevBS is not None): 
                print("WARNING. We are reordering the band structure according to smoothness using the previous iteration BS. ")
            prevBS = prevBS.detach() if prevBS is not None else None
            args_list = [(kidx, hams[iSys], sys, criterion_singleKpt, optimizer, model, cachedMats_info, prevBS) for kidx in range(sys.getNKpts())]

            with mp.Pool(NNConfig['num_cores']) as pool:
                results_systemKpt = pool.starmap(calcEigValsAtK_wGrad_parallel, args_list)
                gradients_systemKpt, trainLoss_systemKpt, eigValsList, extrapolated_eigValList = zip(*results_systemKpt)
            currBS = torch.stack(eigValsList).detach()
            extrapolated_points = torch.stack(extrapolated_eigValList).detach()

            gc.collect()
            gradients_system = merge_dicts(gradients_systemKpt)
            trainLoss_system = torch.sum(torch.tensor(trainLoss_systemKpt))
        
        total_gradients = merge_dicts([total_gradients, gradients_system])
        trainLoss += trainLoss_system

        # Plot each individual band for debugging
        if (NNConfig['smooth_reorder']) and (verbosity>=1): 
            for bandIdx in range(currBS.shape[1]):
                fig, ax = plotBandStruct_reorder(currBS.detach().numpy(), bandIdx)
                ax.plot(np.arange(len(currBS)), extrapolated_points[:, bandIdx].detach().numpy(), "gx:", alpha=0.8, markersize=4)
                ax.set(ylim=(min(extrapolated_points[:, bandIdx])-0.1, max(extrapolated_points[:, bandIdx])+0.1))
                # plot_highlight_kpt(ax, [0,3,6,13,19,26,34,40,50,60,65,70,79,90,100,108])
                fig.savefig(f"{resultsFolder}epoch_{epoch+1}_newBand_{bandIdx}.png")
                fig.savefig(f"{resultsFolder}epoch_{epoch+1}_newBand_{bandIdx}.pdf")
                plt.close()

    # Write the manually accumulated gradients and loss values back into the NN model
    optimizer.zero_grad()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in total_gradients:
                param.grad = total_gradients[name].detach().clone()

    start_time = time.time() if NNConfig['runtime_flag'] else None
    if preAdjustBool: 
        if verbosity>1:
            print_and_inspect_gradients(model, f'{resultsFolder}preEpoch_{pre_epoch+1}_before_gradients.dat', show=True)
            print_and_inspect_NNParams(model, f'{resultsFolder}preEpoch_{pre_epoch+1}_before_params.dat', show=True)
        manual_GD_one_param(model, preAdjustStepSize)
    else:
        optimizer.step()
    end_time = time.time() if NNConfig['runtime_flag'] else None
    print(f"optimizer step, elapsed time: {(end_time - start_time):.2f} seconds") if NNConfig['runtime_flag'] else None

    torch.cuda.empty_cache()
    # print_and_inspect_gradients(model, show=NNConfig['printGrad'])

    return model, trainLoss, currBS


def bandStruct_train_GPU(model, device, NNConfig, systems, hams, atomPPOrder, criterion_singleSystem, criterion_singleKpt, optimizer, scheduler, val_dataset, resultsFolder, cachedMats_info=None):
    trainingCOST_x =[]
    training_COST = []
    validationCOST_x = []
    validation_COST =[]
    file_trainCost = open(f'{resultsFolder}final_training_cost.dat', "w")
    file_valCost = open(f'{resultsFolder}final_validation_cost.dat', "w")
    model.to(device)
    best_validation_loss = float('inf')
    no_improvement_count = 0
    prevBS = None

    pre_min_maxGrad = None
    pre_min_epoch = None
    # pre_adjustments. Optimizing only ONE PARAMETER at a time, which has the largest gradient
    if ('pre_adjust_moves' in NNConfig) and (NNConfig['pre_adjust_moves']>0): 
        for pre_epoch in range(NNConfig['pre_adjust_moves']):
            if ('pre_adjust_stepSize' in NNConfig): 
                pre_adjust_stepSize = NNConfig['pre_adjust_stepSize']
            else: 
                pre_adjust_stepSize = None

            model.train()

            if NNConfig['separateKptGrad']==0: 
                model, trainLoss = trainIter_naive(model, systems, hams, criterion_singleSystem, optimizer, cachedMats_info, NNConfig['runtime_flag'], preAdjustBool=True, preAdjustStepSize=pre_adjust_stepSize, resultsFolder=resultsFolder, pre_epoch=pre_epoch)
            else: 
                model, trainLoss, prevBS = trainIter_separateKptGrad(model, systems, hams, NNConfig, criterion_singleKpt, optimizer, cachedMats_info, preAdjustBool=True, preAdjustStepSize=pre_adjust_stepSize, resultsFolder=resultsFolder, pre_epoch=pre_epoch, prevBS=prevBS.detach() if prevBS is not None else None)

            file_trainCost.write(f"{pre_epoch-NNConfig['pre_adjust_moves']-1}  {trainLoss.item()}\n")
            trainingCOST_x.append(pre_epoch-NNConfig['pre_adjust_moves']-1)
            training_COST.append(trainLoss.item())
            print(f"pre_adjust_moves [{pre_epoch+1}/{NNConfig['pre_adjust_moves']}], training cost: {trainLoss.item():.4f}")
            # print_and_inspect_gradients(model, f'{resultsFolder}preEpoch_{pre_epoch+1}_after_gradients.dat', show=True)
            # print_and_inspect_NNParams(model, f'{resultsFolder}preEpoch_{pre_epoch+1}_after_params.dat', show=True)

            model.eval()
            val_MSE = evalBS_noGrad(model, f'{resultsFolder}preEpoch_{pre_epoch+1}_plotBS.png', f'preEpoch_{pre_epoch+1}', NNConfig, hams, systems, cachedMats_info, writeBS=True)

            torch.save(model.state_dict(), f'{resultsFolder}preEpoch_{pre_epoch+1}_PPmodel.pth')
            torch.cuda.empty_cache()

            maxGrad, _ = judge_well_conditioned_grad(model)
            if pre_min_maxGrad is None or maxGrad <= pre_min_maxGrad:
                print("This is the best pre-adjust epoch so far. ")
                pre_min_maxGrad = maxGrad
                pre_min_epoch = pre_epoch
            print()
        
        model.load_state_dict(torch.load(f'{resultsFolder}preEpoch_{pre_min_epoch+1}_PPmodel.pth'))
        print(f"We have re-loaded back to the preEpoch_{pre_min_epoch+1}, which gives the best-conditioned gradients. ")

    for epoch in range(NNConfig['max_num_epochs']):

        # train
        model.train()
        if NNConfig['separateKptGrad']==0: 
            model, trainLoss = trainIter_naive(model, systems, hams, criterion_singleSystem, optimizer, cachedMats_info, NNConfig['runtime_flag'], resultsFolder=resultsFolder, epoch=epoch)
        else: 
            model, trainLoss, prevBS = trainIter_separateKptGrad(model, systems, hams, NNConfig, criterion_singleKpt, optimizer, cachedMats_info, resultsFolder=resultsFolder, epoch=epoch, prevBS=prevBS.detach() if prevBS is not None else None)
        file_trainCost.write(f"{epoch+1}  {trainLoss.item()}\n")
        trainingCOST_x.append(epoch+1)
        training_COST.append(trainLoss.item())
        print(f"Epoch [{epoch+1}/{NNConfig['max_num_epochs']}], training cost: {trainLoss.item():.4f}")
        if (epoch<=9) or ((epoch + 1) % NNConfig['plotEvery'] == 0):
            print_and_inspect_gradients(model, f'{resultsFolder}epoch_{epoch+1}_gradients.dat', show=True)
            print_and_inspect_NNParams(model, f'{resultsFolder}epoch_{epoch+1}_params.dat', show=True)
        judge_well_conditioned_grad(model)

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
            validationCOST_x.append(epoch+1)
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
    fig_cost = plot_training_validation_cost(trainingCOST_x, training_COST, validation_cost_x=validationCOST_x, validation_cost=validation_COST, ylogBoolean=True, SHOWPLOTS=NNConfig['SHOWPLOTS']);
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

            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_PPmodel.pth', f'{resultsFolder}final_PPmodel.pth')
            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_plotPP.png', f'{resultsFolder}final_plotPP.png')
            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_plotBS.png', f'{resultsFolder}final_plotBS.png')
        elif mc_accept_bool:   # new loss is higher, but we still accept.
            currLoss = newLoss
            currModel = newModel
            file_trainCost.write(f"{iter+1}    {newLoss.item()}    {1}    {currLoss.item()}\n")
            print(f"Accepted. currLoss={currLoss.item():.4f}")

            fig = plotPP(atomPPOrder, val_dataset.q, val_dataset.q, val_dataset.vq_atoms, currModel(val_dataset.q), "ZungerForm", f"mc_iter_{iter+1}", ["-",":" ]*len(atomPPOrder), True, NNConfig['SHOWPLOTS']);
            fig.savefig(f'{resultsFolder}mc_iter_{iter+1}_plotPP.png')
            torch.save(currModel.state_dict(), f'{resultsFolder}mc_iter_{iter+1}_PPmodel.pth')

            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_PPmodel.pth', f'{resultsFolder}final_PPmodel.pth')
            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_plotPP.png', f'{resultsFolder}final_plotPP.png')
            shutil.copy(f'{resultsFolder}mc_iter_{iter+1}_plotBS.png', f'{resultsFolder}final_plotBS.png')
        else:   # don't accept
            file_trainCost.write(f"{iter+1}    {newLoss.item()}    {0}    {currLoss.item()}\n")
            print(f"Not accepted. currLoss={currLoss.item():.4f}")
        
        trial_COST.append(newLoss.item())
        accepted_COST.append(currLoss.item())
    
        plt.close('all')
        torch.cuda.empty_cache()

    model = currModel
        
    fig_cost = plot_mc_cost(trial_COST, accepted_COST, True, NNConfig['SHOWPLOTS']);
    fig_cost.savefig(f'{resultsFolder}final_mc_cost.png')
    file_trainCost.close()
    return (trial_COST, accepted_COST)