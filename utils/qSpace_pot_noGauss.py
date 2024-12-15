import os
import torch
import numpy as np

from .read import read_NNConfigFile, setAllBulkSystems, setNN
from .NN_train import write_PP_qSpace


if __name__ == "__main__":
    inputsFolder = "CALCS/CsPbI3_ultraSmall_round2/mc1_heat5_anneal4_inputs/"
    resultsFolder = "CALCS/CsPbI3_ultraSmall_round2/mc1_heat5_anneal4_results/"
    
    # initialize the NN model (without Gaussian)
    NNConfig = read_NNConfigFile(inputsFolder + 'NN_config.par')
    nSystem = NNConfig['nSystem']
    NNConfig['PPmodel'] = 'Net_celu_RandInit'

    # Read and set up systems
    print(f"\nReading and setting up the BulkSystems.")
    systems, atomPPOrder, nPseudopot, PPparams, totalParams, localPotParams = setAllBulkSystems(nSystem, inputsFolder, resultsFolder)

    # Set up the neural network
    PPmodel = setNN(NNConfig, nPseudopot)

    # read and load the .pth NN from MC or training (with Gaussian)
    if os.path.exists(inputsFolder + 'init_PPmodel.pth'):
        print(f"\n{'#' * 40}\nInitializing the NN with file {inputsFolder}init_PPmodel.pth.")
        # PPmodel.load_state_dict(torch.load(inputsFolder + 'init_PPmodel.pth'))

        state_dict = torch.load(inputsFolder + 'init_PPmodel.pth')
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('neural_network.', '')
            new_state_dict[new_key] = v
        PPmodel.load_state_dict(new_state_dict)

    # Print out the qSpace
    write_PP_qSpace(f'{resultsFolder}qSpace_pot_noGaussian.dat', PPmodel, atomPPOrder)

    # Next step: do real space too 