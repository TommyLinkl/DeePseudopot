import torch

from utils.nn_models import *

layers = [1,32,32,32,3]
PPmodel = Net_relu_xavier_decayGaussian(layers, gaussian_std=2.0)

optimizer = torch.optim.Adam(PPmodel.parameters(), lr=0.00001)
optimizer.load_state_dict(torch.load('CALCS/CsPbI3_16kpts/inputs_5_wAdam/init_AdamState.pth'))

# Accessing the parameters of the optimizer
for i, param_group in enumerate(optimizer.param_groups):
    print(f"Parameter group {i}:")
    print("Learning rate:", param_group['lr'])

    # Accessing the momentum (beta1), beta2, and epsilon of the optimizer for Adam optimizer
    for param_name, param_value in param_group.items():
        if param_name == 'betas':
            print(f"Momentum (beta1): {param_value[0]}")
            print(f"Squared momentum (beta2): {param_value[1]}")
        elif param_name == 'eps':
            print(f"Epsilon: {param_value}")

