import torch
import torch.nn as nn
import random
import numpy as np

torch.set_default_dtype(torch.float64)

# this model accepts a vector for the layers, i.e. [inp, hidden1, hidden2,...hiddenM,out]
# with sigmoid activation
class Net_sig(nn.Module):
    # Constructor
    def __init__(self, Layers):
        super(Net_sig, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)

    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.sigmoid(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

class Net_tanh(nn.Module):
    def __init__(self, Layers,p=0):
        super(Net_tanh, self).__init__()
        self.drop=nn.Dropout(p=p)
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = self.drop(torch.tanh(linear_transform(x)))
            else:
                x = linear_transform(x)
        return x

class Net_relu(nn.Module):
    def __init__(self, Layers):
        super(Net_relu, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.relu(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

class Net_relu_xavier(nn.Module):
    def __init__(self, Layers):
        super(Net_relu_xavier, self).__init__()
        self.hidden_l = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.xavier_normal_(linear.weight)
            self.hidden_l.append(linear)

    def forward(self, x):
        L = len(self.hidden_l)
        for (l, linear_transform) in zip(range(L), self.hidden_l):
            if l < L - 1:
                x = torch.relu(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

class Net_sigmoid_xavier(nn.Module):
    def __init__(self, Layers):
        super(Net_sigmoid_xavier, self).__init__()
        self.hidden_l = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.xavier_normal_(linear.weight)
            self.hidden_l.append(linear)

    def forward(self, x):
        L = len(self.hidden_l)
        for (l, linear_transform) in zip(range(L), self.hidden_l):
            if l < L - 1:
                x = torch.sigmoid(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

class Net_celu_HeInit(nn.Module):
    def __init__(self, Layers):
        super(Net_celu_HeInit, self).__init__()
        self.hidden_l = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            self.hidden_l.append(linear)

    def forward(self, x):
        L = len(self.hidden_l)
        for (l, linear_transform) in zip(range(L), self.hidden_l):
            if l < L - 1:
                x = nn.CELU()(linear_transform(x))
            else:
                x = linear_transform(x)
        return x


class Net_celu_RandInit(nn.Module):
    def __init__(self, Layers):
        super(Net_celu_RandInit, self).__init__()
        self.hidden_l = nn.ModuleList()

        # Save the current random state
        current_random_state = torch.get_rng_state()
        current_numpy_state = np.random.get_state()
        current_python_state = random.getstate()

        # Set a new random seed
        torch.manual_seed(random.randint(0, 2**32 - 1))

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            # Custom normal distribution initialization
            std = 1.0 / (input_size ** 0.5)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            self.hidden_l.append(linear)

        # Restore the original random state
        torch.set_rng_state(current_random_state)
        np.random.set_state(current_numpy_state)
        random.setstate(current_python_state)

    def forward(self, x):
        L = len(self.hidden_l)
        for (l, linear_transform) in zip(range(L), self.hidden_l):
            if l < L - 1:
                x = nn.CELU()(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

# sigmoid activation + uniform initialization
class Net_sig_UniformInit(nn.Module):
    def __init__(self, Layers):
        super(Net_sig_UniformInit, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            linear.weight.data.uniform_(0, 1)
            self.hidden.append(linear)

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.sigmoid(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

# tanh activation + Xavier initialization
# "p" is a dropout parameter, i.e. we randomly "switch off" neurons at probability p
class Net_tanh_XavierInit_dropout(nn.Module):
    def __init__(self, Layers, p=0):
        super(Net_tanh_XavierInit_dropout, self).__init__()
        self.drop=nn.Dropout(p=p)
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip( Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.xavier_uniform_(linear.weight)
            self.hidden.append(linear)

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = self.drop(torch.tanh(linear_transform(x)))
            else:
                x = linear_transform(x)
        return x

# Relu activation + He (Kaiming) initialization
class Net_relu_HeInit(nn.Module):
    def __init__(self, Layers):
        super(Net_relu_HeInit, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            self.hidden.append(linear)

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.relu(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

# sigmoid activation + batch normalization + dropout
class Net_sig_bn(nn.Module):
    def __init__(self, Layers, p=0):
        super(Net_sig_bn, self).__init__()
        self.drop=nn.Dropout(p=p)
        self.hidden_l = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden_l.append(linear)
            batchnorm = nn.BatchNorm1d(output_size)
            self.hidden_bn.append(batchnorm)

    def forward(self, x):
        L = len(self.hidden_l)
        for (l, linear_transform, bn) in zip(range(L), self.hidden_l, self.hidden_bn):
            if l < L - 1:
                x = torch.sigmoid(bn(linear_transform(x)))
            else:
                x = linear_transform(x)
        return x

# with tanh activation + batch normalization + dropout
class Net_tanh_bn(nn.Module):
    def __init__(self, Layers,p=0):
        super(Net_tanh_bn, self).__init__()
        self.drop=nn.Dropout(p=p)
        self.hidden_l = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden_l.append(linear)
            batchnorm = nn.BatchNorm1d(output_size)
            self.hidden_bn.append(batchnorm)

    def forward(self, x):
        L = len(self.hidden_l)
        for (l, linear_transform, bn) in zip(range(L), self.hidden_l, self.hidden_bn):
            if l < L - 1:
                x = torch.tanh(bn(linear_transform(x)))
            else:
                x = linear_transform(x)
        return x

# Relu activation + batch normalization + dropout
class Net_relu_bn(nn.Module):
    def __init__(self, Layers,p=0):
        super(Net_relu_bn, self).__init__()
        self.drop=nn.Dropout(p=p)
        self.hidden_l = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden_l.append(linear)
            batchnorm = nn.BatchNorm1d(output_size)
            self.hidden_bn.append(batchnorm)

    def forward(self, x):
        L = len(self.hidden_l)
        for (l, linear_transform, bn) in zip(range(L), self.hidden_l, self.hidden_bn):
            if l < L - 1:
                x = torch.relu(bn(linear_transform(x)))
            else:
                x = linear_transform(x)
        return x

# tanh activation + Xavier intialization + batch normalization + dropout
class Net_Xavier_BN(nn.Module):
    # Constructor
    def __init__(self, Layers,p=0):
        super(Net_Xavier_BN, self).__init__()
        self.drop=nn.Dropout(p=p)
        self.hidden_l = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.xavier_uniform_(linear.weight)
            self.hidden_l.append(linear)
            batchnorm = nn.BatchNorm1d(output_size)
            self.hidden_bn.append(batchnorm)
    # Prediction
    def forward(self, x):
        L = len(self.hidden_l)
        for (l, linear_transform, bn) in zip(range(L), self.hidden_l, self.hidden_bn):
            if l < L - 1:
                x = torch.tanh(bn(linear_transform(x)))
            else:
                x = linear_transform(x)
        return x

# Relu activation + He (Kaiming) intialization + batch normalization + dropout
class Net_He_BN(nn.Module):
    # Constructor
    def __init__(self, Layers,p=0):
        super(Net_He_BN, self).__init__()
        self.drop=nn.Dropout(p=p)
        self.hidden_l = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            self.hidden_l.append(linear)
            batchnorm = nn.BatchNorm1d(output_size)
            self.hidden_bn.append(batchnorm)
    # Prediction
    def forward(self, x):
        L = len(self.hidden_l)
        for (l, linear_transform, bn) in zip(range(L), self.hidden_l, self.hidden_bn):
            if l < L - 1:
                x = torch.relu(bn(linear_transform(x)))
            else:
                x = linear_transform(x)
        return x
    
#######################################################################################################
    
# this model accepts a vector for the layers, i.e. [inp, hidden1, hidden2,...hiddenM,out]
# Relu activation + Xavier intialization + batch normalization + dropout
class Net_relu_xavier_BN_dropout(nn.Module):
    # Constructor
    def __init__(self, Layers, p=0):
        super(Net_relu_xavier_BN_dropout, self).__init__()
        self.drop=nn.Dropout(p=p)
        self.hidden_l = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.xavier_normal_(linear.weight)     # xavier_uniform_
            self.hidden_l.append(linear)
            batchnorm = nn.BatchNorm1d(output_size)
            self.hidden_bn.append(batchnorm)
    # Prediction
    def forward(self, x):
        L = len(self.hidden_l)
        for (l, linear_transform, bn) in zip(range(L), self.hidden_l, self.hidden_bn):
            if l < L - 1:
                x = torch.relu(bn(linear_transform(x)))
                x = self.drop(x)
            else:
                x = linear_transform(x)
        return x
    

class Net_relu_xavier_BN(nn.Module):
    # Constructor
    def __init__(self, Layers):
        super(Net_relu_xavier_BN, self).__init__()
        self.hidden_l = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.xavier_normal_(linear.weight)     # xavier_uniform_
            self.hidden_l.append(linear)
            batchnorm = nn.BatchNorm1d(output_size)
            self.hidden_bn.append(batchnorm)
    
    # Prediction
    def forward(self, x):
        L = len(self.hidden_l)
        for (l, linear_transform, bn) in zip(range(L), self.hidden_l, self.hidden_bn):
            if l < L - 1:
                x = torch.relu(bn(linear_transform(x)))
            else:
                x = linear_transform(x)
        return x


class ZeroFunction(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        return torch.zeros(batch_size, 2, dtype=x.dtype, device=x.device)

class Net_relu_xavier_BN_dropout_decay(nn.Module):
    def __init__(self, Layers, p=0):
        super(Net_relu_xavier_BN_dropout_decay, self).__init__()
        self.neural_network = Net_relu_xavier_BN_dropout(Layers, p)
        self.mathematical_function = ZeroFunction()
    
    def forward(self, x):
        neural_network_output = self.neural_network(x[x <= 10].view(-1, 1))
        mathematical_function_output = self.mathematical_function(x[x > 10].view(-1, 1))
        output = torch.cat((neural_network_output, mathematical_function_output), dim=0)
        return output
    
class Net_relu_xavier_decay(nn.Module):
    def __init__(self, Layers, decay_rate, decay_center):
        super(Net_relu_xavier_decay, self).__init__()
        self.neural_network = Net_relu_xavier(Layers)
        self.decay_rate = torch.tensor(decay_rate, requires_grad=False)
        self.decay_center = torch.tensor(decay_center, requires_grad=False)
    
    def forward(self, x):
        decay = 1 - 1 / (1 + torch.exp(-self.decay_rate * (x - self.decay_center)))
        output = self.neural_network(x) * decay
        return output
    
class Net_relu_xavier_decay2(nn.Module):
    def __init__(self, Layers):
        super(Net_relu_xavier_decay2, self).__init__()
        self.neural_network = Net_relu_xavier(Layers)
    
    def forward(self, x):
        decay = 1 - 1 / (1 + torch.exp(-1.5 * (x - 6)))
        output = self.neural_network(x) * decay
        return output
    
class Net_celu_HeInit_decay(nn.Module):
    def __init__(self, Layers, decay_rate, decay_center):
        super(Net_celu_HeInit_decay, self).__init__()
        self.neural_network = Net_celu_HeInit(Layers)
        self.decay_rate = torch.tensor(decay_rate, requires_grad=False)
        self.decay_center = torch.tensor(decay_center, requires_grad=False)
    
    def forward(self, x):
        decay = 1 - 1 / (1 + torch.exp(-self.decay_rate * (x - self.decay_center)))
        output = self.neural_network(x) * decay
        return output
    
class Net_celu_HeInit_decay_LSD(nn.Module):
    def __init__(self, Layers, decay_rate, decay_center):
        super(Net_celu_HeInit_decay, self).__init__()
        self.neural_network = Net_celu_HeInit(Layers)
        self.decay_rate = torch.tensor(decay_rate, requires_grad=False)
        self.decay_center = torch.tensor(decay_center, requires_grad=False)
    
        # zero-init the final layer of the subnetwork
        last_layer = self.neural_network.hidden_l[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        
    def forward(self, x):
        q = x[:, 1].unsqueeze(1) # only apply Gaussian decay on q
        decay = 1 - 1 / (1 + torch.exp(-self.decay_rate * (q - self.decay_center)))
        output = self.neural_network(x) * decay
        return output
    
class Net_relu_xavier_decayGaussian(nn.Module):
    def __init__(self, Layers, gaussian_std):
        super(Net_relu_xavier_decayGaussian, self).__init__()
        self.neural_network = Net_relu_xavier(Layers)
        self.gaussian_std = torch.tensor(gaussian_std, requires_grad=False)
    
    def forward(self, x):
        gaussian = torch.exp(-x**2/(2*self.gaussian_std**2))
        output = self.neural_network(x) * gaussian
        return output

class Net_relu_xavier_decayGaussian_LSD(nn.Module):
    def __init__(self, Layers, gaussian_std):
        super(Net_relu_xavier_decayGaussian_LSD, self).__init__()
        self.neural_network = Net_relu_xavier(Layers)
        
        # zero-init the final layer of the subnetwork
        last_layer = self.neural_network.hidden_l[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

        self.gaussian_std = torch.tensor(gaussian_std, requires_grad=False)
    
    def forward(self, x):
        q = x[:, 1].unsqueeze(1)  # only apply Gaussian decay on q
        gaussian = torch.exp(-q**2 / (2 * self.gaussian_std**2))
        return self.neural_network(x) * gaussian

class Net_sigmoid_xavier_decayGaussian(nn.Module):
    def __init__(self, Layers, gaussian_std):
        super(Net_sigmoid_xavier_decayGaussian, self).__init__()
        self.neural_network = Net_sigmoid_xavier(Layers)
        self.gaussian_std = torch.tensor(gaussian_std, requires_grad=False)
    
    def forward(self, x):
        gaussian = torch.exp(-x**2/(2*self.gaussian_std**2))
        output = self.neural_network(x) * gaussian
        return output


class Net_celu_HeInit_decayGaussian(nn.Module):
    def __init__(self, Layers, gaussian_std):
        super(Net_celu_HeInit_decayGaussian, self).__init__()
        self.neural_network = Net_celu_HeInit(Layers)
        self.gaussian_std = torch.tensor(gaussian_std, requires_grad=False)
    
    def forward(self, x):
        gaussian = torch.exp(-x**2/(2*self.gaussian_std**2))
        output = self.neural_network(x) * gaussian
        return output

class Net_celu_HeInit_decayGaussian_LSD(nn.Module):
    def __init__(self, Layers, gaussian_std):
        super(Net_celu_HeInit_decayGaussian_LSD, self).__init__()
        self.neural_network = Net_celu_HeInit(Layers)

        # zero-init the final layer of the subnetwork
        last_layer = self.neural_network.hidden_l[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

        # Placeholder — will be set before first forward pass
        self.register_buffer('N_ref', None)

        # buffer (not a bare attribute) so .to(device, dtype) moves/casts it too —
        # required for GPU and float32 training
        self.register_buffer('gaussian_std', torch.tensor(gaussian_std))

        # Input standardization buffers. Descriptors (~O(0.01-2)) and q ([0,30])
        # have very different scales, which ill-conditions the first layer. These
        # are set from the training data (init_LSD_train_GPU) and applied inside
        # forward, so train-time and inference (ham) use the same transform.
        # Identity by default -> checkpoints without these buffers are unchanged.
        self.register_buffer('in_mean', torch.zeros(Layers[0]))
        self.register_buffer('in_std',  torch.ones(Layers[0]))

    def forward(self, x):
        q = x[:, -1].unsqueeze(1) # only apply Gaussian decay on q
        gaussian = torch.exp(-q**2/(2*self.gaussian_std**2))
        N_ref_expanded = self.N_ref.expand(x.shape[0], -1)
        x_ref = torch.cat([N_ref_expanded, q], dim=1)
        # standardize inputs to the sub-network (gaussian/N_ref stay in raw units)
        xn     = (x     - self.in_mean) / self.in_std
        xn_ref = (x_ref - self.in_mean) / self.in_std
        output = (self.neural_network(xn) - self.neural_network(xn_ref)) * gaussian

        return output

class OscillatingActivation(nn.Module):
    """
    Activation function: x + (1/a) * sin^2(ax)
    Derivative: 1 + sin(2ax), which oscillates around 1 — good for gradient flow.
    'a' controls the oscillation frequency.
    """
    def __init__(self, alpha=1.0):
        super(OscillatingActivation, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2


class Net_osc_HeInit(nn.Module):
    def __init__(self, Layers, alpha=5.0):
        super(Net_osc_HeInit, self).__init__()
        self.hidden_l = nn.ModuleList()
        self.alpha = alpha

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            # He init is still reasonable: activation derivative is 1 + sin(2ax),
            # which has mean ~1 near zero, similar to ReLU-like activations
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            self.hidden_l.append(linear)

    def forward(self, x):
        L = len(self.hidden_l)
        for (l, linear_transform) in zip(range(L), self.hidden_l):
            if l < L - 1:
                x = OscillatingActivation(self.alpha)(linear_transform(x))
            else:
                x = linear_transform(x)
        return x


class Net_osc_HeInit_decayGaussian_LSD(nn.Module):
    def __init__(self, Layers, gaussian_std, alpha=5.0):
        super(Net_osc_HeInit_decayGaussian_LSD, self).__init__()
        self.neural_network = Net_osc_HeInit(Layers, alpha=alpha)

        # Zero-init the final layer, same as the CELU version
        last_layer = self.neural_network.hidden_l[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

        self.gaussian_std = torch.tensor(gaussian_std, requires_grad=False)
        # Placeholder — will be set before first forward pass
        self.register_buffer('G2_ref', torch.tensor(0.0))

    def forward(self, x):
        q = x[:, 1].unsqueeze(1)  # only apply Gaussian decay on q
        gaussian = torch.exp(-q**2 / (2 * self.gaussian_std**2))
        x_ref = torch.zeros_like(x)
        x_ref[:, 0] = self.G2_ref
        x_ref[:, 1] = x[:, 1].clone()
        output = (self.neural_network(x) - self.neural_network(x_ref)) * gaussian
        return output

class Net_celu_HeInit_NOnly(nn.Module):
    """Small CELU network for N modulation functions f(N) and g(N)."""
    def __init__(self, Layers):
        super(Net_celu_HeInit_NOnly, self).__init__()
        self.hidden_l = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            self.hidden_l.append(linear)

    def forward(self, x):
        L = len(self.hidden_l)
        for (l, linear_transform) in zip(range(L), self.hidden_l):
            if l < L - 1:
                x = nn.CELU()(linear_transform(x))
            else:
                x = linear_transform(x)
        return x


class Net_osc_HeInit_FiLM_LSD(nn.Module):
    def __init__(self, q_Layers, N_Layers, gaussian_std, alpha=1.0):
        super(Net_osc_HeInit_FiLM_LSD, self).__init__()

        # q network with oscillating activation - learns base correction shape v(q)
        self.v_q = Net_osc_HeInit(q_Layers, alpha=alpha)

        # Placeholder — will be set before first forward pass
        self.register_buffer('G2_ref', torch.tensor(0.0))

        # N networks with CELU - learn scale f(N) and shift g(N)
        self.f_N = Net_celu_HeInit_NOnly(N_Layers)
        self.g_N = Net_celu_HeInit_NOnly(N_Layers)

        self.gaussian_std = torch.tensor(gaussian_std, requires_grad=False)

    def forward(self, x):
        N = x[:, 0].unsqueeze(1)
        q = x[:, 1].unsqueeze(1)

        gaussian = torch.exp(-q**2 / (2 * self.gaussian_std**2))

        # Enforce BC: Δv(N=0, q) = 0 for all q
        # by subtracting f_N and g_N evaluated at N=0
        #N_ref = torch.ones_like(N) * self.G2_ref
        f = self.f_N(N) #- self.f_N(N_ref).detach()
        g = self.g_N(N) #- self.g_N(N_ref).detach()

        output = (f * self.v_q(q) + g) * gaussian
        return output

class Net_celu_HeInit_FiLM_LSD(nn.Module):
    def __init__(self, q_Layers, N_Layers, gaussian_std):
        super(Net_celu_HeInit_FiLM_LSD, self).__init__()
        
        # q network - learns the base correction shape v(q)
        # input is 1 (just q), output is 1
        self.v_q = Net_celu_HeInit(q_Layers)
        
        # Placeholder — will be set before first forward pass
        self.register_buffer('G2_ref', torch.tensor(0.0))

        # N networks - learn scale f(N) and shift g(N)
        # input is 1 (just N), output is 1
        # both are zero at N=0 by construction (see forward)
        self.f_N = Net_celu_HeInit(N_Layers)
        self.g_N = Net_celu_HeInit(N_Layers)

        self.gaussian_std = torch.tensor(gaussian_std, requires_grad=False)
        
    def forward(self, x):
        N = x[:, 0].unsqueeze(1)
        q = x[:, 1].unsqueeze(1)
        
        gaussian = torch.exp(-q**2 / (2 * self.gaussian_std**2))
        
        # Enforce boundary condition: Δv(N=0, q) = 0 for all q
        # If f_N and g_N both output 0 at N=0, the whole output is 0
        # We enforce this by subtracting the value at N=0
        #N_ref = torch.ones_like(N) * self.G2_ref
        f = self.f_N(N) #- self.f_N(N_ref).detach()
        g = self.g_N(N) #- self.g_N(N_ref).detach()
        
        output = (f * self.v_q(q) + g) * gaussian
        return output

class Net_celu_HeInit_scale_decayGaussian(nn.Module):
    def __init__(self, Layers, gaussian_std, scale):
        super(Net_celu_HeInit_scale_decayGaussian, self).__init__()
        self.layers = Layers
        self.neural_network = Net_celu_HeInit(Layers)

        self.gaussian_std = torch.tensor(gaussian_std, requires_grad=False)
        
        # Assert scale is a tensor and has the correct shape: (Layers[-1],)
        if not torch.is_tensor(scale):
            raise TypeError("scale must be a tensor.")
        if scale.shape != (Layers[-1],):
            raise ValueError(f"scale tensor must have shape ({Layers[-1]},), but got {scale.shape}.")
        self.scale_tensor = scale
    
    def forward(self, x):
        gaussian = torch.exp(-x**2/(2*self.gaussian_std**2))
        output = self.neural_network(x) * gaussian * self.scale_tensor
        return output
    
    def change_scale(self, new_scale): 
        # Assert new_scale is a tensor and has the correct shape: (self.layers[-1],)
        if not torch.is_tensor(new_scale):
            raise TypeError("new_scale must be a tensor.")
        if new_scale.shape != (self.layers[-1],):
            raise ValueError(f"new_scale must have shape ({self.layers[-1]},), but got {new_scale.shape}.")
        
        self.scale_tensor = new_scale




class Net_celu_RandInit_decayGaussian(nn.Module):
    def __init__(self, Layers, gaussian_std):
        super(Net_celu_RandInit_decayGaussian, self).__init__()
        self.neural_network = Net_celu_RandInit(Layers)
        self.gaussian_std = torch.tensor(gaussian_std, requires_grad=False)
    
    def forward(self, x):
        gaussian = torch.exp(-x**2/(2*self.gaussian_std**2))
        output = self.neural_network(x) * gaussian
        return output


class Net_relu_HeInit_decayGaussian(nn.Module):
    def __init__(self, Layers, gaussian_std):
        super(Net_relu_HeInit_decayGaussian, self).__init__()
        self.neural_network = Net_relu_HeInit(Layers)
        self.gaussian_std = torch.tensor(gaussian_std, requires_grad=False)
    
    def forward(self, x):
        gaussian = torch.exp(-x**2/(2*self.gaussian_std**2))
        output = self.neural_network(x) * gaussian
        return output
    

class Net_relu_xavier_BN_decayGaussian(nn.Module):
    def __init__(self, Layers, gaussian_std):
        super(Net_relu_xavier_BN_decayGaussian, self).__init__()
        self.neural_network = Net_relu_xavier_BN(Layers)
        self.gaussian_std = torch.tensor(gaussian_std, requires_grad=False)
    
    def forward(self, x):
        gaussian = torch.exp(-x**2/(2*self.gaussian_std**2))
        output = self.neural_network(x) * gaussian
        return output
    

class Net_relu_xavier_BN_dropout_decayGaussian(nn.Module):
    def __init__(self, Layers, gaussian_std):
        super(Net_relu_xavier_BN_dropout_decayGaussian, self).__init__()
        self.neural_network = Net_relu_xavier_BN_dropout(Layers)
        self.gaussian_std = torch.tensor(gaussian_std, requires_grad=False)
    
    def forward(self, x):
        gaussian = torch.exp(-x**2/(2*self.gaussian_std**2))
        output = self.neural_network(x) * gaussian
        return output