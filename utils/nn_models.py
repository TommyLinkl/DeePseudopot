import torch
import torch.nn as nn

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
    
class Net_relu_xavier_decayGaussian(nn.Module):
    def __init__(self, Layers, gaussian_std):
        super(Net_relu_xavier_decayGaussian, self).__init__()
        self.neural_network = Net_relu_xavier(Layers)
        self.gaussian_std = torch.tensor(gaussian_std, requires_grad=False)
    
    def forward(self, x):
        gaussian = torch.exp(-x**2/(2*self.gaussian_std**2))
        output = self.neural_network(x) * gaussian
        return output


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