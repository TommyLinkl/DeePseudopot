import numpy as np
import torch

MASS = 1.0
HBAR = 1.0
AUTOEV = 27.2114
AUTONM = 0.05291772108
NQGRID = 2048

# Current Cd, Se, S pp's in Zunger's form
CdParams = torch.tensor([-31.4518, 1.3890, -0.0502, 1.6603, 0.0586])
SeParams = torch.tensor([8.4921, 4.3513, 1.3600, 0.3227, 0.1746])
SParams = torch.tensor([7.6697, 4.5192, 1.3456, 0.3035, 0.2087])
PP_order = np.array(["Cd", "Se", "S"])
totalParams = torch.cat((CdParams.unsqueeze(0), SeParams.unsqueeze(0), SParams.unsqueeze(0)), dim=0)