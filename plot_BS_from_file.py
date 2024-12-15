import numpy as np
from utils.pp_func import plotBandStructFromFile

shift = 6.6
BS = np.loadtxt("CALCS/CsPbBr3_32kpts/results_celu_DWInit_4/oldFunc_BS_sys0.dat")
BS[:, 1:] += shift
np.savetxt("CALCS/CsPbBr3_32kpts/results_celu_DWInit_4/oldFunc_shift_BS_sys0.dat", BS)

fig, axs = plotBandStructFromFile("CALCS/CsPbBr3_32kpts/inputs_celu_DWInit_4/expBandStruct_0.par", "CALCS/CsPbBr3_32kpts/results_celu_DWInit_4/oldFunc_shift_BS_sys0.dat")

axs[0].set(ylim=(-12, 15))
axs[1].set(ylim=(-5, 6))
fig.suptitle(f"Shifting the band energies up by {shift} eV")

fig.savefig("CALCS/CsPbBr3_32kpts/results_celu_DWInit_4/oldFunc_shift_BS_sys0_shift7.pdf")