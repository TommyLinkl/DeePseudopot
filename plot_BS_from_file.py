import numpy as np
from utils.pp_func import plotBandStructFromFile

shift = 0.0   # 6.6
BS = np.loadtxt("CALCS_III_V_SOC/alloy_In50Ga50P_rand1_pre_calc_2.3_testSOC_results/initZunger_BS_sys0.dat")
BS[:, 1:] += shift
# np.savetxt("CALCS/CsPbBr3_32kpts/results_celu_DWInit_4/oldFunc_shift_BS_sys0.dat", BS)

fig, axs = plotBandStructFromFile("CALCS_III_V_SOC/alloy_In50Ga50P_rand1_pre_calc_2.3_testSOC_results/initZunger_BS_sys0.dat", "CALCS_III_V_SOC/alloy_In50Ga50P_rand1_pre_calc_2.3_testSOC_results/initZunger_BS_sys0.dat")

axs[0].set(ylim=(5, 15))
axs[1].set(ylim=(5, 15))
fig.suptitle(f"Shifting the band energies up by {shift} eV")

fig.savefig("CALCS_III_V_SOC/alloy_In50Ga50P_rand1_pre_calc_2.3_testSOC_results/initZunger_plotBS_2.pdf")