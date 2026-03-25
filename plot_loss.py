import numpy as np
import matplotlib.pyplot as plt

iterations = np.loadtxt("iterations.dat", skiprows=2)


n_step = iterations.shape[0]

steps_arr = iterations[:, 0]
newMSE = iterations[:, 1]
currMSE = iterations[:, 2]
bestMSE = iterations[:, 3]

fig, ax = plt.subplots()
ax1 = ax.twinx()

ax.plot(steps_arr, newMSE, linewidth=0.5, label="newMSE")
ax.plot(steps_arr, currMSE, linewidth=0.5, label="currMSE")
ax1.plot(steps_arr[20:], bestMSE[20:], linewidth=1.0, color='g', label="bestMSE")

ax.set_xlabel("Monte Carlo Iteration")
ax.set_ylabel("MSE (eV)")

ax.legend(frameon=False)
ax1.legend(frameon=False, loc='upper left')

plt.savefig("iterations.pdf", format='pdf')

