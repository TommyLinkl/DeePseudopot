import numpy as np
import matplotlib.pyplot as plt
import sys

# Functions to compute SO and NL potential
def calc_NL_pot(x, nl1, nl2, shift=1.5):
  return nl1 * np.exp(-(x**2)) + nl2 * np.exp(-(x - shift)**2)

def calc_SO_pot(x, so, width=0.7):
  return so * np.exp(-(x/width)**2)

# Read in necessary data to plot potentials
filename = sys.argv[1]
SO_par = float(sys.argv[2])
NL1_par = float(sys.argv[3])
NL2_par = float(sys.argv[4])

pot_j_0_dat = np.loadtxt(filename)
r_grid = pot_j_0_dat[:, 0]

pot_j_0 = pot_j_0_dat[:, 1]
pot_j_1_2 = pot_j_0 + calc_NL_pot(r_grid, NL1_par, NL2_par) - calc_SO_pot(r_grid, SO_par)
pot_j_3_2 = pot_j_0 + calc_NL_pot(r_grid, NL1_par, NL2_par) + 0.5 * calc_SO_pot(r_grid, SO_par)

fig, ax = plt.subplots(figsize=(4,4))

ax.plot(r_grid, pot_j_0, linewidth=1, label = r"$v_{0}$")
if abs(SO_par) > 1e-8:
  ax.plot(r_grid, pot_j_1_2, linewidth=1, label = r"$v_{1/2}$")
  ax.plot(r_grid, pot_j_3_2, linewidth=1, label = r"$v_{3/2}$")

ax.set_xlim(0, 8.0)
#ax.set_ylim(-3.0,11.0)
ax.set_xlabel("r (a.u.)")
ax.set_ylabel("v(r) (a.u.)")

ax.legend(frameon=False)

plt.savefig(filename + '.pdf', format='pdf', dpi=200)