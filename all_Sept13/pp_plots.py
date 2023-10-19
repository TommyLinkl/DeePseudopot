import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['mathtext.fontset'] = "stix"
rc('axes', linewidth=1.0)
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['xtick.major.size'] = 7
matplotlib.rcParams['ytick.major.size'] = 7
matplotlib.rcParams['xtick.minor.size'] = 4
matplotlib.rcParams['ytick.minor.size'] = 4
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fig, axs = plt.subplots(1,2, figsize=(10, 5))

Cd = np.loadtxt("potCd.par")
Se = np.loadtxt("potSe.par")
S = np.loadtxt("potS.par")
Zn = np.loadtxt("potZn.par")

Ga = np.loadtxt("potGa.par")
In = np.loadtxt("potIn.par")
P = np.loadtxt("potP.par")
As = np.loadtxt("potAs.par")

axs[0].plot(Cd[:,0], Cd[:,1], label="Cd")
axs[0].plot(Zn[:,0], Zn[:,1], label="Zn")
axs[0].plot(Se[:,0], Se[:,1], label="Se")
axs[0].plot(S[:,0], S[:,1], label="S")
axs[0].set(title="II-VI pseudopotentials", xlabel="r (a.u.)", ylabel="v(r) (a.u.)", xlim=(-2, 15), ylim=(-1.7, 1.5))
axs[0].legend(loc="upper right", frameon=False)
axs[0].grid(alpha=0.5)

axs[1].plot(Ga[:,0], Ga[:,1], label="Ga")
axs[1].plot(In[:,0], In[:,1], label="In")
axs[1].plot(P[:,0], P[:,1], label="P")
axs[1].plot(As[:,0], As[:,1], label="As")
axs[1].set(title="III-V pseudopotentials", xlabel="r (a.u.)", ylabel="v(r) (a.u.)", xlim=(-2, 15), ylim=(-1.7, 1.5))
axs[1].legend(loc="upper right", frameon=False)
axs[1].grid(alpha=0.5)

fig.tight_layout()
fig.savefig("pp_plots.pdf")

