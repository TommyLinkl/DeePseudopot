import sys
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.read import setNN_LSD, read_NNConfigFile
#from utils.pp_func import realSpacePot

from compute_G2_nanocrystal import read_conf_par, compute_G2

def plot_pot(pot, atm_label):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(pot[:, 0], pot[:, 1], linewidth=2, label=f"{atm_label}")
    ax.set_xlim(0.0, 10.0)
    ax.set_xlabel("r [Bohr]")
    ax.set_ylabel(r"$\Delta$v^{loc}")
    ax.legend(frameon=False)
    plt.savefig(f"LSD/pot_LSD_{atm_label}.pdf", format="pdf")
    plt.close()

def realSpacePot(vq, qSpacePot, nRGrid, rmax=25):
    dq = vq[1] - vq[0]
    vr = torch.linspace(0, rmax, nRGrid, device=vq.device)    # (nRGrid,)

    vq_ = vq.flatten()            # guarantee (nQGrid,)
    qp_ = qSpacePot.flatten()     # guarantee (nQGrid,)

    # (nRGrid-1, 1) * (1, nQGrid) -> (nRGrid-1, nQGrid)
    sin_term  = torch.sin(vr[1:, None] * vq_[None, :])
    prefactor = 4 * np.pi * dq / (8 * np.pi**3 * vr[1:])     # (nRGrid-1,)
    bulk      = prefactor * (sin_term * (vq_ * qp_)[None, :]).sum(dim=1)

    r0 = (4 * np.pi * dq / (8 * np.pi**3)) * (vq_**2 * qp_).sum()

    rSpacePot = torch.cat([r0.unsqueeze(0), bulk])

    return (vr.view(-1, 1), rSpacePot.view(-1, 1))

def main():
    if len(sys.argv) < 3:
        print("Usage: python gen_lsd_pots.py <conf.par> <material>")
        print("  material: CsPbI3 | CsPbBr3 | CsPbCl3")
        sys.exit(1)

    os.makedirs(f"LSD/", exist_ok=True)

    filepath = sys.argv[1]
    material = sys.argv[2]
    symbols, positions = read_conf_par(filepath)

    atomPPOrder = list(set(symbols))
    NNConfig = read_NNConfigFile('NN_config.par')

    print(f"Read {len(symbols)} atoms from '{filepath}'  (material: {material})")

    G2 = compute_G2(symbols, positions, material)
    print("\nG2 descriptor (one value per atom):")
    for i, (sym, val) in enumerate(zip(symbols, G2)):
        print(f"  Atom {i:4d}  {sym:3s}  G2 = {val: .6g}")

    lsd_layers = [2] + NNConfig['LSD_hiddenLayers'] + [1]
    LSDmodels = {}
    
    for atom in atomPPOrder:
        LSDmodels[atom] = setNN_LSD(NNConfig, layers=lsd_layers)
        #print(f"\nLSDmodel[{atom}] = {LSDmodels[atom]}")
    
    for atomType in set(symbols):
        print(f"Reading in saved LSDmodel for {atomType}")
        LSDmodels[atomType].load_state_dict(torch.load(f"{atomType}_LSDmodel.pth"))
        
    qmax = 30.0
    rmax = 120.0
    nQGrid = 4096
    nRGrid = 4096
    qGrid = torch.linspace(0.0, qmax, nQGrid).view(-1, 1)
    #test_G2s = np.linspace(0.0, -6.0, 5)
    # test_G2s = [-0.36, -0.25, -0.18, -0.08, -0.002, -0.001, -0.0001, 0.0]
    #print(f"test {test_G2s}")
    print(f"\nComputing LSD correction for...")
    #fig, ax = plt.subplots(figsize=(5,5))
    for alpha, symb in enumerate(symbols):
    #for alpha in range(len(test_G2s)):
        #symb = 'Cs'
        print(f"\t{symb}{alpha}")
        # Get the G2 descriptor for this atom and format the NN input
        atom_descr = G2[alpha]
        #atom_descr = test_G2s[alpha]
        
        N_alpha = torch.full_like(qGrid, atom_descr)
        x_input = torch.cat([N_alpha, qGrid], dim=1)

        # Run forward pass of NN to obtain LSD correction in q-space
        qSpacePot = LSDmodels[symb](x_input)

        # Fourier transform into real space to obtain the deltaV correction function for nanocrystals
        (vr, rSpacePot) = realSpacePot(qGrid.view(-1), qSpacePot, nRGrid, rmax)

        # Print out and plot the potentials
        pot = torch.cat((vr, rSpacePot), dim=1).detach().numpy()
        potq = torch.cat((qGrid, qSpacePot), dim=1).detach().numpy()
        np.savetxt(f"LSD/pot_LSD_{symb}{alpha}.dat", pot, delimiter='  ', fmt='%e')
        np.savetxt(f"LSD/pot_q_LSD_{symb}{alpha}.dat", potq, delimiter='  ', fmt='%e')

        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(pot[:, 0], pot[:, 1], linewidth=2, label=f"{symb}{alpha}")
        #ax.plot(pot[:, 0], pot[:, 1], linewidth=2, label=f"N={atom_descr:.2g}")
        ax.set_xlim(0.0, 10.0)
        ax.set_xlabel("r [Bohr]")
        ax.set_ylabel(r"$\Delta v^{loc}$ [a.u.]")
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f"LSD/pot_LSD_{symb}{alpha}.pdf", format="pdf")
        plt.close()

        fig1, ax1 = plt.subplots(figsize=(5,5))
        ax1.plot(potq[:, 0], potq[:, 1], linewidth=2, label=f"{symb}{alpha}")
        ax1.set_xlim(0.0, 10.0)
        ax1.set_xlabel("q [Bohr]")
        ax1.set_ylabel(r"$\Delta v^{loc}$")
        ax1.legend(frameon=False)
        fig1.tight_layout()
        plt.savefig(f"LSD/pot_q_LSD_{symb}{alpha}.pdf", format="pdf")
        plt.close()
    # plt.savefig(f"LSD/pot_LSD_{symb}_range.pdf", format="pdf")
    return


if __name__ == "__main__":
    main()
