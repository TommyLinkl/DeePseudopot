import numpy as np
import matplotlib
import matplotlib.pyplot as plt
AUTOEV = 27.2114

def load_eigVecs(filename): 
    data = np.load(filename)
    array_list = [data[f"arr_{i}"] for i in range(len(data.files))]
    data.close()
    return array_list

def load_H(filename): 
    array = np.load(filename)
    return array

def expectation_value(psi, H, tol=1e-10):
    """
    Calculate the expectation value <psi | H | psi>, with a check for Hermitian Hamiltonian.
    
    Parameters:
    psi (np.ndarray): Wavefunction vector, shape (N,)
    H (np.ndarray): Hamiltonian matrix, shape (N, N)
    tol (float): Numerical tolerance for checking if the expectation value is real.
    
    Returns:
    float: The real expectation value <psi | H | psi>.
    
    Raises:
    ValueError: If the expectation value is not real within the given tolerance, 
                indicating that H might not be Hermitian.
    """
    # Ensure psi is a column vector
    psi_conj = np.conjugate(psi)  # Take the conjugate of the wavefunction
    expectation = np.dot(psi_conj, np.dot(H, psi))  # <psi|H|psi>
    
    if np.abs(expectation.imag) > tol:
        raise ValueError(
            f"The expectation value has a significant imaginary part: {expectation.imag}. "
            "The Hamiltonian might not be Hermitian."
        )
    
    return expectation.real

def calc_T_Vloc_Vnl_SO_psi(dirName, nkpts=150): 
    f = open(f"{dirName}psi_T_psi.dat", "w")
    f.write("# kIdx     <psi|T|psi> for each band (eV)\n")
    g = open(f"{dirName}psi_Vloc_psi.dat", "w")
    g.write("# kIdx     <psi|Vloc|psi> for each band (eV)\n")
    h = open(f"{dirName}psi_Vnl_psi.dat", "w")
    h.write("# kIdx     <psi|Vnl|psi> for each band (eV)\n")
    m = open(f"{dirName}psi_SO_psi.dat", "w")
    m.write("# kIdx     <psi|SO|psi> for each band (eV)\n")

    for kIdx in range(nkpts): 
        print(kIdx)
        eigVecs = load_eigVecs(f"{dirName}hamiltonian_results/eigVec_k{kIdx}.npz")
        f.write(f"{kIdx}")
        g.write(f"{kIdx}")
        h.write(f"{kIdx}")
        m.write(f"{kIdx}")
        T_kidx = load_H(f"{dirName}hamiltonian_results/T_kidx_{kIdx}.npy")
        Vloc_kidx = load_H(f"{dirName}hamiltonian_results/Vloc_kidx_{kIdx}.npy")
        Vnl_kidx = load_H(f"{dirName}hamiltonian_results/Vnl_kidx_{kIdx}.npy")
        SO_kidx = load_H(f"{dirName}hamiltonian_results/SO_kidx_{kIdx}.npy")

        for bandIdx in range(len(eigVecs)):
            exp_T_kidx = expectation_value(eigVecs[bandIdx], T_kidx)
            f.write(f"  {exp_T_kidx*AUTOEV:.5f}")

            exp_Vloc_kidx = expectation_value(eigVecs[bandIdx], Vloc_kidx)
            g.write(f"  {exp_Vloc_kidx*AUTOEV:.5f}")

            exp_Vnl_kidx = expectation_value(eigVecs[bandIdx], Vnl_kidx)
            h.write(f"  {exp_Vnl_kidx*AUTOEV:.5f}")
            
            exp_SO_kidx = expectation_value(eigVecs[bandIdx], SO_kidx)
            m.write(f"  {exp_SO_kidx*AUTOEV:.5f}")
        f.write("\n")
        g.write("\n")
        h.write("\n")
        m.write("\n")
    f.close()
    g.close()
    h.close()
    m.close()
    return

def calc_T_Vloc_Vnl_SO_psi0(dirName, nkpts=50): 
    f = open(f"{dirName}psi0_T_psi0.dat", "w")
    f.write("# kIdx     <psi0|T|psi0> for each band (eV)\n")
    g = open(f"{dirName}psi0_Vloc_psi0.dat", "w")
    g.write("# kIdx     <psi0|Vloc|psi0> for each band (eV)\n")
    h = open(f"{dirName}psi0_Vnl_psi0.dat", "w")
    h.write("# kIdx     <psi0|Vnl|psi0> for each band (eV)\n")
    m = open(f"{dirName}psi0_SO_psi0.dat", "w")
    m.write("# kIdx     <psi0|SO|psi0> for each band (eV)\n")

    eigVecs = load_eigVecs(f"{dirName}hamiltonian_results/eigVec_k0.npz")
    for kIdx in range(nkpts):
        print(kIdx)
        f.write(f"{kIdx}")
        g.write(f"{kIdx}")
        h.write(f"{kIdx}")
        m.write(f"{kIdx}")
        T_kidx = load_H(f"{dirName}hamiltonian_results/T_kidx_{kIdx}.npy")
        Vloc_kidx = load_H(f"{dirName}hamiltonian_results/Vloc_kidx_{kIdx}.npy")
        Vnl_kidx = load_H(f"{dirName}hamiltonian_results/Vnl_kidx_{kIdx}.npy")
        SO_kidx = load_H(f"{dirName}hamiltonian_results/SO_kidx_{kIdx}.npy")

        for bandIdx in range(len(eigVecs)):
            exp_T_kidx = expectation_value(eigVecs[bandIdx], T_kidx)
            f.write(f"  {exp_T_kidx*AUTOEV:.5f}")

            exp_Vloc_kidx = expectation_value(eigVecs[bandIdx], Vloc_kidx)
            g.write(f"  {exp_Vloc_kidx*AUTOEV:.5f}")

            exp_Vnl_kidx = expectation_value(eigVecs[bandIdx], Vnl_kidx)
            h.write(f"  {exp_Vnl_kidx*AUTOEV:.5f}")
            
            exp_SO_kidx = expectation_value(eigVecs[bandIdx], SO_kidx)
            m.write(f"  {exp_SO_kidx*AUTOEV:.5f}")

            # print((exp_T_kidx+exp_Vloc_kidx+exp_Vnl_kidx+exp_SO_kidx)*AUTOEV)
        f.write("\n")
        g.write("\n")
        h.write("\n")
        m.write("\n")

    f.close()
    g.close()
    h.close()
    m.close()
    return

def plot_perturb_total(dirName, ax, ymin=None, ymax=None):
    # read in ref BS, fit BS
    refBS = np.loadtxt(f"{dirName}hamiltonian_inputs/expBandStruct_0.par")[:, 1:]
    fitBS = np.loadtxt(f"{dirName}hamiltonian_results/eval_BS_sys0.dat")[:, 1:]

    # read in 4 files. Calculate the sum
    T_psi0 = np.loadtxt(f"{dirName}psi0_T_psi0.dat")[:, 1:]
    Vloc_psi0 = np.loadtxt(f"{dirName}psi0_Vloc_psi0.dat")[:, 1:]
    Vnl_psi0 = np.loadtxt(f"{dirName}psi0_Vnl_psi0.dat")[:, 1:]
    SO_psi0 = np.loadtxt(f"{dirName}psi0_SO_psi0.dat")[:, 1:]
    sum_psi0 = T_psi0 + Vloc_psi0 + Vnl_psi0 + SO_psi0

    # plot
    for i in range(len(refBS[0])): 
        if i==0: 
            ax.plot(np.arange(len(refBS)), refBS[:, i], "b-", alpha=0.5, label="Ref GW")
        else: 
            ax.plot(np.arange(len(refBS)), refBS[:, i], "b-", alpha=0.5)
    for i in range(len(fitBS[0])): 
        if i==0: 
            ax.plot(np.arange(len(fitBS)), fitBS[:, i], "r-", alpha=0.5, label="Fit")
        else: 
            ax.plot(np.arange(len(fitBS)), fitBS[:, i], "r-", alpha=0.5)
    for i in range(len(sum_psi0[0])): 
        if i==0: 
            ax.plot(np.arange(len(sum_psi0)), sum_psi0[:, i], "-", color="green", alpha=0.5, label="Perturb total")
        else: 
            ax.plot(np.arange(len(sum_psi0)), sum_psi0[:, i], "-", color="green", alpha=0.5, label=None)
    ax.legend()
    ax.grid(alpha=0.5)

    if (ymin is not None) or (ymax is not None):
        current_ylim = ax.get_ylim()
        new_ylim = (ymin if ymin is not None else current_ylim[0], ymax if ymax is not None else current_ylim[1])
        ax.set_ylim(new_ylim)

    return

def plot_components(dirName, ax, ymin=None, ymax=None):
    # read in ref BS, fit BS
    refBS = np.loadtxt(f"{dirName}hamiltonian_inputs/expBandStruct_0.par")[:, 1:]
    fitBS = np.loadtxt(f"{dirName}hamiltonian_results/eval_BS_sys0.dat")[:, 1:]

    # read in 4 files. Calculate the sum
    T_psi0 = np.loadtxt(f"{dirName}psi0_T_psi0.dat")[:, 1:]
    Vloc_psi0 = np.loadtxt(f"{dirName}psi0_Vloc_psi0.dat")[:, 1:]
    Vnl_psi0 = np.loadtxt(f"{dirName}psi0_Vnl_psi0.dat")[:, 1:]
    SO_psi0 = np.loadtxt(f"{dirName}psi0_SO_psi0.dat")[:, 1:]

    # plot
    for i in range(len(refBS[0])): 
        if i==0: 
            ax.plot(np.arange(len(refBS)), refBS[:, i], "b-", alpha=0.5, label="Ref GW")
        else: 
            ax.plot(np.arange(len(refBS)), refBS[:, i], "b-", alpha=0.5)
    for i in range(len(fitBS[0])): 
        if i==0: 
            ax.plot(np.arange(len(fitBS)), fitBS[:, i], "r-", alpha=0.5, label="Fit")
        else: 
            ax.plot(np.arange(len(fitBS)), fitBS[:, i], "r-", alpha=0.5)
    for i in range(len(T_psi0[0])): 
        shift = fitBS[0, i] - T_psi0[0, i]
        if i==0: 
            ax.plot(np.arange(len(T_psi0)), T_psi0[:, i]+shift, "--", color="orange", alpha=0.5, label="<psi0 | T | psi0>, shifted")
        else: 
            ax.plot(np.arange(len(T_psi0)), T_psi0[:, i]+shift, "--", color="orange", alpha=0.5, label=None)
    for i in range(len(Vloc_psi0[0])): 
        shift = fitBS[0, i] - Vloc_psi0[0, i]
        if i==0: 
            ax.plot(np.arange(len(Vloc_psi0)), Vloc_psi0[:, i]+shift, "--", color="cyan", alpha=0.5, label="<psi0 | Vloc | psi0>, shifted")
        else: 
            ax.plot(np.arange(len(Vloc_psi0)), Vloc_psi0[:, i]+shift, "--", color="cyan", alpha=0.5, label=None)
    for i in range(len(Vnl_psi0[0])): 
        shift = fitBS[0, i] - Vnl_psi0[0, i]
        if i==0: 
            ax.plot(np.arange(len(Vnl_psi0)), Vnl_psi0[:, i]+shift, "--", color="darkviolet", alpha=0.5, label="<psi0 | Vnl | psi0>, shifted")
        else: 
            ax.plot(np.arange(len(Vnl_psi0)), Vnl_psi0[:, i]+shift, "--", color="darkviolet", alpha=0.5, label=None)
    for i in range(len(SO_psi0[0])): 
        shift = fitBS[0, i] - SO_psi0[0, i]
        if i==0: 
            ax.plot(np.arange(len(SO_psi0)), SO_psi0[:, i]+shift, "--", color="g", alpha=0.5, label="<psi0 | SO | psi0>, shifted")
        else: 
            ax.plot(np.arange(len(SO_psi0)), SO_psi0[:, i]+shift, "--", color="g", alpha=0.5, label=None)
    
    ax.legend()
    ax.grid(alpha=0.5)

    if (ymin is not None) or (ymax is not None):
        current_ylim = ax.get_ylim()
        new_ylim = (ymin if ymin is not None else current_ylim[0], ymax if ymax is not None else current_ylim[1])
        ax.set_ylim(new_ylim)

    return

if __name__=="__main__": 
    dirName = "CALCS_CsPbI3_dispersion_2/"
    
    calc_T_Vloc_Vnl_SO_psi(dirName)
    calc_T_Vloc_Vnl_SO_psi0(dirName)

    fig, ax = plt.subplots(1,1, figsize=(8,8))
    plot_perturb_total(dirName, ax, ymin=-13, ymax=2)
    ax.set(xlim=(0, 100))
    fig.tight_layout()
    fig.savefig(f"{dirName}plot_perturb_total.pdf")

    fig, ax = plt.subplots(1,1, figsize=(5,8))
    plot_components(dirName, ax, ymin=-9, ymax=2)
    ax.set(xlim=(0, 50))
    fig.tight_layout()
    fig.savefig(f"{dirName}plot_components.pdf")