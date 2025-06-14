import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def plot_matrix(data_trimmed, vmin, vmax, saveFileName, dataName="", figsize=(4,3), showColorBar=False, cmap_string="Reds", squareOffset=0.5, boxSize=35): 
    if np.abs(data_trimmed).max() > vmax or np.abs(data_trimmed).min() < vmin:
        print(f"Warning: {dataName} has values outside [{vmin}, {vmax}]")
        print(np.abs(data_trimmed).max(), np.abs(data_trimmed).min())
    
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap(cmap_string)
    newcolors = cmap(np.linspace(0, 1, 256))
    # Set the first color (corresponding to the minimum value, typically 0) to white
    newcolors[0, :] = np.array([1, 1, 1, 1])
    # Create a new colormap with these colors
    newcmap = mcolors.ListedColormap(newcolors)
    # Use the new colormap in your imshow call
    im = ax.imshow(np.abs(data_trimmed), cmap=newcmap, aspect="equal", vmin=vmin, vmax=vmax)
    # im = ax.imshow(np.abs(data_trimmed), cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    if showColorBar: 
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        # cbar.set_label("Magnitude")
    # ax.set_title(f"Magnitude of {name} (kidx={kidx})")
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw a larger bounding square around the small matrix
    ax.plot([-squareOffset, boxSize-squareOffset, boxSize-squareOffset, -squareOffset, -squareOffset], 
            [-squareOffset, -squareOffset, boxSize-squareOffset, boxSize-squareOffset, -squareOffset], 
            color="black", linestyle="-", linewidth=1)
    ax.scatter([31, 32, 33], [31, 32, 33], color="black", marker=".", s=50)
    ax.scatter([-1], [-1], color="white", marker=".", s=1)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(saveFileName)
    plt.close(fig)
    return


if __name__=="__main__": 
    vmin_T, vmax_T = 0, 0.95
    vmin, vmax = 0, 0.29
    data_dir = "/global/homes/t/tommylin/DeePseudopot/CALCS_CsPbI3_dispersion_2/hamiltonian_results/"
    plot_dir = "/global/homes/t/tommylin/DeePseudopot/CALCS_CsPbI3_dispersion_2/hamiltonian_plots/"
    os.makedirs(plot_dir, exist_ok=True)

    for kidx in range(3): 
        data = np.load(f"{data_dir}T_kidx_{kidx}.npy")
        data = data[:30, :30]
        plot_matrix(data, vmin_T, vmax_T, f"{plot_dir}T_kidx_{kidx}.pdf", dataName=f"T_kidx_{kidx}", showColorBar=True)

        data = np.load(f"{data_dir}Vloc_kidx_{kidx}.npy")
        data = data[:30, :30]
        plot_matrix(data, vmin, vmax, f"{plot_dir}Vloc_kidx_{kidx}.pdf", dataName=f"Vloc_kidx_{kidx}", showColorBar=True)

        data = np.load(f"{data_dir}Vnl_kidx_{kidx}.npy")
        data = data[:30, :30]
        plot_matrix(data, vmin, vmax, f"{plot_dir}Vnl_kidx_{kidx}.pdf", dataName=f"Vnl_kidx_{kidx}", showColorBar=True)

        data = np.load(f"{data_dir}SO_kidx_{kidx}.npy")
        data = data[:30, :30]
        plot_matrix(data, vmin, vmax, f"{plot_dir}SO_kidx_{kidx}.pdf", dataName=f"SO_kidx_{kidx}", showColorBar=True)

        # SOC + NL
        soc = np.load(f"{data_dir}SO_kidx_{kidx}.npy")
        soc = soc[:30, :30]
        nl = np.load(f"{data_dir}Vnl_kidx_{kidx}.npy")
        nl = nl[:30, :30]
        data = soc + nl
        plot_matrix(data, vmin, vmax, f"{plot_dir}NLSOC_kidx_{kidx}.pdf", dataName=f"NLSOC_kidx_{kidx}", showColorBar=True)

        # Total Hamiltonian
        soc = np.load(f"{data_dir}SO_kidx_{kidx}.npy")
        soc = soc[:30, :30]
        nl = np.load(f"{data_dir}Vnl_kidx_{kidx}.npy")
        nl = nl[:30, :30]
        t = np.load(f"{data_dir}T_kidx_{kidx}.npy")
        t = t[:30, :30]
        loc = np.load(f"{data_dir}Vloc_kidx_{kidx}.npy")
        loc = loc[:30, :30]
        data = soc + nl + t + loc
        plot_matrix(data, vmin_T, vmax_T, f"{plot_dir}Htot_kidx_{kidx}.pdf", dataName=f"Htot_kidx_{kidx}", showColorBar=True)

        