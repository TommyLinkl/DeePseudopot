import numpy as np
import matplotlib.pyplot as plt
import re

def parse_bands(filename):
    """
    Parses the QE bands file.
    
    Expected format:
      - A header line like "&plot nbnd= 144, nks=     6 /"
      - For each k-point:
          * A line with the 3 k-point coordinates.
          * Several subsequent lines with band energies (total of nbnd energies).
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Extract nbnd and nks from header line (first line)
    header = lines[0]
    nbnd_match = re.search(r"nbnd=\s*(\d+)", header)
    nks_match  = re.search(r"nks=\s*(\d+)", header)
    if not (nbnd_match and nks_match):
        raise ValueError("Could not parse nbnd or nks from header.")
    nbnd = int(nbnd_match.group(1))
    nks  = int(nks_match.group(1))
    
    k_points = []
    bands = []  # Will store one list (of energies) per k-point
    i = 1  # start after header
    for kp in range(nks):
        # Skip blank lines if any
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        # Read the k-point coordinate line
        k_line = lines[i].strip()
        i += 1
        # Parse three floats for the k-point coordinate
        k_coord = [float(x) for x in k_line.split()]
        k_points.append(k_coord)
        
        # Now read band energies until we have nbnd numbers
        energies = []
        while len(energies) < nbnd and i < len(lines):
            line = lines[i].strip()
            if line == "":
                i += 1
                continue
            parts = line.split()
            energies.extend([float(x) for x in parts])
            i += 1
        if len(energies) != nbnd:
            print(f"Warning: Expected {nbnd} energies but got {len(energies)} for k-point {kp}.")
        bands.append(energies[:nbnd])
    
    return np.array(k_points), np.array(bands)

def compute_k_distances(k_points):
    """
    Computes a cumulative distance along the k-path based on the Euclidean distances
    between successive k-point coordinates.
    """
    distances = [0.0]
    for i in range(1, len(k_points)):
        dk = np.linalg.norm(np.array(k_points[i]) - np.array(k_points[i-1]))
        distances.append(distances[-1] + dk)
    return np.array(distances)

def parse_dos(filename):
    """
    Parses the DOS file.
    
    Expected format:
      - A header line beginning with '#' that includes the Fermi energy, e.g.
        "#  E (eV)   dos(E)     Int dos(E) EFermi =    7.279 eV"
      - Followed by lines with: energy  dos  integrated_dos.
    Returns the energy array, dos array, and the Fermi energy (or None if not found).
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    EFermi = None
    # Look for EFermi in header lines
    for line in lines:
        if line.startswith("#") and "EFermi" in line:
            match = re.search(r"EFermi\s*=\s*([\d\.]+)", line)
            if match:
                EFermi = float(match.group(1))
            break
    
    # Load data; np.loadtxt will skip lines starting with '#' by default.
    dos_data = np.loadtxt(filename, comments="#")
    energy = dos_data[:, 0]
    dos    = dos_data[:, 1]
    return energy, dos, EFermi

if __name__ == "__main__":
    # Parse bands and compute k-path distances
    k_points, bands = parse_bands("GaP.bands.dat")
    k_dist = compute_k_distances(k_points)
    
    # Parse DOS and extract EFermi if available
    dos_energy, dos_vals, EFermi = parse_dos("GaP.dos.dat")
    
    # Create figure with two subplots:
    # Left subplot: band structure; Right subplot: DOS.
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 6),
                                   gridspec_kw={'width_ratios': [3, 1]})
    
    # Plot the band structure.
    # Each band is plotted as a function of the cumulative k distance.
    for band in bands.T:  # bands shape: (nks, nbnd) so we transpose to iterate over bands
        ax1.plot(k_dist, band, color='black', lw=1)
    ax1.set(xlabel="k-path distance", ylabel="Energy (eV)", title="Band Structure")
    ax1.set(ylim=(6, 11))
    ax1.grid(True)
    
    # Highlight the Fermi level if available.
    if EFermi is not None:
        ax1.axhline(EFermi, color='red', ls='--', lw=1, label="EFermi")
        ax1.legend()
    
    # Plot the DOS: energy on the y-axis, DOS on the x-axis.
    ax2.plot(dos_vals, dos_energy, color='blue', lw=1)
    ax2.set(xlabel="DOS", title="Density of States")
    ax2.set(ylim=(6, 11))
    ax2.invert_xaxis()  # so that the DOS plot faces the band structure plot
    ax2.grid(True)
    
    # Draw a horizontal line for EFermi in the DOS plot.
    if EFermi is not None:
        ax2.axhline(EFermi, color='red', ls='--', lw=1)
    
    fig.tight_layout()
    fig.savefig("plot_qe_bandstructure.pdf")

