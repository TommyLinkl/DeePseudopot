import numpy as np
import matplotlib.pyplot as plt
import re

def parse_bands(filename):
    """
    Parse a QE bands output file between the cues 
    "End of band structure calculation" (start) and 
    "Writing all to output data dir" (end) to extract k-point coordinates 
    and band energies.
    
    Parameters:
      filename (str): Path to the QE bands output file.
      
    Returns:
      kpoints (np.ndarray): Array of k-points with shape (n_kpoints, 3).
      bands   (np.ndarray): 2D array of eigenvalues with shape (n_kpoints, n_bands).
    """

    kpoints = []
    eigenvalues = []
    current_eigs = []
    parsing = False  # Flag to indicate we're in the band energy output section
    
    with open(filename, 'r') as f:
        for line in f:
            # Start parsing when the cue is found.
            if "End of band structure calculation" in line:
                parsing = True
                continue  # Skip the cue line itself

            # Stop parsing when the ending cue is reached.
            if "Writing all to output data dir" in line:
                if current_eigs:
                    eigenvalues.append(current_eigs)
                    current_eigs = []
                break

            # Skip lines until we are within the band structure output section.
            if not parsing:
                continue

            # When a new k-point is encountered, store the previous k-point's eigenvalues.
            if "k =" in line:
                if current_eigs:
                    eigenvalues.append(current_eigs)
                    current_eigs = []
                # Remove the "k =" part and any trailing "("; then split to get coordinates.
                k_line = line.split("k =")[1].strip()
                if "(" in k_line:
                    k_line = k_line.split("(")[0].strip()
                # Split the line into components and take the first three as the k-point coordinates.
                # k_coords = k_line.split()
                k_coords = re.findall(r'-?\d*\.\d+', k_line)

                k = [float(coord) for coord in k_coords[:3]]
                kpoints.append(k)
            else:
                # For lines between k-points (and before the end cue), assume they contain eigenenergy values.
                tokens = line.split()
                if tokens:
                    try:
                        # Append each token (or multiple tokens per line) as an eigenenergy.
                        for token in tokens:
                            current_eigs.append(float(token))
                    except ValueError:
                        # If conversion fails, skip this line.
                        pass
        
        # Append the eigenvalues for the last k-point if any.
        if current_eigs:
            eigenvalues.append(current_eigs)
    
    return np.array(kpoints), np.array(eigenvalues)

def parse_bands_pp_output(filename):
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

def compute_k_distances(kpoints):
    """
    Compute the cumulative distances along the k-point path.
    
    Parameters:
      kpoints (np.ndarray): Array of k-points, shape (n_kpoints, 3).
      
    Returns:
      distances (np.ndarray): 1D array with the cumulative k-point distances.
    """
    distances = [0.0]
    for i in range(1, len(kpoints)):
        dk = np.linalg.norm(np.array(kpoints[i]) - np.array(kpoints[i-1]))
        distances.append(distances[-1] + dk)
    return np.array(distances)

def convert_bands(k_dist, bands, writeToFile="forDeepPseudopot.dat", maxBands=32):
    n_bands = bands.shape[1]
    write_BS = np.zeros([len(k_dist), min(maxBands, n_bands)+1])
    write_BS[:,0]=k_dist
    for band in range(min(maxBands, n_bands)):
        write_BS[:, band+1] = bands[:, band]
    
    np.savetxt(writeToFile, write_BS, fmt="%.5f")

if __name__ == '__main__':
    # Name of the QE bands output file
    bands_out_file = "In75_Ga25_P_rand1.bands.dat"
    
    # Parse the bands file
    kpoints, bands = parse_bands_pp_output(bands_out_file)
    if kpoints.size == 0 or bands.size == 0:
        raise ValueError("No k-points or eigenvalues were parsed. Check file format!")
    print(bands[0])
    
    # Compute the cumulative k-point distances along the band path
    k_dist = compute_k_distances(kpoints)
    print(f"There are {len(k_dist)} kpoints. ")

    # Plot and save the band structure
    convert_bands(k_dist, bands, maxBands=90)
