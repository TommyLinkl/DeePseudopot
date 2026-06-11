import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse the plotting parameters
def buildParser():

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--min', type=float, default='-4.0',
            help='lowest energy to plot')
    parser.add_argument('--max', type=float, default='4.0',
            help='max energy to plot')
    parser.add_argument('--fermiEnergy', type=float, default='0.0',
            help='energy of the valence band maximum (just for shifting plots)')
    parser.add_argument('--file', type=str, default='expBandStruct_0.par',
            help='file from which the bandstructure will be read')
    parser.add_argument('--kpoints', type=str, default='kpoints_0.par',
            help='file from which the kpoints will be read')
    parser.add_argument('--ref', type=str, default='', help='reference band structure file')
    return parser

def get_band_window(bands, band_ene_min, band_ene_max):
    """
    bands: ndarray of shape (nkpt, nbands)
    band_ene_min: float
    band_ene_max: float
    
    Returns (n_band_start, n_band_end) inclusive indices
    """
    nkpt, nbands = bands.shape
    # Mask of bands within window at any k
    mask_min = (bands > band_ene_min).any(axis=0)  # True if band ever exceeds min
    mask_max = (bands < band_ene_max).any(axis=0)  # True if band ever below max
    
    # Bands satisfying both
    mask = mask_min & mask_max
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) == 0:
        raise ValueError("No bands found in the specified energy window.")
    
    n_band_start = valid_indices.min()
    n_band_end = valid_indices.max()
    
    return n_band_start, n_band_end

parser = buildParser()
params = parser.parse_args()


# Read in the bandstructure file
bandstruct_filename = params.file

bands = np.loadtxt(bandstruct_filename)
bands[:,1:] = bands[:,1:] - params.fermiEnergy

# If ref band structure exists, read it in
ref_bandstruct_filename = params.ref

try:
  ref_bands = np.loadtxt(ref_bandstruct_filename, ndmin=2)
  ref_bands[:,1:] = ref_bands[:,1:] - params.fermiEnergy
except FileNotFoundError:
  print(f"Reference bandstructure not supplied or not found.")
  ref_bands = None

# Read in the kpoints file
kpoints_filename = params.kpoints
try:
  kpoints_dat = np.loadtxt(kpoints_filename)[:,:-1]
  diffs = np.diff(kpoints_dat, axis=0)        # shape (N-1, 3)
  step_lengths = np.linalg.norm(diffs, axis=1)  # shape (N-1,)
  # prepend 0 so it's the same length N
  kpoints = np.concatenate(([0.0], np.cumsum(step_lengths)))
  # reshape to (N,1)
  #kpoints = kpoints.reshape(-1, 1)
except FileNotFoundError:
  print(f"File not found {kpoints_filename}")
  kpoints = bands[:, 0]

# Plot states within energy window
n_band_start, n_band_end = get_band_window(bands[:, 1:], params.min, params.max)

# Plot bands
fig, ax = plt.subplots()

for ib in range(n_band_start, n_band_end):
  # Index ib + 1 because the first column just holds the x axis grid
  ax.plot(kpoints, bands[:, ib + 1], 'o-', color='b', linewidth='0.8', markersize=1.5)
  if ref_bands is not None:
    ax.scatter(kpoints, ref_bands[:, ib + 1], color='r')

ax.set_xlabel("k")
ax.set_ylabel("E(k)")

plt.savefig(f"{params.file}.pdf", format='pdf', dpi=120)
