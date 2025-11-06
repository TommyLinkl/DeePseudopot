# System and Data Files

## `system_X.par`: Lattice and Species
```
scale = 1.0
cell = read
  0.0 0.5 0.5
  0.5 0.0 0.5
  0.5 0.5 0.0
atoms = frac
  Ga 0.0 0.0 0.0
  N  0.25 0.25 0.25
```
- `scale` multiplies the cell matrix to yield lattice vectors in Bohr.
- `atoms` lists species and fractional positions; species order establishes neural-network outputs and parameter file lookup.

## `input_X.par`: Simulation Metadata
Key-value metadata per system:
- `nBands` (int) – number of band energies provided.
- `maxKE` (float) – kinetic-energy cutoff (Ry) for plane-wave basis generation.
- `idxVB`, `idxCB`, `idxGap` (int) – reference band indices (0-based).
- Plotting controls: `BS_plot_center`, `BS_plot_CBVB_range`, `BS_plot_CBVB_range_zoom`.
- Optional flags: `fit_defPot = 1` to include defect potential targets; `relE_bIdx`, `systemName`, etc., for labeling.

## Spectral Data
- **`kpoints_X.par`** – four columns (`k_x k_y k_z weight`). Weights need not be normalized; the loader handles normalization.
- **`bandWeights_X.par`** – single column of length `nBands`. Non-matching lengths raise errors.
- **`expBandStruct_X.par`** – first column is cumulative k-distance. Remaining columns store band energies (eV). The file can contain comments at the top; `np.loadtxt` handles them once the header is stripped.

## Initialization Data
- **`init_<atom>Params.par`** – nine-line files matching the pseudopotential parameter order described in the README.
- **`init_qSpace_pot.par`** – columns `[q, V_atom1(q), V_atom2(q), ...]`; the loader prints the inferred atom order for verification.
- **`init_PPmodel.pth`** and **`init_AdamState.pth`** (optional) supply pretrained network weights and Adam optimizer state for restarts.

## Optional Data and Auxiliary Controls
- `mcOpts.par`, `mcOpts_ratio.par`, `<atom>ParamSteps.par` fine-tune Monte Carlo run lengths, temperature schedules, and parameter step sizes.
- `expCoupling_*.dat` prepares polarization-resolved coupling matrices; parsed by `BulkSystem.setExpCouplings`.
- `expDefPot.par` or `expDefPot_NEW.par` inject defect formation energies or band-edge shifts.
- `qpoints_X.par` (when available) supplies q-space sampling for special workflows.

With the input files assembled, review [Workflow Modes](workflows.md) to choose the training or refinement strategy that matches your objectives.
