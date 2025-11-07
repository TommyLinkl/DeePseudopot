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

| Key | Type | Applies to | Required? | Notes |
| --- | --- | --- | --- | --- |
| `scale` | float | Entire file | Yes | Multiplies the cell matrix to yield lattice vectors in Bohr. |
| `cell` | 3×3 matrix | Entire file | Yes | Primitive-cell vectors that, when combined with `scale`, define the lattice in Bohr. |
| `atoms` | block (species + fractional coords) | Entire file | Yes | Species order establishes neural-network outputs and parameter file lookup. |

## `input_X.par`: Simulation Metadata
| Key | Type | Applies to | Required? | Notes |
| --- | --- | --- | --- | --- |
| `nBands` | int | System `X` | Yes | Number of band energies provided; must match `bandWeights_X.par` length and `expBandStruct_X.par` columns. |
| `maxKE` | float (Ry) | System `X` | Yes | Plane-wave kinetic-energy cutoff used in Hamiltonian assembly. |
| `idxVB`, `idxCB`, `idxGap` | int | System `X` | Optional but recommended | Zero-based band indices for labeling valence/conduction edges and bandgap references. |
| `BS_plot_center`, `BS_plot_CBVB_range`, `BS_plot_CBVB_range_zoom` | float | System `X` | Optional | Control y-axis windows for the generated band-structure PDFs. |
| `fit_defPot`, `relE_bIdx`, `systemName`, etc. | mixed | Specialized workflows | Optional | Enable defect potentials, relative-energy references, or descriptive labels. |

## Spectral Data
| Key | Type | Applies to | Required? | Notes |
| --- | --- | --- | --- | --- |
| `kpoints_X.par` | text file | System `X` | Yes | Four columns (`k_x k_y k_z weight`). Weights need not be normalized; the loader normalizes internally. |
| `bandWeights_X.par` | text file | System `X` | Yes | Single column of length `nBands`. Any mismatch triggers an error. |
| `expBandStruct_X.par` | text file | System `X` | Yes | Column 0 is cumulative k-distance; remaining columns store band energies (eV). Header comments are allowed. |

## Initialization Data
| Key | Type | Applies to | Required? | Notes |
| --- | --- | --- | --- | --- |
| `init_<atom>Params.par` | text file | Each species | Required unless other init provided | Nine-line files following the parameter order in the README (Zunger coefficients, long-range, SO, nonlocal, strain). |
| `init_qSpace_pot.par` | text file | Initialization | Optional | Columns `[q, V_{atom1}(q), V_{atom2}(q), ...]`; loader echoes the inferred atom order. |
| `init_PPmodel.pth`, `init_AdamState.pth` | binary | Restarts | Optional | Provide pretrained NN weights and Adam optimizer state for seamless restarts. |

## Optional Data and Auxiliary Controls
| Key | Type | Applies to | Required? | Notes |
| --- | --- | --- | --- | --- |
| `mcOpts.par`, `mcOpts_ratio.par`, `<atom>ParamSteps.par` | text files | Monte Carlo runs | Optional | Fine-tune MC iteration counts, temperature schedules, and per-atom step sizes. |
| `expCoupling_*.dat` | text/binary | Optical workflows | Optional | Polarization-resolved coupling matrices consumed by `BulkSystem.setExpCouplings`. |
| `expDefPot.par`, `expDefPot_NEW.par` | text files | Defect workflows | Optional | Provide defect formation energies or band-edge shifts for fitting. |
| `qpoints_X.par` | text file | Specialized workflows | Optional | Supplies q-space sampling for workflows that require explicit q-grid inputs. |

With the input files assembled, review [Workflow Modes](workflows.md) to choose the training or refinement strategy that matches your objectives.
