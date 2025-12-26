# DeepPseudopot

DeepPseudopot is a machine-learned atomistic pseudopotential model that extends the semi-empirical pseudopotential method (SEPM) for simulating large and complex material systems. 

It excels at capturing the electronic structure, photophysics, and charge-carrier dynamics in systems where *ab initio* methods such as GW or hybrid-functional DFT become computationally prohibitive — particularly in nanostructures, alloys, and polymorphic materials.

## How to Cite
Please cite the following paper when referencing DeepPseudopot:

- Lin, K., Coley-O’Rourke, M.J. & Rabani, E. Deep-learning atomistic semi-empirical pseudopotential model for nanomaterials. npj Comput Mater 11, 381 (2025). [https://doi.org/10.1038/s41524-025-01862-5](https://doi.org/10.1038/s41524-025-01862-5)

## Detailed Manual for usage
Please consult [the User's Guide](https://tommylinkl.github.io/DeePseudopot/) for details on input/output file formats, workflow recipes, restart procedures, and troubleshooting tips.  

## Installation & Quick Start
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare an input bundle** with the following at minimum:
   - global training settings `NN_config.par`
   - periodic system definitions `system_X.par`
   - $\mathbf{k}$-point paths `kpoints_X.par`
   - band-specific weights in loss function definition `bandWeights_X.par`
   - reference band structures `expBandStruct_X.par`
   - miscellaneous convergence and plotting-related parameters `input_X.par`
   - initial pseudopotentials for each element `init_<atom>Params.par`
   
   See `docs/manual.md` for details about keyword options and explanations, input units, and restart conventions.
3. **Launch a run**
   ```bash
   python main.py /path/to/inputs/ /path/to/results/
   ```

## Order of Pseudopotential Parameters in the Function Form
The local pseudopotential files `init_<atom>Params.par` and their derivatives expect nine parameters in the following order:
- `ppParams[0]` – `ppParams[3]`: Zunger-form local coefficients
- `ppParams[4]`: long-range parameter (only defined for $N-1$ species to enforce charge neutrality)
- `ppParams[5]`: spin–orbit coupling parameter
- `ppParams[6]`, `ppParams[7]`: nonlocal parameters
- `ppParams[8]`: strain-tensor parameter

## For Developers - Code Repository Layout
- `main.py` – entry point for training pseudopotentials from an input bundle.
- `eval_fullBand.py` – similar to `main.py`, but evaluates full band structures with streamlined utilities and parallelism tuned for inference. 
- `docs/` – user documentation; see `docs/manual.md` for keyword definitions, workflows, and troubleshooting.
- `utils/` – primary implementation modules live here; including readers, Hamiltonian builders, neural network models, training loops, Fourier transforms, and visualization scripts. 
- `test_ham/`, `test_parallel/`, `test_memory/` – regression and stress-test suites covering band-structure accuracy, multiprocessing, eigensolvers, and memory use.

## Extended Toolkit
| Script | Purpose |
| --- | --- |
| `charge_density_from_wfns.py` | Builds real-space charge densities from plane-wave eigenvectors calculated from DeepPseudopot.
| `convert_bgwBS.py` | Translates BerkeleyGW or Quantum ESPRESSO outputs into the DeepPseudopot bundle format.
| `convert_convCell_to_primCell.py` | Converts conventional cells into primitive cells during input preparation.
| `utils/cluster_pp.py` | Performs PCA/K-means analyses to cluster neural network pseudopotentials and assess coverage.
| `inflate_kpoints.py` | Densifies k-point paths for higher-resolution band structure calculations.
| `plot_BS_from_file.py`, `plot_SOC_NL_T_Vloc.py` | Plotting scripts for band structures and decomposed potential components.
