# Installation & Quick Start

1. **Clone the repository**
        
        git clone https://github.com/TommyLinkl/DeePseudopot.git
        cd DeePseudopot

2. **Install dependencies**

        pip install -r requirements.txt

3. **Assemble an input bundle**

    At minimum, include the following input files for a deep-learning pseudopotential training calculation. Please see [Input Data Description](inputs.md) for details on input keywords. 

    | Inputs | Purpose |
    | --- | --- |
    | `NN_config.par` | Global training and optimization settings. |
    | `system_X.par` | Defines the periodic system. |
    | `kpoints_X.par` | $\mathbf{k}$-point paths in Brillouin zone for the band structures. |
    | `bandWeights_X.par` | Per-band weights in the training loss definition. |
    | `expBandStruct_X.par` | Reference band structures used for training. |
    | `input_X.par` | Convergence controls, plot toggles, and miscellaneous simulation knobs. |
    | `init_<atom>Params.par` | Analytic pseudopotential parameters per element, used for model initialization. |

4. **Launch a training run**

    ```
    python main.py /path/to/inputs/ /path/to/results/
    ```

## Code Repository Layout (for developers)
`main.py`
: Entry point for training pseudopotentials from an input bundle.

`eval_fullBand.py`
: Mirrors `main.py` but streamlines evaluation of full band structures with inference-friendly parallelism.

`docs/`
: User-facing documentation, including the Input, Output, Workflow, and Troubleshooting guides published on this site.

`utils/`
: Core implementation modules such as file readers, Hamiltonian builders, neural-network models, training loops, Fourier transforms, and visualization utilities.

`test_ham/`, `test_parallel/`, `test_memory/`
: Regression and stress-test suites covering band-structure accuracy, multiprocessing behavior, eigensolvers, and memory utilization.
