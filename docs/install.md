## Installation & Quick Start

!!! example "Install dependencies"
    ```bash
    pip install -r requirements.txt
    ```

!!! note "Assemble an input bundle"
    At minimum, include the following artifacts (see [Input Data Description](inputs.md) for keyword tables and restart guidance):

    | Artifact | Purpose |
    | --- | --- |
    | `NN_config.par` | Global training and optimization settings. |
    | `system_X.par` | Periodic system definitions and atom ordering. |
    | `kpoints_X.par` | High-symmetry paths and sampling in reciprocal space ($\mathbf{k}$-points). |
    | `bandWeights_X.par` | Per-band weights in the loss definition. |
    | `expBandStruct_X.par` | Reference band structures used for supervision. |
    | `input_X.par` | Convergence controls, plot toggles, and miscellaneous simulation knobs. |
    | `init_<atom>Params.par` | Initial pseudopotentials or analytic seed parameters per element. |

!!! example "Launch a training run"
    ```bash
    python main.py /path/to/inputs/ /path/to/results/
    ```

## For Developers - Code Repository Layout
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
