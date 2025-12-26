# User's Guide for DeepPseudopot

Welcome! This site centralizes setup guides, reference details, and troubleshooting tips for running DeepPseudopot in production workflows.

DeepPseudopot is a machine-learned atomistic pseudopotential model that extends the semi-empirical pseudopotential method (SEPM) for simulating large and complex material systems. 

It excels at capturing the electronic structure, photophysics, and charge-carrier dynamics in systems where *ab initio* methods such as GW or hybrid-functional DFT become computationally prohibitive — particularly in nanostructures, alloys, and polymorphic materials.

## How to Cite
Please cite the following paper when referencing DeepPseudopot:

- Lin, K., Coley-O’Rourke, M.J. & Rabani, E. Deep-learning atomistic semi-empirical pseudopotential model for nanomaterials. npj Comput Mater 11, 381 (2025). [https://doi.org/10.1038/s41524-025-01862-5](https://doi.org/10.1038/s41524-025-01862-5)


## Table of Contents
- [Environment & Installation](install.md)
- [Workflow Modes](workflows.md) – Supported training and refinement workflow modes, plus a step-by-step execution flow.
- [Input Data Description](inputs.md) – Checklist and pointers for every configuration, lattice, and spectral input.
- [Output Data Description](outputs.md) – Catalogue of initialization, checkpoint, and final deliverables.
- [Troubleshooting Guide](troubleshooting.md) – Quick answers to the most common failure modes.
