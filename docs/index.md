# User's Guide for DeepPseudopot

Welcome! This site centralizes setup guides, reference details, and troubleshooting tips for running DeepPseudopot in production workflows.

DeepPseudopot is a machine-learned atomistic pseudopotential model that extends the semi-empirical pseudopotential method (SEPM) for simulating large and complex material systems. 

It excels at capturing the electronic structure, photophysics, and charge-carrier dynamics in systems where *ab initio* methods such as GW or hybrid-functional DFT become computationally prohibitive — particularly in nanostructures, alloys, and polymorphic materials.

## How to Cite
Please cite the following paper when referencing DeepPseudopot:

- Preprint: [arXiv:2505.09846](https://arxiv.org/abs/2505.09846).
- npj Computational Materials article (soon in press).

## Table of Contents
- [Environment & Installation](install.md)
- [Workflow Modes](workflows.md) – Supported training and refinement workflow modes, plus a step-by-step execution flow.
- [Input Data Description](inputs.md) – Checklist and pointers for every configuration, lattice, and spectral input.
- [Output Data Description](outputs.md) – Catalogue of initialization, checkpoint, and final deliverables.
- [Troubleshooting Guide](troubleshooting.md) – Quick answers to the most common failure modes.
