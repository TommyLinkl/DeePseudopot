# NN_pseudopotential_fitting
Semi-empirical pseudopotential methods enable low-cost, high-accuracy quantum chemistry calculations in large systems. They are particularly useful in obtaining accurate eletronic, optical and dynamics properties in nano-scale systems.   

Traditional pseudopotentials typically has a physics-driven function form, with parameters fitted to reproduce high-level *ab initio* or empirical results. Such function forms contain many physical or arbitrary contraints, limiting the degrees of freedom of the pseudopotentials. 

This code parametrizes pseudopotentials for semiconductor nanocrystal systems through physics-inspired deep neural networks, combining the flexibility of neural nets and the necessary physical constraints. The pseudopotentials generated using this method provides better fit to *ab initio* or empirical results and converge faster than traditional pseudopotential functions. 

## Code details

- ``test.ipynb``  
A Jupyter Notebook script that runs ``main.py`` on sample inputs. 

- ``main.py``  
Run this python script to parametrize pseudopotentials through physics-inspired deep neural networks. 

- ``utils/``  
This directory includes classes and functions called by ``main.py``. 
    - ``read.py``: reads in the inputs and constructs the class bulkSystem. 
    - ``nn_models.py``: constructs neural networks. 
    - ``pot_func.py``: includes a variety of utility functions for pseudopotentials. 
    - ``bandStruct.py``: builds the Hamiltonian matrix at every k-point using the NN-pseudopotential. It then diagonalizes Hamiltonian matrices to get corresponding band stuctures. 
    - ``init_NN_train.py``: initialized the neural network by fitting to the latest function form of the pseudopotentials. 
    - ``NN_train.py``: trains the neural network by minimizing the loss with respect to the reference band structures. 

- ``inputs/``  
This directory contains sample input data for running the code, including semiconductor system configurations, expected band structures, k-point inputs, latest parameters for the pseudopotential function form, etc.   

I have done testing on two calculations: 1. ZB_CdSe and 2. InAs, InP, GaAs, GaP systems. Please move the inputs from each folder into the parent ``input/`` directory before running the code for testing. 

- ``results/``  
This directory contains results from the code by running with the sample input data. For reference, results for the two sample runs are collected in separate folders. 

- ``zbCdSe_fitting_numpy.ipynb``
A testing Jupyter notebook script that calculates the zinc-blende CdSe band structure from the Zunger pseudopotential form via numpy. 

- ``zbCdSe_fitting_NN_init.ipynb``
A testing Jupyter notebook script that uses a neural network to fit the band structure of zinc-blende CdSe. Adapts numpy methods to PyTorch tensor library. 

## Theory background

xxx

## TODO

- Add NN hyperparameter optimization (Maybe through RayTune)

- Parallelize across multiple (x4 for Perlmutter) GPUs / multiple GPU cores. Look into data parallelism, Horovod, Model Parallelism for large models. 