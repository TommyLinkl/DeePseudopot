# NN_pseudopotential_fitting
This code parametrizes semi-empirical pseudopotentials through deep neural networks for low-cost, high-accuracy quantum chemistry calculations

## Code details
- main.py
Run this script

- utils
Includes classes and functions called by main.py. 
nn_models.py - constructs neural networks. 
pot_func.py - includes a variety of utility functions about pseudopotentials. 
bandStruct.py - builds the hamiltonian at every k-point using the NN-pseudopotential and diagonalizes them to get the band stuctures. 
init_NN_train.py - initialized the neural network by fitting to the latest function form of the pseudopotentials. 
NN_train.py - trains the neural network by fitting to the band structures. 


- data
Input data for the code, including semiconductor system configurations and setups, expected band structures, k-point inputs, laterst function form of the pseudopotentials. 
read.py - reads in the inputs and constructs the class bulkSystem. 

- zbCdSe_fitting_numpy.ipynb: 
Jupyter notebook that calculates the zinc-blende CdSe band structure from the Zunger pseudopotential form via numpy

- zbCdSe_fitting_NN_init.ipynb: 
Jupyter notebook that uses a neural network to fit the band structure of zinc-blende CdSe. Adapts numpy methods to PyTorch tensor library. 

