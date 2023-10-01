# NN_pseudopotential_fitting
Parametrize semi-empirical pseudopotentials through deep neural networks for low-cost, high-accuracy quantum chemistry calculations

## Code details
- zbCdSe_fitting_numpy.ipynb: Calculates the zinc-blende CdSe band structure from the Zunger pseudopotential form via numpy

- zbCdSe_fitting_NN.ipynb: Uses a neural network to fit the band structure of zinc-blende CdSe. Adapts the numpy methods to PyTorch tensor library. 

- zbCdSe_fitting_NNGPU.ipynb: Accelerates the neural network fitting by using GPU cores. 

- allCdSeS_fitting_NNGPU.ipynb: Extends the neural network fitting to four Cd, Se, S materials: zinc-blende CdSe, zinc-blende CdS, wurtzite CdSe, wurtzite CdS.  

## TODO
- GPU acceleration

- Multiple material systems

- Customizable LOSS function to reflect weights