# symbolic-regression-utilities


Symbolic-Regression-Utilities is a collection of python modules and scripts designed for automating experiments with the [sure independence screening and sparsifying operator (SISSO)](https://github.com/rouyang2017/SISSO) framework. This repository is under development so suggestions and feature requests are always welcome. The project was used in the following publications:

*S. R. Xie, G. R. Stewart, J. J. Hamlin, P. J. Hirschfeld, and R. G. Hennig, Functional Form of the Superconducting Critical Temperature from Machine Learning, Phys. Rev. B 100, (2019).*

*S. R. Xie, P. Kotlarz, R. G. Hennig, and J. C. Nino, Machine Learning of Octahedral Tilting in Oxide Perovskites by Symbolic Classification with Compressed Sensing, Computational Materials Science 180, 109690 (2020).*

Please consider citing these papers if you find this repository helpful.


# Installation
```
conda create --name sru python=3.6
conda activate sru
git clone https://github.com/henniggroup/symbolic-regression-utilities.git
cd symbolic-regression-utilities
pip install -e .
```

#### Requirements
```
python >= 3.6
pint
sympy
```

#### Optional requirements
```
matplotlib
tqdm
```

# Examples
The ```examples``` directory contains two Jupyter notebooks to demonstrate the use of various modules in the package. 
* ```demo_inputs.ipynb``` shows how to prepare and generate input files for SISSO
* ```demo_outputs.ipynb``` shows how to process the output files and filter by units and mathematical constraints.

# Datasets
Two datasets, used in previous publications, are provided in the ```datasets``` directory:
* ```AllenDynes```: symbolic regression of the superconducting critical temperature
* ```octahedral_tilting```: symbolic classification of octahedral tilting
