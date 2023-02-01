# TIGON

# Installment
Packages required:
1. Python3.6 or later
2. torch
3. scipy
4. [TorchDiffEqPack](https://jzkay12.github.io/TorchDiffEqPack/TorchDiffEqPack.odesolver.html)
5. [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
6. random
7. matplotlib
8. functools


# Input Files
`$Dataset.npy`: data coordinates from different time points. One simulation and three single-cell datasets used for TIGON paper are provided in folder `Input/`. 


# Usage
### Inputs:
`--save_dir SAVE_DIR`   Directory for Output Encoding Files. Default value is 'Input/' which store all input files for CLADE  \
`--dataset DATASET`     Name of the data set. Options: 1. GB1; 2. PhoQ. It will load file COMB_LIB.xlsx \
`--encoding ENCODING`  Name of the encoding method; Options: 1. AA; 2.Georgiev. Default: AA \
`--input_dir INPUT_DIR` Directory for Input Files (directory for xlsx files for combinatorial library). Default: Input/\
### Outputs:
`$encoding$.npy` and `$encoding$_normalized.npy` are 3D encoding tensor for combinatorial library. The later one is standardized with zero mean and unit variance. 
`ComboToIndex_$dataset$_$encoding$.pkl`: a dictionary link variant and its index in the xlsx file (`comb_lib`).   
### Examples:
`python3 TIGON.py --encoding AA`


# Sources
## Lineage tracing dataset
Single-cell lineage tracing dataset (raw data of `Lineage.npy`) can be obtained from: [Weinreb, Caleb, et al. "Lineage tracing on transcriptional landscapes links state to fate during differentiation." Science 367.6479 (2020): eaaw3381.](https://www.science.org/doi/full/10.1126/science.aaw3381?casa_token=cmaoSgI9KNQAAAAA%3Ah7lDBD7kPIfZDBTlYDHy9RPVHjX811LOPfxDitvbLiAugMxB1UUWvqMTtzKL4hU3oKdbyfBCw7mmIA)
## EMT dataset
TGFB1 induced EMT from A549 cancer cell line dataset (raw data of `EMT.npy`) can be obtained from: [Cook, David P., and Barbara C. Vanderhyden. "Context specificity of the EMT transcriptional response." Nature communications 11.1 (2020): 1-9.](https://www.nature.com/articles/s41467-020-16066-2)
## Bifurcation dataset
Single-cell qPCR dataset of iPSCs toward cardiomyocytes dataset (raw data of `Bifurcation.npy`) can be obtained from: [Bargaje, Rhishikesh, et al. "Cell population structure prior to bifurcation predicts efficiency of directed differentiation in human induced pluripotent cells." Proceedings of the National Academy of Sciences 114.9 (2017): 2271-2276.](https://www.pnas.org/doi/abs/10.1073/pnas.1621412114)
# Reference