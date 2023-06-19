# TIGON 
(**T**rajectory **I**nference with **G**rowth via **O**ptimal transport and **N**eural network)

# Installment
Packages required:
1. Python=3.6.8
2. torch=1.9.0
3. scipy=1.5.2
4. [TorchDiffEqPack](https://jzkay12.github.io/TorchDiffEqPack/TorchDiffEqPack.odesolver.html)=0.1.0
5. [torchdiffeq](https://github.com/rtqichen/torchdiffeq)=0.2.2


Generating plots to visualize results:
1. numpy=1.19.5
2. seaborn=0.11.2
3. matplotlib=3.1.2


# Input Files
`$Dataset.npy`: data coordinates from different time points. One simulation and three single-cell datasets used for TIGON paper are provided in folder `Input/`. 


# Usage
### Inputs:
`--dataset` Name of the data set. Options: EMT; Lineage; Bifurcation; Simulation, default = 'Simulation'. \
`--input_dir` Input Files Directory, default='Input/'. \
`--save_dir` Output Files Directory, default='Output/'. \
`--timepoints` Time points of data. \
`--niters` Number of traning iterations. \
`--lr` Learning rate. \
`--num_samples` Number of sampling points per epoch. \
`--hidden_dim` Dimension of hidden layer. \
`--n_hiddens` Number of hidden layers for the neural network learning velocity. 

### Outputs:
`ckpt.pth`: save model’s parameters and training errors.

### Examples:
`python3 TIGON.py`
Or
Check `Notebooks/Training.ipynb `

### Plot results:
`python3 plot_result.py`
Or
Check `Notebooks/Visualization.ipynb `

# Sources
## Lineage tracing dataset
Single-cell lineage tracing dataset (raw data of `Lineage.npy`) can be obtained from: [Weinreb, Caleb, et al. "Lineage tracing on transcriptional landscapes links state to fate during differentiation." Science 367.6479 (2020): eaaw3381.](https://www.science.org/doi/full/10.1126/science.aaw3381?casa_token=cmaoSgI9KNQAAAAA%3Ah7lDBD7kPIfZDBTlYDHy9RPVHjX811LOPfxDitvbLiAugMxB1UUWvqMTtzKL4hU3oKdbyfBCw7mmIA)
## EMT dataset
TGFB1 induced EMT from A549 cancer cell line dataset (raw data of `EMT.npy`) can be obtained from: [Cook, David P., and Barbara C. Vanderhyden. "Context specificity of the EMT transcriptional response." Nature communications 11.1 (2020): 1-9.](https://www.nature.com/articles/s41467-020-16066-2)
## Bifurcation dataset
Single-cell qPCR dataset of iPSCs toward cardiomyocytes dataset (raw data of `Bifurcation.npy`) can be obtained from: [Bargaje, Rhishikesh, et al. "Cell population structure prior to bifurcation predicts efficiency of directed differentiation in human induced pluripotent cells." Proceedings of the National Academy of Sciences 114.9 (2017): 2271-2276.](https://www.pnas.org/doi/abs/10.1073/pnas.1621412114)
# Reference
