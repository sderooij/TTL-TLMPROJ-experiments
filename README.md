# README

This repo contains experiments to test the T-LMPROJ model. This is a transductive transfer learning model.

The implementation of the models can be found in another repository: https://github.com/sderooij/tensorlibrary/releases/tag/v0.0.4 , which can be installed using pip.


This repository also contains an implementation of the original LMPROJ model (using cvxpy).

To benchmark the T-LMPROJ we test it against the original LMPROJ model.

### Files

- `synthetic_experiments_LMPROJ.ipynb` is a jupyter notebook detailing how the synthetic data is generated and shows some examples. The data folder contains `.pkl` files with this synthetic data.
- `plot_synthetic_data.py` contains the functions used to plot the data (and save these plots in the `plots` folder).
- `TL_20newsgroups.py` contains the experiments for the 20 Newsgroups datasets. This has not yet been tested with the tSVM model. Currently only compatible with the (T-)LMRPROJ models. 
- `krr.py` contains a python implemetation of the KRR (and KRR_LMPROJ) models. 