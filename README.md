# README

This repo contains experiments to test the T-LMPROJ model. This is a transductive transfer learning model.

The implementation of the model can be found in another repository: 

This repository also contains an implementation of the original LMPROJ model (using cvxpy).

To benchmark the T-LMPROJ we test it against the original LMPROJ model and the transductive SVM model (tSVM). For the transductive SVM model the code from the UniverSVM paper can be used (this is found here: ). A modified Dockerfile was created to be able to use this code (see folder XX for this and how to use it).

### Files

- `synthetic_experiments_LMPROJ.ipynb` is a jupyter notebook detailing how the synthetic data is generated and shows some examples. The data folder contains `.pkl` files with this synthetic data.
- `plot_synthetic_data.py` contains the functions used to plot the data (and save these plots in the `plots` folder).
- `TL_20newsgroups.py` contains the experiments for the 20 Newsgroups datasets. This has not yet been tested with the tSVM model. Currently only compatible with the (T-)LMRPROJ models.
- ` transfer_data_to_libsvm.py` file to transfer the (syntethic) data from .pkl format to the libsvm .dat format. 
- `calculate_tsvm_accuracies.py` file to calculate the accuracy of the tSVM model based on the outputs in .dat format. 
- `krr.py` contains a python implemetation of the KRR (and KRR_LMPROJ) models. 
- 