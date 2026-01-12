from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import numpy as np
import pickle
import pathlib
import os

input_dir = "data/"
save_dir = "C:\\Users\\selinederooij\\surfdrive\\Code\\UniverSVM\\data\\"

unlabeled_value = -3

# iterate over files in input_dir
files = os.listdir(input_dir)
for data_file in files:
    # get filename without .pkl extension
    filename = pathlib.Path(data_file).stem

    # load data from pickle file
    with open(input_dir + data_file, 'rb') as f:
        data = pickle.load(f)

    X_s, y_s, X_t, y_t = data['X_s'], data['y_s'], data['X_t'], data['y_t']

    train_features = np.concatenate([X_s, X_t], axis=0)
    trans_labels = -3*np.ones_like(y_t)
    train_labels = np.concatenate([y_s, trans_labels], axis=0)

    test_features = X_t
    test_labels = y_t

    # save data to libsvm format
    save_folder = save_dir + filename + "/"
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
    train_file = save_folder + "train.dat"
    test_file = save_folder + "test.dat"
    dump_svmlight_file(train_features, train_labels, train_file, zero_based=True)
    dump_svmlight_file(test_features, test_labels, test_file, zero_based=True)



