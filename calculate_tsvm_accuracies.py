import numpy as np
import pathlib
import re
import pickle
from sklearn.metrics import accuracy_score
"""
Parse output files of format:
# function values
f{1} = [ 1.82674 ,1.43144 ,1.33084 ,-0.0390043 ,2.09129 ,2.4343 ,-0.305154 ,1.90423 ,2.44912 ];
"""
def parse_libsvm_output(file_path):
    """
    Parse the output file and extract function values.

    Parameters:
    file_path (str): Path to the output file.

    Returns:
    np.ndarray: Array of function values.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    match = re.search(r'\[([^\]]+)\]', content)

    if match:
        values_str = match.group(1)
        # Convert the string of values into a numpy array
        values = np.fromstring(values_str, sep=',')
        return values
    else:
        raise ValueError("Function values not found in the file.")

# file = "data/predicted_output_universvm.dat"
# values = parse_libsvm_output(file)
# # compute scores
# # test_data
# import pickle
# with open('data/data.pkl', 'rb') as f:
#     data = pickle.load(f)
# X_t = data['X_t']
# y_t = data['y_t']
#
# from sklearn.metrics import f1_score
# y_pred = np.sign(values)
# f1 = f1_score(y_t, y_pred)
# print(f1)

results_directory = "C:\\Users\\selinederooij\\surfdrive\\Code\\UniverSVM\\data\\"
input_dir = "data/"
import os
acc_tsvm = []
for data_file in os.listdir(input_dir):
    # get filename without .pkl extension
    filename = pathlib.Path(data_file).stem

    # load data from pickle file
    with open(input_dir + data_file, 'rb') as f:
        data = pickle.load(f)

    X_s, y_s, X_t, y_t = data['X_s'], data['y_s'], data['X_t'], data['y_t']
    # parse the corresponding output file
    output_file = results_directory + filename + "/output.dat"
    values = parse_libsvm_output(output_file)
    y_pred = np.sign(values)
    acc = accuracy_score(y_t, y_pred)
    acc_tsvm.append(acc)

print(f"Mean accuracy of tSVM over all datasets: {np.mean(acc_tsvm):.4f} ± {np.std(acc_tsvm):.4f}")
