from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from tensorlibrary.learning.transfer import CPKRR_LMPROJ, SVC_LMPROJ
from tensorlibrary.learning import CPKRR
from sklearn.datasets import load_wine

# parameters
sig = 0.3
M = 10
Ld = 1.8 #2.2
R = 5
reg_par = 1e-4
C = 10
RANDOM_STATE = 10
gam = 1/ (sig**2)
N_train = 0.001

# fetch dataset
X, y = load_wine(return_X_y=True)

source_labels = (2, 1)
target_labels = (0, 1)



print(X.shape, y.shape, set(y))
# Output: (48842, 14) (48842,) {-1, 1}

idx = [None]*3
idx[0] = np.where(y == 0)[0].tolist()
idx[1] = np.where(y == 1)[0].tolist()
idx[2] = np.where(y == 2)[0].tolist()

# sample for source
idx_s = idx[source_labels[0]] + idx[source_labels[1]]
y_s = y[idx_s]
X_s = X[idx_s]
# set class 0 to -1
y_s[y_s == source_labels[0]] = -1
y_s[y_s == source_labels[1]] = 1

idx_t = idx[target_labels[0]] + idx[target_labels[1]]
y_t = y[idx_t]
X_t = X[idx_t]
# set class 2 to -1
y_t[y_t==target_labels[0]] = -1
y_t[y_t==target_labels[1]] = 1

# idx = idx0 + idx1
# y = y[idx]
# X = X[idx]
# y = y*2 -1

# undersample the dataset
# N = len(y)
# N_train = int(N_train * N)
# indices = np.random.choice(X.shape[0], N_train, replace=False)
# X_s = X[indices]
# y_s = y[indices]
# X_t = X[~np.isin(np.arange(N), indices)]
# y_t = y[~np.isin(np.arange(N), indices)]
# N = 4000
# indices = np.random.choice(X.shape[0], N, replace=False)
# X = X[indices]
# y = y[indices]
#
# Use k-means clustering to "split" the dataset into source and target domains
from sklearn.cluster import KMeans, SpectralClustering
# from kernel_kmeans import KernelKMeans
# # kmeans = KMeans(n_clusters=2, random_state=42)
# kmeans = KernelKMeans(n_clusters=2, kernel='rbf', gamma=gam)
# labels = kmeans.fit_predict(X)
# # spectral = SpectralClustering(n_clusters=2,  affinity='nearest_neighbors', n_neighbors=5)
# # labels = spectral.fit_predict(X)
# idx_s = labels == 1
# idx_t = labels == 0
# X_s = X[idx_s] # first 4 clusters as source domain
# y_s = y[idx_s]
# X_t = X[idx_t] # last 4 clusters as target domain
# y_t = y[idx_t]
print(f"Source domain samples: {X_s.shape[0]}, Target domain samples: {X_t.shape[0]}")

# standardize features
scaler = MinMaxScaler((-0.5, 0.5))
X_s = scaler.fit_transform(X_s)
X_t = scaler.transform(X_t)

# train source model
source_model = CPKRR(
    feature_map='rbf',
    map_param=sig,
    M=M,
    num_sweeps=10,
    Ld=Ld,
    mu=0,
    max_rank=R,
    reg_par=reg_par,
    train_loss_flag=True,
)

source_model.fit(X_s, y_s)
# evaluate source model on target data
y_t_pred_source = source_model.predict(X_t)
acc_source = accuracy_score(y_t, y_t_pred_source)
print(f"Source model accuracy on target data: {acc_source:.4f}")
# Output: Source model accuracy on target data:

# train adapted model using LMPROJ
adapted_model = CPKRR_LMPROJ(
    feature_map='rbf',
    map_param=sig,
    M=M,
    num_sweeps=10,
    reg_par=reg_par,
    max_rank=R,
    Ld=Ld,
    mu=(X_s.shape[0] + X_t.shape[0]),
    random_init=True,
    train_loss_flag=False,
)

adapted_model.fit(X_s, y_s, x_target=X_t)
# evaluate adapted model on target data
y_t_pred_adapted = adapted_model.predict(X_t)
acc_adapted = accuracy_score(y_t, y_t_pred_adapted)
print(f"Adapted model accuracy on target data: {acc_adapted:.4f}")
# Output: Adapted model accuracy on target data:



