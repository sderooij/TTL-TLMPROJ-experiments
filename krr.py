"""

"""
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from functools import partial

# from sklearn.utils import check_X_y, check_array
# from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    check_random_state,
    check_is_fitted,
    check_array,
    check_X_y,
)
from numbers import Real
from sklearn.metrics import accuracy_score, hinge_loss
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel, laplacian_kernel

# from tensorly import tensor
from copy import deepcopy
from abc import ABC, abstractmethod, ABCMeta


class BaseKRR(BaseEstimator, metaclass=ABCMeta):
    """Abstract class for Tensor Kernel Ridge Regression. Format compatible with scikit-learn.

    Args:
        ABC (ABC): Abstract Base Class
    """
    @abstractmethod
    def __init__(
        self,
        kernel="rbf",
        reg_par=1e-5,
        gamma=1.0,
        mu=0,
        random_state=None,
        class_weight=None,
        train_loss_flag=False,
        debug=False,
        deg=2,
    ):
        self.reg_par = reg_par
        self.random_state = random_state
        self.class_weight = class_weight
        self.mu = mu
        self.train_loss_flag = train_loss_flag
        self.train_loss = []
        self.debug = debug
        self.kernel = kernel
        self.gamma = gamma
        self.deg = deg

    @abstractmethod
    def fit(self, x, y, **kwargs):
        return self

    @abstractmethod
    def predict(self, x, **kwargs):
        pass


class KRR(BaseKRR):
    """Kernel Ridge Regression.

    Args:
        BaseKRR (BaseEstimator): Abstract base class for Tensor Kernel Ridge Regression.
    """
    def __init__(
        self,
        kernel="rbf",
        reg_par=1e-5,
        gamma=1.0,
        deg=2,
        random_state=None,
        class_weight=None,
        train_loss_flag=False,
        debug=False,
    ):
        super().__init__(
            kernel=kernel,
            reg_par=reg_par,
            gamma=gamma,
            random_state=random_state,
            class_weight=class_weight,
            train_loss_flag=train_loss_flag,
            debug=debug,
            deg=deg,
        )

    def fit(self, x, y, *, sample_weights=None):
        self.support_vectors_ = deepcopy(x)

        if self.kernel == "rbf":
            K = rbf_kernel(x, gamma=self.gamma)
        elif self.kernel == "linear":
            K = linear_kernel(x)
        elif self.kernel == "poly":
            K = polynomial_kernel(x, degree=self.deg, gamma=self.gamma)
        elif self.kernel == "laplacian":
            K = laplacian_kernel(x, gamma=self.gamma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        N = x.shape[0]
        if sample_weights is not None or self.class_weight is not None:
            if sample_weights is None:
                sample_weights = np.ones(N)
            if self.class_weight == "balanced":
                idx_p = np.where(y == 1)[0]
                idx_n = np.where(y == -1)[0]
                Np = np.sum(y == 1)
                Nn = np.sum(y == -1)
                Cp = N / (2 * Np)
                Cn = N / (2 * Nn)
                sample_weights = np.ones(N)
                sample_weights[idx_p] = 1/Cp
                sample_weights[idx_n] = 1/Cn
            reg_mat = np.diag(sample_weights)

            self.alpha_ = np.linalg.solve(K + self.reg_par * reg_mat, y)
            # sqrt_w = np.sqrt(sample_weights)
            # K_weighted = (K.T * sqrt_w).T * sqrt_w  # Equivalent to W^{1/2} K W^{1/2}
            # y_weighted = y * sqrt_w
            # self.alpha_ = np.linalg.solve(K_weighted + self.reg_par * np.eye(N), y_weighted * sqrt_w)
        else:
            self.alpha_ = np.linalg.solve(K + self.reg_par * np.eye(N), y)

        return self


    def predict(self, x):
        # check_is_fitted(self, 'alpha_', )
        if self.kernel == "rbf":
            K = rbf_kernel(x, self.support_vectors_, gamma=self.gamma)
        elif self.kernel == "linear":
            K = linear_kernel(x, self.support_vectors_)
        elif self.kernel == "poly":
            K = polynomial_kernel(x, self.support_vectors_, degree=self.deg, gamma=self.gamma)
        elif self.kernel == "laplacian":
            K = laplacian_kernel(x, self.support_vectors_, gamma=self.gamma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        return K @ self.alpha_


class KRRRegressor(KRR, RegressorMixin):
    pass


class KRRClassifier(KRR, ClassifierMixin):
    def __init__(
            self,
            kernel="rbf",
            reg_par=1e-5,
            gamma=1.0,
            deg=2,
            random_state=None,
            class_weight=None,
            train_loss_flag=False,
            debug=False,
    ):
        super().__init__(
            kernel=kernel,
            reg_par=reg_par,
            gamma=gamma,
            random_state=random_state,
            class_weight=class_weight,
            train_loss_flag=train_loss_flag,
            debug=debug,
            deg=deg,
        )

    def fit(self, x, y, *, sample_weights=None):
        self.classes_ = np.unique(y)
        super().fit(x, y, sample_weights=sample_weights)
        return self

    def decision_function(self, x):
        """Compute the decision function for the classifier."""
        check_is_fitted(self, 'alpha_')
        return super().predict(x)

    def predict(self, x):
        """Predict class labels for the input data."""
        decision_values = self.decision_function(x)
        return np.sign(decision_values).astype(int)


class KRR_LMPROJ(KRR):
    def __init__(
        self,
        kernel="rbf",
        reg_par=1e-5,
        gamma=1.0,
        deg=2,
        mu=0,
        random_state=None,
        class_weight=None,
        train_loss_flag=False,
        debug=False,
    ):
        super().__init__(
            kernel=kernel,
            reg_par=reg_par,
            gamma=gamma,
            deg=deg,
            random_state=random_state,
            class_weight=class_weight,
            train_loss_flag=train_loss_flag,
            debug=debug,
        )
        self.mu = mu

    def _compute_kernel_mat(self, x, x_target):
        """
        Compute Omega and Lambda matrices for the LMPROJ method
        Args:
            x: source data \in R^{N_s x D}
            x_target: target data \in R^{N_t x D}

        Returns:
            Omega, Lambda, K_s
        """
        N_s = x.shape[0]
        N_t = x_target.shape[0]
        N = N_s + N_t
        Z = np.concatenate([x, x_target], axis=0)
        self.support_vectors_ = deepcopy(Z)

        # Compute the kernel matrix for the source data
        K_s = rbf_kernel(Z, x, gamma=self.gamma) # N x N_s

        # Compute the kernel matrix for the target data
        K_t= rbf_kernel(Z, x_target, gamma=self.gamma)

        # Combine the kernel matrices to form Omega
        Omega = ((1/(N_s**2) * (K_s @ np.ones((N_s, N_s)) @ K_s.T)
                 + 1/(N_t**2) * (K_t @ np.ones((N_t, N_t)) @ K_t.T))
                 - 1/(N_s*N_t) * (
                         K_s @ np.ones((N_s, N_t)) @ K_t.T
                         + K_t @ np.ones((N_t, N_s)) @ K_s.T
                 ))

        # compute Lambda K(Z,Z) from K_s and K_t
        K = rbf_kernel(Z, Z, gamma=self.gamma)
        return Omega, K, K_s


    def fit(self, x, y, *, x_target=None, sample_weights=None):

        N_s = x.shape[0]
        if self.kernel == 'rbf':
            Omega, K, K_s = self._compute_kernel_mat(x, x_target)

        leftMat = N_s*self.mu * Omega + N_s*self.reg_par * K + K_s @ K_s.T
        rightVec = K_s @ y
        self.alpha_ = np.linalg.solve(leftMat, rightVec)
        return self


class KRR_LMPROJ_Classifier(KRR_LMPROJ, ClassifierMixin):
    def decision_function(self, x):
        """Compute the decision function for the classifier."""
        check_is_fitted(self, 'alpha_')
        return super().predict(x)

    def predict(self, x):
        """Predict class labels for the input data."""
        decision_values = self.decision_function(x)
        return np.sign(decision_values).astype(int)





