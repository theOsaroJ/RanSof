#!/usr/bin/env python3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import json
import pickle
from math import erf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

###############################################################################
# 1) Custom multi-output GaussianProcess with hyperparameter optimization
###############################################################################
class GaussianProcess:
    def __init__(self, length_scale=1.0, noise=0.1, alpha_rq=1.0, batch_size=2000):
        """
        length_scale, noise: initial hyperparameters
        alpha_rq: alpha for RationalQuadratic
        batch_size: used for predict_in_batches
        """
        self.length_scale = length_scale
        self.noise        = noise
        self.alpha_rq     = alpha_rq
        self.batch_size   = batch_size

        self.X_train   = None
        self.y_train   = None
        self.output_dim= None
        self.K_inv     = None

    def rbf_kernel(self, XA, XB, length_scale):
        sqdist = cdist(XA, XB, "sqeuclidean")
        return np.exp(-0.5 * sqdist / (length_scale**2))

    def rq_kernel(self, XA, XB, length_scale, alpha):
        sqdist = cdist(XA, XB, "sqeuclidean")
        factor = 1.0 + sqdist / (2.0*alpha*(length_scale**2))
        return factor ** (-alpha)

    def matern32_kernel(self, XA, XB, length_scale):
        r = cdist(XA, XB, "euclidean")
        sqrt3 = np.sqrt(3.0)
        scaled_r = sqrt3 * r / length_scale
        return (1.0 + scaled_r)*np.exp(-scaled_r)

    def combined_kernel(self, XA, XB, length_scale=None, noise=None, include_noise=False):
        """
        We'll use (RBF + RQ + Matern32)/3 as the combined kernel.
        If include_noise and XA.shape[0]==XB.shape[0], add noise^2 * I on diagonal.
        """
        if length_scale is None:
            length_scale = self.length_scale
        if noise is None:
            noise = self.noise

        K_rbf   = self.rbf_kernel(XA, XB, length_scale)
        K_rq    = self.rq_kernel(XA, XB, length_scale, alpha=self.alpha_rq)
        K_mat32 = self.matern32_kernel(XA, XB, length_scale)

        # Combine equally
        K = (K_rbf + K_rq + K_mat32) / 3.0

        if include_noise and (XA.shape[0] == XB.shape[0]):
            K += (noise**2)*np.eye(XA.shape[0])
        return K

    def fit(self, X_train, Y_train):
        """
        Fit the kernel matrix with current self.length_scale, self.noise
        and invert it once for multi-output.
        """
        self.X_train = X_train
        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1,1)
        self.y_train   = Y_train
        self.output_dim= Y_train.shape[1]

        K = self.combined_kernel(X_train, X_train,
                                 length_scale=self.length_scale,
                                 noise=self.noise,
                                 include_noise=True)
        self.K_inv = np.linalg.inv(K)

    def predict_in_batches(self, X_test):
        N_test = X_test.shape[0]
        mu  = np.zeros((N_test, self.output_dim))
        var = np.zeros((N_test, self.output_dim))

        for start in range(0, N_test, self.batch_size):
            end = min(start + self.batch_size, N_test)
            X_batch = X_test[start:end]

            K_star = self.combined_kernel(X_batch, self.X_train,
                                          length_scale=self.length_scale,
                                          noise=self.noise,
                                          include_noise=False)
            mu_batch = K_star @ self.K_inv @ self.y_train

            K_starstar = self.combined_kernel(X_batch, X_batch,
                                              length_scale=self.length_scale,
                                              noise=self.noise,
                                              include_noise=False)
            cov_batch  = K_starstar - K_star @ self.K_inv @ K_star.T
            var_batch_diag = np.diag(cov_batch)
            var_batch_all  = np.tile(var_batch_diag.reshape(-1,1), (1, self.output_dim))

            mu[start:end,:]  = mu_batch
            var[start:end,:] = var_batch_all

        return mu, var

    def predict(self, X_test):
        return self.predict_in_batches(X_test)

    # ------------------------
    # Negative Log Marginal Likelihood
    # ------------------------
    def neg_log_marginal_likelihood(self, log_params):
        """
        log_params = [log_length_scale, log_noise]
        We'll exponentiate them.
        """
        length_scale_cand = np.exp(log_params[0])
        noise_cand        = np.exp(log_params[1])

        K = self.combined_kernel(self.X_train, self.X_train,
                                 length_scale=length_scale_cand,
                                 noise=noise_cand,
                                 include_noise=True)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e15

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
        n = self.y_train.shape[0]
        ll = 0.0
        for d in range(self.output_dim):
            y_term  = -0.5*(self.y_train[:,d].T @ alpha[:,d])
            logdet  = - np.sum(np.log(np.diag(L)))
            cst     = - 0.5 * n * np.log(2*np.pi)
            ll_comp = y_term + logdet + cst
            ll     += ll_comp
        return -ll

    def optimize_hyperparameters(self, maxiter=50):
        """
        Minimizes neg_log_marginal_likelihood wrt log_length_scale, log_noise
        using L-BFGS-B, then updates self.length_scale, self.noise.
        Then re-fit.
        """
        def objective(log_params):
            return self.neg_log_marginal_likelihood(log_params)

        init_logs = np.array([np.log(self.length_scale),
                              np.log(self.noise)])
        bounds    = [(np.log(1e-5), np.log(1e5)),  # length_scale
                     (np.log(1e-8), np.log(1.0))] # noise

        res = minimize(objective, init_logs,
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter':maxiter})

        self.length_scale = float(np.exp(res.x[0]))
        self.noise        = float(np.exp(res.x[1]))
        self.fit(self.X_train, self.y_train)
        return res
