#!/usr/bin/env python3
"""
ForceGP: A wrapper that trains one Gaussian Process per force component.
Each force component is modeled separately using the custom GP.
"""

import numpy as np
from models.custom_gp import GaussianProcess

class ForceGP:
    def __init__(self, output_dim, length_scale=1e-5, noise=1e-5, alpha_rq=1e-5, batch_size=200):
        """
        Initializes ForceGP with one GP per force component.
        output_dim: number of force components (e.g., n_atoms * 3)
        """
        self.output_dim = output_dim
        self.gp_list = []
        self.X_train = None  # Common training data
        for _ in range(output_dim):
            gp = GaussianProcess(length_scale=length_scale, noise=noise, alpha_rq=alpha_rq, batch_size=batch_size)
            self.gp_list.append(gp)

    def fit(self, X_train, Y_train):
        """
        Fits each GP on its corresponding column of Y_train.
        X_train: shape (n_samples, n_features)
        Y_train: shape (n_samples, output_dim)
        """
        n, d = Y_train.shape
        if d != self.output_dim:
            raise ValueError(f"Expected Y_train with {self.output_dim} columns, got {d}.")
        self.X_train = X_train  # Save common training features
        for i in range(self.output_dim):
            y = Y_train[:, i].reshape(-1, 1)
            self.gp_list[i].fit(X_train, y)

    @property
    def y_train(self):
        """
        Returns the concatenated training targets from each GP as a 2D array.
        """
        return np.hstack([gp.y_train for gp in self.gp_list])

    def predict(self, X_test):
        """
        Predicts each force component.
        Returns:
          mu: shape (n_test, output_dim)
          var: shape (n_test, output_dim)
        """
        mu_list = []
        var_list = []
        for gp in self.gp_list:
            mu, var = gp.predict(X_test)
            mu_list.append(mu)
            var_list.append(var)
        mu = np.hstack(mu_list)
        var = np.hstack(var_list)
        return mu, var

    def optimize_hyperparameters(self, maxiter=10000):
        for gp in self.gp_list:
            gp.optimize_hyperparameters(maxiter=maxiter)
