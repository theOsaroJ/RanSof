#!/usr/bin/env python3
class MultiFidelityCokriging:
    def __init__(self, rho=1.0, low_gp=None, discrepancy_gp=None):
        self.rho = rho
        self.low_gp = low_gp
        self.discrepancy_gp = discrepancy_gp

    def fit(self, X_low, y_low, X_high, y_high):
        y_low_pred, _ = self.low_gp.predict(X_high)
        delta = y_high - self.rho * y_low_pred
        self.discrepancy_gp.fit(X_high, delta)
        self.discrepancy_gp.optimize_hyperparameters()

    def predict(self, X):
        y_low_pred, var_low = self.low_gp.predict(X)
        delta_pred, var_delta = self.discrepancy_gp.predict(X)
        y_pred = self.rho * y_low_pred + delta_pred
        var_pred= var_low + var_delta
