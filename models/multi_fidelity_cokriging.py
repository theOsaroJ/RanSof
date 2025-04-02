#!/usr/bin/env python3
"""
Multi-Fidelity Co-Kriging Model with Transfer Learning

This model expresses the high-fidelity output as:
    y_H(x) = rho * y_L(x) + delta(x)
where:
  - y_L(x) is the low-fidelity prediction,
  - rho is a scaling factor,
  - delta(x) is the discrepancy modeled by a GP.
Transfer learning is applied by initializing the discrepancy GP’s hyperparameters with those from the low-fidelity GP.
"""

class MultiFidelityCokriging:
    def __init__(self, rho=1.0, low_gp=None, discrepancy_gp=None):
        self.rho = rho
        self.low_gp = low_gp
        self.discrepancy_gp = discrepancy_gp
    
    def fit(self, X_low, y_low, X_high, y_high):
        # low_gp is assumed to be pre-trained on low-fidelity data.
        y_low_pred, _ = self.low_gp.predict(X_high)
        delta = y_high - self.rho * y_low_pred
        self.discrepancy_gp.fit(X_high, delta)
        self.discrepancy_gp.optimize_hyperparameters()
    
    def predict(self, X):
        y_low_pred, var_low = self.low_gp.predict(X)
        delta_pred, var_delta = self.discrepancy_gp.predict(X)
        y_pred = self.rho * y_low_pred + delta_pred
        var_pred = var_low + var_delta
        return y_pred, var_pred
