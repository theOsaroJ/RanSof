#!/usr/bin/env python3
"""
Main entry point for the AML Software.

This software predicts high-fidelity (DMRG) energies and forces by progressing
through the fidelity funnel:
    LDA → PBE → r2SCAN → B3LYP → CCSD → DMRG

At each fidelity jump the discrepancy is modeled with a GP whose hyperparameters
are initialized (transfer learning) from the lower-fidelity GP. The active-learning
loop for each level stops either after a specified maximum number of iterations or
once the maximum uncertainty falls below 0.01. Element sequences are extracted from
the multi-molecule XYZ file (smol.xyz) rather than using dummy values.
"""

import os
import json
from active_learning.workflow import run_workflow

def main():
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    run_workflow(config)

if __name__ == "__main__":
    main()
