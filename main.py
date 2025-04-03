#!/usr/bin/env python3
"""
Main entry point for the AML Software.

This software predicts high-fidelity (DMRG) energies and forces by progressing
through the fidelity funnel:
    LDA → PBE → r2SCAN → B3LYP → CCSD → DMRG

The LDA energies and forces are provided by the user via external files
(which must be in the order corresponding to smol.xyz). These serve as the
initial training data.
"""

import os
import json
from active_learning.workflow import run_workflow

def main():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    config_file = os.path.join(base_dir, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    run_workflow(config)

if __name__ == "__main__":
    main()
