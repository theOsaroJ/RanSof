#!/usr/bin/env python3
"""
analysis_multi.py

Analyzes multiple methods (e.g., PBE, r2SCAN, B3LYP, CCSD) in one go.
- Reads final_unlabeled_data_{method}.csv (with Molecule_Index),
- Reads predicted_energy_{method}_unlabeled.dat, predicted_forces_{method}_unlabeled.dat,
- Reads actual_data from {method}_energy.dat, {method}_forces.dat,
- Matches by Molecule_Index to get the actual vs. predicted for each candidate,
- Computes error metrics for energy and for each force coordinate dimension,
- Generates parity plots (energy, and each force dimension x,y,z),
- Generates box plots of error distribution (energy, each dimension),
- Saves a merged CSV for each method with all data.

Usage:
    python3 analysis_multi.py \
        --xyz smol.xyz \
        --actual_dir actual_data \
        --pred_dir output \
        --methods PBE r2SCAN CCSD \
        --output_dir analysis_output
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def read_xyz_molecules(xyz_file):
    """
    Reads a multi-molecule XYZ file, returns:
      - molecules: a list of (n_atoms, 3) float arrays,
      - elem_sequences: list of space-separated element strings
        in the same order.
    """
    if not os.path.isfile(xyz_file):
        raise FileNotFoundError(f"XYZ file '{xyz_file}' not found.")
    molecules = []
    elem_sequences = []
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            natom = int(line)
            i += 1  # skip comment line
            if i < len(lines):
                comment = lines[i].strip()
            i += 1
            mol_coords = []
            elems = []
            for _ in range(natom):
                parts = lines[i].split()
                i += 1
                elems.append(parts[0])
                mol_coords.append([float(x) for x in parts[1:4]])
            molecules.append(np.array(mol_coords))
            elem_sequences.append(" ".join(elems))
        else:
            i += 1
    return molecules, elem_sequences

def flatten_forces(force_str):
    """
    If 'force_str' is space-separated floats, convert to float array.
    Returns shape (n_atoms*3,).
    """
    parts = force_str.strip().split()
    arr = np.array([float(x) for x in parts], dtype=float)
    return arr

def compute_force_coordinate_errors(actual_array, pred_array):
    """
    actual_array, pred_array: shape (n_samples, n_dim) 
    where n_dim = n_atoms * 3
    Returns dict of error metrics for each coordinate dimension.
      e.g. { 'x': (rmse,mae,r2), 'y':(...), 'z':(...), ... for each dimension }
    But we group them in sets of 3 (x,y,z) per atom.

    For convenience, we assume all flattened in the same shape, 
    so dimension k => coordinate dimension across all atoms.
    We'll produce aggregated stats for each dimension index.
    """
    n_samples, n_dim = actual_array.shape
    # We'll break out each dimension in the sense (x of atom0, y of atom0, z of atom0, x of atom1,...)
    # Then produce error metrics for each dimension.

    errors_by_dim = {}
    for d in range(n_dim):
        act = actual_array[:, d]
        prd = pred_array[:, d]
        rmse = np.sqrt(mean_squared_error(act, prd))
        mae  = mean_absolute_error(act, prd)
        r2   = r2_score(act, prd)
        errors_by_dim[d] = (rmse, mae, r2)
    return errors_by_dim

def plot_parity(actual, predicted, label, title, save_path):
    """Simple parity plot for a given 1D coordinate or energy."""
    plt.figure(figsize=(6,6))
    plt.scatter(actual, predicted, s=30, alpha=0.7, edgecolors='k')
    mn = min(np.min(actual), np.min(predicted))
    mx = max(np.max(actual), np.max(predicted))
    plt.plot([mn, mx], [mn, mx], 'r--', label='y = x')
    plt.xlabel(f"Actual {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_box(errors, label, title, save_path):
    """Box plot for 1D array of errors."""
    plt.figure(figsize=(6,4))
    sns.boxplot(y=errors)
    plt.ylabel(f"{label} Error")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze multiple methods at once, per coordinate forces.")
    parser.add_argument("--xyz", type=str, required=True, help="Path to the input smol.xyz file.")
    parser.add_argument("--actual_dir", type=str, required=True, help="Folder with actual data .dat files.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Folder with predicted data (unlabeled).")
    parser.add_argument("--methods", nargs="+", required=True,
                        help="List of method names, e.g. PBE r2SCAN CCSD etc.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save analysis outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    # Read the full XYZ for reference (to get number of atoms, etc.)
    molecules, elem_sequences = read_xyz_molecules(args.xyz)
    n_mol = len(molecules)
    n_atoms = molecules[0].shape[0]  # each molecule => shape (n_atoms,3)
    n_force_dim = n_atoms * 3

    for method in args.methods:
        print(f"\n=== Analyzing method: {method} ===")

        # 1) Load final_unlabeled_data_{method}.csv to get candidate indices
        unlabeled_csv = os.path.join(args.pred_dir, f"final_unlabeled_data_{method}.csv")
        if not os.path.isfile(unlabeled_csv):
            print(f"Skipping method {method}: {unlabeled_csv} not found.", file=sys.stderr)
            continue
        df_unlabeled = pd.read_csv(unlabeled_csv)
        if "Molecule_Index" not in df_unlabeled.columns:
            print(f"Skipping method {method}: 'Molecule_Index' not in columns of {unlabeled_csv}.", file=sys.stderr)
            continue
        candidate_indices = df_unlabeled["Molecule_Index"].values.astype(int)
        n_candidates = len(candidate_indices)

        # 2) Load predicted energies/forces
        energy_file = os.path.join(args.pred_dir, f"predicted_energy_{method}_unlabeled.dat")
        forces_file = os.path.join(args.pred_dir, f"predicted_forces_{method}_unlabeled.dat")
        if not os.path.isfile(energy_file) or not os.path.isfile(forces_file):
            print(f"Skipping {method}: predicted data files not found in {args.pred_dir}.", file=sys.stderr)
            continue
        pred_energy = np.loadtxt(energy_file)
        if pred_energy.ndim == 1:
            pred_energy = pred_energy.reshape(-1, 1)
        pred_forces = np.loadtxt(forces_file)
        if pred_forces.ndim == 1:
            pred_forces = pred_forces.reshape(-1, n_force_dim)
        if pred_energy.shape[0] != n_candidates or pred_forces.shape[0] != n_candidates:
            print(f"Warning: mismatch in number of candidates for {method}.", file=sys.stderr)

        # 3) Load actual data from actual_dir
        actual_energy_dat = os.path.join(args.actual_dir, f"{method}_energy.dat")
        actual_forces_dat = os.path.join(args.actual_dir, f"{method}_force.dat")
        if not os.path.isfile(actual_energy_dat) or not os.path.isfile(actual_forces_dat):
            print(f"Skipping {method}: actual data files not found in {args.actual_dir}.", file=sys.stderr)
            continue
        full_actual_energy = np.loadtxt(actual_energy_dat)
        if full_actual_energy.ndim == 1:
            full_actual_energy = full_actual_energy.reshape(-1, 1)
        full_actual_forces = np.loadtxt(actual_forces_dat)
        if full_actual_forces.ndim == 1:
            full_actual_forces = full_actual_forces.reshape(1, -1)
        
        # 4) Subset the actual data to only the candidate indices
        #    (assuming the data files have a row for each molecule in the same order as smol.xyz).
        actual_energy_cand = full_actual_energy[candidate_indices, :]
        actual_forces_cand = full_actual_forces[candidate_indices, :]

        # 5) Build a merged DataFrame
        # We'll store flattened forces as space-separated columns.
        def arr_to_str(arr):
            return " ".join(f"{x:.8f}" for x in arr)
        candidate_elem_seq = [elem_sequences[idx] for idx in candidate_indices]
        df_merged = pd.DataFrame({
            "Molecule_Index": candidate_indices,
            "Element_Sequence": candidate_elem_seq,
            "Actual_Energy": actual_energy_cand.ravel(),
            "Predicted_Energy": pred_energy.ravel(),
            "Actual_Forces": [arr_to_str(row) for row in actual_forces_cand],
            "Predicted_Forces": [arr_to_str(row) for row in pred_forces]
        })
        merged_csv_path = os.path.join(args.output_dir, f"merged_results_{method}.csv")
        df_merged.to_csv(merged_csv_path, index=False)

        # 6) Compute error metrics for energy
        E_act = df_merged["Actual_Energy"].values
        E_prd = df_merged["Predicted_Energy"].values
        rmseE = np.sqrt(mean_squared_error(E_act, E_prd))
        maeE  = mean_absolute_error(E_act, E_prd)
        r2E   = r2_score(E_act, E_prd)
        print(f"{method} Energy -> RMSE: {rmseE:.6f}, MAE: {maeE:.6f}, R^2: {r2E:.6f}")

        # 7) For forces, parse them back to float arrays, shape (n_candidates, n_dim).
        #    Then compute per-coordinate errors and aggregator.
        act_forces_mat = []
        prd_forces_mat = []
        for i in range(n_candidates):
            act = flatten_forces(df_merged["Actual_Forces"].iloc[i])
            prd = flatten_forces(df_merged["Predicted_Forces"].iloc[i])
            act_forces_mat.append(act)
            prd_forces_mat.append(prd)
        act_forces_mat = np.array(act_forces_mat)  # shape (n_candidates, n_dim)
        prd_forces_mat = np.array(prd_forces_mat)
        n_dim = act_forces_mat.shape[1]

        # We'll do overall aggregator for each dimension, but let's also do a global aggregator
        # for all coordinates combined.
        # coordinate_stats[d] => (rmse, mae, r2) for dimension d
        coordinate_stats = {}
        for d in range(n_dim):
            aD = act_forces_mat[:, d]
            pD = prd_forces_mat[:, d]
            rmse_d = np.sqrt(mean_squared_error(aD, pD))
            mae_d  = mean_absolute_error(aD, pD)
            r2_d   = r2_score(aD, pD)
            coordinate_stats[d] = (rmse_d, mae_d, r2_d)

        # We'll also produce aggregated error across all coordinates:
        # Flatten all coordinates into a single vector.
        all_act = act_forces_mat.flatten()
        all_prd = prd_forces_mat.flatten()
        rmseF = np.sqrt(mean_squared_error(all_act, all_prd))
        maeF  = mean_absolute_error(all_act, all_prd)
        r2F   = r2_score(all_act, all_prd)
        print(f"{method} Forces -> RMSE: {rmseF:.6f}, MAE: {maeF:.6f}, R^2: {r2F:.6f}")

        # 8) Plots
        # 8.1) Parity for energy
        energy_parity_path = os.path.join(args.output_dir, f"parity_energy_{method}.png")
        plot_parity(E_act, E_prd, "Energy", f"{method}: Energy Parity", energy_parity_path)

        # 8.2) Parity for force coordinates
        # We'll group them in sets of (x,y,z) if possible:
        # e.g. dimension d => coordinate d % 3 for atom d//3
        # We'll produce separate parity plots for x, y, z overall aggregator if you'd like.
        # For simplicity, we'll produce 3 if n_dim%3==0, else all dims individually.
        if n_dim % 3 == 0:
            num_atoms = n_dim // 3
            # We'll produce 3 big sets: x coords for all atoms, y coords, z coords
            # x coords: dims = 0,3,6,... y coords: 1,4,7,... z coords: 2,5,8,...
            x_dims = list(range(0, n_dim, 3))
            y_dims = list(range(1, n_dim, 3))
            z_dims = list(range(2, n_dim, 3))
            # Flatten them:
            x_act = act_forces_mat[:, x_dims].flatten()
            x_prd = prd_forces_mat[:, x_dims].flatten()
            y_act = act_forces_mat[:, y_dims].flatten()
            y_prd = prd_forces_mat[:, y_dims].flatten()
            z_act = act_forces_mat[:, z_dims].flatten()
            z_prd = prd_forces_mat[:, z_dims].flatten()
            # Plot each
            plot_parity(x_act, x_prd, "Force-X", f"{method}: Force-X Parity",
                        os.path.join(args.output_dir, f"parity_forces_x_{method}.png"))
            plot_parity(y_act, y_prd, "Force-Y", f"{method}: Force-Y Parity",
                        os.path.join(args.output_dir, f"parity_forces_y_{method}.png"))
            plot_parity(z_act, z_prd, "Force-Z", f"{method}: Force-Z Parity",
                        os.path.join(args.output_dir, f"parity_forces_z_{method}.png"))
        else:
            # We'll produce a parity for each dimension d individually.
            # This might create many plots if you have many atoms. Adjust as needed.
            for d in range(n_dim):
                aD = act_forces_mat[:, d]
                pD = prd_forces_mat[:, d]
                plot_parity(aD, pD, f"Force-dim{d}", f"{method}: Force-dim{d} Parity",
                            os.path.join(args.output_dir, f"parity_forces_dim{d}_{method}.png"))

        # 8.3) Box plots of energy error
        energy_errors = np.abs(E_act - E_prd)
        box_energy_path = os.path.join(args.output_dir, f"box_energy_error_{method}.png")
        plot_box(energy_errors, "Energy", f"{method}: Energy Error Dist.", box_energy_path)

        # 8.4) Box plots of force coordinate errors
        # We'll do a single aggregator for all dims combined as well as x,y,z if n_dim%3==0.
        all_forces_errors = np.abs(act_forces_mat - prd_forces_mat).flatten()
        box_force_all_path = os.path.join(args.output_dir, f"box_forces_error_agg_{method}.png")
        plot_box(all_forces_errors, "Forces(all)", f"{method}: Force All-Coords Error Dist.", box_force_all_path)

        # If n_dim%3==0, also produce separate x,y,z aggregator box plots
        if n_dim % 3 == 0:
            x_err = np.abs(act_forces_mat[:, x_dims] - prd_forces_mat[:, x_dims]).flatten()
            y_err = np.abs(act_forces_mat[:, y_dims] - prd_forces_mat[:, y_dims]).flatten()
            z_err = np.abs(act_forces_mat[:, z_dims] - prd_forces_mat[:, z_dims]).flatten()
            plot_box(x_err, "Force-X", f"{method}: Force-X Error Dist.",
                     os.path.join(args.output_dir, f"box_forces_x_error_{method}.png"))
            plot_box(y_err, "Force-Y", f"{method}: Force-Y Error Dist.",
                     os.path.join(args.output_dir, f"box_forces_y_error_{method}.png"))
            plot_box(z_err, "Force-Z", f"{method}: Force-Z Error Dist.",
                     os.path.join(args.output_dir, f"box_forces_z_error_{method}.png"))

        print(f"Done analyzing {method}. Results saved in {args.output_dir}.")

if __name__ == "__main__":
    main()
