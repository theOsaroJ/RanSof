#!/usr/bin/env python3
"""
analysis_multi_ref_forces.py

Analyzes multiple methods, subtracting both a reference energy and a reference forces array
so that you only see (actual - ref) and (predicted - ref) for energies and forces.

We rename the columns in the final CSV to:
   Actual_Energy_Diff, Predicted_Energy_Diff
   Actual_Forces_Diff, Predicted_Forces_Diff
to emphasize these are differences from the references.

Script steps:
  1) For each method in --methods:
     - read final_unlabeled_data_{method}.csv => candidate indices
     - read predicted_energy_{method}_unlabeled.dat, predicted_forces_{method}_unlabeled.dat
     - read actual energies/forces from actual_dir => {method}_energy.dat, {method}_forces.dat
     - read reference energy => ref_energy_{method}.dat (float)
     - read reference forces => ref_forces_{method}.dat (n_atoms*3 floats in one line)
     - for each candidate => do (actual_energy - ref_energy), (pred_energy - ref_energy),
                              (actual_forces - ref_forces), (pred_forces - ref_forces)
     - store them in CSV as Actual_Energy_Diff, Predicted_Energy_Diff, Actual_Forces_Diff, Predicted_Forces_Diff
     - compute error metrics, plot parity (energy diff & each force dimension), box plots

Usage Example:
  python3 analysis_multi_ref_forces.py \
      --xyz smol.xyz \
      --actual_dir actual_data \
      --pred_dir output \
      --ref_dir reference_energies \
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
    """Reads a multi-molecule XYZ file => returns (molecules, elem_sequences)."""
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
            comment = lines[i].strip() if i < len(lines) else ""
            i += 1
            coords = []
            elems = []
            for _ in range(natom):
                parts = lines[i].split()
                i += 1
                elems.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])
            molecules.append(np.array(coords))
            elem_sequences.append(" ".join(elems))
        else:
            i += 1
    return molecules, elem_sequences

def load_reference_energy(ref_file):
    """Reads ref_energy_{method}.dat => single float."""
    if not os.path.isfile(ref_file):
        raise FileNotFoundError(f"Reference energy file '{ref_file}' not found.")
    with open(ref_file, 'r') as f:
        val_str = f.read().strip()
    return float(val_str)

def load_reference_forces(ref_file, n_dim):
    """
    Reads ref_forces_{method}.dat => single line with n_dim floats (n_atoms*3).
    Returns a 1D array shape (n_dim,).
    """
    if not os.path.isfile(ref_file):
        raise FileNotFoundError(f"Reference forces file '{ref_file}' not found.")
    arr = np.loadtxt(ref_file)
    if arr.ndim == 0:
        # single float => not valid if n_dim>1
        if n_dim > 1:
            raise ValueError(f"Reference forces file only has one float, but we expect {n_dim}.")
        arr = np.array([arr])
    elif arr.ndim == 1:
        # ok if length == n_dim
        if arr.shape[0] != n_dim:
            raise ValueError(f"Reference forces shape mismatch: file has {arr.shape[0]} floats but expect {n_dim}.")
    else:
        raise ValueError(f"Reference forces file is multi-dim => invalid shape {arr.shape}.")
    return arr

def flatten_forces(force_str):
    """Parses a space-separated float string => 1D float array."""
    arr = np.array([float(x) for x in force_str.split()], dtype=float)
    return arr

def plot_parity(actual, predicted, label, title, save_path):
    plt.figure(figsize=(6,6))
    plt.scatter(actual, predicted, s=30, alpha=0.7, edgecolors='k')
    mn = min(np.min(actual), np.min(predicted))
    mx = max(np.max(actual), np.max(predicted))
    plt.plot([mn,mx],[mn,mx],'r--')
    plt.xlabel(f"Actual {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_box(errors, label, title, save_path):
    plt.figure(figsize=(6,4))
    sns.boxplot(y=errors)
    plt.ylabel(f"{label} Error")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze multiple methods with reference energies & forces.")
    parser.add_argument("--xyz", type=str, required=True, help="Path to smol.xyz.")
    parser.add_argument("--actual_dir", type=str, required=True, help="Folder with {method}_energy.dat, {method}_forces.dat")
    parser.add_argument("--pred_dir", type=str, required=True, help="Folder with final_unlabeled_data, predicted energy/forces, etc.")
    parser.add_argument("--ref_dir", type=str, required=True, help="Folder with ref_energy_{method}.dat, ref_forces_{method}.dat")
    parser.add_argument("--methods", nargs="+", required=True, help="List of method names. E.g. PBE, r2SCAN, CCSD, etc.")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save analysis outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    # Read XYZ => get n_atoms
    molecules, elem_sequences = read_xyz_molecules(args.xyz)
    n_mol = len(molecules)
    n_atoms = molecules[0].shape[0]
    n_force_dim = n_atoms*3

    for method in args.methods:
        print(f"\n=== Analyzing method: {method}, subtracting reference energy & forces ===")

        # 1) Load references
        ref_energy_file = os.path.join(args.ref_dir, f"{method}_energy.dat")
        ref_forces_file = os.path.join(args.ref_dir, f"{method}_force.dat")
        try:
            refE = load_reference_energy(ref_energy_file)
        except Exception as e:
            print(f"Skipping {method}: error reading reference energy => {e}", file=sys.stderr)
            continue
        try:
            refF = load_reference_forces(ref_forces_file, n_force_dim)
        except Exception as e:
            print(f"Skipping {method}: error reading reference forces => {e}", file=sys.stderr)
            continue
        print(f"{method} reference energy: {refE:.6f}")
        print(f"{method} reference forces shape: {refF.shape}")

        # 2) Load final_unlabeled_data_{method}.csv => candidate indices
        unlabeled_csv = os.path.join(args.pred_dir, f"final_unlabeled_data_{method}.csv")
        if not os.path.isfile(unlabeled_csv):
            print(f"Skipping {method}: no file => {unlabeled_csv}", file=sys.stderr)
            continue
        df_unlabeled = pd.read_csv(unlabeled_csv)
        if "Molecule_Index" not in df_unlabeled.columns:
            print(f"Skipping {method}: no 'Molecule_Index' in {unlabeled_csv}", file=sys.stderr)
            continue
        candidate_indices = df_unlabeled["Molecule_Index"].values.astype(int)
        n_candidates = len(candidate_indices)

        # 3) Load predicted energy/forces
        pred_energy_file = os.path.join(args.pred_dir, f"predicted_energy_{method}_unlabeled.dat")
        pred_forces_file = os.path.join(args.pred_dir, f"predicted_forces_{method}_unlabeled.dat")
        if (not os.path.isfile(pred_energy_file)) or (not os.path.isfile(pred_forces_file)):
            print(f"Skipping {method}: missing predicted data in {args.pred_dir}", file=sys.stderr)
            continue
        predE = np.loadtxt(pred_energy_file)
        if predE.ndim == 1:
            predE = predE.reshape(-1,1)
        predF = np.loadtxt(pred_forces_file)
        if predF.ndim == 1:
            predF = predF.reshape(-1, n_force_dim)
        if predE.shape[0] != n_candidates or predF.shape[0] != n_candidates:
            print(f"Warning: mismatch #rows for {method}", file=sys.stderr)

        # 4) Load actual data
        actual_energy_path = os.path.join(args.actual_dir, f"{method}_energy.dat")
        actual_forces_path = os.path.join(args.actual_dir, f"{method}_force.dat")
        if (not os.path.isfile(actual_energy_path)) or (not os.path.isfile(actual_forces_path)):
            print(f"Skipping {method}: missing actual data => {actual_energy_path}, {actual_forces_path}", file=sys.stderr)
            continue
        fullE = np.loadtxt(actual_energy_path)
        if fullE.ndim == 1:
            fullE = fullE.reshape(-1,1)
        fullF = np.loadtxt(actual_forces_path)
        if fullF.ndim == 1:
            fullF = fullF.reshape(1, -1)

        # 5) Subset for the candidate indices
        actE_cand = fullE[candidate_indices, :]
        actF_cand = fullF[candidate_indices, :]

        # 6) Subtract references
        # Energy => subtract refE
        actE_cand = actE_cand - refE
        predE = predE - refE
        # Forces => subtract refF
        # We do it for each row => shape (n_candidates, n_force_dim).
        for i in range(n_candidates):
            actF_cand[i, :]  = actF_cand[i, :] - refF
            predF[i, :]      = predF[i, :] - refF

        # 7) Build merged DataFrame
        def arr_to_str(arr):
            return " ".join(f"{x:.8f}" for x in arr)
        candidate_elem_seq = [elem_sequences[idx] for idx in candidate_indices]
        df_merged = pd.DataFrame({
            "Molecule_Index": candidate_indices,
            "Element_Sequence": candidate_elem_seq,
            "Actual_Energy_Diff": actE_cand.ravel(),
            "Predicted_Energy_Diff": predE.ravel(),
            # Store the force diffs as space-separated strings
            "Actual_Forces_Diff": [arr_to_str(actF_cand[i,:]) for i in range(n_candidates)],
            "Predicted_Forces_Diff": [arr_to_str(predF[i,:]) for i in range(n_candidates)]
        })
        out_csv = os.path.join(args.output_dir, f"merged_results_{method}.csv")
        df_merged.to_csv(out_csv, index=False)

        # 8) Compute error metrics for energy (Diff)
        E_act = df_merged["Actual_Energy_Diff"].values
        E_prd = df_merged["Predicted_Energy_Diff"].values
        rmseE = np.sqrt(mean_squared_error(E_act, E_prd))
        maeE  = mean_absolute_error(E_act, E_prd)
        r2E   = r2_score(E_act, E_prd)
        print(f"{method}: Energy(Diff) => RMSE: {rmseE:.6f}, MAE: {maeE:.6f}, R^2: {r2E:.6f}")

        # 9) Force arrays => shape (n_candidates, n_force_dim)
        # parse them back from the DF
        actF_mat = []
        prdF_mat = []
        for i in range(n_candidates):
            aF_str = df_merged["Actual_Forces_Diff"].iloc[i]
            pF_str = df_merged["Predicted_Forces_Diff"].iloc[i]
            aF = np.array([float(x) for x in aF_str.split()], dtype=float)
            pF = np.array([float(x) for x in pF_str.split()], dtype=float)
            actF_mat.append(aF)
            prdF_mat.append(pF)
        actF_mat = np.array(actF_mat)
        prdF_mat = np.array(prdF_mat)
        # aggregator across all coords
        allA = actF_mat.flatten()
        allP = prdF_mat.flatten()
        rmseF = np.sqrt(mean_squared_error(allA, allP))
        maeF  = mean_absolute_error(allA, allP)
        r2F   = r2_score(allA, allP)
        print(f"{method}: Force(Diff) => RMSE: {rmseF:.6f}, MAE: {maeF:.6f}, R^2: {r2F:.6f}")

        # 10) Plots
        # 10.1) energy parity
        en_parity_path = os.path.join(args.output_dir, f"parity_energy_diff_{method}.png")
        plot_parity(E_act, E_prd, "Energy Diff", f"{method}: (energy - ref)", en_parity_path)

        # 10.2) energy box
        E_err = np.abs(E_act - E_prd)
        boxE_path = os.path.join(args.output_dir, f"box_energy_diff_error_{method}.png")
        plot_box(E_err, "EnergyDiff", f"{method} Energy-Ref Error Dist.", boxE_path)

        # 10.3) Force coordinate parity
        n_dim = n_force_dim
        if n_dim % 3 == 0:
            x_dims = list(range(0, n_dim, 3))
            y_dims = list(range(1, n_dim, 3))
            z_dims = list(range(2, n_dim, 3))
            xA = actF_mat[:, x_dims].flatten()
            xP = prdF_mat[:, x_dims].flatten()
            yA = actF_mat[:, y_dims].flatten()
            yP = prdF_mat[:, y_dims].flatten()
            zA = actF_mat[:, z_dims].flatten()
            zP = prdF_mat[:, z_dims].flatten()
            plot_parity(xA, xP, "ForceX(Diff)", f"{method}: ForceX-Ref parity",
                        os.path.join(args.output_dir, f"parity_forces_x_diff_{method}.png"))
            plot_parity(yA, yP, "ForceY(Diff)", f"{method}: ForceY-Ref parity",
                        os.path.join(args.output_dir, f"parity_forces_y_diff_{method}.png"))
            plot_parity(zA, zP, "ForceZ(Diff)", f"{method}: ForceZ-Ref parity",
                        os.path.join(args.output_dir, f"parity_forces_z_diff_{method}.png"))
        else:
            # one parity per dimension
            for d in range(n_dim):
                aD = actF_mat[:,d]
                pD = prdF_mat[:,d]
                plot_parity(aD, pD, f"Force-dim{d}(Diff)", f"{method} Force-dim{d} minus ref",
                            os.path.join(args.output_dir, f"parity_forces_dim{d}_diff_{method}.png"))

        # 10.4) Force box
        all_f_err = np.abs(actF_mat - prdF_mat).flatten()
        boxF_path = os.path.join(args.output_dir, f"box_forces_diff_error_all_{method}.png")
        plot_box(all_f_err, "Forces(All coords,Diff)", f"{method} Force-Ref All-Coord Error Dist.", boxF_path)
        if n_dim % 3 == 0:
            x_err = np.abs(actF_mat[:, x_dims] - prdF_mat[:, x_dims]).flatten()
            y_err = np.abs(actF_mat[:, y_dims] - prdF_mat[:, y_dims]).flatten()
            z_err = np.abs(actF_mat[:, z_dims] - prdF_mat[:, z_dims]).flatten()
            plot_box(x_err, "ForceX(Diff)", f"{method} ForceX-Ref Error Dist.",
                     os.path.join(args.output_dir, f"box_forces_x_diff_{method}.png"))
            plot_box(y_err, "ForceY(Diff)", f"{method} ForceY-Ref Error Dist.",
                     os.path.join(args.output_dir, f"box_forces_y_diff_{method}.png"))
            plot_box(z_err, "ForceZ(Diff)", f"{method} ForceZ-Ref Error Dist.",
                     os.path.join(args.output_dir, f"box_forces_z_diff_{method}.png"))

        print(f"Done with {method} => results in {args.output_dir}")

if __name__ == "__main__":
    main()
