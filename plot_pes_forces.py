#!/usr/bin/env python3
"""
plot_pes_forces_merged_toggle_dims.py

This script processes merged results CSV files for one or more methods, adds a computed bond length
(from an XYZ file) to each row, optionally subtracts reference energy/force values, and produces a
two-panel line plot versus bond length:
  - Top subplot: Actual vs. Predicted energies.
  - Bottom subplot: Actual vs. Predicted forces for selected force dimensions.

Expected input:
  - Merged CSV files in a directory (--merged_dir) named:
        merged_results_{method}.csv
    Each CSV must include at least:
        Molecule_Index (0-based by default)
        Actual_Energy and Predicted_Energy    (or raw energies)
        Actual_Forces and Predicted_Forces    (stored as space-separated strings)
  - An XYZ file (--xyz) with all molecule geometries.
  - If --subtract_ref is "true", reference files (one per method) exist in --ref_dir:
        ref_energy_{method}.dat  and ref_forces_{method}.dat
  - The user can choose which force dimensions to plot using --force_dims.
  
Usage example:
  python3 plot_pes_forces_merged_toggle_dims.py \
      --xyz smol.xyz \
      --merged_dir analysis_output \
      --output_dir plots \
      --subtract_ref true \
      --ref_dir Gstate \
      --methods PBE r2SCAN CCSD \
      --atom1 0 --atom2 1 \
      --force_dims all
      # (Do not include --one_based if Molecule_Index is 0-based.)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# --------------------- Utility Functions ---------------------

def read_xyz_molecules(xyz_file):
    """
    Reads a multi-molecule XYZ file.
    Returns:
       molecules: a list of np.array (each of shape (n_atoms, 3))
    """
    if not os.path.isfile(xyz_file):
        raise FileNotFoundError(f"XYZ file '{xyz_file}' not found.")
    molecules = []
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            natom = int(line)
            i += 1  # skip comment line
            i += 1  # skip comment line
            coords = []
            for _ in range(natom):
                parts = lines[i].split()
                i += 1
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                coords.append([x, y, z])
            molecules.append(np.array(coords))
        else:
            i += 1
    return molecules

def compute_bond_length(mol_coords, a1=0, a2=1):
    """
    Computes the Euclidean distance between atoms a1 and a2 of a molecule.
    Modify this function if you wish to use another definition (e.g., overall norm).
    """
    diff = mol_coords[a1, :] - mol_coords[a2, :]
    return np.linalg.norm(diff)

def parse_forces(force_str):
    """Converts a space-separated string to a numpy array of floats."""
    return np.array([float(x) for x in force_str.strip().split()], dtype=float)

def load_ref_energy(ref_file):
    """Loads a single float from ref_file."""
    with open(ref_file, 'r') as f:
        val = f.read().strip()
    return float(val)

def load_ref_forces(ref_file, n_dim):
    """
    Loads a line of space-separated floats from ref_file and verifies it has length n_dim.
    """
    arr = np.loadtxt(ref_file)
    if arr.ndim == 0:
        if n_dim > 1:
            raise ValueError(f"Reference forces file {ref_file} has only 1 float but expected {n_dim}.")
        arr = np.array([arr])
    elif arr.ndim == 1:
        if arr.shape[0] != n_dim:
            raise ValueError(f"Reference forces file {ref_file} has {arr.shape[0]} values but expected {n_dim}.")
    else:
        raise ValueError(f"Reference forces file {ref_file} has invalid shape {arr.shape}.")
    return arr

def get_force_dims(force_dims_str, n_dim):
    """
    Parses the --force_dims argument.
    If "all", returns list(range(n_dim)).
    Otherwise, returns a list of integer indices.
    """
    if force_dims_str.lower() == "all":
        return list(range(n_dim))
    try:
        dims = [int(x.strip()) for x in force_dims_str.split(",")]
        for d in dims:
            if d < 0 or d >= n_dim:
                raise ValueError(f"Force dimension {d} not valid for n_dim = {n_dim}")
        return dims
    except Exception as e:
        raise ValueError(f"Error parsing --force_dims: {e}")

def plot_pes_and_forces(bondX, Eact, Eprd, forces_act, forces_prd, method, use_ref, dims_to_plot, out_path):
    """
    Creates a figure with two subplots:
      - Subplot 1: Potential Energy (actual vs. predicted) vs. bond length.
      - Subplot 2: For each selected force dimension, plots actual and predicted forces vs. bond length.
    Uses fixed colors for energy (green for actual, orange for predicted) and two colormaps for forces.
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,10), sharex=True)
    
    # Energy plot.
    axes[0].plot(bondX, Eact, marker="o", linestyle="-", color="green", label="Actual Energy")
    axes[0].plot(bondX, Eprd, marker="+", linestyle="--", color="orange", label="Predicted Energy")
    axes[0].set_ylabel("Energy " + ("(Diff)" if use_ref else ""))
    axes[0].set_title(f"{method}: Potential Energy vs Bond Length")
    axes[0].legend()
    
    # Forces plot.
    n_plot = len(dims_to_plot)
    # Use separate colormaps for actual vs. predicted forces.
    colors_actual = [plt.cm.viridis(i/n_plot) for i in range(n_plot)]
    colors_pred = [plt.cm.plasma(i/n_plot) for i in range(n_plot)]
    for i, d in enumerate(dims_to_plot):
        axes[1].plot(bondX, forces_act[:, d],
                     marker="o", linestyle="-", color=colors_actual[i],
                     label=f"Actual Force dim {d}")
        axes[1].plot(bondX, forces_prd[:, d],
                     marker="+", linestyle="--", color=colors_pred[i],
                     label=f"Predicted Force dim {d}")
    axes[1].set_ylabel("Forces " + ("(Diff)" if use_ref else ""))
    axes[1].set_xlabel("Bond Length (units)")
    axes[1].set_title("Force Components vs Bond Length")
    axes[1].legend()
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[{method}] Saved plot: {out_path}")

# --------------------- Main Processing Function ---------------------

def process_method(method, merged_dir, output_dir, xyz_file, use_ref, ref_dir,
                   actE_col, prdE_col, actF_col, prdF_col, one_based, atom1, atom2, force_dims_str, bond_arr):
    """
    Processes one method:
      1. Loads merged_results_{method}.csv from merged_dir.
      2. Optionally subtracts reference values if use_ref is True.
      3. Adds a BondLength column mapping Molecule_Index to bond lengths from bond_arr.
      4. Saves a final CSV as final_merged_{method}.csv.
      5. Plots two subplots (energy and forces vs. bond length) and saves the plot.
    """
    csv_file = os.path.join(merged_dir, f"merged_results_{method}.csv")
    if not os.path.isfile(csv_file):
        print(f"Method {method}: Skipping; {csv_file} not found.", file=sys.stderr)
        return
    df = pd.read_csv(csv_file)
    if "Molecule_Index" not in df.columns:
        print(f"Method {method}: Skipping; no 'Molecule_Index' column.", file=sys.stderr)
        return

    # If subtracting references, work on raw columns.
    if use_ref:
        if actE_col in df.columns and prdE_col in df.columns and actF_col in df.columns and prdF_col in df.columns:
            row0 = df[actF_col].iloc[0]
            n_dim = len(parse_forces(row0))
            refE_file = os.path.join(ref_dir, f"{method}_energy.dat")
            refF_file = os.path.join(ref_dir, f"{method}_force.dat")
            if not os.path.isfile(refE_file) or not os.path.isfile(refF_file):
                print(f"Method {method}: Missing reference files {refE_file} or {refF_file}. Skipping.", file=sys.stderr)
                return
            try:
                refE = load_ref_energy(refE_file)
                refF = load_ref_forces(refF_file, n_dim)
            except Exception as e:
                print(f"Method {method}: Error reading references: {e}", file=sys.stderr)
                return
            df["Actual_Energy_Diff"] = df[actE_col] - refE
            df["Predicted_Energy_Diff"] = df[prdE_col] - refE

            def subtract_force(row):
                arr = parse_forces(row)
                return " ".join(f"{x:.8f}" for x in (arr - refF))
            df["Actual_Forces_Diff"] = df[actF_col].apply(subtract_force)
            df["Predicted_Forces_Diff"] = df[prdF_col].apply(subtract_force)
            actE_name = "Actual_Energy_Diff"
            prdE_name = "Predicted_Energy_Diff"
            actF_name = "Actual_Forces_Diff"
            prdF_name = "Predicted_Forces_Diff"
        else:
            print(f"Method {method}: Missing raw columns for reference subtraction.", file=sys.stderr)
            return
    else:
        for col in [actE_col, prdE_col, actF_col, prdF_col]:
            if col not in df.columns:
                print(f"Method {method}: Column '{col}' not found.", file=sys.stderr)
                return
        actE_name = actE_col
        prdE_name = prdE_col
        actF_name = actF_col
        prdF_name = prdF_col

    # Map Molecule_Index to BondLength using bond_arr.
    def bond_lookup(idx):
        return bond_arr[idx-1] if one_based else bond_arr[idx]
    df["BondLength"] = df["Molecule_Index"].apply(bond_lookup)
    df = df.sort_values("BondLength").reset_index(drop=True)

    # Save final merged CSV.
    final_csv = os.path.join(output_dir, f"final_merged_{method}.csv")
    df.to_csv(final_csv, index=False)
    print(f"[{method}] Saved final merged CSV: {final_csv}")

    # Prepare arrays for plotting.
    bondX = df["BondLength"].values
    Eact = df[actE_name].values
    Eprd = df[prdE_name].values

    n_data = df.shape[0]
    sample_forces = parse_forces(df[actF_name].iloc[0])
    n_dim = len(sample_forces)
    forces_act = np.array([parse_forces(df[actF_name].iloc[i]) for i in range(n_data)])
    forces_prd = np.array([parse_forces(df[prdF_name].iloc[i]) for i in range(n_data)])

    dims_to_plot = get_force_dims(force_dims_str, n_dim)

    # Plotting: Use fixed distinct colors for energy and two distinct colormaps for forces.
    plot_file = os.path.join(output_dir, f"bond_curve_{method}.png")
    plot_pes_and_forces(bondX, Eact, Eprd, forces_act, forces_prd, method, use_ref, dims_to_plot, plot_file)

# --------------------- Main ---------------------

def main():
    parser = argparse.ArgumentParser(description="Process merged results and plot PES & forces vs. bond length for multiple methods.")
    parser.add_argument("--xyz", required=True, help="Path to multi-molecule XYZ file (e.g., smol.xyz).")
    parser.add_argument("--merged_dir", required=True, help="Directory containing merged_results_{method}.csv files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save final merged CSVs and plots (auto-named by method).")
    parser.add_argument("--subtract_ref", default="false", help="'true' to subtract reference values; 'false' to skip. (Default false)")
    parser.add_argument("--ref_dir", default="", help="Directory with ref_energy_{method}.dat and ref_forces_{method}.dat (if subtract_ref true).")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="List of method names (e.g., PBE, r2SCAN, CCSD). If omitted, all CSVs matching merged_results_*.csv in merged_dir are processed.")
    parser.add_argument("--atom1", type=int, default=0, help="Atom index for bond calculation (default 0).")
    parser.add_argument("--atom2", type=int, default=1, help="Atom index for bond calculation (default 1).")
    parser.add_argument("--one_based", action="store_true", help="If set, Molecule_Index in CSV is 1-based; otherwise 0-based (default).")
    parser.add_argument("--actE_col", default="Actual_Energy", help="Column name for actual energy.")
    parser.add_argument("--prdE_col", default="Predicted_Energy", help="Column name for predicted energy.")
    parser.add_argument("--actF_col", default="Actual_Forces", help="Column name for actual forces.")
    parser.add_argument("--prdF_col", default="Predicted_Forces", help="Column name for predicted forces.")
    parser.add_argument("--force_dims", default="all", help="Force dimensions to plot, either 'all' or a comma-separated list (e.g. '0,3').")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine the list of methods.
    if args.methods and len(args.methods) > 0:
        methods = args.methods
    else:
        pattern = os.path.join(args.merged_dir, "merged_results_*.csv")
        files = glob(pattern)
        methods = []
        for f in files:
            base = os.path.basename(f)
            if base.startswith("merged_results_") and base.endswith(".csv"):
                m = base[len("merged_results_"):-len(".csv")]
                methods.append(m)
        if not methods:
            print("No merged_results_*.csv files found in merged_dir.", file=sys.stderr)
            sys.exit(1)

    # Read the XYZ file once and compute bond lengths for all molecules.
    molecules = read_xyz_molecules(args.xyz)
    n_mol = len(molecules)
    bond_arr = [compute_bond_length(mol, a1=args.atom1, a2=args.atom2) for mol in molecules]
    bond_arr = np.array(bond_arr)

    # Process each method.
    for m in methods:
        try:
            process_method(m,
                           merged_dir=args.merged_dir,
                           output_dir=args.output_dir,
                           xyz_file=args.xyz,
                           use_ref=(args.subtract_ref.lower()=="true"),
                           ref_dir=args.ref_dir,
                           actE_col=args.actE_col,
                           prdE_col=args.prdE_col,
                           actF_col=args.actF_col,
                           prdF_col=args.prdF_col,
                           one_based=args.one_based,
                           atom1=args.atom1,
                           atom2=args.atom2,
                           force_dims_str=args.force_dims,
                           bond_arr=bond_arr)
        except Exception as e:
            print(f"Error processing method {m}: {e}", file=sys.stderr)

if __name__=="__main__":
    main()
