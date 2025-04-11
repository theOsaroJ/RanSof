#!/usr/bin/env python3
"""
plot_pes_forces_merged.py

This script processes merged results for one or more methods. It assumes that in a given 
merged results directory (--merged_dir), you have CSV files named like:

    merged_results_{method}.csv

Each such CSV must contain at least:
  - Molecule_Index         : the index of the molecule (by default 0-based; use --one_based if 1-based)
  - Actual_Energy          : the raw (or already diff) actual energy
  - Predicted_Energy       : the raw (or diff) predicted energy
  - Actual_Forces          : the raw (or diff) actual forces, stored as a space‑separated string (all coordinates)
  - Predicted_Forces       : the corresponding predicted forces

Optionally, if --subtract_ref is true, the script will read reference files from --ref_dir:
  ref_energy_{method}.dat  (one float per method) and
  ref_forces_{method}.dat  (one line with (n_atoms*3) floats)
and subtract these from the raw values to create new columns with suffix “_Diff.”

The script also reads an XYZ file (--xyz) and computes a bond length for each molecule.
Here the bond length is computed as the Euclidean distance between atom indices specified 
by --atom1 and --atom2 (default 0 and 1). (You may change this function if needed.)

It then adds a “BondLength” column (by matching the Molecule_Index from the CSV with the computed 
bond length from the XYZ) and sorts the rows by BondLength. Finally, it writes out a new CSV and creates 
a plot (two subplots: (i) potential energy [actual vs. predicted] vs. BondLength and (ii) force curves
(vs. BondLength) for each coordinate dimension).

Usage example:
  python3 plot_pes_forces_merged.py \
      --xyz smol.xyz \
      --merged_dir analysis_output \
      --output_dir plots \
      --subtract_ref true \
      --ref_dir refs \
      --methods PBE r2SCAN CCSD \
      --atom1 0 --atom2 1 --one_based

If --one_based is provided the script assumes the CSV Molecule_Index is 1-based; otherwise, it’s 0-based.
If --subtract_ref is "true", the script reads the reference files and subtracts them.
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
      molecules: a list of np.array (shape: (n_atoms, 3)) for each molecule.
      (Element sequences are not used here.)
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
    Computes the Euclidean distance between atom indices a1 and a2 in a molecule.
    If you wish to use a different definition (e.g. overall norm), modify this function.
    """
    diff = mol_coords[a1, :] - mol_coords[a2, :]
    return np.linalg.norm(diff)

def parse_forces(force_str):
    """Parses a space-separated string of numbers into a numpy 1D float array."""
    return np.array([float(x) for x in force_str.strip().split()], dtype=float)

def load_ref_energy(ref_file):
    """Loads and returns a single float from ref_file."""
    with open(ref_file, 'r') as f:
        val = f.read().strip()
    return float(val)

def load_ref_forces(ref_file, n_dim):
    """
    Loads a single line of space-separated floats from ref_file and ensures it has the required dimension.
    """
    arr = np.loadtxt(ref_file)
    if arr.ndim == 0:
        if n_dim > 1:
            raise ValueError(f"Reference forces file {ref_file} has only 1 float but expected {n_dim}.")
        arr = np.array([arr])
    elif arr.ndim == 1:
        if arr.shape[0] != n_dim:
            raise ValueError(f"Reference forces file {ref_file} has length {arr.shape[0]}, expected {n_dim}.")
    else:
        raise ValueError(f"Reference forces file {ref_file} has invalid shape {arr.shape}.")
    return arr

def plot_pes_and_forces(bondX, Eact, Eprd, forces_act, forces_prd, method, use_ref, output_path):
    """
    Plots two subplots:
      Subplot 1: Energy (Actual vs. Predicted) vs. Bond Length.
      Subplot 2: For each force dimension, plots Actual and Predicted forces vs. Bond Length.
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,10), sharex=True)
    # Energy plot
    axes[0].plot(bondX, Eact, "o-", label="Actual Energy")
    axes[0].plot(bondX, Eprd, "s--", label="Predicted Energy")
    axes[0].set_ylabel("Energy " + ("(Diff)" if use_ref else ""))
    axes[0].set_title(f"{method}: Potential Energy vs. Bond Length")
    axes[0].legend()

    # Forces plot
    n_dim = forces_act.shape[1]
    color_cycle = plt.cm.tab20(np.linspace(0,1,n_dim))
    for d in range(n_dim):
        axes[1].plot(bondX, forces_act[:, d],
                     marker='o', linestyle='-',
                     color=color_cycle[d],
                     alpha=0.7,
                     label=f"Actual F-d{d}" if d < 3 else None)
        axes[1].plot(bondX, forces_prd[:, d],
                     marker='s', linestyle='--',
                     color=color_cycle[d],
                     alpha=0.7,
                     label=f"Predicted F-d{d}" if d < 3 else None)
    axes[1].set_ylabel("Forces " + ("(Diff)" if use_ref else ""))
    axes[1].set_xlabel("Bond Length (units?)")
    axes[1].set_title("Forces vs. Bond Length")
    if n_dim < 6:
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

# --------------------- Processing Function per Method ---------------------

def process_method(method, merged_dir, output_dir, xyz_file, use_ref, ref_dir,
                   actE_col, prdE_col, actF_col, prdF_col, one_based, atom1, atom2, bond_arr):
    """
    Processes one method. It:
      1. Loads merged_results_{method}.csv.
      2. Optionally subtracts reference values if use_ref == True.
      3. Adds a 'BondLength' column based on Molecule_Index and the provided bond_arr.
      4. Sorts by bond length and writes a final CSV (named automatically by the method).
      5. Generates a 2-panel line plot of energy and forces versus bond length.
    """
    csv_file = os.path.join(merged_dir, f"merged_results_{method}.csv")
    if not os.path.isfile(csv_file):
        print(f"Method {method}: Skipping, file {csv_file} not found.", file=sys.stderr)
        return
    df = pd.read_csv(csv_file)
    if "Molecule_Index" not in df.columns:
        print(f"Method {method}: Skipping, 'Molecule_Index' column missing.", file=sys.stderr)
        return

    # If subtracting references, work on raw columns to create new _Diff columns.
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
            print(f"Method {method}: Missing raw columns for energy/forces. Skipping reference subtraction.", file=sys.stderr)
            return
    else:
        # Use the provided column names.
        for col in [actE_col, prdE_col, actF_col, prdF_col]:
            if col not in df.columns:
                print(f"Method {method}: Column {col} not found in CSV. Skipping.", file=sys.stderr)
                return
        actE_name = actE_col
        prdE_name = prdE_col
        actF_name = actF_col
        prdF_name = prdF_col

    # Add BondLength column. Assume Molecule_Index is 0-based or 1-based as indicated.
    def bond_lookup(idx):
        return bond_arr[idx-1] if one_based else bond_arr[idx]
    df["BondLength"] = df["Molecule_Index"].apply(bond_lookup)
    df = df.sort_values("BondLength").reset_index(drop=True)

    # Save final CSV automatically named by method.
    final_csv = os.path.join(output_dir, f"final_merged_{method}.csv")
    df.to_csv(final_csv, index=False)
    print(f"[{method}] Saved final merged CSV: {final_csv}")

    # Prepare data arrays for plotting.
    bondX = df["BondLength"].values
    Eact = df[actE_name].values
    Eprd = df[prdE_name].values
    # For forces, we assume forces are stored as space-separated strings.
    n_data = df.shape[0]
    # Parse the forces for the first row to determine dimension.
    arr0 = parse_forces(df[actF_name].iloc[0])
    n_dim = len(arr0)
    forces_act = np.array([parse_forces(df[actF_name].iloc[i]) for i in range(n_data)])
    forces_prd = np.array([parse_forces(df[prdF_name].iloc[i]) for i in range(n_data)])

    # Plot
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,10), sharex=True)
    # Energy subplot
    axes[0].plot(bondX, Eact, "o-", label="Actual Energy")
    axes[0].plot(bondX, Eprd, "s--", label="Predicted Energy")
    ylabel_energy = "Energy - ref" if use_ref else "Energy"
    axes[0].set_ylabel(ylabel_energy)
    axes[0].set_title(f"{method}: Potential Energy vs. Bond Length")
    axes[0].legend()
    # Forces subplot
    import matplotlib.cm as cm
    color_cycle = cm.tab20(np.linspace(0,1,n_dim))
    for d in range(n_dim):
        axes[1].plot(bondX, forces_act[:, d],
                     marker='o', linestyle='-', color=color_cycle[d],
                     alpha=0.7, label=(f"Actual F-d{d}" if d<3 else None))
        axes[1].plot(bondX, forces_prd[:, d],
                     marker='s', linestyle='--', color=color_cycle[d],
                     alpha=0.7, label=(f"Predicted F-d{d}" if d<3 else None))
    ylabel_force = "Forces - ref" if use_ref else "Forces"
    axes[1].set_ylabel(ylabel_force)
    axes[1].set_xlabel("Bond Length (units)")
    axes[1].set_title("Forces vs. Bond Length")
    if n_dim < 6:
        axes[1].legend()
    fig.tight_layout()

    # Auto-generate plot file name by method.
    plot_file = os.path.join(output_dir, f"bond_curve_{method}.png")
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)
    print(f"[{method}] Saved plot: {plot_file}")

# --------------------- Main ---------------------

def main():
    parser = argparse.ArgumentParser(description="Process merged results and plot PES & forces vs. bond length for multiple methods.")
    parser.add_argument("--xyz", required=True, help="Path to multi-molecule XYZ file (e.g., smol.xyz).")
    parser.add_argument("--merged_dir", required=True, help="Directory containing merged_results_{method}.csv files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save final merged CSVs and plots (names auto-generated).")
    parser.add_argument("--subtract_ref", default="false",
                        help="'true' to subtract reference energy/forces; 'false' to skip. (Default false)")
    parser.add_argument("--ref_dir", default="",
                        help="Directory with ref_energy_{method}.dat and ref_forces_{method}.dat (if subtract_ref true).")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="List of method names (e.g., PBE, r2SCAN, CCSD). If omitted, all CSVs matching merged_results_*.csv in merged_dir are processed.")
    parser.add_argument("--atom1", type=int, default=0, help="Atom index for bond calculation (default 0).")
    parser.add_argument("--atom2", type=int, default=1, help="Atom index for bond calculation (default 1).")
    parser.add_argument("--one_based", action="store_true",
                        help="If set, Molecule_Index in CSV is 1-based; otherwise, it is 0-based.")
    parser.add_argument("--actE_col", default="Actual_Energy", help="Column name for actual energy.")
    parser.add_argument("--prdE_col", default="Predicted_Energy", help="Column name for predicted energy.")
    parser.add_argument("--actF_col", default="Actual_Forces", help="Column name for actual forces.")
    parser.add_argument("--prdF_col", default="Predicted_Forces", help="Column name for predicted forces.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine methods to process: if not explicitly provided, scan merged_dir.
    if args.methods and len(args.methods) > 0:
        methods = args.methods
    else:
        from glob import glob
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

    # First, read the XYZ file once and compute the bond lengths for all molecules.
    molecules = read_xyz_molecules(args.xyz)
    n_mol = len(molecules)
    bond_arr = []
    for i, coords in enumerate(molecules):
        b_len = compute_bond_length(coords, a1=args.atom1, a2=args.atom2)
        bond_arr.append(b_len)
    bond_arr = np.array(bond_arr)

    # Process each method:
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
                           bond_arr=bond_arr)
        except Exception as e:
            print(f"Error processing method {m}: {e}", file=sys.stderr)

if __name__=="__main__":
    main()
