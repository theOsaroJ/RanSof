#!/usr/bin/env python3
"""
Utility functions for I/O operations.
"""

import numpy as np

def write_xyz(molecule_list, molecule_ids, element_sequences, filename):
    """
    Writes a list of molecules to an XYZ file.
    Each molecule is a list of [x, y, z] coordinates.
    """
    with open(filename, "w") as f:
        for i, mol in enumerate(molecule_list):
            num_atoms = len(mol)
            f.write(f"{num_atoms}\n")
            f.write(f"Molecule {molecule_ids[i]}\n")
            elems = element_sequences[i].split()
            for j in range(num_atoms):
                x, y, z = mol[j]
                f.write(f"{elems[j]} {x:.8f} {y:.8f} {z:.8f}\n")

def read_xyz_multimol(file_path):
    """
    Reads a multi-molecule XYZ file.
    Returns:
      - molecules: list of lists; each inner list contains [x, y, z] for each atom.
      - elem_sequences: list of strings (e.g., "C H H H").
    """
    molecules = []
    elem_sequences = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            natom = int(line)
            i += 1  # skip comment line
            i += 1
            mol = []
            elems = []
            for _ in range(natom):
                parts = lines[i].split()
                i += 1
                sym = parts[0]
                x, y, z = map(float, parts[1:4])
                mol.append([x, y, z])
                elems.append(sym)
            molecules.append(mol)
            elem_sequences.append(" ".join(elems))
        else:
            i += 1
    return molecules, elem_sequences

def load_forces(file_path, expected_rows, expected_cols):
    """
    Loads a forces file with one nonempty line per molecule.
    Each line is a flattened force vector.
    Checks that the number of rows and columns match expectations.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append([float(x) for x in line.split()])
    data = np.array(data)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {data.shape[0]} from {file_path}")
    if data.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, got {data.shape[1]} from {file_path}")
    return data

def load_latest_value(file_path, expected_shape):
    """
    Loads the simulation output from file_path and returns the last available row
    reshaped into expected_shape. Raises an exception if reshaping is not possible.
    """
    data = np.loadtxt(file_path)
    if np.isscalar(data) or (hasattr(data, "ndim") and data.ndim == 0):
        data = np.array([data])
    if data.ndim == 1:
        data = np.array([data[-1]])
    else:
        data = data[-1:]
    try:
        data = data.reshape(expected_shape)
    except Exception as e:
        raise ValueError(f"Reshape error for file {file_path}: {e}")
    return data
