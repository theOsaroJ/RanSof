#!/usr/bin/env python3
"""
Utility functions for I/O operations.
"""

def write_xyz(x_array, molecule_ids, element_sequences, filename):
    """
    Write an XYZ file from a set of flattened coordinate arrays.
    Each row in x_array is assumed to contain coordinates for one molecule.
    """
    with open(filename, "w") as f:
        for i, coords in enumerate(x_array):
            num_atoms = len(coords) // 3
            f.write(f"{num_atoms}\n")
            f.write(f"Molecule {molecule_ids[i]}\n")
            elems = element_sequences[i].split()
            for j in range(num_atoms):
                x, y, z = coords[3*j:3*j+3]
                f.write(f"{elems[j]} {x:.8f} {y:.8f} {z:.8f}\n")

def read_xyz_multimol(file_path):
    """
    Reads a multi-molecule XYZ file and returns:
      (molecules, element_sequences)
    Each molecule is a list of [x, y, z] coordinates, and element_sequences is a list
    of space-separated element symbols.
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
