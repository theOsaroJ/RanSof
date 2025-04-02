#!/usr/bin/env python3

import argparse
import os
from functools import partial
from time import perf_counter
from multiprocessing import Pool
import numpy as np

from pyscf import gto, scf, cc, grad
from pyscf.scf import addons

# 1 Bohr = 0.52917721092 Å
BOHR2ANG = 0.52917721092
ANG2BOHR = 1.0 / BOHR2ANG
Ha_to_eV = 27.2114

def package_data(atom_charges, atom_coords):
    """
    Packages each molecule's atomic charges and coordinates into
    a list of (Z, (x, y, z)) in Angstrom.
    """
    all_mols = []
    for i in range(atom_charges.shape[0]):
        Z_list = atom_charges[i]
        R_list = atom_coords[i]
        # Build (Z, coords) for each atom
        mol_data = list(zip(Z_list, R_list))
        all_mols.append(mol_data)
    return all_mols

def run_ccsd(atom, args):
    """
    Runs SCF + CCSD on a single molecule described by
    atom = [(Z, (x, y, z in Ang)), ...].
    Returns a CCSD object (mycc) on success or None if fails.
    """
    try:
        mol = gto.Mole()
        mol.unit = 'Angstrom'  # We interpret coords as Angstrom
        mol.atom = []
        for (Z, xyz) in atom:
            mol.atom.append((int(Z), tuple(xyz)))
        mol.basis = args.basisset
        mol.verbose = args.verbose
        mol.build()

        mf = scf.RHF(mol)
        if not mf.kernel():
            print("SCF failed; trying SOSCF fallback...", flush=True)
            mf = addons.convert_to_soscf(mf)
            if not mf.kernel():
                print("SOSCF also failed; trying smaller DIIS space...", flush=True)
                mf.diis_space = 5
                mf.max_cycle = 100
                if not mf.kernel():
                    print("SCF attempts all failed.", flush=True)
                    return None

        mycc = cc.CCSD(mf)
        if not mycc.kernel():
            print("CCSD not converged; adjusting conv_tol...", flush=True)
            mycc.conv_tol = 1e-6
            mycc.max_cycle = 100
            if not mycc.kernel():
                print("CCSD attempts all failed for this molecule.", flush=True)
                return None

        return mycc

    except Exception as e:
        print(f"Error in run_ccsd: {e}", flush=True)
        return None

def analytic_ccsd_forces(mycc):
    """
    Analytic CCSD gradient => shape (n_atoms, 3) in atomic units (Hartree/Bohr).
    Convert to eV/Å before returning.
    """
    try:
        cc_grad = grad.ccsd.Gradients(mycc)
        # gradient: dE/dR => shape (n_atoms, 3)
        # force = -gradient
        grad_au = cc_grad.kernel()  # Hartree/Bohr
        force_au = -grad_au         # negative gradient => force
        # Convert to eV/Å:
        force_eV_A = force_au * (Ha_to_eV / BOHR2ANG)
        return force_eV_A
    except Exception as e:
        print(f"Error in analytic_ccsd_forces: {e}")
        return None

def ccsd_energy(atom_data, args):
    """
    Build a new Mole object (unit=Angstrom), run CCSD, return total energy in Hartree.
    Used by the FD approach for +/- displacements.
    """
    try:
        mol = gto.Mole()
        mol.unit = 'Angstrom'
        mol.atom = []
        for (Z, xyz) in atom_data:
            mol.atom.append((int(Z), tuple(xyz)))
        mol.basis = args.basisset
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        if not mf.kernel():
            mf = addons.convert_to_soscf(mf)
            mf.kernel()

        mycc = cc.CCSD(mf)
        mycc.kernel()
        return mycc.e_tot  # in Hartree
    except Exception as e:
        print(f"Error in ccsd_energy: {e}")
        return None

def finite_difference_ccsd_forces(base_mycc, args, delta=0.01):
    """
    Finite difference forces from CCSD energies:
      F_i = - (E+ - E-) / (2*delta).
    'delta' is in Angstrom. Coordinates from base_mycc are in Bohr,
    so we convert to Angstrom. Returns an (n_atoms, 3) array in eV/Å.
    """
    try:
        mol = base_mycc._scf.mol
        coords_bohr = mol.atom_coords()  # shape (n_atoms, 3) in Bohr
        n_atoms = coords_bohr.shape[0]
        coords_ang = coords_bohr * BOHR2ANG  # convert to Angstrom

        # We also need the atomic numbers
        atom_data = []
        for i in range(n_atoms):
            Znum = mol.atom_charge(i)  # integer atomic number
            atom_data.append((Znum, coords_ang[i]))

        forces = np.zeros((n_atoms, 3), dtype=float)

        for i_atom in range(n_atoms):
            for dim in range(3):
                # plus displacement
                plus_data = []
                for idx, (Z, xyz) in enumerate(atom_data):
                    px = np.array(xyz)
                    if idx == i_atom:
                        px[dim] += delta
                    plus_data.append((Z, px))

                # minus displacement
                minus_data = []
                for idx, (Z, xyz) in enumerate(atom_data):
                    mx = np.array(xyz)
                    if idx == i_atom:
                        mx[dim] -= delta
                    minus_data.append((Z, mx))

                Eplus = ccsd_energy(plus_data, args)
                Eminus = ccsd_energy(minus_data, args)
                if Eplus is None or Eminus is None:
                    print("Finite difference: SCF/CCSD failed for +/- displacement.")
                    return None

                dE_dX_Ha_A = (Eplus - Eminus) / (2.0 * delta)
                # Force = negative derivative => F = -dE/dx
                # Convert to eV/Å
                F_eV_A = -dE_dX_Ha_A * Ha_to_eV
                forces[i_atom, dim] = F_eV_A

        return forces
    except Exception as e:
        print(f"Error in FD forces: {e}")
        return None

def save_data(mycc, mol_index, args, delta=0.01):
    """
    Saves geometry, total energy, analytic CCSD forces, and optional FD CCSD forces.
    If --use_fd is not provided, we skip FD entirely and do NOT write the FD file.
    """
    folder = os.path.abspath(args.save_path)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    f_struct   = os.path.join(folder, f"{args.name_molecule}_structure.dat")
    f_energy   = os.path.join(folder, f"{args.name_molecule}_energy.dat")
    f_force    = os.path.join(folder, f"{args.name_molecule}_force.dat")
    f_force_fd = os.path.join(folder, f"{args.name_molecule}_force_fd.dat")

    mol = mycc._scf.mol
    coords_bohr = mol.atom_coords()  # shape (n_atoms, 3) in Bohr
    coords_ang = coords_bohr * BOHR2ANG
    coords_flat = coords_ang.flatten()

    # Convert CCSD total energy to eV
    E_tot_eV = mycc.e_tot * Ha_to_eV

    # 1) Analytic CCSD force => eV/Å
    ccsd_f = analytic_ccsd_forces(mycc)
    if ccsd_f is None:
        ccsd_f = np.zeros_like(coords_bohr)  # shape (n_atoms, 3)
    ccsd_f_flat = ccsd_f.flatten()

    # Write geometry, total energy, analytic forces
    with open(f_struct, "a") as fs:
        fs.write(" ".join(map(str, coords_flat)) + "\n")

    with open(f_energy, "a") as fe:
        fe.write(f"{E_tot_eV}\n")

    with open(f_force, "a") as ff:
        ff.write(" ".join(map(str, ccsd_f_flat)) + "\n")

    # 2) If user sets --use_fd, compute and write FD forces
    if args.use_fd:
        fd_f = finite_difference_ccsd_forces(mycc, args, delta=delta)
        if fd_f is None:
            fd_f = np.zeros_like(coords_bohr)
        fd_f_flat = fd_f.flatten()

        with open(f_force_fd, "a") as ffd:
            ffd.write(" ".join(map(str, fd_f_flat)) + "\n")

def process_molecule(mol_data, args):
    """
    For each molecule:
    1) run CCSD
    2) compute analytic forces
    3) optionally compute FD forces if --use_fd
    4) save geometry, energies, forces
    """
    mol_index, atom = mol_data
    try:
        mycc = run_ccsd(atom, args)
        if mycc is None:
            print(f"Molecule {mol_index}: CCSD failed => skipping.", flush=True)
            return False

        save_data(mycc, mol_index, args, delta=args.fd_delta)
        return True
    except Exception as e:
        print(f"Error in process_molecule: {e}")
        return False

def read_xyz_multimol(file_path):
    """
    Reads a multi-molecule XYZ file in Angstrom.
    Returns: 
      charges_tot -> list of atomic-number lists
      coords_tot  -> list of Nx3 coordinates in Angstrom
    """
    atomic_number = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'Cl':17}
    charges_all = []
    coords_all  = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            natom = int(line)
            i += 1
            # Possibly skip a comment line
            if i < len(lines):
                i += 1
            z_list = []
            c_list = []
            for _ in range(natom):
                parts = lines[i].split()
                i += 1
                sym = parts[0]
                x, y, z = map(float, parts[1:4])
                if sym not in atomic_number:
                    raise ValueError(f"Unknown symbol '{sym}'. Please extend atomic_number dict.")
                z_list.append(atomic_number[sym])
                c_list.append([x,y,z])
            charges_all.append(z_list)
            coords_all.append(c_list)
        else:
            i += 1

    return np.array(charges_all, dtype=object), np.array(coords_all, dtype=object)

def main(args):
    if not os.path.isfile(args.xyz_path):
        raise FileNotFoundError(f"File {args.xyz_path} not found!")
    t0 = perf_counter()

    # read multi-XYZ
    atom_charges, atom_coords = read_xyz_multimol(args.xyz_path)
    # package data
    data_list = package_data(atom_charges, atom_coords)

    from multiprocessing import Pool
    with Pool(args.n_workers) as p:
        fn = partial(process_molecule, args=args)
        for i, success in enumerate(p.imap_unordered(fn, enumerate(data_list))):
            if not success:
                print(f"Molecule {i}: CCSD or calculation failed!")
            if perf_counter() - t0 > 30:
                #print(f"Processed {i+1} molecules so far...", flush=True)
                t0 = perf_counter()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CCSD script with optional finite-difference force calculation"
    )
    parser.add_argument("xyz_path", type=str, help="Multi-molecule XYZ file in Angstrom.")
    parser.add_argument("--save_path", default="./", help="Directory for result files.")
    parser.add_argument("--name_molecule", default="molecule", help="Prefix for saved files.")
    parser.add_argument("--verbose", type=int, default=0, help="PySCF verbosity.")
    parser.add_argument("--basisset", default="ccpvdz", help="Basis set name.")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--fd_delta", type=float, default=0.01,
                        help="Displacement in Angstrom for the finite-difference approach.")
    # New: a boolean argument to toggle FD or skip it entirely
    parser.add_argument("--use_fd", action="store_true",
                        help="If provided, compute FD CCSD forces. Otherwise skip FD.")
    args = parser.parse_args()

    main(args)
