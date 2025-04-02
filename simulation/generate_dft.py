#!/usr/bin/env python3
import argparse
import os
from functools import partial
from time import perf_counter
from multiprocessing import Pool
import numpy as np

from pyscf import gto, dft, scf
from pyscf.dft import numint, libxc

BOHR2ANG = 0.52917721092
ANG2BOHR = 1.0 / BOHR2ANG
Ha_to_eV = 27.2114

def eval_rho_end_exc(mf, dm=None):
    """
    Calculates the electronic density (rho) and XC energy density (exc) on the real-space grid.
    (Optional utility, not required for forces.)
    """
    if dm is None:
        dm = mf.make_rdm1()
    xc_code = mf.xc
    xc_type = libxc.xc_type(xc_code)

    ao = numint.eval_ao(mf.mol, mf.grids.coords, deriv=0)
    rho_data = numint.eval_rho(mf.mol, ao, dm, xctype=xc_type)
    exc, *_ = libxc.eval_xc(xc_code, rho_data, deriv=0)

    rho = rho_data[0]
    return rho.flatten(), exc.flatten()

def package_data(atom_charges, atom_coords):
    """
    Packages data from the dataset into (Z, coords) for each molecule
    in Angstrom. Returns a list of "molecules", each a list of (Z,(x,y,z)).
    """
    all_mols = []
    for i in range(len(atom_charges)):
        Z_list = atom_charges[i]
        R_list = atom_coords[i]
        mol_data = list(zip(Z_list, R_list))  # e.g. [(Z1,(x1,y1,z1)), ...]
        all_mols.append(mol_data)
    return all_mols

def run_dft(atom, args):
    """
    Builds a PySCF Mole in Angstrom, runs DFT with given XC, returns the mf object.
    """
    mol = gto.Mole(unit="Angstrom")
    mol.atom = []
    for (Z, xyz) in atom:
        mol.atom.append((int(Z), tuple(xyz)))  # Z is atomic number, coords in Angstrom
    mol.basis = args.basisset
    mol.spin = args.spin
    mol.charge = args.charge
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = args.xc
    mf.grids.level = args.grid_level
    mf.verbose = args.verbose
    mf.kernel()

    if not mf.converged:
        # Attempt newton SCF if standard DFT fails to converge
        mf = scf.newton(mf)
        mf.kernel()

    return mf

def analytic_dft_forces(mf):
    """
    Computes analytic DFT forces (in eV/Å).
    'mf.nuc_grad_method().kernel()' => gradient in atomic units: (Hartree/Bohr).
      Force = -Grad
    We convert to eV/Å.
    """
    g = mf.nuc_grad_method()
    grad_au = g.kernel()        # shape (n_atoms, 3)
    force_au = -grad_au         # negative of gradient
    # Convert from Hartree/Bohr -> eV/Å
    force_eV_A = force_au * (Ha_to_eV / BOHR2ANG)
    return force_eV_A

def dft_energy(atom_data, args):
    """
    Builds a Mole in Angstrom, runs DFT, returns total energy in Hartree.
    For FD approach only.
    """
    mol = gto.Mole(unit="Angstrom")
    mol.atom = []
    for (Z, xyz) in atom_data:
        mol.atom.append((int(Z), tuple(xyz)))
    mol.basis = args.basisset
    mol.spin = args.spin
    mol.charge = args.charge
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = args.xc
    mf.grids.level = args.grid_level
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf)
        mf.kernel()
    return mf.e_tot

def finite_difference_dft_forces(base_mf, args, delta=0.01):
    """
    Finite-difference DFT forces:
      F_i = -( E(r+delta) - E(r-delta) ) / (2*delta )
    'delta' is in Angstrom.
    1) Extract geometry from base_mf (in Bohr), convert to Angstrom,
    2) For each +/- displacement, compute DFT energy in Hartree,
    3) Force => eV/Å
    """
    mol = base_mf.mol
    coords_bohr = mol.atom_coords()  # shape (n_atoms, 3) in Bohr
    n_atoms = coords_bohr.shape[0]
    coords_ang = coords_bohr * BOHR2ANG  # convert to Angstrom

    Z_list = [mol.atom_charge(i) for i in range(n_atoms)]
    forces = np.zeros((n_atoms, 3), dtype=float)
    for i_atom in range(n_atoms):
        for dim in range(3):
            plus_data = []
            minus_data = []
            for idx in range(n_atoms):
                Z = Z_list[idx]
                posA = np.array(coords_ang[idx], dtype=float)
                if idx == i_atom:
                    pos_plus = posA.copy()
                    pos_plus[dim] += delta
                    pos_minus = posA.copy()
                    pos_minus[dim] -= delta
                    plus_data.append((Z, pos_plus))
                    minus_data.append((Z, pos_minus))
                else:
                    plus_data.append((Z, posA))
                    minus_data.append((Z, posA))

            e_plus  = dft_energy(plus_data, args)
            e_minus = dft_energy(minus_data, args)
            dE_dX = (e_plus - e_minus) / (2.0 * delta)  # Hartree/Ang
            # Force is negative derivative
            forces[i_atom, dim] = -dE_dX * Ha_to_eV  # eV/Å

    return forces

def save_data(mf, mol_index, args, folder=None, delta=0.01):
    """
    Saves total energy, analytic DFT forces, and FD DFT forces (ONLY if --use_fd).
    If --use_fd is not specified, it will NOT compute or write FD data.
    """
    folder = os.path.abspath(folder or ".")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    f_energy    = os.path.join(folder, f"{args.name_molecule}_energy.dat")
    f_force     = os.path.join(folder, f"{args.name_molecule}_force.dat")

    # 1) DFT total energy in eV
    E_eV = mf.e_tot * Ha_to_eV
    # 2) Analytic DFT force => eV/Å
    forces_analytic = analytic_dft_forces(mf)
    if forces_analytic is None:
        forces_analytic = np.zeros((mf.mol.natm, 3))
    f_analytic_flat = forces_analytic.flatten()

    # Save energies and analytic forces
    with open(f_energy, "a") as fe:
        fe.write(f"{E_eV}\n")

    with open(f_force, "a") as ff:
        ff.write(' '.join(map(str, f_analytic_flat)) + "\n")

    # If user wants FD, do it now
    if args.use_fd:
        f_force_fd  = os.path.join(folder, f"{args.name_molecule}_force_fd.dat")
        fd_forces = finite_difference_dft_forces(mf, args, delta=delta)
        if fd_forces is None:
            fd_forces = np.zeros((mf.mol.natm, 3))
        f_fd_flat = fd_forces.flatten()
        with open(f_force_fd, "a") as ffd:
            ffd.write(' '.join(map(str, f_fd_flat)) + "\n")

def process_mol(mol_data, atom_charges, args):
    """
    Runs DFT on each molecule. If converged, saves energies & forces. 
    FD is optional, controlled by args.use_fd.
    """
    mol_index, atom = mol_data
    mf = run_dft(atom, args)
    if mf.converged:
        save_data(mf, mol_index, args, folder=args.save_path, delta=args.fd_delta)
    return mf.converged

def read_xyz_multimol(file_path):
    """
    Reads a multi-molecule XYZ file (in Angstrom).
    Returns two lists: [charges_tot], [coords_tot].
    Example:
      charges_tot[i] -> list of atomic numbers for molecule i
      coords_tot[i]  -> Nx3 list of [x,y,z] for molecule i in Angstrom
    """
    atomic_number = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Cl': 17}
    charges_all = []
    coords_all  = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.isdigit():
                natom = int(line)
                i += 1  # skip or consume the comment line if present
                _ = lines[i].strip()
                i += 1
                c_list = []
                z_list = []
                for _ in range(natom):
                    parts = lines[i].split()
                    i += 1
                    sym = parts[0]
                    x, y, z = map(float, parts[1:4])
                    z_list.append(atomic_number[sym])
                    c_list.append([x,y,z])
                charges_all.append(z_list)
                coords_all.append(c_list)
            else:
                i += 1

    return charges_all, coords_all

def main(args):
    start_t = perf_counter()
    xyz_file = args.xyz_path
    assert os.path.isfile(xyz_file), f"File {xyz_file} not found!"

    atom_charges, atom_coords = read_xyz_multimol(xyz_file)
    # Build data for parallel processing
    data = package_data(atom_charges, atom_coords)

    from multiprocessing import Pool
    with Pool(args.n_workers) as p:
        fn = partial(process_mol, atom_charges=atom_charges, args=args)
        for i, converged in enumerate(p.imap_unordered(fn, enumerate(data))):
            if not converged:
                print(f"Failed to converge for molecule {i}.", flush=True)
            if perf_counter() - start_t > 30:
                #print(f"Processed {i} molecules so far...", flush=True)
                start_t = perf_counter()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz_path", type=str, help="Multi-molecule XYZ file (Angstrom)")
    parser.add_argument("--save_path", type=str, default="./",
                        help="Directory to save energy and force files.")
    parser.add_argument("--name_molecule", type=str, default="molecule",
                        help="Name prefix for saved files.")
    parser.add_argument("--verbose", type=int, default=0,
                        help="PySCF verbose level.")
    parser.add_argument("--xc", type=str, default="LDA",
                        help="XC functional code (e.g. 'PBE', 'B3LYP').")
    parser.add_argument("--basisset", type=str, default="ccpvdz",
                        help="Basis set for DFT.")
    parser.add_argument("--grid_level", type=int, default=1,
                        help="DFT integration grid level (0-9).")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Number of parallel worker processes.")
    parser.add_argument("--spin", type=int, default=0,
                        help="Spin multiplicity factor. For closed-shell, set 0.")
    parser.add_argument("--charge", type=int, default=0,
                        help="Total molecular charge.")
    parser.add_argument("--fd_delta", type=float, default=0.01,
                        help="Displacement in Angstrom for finite-difference forces.")

    # New: a boolean flag that, if omitted, we do NOT compute FD at all
    parser.add_argument("--use_fd", action="store_true",
                        help="If set, compute finite-difference forces. Otherwise skip FD entirely.")

    args = parser.parse_args()
    main(args)
