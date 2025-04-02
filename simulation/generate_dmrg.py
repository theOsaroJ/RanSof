#!/usr/bin/env python3
import argparse
import os
from functools import partial
from time import perf_counter
from multiprocessing import Pool
import numpy as np

from pyscf import gto, scf, mcscf, dmrgscf

# Attempt to locate BLOCK DMRG executable
dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''

##############################################################################
# Constants
##############################################################################
HARTREE_TO_EV = 27.2114
BOHR2ANG      = 0.52917721092  # 1 Bohr = 0.529177 Å
ANG2BOHR      = 1.0 / BOHR2ANG

##############################################################################
# 1) Read multi-XYZ
##############################################################################
def read_xyz_multimol(file_path):
    """
    Reads a multi-molecule XYZ file in Angstrom:
      NAtoms
      (comment line?)
      AtomSym X Y Z
      ...
    Returns (charges_tot, coords_tot), each a list of length #molecules,
    where charges_tot[i] -> list of atomic numbers,
          coords_tot[i]  -> Nx3 list of [x,y,z].
    """
    atomic_number = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'Cl':17}
    charges_tot = []
    coords_tot  = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            natom = int(line)
            i += 1
            # skip a comment line (if present)
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
                    raise ValueError(f"Unknown atom symbol: {sym}. Extend dictionary if needed.")
                z_list.append(atomic_number[sym])
                c_list.append([x,y,z])
            charges_tot.append(z_list)
            coords_tot.append(c_list)
        else:
            i += 1

    return np.array(charges_tot, dtype=object), np.array(coords_tot, dtype=object)

##############################################################################
# 2) Package data
##############################################################################
def package_data(atom_charges, atom_coords):
    """
    Returns a list of snapshots, each snapshot is [ (Z, (x,y,z)), ... ] in Angstrom.
    """
    out = []
    for c_list, r_list in zip(atom_charges, atom_coords):
        one_mol = list(zip(c_list, r_list))
        out.append(one_mol)
    return out

##############################################################################
# 3) Build and run a DMRG-based MCSCF on the original geometry
##############################################################################
def run_dmrg(atom_list, args):
    """
    Builds a PySCF Mole from (Z, (x,y,z)) in Angstrom,
    runs DMRG. Returns mc object for further analysis.
    """
    mol = gto.Mole()
    mol.unit = "Angstrom"
    mol.atom = []
    for (Z, (xA, yA, zA)) in atom_list:
        mol.atom.append((int(Z), (xA, yA, zA)))
    mol.basis  = args.basisset
    mol.spin   = args.spin
    mol.charge = args.charge
    mol.build()

    mf = scf.RHF(mol)
    mf.verbose = args.verbose
    mf.run()

    mc = mcscf.CASCI(mf, args.num_orbitals, args.num_electrons)
    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=args.max_dmrg_bond_dimension)
    # set the maximum DMRG iterations
    mc.fcisolver.maxIter = args.max_dmrg_iterations
    mc.verbose = args.verbose
    mc.kernel()

    return mc

##############################################################################
# 4) Attempt to get analytic gradient for the entire geometry
##############################################################################
def analytic_dmrg_forces(mc):
    """
    If supported, PySCF MCSCF gradient is in atomic units (Hartree/Bohr).
    Force = -Grad. Convert to eV/Å. If not available, returns None.
    """
    try:
        grad_method = mc.nuc_grad_method()  # might fail if not implemented
        grad_au = grad_method.kernel()      # shape (n_atoms, 3), in Hartree/Bohr
        force_au = -grad_au                 # force is negative of gradient
        force_eV_A = force_au * (HARTREE_TO_EV / BOHR2ANG)
        return force_eV_A
    except AttributeError:
        print("Analytic gradient not implemented for this MCSCF/DMRG scenario.")
        return None
    except Exception as e:
        print(f"Error computing analytic gradient: {e}")
        return None

##############################################################################
# 5) Rebuild geometry -> run DMRG -> get total energy in Hartree
##############################################################################
def dmrg_energy(atom_list, args):
    """
    Build a molecule from (Z,(x,y,z)) in Angstrom, run DMRG, return e_tot in Hartree.
    Used by the FD approach for +/- displacements.
    """
    mol = gto.Mole()
    mol.unit = "Angstrom"
    mol.atom = []
    for (Z, (xA, yA, zA)) in atom_list:
        mol.atom.append((int(Z), (xA, yA, zA)))
    mol.basis  = args.basisset
    mol.spin   = args.spin
    mol.charge = args.charge
    mol.build()

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    mc = mcscf.CASCI(mf, args.num_orbitals, args.num_electrons)
    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=args.max_dmrg_bond_dimension)
    mc.fcisolver.maxIter = args.max_dmrg_iterations
    mc.verbose = 0
    mc.kernel()

    return mc.e_tot

##############################################################################
# 6) Finite-Difference forces for the entire geometry
##############################################################################
def finite_difference_dmrg_forces(base_mc, args, delta=0.01):
    """
    For each atom/dimension, do:
       E+ = E(r + delta along that dimension)
       E- = E(r - delta)
       F = -(E+ - E-) / (2*delta)
    geometry is in Angstrom, energies in Hartree -> final forces eV/Å
    """
    mol = base_mc._scf.mol
    n_atoms = mol.natm

    # PySCF internal coords are in Bohr, so convert to Ang for shifting
    coords_bohr = mol.atom_coords()  # shape (n_atoms,3) in Bohr
    coords_ang  = coords_bohr * BOHR2ANG

    # get atomic numbers
    Z_list = [mol.atom_charge(i) for i in range(n_atoms)]
    forces = np.zeros((n_atoms,3), dtype=float)

    for i_atom in range(n_atoms):
        for dim in range(3):
            # build plus geometry
            plus_coords = coords_ang.copy()
            plus_coords[i_atom, dim] += delta
            plus_atom_list = list(zip(Z_list, plus_coords))

            # build minus geometry
            minus_coords = coords_ang.copy()
            minus_coords[i_atom, dim] -= delta
            minus_atom_list = list(zip(Z_list, minus_coords))

            E_plus  = dmrg_energy(plus_atom_list, args)  # Hartree
            E_minus = dmrg_energy(minus_atom_list, args) # Hartree

            dE_dX_Ha_per_A = (E_plus - E_minus) / (2.0 * delta)
            # Force = negative derivative, convert Hartree -> eV
            F_eV_A = - dE_dX_Ha_per_A * HARTREE_TO_EV
            forces[i_atom, dim] = F_eV_A

    return forces

##############################################################################
# 7) Save geometry, energies, forces
##############################################################################
def save_data(mc, mol_index, args, delta=0.01):
    """
    Saves:
     - geometry (in Angstrom)
     - total DMRG energy (in eV)
     - entire analytic force vector (if available) in eV/Å
     - FD force if --use_fd is set, otherwise skip FD
    """
    folder = os.path.abspath(args.save_path)
    os.makedirs(folder, exist_ok=True)

    f_struct  = os.path.join(folder, f"{args.name_molecule}_structure.dat")
    f_energy  = os.path.join(folder, f"{args.name_molecule}_energy.dat")
    f_force   = os.path.join(folder, f"{args.name_molecule}_force.dat")
    f_fdforce = os.path.join(folder, f"{args.name_molecule}_force_fd.dat")

    mol = mc._scf.mol
    n_atoms = mol.natm
    coords_bohr = mol.atom_coords()  # shape (n_atoms,3) in Bohr
    coords_ang  = coords_bohr * BOHR2ANG

    # Flatten geometry
    coords_flat = coords_ang.flatten()

    # total energy in eV
    E_tot_eV = mc.e_tot * HARTREE_TO_EV

    # Attempt analytic forces
    an_forces = analytic_dmrg_forces(mc)
    if an_forces is None:
        an_forces = np.zeros((n_atoms,3), dtype=float)
    an_forces_flat = an_forces.flatten()

    # Write geometry, total energy, analytic forces
    with open(f_struct, "a") as fs:
        fs.write(" ".join(map(str, coords_flat)) + "\n")

    with open(f_energy, "a") as fe:
        fe.write(f"{E_tot_eV}\n")

    with open(f_force, "a") as ff:
        ff.write(" ".join(map(str, an_forces_flat)) + "\n")

    # If user wants FD, do it. Otherwise skip
    if args.use_fd:
        fd_forces = finite_difference_dmrg_forces(mc, args, delta=delta)
        fd_forces_flat = fd_forces.flatten()
        with open(f_fdforce, "a") as ffd:
            ffd.write(" ".join(map(str, fd_forces_flat)) + "\n")

##############################################################################
# 8) Process a single molecule
##############################################################################
def process_mol(mol_data, args):
    """
    1) Build DMRG for the original geometry.
    2) Save geometry, total energy, entire forces (analytic), FD if requested.
    """
    mol_index, atom_list = mol_data
    mc = run_dmrg(atom_list, args)
    if mc.converged:
        save_data(mc, mol_index, args, delta=args.fd_delta)
    return mc.converged

##############################################################################
# 9) Main
##############################################################################
def main(args):
    t0 = perf_counter()
    xyz_file = args.xyz_path
    if not os.path.isfile(xyz_file):
        raise FileNotFoundError(f"File {xyz_file} not found.")

    # 1) read multi-XYZ
    atom_charges, atom_coords = read_xyz_multimol(xyz_file)
    # 2) package
    data_list = package_data(atom_charges, atom_coords)

    # 3) parallel
    from multiprocessing import Pool
    with Pool(args.n_workers) as p:
        fn = partial(process_mol, args=args)
        for i, converged in enumerate(p.imap_unordered(fn, enumerate(data_list))):
            if not converged:
                print(f"Molecule {i}: DMRG not converged!", flush=True)
            if perf_counter() - t0 > 30:
                print(f"Processed {i+1} molecules so far...", flush=True)
                t0 = perf_counter()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMRG script with optional FD forces.")
    parser.add_argument("xyz_path", type=str, help="Path to multi-XYZ file (Angstrom).")
    parser.add_argument("--save_path", default="./", help="Directory to save results.")
    parser.add_argument("--name_molecule", default="molecule", help="Prefix for saved files.")
    parser.add_argument("--verbose", type=int, default=4, help="PySCF verbosity.")
    parser.add_argument("--basisset", type=str, default="ccpvdz", help="Basis set.")
    parser.add_argument("--spin", type=int, default=0, help="Total spin.")
    parser.add_argument("--charge", type=int, default=0, help="Molecular charge.")
    parser.add_argument("--num_orbitals", type=int, default=6, help="Active orbitals for CAS.")
    parser.add_argument("--num_electrons", type=int, default=6, help="Active electrons for CAS.")
    parser.add_argument("--max_dmrg_bond_dimension", type=int, default=50,
                        help="Max bond dimension for DMRG.")
    parser.add_argument("--max_dmrg_iterations", type=int, default=100,
                        help="Max number of DMRG iterations.")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of parallel processes.")
    parser.add_argument("--fd_delta", type=float, default=0.01,
                        help="Finite-displacement (Å) for FD forces.")
    # NEW: boolean flag to toggle FD
    parser.add_argument("--use_fd", action="store_true",
                        help="If set, compute FD forces. If omitted, skip FD.")
    args = parser.parse_args()

    main(args)
