#!/usr/bin/env python3
import os
import sys
import logging
import numpy as np
import pandas as pd
from utils.io_utils import read_xyz_multimol, write_xyz, load_forces, load_latest_value

# Save the training data (XYZ) file.
def save_training_xyz(X_train, original_geom, element_sequences, molecule_ids, filename):
    n = len(molecule_ids)
    if n == 0 or len(original_geom) == 0:
        return
    n_atoms = len(original_geom[0])
    X_coord = X_train[:n, :n_atoms * 3]
    molecules = []
    for row in X_coord:
        coords = np.array(row).reshape(n_atoms, 3).tolist()
        molecules.append(coords)
    write_xyz(molecules, molecule_ids, element_sequences[:n], filename)

# Split the data indices based on energy values.
def split_prior_by_energy(Y_lower, n_train):
    n_total = Y_lower.shape[0]
    if n_total <= n_train:
        return np.arange(n_total), np.array([])
    else:
        sorted_idx = np.argsort(Y_lower.ravel())
        n_low = int(n_train * 0.3)
        n_high = int(n_train * 0.3)
        n_mean = n_train - n_low - n_high
        idx_low = sorted_idx[:n_low]
        idx_high = sorted_idx[-n_high:]
        mean_val = np.mean(Y_lower)
        abs_diff = np.abs(Y_lower.ravel() - mean_val)
        smean = np.argsort(abs_diff)
        remain = [i for i in smean if i not in idx_low and i not in idx_high]
        idx_mean = np.array(remain[:n_mean])
        training_idx = np.concatenate((idx_low, idx_high, idx_mean))
    candidate_idx = np.array([i for i in range(n_total) if i not in training_idx])
    return training_idx, candidate_idx

# This function runs the active learning (AL) loop for a single transition.
def active_learning_correction(lower_level, higher_level, X_candidate, X_geom_all,
                               gp_model_energy, gp_model_forces,
                               Y_lower_energy_all, Y_lower_forces_all,
                               candidate_elem_seq, candidate_indices,
                               training_ids, training_indices,
                               al_cfg, funnel, final_E, final_F):
    logging.info(f"Starting AL from {lower_level} -> {higher_level}.")
    min_threshold = al_cfg["min_std_threshold"]
    max_iterations = al_cfg["al_iterations_per_level"].get(higher_level, None)
    window_size = al_cfg["window_size"]
    conv_tol = al_cfg["convergence_tolerance"]
    default_unc = al_cfg["default_uncertainty_threshold"]
    fixed_mode = (max_iterations is not None)
    iteration = 0
    stdE_window = []
    stdF_window = []
    while True:
        if X_candidate.shape[0] == 0:
            logging.info("No unlabeled candidates remain. Stopping AL loop.")
            break
        iteration += 1
        logging.info(f"AL iteration {iteration} starting for {lower_level} -> {higher_level}.")
        if fixed_mode and iteration > max_iterations:
            logging.info(f"Reached fixed iteration count ({max_iterations}). Stopping AL loop.")
            break
        # Predict uncertainties for all candidates.
        _, varE = gp_model_energy.predict(X_candidate)
        stdE = np.sqrt(np.clip(varE, 1e-8, None))
        _, varF = gp_model_forces.predict(X_candidate)
        stdF = np.sqrt(np.clip(varF, 1e-8, None))
        mean_stdF = np.mean(stdF, axis=1)
        avg_std_e = float(np.mean(stdE))
        avg_std_f = float(np.mean(mean_stdF))
        logging.info(f"AL iter {iteration}: avg std energy = {avg_std_e:.6f}, avg std forces = {avg_std_f:.6f}")
        stdE_window.append(avg_std_e)
        stdF_window.append(avg_std_f)
        # Check convergence over a rolling window.
        if (not fixed_mode) and (iteration >= window_size):
            wE = np.mean(stdE_window[-window_size:])
            wF = np.mean(stdF_window[-window_size:])
            if abs(avg_std_e - wE) < conv_tol and abs(avg_std_f - wF) < conv_tol:
                logging.info(f"AL converged by rolling window at iteration {iteration}.")
                break
            if np.max(stdE) < default_unc and np.max(mean_stdF) < default_unc:
                logging.info(f"All uncertainties below threshold at iteration {iteration}.")
                break
        # Select candidate if uncertainties exceed thresholds.
        if avg_std_e > min_threshold or avg_std_f > min_threshold:
            combined_uncert = np.maximum(stdE.ravel(), mean_stdF)
            chosen_local_idx = int(np.argmax(combined_uncert))
            chosen_orig_idx = candidate_indices[chosen_local_idx]
            logging.info(f"AL iter {iteration}: Selected candidate original index {chosen_orig_idx}.")
            # Remove candidate from the unlabeled pool.
            X_candidate = np.delete(X_candidate, chosen_local_idx, axis=0)
            c_elem = candidate_elem_seq.pop(chosen_local_idx)
            candidate_indices = np.delete(candidate_indices, chosen_local_idx, axis=0)
            xyz_file = f"selected_uncertain_structures_iter{iteration}.xyz"
            mol_geom = X_geom_all[chosen_orig_idx]
            n_atoms = len(mol_geom)
            with open(xyz_file, "w") as fff:
                fff.write(f"{n_atoms}\n")
                fff.write(f"Molecule {chosen_orig_idx+1}\n")
                elems_arr = np.array(c_elem.split())
                for at in range(n_atoms):
                    x, y, z = mol_geom[at]
                    fff.write(f"{elems_arr[at]} {x:.8f} {y:.8f} {z:.8f}\n")
            logging.info(f"Temporary XYZ file written: {xyz_file}")
            # Run lower-level simulation if the lower level is not the base.
            if lower_level != funnel[0]:
                if lower_level in ["CCSD", "DMRG"]:
                    if lower_level == "CCSD":
                        cmd_lo = f"python3 simulation/generate_ccsd.py {xyz_file} --save_path simulation_data --name_molecule m{lower_level.lower()} --basisset 6311g --n_workers 1"
                    else:
                        cmd_lo = f"python3 simulation/generate_dmrg.py {xyz_file} --save_path simulation_data --name_molecule m{lower_level.lower()} --basisset 6311g --num_orbitals 6 --num_electrons 6 --max_dmrg_bond_dimension 100 --max_dmrg_iterations 500 --n_workers 1 --verbose 0"
                else:
                    cmd_lo = f"python3 simulation/generate_dft.py {xyz_file} --save_path simulation_data --name_molecule m{lower_level.lower()} --xc {lower_level} --basisset 6311g --n_workers 1 --grid_level 6 --spin 0 --charge 0 --fd_delta 0.001"
                logging.info(f"Running lower-level sim: {cmd_lo}")
                os.system(cmd_lo)
            # Run higher-level simulation.
            if higher_level in ["CCSD", "DMRG"]:
                if higher_level == "CCSD":
                    cmd_hi = f"python3 simulation/generate_ccsd.py {xyz_file} --save_path simulation_data --name_molecule m{higher_level.lower()} --basisset 6311g --n_workers 1"
                else:
                    cmd_hi = f"python3 simulation/generate_dmrg.py {xyz_file} --save_path simulation_data --name_molecule m{higher_level.lower()} --basisset 6311g --num_orbitals 6 --num_electrons 6 --max_dmrg_bond_dimension 100 --max_dmrg_iterations 500 --n_workers 1 --verbose 0"
            else:
                cmd_hi = f"python3 simulation/generate_dft.py {xyz_file} --save_path simulation_data --name_molecule m{higher_level.lower()} --xc {higher_level} --basisset 6311g --n_workers 1 --grid_level 6 --spin 0 --charge 0 --fd_delta 0.001"
            logging.info(f"Running higher-level sim: {cmd_hi}")
            os.system(cmd_hi)
            sp = "simulation_data"
            lo_pref = f"m{lower_level.lower()}"
            hi_pref = f"m{higher_level.lower()}"
            try:
                if lower_level == funnel[0]:
                    new_loE = Y_lower_energy_all[chosen_orig_idx].reshape(1, 1)
                    new_loF = Y_lower_forces_all[chosen_orig_idx].reshape(1, n_atoms * 3)
                else:
                    new_loE = load_latest_value(os.path.join(sp, f"{lo_pref}_energy.dat"), (1, 1))
                    new_loF = load_latest_value(os.path.join(sp, f"{lo_pref}_force.dat"), (1, n_atoms * 3))
                if higher_level == funnel[0]:
                    new_hiE = new_loE
                    new_hiF = new_loF
                else:
                    new_hiE = load_latest_value(os.path.join(sp, f"{hi_pref}_energy.dat"), (1, 1))
                    new_hiF = load_latest_value(os.path.join(sp, f"{hi_pref}_force.dat"), (1, n_atoms * 3))
                # For CCSD/DMRG transitions, compute target delta as – (simulated – lower)
                if higher_level in ["CCSD", "DMRG"]:
                    deltaE = - (new_hiE - new_loE)
                    deltaF = - (new_hiF - new_loF)
                else:
                    deltaE = new_hiE - new_loE
                    deltaF = new_hiF - new_loF
            except Exception as e:
                logging.error(f"Simulation for candidate {chosen_orig_idx} failed => skipping GP update: {e}")
                continue
            from numpy import hstack, vstack
            coords_flat = np.array(X_geom_all[chosen_orig_idx]).flatten().reshape(1, -1)
            lower_idx_val = funnel.index(lower_level)
            if lower_idx_val == 0:
                new_desc = hstack((coords_flat, new_loE, new_loF))
            else:
                cumulative = []
                for k in range(lower_idx_val, 0, -1):
                    deltaE_k = final_E[funnel[k]] - final_E[funnel[k-1]]
                    deltaF_k = final_F[funnel[k]] - final_F[funnel[k-1]]
                    cumulative.append(deltaE_k[chosen_orig_idx:chosen_orig_idx+1])
                    cumulative.append(deltaF_k[chosen_orig_idx:chosen_orig_idx+1])
                cum_array = np.hstack(cumulative)
                new_desc = hstack((coords_flat, new_loE, new_loF, cum_array))
            X_train_new = vstack((gp_model_energy.X_train, new_desc))
            Y_train_new = vstack((gp_model_energy.y_train, deltaE))
            gp_model_energy.fit(X_train_new, Y_train_new)
            gp_model_energy.optimize_hyperparameters()
            X_train_new_f = vstack((gp_model_forces.X_train, new_desc))
            Y_train_new_f = vstack((gp_model_forces.y_train, deltaF))
            gp_model_forces.fit(X_train_new_f, Y_train_new_f)
            gp_model_forces.optimize_hyperparameters()
            training_ids.append(str(chosen_orig_idx + 1))
            training_indices = np.append(training_indices, chosen_orig_idx)
            logging.info(f"AL iteration {iteration} complete; remaining unlabeled: {X_candidate.shape[0]}")
        else:
            logging.info(f"Uncertainty below threshold at iteration {iteration}.")
            if (not fixed_mode) and (iteration >= window_size):
                logging.info(f"Reached window_size ({window_size}) iterations; stopping AL loop.")
                break
    return (X_candidate, Y_lower_energy_all, Y_lower_forces_all,
            gp_model_energy, gp_model_forces,
            candidate_elem_seq, candidate_indices,
            training_ids, training_indices)

def run_workflow(config):
    os.makedirs(config["paths"]["output_dir"], exist_ok=True)
    os.makedirs("simulation_data", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=os.path.join(config["paths"]["logs_dir"], "app.log"),
        filemode="a"
    )
    logging.info("Starting full multi-level AML workflow...")
    funnel = config["funnel"]
    paths = config["paths"]
    al_cfg = config["active_learning"]
    
    from utils.io_utils import read_xyz_multimol, load_forces
    from models.custom_gp import GaussianProcess
    from models.force_gp import ForceGP
    X_geom_all, elem_sequences_all = read_xyz_multimol(paths["input_xyz"])
    n_mol = len(X_geom_all)
    n_atoms = len(X_geom_all[0])
    try:
        Y_loE = np.loadtxt(paths["initial_energy_file"]).reshape(-1, 1)
        Y_loF = load_forces(paths["initial_forces_file"], n_mol, n_atoms * 3)
    except Exception as e:
        logging.error(f"Error reading lowest fidelity data: {e}")
        sys.exit(1)
    
    import pandas as pd
    df_lo = pd.DataFrame({
        "Molecule": np.arange(1, n_mol + 1),
        "Element_Sequence": elem_sequences_all,
        "Flattened_Coords": [" ".join(map(str, np.array(arr).flatten())) for arr in X_geom_all],
        f"{funnel[0]}_Energy": Y_loE.ravel(),
        f"{funnel[0]}_Forces": [" ".join(map(str, row)) for row in Y_loF]
    })
    outdir = paths["output_dir"]
    df_lo.to_csv(os.path.join(outdir, f"lowest_fidelity_{funnel[0]}.csv"), index=False)
    logging.info("Saved lowest fidelity data.")
    
    coords_flat = np.array([np.array(m).flatten() for m in X_geom_all])
    X_all_desc = np.hstack((coords_flat, Y_loE, Y_loF))
    final_E = {funnel[0]: Y_loE}
    final_F = {funnel[0]: Y_loF}
    
    for i in range(len(funnel) - 1):
        lower = funnel[i]
        higher = funnel[i+1]
        logging.info(f"=== Transition from {lower} -> {higher} ===")
        lower_idx = funnel.index(lower)
        if lower_idx == 0:
            X_features = np.hstack((coords_flat, final_E[lower], final_F[lower]))
        else:
            cumulative = []
            for k in range(lower_idx, 0, -1):
                deltaE_k = final_E[funnel[k]] - final_E[funnel[k-1]]
                deltaF_k = final_F[funnel[k]] - final_F[funnel[k-1]]
                cumulative.append(deltaE_k)
                cumulative.append(deltaF_k)
            cum_array = np.hstack(cumulative)
            X_features = np.hstack((coords_flat, final_E[lower], final_F[lower], cum_array))
        n_train = al_cfg["n_training_points"].get(higher, 20)
        train_idx, candidate_idx = split_prior_by_energy(final_E[lower], n_train)
        train_idx = np.array(train_idx, dtype=int)
        candidate_idx = np.array(candidate_idx, dtype=int)
        X_train = X_features[train_idx, :]
        X_candidate = X_features[candidate_idx, :]
        candidate_elem_seq = [elem_sequences_all[j] for j in candidate_idx]
        training_indices = np.array(train_idx)
        training_ids = [str(j+1) for j in train_idx]
        
        from utils.io_utils import write_xyz
        train_mols = [X_geom_all[j] for j in train_idx]
        train_elems = [elem_sequences_all[j] for j in train_idx]
        train_xyz = os.path.join("temp", f"training_{higher}.xyz")
        write_xyz(train_mols, training_ids, train_elems, train_xyz)
        
        sim_cmd = config["simulation_commands"][higher].format(xyz=train_xyz)
        logging.info(f"Running simulation for training set at {higher}: {sim_cmd}")
        os.system(sim_cmd)
        
        prefix = f"m{higher.lower()}"
        sp = "simulation_data"
        efile = os.path.join(sp, f"{prefix}_energy.dat")
        ffile = os.path.join(sp, f"{prefix}_force.dat")
        try:
            nextE = np.loadtxt(efile).reshape(-1, 1)
            nextF = np.loadtxt(ffile).reshape(-1, n_atoms * 3)
        except Exception as e:
            logging.error(f"Error loading simulation outputs for {higher}: {e}")
            sys.exit(1)
        if os.path.exists(train_xyz):
            os.remove(train_xyz)
        if higher in ["CCSD", "DMRG"]:
            deltaE_train = - (nextE - final_E[lower][train_idx, :])
            deltaF_train = - (nextF - final_F[lower][train_idx, :])
        else:
            deltaE_train = nextE - final_E[lower][train_idx, :]
            deltaF_train = nextF - final_F[lower][train_idx, :]
        
        from models.custom_gp import GaussianProcess
        from models.force_gp import ForceGP
        gp_en = GaussianProcess(length_scale=1, noise=1e-2, alpha_rq=1, batch_size=200)
        gp_fr = ForceGP(output_dim=n_atoms * 3, length_scale=1e4, noise=1e-5, alpha_rq=10000, batch_size=200)

        gp_en.fit(X_train, deltaE_train)
        gp_fr.fit(X_train, deltaF_train)
        gp_en.optimize_hyperparameters()
        gp_fr.optimize_hyperparameters()
        logging.info(f"{higher} training: GP energy X shape: {gp_en.X_train.shape}, Y shape: {gp_en.y_train.shape}")
        
        pd.DataFrame(gp_en.X_train).to_csv(os.path.join(outdir, f"initial_training_features_{higher}.csv"), index=False)
        pd.DataFrame(gp_en.y_train, columns=["delta_energy"]).to_csv(os.path.join(outdir, f"initial_training_deltaE_{higher}.csv"), index=False)
        pd.DataFrame(gp_fr.predict(gp_fr.X_train)[0]).to_csv(os.path.join(outdir, f"initial_training_deltaF_{higher}.csv"), index=False)
        save_training_xyz(gp_en.X_train, train_mols, train_elems, training_ids, os.path.join(outdir, f"initial_training_{higher}.xyz"))
        # Save initial unlabeled data with molecule index.
        df_candidate_init = pd.DataFrame(X_candidate, columns=[f"feature_{j}" for j in range(X_candidate.shape[1])])
        df_candidate_init["Molecule_Index"] = candidate_idx
        df_candidate_init.to_csv(os.path.join(outdir, f"initial_unlabeled_data_{higher}.csv"), index=False)
        
        # Run the active learning (AL) loop for this transition.
        from active_learning.workflow import active_learning_correction
        (X_candidate, Y_lower_energy_all, Y_lower_forces_all,
         gp_en, gp_fr, candidate_elem_seq, candidate_idx,
         training_ids, training_indices) = active_learning_correction(
            lower_level=lower, higher_level=higher,
            X_candidate=X_candidate,
            X_geom_all=X_geom_all,
            gp_model_energy=gp_en,
            gp_model_forces=gp_fr,
            Y_lower_energy_all=final_E[lower],
            Y_lower_forces_all=final_F[lower],
            candidate_elem_seq=candidate_elem_seq,
            candidate_indices=candidate_idx,
            training_ids=training_ids,
            training_indices=training_indices,
            al_cfg=al_cfg,
            funnel=funnel,
            final_E=final_E,
            final_F=final_F
         )
        
        # Re-compute full features for final prediction.
        if funnel.index(lower) == 0:
            X_features = np.hstack((coords_flat, final_E[lower], final_F[lower]))
        else:
            cumulative = []
            for k in range(funnel.index(lower), 0, -1):
                deltaE_k = final_E[funnel[k]] - final_E[funnel[k-1]]
                deltaF_k = final_F[funnel[k]] - final_F[funnel[k-1]]
                cumulative.append(deltaE_k)
                cumulative.append(deltaF_k)
            cum_array = np.hstack(cumulative)
            X_features = np.hstack((coords_flat, final_E[lower], final_F[lower], cum_array))
        E_pred_full, _ = gp_en.predict(X_features)
        F_pred_full, _ = gp_fr.predict(X_features)
        if higher in ["CCSD", "DMRG"]:
            E_pred_full = -E_pred_full
            F_pred_full = -F_pred_full
        final_E[higher] = final_E[lower] + E_pred_full
        final_F[higher] = final_F[lower] + F_pred_full
        
        combined = np.hstack((gp_en.X_train, gp_en.y_train, gp_fr.y_train))
        headers = [f"feature_{j}" for j in range(gp_en.X_train.shape[1])]
        headers.append("delta_energy")
        headers += [f"delta_force_{j}" for j in range(gp_fr.y_train.shape[1])]
        pd.DataFrame(combined, columns=headers).to_csv(os.path.join(outdir, f"final_training_{higher}.csv"), index=False)
        
        Y_candidate_E = final_E[lower][candidate_idx, :]
        Y_candidate_F = final_F[lower][candidate_idx, :]
        pred_delta_E, _ = gp_en.predict(X_candidate)
        pred_delta_F, _ = gp_fr.predict(X_candidate)
        if higher in ["CCSD", "DMRG"]:
            pred_delta_E = -pred_delta_E
            pred_delta_F = -pred_delta_F
        final_E_unlabeled = Y_candidate_E + pred_delta_E
        final_F_unlabeled = Y_candidate_F + pred_delta_F
        np.savetxt(os.path.join(outdir, f"predicted_energy_{higher}_unlabeled.dat"), final_E_unlabeled, fmt="%.8f")
        np.savetxt(os.path.join(outdir, f"predicted_forces_{higher}_unlabeled.dat"),
                   np.array([row.flatten() for row in final_F_unlabeled]), fmt="%.8f")
        # Save final unlabeled candidate data with molecule indices.
        df_candidate_final = pd.DataFrame(X_candidate, columns=[f"feature_{j}" for j in range(X_candidate.shape[1])])
        if X_candidate.shape[0] != len(candidate_idx):
            logging.error(f"Mismatch in candidate indices: X_candidate has {X_candidate.shape[0]} rows but candidate_idx has {len(candidate_idx)} entries.")
            candidate_idx = candidate_idx[:X_candidate.shape[0]]
        df_candidate_final["Molecule_Index"] = candidate_idx
        df_candidate_final.to_csv(os.path.join(outdir, f"final_unlabeled_data_{higher}.csv"), index=False)
        logging.info(f"Completed transition from {lower} to {higher}.")
    logging.info("Multi-level AML workflow complete for the entire funnel.")

if __name__ == "__main__":
    run_workflow({})
