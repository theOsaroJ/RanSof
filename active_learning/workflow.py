#!/usr/bin/env python3
"""
Active Learning Workflow for Delta Correction with Transfer Learning

This module:
  - Loads the user-provided initial LDA energies and forces.
  - Reads smol.xyz to extract molecular coordinates and element sequences.
  - Runs an active-learning loop for each fidelity jump (e.g., LDA → PBE, etc.).
  - If a maximum iteration count is not specified for a level, the loop continues until
    the maximum uncertainty in the candidate set is below a default threshold (0.01).
  - Saves predictions (energies and flattened forces) at each fidelity level.
"""

import os
import sys
import logging
import numpy as np
from utils.io_utils import write_xyz, read_xyz_multimol

def active_learning_correction(lower_level, higher_level, X_features, gp_model_energy, gp_model_forces,
                               Y_lower_energy, Y_lower_forces, elem_sequences, acq_method, num_new_points,
                               min_threshold, max_iterations, window_size, convergence_tolerance,
                               default_uncertainty_threshold):
    logging.info(f"Starting AL for delta {higher_level} - {lower_level}")
    std_energy_window = []
    std_forces_window = []
    it = 0
    while True:
        it += 1
        if max_iterations is not None and it > max_iterations:
            logging.info(f"Reached maximum iterations ({max_iterations}) for delta {higher_level} - {lower_level}")
            break

        logging.info(f"Iteration {it} for delta {higher_level} - {lower_level}")
        candidate_X = X_features
        if gp_model_energy.X_train is not None:
            mean_delta_energy, var_delta_energy = gp_model_energy.predict(candidate_X)
        else:
            mean_delta_energy = np.zeros((candidate_X.shape[0], 1))
            var_delta_energy = np.ones((candidate_X.shape[0], 1)) * 10
        std_delta_energy = np.sqrt(var_delta_energy)
        
        if gp_model_forces.X_train is not None:
            mean_delta_forces, var_delta_forces = gp_model_forces.predict(candidate_X)
        else:
            mean_delta_forces = np.zeros((candidate_X.shape[0], 1))
            var_delta_forces = np.ones((candidate_X.shape[0], 1)) * 10
        std_delta_forces = np.sqrt(var_delta_forces)
        
        avg_std_energy = np.mean(std_delta_energy)
        avg_std_forces = np.mean(std_delta_forces)
        max_std_energy = np.max(std_delta_energy)
        max_std_forces = np.max(std_delta_forces)
        logging.info(f"Iteration {it} uncertainties: Avg Energy {avg_std_energy:.6f}, Avg Forces {avg_std_forces:.6f}, Max Energy {max_std_energy:.6f}, Max Forces {max_std_forces:.6f}")
        
        std_energy_window.append(avg_std_energy)
        std_forces_window.append(avg_std_forces)
        if len(std_energy_window) >= window_size:
            window_avg_energy = np.mean(std_energy_window[-window_size:])
            window_avg_forces = np.mean(std_forces_window[-window_size:])
            if (abs(avg_std_energy - window_avg_energy) < convergence_tolerance and
                abs(avg_std_forces - window_avg_forces) < convergence_tolerance):
                logging.info(f"Convergence achieved for delta {higher_level} - {lower_level} at iteration {it}")
                break
        
        if max_std_energy < default_uncertainty_threshold and max_std_forces < default_uncertainty_threshold:
            logging.info(f"Max uncertainty below threshold {default_uncertainty_threshold} for delta {higher_level} - {lower_level} at iteration {it}")
            break
        
        if avg_std_energy > min_threshold or avg_std_forces > min_threshold:
            if acq_method == 'std':
                idx = np.argsort(std_delta_energy.ravel())[-num_new_points:]
            else:
                idx = np.argsort(std_delta_energy.ravel())[-num_new_points:]
            new_X = candidate_X[idx]
            selected_elems = [elem_sequences[i] for i in idx]
            selected_ids = [str(i) for i in idx]
            temp_xyz = "selected_uncertain_structures.xyz"
            write_xyz(new_X, selected_ids, selected_elems, temp_xyz)
            logging.info(f"New candidate points written to {temp_xyz}")
            
            os.system(f"python simulation/generate_dft.py {temp_xyz} --save_path ./ --name_molecule m{lower_level.lower()} --xc {lower_level} --basisset 6311g --n_workers 1 --grid_level 6 --spin 0 --charge 0 --fd_delta 0.001")
            if higher_level in ["PBE", "r2SCAN", "B3LYP"]:
                os.system(f"python simulation/generate_dft.py {temp_xyz} --save_path ./ --name_molecule m{higher_level.lower()} --xc {higher_level} --basisset 6311g --n_workers 1 --grid_level 6 --spin 0 --charge 0 --fd_delta 0.001")
            elif higher_level == "CCSD":
                os.system(f"python simulation/generate_ccsd.py {temp_xyz} --save_path ./ --name_molecule mccsd --basisset 6311g --n_workers 1")
            elif higher_level == "DMRG":
                os.system(f"python simulation/generate_dmrg.py {temp_xyz} --save_path ./ --name_molecule mdmrg --basisset 6311g --num_orbitals 6 --num_electrons 6 --max_dmrg_bond_dimension 100 --max_dmrg_iterations 500 --n_workers 1 --verbose 0")
            
            lower_prefix = "m" + lower_level.lower()
            higher_prefix = "m" + higher_level.lower()
            try:
                new_Y_lower_energy = np.loadtxt(f"{lower_prefix}_energy.dat").reshape(-1, 1)
                new_Y_lower_forces = np.loadtxt(f"{lower_prefix}_force.dat").reshape(-1, 3)
                new_Y_higher_energy = np.loadtxt(f"{higher_prefix}_energy.dat").reshape(-1, 1)
                new_Y_higher_forces = np.loadtxt(f"{higher_prefix}_force.dat").reshape(-1, 3)
            except Exception as e:
                logging.error(f"Error loading outputs for {higher_level}: {e}")
                continue
            os.system("rm *dat")
            
            delta_energy_new = new_Y_higher_energy - new_Y_lower_energy
            delta_forces_new = new_Y_higher_forces - new_Y_lower_forces
            
            if gp_model_energy.X_train is not None:
                X_delta = np.vstack((gp_model_energy.X_train, new_X))
                Y_delta_energy = np.vstack((gp_model_energy.y_train, delta_energy_new))
                X_delta_forces = np.vstack((gp_model_forces.X_train, new_X))
                Y_delta_forces = np.vstack((gp_model_forces.y_train, delta_forces_new))
            else:
                X_delta = new_X
                Y_delta_energy = delta_energy_new
                X_delta_forces = new_X
                Y_delta_forces = delta_forces_new
            
            gp_model_energy.fit(X_delta, Y_delta_energy)
            gp_model_forces.fit(X_delta_forces, Y_delta_forces)
            gp_model_energy.optimize_hyperparameters()
            gp_model_forces.optimize_hyperparameters()
            logging.info(f"Updated GP models for delta {higher_level} - {lower_level} with new points: {new_X.flatten()}")
            
            X_features = np.vstack((X_features, new_X))
            Y_lower_energy = np.vstack((Y_lower_energy, new_Y_lower_energy))
            Y_lower_forces = np.vstack((Y_lower_forces, new_Y_lower_forces))
            elem_sequences.extend(selected_elems)
        else:
            logging.info(f"Uncertainty below threshold; no new points selected at iteration {it}")
    return X_features, Y_lower_energy, Y_lower_forces, gp_model_energy, gp_model_forces, elem_sequences

def run_workflow(config):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=os.path.join(config["paths"]["logs_dir"], "app.log"),
                        filemode='a')
    logging.info("Starting delta-learning workflow")
    
    funnel = config["funnel"]
    paths = config["paths"]
    al_config = config["active_learning"]
    
    # Instead of computing LDA data, load the provided initial LDA data.
    try:
        Y_base_energy = np.loadtxt(paths["initial_energy_file"]).reshape(-1, 1)
        Y_base_forces = np.loadtxt(paths["initial_forces_file"]).reshape(-1, 3)
    except Exception as e:
        logging.error(f"Error loading initial LDA data: {e}")
        sys.exit(1)
    current_energy = Y_base_energy.copy()
    current_forces = Y_base_forces.copy()
    
    # Extract features and element sequences from the input XYZ.
    from utils.io_utils import read_xyz_multimol
    _, elem_sequences = read_xyz_multimol(paths["input_xyz"])
    # In production, extract real molecular descriptors; here we use a dummy linear space.
    X_features = np.linspace(0, 10, Y_base_energy.shape[0]).reshape(-1, 1)
    
    from models.custom_gp import GaussianProcess
    gp_models_energy = {}
    gp_models_forces = {}
    for level in funnel[1:]:
        gp_models_energy[level] = GaussianProcess(length_scale=1e5, noise=1e-2, alpha_rq=10, batch_size=200)
        gp_models_forces[level] = GaussianProcess(length_scale=1e5, noise=1e-2, alpha_rq=10, batch_size=200)
    
    # Loop over fidelity levels starting from the second (e.g., PBE, etc.).
    for i in range(1, len(funnel)):
        lower_level = funnel[i-1]
        higher_level = funnel[i]
        logging.info(f"Processing delta correction: {higher_level} - {lower_level}")
        max_iters = al_config.get("al_iterations_per_level", {}).get(higher_level, None)
        (X_features, Y_base_energy, Y_base_forces, 
         gp_models_energy[higher_level], gp_models_forces[higher_level],
         elem_sequences) = active_learning_correction(
            lower_level, higher_level, X_features, gp_models_energy[higher_level],
            gp_models_forces[higher_level], Y_base_energy, Y_base_forces, elem_sequences,
            acq_method=al_config["acquisition_method"],
            num_new_points=al_config["num_new_points"],
            min_threshold=al_config["min_std_threshold"],
            max_iterations=max_iters,
            window_size=al_config["window_size"],
            convergence_tolerance=al_config["convergence_tolerance"],
            default_uncertainty_threshold=al_config["default_uncertainty_threshold"]
        )
        delta_energy_pred, _ = gp_models_energy[higher_level].predict(X_features)
        delta_forces_pred, _ = gp_models_forces[higher_level].predict(X_features)
        current_energy += delta_energy_pred
        current_forces += delta_forces_pred
        logging.info(f"Updated overall prediction with delta from {higher_level}")
        
        output_dir = paths["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        energy_file = os.path.join(output_dir, f"predicted_energy_{higher_level}.dat")
        np.savetxt(energy_file, current_energy)
        flattened_forces = np.array([row.flatten() for row in current_forces])
        forces_file = os.path.join(output_dir, f"predicted_forces_{higher_level}.dat")
        np.savetxt(forces_file, flattened_forces)
        logging.info(f"Saved predictions for {higher_level}: energies and flattened forces.")
    
    os.makedirs(paths["output_dir"], exist_ok=True)
    np.savetxt(os.path.join(paths["output_dir"], "final_energy_prediction.dat"), current_energy)
    np.savetxt(os.path.join(paths["output_dir"], "final_forces_prediction.dat"), 
               np.array([row.flatten() for row in current_forces]))
    logging.info("Delta-learning workflow completed. Final predictions saved.")
