#!/bin/bash
#$ -q hpc@@colon
#$ -N n2_lda
#$ -pe smp 4

python3 main.py 

python3 analysis.py --xyz smol.xyz --actual_dir actual_data  --pred_dir output --methods PBE r2SCAN B3LYP CCSD --output_dir analysis_output > 1_output.txt

python3 analysis_ref.py --xyz smol.xyz --actual_dir actual_data --pred_dir output --ref_dir Gstate --methods PBE r2SCAN B3LYP CCSD --output_dir analysis_output_normalized > 2_output.txt

python3 plot_pes_forces.py --xyz smol.xyz  --merged_dir analysis_output --output_dir plot_pes --subtract_ref true  --ref_dir Gstate --atom1 0 --atom2 1 --methods PBE r2SCAN B3LYP CCSD > 3_output.txt
