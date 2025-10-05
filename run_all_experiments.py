import subprocess
import sys

# Define the base command to avoid repetition
BASE_PYTHON_CMD = [sys.executable, "main_train.py", "--epochs", "300", "--batch_size", "2", "--lr", "1e-5",]

# A list of all experiments to run. Each experiment is a list of additional command-line args.
experiments = [
    # --- Phase 1: GNN Generalist Baselines (Multi-modal) ---
    #["--model_type", "gcn", "--data_dir", "data/h5_multi_modal", "--ablation_one_hot", "--experiment_suffix", "generalist"],
    #["--model_type", "gat", "--data_dir", "data/h5_multi_modal", "--ablation_one_hot", "--experiment_suffix", "generalist"],
    #["--model_type", "mpnn", "--data_dir", "data/h5_multi_modal", "--ablation_one_hot", "--experiment_suffix", "pinn_generalist",
     #"--use_pinn", "True",
     #"--pinn_weight", "1e7" ],
    #["--model_type", "transformer", "--data_dir", "data/h5_multi_modal", "--ablation_one_hot", "--experiment_suffix", "generalist"],

    # --- Phase 2: GNN Specialist Baselines (Unimodal) ---
    #["--model_type", "gcn", "--data_dir", "data/h5_raw", "--experiment_suffix", "specialist_bending_y"],
    #["--model_type", "gat", "--data_dir", "data/h5_raw", "--experiment_suffix", "specialist_bending_y"],
    ["--model_type", "mpnn", "--data_dir", "data/h5_raw", "--experiment_suffix", "pinn_specialist_bending_y",
     "--use_pinn", "True",
     "--pinn_weight", "500",
      "--hidden_size", "64" ],
    #["--model_type", "transformer", "--data_dir", "data/h5_raw", "--experiment_suffix", "specialist_bending_y"],

    # --- Phase 3: U-Net Baselines (Multi-modal Voxel Data) ---
   #["--model_type", "unet_small", "--data_dir", "data/voxelized_big", "--batch_size", "8", "--experiment_suffix", "baseline"],
    #["--model_type", "unet_small", "--data_dir", "data/voxelized_big", "--batch_size", "8", "--use_attention", "--experiment_suffix", "attention"],
]

print(f"--- Starting a sequence of {len(experiments)} experiments ---")

for i, experiment_args in enumerate(experiments):
    print(f"\n{'='*20} Experiment {i+1}/{len(experiments)} {'='*20}")
    
    # Combine the base command with the specific args for this experiment
    command = BASE_PYTHON_CMD + experiment_args
    
    print(f"Running command: {' '.join(command)}\n")
    
    try:
        # subprocess.run will wait for the command to complete.
        # check=True will raise an exception if the command returns a non-zero exit code (i.e., it fails).
        subprocess.run(command, check=True)
        print(f"--- Experiment {i+1} completed successfully. ---")
    except subprocess.CalledProcessError as e:
        print(f"!!! ERROR: Experiment {i+1} failed with exit code {e.returncode} !!!")
        print("Stopping the sequence.")
        # Stop the entire script if one experiment fails
        break
    except KeyboardInterrupt:
        print("\nUser interrupted the process. Stopping the sequence.")
        break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break

print(f"\n--- Experiment sequence finished. ---")