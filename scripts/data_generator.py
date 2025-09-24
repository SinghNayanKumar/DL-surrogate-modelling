import os
import subprocess
import numpy as np
from scipy.stats import qmc

# =============================================================================
# 1. DEFINE THE PARAMETER SPACE
# =============================================================================
# Define the number of simulations to run
NUM_SAMPLES = 10 # Start with a small number for testing, e.g., 10. Then increase to 1000+.

# Define the ranges for each continuous parameter [min, max]
param_space = {
    "beam_length": [280.0, 320.0],
    "flange_width": [90.0, 110.0],
    "flange_thickness": [13.0, 17.0],
    "web_thickness": [8.0, 12.0],
    "beam_depth": [140.0, 160.0],
    "fillet_radius": [10.0, 14.0],
    "youngs_modulus": [190e3, 210e3],  # Steel variations
    "poissons_ratio": [0.28, 0.32],
    "force_magnitude": [1000.0, 2000.0],
}

# Define categorical parameters
load_types = ["bending_y", "bending_x", "torsion"]
load_distributions = ["uniform", "linear_y"]

# =============================================================================
# 2. SAMPLE THE PARAMETER SPACE USING LATIN HYPERCUBE SAMPLING (LHS)
# =============================================================================
keys = list(param_space.keys())
l_bounds = [param_space[key][0] for key in keys]
u_bounds = [param_space[key][1] for key in keys]

sampler = qmc.LatinHypercube(d=len(keys))
samples = sampler.random(n=NUM_SAMPLES)
scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

# =============================================================================
# 3. RUN THE SIMULATIONS IN A LOOP
# =============================================================================
print(f"--- Starting data generation for {NUM_SAMPLES} simulations ---")

for i in range(NUM_SAMPLES):
    print(f"\n{'='*20} Running Simulation ID: {i}/{NUM_SAMPLES-1} {'='*20}")

    params = {keys[j]: scaled_samples[i, j] for j in range(len(keys))}
    chosen_load_type = np.random.choice(load_types)
    chosen_distribution = np.random.choice(load_distributions)

    command = ["python", "scripts/run_ibeam_parametric.py", f"--sim_id={i}"]
    for key, value in params.items():
        command.append(f"--{key}={value}")
    command.append(f"--load_type={chosen_load_type}")
    command.append(f"--load_distribution={chosen_distribution}")
    
    print(f"Executing command: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"--- Simulation {i} completed successfully. ---")
    except subprocess.CalledProcessError as e:
        print(f"!!! ERROR: Simulation {i} failed! !!!")
        print(f"Return Code: {e.returncode}")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        # Uncomment the next line to stop the whole process if one simulation fails
        # break

print(f"\n--- Data generation complete. {NUM_SAMPLES} simulations attempted. ---")