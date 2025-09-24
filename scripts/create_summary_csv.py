import os
import glob
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = "output_data_3d"
OUTPUT_CSV_FILE = "simulation_summary.csv"

# =============================================================================
# Main Script Logic
# =============================================================================
def create_summary_csv():
    """
    Processes H5 files to extract input parameters and summary statistics
    (Quantities of Interest) from the output fields, saving to a single CSV.
    """
    print("--- Starting Feature Extraction Process ---")
    
    h5_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.h5")))
    if not h5_files:
        print(f"Error: No H5 files found in '{INPUT_DIR}'. Please generate data first.")
        return

    print(f"Found {len(h5_files)} files to process.")
    
    all_results = []

    for h5_path in tqdm(h5_files, desc="Extracting Features"):
        try:
            with h5py.File(h5_path, 'r') as f:
                # This dictionary will become one row in our CSV
                sim_data = {}

                # 1. Extract all scalar input parameters from H5 attributes
                for key, value in f.attrs.items():
                    sim_data[key] = value

                # 2. Load the output field data
                displacement = f['displacement'][:]
                von_mises = f['von_mises_stress'][:]

                # 3. Calculate Quantities of Interest (QoI)
                # Displacement magnitude
                disp_magnitude = np.linalg.norm(displacement, axis=1)

                # --- Add QoIs to our dictionary ---
                # Max absolute displacement in each direction
                sim_data['max_abs_disp_x'] = np.max(np.abs(displacement[:, 0]))
                sim_data['max_abs_disp_y'] = np.max(np.abs(displacement[:, 1]))
                sim_data['max_abs_disp_z'] = np.max(np.abs(displacement[:, 2]))
                
                # Min (max negative) displacement in the primary bending direction
                sim_data['min_disp_y'] = np.min(displacement[:, 1])

                # Max displacement magnitude
                sim_data['max_disp_magnitude'] = np.max(disp_magnitude)

                # Max and Mean Von Mises Stress
                sim_data['max_von_mises'] = np.max(von_mises)
                sim_data['mean_von_mises'] = np.mean(von_mises)

                all_results.append(sim_data)

        except Exception as e:
            print(f"Error processing file {h5_path}: {e}")

    if not all_results:
        print("No data was processed. Exiting.")
        return

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(all_results)
    
    # Save the DataFrame to a CSV file
    df.to_csv(OUTPUT_CSV_FILE, index=False)

    print("\n--- Feature Extraction Complete ---")
    print(f"Summary data saved to '{OUTPUT_CSV_FILE}'")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

if __name__ == "__main__":
    create_summary_csv()