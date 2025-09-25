import os
import glob
import h5py
import numpy as np

# --- Configuration ---
DATA_DIRECTORY = 'data/h5_raw'

def main():
    """
    This script iterates through all .h5 files and checks if the 'topology'
    dataset is valid for tetrahedral meshes.
    """
    print(f"--- Starting Data Integrity Check in '{DATA_DIRECTORY}' ---")
    
    all_h5_files = sorted(glob.glob(os.path.join(DATA_DIRECTORY, '*.h5')))
    
    if not all_h5_files:
        print("No .h5 files found. Exiting.")
        return
        
    found_issues = False
    for h5_path in all_h5_files:
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'topology' not in f:
                    print(f"[ERROR] File: {os.path.basename(h5_path)} -> Missing 'topology' dataset.")
                    found_issues = True
                    continue
                
                topology_arr = f['topology'][:]
                
                # The critical check: Is the total number of elements a multiple of 4?
                if topology_arr.size % 4 != 0:
                    print(f"[ERROR] File: {os.path.basename(h5_path)} -> Corrupted topology data!")
                    print(f"        Shape: {topology_arr.shape}, Total Size: {topology_arr.size}")
                    print(f"        Size is not divisible by 4, so it cannot be a pure tetrahedral mesh.")
                    found_issues = True

        except Exception as e:
            print(f"[ERROR] File: {os.path.basename(h5_path)} -> Could not be read. Error: {e}")
            found_issues = True
            
    print("-" * 50)
    if not found_issues:
        print("✅ Success! All files seem to have valid tetrahedral topology.")
    else:
        print("❌ Found issues in one or more files. Please inspect or remove the files listed above.")
    print("-" * 50)

if __name__ == '__main__':
    main()