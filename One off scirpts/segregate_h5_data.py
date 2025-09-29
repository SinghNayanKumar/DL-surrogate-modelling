import os
import glob
import h5py
import shutil
from tqdm import tqdm

def segregate_data(source_dir, base_dest_dir, unimodal_type='bending_y'):
    """
    Reads H5 files from a source directory, checks their 'load_type' attribute,
    and copies them into new 'multi_modal' and 'unimodal' subdirectories.

    Args:
        source_dir (str): Path to the directory containing all raw .h5 files.
        base_dest_dir (str): Path to the base directory where new folders will be created.
        unimodal_type (str): The 'load_type' to be considered for the unimodal dataset.
    """
    # 1. Define destination paths
    multi_modal_dest = os.path.join(base_dest_dir, 'h5_multi_modal')
    unimodal_dest = os.path.join(base_dest_dir, f'h5_unimodal_{unimodal_type}')
    
    # 2. Create destination directories if they don't exist
    os.makedirs(multi_modal_dest, exist_ok=True)
    os.makedirs(unimodal_dest, exist_ok=True)
    print(f"Destination directories created/ensured at:\n- {multi_modal_dest}\n- {unimodal_dest}")

    # 3. Find all source files
    source_files = glob.glob(os.path.join(source_dir, '*.h5'))
    if not source_files:
        print(f"Error: No .h5 files found in '{source_dir}'. Please check the path.")
        return

    print(f"\nFound {len(source_files)} total .h5 files to process...")
    
    unimodal_count = 0
    # 4. Loop through every file
    for file_path in tqdm(source_files, desc="Segregating H5 files"):
        try:
            # 5a. Always copy the file to the multi-modal directory
            shutil.copy(file_path, multi_modal_dest)
            
            # 5b. Read the attribute to decide if it belongs in the unimodal set
            with h5py.File(file_path, 'r') as f:
                load_type = f.attrs.get('load_type')
            
            # 5c. If it matches, copy to the unimodal directory as well
            if load_type == unimodal_type:
                shutil.copy(file_path, unimodal_dest)
                unimodal_count += 1

        except Exception as e:
            print(f"Could not process file {os.path.basename(file_path)}: {e}")

    print("\n--- Segregation Complete ---")
    print(f"Total files copied to multi-modal directory: {len(source_files)}")
    print(f"Total files copied to unimodal directory: {unimodal_count}")
    print("----------------------------")

if __name__ == '__main__':
    # --- CONFIGURE YOUR PATHS HERE ---
    # This is the directory where your current 1500+ mixed .h5 files are.
    SOURCE_H5_DIRECTORY = "data/h5_raw" 
    
    # This is the base 'data' directory. The script will create subfolders inside it.
    DESTINATION_BASE_DIRECTORY = "data"
    
    segregate_data(SOURCE_H5_DIRECTORY, DESTINATION_BASE_DIRECTORY)