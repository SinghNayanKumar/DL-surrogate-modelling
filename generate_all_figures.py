import os
import subprocess
import sys

# --- Configuration ---
VISUALIZATION_DATA_DIR = "results/for_figures"
FIGURE_OUTPUT_DIR = "results/paper_figures"
SAMPLE_INDEX_FOR_COMPARISON = 0
DEFORMATION_SCALE = 50.0 # Exaggeration factor for deformation plots

def run_command(command):
    python_executable = sys.executable
    full_command = [python_executable] + command
    print(f"\nExecuting command:\n{' '.join(full_command)}")
    try:
        output_dir = "an unknown directory"
        if "--output_dir" in command:
            try:
                output_dir_index = command.index("--output_dir") + 1
                output_dir = command[output_dir_index]
            except IndexError:
                pass
        
        # Use capture_output=True to get stdout/stderr
        result = subprocess.run(full_command, check=True, capture_output=True, text=True, timeout=300)
        
        print(f"Successfully generated visuals in: {output_dir}")
        # Optional: Print stdout if there's useful info
        if result.stdout:
            print("--- Script Output ---")
            print(result.stdout)
            
    except subprocess.CalledProcessError as e:
        # Print the full error for debugging
        print(f"\n--- ERROR: The visualization script failed. ---")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        print("-" * 40)

def main():
    print("="*60); print("      AUTOMATED FIGURE GENERATION SCRIPT"); print("="*60)
    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

    # --- Figure 5: GNN vs. U-Net ---
    print("\n--- Generating visuals for Figure 5: GNN vs. U-Net ---")
    output_dir_fig5 = os.path.join(FIGURE_OUTPUT_DIR, "figure_5_architectural_showdown")
    
    gnn_file = os.path.join(VISUALIZATION_DATA_DIR, f"Transformer_-_HIGH_SIGNAL_Specialist_testsample_{SAMPLE_INDEX_FOR_COMPARISON}.npz")
    
    # --- THIS IS THE FIX ---
    # The filename is corrected to match the file that actually exists on your computer.
    # I have removed the incorrect '+' from the string.
    unet_file = os.path.join(VISUALIZATION_DATA_DIR, f"U-Net_Attn-High-Res_-_HIGH_SIGNAL_Specialist_testsample_{SAMPLE_INDEX_FOR_COMPARISON}.npz")
    # --- END FIX ---
    
    print(f"Checking for GNN file:   '{gnn_file}'")
    print(f"File exists? --> {os.path.exists(gnn_file)}")
    print(f"Checking for U-Net file: '{unet_file}'")
    print(f"File exists? --> {os.path.exists(unet_file)}")

    if os.path.exists(gnn_file) and os.path.exists(unet_file):
        command = ["create_figure_visuals.py", "--files", gnn_file, unet_file, "--names", "GNN", "U-Net",
                   "--output_dir", output_dir_fig5, "--warp_factor", str(DEFORMATION_SCALE)]
        run_command(command)
    else:
        print("!!! SKIPPING Figure 5 logic: One or both files were not found.")

    # --- Figure 7: MPNN vs. MPNN-PINN ---
    print("\n--- Generating visuals for Figure 7: PINN Advantage ---")
    output_dir_fig7 = os.path.join(FIGURE_OUTPUT_DIR, "figure_7_pinn_advantage")
    
    mpnn_file = os.path.join(VISUALIZATION_DATA_DIR, f"MPNN_-_HIGH_SIGNAL_Specialist_testsample_{SAMPLE_INDEX_FOR_COMPARISON}.npz")
    pinn_file = os.path.join(VISUALIZATION_DATA_DIR, f"MPNN_PINN-_HIGH_SIGNAL_Specialist_testsample_{SAMPLE_INDEX_FOR_COMPARISON}.npz")

    print(f"Checking for MPNN file: '{mpnn_file}'")
    print(f"File exists? --> {os.path.exists(mpnn_file)}")
    print(f"Checking for PINN file: '{pinn_file}'")
    print(f"File exists? --> {os.path.exists(pinn_file)}")

    if os.path.exists(mpnn_file) and os.path.exists(pinn_file):
        command = ["create_figure_visuals.py", "--files", mpnn_file, pinn_file, "--names", "MPNN", "MPNN-PINN",
                   "--output_dir", output_dir_fig7, "--warp_factor", str(DEFORMATION_SCALE)]
        run_command(command)
    else:
        print("!!! SKIPPING Figure 7 logic: One or both files were not found.")
        
    # --- Figure 6: Generalist Versatility (Using the Torsion sample) ---
    print("\n--- Generating visuals for Figure 6: Generalist Versatility ---")
    output_dir_fig6 = os.path.join(FIGURE_OUTPUT_DIR, "figure_6_generalist_versatility")
    generalist_file = os.path.join(VISUALIZATION_DATA_DIR, "MPNN_-_PINN_LOW_SIGNAL_Generalist_testsample_13.npz")
    
    if os.path.exists(generalist_file):
         command = ["create_figure_visuals.py", "--files", generalist_file, "--names", "MPNN-PINN_Generalist",
                    "--output_dir", output_dir_fig6, "--warp_factor", str(DEFORMATION_SCALE)]
         run_command(command)
    else:
        print(f"!!! SKIPPING Figure 6: File not found at '{generalist_file}'")
                
    print("\n" + "="*60); print("Figure generation script finished."); print("="*60)

if __name__ == '__main__':
    main()