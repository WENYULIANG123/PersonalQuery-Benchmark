
import os
import subprocess
import time

def run_script(script_path, desc):
    print(f"\n{'='*20} Starting: {desc} {'='*20}")
    start_time = time.time()
    
    # We use the sbatch wrapper script to run our scripts on the cluster as required
    # But for a local main orchestrator, we can use subprocess.run if we assume 
    # the environment is already set up or wrap each call.
    # To follow User rules strictly, we wrap python calls.
    wrapper = "/home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py"
    cmd = [
        "python3", wrapper, 
        f"source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && conda activate /home/wlia0047/ar57_scratch/wenyu/stark && python -u {script_path}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    end_time = time.time()
    print(result.stdout)
    if result.stderr:
        print(f"Errors/Warnings:\n{result.stderr}")
        
    print(f"{'='*20} Finished: {desc} (Time: {end_time - start_time:.2f}s) {'='*20}\n")
    return result.returncode == 0

def main():
    base_path = "/home/wlia0047/ar57/wenyu/stark/code/preprocessing"
    
    steps = [
        (os.path.join(base_path, "extract_direct_attributes.py"), "Extracting Direct Attributes (Brand, Category, Price)"),
        (os.path.join(base_path, "extract_style_values.py"), "Extracting Style Attributes (Color, Size from Reviews)"),
        (os.path.join(base_path, "extract_details_values.py"), "Extracting Technical Details (Material, Dimensions)"),
        (os.path.join(base_path, "map_colors.py"), "Mapping Colors to Base Colors and Cleaning Noise"),
        (os.path.join(base_path, "map_sizes.py"), "Normalizing Sizes and Cleaning Noise"),
        (os.path.join(base_path, "map_styles.py"), "Normalizing Styles and Cleaning Noise"),
        (os.path.join(base_path, "map_other_attributes.py"), "Normalizing Remaining Attributes (Length, Pattern, Material, etc.)")
    ]
    
    print("🚀 Starting Preprocessing Pipeline Orchestration...")
    
    for script, desc in steps:
        success = run_script(script, desc)
        if not success:
            print(f"❌ Pipeline failed at step: {desc}. Terminating.")
            return

    print("✅ Preprocessing Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
