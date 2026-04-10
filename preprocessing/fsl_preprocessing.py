import subprocess
import os

def skull_strip(input_path, output_path):
    subprocess.run(["bet", input_path, output_path, "-f", "0.5", "-g", "0"], check=True)
    print(f"Done: {os.path.basename(output_path)}")

def preprocess_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".nii.gz") or filename.endswith(".nii"):
            in_path = os.path.join(input_folder, filename)
            out_name = filename.replace(".nii.gz", "_brain.nii.gz").replace(".nii", "_brain.nii.gz")
            out_path = os.path.join(output_folder, out_name)

            if os.path.exists(out_path):
                print(f"Skipping {filename} — already processed")
                continue

            print(f"Processing: {filename}")
            skull_strip(in_path, out_path)