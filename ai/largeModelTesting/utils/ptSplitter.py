'''
Description:
Utility script to split large preprocessed PyTorch `.pt` files into smaller chunks
for more efficient loading and training. This is especially useful when working 
with large datasets (e.g., Pix2Struct SciCap preprocessed patches) to reduce memory
overhead and improve batch processing performance.

Key Features:
1. Configurable maximum samples per output file (`MAX_SAMPLES_PER_FILE`).
2. Splits each `.pt` file in the input directory into multiple smaller `.pt` files.
3. Preserves original file names with `_partN` suffix for easy identification.
4. Saves all split files into a dedicated output directory (`OUTPUT_DIR`).
5. Prints progress and summary for each file.

Usage:
    1. Set `BASE_DIR` to the folder containing original `.pt` files.
    2. Set `OUTPUT_DIR` to the folder where split files should be saved.
    3. Adjust `MAX_SAMPLES_PER_FILE` as needed.
    4. Run the script to generate smaller `.pt` chunks.
'''


import os
import torch
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = r"C:/Users/nawaa/Downloads/scicap_data_preprocessed/Preprocessed-Patches"
OUTPUT_DIR = r"C:/Users/nawaa/Downloads/scicap_data_preprocessed/Preprocessed-Patches-Split"
MAX_SAMPLES_PER_FILE = 500  # split each file into chunks of ~500 samples

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# FUNCTION TO SPLIT A SINGLE PT FILE
# ----------------------------
def split_pt_file(file_path, output_dir, max_samples=500):
    data = torch.load(file_path, map_location="cpu")
    total_samples = len(data)
    
    print(f"Splitting {file_path} ({total_samples} samples)...")
    
    # split into chunks
    for i in range(0, total_samples, max_samples):
        chunk = data[i:i + max_samples]
        chunk_filename = f"{Path(file_path).stem}_part{i//max_samples + 1}.pt"
        chunk_path = os.path.join(output_dir, chunk_filename)
        torch.save(chunk, chunk_path)
        print(f"  Saved chunk: {chunk_path} ({len(chunk)} samples)")

# ----------------------------
# PROCESS ALL FILES
# ----------------------------
all_files = [f for f in os.listdir(BASE_DIR) if f.endswith(".pt")]

for f in all_files:
    file_path = os.path.join(BASE_DIR, f)
    split_pt_file(file_path, OUTPUT_DIR, MAX_SAMPLES_PER_FILE)

print("\n All files split successfully!")
