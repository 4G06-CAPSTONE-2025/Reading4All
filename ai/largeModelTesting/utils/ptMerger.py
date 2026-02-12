'''
Description:
Utility script to merge multiple preprocessed PyTorch `.pt` batch files into larger
files for more efficient dataset management and training. This is useful when 
datasets are initially saved in many small batches and a larger consolidated 
batch size is preferred for faster loading during training.

Key Features:
1. Merges a configurable number of batch files (`merge_factor`) into a single `.pt` file.
2. Preserves the dataset split (e.g., 'train', 'val', 'test') in the merged file names.
3. Outputs merged files to a dedicated directory (`output_dir`) for organized storage.
4. Prints progress and the number of samples in each merged file.

Usage:
    1. Set `preproc_dir` to the folder containing original batch `.pt` files.
    2. Set `output_dir` for merged output files.
    3. Adjust `split` and `merge_factor` as needed.
    4. Run the script to create merged batch files for downstream training.
'''


import torch, os

preproc_dir = r"C:/Users/nawaa/Downloads/scicap_data_preprocessed/Preprocessed-Patches"
output_dir = r"C:/Users/nawaa/Downloads/scicap_data_preprocessed/Preprocessed-Patches-Merged"
os.makedirs(output_dir, exist_ok=True)

split = "val"
batch_files = sorted([f for f in os.listdir(preproc_dir) if f.startswith(f"{split}_processed_batch")])

merge_factor = 10  # Merge 10 batches per file
for i in range(0, len(batch_files), merge_factor):
    merged_batches = []
    for f in batch_files[i:i+merge_factor]:
        data = torch.load(os.path.join(preproc_dir, f))
        merged_batches.extend(data)
    out_file = os.path.join(output_dir, f"{split}_merged_batch_{i//merge_factor}.pt")
    torch.save(merged_batches, out_file)
    print(f"Saved {out_file} with {len(merged_batches)} samples")
