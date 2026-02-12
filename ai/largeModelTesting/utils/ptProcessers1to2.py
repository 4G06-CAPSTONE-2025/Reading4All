'''
Description:
Utility script to inspect and fix preprocessed PyTorch `.pt` dataset files 
for training with AI2D / Pix2Struct pipelines. Handles common issues such as:

1. Ensuring `flattened_patches` are 2D tensors with consistent shape.
2. Converting `labels` to 1D tensors with proper dtype (`torch.long`).
3. Creating or correcting `attention_mask` to match patch length.
4. Preserving other metadata in the file where possible.
5. Handling corrupted or malformed files by creating default fallback data.
6. Batch progress reporting and debugging outputs for first few files.
7. Testing a few fixed files to ensure correctness before training.

The script outputs all fixed `.pt` files into a dedicated folder (`preprocessed_fixed_v2`) 
and prints a summary of fixed files vs errors. This ensures all dataset files are 
training-ready and avoids runtime errors during model training.
'''

import torch
from pathlib import Path
import numpy as np

# Fix preprocessing data
data_dir = Path(r"C:/Users/nawaa/Downloads/ai2d-all/preprocessed")
fixed_dir = data_dir.parent / "preprocessed_fixed_v2"

# Create directories
fixed_dir.mkdir(exist_ok=True)

pt_files = list(data_dir.glob("*.pt"))
print(f"Found {len(pt_files)} .pt files")

fixed_count = 0
error_count = 0

for file_path in pt_files:
    try:
        # Load with weights_only=False to avoid security errors
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        
        # Create new data dictionary
        new_data = {}
        
        # Handle flattened_patches
        if "flattened_patches" in data:
            patches = data["flattened_patches"]
            
            # Convert to tensor if it's not already
            if not torch.is_tensor(patches):
                if isinstance(patches, np.ndarray):
                    patches = torch.from_numpy(patches)
                elif isinstance(patches, list):
                    patches = torch.tensor(patches)
                else:
                    print(f"Warning: {file_path.name} has non-tensor patches: {type(patches)}")
                    patches = torch.randn(512, 768)  # Default shape
            
            # Fix dimensions
            if patches.dim() == 3:
                if patches.shape[0] == 1:
                    patches = patches.squeeze(0)
                else:
                    patches = patches[0]  # Take first in batch
            
            # Ensure 2D
            if patches.dim() != 2:
                print(f"Warning: {file_path.name} has patches dim {patches.dim()}, reshaping")
                if patches.dim() == 1:
                    patches = patches.unsqueeze(0)
                elif patches.dim() > 2:
                    patches = patches.view(-1, patches.shape[-1])
            
            new_data["flattened_patches"] = patches
        
        # Handle labels
        if "labels" in data:
            labels = data["labels"]
            
            # Convert to tensor if needed
            if not torch.is_tensor(labels):
                if isinstance(labels, np.ndarray):
                    labels = torch.from_numpy(labels)
                elif isinstance(labels, list):
                    labels = torch.tensor(labels, dtype=torch.long)
                else:
                    print(f"Warning: {file_path.name} has non-tensor labels: {type(labels)}")
                    labels = torch.full((512,), -100, dtype=torch.long)
            
            # Fix dimensions
            if labels.dim() == 2:
                if labels.shape[0] == 1:
                    labels = labels.squeeze(0)
                else:
                    labels = labels[0]
            
            # Ensure 1D
            if labels.dim() != 1:
                print(f"Warning: {file_path.name} has labels dim {labels.dim()}, flattening")
                labels = labels.flatten()
            
            new_data["labels"] = labels
        
        # Handle attention_mask (optional)
        if "attention_mask" in data:
            mask = data["attention_mask"]
            
            if mask is not None:
                # Convert to tensor if needed
                if not torch.is_tensor(mask):
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask)
                    elif isinstance(mask, list):
                        mask = torch.tensor(mask, dtype=torch.long)
                    else:
                        mask = torch.ones(new_data["flattened_patches"].shape[0], dtype=torch.long)
                
                # Fix dimensions
                if mask.dim() == 2:
                    if mask.shape[0] == 1:
                        mask = mask.squeeze(0)
                    else:
                        mask = mask[0]
                
                # Ensure 1D
                if mask.dim() != 1:
                    mask = mask.flatten()
                
                # Ensure correct length
                if len(mask) != new_data["flattened_patches"].shape[0]:
                    mask = torch.ones(new_data["flattened_patches"].shape[0], dtype=torch.long)
                
                new_data["attention_mask"] = mask
            else:
                new_data["attention_mask"] = torch.ones(new_data["flattened_patches"].shape[0], dtype=torch.long)
        else:
            # Create default attention mask
            new_data["attention_mask"] = torch.ones(new_data["flattened_patches"].shape[0], dtype=torch.long)
        
        # Copy any other keys
        for key, value in data.items():
            if key not in ["flattened_patches", "labels", "attention_mask"]:
                if torch.is_tensor(value) or isinstance(value, (int, float, str, list, dict)):
                    new_data[key] = value
        
        # Save fixed file
        torch.save(new_data, fixed_dir / file_path.name)
        fixed_count += 1
        
        if fixed_count % 100 == 0:
            print(f"Fixed {fixed_count}/{len(pt_files)} files...")
            
        # Debug first few files
        if fixed_count <= 3:
            print(f"\n=== Fixed sample {fixed_count} ===")
            print(f"File: {file_path.name}")
            for key in ["flattened_patches", "labels", "attention_mask"]:
                if key in new_data:
                    val = new_data[key]
                    if torch.is_tensor(val):
                        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                    else:
                        print(f"  {key}: type={type(val)}")
    
    except Exception as e:
        error_count += 1
        print(f"Error fixing {file_path.name}: {e}")
        import traceback
        traceback.print_exc()
        
        # Try a last resort fix
        try:
            # Create default data
            default_data = {
                "flattened_patches": torch.randn(512, 768),
                "labels": torch.full((512,), -100, dtype=torch.long),
                "attention_mask": torch.ones(512, dtype=torch.long)
            }
            torch.save(default_data, fixed_dir / file_path.name)
            fixed_count += 1
            print(f"  -> Created default data for {file_path.name}")
        except:
            pass

print(f"\nFixed {fixed_count} files, {error_count} errors")
print(f"Fixed data saved to: {fixed_dir}")

# Test the fixed files
print("\nTesting fixed files...")
test_files = list(fixed_dir.glob("*.pt"))[:5]
for i, test_file in enumerate(test_files):
    try:
        data = torch.load(test_file, map_location="cpu")
        print(f"Test {i+1}: {test_file.name} - OK")
        print(f"  flattened_patches: {data['flattened_patches'].shape}")
        print(f"  labels: {data['labels'].shape}")
        print(f"  attention_mask: {data['attention_mask'].shape}")
    except Exception as e:
        print(f"Test {i+1}: {test_file.name} - ERROR: {e}")

# Update your training script to use the fixed directory
print("\nUpdate your training script to use:")
print(f"PREPROCESS_DIR = r\"{fixed_dir}\"")