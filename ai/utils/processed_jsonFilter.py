'''
Custom script used during NF's iter#2

Description:
This script filters and prepares the SciCap dataset for model training or evaluation. 
It processes JSON caption files and preprocessed images to create cleaned, ready-to-use 
datasets for the 'train', 'val', and 'test' splits. Specifically, it:

1. Searches preprocessed image folders to confirm the existence and validity of images.
2. Reads caption JSONs corresponding to each split and extracts figure IDs and captions.
3. Filters out entries with missing images or invalid captions (e.g., too short).
4. Truncates captions to a maximum of 500 characters.
5. Saves the filtered dataset as JSON files in a dedicated 'Filtered-Captions' folder.

The resulting JSONs contain only valid image-caption pairs, ensuring a high-quality dataset 
for downstream tasks such as image-to-text modeling or alt text generation.
'''

import os
import json
from pathlib import Path
from PIL import Image

# ================== CONFIG ==================
# Preprocessed image folders (resized / cleaned)
PREPROCESS_BASE = Path(r"C:/Users/nawaa/Downloads/scicap_data_preprocessed")
PREPROCESS_FOLDERS = [
    PREPROCESS_BASE / "SciCap-Yes-Subfig-Img",
    PREPROCESS_BASE / "Scicap-No-Subfig-Img"
]

# Captions are in the original folder
CAP_DIR = Path(r"C:/Users/nawaa/Downloads/scicap_data_extracted/scicap_data/SciCap-Caption-All")

# Where to save filtered JSONs
OUTPUT_DIR = PREPROCESS_BASE / "Filtered-Captions"
OUTPUT_DIR.mkdir(exist_ok=True)

# Dataset splits
SPLITS = ["train", "val", "test"]

# ================== HELPER ==================
def find_preprocessed_image(img_id, split):
    """
    Search only in preprocessed folders for a given split
    Returns first valid image path or None
    """
    for folder in PREPROCESS_FOLDERS:
        candidate = folder / split / f"{img_id}.png"
        if candidate.exists():
            try:
                with Image.open(candidate) as im:
                    im.verify()  # raises if truncated
                return candidate
            except Exception:
                continue
    return None

# ================== MAIN FILTER ==================
for split in SPLITS:
    # Auto-detect caption folder for this split
    possible_folders = [f for f in CAP_DIR.iterdir() if f.is_dir() and split.lower() in f.name.lower()]
    if not possible_folders:
        print(f"[WARNING] No caption folder found for split '{split}' in {CAP_DIR}")
        continue
    cap_split_dir = possible_folders[0]

    filtered_records = []

    for jf in os.listdir(cap_split_dir):
        if not jf.endswith(".json"):
            continue
        json_path = cap_split_dir / jf
        try:
            with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            item = data[0] if isinstance(data, list) else data

            # Extract image ID
            img_id = os.path.splitext(item.get("figure-ID", item.get("figure_id", "")))[0].lower().strip()
            if not img_id:
                continue

            # Extract caption
            caption = str(item.get("0-originally-extracted", item.get("caption", ""))).strip()
            if len(caption) < 5:
                continue

            # Only include image if it exists and is valid in preprocessed folders
            img_path = find_preprocessed_image(img_id, split.lower())
            if img_path:
                filtered_records.append({
                    "image_path": str(img_path),
                    "caption": caption[:500]  # truncate to max length
                })
            else:
                print(f"[SKIP] No valid preprocessed image for {img_id}, skipping.")

        except Exception as e:
            print(f"[ERROR] Failed to process {json_path}: {e}")
            continue

    # Save filtered JSON for this split
    out_file = OUTPUT_DIR / f"{split}_filtered.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(filtered_records, f, indent=2)

    print(f"[INFO] Saved {len(filtered_records)} valid samples for split '{split}' -> {out_file}")