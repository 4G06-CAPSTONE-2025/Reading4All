import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

CSV_PATH = "ai/annotations/annotated_physics_data(Sheet1).csv"
local_prefix = "/Users/francinebulaclac/Desktop/capstone/"
relative_folder = "ai/data/"
IMAGE_ROOT = Path(local_prefix+relative_folder)
TRAIN_DIR = Path("ai/train/train_data")
VAL_DIR = Path("ai/train/val_data")
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# Create output folders
TRAIN_DIR.mkdir(exist_ok=True)
VAL_DIR.mkdir(exist_ok=True)

# # Load CSV
# df = pd.read_csv(CSV_PATH)

# # Train / validation split
# train_df, val_df = train_test_split(
#     df,
#     test_size=1 - TRAIN_SPLIT,
#     random_state=RANDOM_SEED,
#     shuffle=True
# )

# def copy_images_and_update_paths(split_df, target_dir):
#     new_rows = []

#     for _, row in split_df.iterrows():
#         src_path = Path(row["Image-Path"])

#         # If image path is relative, resolve it from IMAGE_ROOT
#         if not src_path.is_absolute():
#             src_path = IMAGE_ROOT / src_path

#         if not src_path.exists():
#             raise FileNotFoundError(f"Image not found: {src_path}")

#         dst_path = target_dir / src_path.name
#         shutil.copy(src_path, dst_path)

#         # Update Image-Path to new relative location
#         row = row.copy()
#         row["Image-Path"] = str(dst_path)
#         new_rows.append(row)

#     return pd.DataFrame(new_rows)

# # Copy images + update paths
# train_df = copy_images_and_update_paths(train_df, TRAIN_DIR)
# val_df = copy_images_and_update_paths(val_df, VAL_DIR)

# # Save CSVs
# train_df.to_csv("train.csv", index=False)
# val_df.to_csv("val.csv", index=False)

# print("✅ Dataset split complete")
# print(f"Train samples: {len(train_df)}")
# print(f"Validation samples: {len(val_df)}")
