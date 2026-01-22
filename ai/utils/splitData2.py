import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

TESTER = 1    # CHANGE THIS 
CSV_PATH = "ai/annotations/annotated_physics_data(Sheet1).csv"
local_root = "/Users/fizasehar/Downloads/"        # CHANGE THIS
IMAGE_ROOT = Path(local_root)
TRAIN_DIR = Path("ai/train/train_data")
VAL_DIR = Path("ai/train/val_data")
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

'''
Directery will look like: 
>ai
    >train
        >train_data **(splitData2.py makes this)**
        >val_data  **(splitData2.py makes this)**
        base_train.py
        blip_baseline.py
        train.csv **(splitData2.py makes this)**
        val.csv  **(splitData2.py makes this)**
'''

# deletes folders if they already exist and remakes 
for d in [TRAIN_DIR, VAL_DIR]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

# load csv
df = pd.read_csv(CSV_PATH)


# filters by tester - COMMENT IF WANT TO TEST BOTH 
df = df[df["Tester-ID"] == TESTER].copy()


# split train and val images
train_df, val_df = train_test_split(
    df,
    test_size=1 - TRAIN_SPLIT,
    random_state=RANDOM_SEED,
    shuffle=True
)

def copyImages(split_df, target_dir):
    new_rows = []

    for _, row in split_df.iterrows():
        src_path = Path(row["Image-Path"])

        # If image path is relative, resolve it from IMAGE_ROOT
        if not src_path.is_absolute():
            src_path = IMAGE_ROOT / src_path

        if not src_path.exists():
            raise FileNotFoundError(f"Image not found: {src_path}")

        dst_path = target_dir / src_path.name
        shutil.copy(src_path, dst_path)

        # updates the image path to new location 
        row = row.copy()
        row["Image-Path"] = str(dst_path)
        new_rows.append(row)

    return pd.DataFrame(new_rows)

# copy images
train_df = copyImages(train_df, TRAIN_DIR)
val_df = copyImages(val_df, VAL_DIR)

# savs csvs to get labels 
train_df.to_csv("ai/train/train.csv", index=False)
val_df.to_csv("ai/train/val.csv", index=False)
