import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

TESTER = 1
CSV_PATH = "/ai/annotations/annotated_physics_data(Sheet1).csv"
local_root = "/Users/fizasehar/Downloads/"
IMAGE_ROOT = Path(local_root)

TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

OUT_ROOT = Path("ai/data") / f"tester_{TESTER}"

TRAIN_DIR = OUT_ROOT / "train_data"
VAL_DIR = OUT_ROOT / "val_data"

if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

df["Tester-ID"] = pd.to_numeric(df["Tester-ID"], errors="coerce")
df = df[df["Tester-ID"] == TESTER].copy()

if df.empty:
    raise SystemExit(
        f"ERROR: 0 rows after filtering Tester-ID == {TESTER}. "
        f"Unique Tester-ID values: {df['Tester-ID'].dropna().unique()[:20]}"
    )

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
        if not src_path.is_absolute():
            src_path = IMAGE_ROOT / src_path
        if not src_path.exists():
            raise FileNotFoundError(f"Image not found: {src_path}")
        dst_path = target_dir / src_path.name
        shutil.copy(src_path, dst_path)
        row = row.copy()
        row["Image-Path"] = str(dst_path)
        new_rows.append(row)
    return pd.DataFrame(new_rows)

train_df = copyImages(train_df, TRAIN_DIR)
val_df = copyImages(val_df, VAL_DIR)

train_df.to_csv(OUT_ROOT / "train.csv", index=False)
val_df.to_csv(OUT_ROOT / "val.csv", index=False)
