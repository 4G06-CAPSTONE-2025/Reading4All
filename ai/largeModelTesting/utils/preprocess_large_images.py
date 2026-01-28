'''
Custom preprocessing script to resize large images (>512 pixels in width or height) 
for more efficient training and reduced disk I/O on NF's dedicated training PC. 

Key Features:
1. Recursively scans specified source image folders for PNG images.
2. Converts all images to RGB and resizes any image exceeding MAX_SIZE while maintaining aspect ratio.
3. Saves resized images to a designated output directory in JPEG format with quality=90.
4. Preserves folder structure relative to the source directory.
5. Includes basic error handling to skip and log problematic files.

Configurable parameters:
- SRC_DIR: source directory containing raw images.
- DST_DIR: destination directory for resized images.
- MAX_SIZE: maximum allowed width or height before resizing.
'''
from PIL import Image
from pathlib import Path

SRC_DIR = Path(r"C:/Users/nawaa/Downloads/scicap_data_extracted/scicap_data")
DST_DIR = Path(r"C:/Users/nawaa/Downloads/scicap_data_preprocessed")
DST_DIR.mkdir(parents=True, exist_ok=True)

MAX_SIZE = 512  # Only resize if width or height > MAX_SIZE

# Only process image folders
image_folders = ["Scicap-No-Subfig-Img", "SciCap-Yes-Subfig-Img"]

for folder in image_folders:
    folder_path = SRC_DIR / folder
    for img_path in folder_path.rglob("*.png"):
        try:
            rel_path = img_path.relative_to(SRC_DIR)
            out_path = DST_DIR / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_SIZE:
                scale = MAX_SIZE / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.BILINEAR)

            img.save(out_path, format="JPEG", quality=90)
        except Exception as e:
            print(f"Failed {img_path}: {e}")