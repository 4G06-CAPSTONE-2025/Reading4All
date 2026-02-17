import os
import pandas as pd

'''
This script validates that each image path in the combinedData.csv under /ai/annotations/ is in 
the folder of images in your local system
'''


# path to your CSV file
CSV_PATH = "ai/annotations/combinedData.csv"
# col name in CSV that contains image paths 
IMAGE_COLUMN = "Image-Path"
# path to the local folder containing images
# ***change this path to the folder path of the images in ur local system** 
IMAGE_FOLDER = "/Users/francinebulaclac/Desktop/capstone/ai/data"

# allowed image extensions (adjust if needed)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}



df = pd.read_csv(CSV_PATH)
csv_image_paths = df[IMAGE_COLUMN].astype(str).tolist()

missing_images = []

for img_path in csv_image_paths:
    image_name = os.path.basename(img_path)
    local_image_path = os.path.join(IMAGE_FOLDER, image_name)
    # check if file doesnt exist
    if not os.path.isfile(local_image_path):
        missing_images.append(image_name)


# Gets images that exist locally but are not in the CSV
local_images = {
    f for f in os.listdir(IMAGE_FOLDER)
    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
}
csv_images_set = {os.path.basename(p) for p in csv_image_paths}
extra_images = local_images - csv_images_set


print("Results:")

if missing_images:
    print(f"\nMissing images ({len(missing_images)}):")
    for img in missing_images:
        print(f"  - {img}")
else:
    print("\nAll CSV images exist locally.")

if extra_images:
    print(f"\nExtra local images not in CSV ({len(extra_images)}):")
    for img in sorted(extra_images):
        print(f"  - {img}")
else:
    print("\nNo extra images found.")

