from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import pandas as pd
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

labels = [
    "graph",
    "waves and fields diagram",
    "mechanics diagram",
    "crystal lattice diagram",
    "general physics diagram"
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_DIR = os.path.join(BASE_DIR, "Images")

def find_file(name):
    for f in os.listdir(IMAGE_DIR):
        if name in f:
            return f
    return None

def classify_image(image_path):
    real_name = find_file(image_path)
    if real_name is None:
        return "error"

    full_path = os.path.join(IMAGE_DIR, real_name)

    try:
        image = Image.open(full_path).convert("RGB")
    except:
        return "error"

    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

    return labels[probs.argmax().item()]

csv_path = os.path.join(BASE_DIR, "annotations", "combinedData.csv")
df = pd.read_csv(csv_path)

files = sorted([f for f in os.listdir(IMAGE_DIR) if not f.startswith('.')])
df["Image-Path"] = files[:len(df)]

df["Predicted-Type"] = df["Image-Path"].apply(classify_image)
df.to_csv(os.path.join(BASE_DIR, "classified_output.csv"), index=False)

print(df[["Image-Path", "Category-of-Image", "Predicted-Type"]].head())