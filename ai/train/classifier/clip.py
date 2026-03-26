from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import pandas as pd
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_DIR = os.path.join(BASE_DIR, "Images")
CSV_PATH = os.path.join(BASE_DIR, "annotations", "combinedData.csv")

df = pd.read_csv(CSV_PATH)

labels = df["Category-of-Image"].unique().tolist()

optimizer = torch.optim.AdamW(clip_model.parameters(), lr=5e-6)

clip_model.train()

for epoch in range(5):
    print("Epoch:", epoch + 1)

    for _, row in df.iterrows():
        image_name = row["Image-Path"]
        label = row["Category-of-Image"]

        image_path = os.path.join(IMAGE_DIR, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
        except:
            continue

        inputs = clip_processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image

        target = torch.tensor([labels.index(label)]).to(device)
        loss = torch.nn.functional.cross_entropy(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# save trained model
MODEL_DIR = os.path.join(BASE_DIR, "models", "clip-physics")
os.makedirs(MODEL_DIR, exist_ok=True)

clip_model.save_pretrained(MODEL_DIR)
clip_processor.save_pretrained(MODEL_DIR)

print("Training complete. Model saved to:", MODEL_DIR)