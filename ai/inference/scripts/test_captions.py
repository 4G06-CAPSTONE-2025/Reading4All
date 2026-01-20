import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

MODEL_DIR = "ai/train"
IMAGE_PATH = "/Users/francinebulaclac/Desktop/Reading4All/ai/train/val_data/image14.png"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")

inputs = processor(
    images=image,
    text="Describe this physics diagram:",
    return_tensors="pt"
).to(DEVICE)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_length=128
    )

caption = processor.decode(output_ids[0], skip_special_tokens=True)
print("Caption:")
print(caption)
