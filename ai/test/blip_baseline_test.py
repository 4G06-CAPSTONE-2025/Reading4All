import json, torch, os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

MODEL_NAME = "Salesforce/blip-image-captioning-base"  # change to pretrained model pls
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Example
print(generate_caption("path/to/sample_image.jpg"))