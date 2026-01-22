import os
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

MODEL_DIR = "ai/models/TesterOneFrozen"
IMAGE_DIR = "/Users/fizasehar/GitHub/Reading4All/ai/data/tester_1/val_data"  
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PROMPT = "Describe this physics diagram:"

def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}

processor = AutoProcessor.from_pretrained(MODEL_DIR, use_fast=False)
model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

image_paths = sorted([p for p in Path(IMAGE_DIR).iterdir() if p.is_file() and is_image(p)])

print(f"Found {len(image_paths)} images in {IMAGE_DIR}\n")

for p in image_paths:
    try:
        image = Image.open(p).convert("RGB")

        inputs = processor(images=image, text=PROMPT, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=6,
                length_penalty=0.8,
                no_repeat_ngram_size=4,
                repetition_penalty=1.35,
                early_stopping=True,
            )

        caption = processor.decode(ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(f"{p.name}: {caption}")

    except Exception as e:
        print(f"{p.name}: ERROR -> {e}")
