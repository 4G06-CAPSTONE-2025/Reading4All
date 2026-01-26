"""
Stage 2 Pix2Struct Single-Image Inference Script

This script generates a structured semantic representation for a single AI2D diagram 
using a pre-trained Stage 2 Pix2Struct model.

Workflow:
1. Set device to GPU if available, otherwise CPU.
2. Load the Pix2Struct processor and model from a specified directory.
3. Load a single image and convert it to RGB.
4. Process the image using the Pix2Struct processor to get model inputs.
5. Perform inference with the model using `generate()`.
   - Uses `flattened_patches` if available, otherwise falls back to `pixel_values`.
6. Decode the generated output to obtain the structured text.
7. Print the structured representation for the image.

Dependencies:
- torch
- transformers
- PIL (Pillow)

Example usage:
$ python iter2_stage2_validation.py
"""
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_dir = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/model/stage2_structured"
processor = Pix2StructProcessor.from_pretrained(model_dir)
model = Pix2StructForConditionalGeneration.from_pretrained(model_dir)
model.to(DEVICE)
model.eval()

img_path = r"C:/Users/nawaa/Downloads/ai2d-all/ai2d/images/97.png"
img = Image.open(img_path).convert("RGB")

inputs = processor(images=[img], return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

with torch.no_grad():
    # Use pixel_values if flattened_patches fails
    try:
        out = model.generate(flattened_patches=inputs["flattened_patches"], max_new_tokens=256)
    except KeyError:
        out = model.generate(pixel_values=inputs["pixel_values"], max_new_tokens=256)

text = processor.decode(out[0], skip_special_tokens=True)
print(text)