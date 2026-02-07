'''
This is going to be the inference handler in the hugging face repo
'''

import io
import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# hugging face provides gpu if avail 
device = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT = "Describe this physics diagram:"

# loads from HF repo root 
processor = AutoProcessor.from_pretrained(".", use_fast=False)
model = BlipForConditionalGeneration.from_pretrained(".").to(device)
model.eval()

def predict(data):
    """
    This is the HF Inference Endpoint entry point.
    It receives raw image bytes and returns alt text.
    """

    # HF sends image bytes under "inputs"
    image_bytes = data["inputs"]
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(
        images=image,
        text=PROMPT,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=6,
            length_penalty=0.8,
            no_repeat_ngram_size=4,
            repetition_penalty=1.35,
            early_stopping=True,
        )

    caption = processor.decode(
        ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return {
        "alt_text": caption
    }
