import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

MODEL_DIR = "ai/models/TesterOneFrozen"
IMAGE_PATH = "/Users/fizasehar/GitHub/Reading4All/ai/train/val_data/image14.png"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")

inputs = processor(
    images=image,
    text="Describe this physics diagram:",
    return_tensors="pt"
)
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
print(caption)
