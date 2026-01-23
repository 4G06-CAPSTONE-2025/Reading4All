#TODO: write config file for this script and remove hardcoding for scicap training

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image_path = "inference//images//autoencoder.png"
image = Image.open(image_path).convert("RGB")

inputs = processor(image, return_tensors="pt")
output_ids = model.generate(**inputs, max_length=120)

caption = processor.decode(output_ids[0], skip_special_tokens=True)
print(caption)