'''
This is going to be the inference handler in the hugging face repo
'''
'''
This is going to be the inference handler in the hugging face repo
'''
import io
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

class EndpointHandler:
    def __init__(self, path=""):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForVision2Seq.from_pretrained(path).to(self.device)
        self.model.eval()

        self.prompt = "Describe this physics diagram:"

    def __call__(self, data):
        # hugging Face sends raw image bytes
        image_bytes = data["inputs"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = self.processor(
            images=image,
            text=self.prompt,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=6,
                length_penalty=0.8,
                no_repeat_ngram_size=4,
                repetition_penalty=1.35,
                early_stopping=True,
            )

        caption = self.processor.decode(
            ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return {"alt_text": caption}
