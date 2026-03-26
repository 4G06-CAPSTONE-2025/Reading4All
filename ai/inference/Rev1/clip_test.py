from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
BLIP_PATH = "/Users/fizasehar/GitHub/Reading4All/ai/models/blip-physics"
CLIP_PATH = "/Users/fizasehar/GitHub/Reading4All/ai/models/clip-physics"

blip_processor = BlipProcessor.from_pretrained(BLIP_PATH)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_PATH)

clip_processor = CLIPProcessor.from_pretrained(CLIP_PATH)
clip_model = CLIPModel.from_pretrained(CLIP_PATH)

labels = [
    "a graph or chart",
    "a waves or fields diagram",
    "a mechanics forces and motion diagram",
    "a general physics diagram"
]

def classify_image(image):
    inputs = clip_processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return labels[probs.argmax().item()]

def generate_prompt(label):
    if "graph" in label:
        return "Describe the graph including axes, variables, and trends."
    
    elif "waves" in label or "fields" in label:
        return "Describe the wave or field diagram and relationships."

    elif "mechanics" in label:
        return "Explain the forces, motion, and physical setup."

    else:
        return "Describe the physics diagram clearly."
    
def generate_caption(image, prompt):
    inputs = blip_processor(image, prompt, return_tensors="pt")
    output_ids = blip_model.generate(**inputs, max_length=120)
    return blip_processor.decode(output_ids[0], skip_special_tokens=True)

image_path = "ai/Images/page0643_img002.png"
image = Image.open(image_path).convert("RGB")

label = classify_image(image)
prompt = generate_prompt(label)
caption = generate_caption(image, prompt)

final_caption = f"{label}. {caption}"

print("Image:", image_path)
print("Predicted Type:", label)
print("Prompt Used:", prompt)
print("Final Caption:", final_caption)