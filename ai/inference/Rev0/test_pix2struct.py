from pathlib import Path
import torch
from PIL import Image
import logging
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

MODEL_DIR = "C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/model/iter3_pix2struct_3epochs" #custom for NF'c PC
LOGGER_PATH = "logger"
IMAGE_DIR = "C:/Users/nawaa/OneDrive/Desktop/capstoneAI/trainSplit/val_data"  #custom for NF'c PC
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE = Path(LOGGER_PATH) / "log_3epochs.txt"
PROMPT = "Describe this physics diagram:"

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("blip_eval_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

logger = setup_logger(LOG_FILE)


def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}

logger.info("=" * 90)
logger.info(f"RUN | MODEL_DIR={MODEL_DIR} | IMAGE_DIR={IMAGE_DIR} | DEVICE={DEVICE}")
logger.info("=" * 90)


processor = Pix2StructProcessor.from_pretrained(MODEL_DIR)
model = Pix2StructForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

image_paths = sorted([p for p in Path(IMAGE_DIR).iterdir() if p.is_file() and is_image(p)])

logger.info(f"Found {len(image_paths)} images in {IMAGE_DIR}")

for p in image_paths:
    try:
        image = Image.open(p).convert("RGB")

        inputs = processor(
            images=image,
            text=PROMPT,
            return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        logger.info(f"{p.name}: starting generation")
        with torch.no_grad():
            ids = model.generate (
                **inputs,
                max_new_tokens=512,
                num_beams=2,
                length_penalty=0.8,
                no_repeat_ngram_size=4,
                repetition_penalty=1.35,
                early_stopping=True,
            )
        logger.info(f"{p.name}: generation finished")

        caption = processor.decode(ids[0], skip_special_tokens=True)
        logger.info(f"{p.name}: {caption}")


    except Exception as e:
        logger.exception(f"{p.name}: ERROR")
