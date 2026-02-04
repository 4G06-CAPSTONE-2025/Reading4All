from pathlib import Path
import torch
from PIL import Image
import logging
from transformers import AutoProcessor, BlipForConditionalGeneration
import pandas as pd

MODEL_DIR = "ai/train/models"
LOGGER_PATH = "ai/logger"
IMAGE_DIR = "/Users/francinebulaclac/Desktop/capstone/ai/trainSplit/val_data2"  
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LOG_FILE = Path(LOGGER_PATH) / "log_BLIP_3epochs2.txt"
PROMPT = "Describe this physics diagram:"

# For getting draft captions - MOONBEAM
VAL_CSV = "/Users/francinebulaclac/Desktop/capstone/ai/trainSplit/val.csv"
CSV_OUT_DIR = "/Users/francinebulaclac/Desktop/Reading4All/ai/train"

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


processor = AutoProcessor.from_pretrained(MODEL_DIR, use_fast=False)
model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

image_paths = sorted([p for p in Path(IMAGE_DIR).iterdir() if p.is_file() and is_image(p)])

logger.info(f"Found {len(image_paths)} images in {IMAGE_DIR}")


# Get modified alt text from val csv
val_df = pd.read_csv(VAL_CSV)
IMAGE_PATH_COL = "Image-Path"
MOD_ALT_COL = "Modified-Alt-Text"

val_df["filename"] = val_df[IMAGE_PATH_COL].apply(lambda x: Path(x).name)
alt_text_lookup = dict(zip(val_df["filename"], val_df[MOD_ALT_COL]))

draftAltText = []


for p in image_paths:
    try:
        image = Image.open(p).convert("RGB")

        inputs = processor(images=image, text=PROMPT, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            ids = model.generate (
                **inputs,
                max_new_tokens=512,
                num_beams=6,
                length_penalty=0.8,
                no_repeat_ngram_size=4,
                repetition_penalty=1.35,
                early_stopping=True,
            )

        caption = processor.decode(ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # get matching modified alt text 
        modified_alt_text = alt_text_lookup.get(p.name, "")
        draftAltText.append({
            "image_path": str(p),
            "generated_caption": caption,
            "modified_alt_text": modified_alt_text,
        })

        logger.info(f"{p.name}: {caption}")


    except Exception as e:
        logger.exception(f"{p.name}: ERROR")

# saves to csv for moonbeam
CSV_OUT_DIR = Path(CSV_OUT_DIR)
CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)

out_csv = CSV_OUT_DIR / "BLIP_draftCaptions.csv"
out_df = pd.DataFrame(draftAltText)
out_df.to_csv(out_csv, index=False, encoding="utf-8")

logger.info(f"Saved predictions to {out_csv}")


