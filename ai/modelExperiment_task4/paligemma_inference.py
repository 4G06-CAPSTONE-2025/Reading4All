from pathlib import Path
import torch
from PIL import Image
import logging
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_DIR      = "google/paligemma-3b-pt-224"  # or local path to fine-tuned model
IMAGE_DIR      = "C:/Users/nawaa/OneDrive/Desktop/capstoneAI/trainSplit/val_data" #custom for nf's pc
LOG_FILE       = Path("C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/logger") / "paligemma_infer.txt"
PROMPT         = "caption en"                  # PaliGemma task prefix
MAX_NEW_TOKENS = 256
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# Other useful PROMPT options:
#   "Describe this image:"
#   "answer en What is shown in this image?"
#   "caption en"   ← standard captioning prefix
# ─────────────────────────────────────────────────────────────────────────────


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("paligemma_infer")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}


logger = setup_logger(LOG_FILE)

logger.info("=" * 90)
logger.info(f"RUN | MODEL={MODEL_DIR} | IMAGE_DIR={IMAGE_DIR} | DEVICE={DEVICE}")
logger.info("=" * 90)

logger.info("Loading PaliGemma processor and model …")
processor = PaliGemmaProcessor.from_pretrained(MODEL_DIR)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)
model.eval()
logger.info("Model loaded.")

image_paths = sorted([p for p in Path(IMAGE_DIR).iterdir() if p.is_file() and is_image(p)])
logger.info(f"Found {len(image_paths)} images in {IMAGE_DIR}")

for p in image_paths:
    try:
        image = Image.open(p).convert("RGB")

        inputs = processor(
            text=PROMPT,
            images=image,
            return_tensors="pt",
        ).to(DEVICE)

        logger.info(f"{p.name}: starting generation")
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=2,
                length_penalty=0.8,
                no_repeat_ngram_size=4,
                repetition_penalty=1.35,
                early_stopping=True,
            )
        logger.info(f"{p.name}: generation finished")

        caption = processor.decode(ids[0], skip_special_tokens=True)
        logger.info(f"{p.name}: {caption}")

    except Exception:
        logger.exception(f"{p.name}: ERROR")