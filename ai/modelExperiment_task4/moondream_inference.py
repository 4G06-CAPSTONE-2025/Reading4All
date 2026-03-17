from pathlib import Path
import torch
from PIL import Image
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_DIR   = "vikhyatk/moondream2"          # or local path to fine-tuned model
IMAGE_DIR   = "C:/Users/nawaa/OneDrive/Desktop/capstoneAI/trainSplit/val_data" #custom for nf's pc
LOG_FILE    = Path("C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/logger") / "moondream_infer.txt"
PROMPT      = "Describe this image:"
REVISION    = "2025-01-09"                   # pin stable HF revision
MAX_NEW_TOKENS = 256
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("moondream_infer")
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

logger.info("Loading Moondream2 model and tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, revision=REVISION, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    revision=REVISION,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)
model.eval()
logger.info("Model loaded.")

image_paths = sorted([p for p in Path(IMAGE_DIR).iterdir() if p.is_file() and is_image(p)])
logger.info(f"Found {len(image_paths)} images in {IMAGE_DIR}")

for p in image_paths:
    try:
        image = Image.open(p).convert("RGB")

        logger.info(f"{p.name}: encoding image …")
        enc_image = model.encode_image(image)

        logger.info(f"{p.name}: generating caption …")
        caption = model.answer_question(enc_image, PROMPT, tokenizer)

        logger.info(f"{p.name}: {caption}")

    except Exception:
        logger.exception(f"{p.name}: ERROR")