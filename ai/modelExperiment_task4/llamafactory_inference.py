from pathlib import Path
import torch
from PIL import Image
import logging

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_DIR    = "meta-llama/Llama-3.2-11B-Vision-Instruct"  # HF ID or local path
ADAPTER_DIR  = None          # path to LoRA adapter folder, or None for base model
TEMPLATE     = "mllama"      # chat template: llama3 | qwen | intern_vl | llava | mllama
IMAGE_DIR    = "C:/Users/nawaa/OneDrive/Desktop/capstoneAI/trainSplit/val_data" #custom for nf's pc
LOG_FILE     = Path("C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/logger") / "llamafactory_infer.txt"
PROMPT       = "Describe this image in detail."
MAX_NEW_TOKENS = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Other model / template combos:
#   "Qwen/Qwen2-VL-7B-Instruct"         template="qwen2_vl"
#   "llava-hf/llava-1.5-7b-hf"          template="llava"
#   "OpenGVLab/InternVL2-8B"             template="intern_vl"
#   "meta-llama/Llama-3.2-11B-Vision-Instruct"  template="mllama"
# ─────────────────────────────────────────────────────────────────────────────


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("llamafactory_infer")
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
logger.info(f"RUN | MODEL={MODEL_DIR} | ADAPTER={ADAPTER_DIR} | IMAGE_DIR={IMAGE_DIR} | DEVICE={DEVICE}")
logger.info("=" * 90)

try:
    from llamafactory.chat import ChatModel
except ImportError:
    logger.error("llamafactory not installed. Run: pip install llamafactory")
    raise

logger.info("Loading model via LLaMA-Factory ChatModel …")
chat_args = dict(
    model_name_or_path=MODEL_DIR,
    adapter_name_or_path=ADAPTER_DIR,
    finetuning_type="lora" if ADAPTER_DIR else "full",
    template=TEMPLATE,
    infer_dtype="float16" if DEVICE == "cuda" else "float32",
)
chat_model = ChatModel(chat_args)
logger.info("Model loaded.")

image_paths = sorted([p for p in Path(IMAGE_DIR).iterdir() if p.is_file() and is_image(p)])
logger.info(f"Found {len(image_paths)} images in {IMAGE_DIR}")

for p in image_paths:
    try:
        image = Image.open(p).convert("RGB")

        messages = [
            {"role": "user", "content": PROMPT}
        ]

        logger.info(f"{p.name}: starting generation")

        response = chat_model.chat(
            messages,
            images=[image], 
            max_new_tokens=MAX_NEW_TOKENS
        )

        logger.info(f"{p.name}: generation finished")

        caption = response[0].response_text
        logger.info(f"{p.name}: {caption}")

    except Exception:
        logger.exception(f"{p.name}: ERROR")