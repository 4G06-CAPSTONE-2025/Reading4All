import sys, os, torch, json, random
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration, AutoTokenizer
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

# ---- utils ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logging_utils import setup_logger, log_metrics
from utils.progress_utils import progress
from utils.safety_utils import log_health
from utils.reproducibility import set_seed

# ---------------- PATHS ----------------
BASE_DIR = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai"
MODEL_DIR = os.path.join(BASE_DIR, "model")
STAGE1_MODEL = os.path.join(MODEL_DIR, "pix2struct_publaynet_stage1_20260120_1156")
OUT_DIR = os.path.join(MODEL_DIR, "stage2_semantic")
DATASET_PATH = r"C:/Users/nawaa/Downloads/ai2d-all/ai2d"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- SEED ----------------
set_seed(42)

# ---------------- LOGGER ----------------
logger = setup_logger("stage2_semantic", OUT_DIR)
logger.info(f"Device: {DEVICE}")

# ---------------- PROCESSOR ----------------
tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL)
processor = Pix2StructProcessor.from_pretrained(STAGE1_MODEL)
max_len = 512

# ---------------- SAFE IMAGE ----------------
def safe_image(img_path: Path, target_size=(224, 224)):
    try:
        with Image.open(img_path) as img:
            img.verify()
        img = Image.open(img_path).convert("RGB")
        img.thumbnail(target_size, Image.LANCZOS)
        pad_w, pad_h = target_size[0] - img.width, target_size[1] - img.height
        return ImageOps.expand(
            img,
            (pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2),
            fill=(255,255,255)
        )
    except Exception as e:
        logger.warning(f"Bad image {img_path.name}: {e}")
        return None

# ---------------- JSON LOADER ----------------
def load_json_file(path: Path):
    if not path or not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start:end+1]) if start != -1 else None
    except Exception:
        return None

# ---------------- EMPHASIS DETECTION ----------------
def detect_emphasis(img: Image.Image):
    """
    Detect visual emphasis using red-saturation dominance.
    Returns (present: bool, region: str)
    """
    arr = np.array(img).astype(np.float32) / 255.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    red_mask = (r > 0.6) & (r > g + 0.2) & (r > b + 0.2)
    red_ratio = red_mask.mean()

    if red_ratio > 0.01:
        region = "localized" if red_ratio < 0.1 else "dominant"
        return True, region
    return False, "none"

# ---------------- SEMANTIC TARGET ----------------
def build_semantic_target(annotation_json, question_json, emphasis_info):
    lines = ["OBJECTS:"]

    if isinstance(annotation_json, dict):
        for k in annotation_json.keys():
            lines.append(f"- {k}")
    else:
        lines.append("- unknown")

    lines.append("\nRELATIONS:")
    if question_json and "questions" in question_json:
        for q in question_json["questions"].keys():
            lines.append(f"- {q}")
    else:
        lines.append("- none")

    present, region = emphasis_info
    lines.append("\nEMPHASIS:")
    lines.append(f"- red_highlight_present: {'yes' if present else 'no'}")
    lines.append(f"- emphasis_region: {region}")

    return "\n".join(lines)

# ---------------- DATASET ----------------
class AI2DSemanticDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.images = sorted((self.root / "images").glob("*.png"))
        self.ann = self.root / "annotations"
        self.q = self.root / "questions"

    def __iter__(self):
        for img_path in self.images:
            img = safe_image(img_path)
            if img is None:
                continue

            ann_json = load_json_file(self.ann / img_path.name)
            q_json = load_json_file(self.q / img_path.name)
            emphasis = detect_emphasis(img)

            yield {
                "image": img,
                "target": build_semantic_target(ann_json, q_json, emphasis)
            }

# ---------------- MODEL ----------------
model = Pix2StructForConditionalGeneration.from_pretrained(
    STAGE1_MODEL, torch_dtype=torch.float32
).to(DEVICE)

for p in model.encoder.parameters():
    p.requires_grad = False

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
scaler = GradScaler()
model.train()

dataset = list(AI2DSemanticDataset(DATASET_PATH))
logger.info(f"Loaded {len(dataset)} samples")

# ---------------- TRAINING ----------------
for step in progress(range(len(dataset)), desc="Stage 2 Semantic Training"):
    sample = dataset[step]

    labels = processor.tokenizer(
        sample["target"],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    ).input_ids

    labels[labels == tokenizer.pad_token_id] = -100
    labels = labels.to(DEVICE)

    inputs = processor(images=[sample["image"]], return_tensors="pt").to(DEVICE)

    optimizer.zero_grad()
    with autocast(device_type="cuda"):
        out = model(**inputs, labels=labels)
        loss = out.loss

    if torch.isnan(loss):
        continue

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if step % 50 == 0:
        logger.info(f"Step {step} | Loss {loss.item():.4f}")
        log_health(logger)

# ---------------- SAVE ----------------
model.save_pretrained(OUT_DIR)
processor.save_pretrained(OUT_DIR)
logger.info("STAGE 2 SEMANTIC TRAINING COMPLETE")