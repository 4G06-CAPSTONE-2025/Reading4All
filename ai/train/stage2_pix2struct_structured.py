import os, torch, sys, json
from pathlib import Path
from PIL import Image
import numpy as np

from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from torch.optim import AdamW

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from utils.reproducibility import set_seed, save_config
from utils.logging_utils import setup_logger, log_metrics
from utils.progress_utils import progress
from utils.safety_utils import log_health, cooldown

BASE_DIR = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai"
MODEL_DIR = os.path.join(BASE_DIR, "model")

CFG = {
    "model": os.path.join(MODEL_DIR, "stage1_visual", "epoch_2"),
    "dataset_path": r"C:\Users\nawaa\Downloads\ai2d-all\ai2d",
    "epochs": 1,
    "max_steps": 7000,
    "lr": 3e-5,
    "seed": 42
}
OUT = os.path.join(MODEL_DIR, "stage2_structured")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set seed and setup logging
set_seed(CFG["seed"])
save_config(CFG, OUT)
logger = setup_logger("stage2", OUT)

# Custom AI2D Dataset Loader for PNG annotations
class AI2DDataset:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        
        # Load all original image files from images/ folder
        self.images_dir = self.dataset_path / "images"
        self.image_files = []
        
        # Check for common image extensions
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        for ext in extensions:
            self.image_files.extend(list(self.images_dir.glob(ext)))
        
        # Sort by numeric id if possible
        try:
            self.image_files.sort(key=lambda x: int(x.stem))
        except:
            self.image_files.sort()
        
        logger.info(f"Found {len(self.image_files)} original images in dataset")
        
        # Load annotation images if available
        self.annotations_dir = self.dataset_path / "annotations"
        self.has_annotations = self.annotations_dir.exists()
        
        if self.has_annotations:
            # Get annotation files (should be PNGs with same names as images)
            self.annotation_files = []
            for ext in extensions:
                self.annotation_files.extend(list(self.annotations_dir.glob(ext)))
            
            # Create mapping from image_id to annotation path
            self.annotation_map = {}
            for ann_file in self.annotation_files:
                img_id = ann_file.stem
                self.annotation_map[img_id] = ann_file
            
            logger.info(f"Found {len(self.annotation_files)} annotation images")
        
        # Load categories if available
        self.categories_file = self.dataset_path / "categories.json"
        self.categories = {}
        if self.categories_file.exists():
            try:
                with open(self.categories_file, 'r') as f:
                    self.categories = json.load(f)
                logger.info(f"Loaded categories for {len(self.categories)} images")
            except:
                logger.warning("Could not load categories.json")
        
        # Load questions if available
        self.questions_dir = self.dataset_path / "questions"
        self.questions = {}
        if self.questions_dir.exists():
            # Try to load questions.json if it exists
            questions_file = self.questions_dir / "questions.json"
            if questions_file.exists():
                try:
                    with open(questions_file, 'r') as f:
                        self.questions = json.load(f)
                    logger.info(f"Loaded questions for {len(self.questions)} images")
                except:
                    logger.warning("Could not load questions.json")
            else:
                # Check for individual question files
                question_files = list(self.questions_dir.glob("*.json"))
                for q_file in question_files:
                    try:
                        with open(q_file, 'r') as f:
                            q_data = json.load(f)
                            img_id = q_file.stem
                            self.questions[img_id] = q_data
                    except:
                        pass
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_id = img_path.stem
        
        try:
            # Load original image
            image = Image.open(img_path).convert("RGB")
            
            # Load annotation image if available
            annotation_image = None
            if self.has_annotations and img_id in self.annotation_map:
                ann_path = self.annotation_map[img_id]
                try:
                    annotation_image = Image.open(ann_path).convert("RGB")
                except:
                    logger.warning(f"Could not load annotation for {img_id}")
            
            # Get category if available
            category = self.categories.get(img_id, "unknown")
            
            # Get question if available
            question_text = ""
            if img_id in self.questions:
                q_data = self.questions[img_id]
                # Handle different question formats
                if isinstance(q_data, dict):
                    if "question" in q_data:
                        question_text = q_data["question"]
                    elif "questions" in q_data and isinstance(q_data["questions"], list) and len(q_data["questions"]) > 0:
                        question_text = q_data["questions"][0]
                elif isinstance(q_data, list) and len(q_data) > 0:
                    question_text = q_data[0]
                elif isinstance(q_data, str):
                    question_text = q_data
            
            return {
                "image": image,
                "annotation_image": annotation_image,
                "image_id": img_id,
                "category": category,
                "question": question_text
            }
        except Exception as e:
            logger.warning(f"Error loading image {img_id}: {e}")
            # Return a fallback item
            return {
                "image": Image.new('RGB', (512, 512), color='white'),
                "annotation_image": None,
                "image_id": img_id,
                "category": "error",
                "question": ""
            }
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

# Initialize dataset
dataset_path = CFG["dataset_path"]
if not os.path.exists(dataset_path):
    logger.error(f"Dataset not found at: {dataset_path}")
    logger.error("Please update the 'dataset_path' in CFG to point to your AI2D dataset location")
    exit(1)

dataset = AI2DDataset(dataset_path)

# Initialize model and processor
logger.info(f"Loading model from: {CFG['model']}")
processor = Pix2StructProcessor.from_pretrained(CFG["model"])
model = Pix2StructForConditionalGeneration.from_pretrained(
    CFG["model"], torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

model.gradient_checkpointing_enable()
optimizer = AdamW(model.parameters(), lr=CFG["lr"])

logger.info(f"Starting training on {DEVICE}")
logger.info(f"Dataset size: {len(dataset)} images")
logger.info(f"Max steps: {CFG['max_steps']}")

# Training loop
step = 0
model.train()
loss_history = []

for epoch in range(CFG["epochs"]):
    logger.info(f"Starting epoch {epoch + 1}/{CFG['epochs']}")
    
    for sample in progress(dataset, "Structured learning"):
        if step >= CFG["max_steps"]:
            logger.info(f"Reached max steps ({CFG['max_steps']})")
            break
        
        try:
            # Create prompt - use question if available, otherwise use default
            if sample["question"] and len(sample["question"]) > 5:
                prompt = sample["question"]
            else:
                # Create informative prompt based on category
                category = sample["category"]
                if category != "unknown":
                    prompt = f"Describe this {category} diagram and its components"
                else:
                    prompt = "describe the diagram components and their relationships"
            
            # Prepare inputs - use both original image and annotation if available
            # For now, we'll just use the original image
            inputs = processor(
                images=sample["image"],
                text=prompt,
                return_tensors="pt"
            ).to(DEVICE)
            
            # Forward pass
            out = model(**inputs, labels=inputs["input_ids"])
            loss = out.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            step += 1
            loss_history.append(loss.item())
            
            # Logging and checkpointing
            if step % 10 == 0:
                avg_loss = sum(loss_history[-10:]) / min(len(loss_history), 10)
                logger.info(f"Step {step}: Loss = {loss.item():.4f}, Avg Loss = {avg_loss:.4f}")
            
            if step % 250 == 0:
                log_health(logger)
                log_metrics({
                    "step": step, 
                    "loss": loss.item(), 
                    "avg_loss": avg_loss,
                    "image_id": sample["image_id"],
                    "category": sample["category"]
                }, OUT)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cooldown()
            
            if step % 1000 == 0:
                # Save intermediate checkpoint
                checkpoint_dir = f"{OUT}/step_{step}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                processor.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint at step {step}")
        
        except Exception as e:
            logger.error(f"Error at step {step} (image {sample.get('image_id', 'unknown')}): {e}")
            # Skip this sample but continue training
            continue
        
        if step >= CFG["max_steps"]:
            break

# Save final model
logger.info("Saving final model...")
model.save_pretrained(OUT)
processor.save_pretrained(OUT)
logger.info("STAGE 2 COMPLETE")