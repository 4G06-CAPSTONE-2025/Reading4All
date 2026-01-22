"""
Stage 1: Train Pix2Struct on SciCap for scientific diagram understanding
Uses the correct SciCap folder structure
"""

import os
import sys
import json
import torch
import time
import random
import multiprocessing
import psutil
import re  # Added for cleaning text
from datetime import datetime
from pathlib import Path
from PIL import Image

# ============================================================
# DEBUG SETTINGS
# ============================================================
# Enable detailed CUDA error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128"
# Disable warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')
# ============================================================


# Set multiprocessing start method for Windows
multiprocessing.set_start_method('spawn', force=True)

# Add utils folder to path
SCRIPT_DIR = Path(__file__).parent.absolute()
UTILS_DIR = SCRIPT_DIR.parent / "utils"
sys.path.append(str(UTILS_DIR))

from transformers import (
    Pix2StructProcessor,
    Pix2StructForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

# Import utilities
try:
    from logging_utils import setup_logger, log_metrics
    from reproducibility import set_seed, save_config
    from progress_utils import progress
    from safety_utils import log_health, cooldown
except ImportError as e:
    print(f"Error importing utilities: {e}")
    print(f"Make sure utils folder exists at: {UTILS_DIR}")
    sys.exit(1)

# ============================================================
# CONFIGURATION - UPDATED WITH DEBUG SETTINGS
# ============================================================
class Config:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    AI_DIR = os.path.dirname(SCRIPT_DIR)
    
    CONFIG_PATH = os.path.join(SCRIPT_DIR, "paths_config.json")
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            paths_config = json.load(f)
        BASE = paths_config["base_data_dir"]
        IMG_NO = os.path.join(BASE, paths_config["img_no_dir"])
        IMG_YES = os.path.join(BASE, paths_config["img_yes_dir"])
        CAP_ALL = os.path.join(BASE, paths_config["captions_dir"])
    else:
        BASE = r"C:/Users/nawaa/Downloads/scicap_data_extracted/scicap_data"
        IMG_NO = os.path.join(BASE, "Scicap-No-Subfig-Img")
        IMG_YES = os.path.join(BASE, "SciCap-Yes-Subfig-Img")
        CAP_ALL = os.path.join(BASE, "SciCap-Caption-All")
    
    MODEL_BASE_DIR = os.path.join(AI_DIR, "model")
    os.makedirs(MODEL_BASE_DIR, exist_ok=True)
    
    BASE_MODEL = "google/pix2struct-base"
    
    # DEBUG SETTINGS - START SMALL
    MAX_TRAIN_SAMPLES = 500  # Start small for debugging
    MAX_VAL_SAMPLES = 100    # Start small for debugging
    
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 4
    LEARNING_RATE = 3e-5
    EPOCHS = 1  # Only 1 epoch for debugging
    WARMUP_STEPS = 100
    MAX_LENGTH = 512
    
    FREEZE_EARLY_LAYERS = True
    NUM_FROZEN_BLOCKS = 2
    
    MODEL_PREFIX = "pix2struct_scicap_stage1"
    LOGGING_STEPS = 10
    
    WORKERS = 2  # Conservative for debugging

# ============================================================
# INITIALIZATION
# ============================================================
config = Config()
set_seed(42)

MODEL_NAME = f"{config.MODEL_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M')}"
OUTPUT_DIR = os.path.join(config.MODEL_BASE_DIR, MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = setup_logger("stage1_scicap", OUTPUT_DIR)

logger.info("=" * 60)
logger.info("STAGE 1: SCIENTIFIC DIAGRAM CAPTIONING - DEBUG MODE")
logger.info("=" * 60)
logger.info(f"DEBUG: Small dataset ({config.MAX_TRAIN_SAMPLES} train, {config.MAX_VAL_SAMPLES} val)")
logger.info(f"DEBUG: 1 epoch only to find CUDA error")
logger.info("=" * 60)
logger.info(f"SciCap structure:")
logger.info(f"  Captions: {config.CAP_ALL}/{{train,val,test}}/{{JSON files}}")
logger.info(f"  No-subfig: {config.IMG_NO}/{{train,val,test}}/{{PNG files}}")
logger.info(f"  Yes-subfig: {config.IMG_YES}/train/{{PNG files}} (only train)")
logger.info("=" * 60)

# Verify folders
for folder_name, folder_path in [
    ("Base", config.BASE),
    ("Img No", config.IMG_NO),
    ("Img Yes", config.IMG_YES),
    ("Cap All", config.CAP_ALL)
]:
    exists = os.path.exists(folder_path)
    logger.info(f"  {folder_name}: {'[OK]' if exists else '[MISSING]'} {folder_path}")
    if not exists:
        logger.error(f"Missing folder: {folder_path}")
        sys.exit(1)

# ============================================================
# DATASET CLASS
# ============================================================
class SciCapDataset(torch.utils.data.Dataset):
    def __init__(self, records, image_size=512):
        self.records = records
        self.image_size = image_size
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {rec['image_path']}: {e}")
            img = Image.new("RGB", (self.image_size, self.image_size), color=(255, 255, 255))
        
        return {
            "image": img,
            "caption": rec["caption"],
            "image_path": rec["image_path"]
        }

# ============================================================
# FIXED SCI CAP LOADER - PROPERLY HANDLES JSON STRUCTURE
# ============================================================
def debug_json_structure(json_path, logger):
    """Debug the structure of a JSON file"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"\n=== DEBUG JSON STRUCTURE: {os.path.basename(json_path)} ===")
        
        def print_structure(obj, indent=0, max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return " ..."
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        logger.info("  " * indent + f"{key}: {type(value).__name__}")
                        print_structure(value, indent + 1, max_depth, current_depth + 1)
                    else:
                        # Show first 50 chars of string values
                        if isinstance(value, str):
                            display = value[:50] + ("..." if len(value) > 50 else "")
                        else:
                            display = str(value)
                        logger.info("  " * indent + f"{key}: {display}")
            elif isinstance(obj, list):
                logger.info("  " * indent + f"List with {len(obj)} items")
                if obj and len(obj) > 0:
                    print_structure(obj[0], indent + 1, max_depth, current_depth + 1)
        
        print_structure(data)
        
        # Specifically check for caption fields
        logger.info("\nCaption field check:")
        def find_caption_fields(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if "caption" in key.lower():
                        logger.info(f"  Found caption field at {path}.{key}")
                        if isinstance(value, str):
                            logger.info(f"    Value: {value[:100]}...")
                    if isinstance(value, (dict, list)):
                        find_caption_fields(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj[:2]):  # Check first 2 items only
                    find_caption_fields(item, f"{path}[{i}]")
        
        find_caption_fields(data)
        
    except Exception as e:
        logger.error(f"Error debugging JSON: {e}")

def load_split(split, max_samples=None):
    """Load SciCap data for a specific split (train/val/test) - FIXED VERSION"""
    # Step 1: Find caption folder (CAP_ALL/split/)
    caption_folder = os.path.join(config.CAP_ALL, split.lower())
    if not os.path.exists(caption_folder):
        # Try capitalized
        caption_folder = os.path.join(config.CAP_ALL, split.capitalize())
        if not os.path.exists(caption_folder):
            raise FileNotFoundError(f"No caption folder found for split '{split}' at: {os.path.join(config.CAP_ALL, split)}")
    
    logger.info(f"Loading {split} split from: {caption_folder}")
    logger.info(f"  Caption files in: {caption_folder}")
    logger.info(f"  Looking for images in:")
    logger.info(f"    - {config.IMG_NO}/{split.lower()}/")
    if split.lower() == "train":
        logger.info(f"    - {config.IMG_YES}/train/ (yes-subfig images)")
    
    records = []
    corrupted_images = []
    skipped = 0
    json_errors = 0
    
    # Get JSON files
    json_files = [f for f in os.listdir(caption_folder) if f.endswith(".json")]
    if not json_files:
        logger.error(f"No JSON files found in {caption_folder}")
        return []
    
    logger.info(f"Found {len(json_files)} JSON files")
    
    # DEBUG: Inspect first JSON file structure
    if json_files and len(json_files) > 0:
        debug_json_structure(os.path.join(caption_folder, json_files[0]), logger)
    
    # Shuffle for random sampling
    random.shuffle(json_files)
    
    for jf in progress(json_files, desc=f"Loading {split}"):
        if max_samples and len(records) >= max_samples:
            logger.info(f"Reached max_samples ({max_samples}), stopping...")
            break
            
        json_path = os.path.join(caption_folder, jf)
        try:
            with open(json_path, "r", encoding="utf-8", errors='ignore') as f:
                data = json.load(f)
            
            # Handle both list and dict formats
            if isinstance(data, list):
                # Some files might be lists of objects
                items = data
            elif isinstance(data, dict):
                items = [data]
            else:
                logger.warning(f"Unexpected JSON format in {jf}: {type(data)}")
                skipped += 1
                continue
            
            for item in items:
                if max_samples and len(records) >= max_samples:
                    break
                
                # Get image ID
                img_id = item.get("figure-ID", "")
                if not img_id:
                    # Try alternative field names
                    img_id = item.get("figure_id", "")
                    img_id = item.get("figureID", "")
                
                if not img_id:
                    logger.debug(f"No figure-ID found in {jf}, skipping")
                    skipped += 1
                    continue
                
                # Clean image ID
                img_id = os.path.splitext(img_id)[0].lower().strip()
                if not img_id:
                    skipped += 1
                    continue
                
                # Get caption - MULTIPLE SOURCES TO TRY
                caption = ""
                
                # Try "0-originally-extracted" (most reliable)
                if "0-originally-extracted" in item:
                    caption = item["0-originally-extracted"]
                    # Remove "Figure X: " prefix if present
                    if caption.lower().startswith("figure"):
                        colon_pos = caption.find(":")
                        if colon_pos != -1:
                            caption = caption[colon_pos + 1:].strip()
                
                # If not found, try "2-normalized" -> "2-1-basic-num" -> "caption"
                elif not caption and "2-normalized" in item:
                    normalized = item["2-normalized"]
                    if isinstance(normalized, dict) and "2-1-basic-num" in normalized:
                        basic_num = normalized["2-1-basic-num"]
                        if isinstance(basic_num, dict) and "caption" in basic_num:
                            caption = basic_num["caption"]
                
                # If still not found, try top-level "caption"
                elif not caption and "caption" in item:
                    caption = item["caption"]
                
                # Clean caption
                caption = str(caption).strip()
                
                # Remove control characters and invalid Unicode
                caption = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', caption)
                caption = caption.replace('\x00', '').replace('\ufffd', '')
                caption = caption.replace('BRACKET-TK', '').strip()
                
                if len(caption) < 5:
                    logger.debug(f"Caption too short in {jf}: '{caption[:50]}...'")
                    skipped += 1
                    continue
                
                # Step 2: Search for image in correct locations
                img_found = False
                img_path = None
                
                # First: Look in No-Subfig folder (has train/val/test subfolders)
                no_subfig_path = os.path.join(config.IMG_NO, split.lower())
                if os.path.exists(no_subfig_path):
                    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                        test_path = os.path.join(no_subfig_path, img_id + ext)
                        if os.path.exists(test_path):
                            img_path = test_path
                            img_found = True
                            break
                
                # Second: If training split, also check Yes-Subfig folder (only has train)
                if not img_found and split.lower() == "train":
                    yes_subfig_path = os.path.join(config.IMG_YES, "train")
                    if os.path.exists(yes_subfig_path):
                        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                            test_path = os.path.join(yes_subfig_path, img_id + ext)
                            if os.path.exists(test_path):
                                img_path = test_path
                                img_found = True
                                break
                
                # Third: Try with different extensions/case variations
                if not img_found:
                    # Try with original case
                    original_img_id = item.get("figure-ID", "")
                    original_img_id = os.path.splitext(original_img_id)[0]
                    
                    # Try no-subfig folder with original case
                    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                        test_path = os.path.join(no_subfig_path, original_img_id + ext)
                        if os.path.exists(test_path):
                            img_path = test_path
                            img_found = True
                            break
                
                if not img_found:
                    logger.debug(f"Image not found for {img_id} in {split}")
                    skipped += 1
                    continue
                
                # Step 3: Verify image can be loaded
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    records.append({
                        "image_path": img_path,
                        "caption": caption,
                        "image_id": img_id,
                        "json_file": jf
                    })
                    
                    # Debug logging - show first few samples
                    if len(records) <= 5:
                        logger.debug(f"  Sample {len(records)}:")
                        logger.debug(f"    ID: {img_id}")
                        logger.debug(f"    JSON: {jf}")
                        logger.debug(f"    Image: {os.path.basename(img_path)}")
                        logger.debug(f"    Caption: {caption[:100]}...")
                
                except (IOError, OSError, SyntaxError) as e:
                    corrupted_images.append(img_path)
                    skipped += 1
                    logger.debug(f"Corrupted image {img_path}: {e}")
                
                # Progress update
                if len(records) % 100 == 0 and len(records) > 0:
                    logger.info(f"  Loaded {len(records)} samples...")
        
        except json.JSONDecodeError as e:
            json_errors += 1
            if json_errors <= 5:
                logger.warning(f"JSON decode error in {jf}: {e}")
            continue
        except Exception as e:
            json_errors += 1
            if json_errors <= 5:
                logger.warning(f"Error processing {jf}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            continue
    
    logger.info(f"Successfully loaded {len(records)} samples for {split}")
    logger.info(f"Skipped {skipped} samples")
    if json_errors > 0:
        logger.info(f"JSON errors: {json_errors}")
    if corrupted_images:
        logger.info(f"Corrupted images: {len(corrupted_images)}")
    
    # Show sample statistics
    if records:
        avg_caption_len = sum(len(r["caption"]) for r in records) / len(records)
        logger.info(f"Average caption length: {avg_caption_len:.1f} characters")
        
        # Show a few sample captions for debugging
        logger.info("Sample captions (first 3):")
        for i, rec in enumerate(records[:3]):
            logger.info(f"  {i+1}. {rec['caption'][:100]}...")
    
    return records

# ============================================================
# DEBUG DATA COLLATOR - FIXES CUDA ASSERTION ERROR
# ============================================================
# ============================================================
# DEBUG DATA COLLATOR - FIXED VERSION
# ============================================================
class DebugSciCapDataCollator:
    """Data collator with debugging for token ID issues - FIXED VERSION"""
    def __init__(self, processor, max_length, logger):
        self.processor = processor
        self.max_length = max_length
        self.logger = logger
        self.vocab_size = len(processor.tokenizer)
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.eos_token_id = processor.tokenizer.eos_token_id
        self.bos_token_id = processor.tokenizer.bos_token_id
        
        # Log tokenizer details
        self.logger.info(f"Collator initialized:")
        self.logger.info(f"  Vocab size: {self.vocab_size}")
        self.logger.info(f"  Pad token ID: {self.pad_token_id}")
        self.logger.info(f"  EOS token ID: {self.eos_token_id}")
        self.logger.info(f"  BOS token ID: {self.bos_token_id}")
        
    def __call__(self, batch):
        """Collate batch with validation - FIXED VERSION"""
        images = [item["image"] for item in batch]
        captions = [item["caption"] for item in batch]
        
        # DEBUG: Check captions
        for i, caption in enumerate(captions):
            if not caption or len(caption.strip()) == 0:
                self.logger.warning(f"Empty caption at index {i}, using placeholder")
                captions[i] = "Scientific diagram"
            
            # Ensure caption is a string
            captions[i] = str(captions[i])
        
        # Process images
        try:
            image_inputs = self.processor(
                images=images,
                return_tensors="pt",
                max_patches=self.max_length
            )
        except Exception as e:
            self.logger.error(f"Error processing images: {e}")
            # Create dummy image inputs
            image_inputs = {
                "flattened_patches": torch.zeros((len(batch), self.max_length, 768)),
                "attention_mask": torch.ones((len(batch), self.max_length))
            }
        
        # Tokenize captions with careful settings
        try:
            text_inputs = self.processor.tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                add_special_tokens=True,
                return_attention_mask=True
            )
        except Exception as e:
            self.logger.error(f"Error tokenizing captions: {e}")
            # Create dummy text inputs
            text_inputs = {
                "input_ids": torch.ones((len(batch), self.max_length), dtype=torch.long) * self.pad_token_id,
                "attention_mask": torch.zeros((len(batch), self.max_length), dtype=torch.long)
            }
        
        # DEBUG: Validate token IDs
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs.get("attention_mask", None)
        
        # Create labels (shifted right for teacher forcing)
        labels = input_ids.clone()
        
        # Set padding tokens to -100 (ignore in loss)
        labels[labels == self.pad_token_id] = -100
        
        # DEBUG: Print first batch for inspection
        if hasattr(self, '_first_batch') and not self._first_batch:
            self._first_batch = True
            self.logger.info("\n=== FIRST BATCH DEBUG ===")
            self.logger.info(f"Number of samples: {len(batch)}")
            self.logger.info(f"Image patches shape: {image_inputs.get('flattened_patches', 'N/A').shape}")
            self.logger.info(f"Input IDs shape: {input_ids.shape}")
            self.logger.info(f"Labels shape: {labels.shape}")
            
            # Show token IDs for first caption
            sample_input_ids = input_ids[0].tolist()
            sample_labels = labels[0].tolist()
            
            # Remove padding from display
            non_pad_input = [t for t in sample_input_ids if t != self.pad_token_id]
            non_pad_labels = [t for t in sample_labels if t != -100]
            
            self.logger.info(f"Sample input IDs (non-pad): {non_pad_input[:20]}...")
            self.logger.info(f"Sample labels (non-ignore): {non_pad_labels[:20]}...")
            
            # Check for invalid token IDs
            invalid_input = [t for t in sample_input_ids if t >= self.vocab_size]
            invalid_labels = [t for t in sample_labels if t != -100 and t >= self.vocab_size]
            
            if invalid_input:
                self.logger.warning(f"Invalid input token IDs: {invalid_input}")
            if invalid_labels:
                self.logger.warning(f"Invalid label token IDs: {invalid_labels}")
            
            # Decode tokens back to text
            try:
                decoded = self.processor.tokenizer.decode(sample_input_ids, skip_special_tokens=True)
                self.logger.info(f"Decoded caption: {decoded[:100]}...")
            except:
                self.logger.warning("Could not decode tokens")
        
        # Validate all token IDs
        self.validate_token_ids(input_ids, "input_ids")
        if labels is not None:
            self.validate_labels(labels)
        
        # Return proper format for Pix2Struct
        return {
            "flattened_patches": image_inputs["flattened_patches"],
            "attention_mask": image_inputs["attention_mask"],
            "labels": labels,
            # Don't provide decoder_input_ids - model will handle it
        }
    
    def validate_token_ids(self, token_ids, name):
        """Validate token IDs are within vocabulary"""
        if token_ids is None:
            return
        
        # Convert to tensor if needed
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids)
        
        # Check for NaN or inf
        if torch.isnan(token_ids).any() or torch.isinf(token_ids).any():
            self.logger.error(f"{name} contains NaN or inf values!")
        
        # Check bounds
        max_id = token_ids.max().item()
        min_id = token_ids.min().item()
        
        if max_id >= self.vocab_size:
            self.logger.error(f"{name} contains token ID {max_id} >= vocab size {self.vocab_size}")
            # Find and fix invalid tokens
            invalid_mask = token_ids >= self.vocab_size
            invalid_count = invalid_mask.sum().item()
            if invalid_count > 0:
                self.logger.error(f"Found {invalid_count} invalid token IDs")
                # Replace with pad token
                token_ids[invalid_mask] = self.pad_token_id
        
        if min_id < 0:
            # Negative values are OK for labels (-100 means ignore)
            if name == "labels":
                self.logger.debug(f"Labels contain negative values (ignore tokens): {min_id}")
            else:
                self.logger.error(f"{name} contains negative token ID {min_id}")
                # Replace negatives with pad token
                invalid_mask = token_ids < 0
                token_ids[invalid_mask] = self.pad_token_id
        
        # Log statistics
        unique_tokens = torch.unique(token_ids[token_ids != self.pad_token_id])
        self.logger.debug(f"{name}: shape={token_ids.shape}, range=[{min_id}, {max_id}], unique={len(unique_tokens)}")
    
    def validate_labels(self, labels):
        """Validate labels for training"""
        if labels is None:
            return
        
        # Count ignored tokens (-100)
        ignored = (labels == -100).sum().item()
        total = labels.numel()
        
        self.logger.debug(f"Labels: {ignored}/{total} tokens ignored ({(ignored/total)*100:.1f}%)")
        
        # Check for any token IDs in labels that should be ignored
        valid_labels = labels[labels != -100]
        if len(valid_labels) > 0:
            max_label = valid_labels.max().item()
            min_label = valid_labels.min().item()
            
            if max_label >= self.vocab_size:
                self.logger.error(f"Label contains token ID {max_label} >= vocab size {self.vocab_size}")
                # Fix invalid labels
                invalid_mask = (labels != -100) & (labels >= self.vocab_size)
                labels[invalid_mask] = self.pad_token_id
            
            if min_label < 0:
                # Should only be -100
                other_negative = ((labels < 0) & (labels != -100)).sum().item()
                if other_negative > 0:
                    self.logger.error(f"Labels contain {other_negative} negative values other than -100")


# ============================================================
# CUSTOM CALLBACKS
# ============================================================
class ScicapTrainingCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % config.LOGGING_STEPS == 0:
            metrics = {
                "step": state.global_step,
                "epoch": state.epoch,
                "loss": state.log_history[-1].get("loss", "N/A") if state.log_history else "N/A"
            }
            log_metrics(metrics, OUTPUT_DIR)
            
            if state.global_step % 20 == 0:
                log_health(self.logger)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = (time.time() - self.start_time) / 60
        self.logger.info(f"Epoch {int(state.epoch)} completed in {epoch_time:.1f} minutes")
        
        epoch_metrics = {
            "epoch": int(state.epoch),
            "global_step": state.global_step,
            "time_minutes": epoch_time
        }
        if state.log_history:
            last_log = state.log_history[-1]
            epoch_metrics.update({
                k: v for k, v in last_log.items() 
                if k in ["loss", "eval_loss", "learning_rate"]
            })
        
        log_metrics(epoch_metrics, OUTPUT_DIR)
        self.start_time = time.time()

# ============================================================
# MODEL SETUP
# ============================================================
def setup_model():
    """Load and configure Pix2Struct"""
    logger.info("\n=== Loading Pix2Struct Model ===")
    
    model = Pix2StructForConditionalGeneration.from_pretrained(config.BASE_MODEL)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = 0
    
    logger.info("\n=== Layer Freezing Strategy ===")
    
    if config.FREEZE_EARLY_LAYERS:
        for name, param in model.named_parameters():
            if name.startswith("encoder.embed") or any(
                name.startswith(f"encoder.block.{i}") 
                for i in range(config.NUM_FROZEN_BLOCKS)
            ):
                param.requires_grad = False
            else:
                param.requires_grad = True
                trainable_params += param.numel()
    else:
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = total_params
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable percentage: {100 * trainable_params/total_params:.1f}%")
    
    return model

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def main():
    global processor

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Save configuration
    config_dict = {
        "model_name": MODEL_NAME,
        "base_model": config.BASE_MODEL,
        "scicap_base": config.BASE,
        "max_train_samples": config.MAX_TRAIN_SAMPLES,
        "max_val_samples": config.MAX_VAL_SAMPLES,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "grad_accum_steps": config.GRAD_ACCUM_STEPS,
        "learning_rate": config.LEARNING_RATE,
        "max_length": config.MAX_LENGTH,
        "freeze_early_layers": config.FREEZE_EARLY_LAYERS,
        "num_frozen_blocks": config.NUM_FROZEN_BLOCKS,
    }
    
    save_config(config_dict, OUTPUT_DIR)
    
    # Load processor
    logger.info("\n1. Loading Pix2Struct processor...")
    processor = Pix2StructProcessor.from_pretrained(config.BASE_MODEL)
    processor.image_processor.max_patches = config.MAX_LENGTH
    
    # Log tokenizer info
    logger.info(f"Tokenizer info:")
    logger.info(f"  Vocab size: {len(processor.tokenizer)}")
    logger.info(f"  Pad token: {processor.tokenizer.pad_token} (ID: {processor.tokenizer.pad_token_id})")
    logger.info(f"  EOS token: {processor.tokenizer.eos_token} (ID: {processor.tokenizer.eos_token_id})")
    
    # Load datasets
    logger.info("\n2. Loading SciCap datasets...")
    
    # Load train data
    logger.info("Loading training data...")
    train_data = load_split("train", max_samples=config.MAX_TRAIN_SAMPLES)
    
    # Load validation data
    logger.info("Loading validation data...")
    val_data = load_split("val", max_samples=config.MAX_VAL_SAMPLES)
    
    if not train_data:
        logger.error("No training data loaded!")
        return None, None, None
    if not val_data:
        logger.error("No validation data loaded!")
        return None, None, None
    
    # Create datasets
    train_dataset = SciCapDataset(train_data)
    eval_dataset = SciCapDataset(val_data)
    
    logger.info(f"\nFinal dataset sizes:")
    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(eval_dataset)} samples")
    
    # Setup model
    logger.info("\n3. Setting up model...")
    model = setup_model()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model moved to: {device}")
    
    # Training arguments - simplified for debugging
    # Training arguments - simplified for debugging
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=config.WARMUP_STEPS,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_strategy="steps",
        logging_steps=config.LOGGING_STEPS,
        save_total_limit=2,
        report_to="none",
        fp16=False,  # Disable mixed precision for debugging
        dataloader_num_workers=0,  # Use 0 workers to avoid multiprocessing issues
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Disable pin memory for debugging
        gradient_checkpointing=False,
        dataloader_drop_last=True,  # Drop last incomplete batch
        # Add debugging options
        disable_tqdm=False,  # Show progress bars
        optim="adamw_torch",  # Use torch optimizer
        seed=42,
    )
    
    # Create debug collator
    data_collator = DebugSciCapDataCollator(processor, config.MAX_LENGTH, logger)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[ScicapTrainingCallback(logger)],
    )
    
    # Train
    logger.info("\n4. Starting training (DEBUG MODE)...")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(eval_dataset)}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Batch size: {config.BATCH_SIZE} (effective: {config.BATCH_SIZE * config.GRAD_ACCUM_STEPS})")
    logger.info(f"Learning rate: {config.LEARNING_RATE}")
    logger.info(f"Workers: {config.WORKERS}")
    
    log_health(logger)
    
    try:
        start_time = time.time()
        trainer.train()
        training_time = (time.time() - start_time) / 60
        
        logger.info(f"\nTraining completed in {training_time:.1f} minutes")
        
        # Save model
        logger.info("\n5. Saving model...")
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        
        logger.info(f"Model saved to: {OUTPUT_DIR}")
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OUT OF MEMORY! Training stopped.")
        return None, None, None
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None
    
    return model, processor, OUTPUT_DIR

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    try:
        logger.info("Starting Stage 1 training with SciCap (DEBUG MODE)...")
        log_health(logger)
        
        model, processor, output_dir = main()
        
        if model:
            log_health(logger)
            logger.info("Stage 1 execution completed successfully!")
            
            # If debug worked, suggest increasing dataset size
            logger.info("\n" + "=" * 60)
            logger.info("DEBUG SUCCESSFUL!")
            logger.info("To train on full dataset, update config:")
            logger.info("  MAX_TRAIN_SAMPLES = None")
            logger.info("  MAX_VAL_SAMPLES = None")
            logger.info("  EPOCHS = 4")
            logger.info("  WORKERS = 4")
            logger.info("=" * 60)
        else:
            logger.error("Training failed during debug phase")
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)