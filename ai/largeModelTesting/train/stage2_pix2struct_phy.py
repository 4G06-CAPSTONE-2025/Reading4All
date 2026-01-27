"""
Stage 1 Fine-Tuning – Physics Diagram Alt-Text Generation (Pix2Struct)
=====================================================================

This script fine-tunes an existing **Stage 1 Pix2Struct model** (continued from
model #1946) on a **custom, annotated physics textbook dataset** to improve
high-quality, human-readable alt-text generation for STEM diagrams.

It is designed as a **robust, production-grade fine-tuning pipeline** with
strong emphasis on:
• Data validation
• Reproducibility
• Training stability
• Monitoring for mode collapse and repetition
• Windows + consumer GPU compatibility

---------------------------------------------------------------------
Pipeline Context
---------------------------------------------------------------------
This script operates at **Stage 1** of a multi-stage accessibility pipeline:

    Physics Diagram Image
        ↓
    Stage 1 (THIS SCRIPT)
        → Natural-language alt text
        ↓
    Stage 2 (AI2D Semantic Parsing)
        → Structured diagram representation
        ↓
    Stage 3 (LLM Refinement / MAJUNGA)
        → Highly accessible, pedagogical alt text

Stage 1 focuses on **direct image → alt-text alignment**, grounded in
real textbook annotations.

---------------------------------------------------------------------
Hardware & Platform Assumptions
---------------------------------------------------------------------
• Operating System: Windows 10 / 11
• GPU: NVIDIA RTX / GTX (CUDA-enabled)
• Mixed precision training enabled when CUDA is available
• Small per-device batch size with gradient accumulation
• Designed to run safely on limited VRAM setups

---------------------------------------------------------------------
Model & Training Strategy
---------------------------------------------------------------------
• Base model: Pix2StructForConditionalGeneration
• Initialization: Pretrained Stage 1 checkpoint (model #1946 lineage)
• Encoder: Optionally frozen (configurable)
• Decoder: Fully trainable
• Optimizer: AdamW
• Learning rate: 1e-5 (low LR for careful fine-tuning)
• Effective batch size: BATCH_SIZE × GRAD_ACCUM_STEPS
• Max sequence length: 768 tokens (long, descriptive alt text)

---------------------------------------------------------------------
Dataset Description
---------------------------------------------------------------------
Input data is provided via a CSV file containing:
• Image paths (relative or absolute)
• Modified alt text (preferred)
• Original captions (fallback)
• Diagram metadata (ID, page number, index)

The loader is intentionally defensive:
• Tries multiple path resolutions per image
• Skips samples with missing images or text
• Logs skipped cases for auditability

Each valid sample consists of:
    {
        image: PIL.Image,
        alt_text: str,
        image_path: str
    }

---------------------------------------------------------------------
Data Splitting
---------------------------------------------------------------------
• Train / Validation / Test splits are configurable
• Default: 90% train, 10% validation
• Deterministic splitting using a fixed random seed
• Split IDs are saved for reproducibility

---------------------------------------------------------------------
Data Collation
---------------------------------------------------------------------
The custom data collator:
• Uses Pix2StructProcessor for image encoding
• Tokenizes alt text with padding + truncation
• Masks padding tokens with -100 for loss computation
• Returns tensors compatible with HuggingFace Trainer

---------------------------------------------------------------------
Metrics & Training Health Monitoring
---------------------------------------------------------------------
In addition to loss, the pipeline computes:
• ROUGE-1 / ROUGE-2 / ROUGE-L
• Prediction diversity (uniqueness ratio)
• Average prediction length
• Word-level repetition scores

These metrics are used to detect:
• Mode collapse
• Over-repetition
• Degenerate outputs

Warnings are logged automatically if thresholds are crossed.

---------------------------------------------------------------------
Logging & Checkpointing
---------------------------------------------------------------------
• Detailed logging to both console and file
• Periodic evaluation and checkpoint saving
• Best model selected based on validation loss
• Final artifacts include:
    - Trained model
    - Processor
    - Training metrics
    - Evaluation results
    - Sample predictions
    - Data split manifest

All outputs are saved to a timestamped directory.

---------------------------------------------------------------------
Intended Use
---------------------------------------------------------------------
This script is intended for:
• Accessibility research
• Physics / STEM education tooling
• Diagram-to-text model specialization
• Producing high-quality training checkpoints for later stages

It is **not** intended for inference-only usage.

---------------------------------------------------------------------
"""

import os
import sys
import csv
import json
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Pix2StructProcessor,
    Pix2StructForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

import evaluate

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Training configuration"""
    
    STAGE1_MODEL = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/model/pix2struct_physics_finetuned_20260125_0047/final_model"
    CSV_PATH = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/physics_textbook_extracted/extracted_images.csv"
    IMAGE_BASE_DIR = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/physics_textbook_extracted/images"
    OUTPUT_DIR = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/model/pix2struct_physicsTxtbk"
    
    # Training hyperparameters
    BATCH_SIZE = 1  # Increase if you have more GPU memory
    GRAD_ACCUM_STEPS = 8  # Effective batch size = 32
    LEARNING_RATE = 1e-5  # Lower LR for fine-tuning
    EPOCHS = 10  # More epochs for small dataset
    WARMUP_RATIO = 0.1
    MAX_LENGTH = 768  # Longer to accommodate detailed alt-text
    
    # Model settings
    FREEZE_ENCODER = True  # Don't freeze - allow adaptation
    FREEZE_FIRST_N_BLOCKS = 0  # Or set to 2 for minimal freezing
    
    # Data settings
    TRAIN_SPLIT = 0.90  # 85% for training
    VAL_SPLIT = 0.10    # 10% for validation
    TEST_SPLIT = 0.0   # 5% for testing
    
    # Logging and checkpointing
    LOGGING_STEPS = 10
    EVAL_STEPS = 50
    SAVE_STEPS = 50
    SAVE_TOTAL_LIMIT = 3
    
    # Reproducibility
    SEED = 42
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(output_dir: str) -> None:
    """Setup logging directory"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ============================================================
# DATA LOADING
# ============================================================

def load_csv_data(csv_path: str) -> List[Dict]:
    """
    Load the annotated physics data from CSV
    Returns list of dicts with 'image_path' and 'alt_text'
    """
    data = []
    skipped_no_image = 0
    skipped_no_text = 0
    
    print(f"\n{'='*60}")
    print(f"Loading data from: {csv_path}")
    print(f"{'='*60}\n")
    
    # First verify CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at: {csv_path}")
        
        # Try alternative path
        alt_path = csv_path.replace("C:/Users/nawaa/", "C:/Users/nawaa/OneDrive/")
        if os.path.exists(alt_path):
            print(f"✓ Found at alternative path: {alt_path}")
            csv_path = alt_path
        else:
            print(f"Also tried: {alt_path} (not found)")
            return data
    
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            # Use csv.reader to handle multi-line fields properly
            reader = csv.DictReader(f)
            
            # Print headers for debugging
            print(f"CSV Headers found: {reader.fieldnames}")
            
            for idx, row in enumerate(reader):
                image_path = row.get('Image-Path', '').strip()
                
                # If no image path, skip
                if not image_path or image_path.lower() in ['', 'nan', 'none', 'null']:
                    skipped_no_image += 1
                    continue
                
                # Try multiple text sources in order of preference
                alt_text = ''
                
                # 1. Modified-Alt-Text (if available and valid)
                mod_text = row.get('Modified-Alt-Text', '').strip()
                if mod_text and mod_text.lower() not in ['', 'nan', 'none', 'null']:
                    alt_text = mod_text
                
                # 2. Original Caption (if Modified-Alt-Text is empty)
                elif row.get('Caption', '').strip():
                    alt_text = row.get('Caption', '').strip()
                
                # 3. Generate descriptive text from metadata
                elif row.get('Diagram-ID', '').strip():
                    # Create descriptive text from available metadata
                    diagram_id = row.get('Diagram-ID', '').strip()
                    page_num = row.get('Page-Number', '').strip()
                    alt_text = f"Physics diagram {diagram_id}"
                    if page_num:
                        alt_text += f" from page {page_num}"
                
                # If still no text, skip
                if not alt_text:
                    skipped_no_text += 1
                    continue
                
                # Remove any surrounding quotes
                if alt_text.startswith('"') and alt_text.endswith('"'):
                    alt_text = alt_text[1:-1]
                if image_path.startswith('"') and image_path.endswith('"'):
                    image_path = image_path[1:-1]
                
                # Clean up the alt text
                alt_text = ' '.join(alt_text.split())
                
                # Store just the filename for easier lookup
                # (assuming images are in the IMAGE_BASE_DIR)
                data.append({
                    'image_path': image_path,  # Keep original
                    'filename': os.path.basename(image_path),  # Add filename
                    'alt_text': alt_text,
                    'diagram_id': row.get('Diagram-ID', ''),
                    'page_number': row.get('Page-Number', ''),
                    'image_index': row.get('Image-Index', ''),
                    'unique_id': f"{row.get('Diagram-ID', '')}_{idx}"
                })
                
                # Print first few for debugging
                if idx < 3:
                    print(f"\nSample row {idx}:")
                    print(f"  Image path: {image_path}")
                    print(f"  Filename: {os.path.basename(image_path)}")
                    print(f"  Text: {alt_text[:100]}..." if len(alt_text) > 100 else f"  Text: {alt_text}")
    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Data Loading Summary:")
    print(f"  Total valid samples: {len(data)}")
    print(f"  Skipped (no image): {skipped_no_image}")
    print(f"  Skipped (no text): {skipped_no_text}")
    print(f"{'='*60}\n")
    
    return data

def split_data(data: List[Dict], train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42):
    """Split data into train/val/test sets"""
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Shuffle data
    random.seed(seed)
    data_shuffled = data.copy()
    random.shuffle(data_shuffled)
    
    n = len(data_shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_data = data_shuffled[:n_train]
    val_data = data_shuffled[n_train:n_train + n_val]
    test_data = data_shuffled[n_train + n_val:]
    
    print(f"\n{'='*60}")
    print(f"Data Split:")
    print(f"  Training:   {len(train_data)} samples ({len(train_data)/n*100:.1f}%)")
    print(f"  Validation: {len(val_data)} samples ({len(val_data)/n*100:.1f}%)")
    print(f"  Test:       {len(test_data)} samples ({len(test_data)/n*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return train_data, val_data, test_data

# ============================================================
# DATASET CLASS
# ============================================================

class PhysicsAltTextDataset(Dataset):
    """
    Dataset for physics diagram alt-text pairs
    """
    
    def __init__(
        self, 
        data: List[Dict], 
        processor: Pix2StructProcessor,
        image_base_dir: str,
        max_length: int = 768,
        augment: bool = False
    ):
        self.data = data
        self.processor = processor
        self.image_base_dir = Path(image_base_dir)
        self.max_length = max_length
        self.augment = augment
        
        # Verify images exist
        self.valid_indices = []
        missing_images = []
        
        for idx, item in enumerate(self.data):
            # Try different path combinations
            img_path = self._find_image_path(item['image_path'])
            
            if img_path and img_path.exists():
                self.valid_indices.append(idx)
            else:
                missing_images.append(item['image_path'])
        
        if missing_images:
            print(f"\n Warning: {len(missing_images)} images not found:")
            for img in missing_images[:5]:  # Show first 5
                print(f"  - {img}")
            if len(missing_images) > 5:
                print(f"  ... and {len(missing_images) - 5} more")
        
        print(f"\n Dataset initialized with {len(self.valid_indices)}/{len(self.data)} valid samples")
    
    def _find_image_path(self, relative_path: str) -> Optional[Path]:
        """Try to find the image in various locations"""
        
        # Clean the path
        relative_path = relative_path.strip()
        
        # Remove any quotes
        relative_path = relative_path.replace('"', '').replace("'", "")
        
        # Debug: print what we're looking for
        # print(f"\nLooking for: {relative_path}")
        
        # Try as absolute path first
        if os.path.isabs(relative_path):
            if Path(relative_path).exists():
                return Path(relative_path)
        
        # Get just the filename
        filename = Path(relative_path).name
        
        # Try different combinations
        candidates = [
            # Direct combination
            self.image_base_dir / relative_path,
            # Just the filename in the base directory
            self.image_base_dir / filename,
            # Try one level up (in case relative_path already has "images/")
            self.image_base_dir.parent / relative_path,
            # Try in CSV directory
            Path(config.CSV_PATH).parent / relative_path,
            Path(config.CSV_PATH).parent / filename,
        ]
        
        for candidate in candidates:
            candidate = candidate.resolve()
            # print(f"  Trying: {candidate}")
            if candidate.exists():
                # print(f"  ✓ Found!")
                return candidate
        
        # If we get here, print debug info
        print(f"\n❌ Could not find image: {relative_path}")
        print(f"   Filename: {filename}")
        print(f"   Base directory: {self.image_base_dir}")
        print(f"   Base exists: {self.image_base_dir.exists()}")
        
        # List what's in the base directory
        if self.image_base_dir.exists():
            try:
                files = list(self.image_base_dir.iterdir())
                print(f"   Files in base directory ({len(files)} total):")
                # Show files that might be matches
                matching = [f for f in files if filename in f.name]
                for f in matching[:10]:
                    print(f"     - {f.name}")
            except Exception as e:
                print(f"   Error listing files: {e}")
        
        return None
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        item = self.data[actual_idx]
        
        # Load image
        img_path = self._find_image_path(item['image_path'])
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Optional augmentation
            if self.augment and random.random() < 0.3:
                # Simple augmentation: slight rotation or brightness adjustment
                from PIL import ImageEnhance
                if random.random() < 0.5:
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(random.uniform(0.8, 1.2))
            
            alt_text = item['alt_text']
            
            return {
                'image': image,
                'alt_text': alt_text,
                'image_path': str(img_path)
            }
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return first valid item as fallback
            return self.__getitem__(0)

# ============================================================
# DATA COLLATOR
# ============================================================

class AltTextDataCollator:
    """
    Collator that properly encodes images and text
    """
    
    def __init__(self, processor: Pix2StructProcessor, max_length: int = 768):
        self.processor = processor
        self.max_length = max_length
        self.pad_token_id = processor.tokenizer.pad_token_id
    
    def __call__(self, batch):
        # Filter out None values
        batch = [item for item in batch if item is not None]
        
        if not batch:
            return None
        
        # Extract images and texts
        images = [item['image'] for item in batch]
        texts = [item['alt_text'] for item in batch]
        
        # Process images using the processor
        encoding = self.processor(
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        # Tokenize texts
        text_encoding = self.processor.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # Create labels (mask padding tokens)
        labels = text_encoding.input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        
        return {
            'flattened_patches': encoding.flattened_patches,
            'attention_mask': encoding.attention_mask,
            'labels': labels,
            'decoder_attention_mask': text_encoding.attention_mask
        }

# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred, tokenizer):
    """
    Compute metrics to monitor training health
    """
    predictions, labels = eval_pred
    
    # Handle different prediction formats safely
    try:
        if predictions.ndim == 3:
            # predictions are logits, take argmax
            predictions = np.argmax(predictions, axis=-1)
        elif predictions.ndim == 2:
            # predictions are token IDs
            pass
        else:
            raise ValueError(f"Unexpected predictions shape: {predictions.shape}")
            
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels with pad token for decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute uniqueness (detect mode collapse)
        unique_preds = len(set(decoded_preds))
        uniqueness_ratio = unique_preds / len(decoded_preds) if decoded_preds else 0
        
        # Compute average length
        avg_pred_length = np.mean([len(p.split()) for p in decoded_preds])
        avg_label_length = np.mean([len(l.split()) for l in decoded_labels])
        
        # Check for repetition within predictions
        repetition_scores = []
        for pred in decoded_preds:
            words = pred.split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                repetition_scores.append(unique_ratio)
        
        avg_uniqueness = np.mean(repetition_scores) if repetition_scores else 0
        
        # Compute ROUGE scores
        try:
            rouge = evaluate.load("rouge")
            rouge_scores = rouge.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True
            )
        except Exception as e:
            print(f"Warning: Could not compute ROUGE scores: {e}")
            rouge_scores = {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0
            }
        
        metrics = {
            "unique_predictions": unique_preds,
            "uniqueness_ratio": uniqueness_ratio,
            "avg_pred_length": avg_pred_length,
            "avg_label_length": avg_label_length,
            "avg_word_uniqueness": avg_uniqueness,
            "rouge1": rouge_scores.get("rouge1", 0.0),
            "rouge2": rouge_scores.get("rouge2", 0.0),
            "rougeL": rouge_scores.get("rougeL", 0.0),
        }
        
    except Exception as e:
        print(f"Error computing metrics: {e}")
        # Return dummy metrics if computation fails
        metrics = {
            "unique_predictions": 0,
            "uniqueness_ratio": 0.0,
            "avg_pred_length": 0.0,
            "avg_label_length": 0.0,
            "avg_word_uniqueness": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
        }
    
    return metrics

# ============================================================
# CALLBACK
# ============================================================

class DetailedLoggingCallback(TrainerCallback):
    """Callback for detailed logging during training"""
    
    def __init__(self, logger, tokenizer):
        self.logger = logger
        self.tokenizer = tokenizer
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self.start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("Training started")
        self.logger.info("=" * 60)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Log training progress
            if 'loss' in logs:
                self.logger.info(
                    f"Step {state.global_step}: "
                    f"loss={logs['loss']:.4f}"
                )
            
            # Log evaluation results
            if 'eval_loss' in logs:
                self.logger.info(
                    f"\nEvaluation at step {state.global_step}:"
                )
                for key, value in logs.items():
                    if key.startswith('eval_'):
                        self.logger.info(f"  {key}: {value:.4f}")
            
            # Monitor for issues
            if 'eval_avg_word_uniqueness' in logs:
                if logs['eval_avg_word_uniqueness'] < 0.3:
                    self.logger.warning(
                        " Low word uniqueness detected - possible repetition!"
                    )
            
            if 'eval_uniqueness_ratio' in logs:
                if logs['eval_uniqueness_ratio'] < 0.5:
                    self.logger.warning(
                        " Low prediction diversity - possible mode collapse!"
                    )
    
    def on_epoch_end(self, args, state, control, **kwargs):
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Epoch {state.epoch} completed")
        self.logger.info(f"{'='*60}\n")

# ============================================================
# MODEL SETUP
# ============================================================

def setup_model(model_path: str, freeze_encoder: bool = False, freeze_first_n: int = 0):
    """
    Load the Stage 1 model and optionally freeze layers
    """
    
    print(f"\n{'='*60}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*60}\n")
    
    model = Pix2StructForConditionalGeneration.from_pretrained(model_path)
    processor = Pix2StructProcessor.from_pretrained(model_path)
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optional freezing
    if freeze_encoder:
        print("Freezing entire encoder...")
        for name, param in model.named_parameters():
            if name.startswith("encoder"):
                param.requires_grad = False
    
    elif freeze_first_n > 0:
        print(f"Freezing first {freeze_first_n} encoder blocks...")
        
        # Freeze embedding
        for name, param in model.named_parameters():
            if name.startswith("encoder.embed"):
                param.requires_grad = False
        
        # Freeze first N blocks
        for i in range(freeze_first_n):
            for name, param in model.named_parameters():
                if name.startswith(f"encoder.block.{i}"):
                    param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # # Enable gradient checkpointing for memory efficiency
    # if hasattr(model.encoder, "block"):
    #     for module in model.encoder.block:
    #         if hasattr(module, 'gradient_checkpointing_enable'):
    #             module.gradient_checkpointing_enable()
    #     print(" Gradient checkpointing enabled for encoder")
    
    # if hasattr(model.decoder, "block"):
    #     for module in model.decoder.block:
    #         if hasattr(module, 'gradient_checkpointing_enable'):
    #             module.gradient_checkpointing_enable()
    #     print(" Gradient checkpointing enabled for decoder")
    
    return model, processor

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():
    """Main training function"""
    
    # Set seed
    set_seed(config.SEED)
    
    # Setup output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"{config.OUTPUT_DIR}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {config.DEVICE}")
    
    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump({
            'stage1_model': config.STAGE1_MODEL,
            'batch_size': config.BATCH_SIZE,
            'grad_accum_steps': config.GRAD_ACCUM_STEPS,
            'learning_rate': config.LEARNING_RATE,
            'epochs': config.EPOCHS,
            'max_length': config.MAX_LENGTH,
            'freeze_encoder': config.FREEZE_ENCODER,
            'freeze_first_n_blocks': config.FREEZE_FIRST_N_BLOCKS,
            'seed': config.SEED,
        }, f, indent=2)
    
    # Load data
    data = load_csv_data(config.CSV_PATH)
    
    if len(data) < 10:
        logger.warning(f" Only {len(data)} samples found - this may not be enough!")
        logger.warning("Consider using data augmentation or synthetic data generation")
    
    # Split data
    train_data, val_data, test_data = split_data(
        data,
        config.TRAIN_SPLIT,
        config.VAL_SPLIT,
        config.TEST_SPLIT,
        config.SEED
    )
    
    # Save splits for reproducibility
    splits_path = os.path.join(output_dir, "data_splits.json")
    with open(splits_path, 'w') as f:
        json.dump({
            'train': [d['unique_id'] for d in train_data],
            'val': [d['unique_id'] for d in val_data],
            'test': [d['unique_id'] for d in test_data],
        }, f, indent=2)
    
    # Load model and processor
    model, processor = setup_model(
        config.STAGE1_MODEL,
        freeze_encoder=config.FREEZE_ENCODER,
        freeze_first_n=config.FREEZE_FIRST_N_BLOCKS
    )
    
    # Move model to device
    model.to(config.DEVICE)
    logger.info(f"Model loaded on {config.DEVICE}")
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = PhysicsAltTextDataset(
        train_data,
        processor,
        config.IMAGE_BASE_DIR,
        config.MAX_LENGTH,
        augment=True  # Enable augmentation for training
    )
    
    val_dataset = PhysicsAltTextDataset(
        val_data,
        processor,
        config.IMAGE_BASE_DIR,
        config.MAX_LENGTH,
        augment=False
    )
    
    # Create data collator
    data_collator = AltTextDataCollator(processor, config.MAX_LENGTH)
    
    # Test data collator
    logger.info("\nTesting data collator...")
    try:
        test_sample = train_dataset[0]
        test_batch = data_collator([test_sample])
        logger.info(f" Data collator test passed")
        logger.info(f"  Batch keys: {test_batch.keys()}")
        logger.info(f"  Flattened patches shape: {test_batch['flattened_patches'].shape}")
        logger.info(f"  Labels shape: {test_batch['labels'].shape}")
    except Exception as e:
        logger.error(f" Data collator test failed: {e}")
        raise
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        
        # Optimization
        learning_rate=config.LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=config.WARMUP_RATIO,
        max_grad_norm=1.0,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        
        # Saving
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging
        logging_strategy="steps",
        logging_steps=config.LOGGING_STEPS,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        
        # Performance
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        #gradient_checkpointing=True,
        
        # Misc
        remove_unused_columns=False,
        seed=config.SEED,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[DetailedLoggingCallback(logger, processor.tokenizer)],
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, processor.tokenizer)
    )
    
    # Print training info
    logger.info("\n" + "="*60)
    logger.info("Training Configuration:")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")
    logger.info(f"  Batch size: {config.BATCH_SIZE}")
    logger.info(f"  Gradient accumulation: {config.GRAD_ACCUM_STEPS}")
    logger.info(f"  Effective batch size: {config.BATCH_SIZE * config.GRAD_ACCUM_STEPS}")
    logger.info(f"  Epochs: {config.EPOCHS}")
    logger.info(f"  Learning rate: {config.LEARNING_RATE}")
    logger.info(f"  Max sequence length: {config.MAX_LENGTH}")
    logger.info(f"  Total training steps: {len(train_dataset) // (config.BATCH_SIZE * config.GRAD_ACCUM_STEPS) * config.EPOCHS}")
    logger.info("="*60 + "\n")
    
    # Train
    try:
        logger.info("Starting training...\n")
        train_result = trainer.train()
        
        # Save final model
        logger.info(f"\n{'='*60}")
        logger.info("Training completed!")
        logger.info(f"{'='*60}\n")
        
        final_model_dir = os.path.join(output_dir, "final_model")
        trainer.save_model(final_model_dir)
        processor.save_pretrained(final_model_dir)
        
        logger.info(f" Model saved to: {final_model_dir}")
        
        # Save training metrics
        metrics_path = os.path.join(output_dir, "train_results.json")
        with open(metrics_path, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f" Metrics saved to: {metrics_path}")
        
        # Final evaluation on validation set
        logger.info("\nRunning final evaluation on validation set...")
        eval_results = trainer.evaluate()
        
        eval_path = os.path.join(output_dir, "eval_results.json")
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f" Evaluation results saved to: {eval_path}")
        
        # Print final metrics
        logger.info("\nFinal Evaluation Metrics:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Generate sample predictions
        logger.info("\n" + "="*60)
        logger.info("Generating sample predictions...")
        logger.info("="*60 + "\n")
        
        model.eval()
        sample_predictions = []
        
        for i in range(min(5, len(val_dataset))):
            sample = val_dataset[i]
            image = sample['image']
            true_text = sample['alt_text']
            
            # Generate prediction
            inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_length=config.MAX_LENGTH,
                    num_beams=4,
                    early_stopping=True
                )
            
            predicted_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            sample_predictions.append({
                'image_path': sample['image_path'],
                'true_alt_text': true_text,
                'predicted_alt_text': predicted_text
            })
            
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Image: {sample['image_path']}")
            logger.info(f"  True: {true_text[:200]}...")
            logger.info(f"  Pred: {predicted_text[:200]}...")
            logger.info("")
        
        # Save sample predictions
        samples_path = os.path.join(output_dir, "sample_predictions.json")
        with open(samples_path, 'w') as f:
            json.dump(sample_predictions, f, indent=2)
        
        logger.info(f" Sample predictions saved to: {samples_path}")
        
        logger.info("\n" + "="*60)
        logger.info("Training pipeline completed successfully!")
        logger.info(f"All outputs saved to: {output_dir}")
        logger.info("="*60 + "\n")
        
        return trainer, model, processor
        
    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error(f"Error during training: {e}")
        logger.error(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        raise

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Physics Alt-Text Fine-Tuning Pipeline")
    print("="*60 + "\n")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f" GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(" No GPU detected - training will be slow")
    
    print("\nStarting training...\n")
    
    try:
        trainer, model, processor = main()
        print("\n Pipeline completed successfully!")
    except KeyboardInterrupt:
        print("\n\n Training interrupted by user")
    except Exception as e:
        print(f"\n Pipeline failed with error: {e}")
        raise