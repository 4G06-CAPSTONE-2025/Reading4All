'''
Description:
End-to-end preprocessing pipeline for the AI2D dataset, designed to generate
semantically rich training targets for Pix2Struct-based models.

This script processes raw AI2D diagram images alongside their annotations and
question files to produce training-ready `.pt` files containing:
- Vision embeddings (pixel values or flattened patches)
- Tokenized semantic targets
- Human-readable semantic target text (for debugging and analysis)

Key Features:
1. Loads AI2D images, annotations, and question JSON files with robust error handling.
2. Performs multi-type visual emphasis detection using OpenCV:
   - Color emphasis (red, green, blue, yellow)
   - Edge density
   - Contrast
   - Dominant object size
   - Spatial focus region (center, left, right, top, bottom)
   - Grayscale handling
3. Extracts structured semantic content from AI2D ground truth:
   - Diagram objects and labeled segments
   - Diagram elements and BLIP annotations
   - Associated questions
4. Combines annotations, questions, and visual emphasis into a single
   structured semantic target text.
5. Uses a Stage-1 Pix2Struct model to preprocess images into model-compatible
   tensors.
6. Tokenizes semantic targets with padding and masking (`-100`) for loss-safe training.
7. Saves one `.pt` file per image to support efficient, resumable preprocessing.

Intended Use:
- Stage-2 or multimodal fine-tuning of Pix2Struct models
- Learning object recognition, relationships, and visual emphasis in diagrams
- Downstream tasks such as structured diagram understanding or alt-text generation

Output:
Each saved `.pt` file contains:
{
  "pixel_values": Tensor,
  "labels": Tensor,
  "target_text": str
}

All outputs are written to the configured PREPROCESS_DIR.

'''

import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import Pix2StructProcessor, AutoTokenizer
import cv2

# ------------------------------
# PATHS
# ------------------------------
AI2D_ROOT = r"C:/Users/nawaa/Downloads/ai2d-all/ai2d"
PREPROCESS_DIR = r"C:/Users/nawaa/Downloads/ai2d-all/preprocessed"
STAGE1_MODEL = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/model/pix2struct_scicap_stage1_20260122_1946"

IMG_DIR = Path(AI2D_ROOT) / "images"
ANN_DIR = Path(AI2D_ROOT) / "annotations"
Q_DIR = Path(AI2D_ROOT) / "questions"

os.makedirs(PREPROCESS_DIR, exist_ok=True)

# ------------------------------
# LOAD PROCESSOR & TOKENIZER
# ------------------------------
processor = Pix2StructProcessor.from_pretrained(STAGE1_MODEL)
tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL)

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def safe_json_load(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
            if content.startswith('\ufeff'):
                content = content[1:]
            return json.loads(content)
    except:
        return None

def find_json_files(img_stem):
    # Look for annotation JSON
    ann_file = ANN_DIR / f"{img_stem}.json"
    if not ann_file.exists():
        ann_file = None

    # Look for question JSON
    q_file = Q_DIR / f"{img_stem}.json"
    if not q_file.exists():
        q_file = None

    return ann_file, q_file

# ------------------------------
# EMPHASIS DETECTION FUNCTIONS
# ------------------------------
def detect_multitype_emphasis(img: Image.Image):
    img_np = np.array(img)
    is_grayscale = len(img_np.shape) == 2 or (len(img_np.shape) == 3 and img_np.shape[2] == 1)
    
    if is_grayscale:
        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=2)
        img_np = np.repeat(img_np, 3, axis=2)
    
    color_ranges = {
        "red": ((0, 100, 100), (10, 255, 255)),
        "green": ((50, 100, 100), (70, 255, 255)),
        "blue": ((100, 100, 100), (130, 255, 255)),
        "yellow": ((20, 100, 100), (30, 255, 255)),
    }
    
    color_scores = {k: 0.0 for k in color_ranges} if is_grayscale else {
        k: color_emphasis(img_np, np.array(v[0]), np.array(v[1])) for k,v in color_ranges.items()
    }
    
    edge_score = edge_emphasis(img_np)
    contrast_score = high_contrast(img_np)
    size_score = size_emphasis(img_np)
    
    overall_score = (0.1 * max(color_scores.values()) + 0.4 * edge_score +
                     0.3 * contrast_score + 0.2 * size_score) if is_grayscale else \
                    (0.3 * max(color_scores.values()) + 0.3 * edge_score +
                     0.2 * contrast_score + 0.2 * size_score)
    
    present = overall_score > 0.02
    
    if not is_grayscale:
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        sat_mask = (hsv[:,:,1]/255.0) > 0.5
        region = emphasis_region(sat_mask) if present else "none"
    else:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        intensity_mask = gray > 128
        region = emphasis_region(intensity_mask) if present else "none"
    
    return {
        **color_scores,
        "high_contrast": "present" if contrast_score > 0.05 else "none",
        "edge_density": "high" if edge_score > 0.05 else "low",
        "large_object": "yes" if size_score > 0.25 else "no",
        "region": region,
        "present": "yes" if present else "no",
        "overall_score": overall_score,
        "is_grayscale": is_grayscale
    }

def color_emphasis(img_np, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return np.mean(mask>0)

def edge_emphasis(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,100,200)
    return edges.mean()/255.0

def size_emphasis(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0
    areas = [cv2.contourArea(c) for c in contours]
    return max(areas)/(img_np.shape[0]*img_np.shape[1])

def high_contrast(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return gray.std()/255.0

def emphasis_region(mask):
    h,w = mask.shape
    ys,xs = np.where(mask)
    if len(xs)==0: return "none"
    cx,cy = xs.mean()/w, ys.mean()/h
    if 0.33<cx<0.66 and 0.33<cy<0.66: return "center"
    if cx<0.33: return "left"
    if cx>0.66: return "right"
    if cy<0.33: return "top"
    return "bottom"

# ------------------------------
# AI2D EXTRACTION FUNCTIONS
# ------------------------------
def extract_ai2d_objects(annotation):
    if not annotation: return ["No annotations found"]
    objects=[]
    if "segmentationGroups" in annotation:
        for g in annotation["segmentationGroups"]:
            for s in g.get("segments",[]):
                if "text" in s and s["text"]: objects.append(f"SEGMENT: {s['text']}")
    if "diagramElements" in annotation:
        for k,v in annotation["diagramElements"].items():
            obj_text = v.get("text","")
            obj_type = v.get("type","element")
            objects.append(f"{obj_type.upper()}: {obj_text}" if obj_text else f"{obj_type.upper()}: {k}")
    if "blip" in annotation:
        for k,v in annotation["blip"].items():
            if isinstance(v,str) and v.strip(): objects.append(f"BLIP_{k}: {v}")
    seen=set()
    return [o for o in objects if not (o in seen or seen.add(o))] or ["Generic diagram element"]

def extract_ai2d_questions(question_data):
    if not question_data: return ["No questions"]
    questions=[]
    if "questions" in question_data:
        for qid,qinfo in question_data["questions"].items():
            if "question" in qinfo: questions.append(qinfo["question"])
    return questions or ["What is this diagram about?"]

def build_semantic_target(annotation, question_data, emphasis_dict):
    lines=["OBJECTS (from ground truth annotations):"]
    objects = extract_ai2d_objects(annotation)[:15]
    for obj in objects: lines.append(f"- {obj}")
    lines.append("\nRELATIONS AND QUESTIONS:")
    questions = extract_ai2d_questions(question_data)[:5]
    for i,q in enumerate(questions): lines.append(f"Q{i+1}: {q}")
    lines.append("\nVISUAL EMPHASIS (detected):")
    emphasis_summary=[]
    if emphasis_dict.get("is_grayscale",False): emphasis_summary.append("Grayscale diagram")
    if emphasis_dict["present"]=="yes":
        if not emphasis_dict.get("is_grayscale",False):
            colors = ["red","green","blue","yellow"]
            prominent_color=max(colors,key=lambda c:emphasis_dict.get(c,0))
            if emphasis_dict.get(prominent_color,0)>0.05:
                emphasis_summary.append(f"Prominent {prominent_color} elements")
        if emphasis_dict.get("high_contrast")=="present": emphasis_summary.append("High contrast areas")
        if emphasis_dict.get("edge_density")=="high": emphasis_summary.append("Detailed edges")
        if emphasis_dict.get("large_object")=="yes": emphasis_summary.append("Dominant large object")
        emphasis_summary.append(f"Focus region: {emphasis_dict.get('region','unknown')}")
    lines.extend([f"- {e}" for e in emphasis_summary] if emphasis_summary else ["- No strong visual emphasis detected"])
    return "\n".join(lines)

# ------------------------------
# MAIN PREPROCESS LOOP
# ------------------------------
image_paths = sorted(IMG_DIR.glob("*.png"))
print(f"Found {len(image_paths)} images in images folder")

success_count = 0
for i, img_path in enumerate(image_paths):
    if i % 100 == 0:
        print(f"Processing {i}/{len(image_paths)}...")

    out_path = Path(PREPROCESS_DIR) / f"{img_path.stem}.pt"
    if out_path.exists():
        success_count += 1
        continue

    try:
        img_stem = img_path.stem
        ann_file, q_file = find_json_files(img_stem)

        # Load JSON, or use empty dict if missing
        annotation = safe_json_load(ann_file) if ann_file else {}
        question_data = safe_json_load(q_file) if q_file else {}

        # Open image
        with Image.open(img_path) as img:
            # Preserve original mode for emphasis detection
            orig_img = img.copy()

            # Convert to RGB for processor
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize to 224x224
            img = img.resize((224, 224))

            # Convert to NumPy float32 array scaled 0-1
            img_np = np.array(img).astype(np.float32) / 255.0

            # Run processor
            try:
                encoding = processor(images=[img_np], return_tensors="pt")

                # Handle both keys
                if "pixel_values" in encoding:
                    pixel_values = encoding["pixel_values"].squeeze(0)
                elif "flattened_patches" in encoding:
                    pixel_values = encoding["flattened_patches"].squeeze(0)
                else:
                    print(f"Processor returned unexpected keys {encoding.keys()} for {img_path.name}, skipping")
                    continue

            except Exception as e:
                print(f"Processor failed on {img_path.name}: {e}")
                continue

        # Detect visual emphasis
        emphasis = detect_multitype_emphasis(orig_img)

        # Build semantic target text
        target_text = build_semantic_target(annotation, question_data, emphasis)

        # Tokenize target text
        labels = tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        labels[labels == tokenizer.pad_token_id] = -100

        # Save preprocessed tensor
        torch.save({
            "pixel_values": pixel_values,
            "labels": labels,
            "target_text": target_text
        }, out_path)

        success_count += 1

    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")

print(f"\nSuccessfully processed {success_count}/{len(image_paths)} images")
print(f"Saved to: {PREPROCESS_DIR}")