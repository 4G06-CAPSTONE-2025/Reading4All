# ai/inference/

This folder contains the inference pipeline for running the fine-tuned BLIP alt-text generation model on new images.

---

## Folder Structure

| Folder | Description |
|---|---|
| `POC/` | Early proof-of-concept inference scripts |
| `rev0/` | First revision — batch inference using pretrained BLIP as a baseline |
| `rev1/` | Latest revision — inference using the fine-tuned model (use this one) |

Root-level files:

| File | Description |
|---|---|
| `README.md` | Existing inference-specific notes |
| `config.yaml` | Model and path configuration — edit this to point to your model and data |
| `requirements.txt` | Inference-specific Python dependencies |

---

## Quickstart

### 1. Set up a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install torch torchvision transformers pillow pyyaml
```

### 3. Configure paths

Edit `config.yaml` to set:

```yaml
MODEL_DIR: "PATH TO THE MODEL FOLDER"
IMAGE_DIR: "IMAGE INPUT PATH"
LOGGER_PATH: "PATH TO A FOLDER FOR LOGS"
PROMPT: "Describe this physics diagram:"
```

### 4. Run inference

```bash
python3 ai/inference/rev1/inference.py
```

### 5. View output

Generated captions are written to the log file specified in `LOGGER_PATH`, one line per image in the format:

```
<image_name>: <caption>
```

If an image fails, the log will contain `ERROR` followed by a stack trace, and the script will continue to the next image.

---

## How It Works

1. Loads the BLIP processor and fine-tuned model from `MODEL_DIR`
2. Scans `IMAGE_DIR` for image files (`.png`, `.jpg`, `.jpeg`, `.webp`)
3. For each image: opens it with PIL, runs BLIP generation using the configured `PROMPT`, decodes the output into a caption string
4. Appends each result to the log file
5. Handles per-image errors gracefully without stopping the batch

---

## Model Architecture (BLIP)

The deployed model is based on `Salesforce/blip-image-captioning-base`, which combines two transformer components:

- **Vision Transformer (ViT)** — splits the input image into flattened patches with positional encodings and encodes them into visual feature representations
- **GPT-2 language model decoder** — takes the visual features as input and generates a caption token by token

The ViT extracts visual features and GPT-2 produces a natural language description. The fine-tuning process adapts this base model specifically to technical and educational diagrams.

---

## Notes

- `rev0/` uses the pretrained model only — useful for baseline comparisons
- `rev1/` uses the fine-tuned checkpoint — use this for production or evaluation
- `POC/` is kept for reference but should not be used for new runs
