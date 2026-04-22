# ai/

This folder contains all AI-related work for Reading4All — including dataset annotations, model training, inference pipelines, experiment logs, and shared utilities.

The final deployed model is a custom fine-tuned vision-language model trained to generate descriptive alt-text for technical and educational diagrams.

---

## Folder Structure

| Folder | Purpose |
|---|---|
| `annotations/` | Annotated CSV datasets used for training and evaluation |
| `inference/` | Inference scripts for running the trained model on new images |
| `train/` | Training scripts for fine-tuning and comparing model variants |
| `logger/` | Inference output logs from model runs |
| `modelExperiment_task4/` | Inference experiment scripts for Task 4 model candidates |
| `results_log/` | Markdown summaries of training and inference results |
| `utils/` | Shared utility scripts for data splitting and validation |
| `largeModelTesting/` | ⚠️ Archived — early experiments with large models (Moondream, PaliGemma). No longer active. |

---

## Models Explored

The team evaluated and trained across several vision-language models before arriving at the final fine-tuned model:

- **BLIP** (`Salesforce/blip-image-captioning-base`) — primary model, fine-tuned and deployed
- **Pix2Struct** — trained as a comparison baseline
- **LLaMAFactory** — tested for inference
- **Moondream** — tested for inference (archived)
- **PaliGemma** — tested for inference (archived)

---

## Requirements

Python 3.9+ is recommended. Install core dependencies:

```bash
pip install torch torchvision transformers pillow pyyaml
```

See individual subfolder READMEs for model-specific dependencies.

---

## Entry Points

| Task | Script |
|---|---|
| Run inference on images | `ai/inference/rev1/` |
| Train / fine-tune a model | `ai/train/train.py` |
| Split dataset for training | `ai/utils/splitData.py` |
| Validate image paths | `ai/utils/validateImagePaths.py` |
