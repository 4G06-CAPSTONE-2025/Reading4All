# ai/modelExperiment_task4/

This folder contains inference experiment scripts written as part of Task 4, used to evaluate and compare multiple model candidates before selecting the final deployed model.

---

## Files

| File | Description |
|---|---|
| `llamafactory_inference.py` | Runs inference using the LLaMAFactory model |
| `moondream_inference.py` | Runs inference using the Moondream model |
| `paligemma_inference.py` | Runs inference using the PaliGemma model |

---

## Purpose

These scripts were written to test three alternative model candidates side-by-side against the BLIP baseline. Each script follows the same structure: load the model, run it on a set of test images, and output captions for comparison.

Results from these runs are stored in `ai/logger/`.

---

## Notes

- These scripts are experiment artifacts from Task 4 — they are not part of the production inference pipeline
- The final deployed model is BLIP; see `ai/inference/` for the production scripts
- LLaMAFactory, Moondream, and PaliGemma were ruled out based on caption quality and performance trade-offs observed in these runs
