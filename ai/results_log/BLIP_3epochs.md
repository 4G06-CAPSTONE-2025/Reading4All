# BLIP (3 Epochs) – Training & Inference Results

## Model Overview
- **Model:** Salesforce/blip-image-captioning-base (3 epochs)

---

## Training Configuration

The model was fine-tuned using the following `TrainingArguments`:

```python
TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    num_train_epochs=3,
    warmup_ratio=0.05,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    remove_unused_columns=False,
    fp16=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)
```

## Inference Results

Below are representative outputs from the model using the prompt: “describe this physics diagram:”

### Mechanics & Vectors
For an **inclined plane** diagram such as image `/blockDiagram3.png`:
- Identifies a block on an inclined surface
- Recognizes force vectors and direction arrows
- Understands qualitative force interactions
- **Failure:** Occasionally vague about vector labels and components

For an **oscillating or vector** diagram such as image `/angularVelocity.png`:
- Correctly identifies oscillation and perpendicular force directions
- Understands spatial relationships between vectors
- **Failure:** Overgeneralizes motion as “oscillating in all directions”

### Circuits & Electronics


### Graphs & Plots

### Electricity & Magnetism

### Specialized Physics 

### Failure Observed 

### Overall Strengths and Weeknesses