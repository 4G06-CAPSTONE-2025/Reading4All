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

### Circuits
For a **circuit** diagram such as image `/circuitDiagram4.png`:
- Recognizes a circuit with voltage source, resistor, and capacitor
- Identifies series/parallel ambiguity
-  **Failure:** Struggles with exact topology such as component count and configuration

### Graphs & Plots
For **graph or plot** diagrams such as images `/densityTemperature.png` and `/energyTime.png`:
- Correctly identifies axes and oscillatory behavior
- Recognizes periodic trends
- Identifies linear trend and data points
- Understands the relationship between variables 
- **Failure:** Some axis confusion (i.e. energy vs time swapped in alt text description of `/energyTime.png`)
- **Failure:** Uses imprecise statistical language: description is vague and general, and does not clearly explain what the data shows in measurable or scientific terms.

### Electricity & Magnetism
For a **Electric field or Magnetic field** diagrams such as image `/electricField.png`:
- Correctly identifies field lines and direction arrows
- Distinguishes positive/negative charge conventions
- Good spatial reasoning for E-field visualization
- Identifies current direction and magnetic field orientation
- Recognizes uniform magnetic fields
- Correctly interprets loop-field interaction diagrams

### Specialized Physics 
For **advanced or specialized** diagrams such as `image3.png`
- Identifies EUV systems and resolution trends
- Recognizes labeled axes and technological progression
- Crystal Lattice Diagrams
- Correctly distinguishes lattice structures (bcc, fcc, hexagonal)
- Understands spatial symmetry visually
❌ Terminology repetition (“euv vs euv”) and semantic drift
❌ Does not consistently name lattice types

### Failure Observed 
- **Semantic drift in long captions:** As captions become longer, the model sometimes loses focus on the original visual content and introduces unrelated or repetitive concepts.
- **Hallucinated entities in abstract diagrams:** In highly abstract or schematic diagrams, the model often mentions objects,  labels, or physical entities that are not actually present in the image.
- **Overuse of generic descriptive phrases:** The model frequently relies on vague wording such as “illustrates the relationship” or “shows a pattern” instead of explicitly describing axes, variables, or physical interpretations.
- **Weak performance on highly theoretical diagrams**: Diagrams involving subatomic particles, quarks, or highly theoretical concepts are often described inaccurately or too generally, suggesting limited domain-specific understanding.

### Overall Strengths and Weeknesses
**Strengths**
- Strong spatial reasoning
- Accurate high-level physics interpretation
- Good handling of arrows, vectors, and field lines

**Weakenesses**
- Lacks equation-level precision
- Inconsistent variable naming
- Weak domain specificity without prompt constraints