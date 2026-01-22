## AI Development Schedule  

### Week 1 (Jan 5-11, 2026) -> Physics Dataset Labeling + SciCap Baseline  
Goal: Create a small, high-quality physics-only ground-truth set and train a SciCap baseline checkpoint.  
Tasks:  
- Dataset labeling (Physics):  
    - Labeled 100 physics images (STEM/physics-relevant diagrams/figures).  
    - Wrote human alt-text for each image in a CSV file (ground-truth dataset).  
    - Two team members completed labeling + alt-text writing (internal team deliverable due Sunday).  
- Baseline training:  
    - One team member trained / fine-tuned the model on SciCap data (baseline checkpoint).  
Outcome:  
- `physics_alt_text.csv` with 100 rows (image reference + human-written alt-text).  
- SciCap baseline checkpoint ready for rollout and comparison against the labeled physics set.  

### Week 2 (Jan 12-18, 2026) -> Physics Fine-Tuning (Three-Model Experiment) (Due Sun Jan 18)  
Goal: Fine-tune on physics-labeled data using three training strategies and compare which improves caption quality the most.  
Tasks:  
- Physics labeled data training:  
    - Use the internally labeled physics dataset (ground-truth CSV) as training data.  
- Relabeled set using tester feedback:  
    - Prepare a relabeled subset based on tester feedback on previous alt-text.  
    - Use 50 images for this tester-feedback relabeled set.  
- Three-model experiment:  
    - Train Model A: trained on the original physics-labeled ground-truth set.  
    - Train Model B: trained on the tester-feedback relabeled set (50-image set).  
    - Train Model C: trained on a combined dataset (original physics-labeled set + tester-feedback relabeled set).  
- Save checkpoints + training logs for all runs (loss curves + run configs).  
Outcome:  
- Three physics-focused checkpoints (Model A vs Model B vs Model C).  
- Comparison notes on which training strategy improves correctness, clarity, and appropriate length/detail.  

### Week 3 (Jan 19-25, 2026) -> Testing + Model Selection (Compare All Three)  
Goal: Evaluate Model A vs Model B vs Model C using the physics ground-truth and pick the best checkpoint for gating + human study.  
Tasks:  
- Run side-by-side inference tests across representative physics diagrams/figures.  
- Evaluate against `physics_alt_text.csv` using:  
    - Internal human review (primary for selection)  
    - BLEU/ROUGE-L (secondary reference)  
- Identify failure modes per model (too vague, misses key physics relationships, too long, symbol-heavy, not accessible).  
- Select the best checkpoint to proceed with (and document why).  
Outcome:  
- Final selected physics checkpoint for downstream gating + evaluation.  
- Documented failure modes to drive WCAG + internal metric rules.  

### Week 4 (Jan 26-Feb 1, 2026) -> Accessibility & Internal Metric Compliance (WCAG + Gating)  
Goal: Ensure captions are WCAG-aligned and meet internal format/verbosity requirements before human testing.  
Tasks:  
- WCAG accessibility checks:  
    - Implement rule-based validation for:  
        - Readable sentence structure  
        - Removal of inaccessible symbols  
        - Avoidance of color-only references  
- Internal metric enforcement:  
    - Define and enforce length constraints by image type  
    - Apply verbosity rules to prevent overly detailed outputs  
- Output gating:  
    - Automatically reject or regenerate captions that fail checks  
    - Ensure only compliant captions proceed downstream  
Outcome:  
- WCAG-aligned alt text outputs.  
- Length and verbosity constraints enforced.  
- Gating + compliance logging ready for user testing.  

### Week 5 (Feb 2-8, 2026) -> Human-in-the-Loop Evaluation  
Goal: Collect structured human ratings on caption quality, accessibility, and learning usefulness.  
Tasks:  
- Tester study setup:  
    - Recruit testers to evaluate generated alt text  
    - Provide clear rating instructions using defined Likert scales  
- Metric collection:  
    - Collect ratings for:  
        - Learning objective sufficiency  
        - Length appropriateness  
        - Detail sufficiency  
        - Accessibility format  
        - Usability (0–4)  
        - Learning impact (0–3)  
- Qualitative feedback:  
    - Capture open-ended tester comments  
    - Identify recurring strengths and failure cases  
Outcome:  
- Human-evaluated accessibility dataset.  
- Annotated examples of successful and unsuccessful captions.  

### Week 6 (Feb 9-15, 2026) -> Feedback Metrics & Feedback Loop  
Goal: Turn tester feedback into actionable improvements without requiring full retraining.  
Tasks:  
- Metric aggregation:  
    - Aggregate quantitative tester scores  
    - Summarize qualitative feedback themes  
- Feedback module implementation:  
    - Create a feedback metrics processing module  
    - Analyze trends across image types and caption styles  
- Feedback loop logic:  
    - Adjust prompts and constraints based on tester feedback  
    - Refine generation behavior without full model retraining  
Outcome:  
- Operational feedback loop.  
- Prompt-level model improvements applied.  
- Metrics actively used (learning objective sufficiency, accessibility format, usability, learning impact).  

### Week 7 (Feb 16-22, 2026) -> Evaluation, Results, Integration, and Freeze  
Goal: Finalize system for submission with stable end-to-end behavior and final metrics captured.  
Tasks:  
- Baseline comparison:  
    - Evaluate model outputs using BLEU and ROUGE-L  
    - Compare baseline BLIP vs fine-tuned + gated system  
- Metric analysis:  
    - Analyze correlations between human-rated metrics  
    - Identify trade-offs between length, detail, and usability  
- Results visualization:  
    - Generate tables and figures for report  
    - Summarize quantitative and qualitative findings  
- System integration + freeze:  
    - Lock frontend, backend, and model versions  
    - Run end-to-end pipeline tests and handle edge cases  
    - Capture final metric results and freeze system state for submission  
Outcome:  
- Complete results and analysis section.  
- Fully integrated, stable system.  
- Submission-ready capstone project with finalized metrics.  
