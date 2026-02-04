# Moonbeam-2 Research 

## 1. What is Moonbeam-2?

Moonbeam-2 is a *text-only large language model (LLM)* designed for high-quality natural language generation, reasoning, and rewriting tasks. Unlike multimodal models such as BLIP, Moonbeam-2 *does not process images* directly. Instead, it operates purely on textual inputs and outputs, making it well-suited for tasks that require linguistic precision, coherence, and domain-aware explanation.

Key characteristics of Moonbeam-2:

* **Causal language model** (decoder-only architecture)
* Optimized for **instruction-following and rewriting** tasks
* Strong performance on **structured explanations**, summaries, and refinements
* Smaller in size compared to very large LLMs which makes it feasible to fine-tune on small and curated datasets 

Overall, Moonbeam-2 is the best at taking *imperfect or rough text* and transforming it into clearer, more accurate, and more readable language. This makes it particularly valuable in educational and accessibility-focused pipelines.

---

## 2. What Moonbeam-2 Can't Do

What Moonbeam-2 does **not** do:

* It **cannot ingest images** or visual features
* It **cannot perform image captioning on its own**
* It doesn't replace vision–language models such as BLIP

Therefore, Moonbeam-2 shouldn't be viewed as a standalone solution for image-to-text generation. Instead, it functions best as a **second-stage language refinement model**.

---

## 3. How Moonbeam-2 Fits in the Reading4ALL System: 

### Project Goal (Context)

The goal of this project is to generate **high-quality alt text for academic images**, currently for physics diagrams. Therefore, effective alt text in this context must:

* Accurately describe visual content
* Use **domain-appropriate terminology**
* Be clear, structured, and accessible for screen readers
* Avoid ambiguity, hallucination, or unnecessary verbosity

### Using a Two-Stage Model

A single model rarely excels at both **visual grounding** and **pedagogically strong language generation**, especially with limited training data. For this reason, a two-stage approach is more robust:

1. **First Stage: Vision–Language Model (e.g., BLIP)**

   * Converts the image into a *draft caption*
   * Grounds the description in actual visual content

2. **Second Stage: Moonbeam-2 (Text-Only Refinement)**

   * Takes the draft caption as input
   * Rewrites it into polished, accurate, and accessible academic alt text

Moonbeam-2 fits  into **stage two** of this pipeline.

---

## 4. Why Moonbeam-2 Is Well-Suited for Alt Text Refinement

Moonbeam-2 aligns with the needs of academic alt text generation in several key ways:

### Language Precision

Academic alt text often requires careful phrasing such as defining variables, describing axes, or explaining relationships. Moonbeam-2 performs well at:

* Clarifying sentence structure
* Reducing redundancy
* Maintaining logical flow

### Accessibility-Oriented Rewriting

Alt text must be understandable when read aloud by screen readers. Therefore, Moonbeam-2 can be trained to:

* Expand abbreviations
* Use explicit relational language 
* Avoid vague references like “this” or “that”

### Small Dataset Compatibility

Given a dataset of approximately 500 curated examples (currently for Rev 0), Moonbeam-2 is a strong choice because:

* It can be fine-tuned with **low learning rates**
* It benefits from **instruction-style prompts**
* It generalizes well when used as a rewriting model rather than a generator from scratch

---

## 5. Conceptual Pipeline Using Moonbeam-2

The intended workflow for this project is:

```
Academic Image
   ↓
Vision–Language Model (BLIP/pix2struct for Rev 0)
   ↓
Draft Caption
   ↓
Moonbeam-2 (Fine-Tuned)
   ↓
Final Academic Alt Text
```

In this design, Moonbeam-2 acts as a **language specialist**, ensuring the final alt text meets academic, pedagogical, and accessibility standards.

---

## Reason for research 
- Moonbeam 2 was reccommended to the team by Fazmin (An AI specialist at McMaster)
- To use as a candidate model for the project 