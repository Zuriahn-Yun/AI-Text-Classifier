# AI Text Classifier Findings

This repository includes a completed notebook with outputs at:

- `notebooks/classifier_completed.ipynb`

## Dataset

- Combined HC3 and GPT-wiki-intro text samples.
- After filtering short / non-English texts, the final dataset contains **34,579 samples**:
  - **AI:** 19,540
  - **Human:** 15,039

## Features

The classifier uses **39 statistical / linguistic features**, including:

- length and sentence statistics
- burstiness
- type-token ratio
- punctuation ratios
- readability metrics
- POS-tag ratios
- GPT-2 perplexity

## Model comparison

Validation performance:

- Logistic Regression: accuracy **0.9618**, macro F1 **0.9611**
- Random Forest: accuracy **0.9800**, macro F1 **0.9797**
- XGBoost: accuracy **0.9881**, macro F1 **0.9879**

## Best model

**XGBoost** performed best.

- 5-fold CV macro F1: **0.9894 ± 0.0014**
- Test macro F1: **0.9893**
- Test set size: **6,916** samples
- Test classification report showed about **0.99 precision / recall / F1** for both human and AI classes.

## Inference smoke test

The saved model round-tripped successfully and produced high-confidence predictions on sample AI and human texts.
