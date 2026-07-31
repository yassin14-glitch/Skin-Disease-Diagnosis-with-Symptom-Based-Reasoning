# Skin Disease Diagnosis with Symptom-Based Reasoning

A multimodal deep learning system for dermoscopic skin lesion classification that fuses **image data** with **patient clinical context** (age, sex, lesion location) — rather than relying on the image alone — to produce a diagnosis, per-class confidence scores, a clinically actionable risk tier, and an independent ABCDE symptom check.

> Graduation Project — Digital Egypt Pioneers Initiative (DEPI) × eYouth (MCIT)
> Supervised by Eng. Mahmoud Talaat

---

## Team

| Name | Role |
|---|---|
| Yassin Taher | ML Engineer |
| Rawda Amr | Data Engineer |
| Mariam Akram | Computer Vision Engineer |
| Fady Mamdouh | Deployment Engineer |

---

## Problem Statement

Dermatology faces a shortage of specialists relative to demand, so patients often wait weeks for an in-person evaluation of a suspicious lesion. Image-only classifiers can also be misled, since visually similar lesions can carry very different malignancy risk depending on a patient's age, sex, and lesion site. This project fuses image and clinical evidence and clearly communicates risk level, to help shorten time-to-triage and prioritize urgent cases for both patients and clinicians.

## Approach

A three-stage pipeline:

1. **Capture** — user uploads a dermoscopic photo along with age, sex, and lesion location.
2. **Fuse & Classify** — a multimodal network fuses image features with encoded clinical data to predict one of 7 diagnostic classes.
3. **Risk-Tier & Explain** — the prediction is mapped to a risk tier (Critical / High / Moderate / Low), paired with per-class confidence, Grad-CAM visual explanations, and an independent, rule-based ABCDE symptom checklist — shown separately so it never silently overrides the model's output.

Deployed as a **FastAPI backend** with a drag-and-drop web GUI, intended for teledermatology pre-screening, a second-opinion tool for dermatologists/GPs, and fast initial guidance for patients.

## Architecture

- **Image branch — EfficientNet-B4**: ImageNet-pretrained CNN backbone (via `timm`), fine-tuned end-to-end.
- **Clinical branch — MLP encoder**: `18 → 64 → 32` (ReLU + Dropout 0.2) projecting the structured clinical vector (age, sex, one-hot lesion location) into a 32-dim embedding.
- **Fusion head**: concatenated embeddings → `Linear → ReLU → Dropout 0.3 → Linear` → softmax over 7 classes.

## Dataset

- **Source**: [HAM10000](https://doi.org/10.1038/sdata.2018.161) (ISIC 2018 Task 3) — dermoscopic images + patient metadata.
- **Classes (7)**: Melanoma (MEL), Melanocytic Nevus (NV), Basal Cell Carcinoma (BCC), Actinic Keratosis (AKIEC), Benign Keratosis (BKL), Dermatofibroma (DF), Vascular Lesion (VASC).
- **Size / split**: 10,015 images → stratified 60/20/20 (6,009 / 2,003 / 2,003), verified zero image overlap between splits.
- **Class imbalance**: NV made up over half the dataset (~6,700 images) vs. under 100 for DF — a ~58:1 imbalance, addressed via oversampling + `WeightedRandomSampler`.

## Techniques

- Transfer learning (ImageNet-pretrained EfficientNet-B4)
- Late-fusion multimodal learning
- Grad-CAM explainability
- Test-Time Augmentation (3-view: original, h-flip, v-flip)
- Mixed-precision (AMP) training
- Rule-based ABCDE symptom scoring (kept independent of the model)

**Stack**: Python, PyTorch, timm, Pandas, NumPy, scikit-learn, OpenCV, FastAPI, Google Colab, CUDA

## Experiments

| Experiment | Optimizer | Loss | Scheduler | Best Val Macro F1 |
|---|---|---|---|---|
| **Baseline (selected)** | Adam (lr 1e-4) | Cross-Entropy | ReduceLROnPlateau | **0.8254** |
| Improved | AdamW (lr 1e-4, wd 1e-4) | Focal Loss (γ=2) | Cosine Annealing | 0.7874 |

## Results

Final test-set performance (2,003 held-out images):

- **Accuracy**: 86.9%
- **Macro F1**: 0.786
- **Weighted F1**: 0.870

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| MEL | 0.65 | 0.63 | 0.64 | 223 |
| NV | 0.92 | 0.96 | 0.94 | 1,341 |
| BCC | 0.78 | 0.81 | 0.79 | 103 |
| AKIEC | 0.76 | 0.63 | 0.69 | 65 |
| BKL | 0.82 | 0.68 | 0.74 | 220 |
| DF | 0.89 | 0.74 | 0.81 | 23 |
| VASC | 0.92 | 0.86 | 0.89 | 28 |

**Key findings**: NV was classified most reliably; rare classes (DF, VASC) benefited clearly from oversampling. Melanoma (MEL) — the most clinically dangerous class — remained hardest to classify (0.64 F1), most often confused with NV (69/223 cases), mirroring the difficulty dermatologists face with single-image diagnosis.

This result (86.9% accuracy, 0.786 macro F1) compares favorably to a reported HAM10000 image-only CNN baseline (~85% top-1 accuracy) and typical image-only ResNet/DenseNet studies (macro F1 ~0.65–0.75).

## Limitations & Future Work

Melanoma recall (0.64 F1) still lags and is frequently confused with benign NV — the system is a **preliminary triage aid**, not a replacement for specialist evaluation.

Planned improvements: lesion segmentation pre-classification, confidence calibration (temperature scaling), a patient-guidance chatbot, native mobile deployment, and hybrid heads (EfficientNet features + XGBoost/Random Forest/SVM) to lift minority-class and melanoma recall.

## Repository Structure

```
├── notebooks/        # Training & experimentation notebooks
├── app/              # FastAPI backend + web GUI
├── docs/             # Full reports, documentation, presentation
├── assets/           # Images, diagrams, infographic
└── README.md
```

## Disclaimer

This tool is a research prototype intended for triage support only. It is **not** a certified medical device and should not be used as a substitute for professional dermatological diagnosis.
