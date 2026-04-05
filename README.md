# AISEHack 2026 - Theme 1: Flood Detection
[cite_start]**Team:** Quantum and Al CDAC CINE [cite: 3]

## Overview
[cite_start]This repository contains the pipeline for **3-class flood segmentation** (No Flood, Flood, Water Body) using satellite data[cite: 7, 8]. [cite_start]The project evolved significantly during development, pivoting from a complex ensemble model to a highly optimized, single fine-tuned geospatial foundation model[cite: 7, 52]. 

---

## 1. Initial Approach (Archived)
Our initial development strategy focused on a heavy, complex pipeline aiming for maximum theoretical performance:
* **Architecture:** An ensemble of Prithvi EO v2 600M (60% weight) and 300M (40% weight) models.
* [cite_start]**Inputs:** 8-channel inputs combining optical imagery with SAR and GEE-derived auxiliary data (DEM, slope, rainfall, water maps)[cite: 52].
* **Training Strategy:** 5-fold stratified cross-validation with optional pseudo-labeling.
* **Inference:** Heavy Test-Time Augmentation (D4 TTA: 8 orientations × 3 scales for 24 passes per patch) and extensive threshold sweeping.

### Why We Pivoted
[cite_start]The initial approach encountered critical silent data failures and pipeline roadblocks[cite: 75, 76]:
* [cite_start]**Data Retrieval Failures:** The 3-day GEE date window for Sentinel-1 resulted in zero overpasses for most patches, silently returning arrays of zeros without throwing errors[cite: 77]. 
* [cite_start]**Overfitting & Generalization:** Fine-tuning massive backbones (600M/300M) on the limited ~40-patch labelled corpus proved highly unstable, particularly when attempting to adapt to the 8-channel input distribution[cite: 82]. 
* [cite_start]**Pipeline Complexity:** Pseudo-labeling required a clean base model run, which was blocked by the SAR zero-fill failure[cite: 82].

---

## 2. Final Submission Pipeline
[cite_start]The final submitted pipeline emphasizes stability, efficient transfer learning, and precise data handling over raw model scale[cite: 10, 52]. 

### Architecture
* [cite_start]**Model Base:** Prithvi EO v2 Tiny-TL Vision Transformer[cite: 16].
* [cite_start]**Decoder:** UperNet decoder with 128 channels[cite: 18].
* [cite_start]**Strategy:** **Backbone Freezing.** The pre-trained Prithvi encoder weights are frozen entirely to prevent catastrophic forgetting on the small dataset[cite: 10, 11]. [cite_start]The UperNet decoder and segmentation head adapt exclusively to the task[cite: 10]. 

### Data & Input Processing
* [cite_start]**Input Features:** 6-channel multi-spectral input (HH, HV, Green, Red, NIR, SWIR) natively sourced from the IBM Flood Dataset Phase 2[cite: 8, 27, 52]. 
* [cite_start]**Data Loader:** TerraTorch's `GenericNonGeoSegmentationDataModule`[cite: 27].
* [cite_start]**Augmentation Suite:** D4 symmetry augmentations (random flips and 90° rotations), ShiftScaleRotate, RandomBrightnessContrast, Gauss Noise, and Coarse Dropout (4-8 rectangular holes of 32-64 px)[cite: 13].

### Training Configuration
[cite_start]To combat the severe class imbalance (~65% No-Flood, ~15% Flood, ~20% Water Body), the training loop employs targeted strategies[cite: 29]:
* [cite_start]**Loss Function:** FocalDice Loss[cite: 12]. 
* [cite_start]**Inverse-Frequency Weighting:** Class weights are computed analytically from pixel-frequency distribution and applied to the loss function (No-Flood: 0.51, Flood: 2.19, Water Body: 1.71)[cite: 12, 30]. 

| Parameter | Value |
| :--- | :--- |
| **Batch Size** | [cite_start]4 [cite: 25] |
| **Optimizer** | [cite_start]AdamW (weight decay = 0.05) [cite: 25] |
| **Learning Rate** | [cite_start]1e-4 [cite: 25] |
| **LR Scheduler** | [cite_start]ReduceLROnPlateau (factor = 0.5, patience = 20) [cite: 25] |
| [cite_start]**Precision** | bf16-mixed [cite: 25] |
| **Max Epochs** | [cite_start]100 (Early Stopping Patience = 50) [cite: 25] |

### Inference & Submission Format
* [cite_start]**TTA Removed:** Test-Time Augmentation was removed from the inference path to simplify prediction and reduce inference time[cite: 55]. 
* [cite_start]**Submission Generator:** A dedicated post-processing module converts predicted GeoTIFF masks to the required Column-Major RLE format[cite: 14].
* [cite_start]**Validation:** The generator includes a built-in RLE round-trip verification (encode -> decode -> compare)[cite: 14].
* [cite_start]**Empty Mask Handling:** Explicit `0 0` encoding is correctly produced for flood-free patches[cite: 14, 42].

---

## 3. Requirements
* Python 3.9+
* PyTorch + CUDA GPU (bf16-mixed support recommended)
* Key Libraries: `terratorch`, `albumentations`, `rasterio`

## 4. AI Assistance Declaration
[cite_start]This project made use of Claude (Anthropic, claude-sonnet-4.6) as an AI coding assistant during Phase 2 development[cite: 57]. [cite_start]Claude acted primarily as a technical accelerator for reviewing code, updating loss formulations, refactoring the augmentation pipeline, and implementing the RLE submission generator[cite: 58, 63, 64, 65]. [cite_start]All architectural decisions, training configurations, and scientific methodology remain entirely the work of Team Quantum and Al[cite: 67, 70].
