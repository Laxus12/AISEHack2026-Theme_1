# AISEHack 2026 - Theme 1: Flood Detection
**Team:** Quantum and Al CDAC CINE [cite: 3]

## Overview
This repository contains the pipeline for **3-class flood segmentation** (No Flood, Flood, Water Body) using satellite data[cite: 7, 8].  The project evolved significantly during development, pivoting from a complex ensemble model to a highly optimized, single fine-tuned geospatial foundation model[cite: 7, 52]. 

---

## 1. Initial Approach (Archived)
Our initial development strategy focused on a heavy, complex pipeline aiming for maximum theoretical performance:
* **Architecture:** An ensemble of Prithvi EO v2 600M (60% weight) and 300M (40% weight) models.
* **Inputs:** 8-channel inputs combining optical imagery with SAR and GEE-derived auxiliary data (DEM, slope, rainfall, water maps)[cite: 52].
* **Training Strategy:** 5-fold stratified cross-validation with optional pseudo-labeling.
* **Inference:** Heavy Test-Time Augmentation (D4 TTA: 8 orientations × 3 scales for 24 passes per patch) and extensive threshold sweeping.

### Why We Pivoted
The initial approach encountered critical silent data failures and pipeline roadblocks[cite: 75, 76]:
* **Data Retrieval Failures:** The 3-day GEE date window for Sentinel-1 resulted in zero overpasses for most patches, silently returning arrays of zeros without throwing errors[cite: 77]. 
* **Overfitting & Generalization:** Fine-tuning massive backbones (600M/300M) on the limited ~40-patch labelled corpus proved highly unstable, particularly when attempting to adapt to the 8-channel input distribution[cite: 82]. 
* **Pipeline Complexity:** Pseudo-labeling required a clean base model run, which was blocked by the SAR zero-fill failure[cite: 82].

---

## 2. Final Submission Pipeline
The final submitted pipeline emphasizes stability, efficient transfer learning, and precise data handling over raw model scale[cite: 10, 52]. 

### Architecture
*  **Model Base:** Prithvi EO v2 Tiny-TL Vision Transformer[cite: 16].
*  **Decoder:** UperNet decoder with 128 channels[cite: 18].
*  **Strategy:** **Backbone Freezing.** The pre-trained Prithvi encoder weights are frozen entirely to prevent catastrophic forgetting on the small dataset[cite: 10, 11].  The UperNet decoder and segmentation head adapt exclusively to the task[cite: 10]. 

### Data & Input Processing
*  **Input Features:** 6-channel multi-spectral input (HH, HV, Green, Red, NIR, SWIR) natively sourced from the IBM Flood Dataset Phase 2[cite: 8, 27, 52]. 
*  **Data Loader:** TerraTorch's `GenericNonGeoSegmentationDataModule`[cite: 27].
*  **Augmentation Suite:** D4 symmetry augmentations (random flips and 90° rotations), ShiftScaleRotate, RandomBrightnessContrast, Gauss Noise, and Coarse Dropout (4-8 rectangular holes of 32-64 px)[cite: 13].

### Training Configuration
 To combat the severe class imbalance (~65% No-Flood, ~15% Flood, ~20% Water Body), the training loop employs targeted strategies[cite: 29]:
*  **Loss Function:** FocalDice Loss[cite: 12]. 
*  **Inverse-Frequency Weighting:** Class weights are computed analytically from pixel-frequency distribution and applied to the loss function (No-Flood: 0.51, Flood: 2.19, Water Body: 1.71)[cite: 12, 30]. 

| Parameter | Value |
| :--- | :--- |
| **Batch Size** |  4 [cite: 25] |
| **Optimizer** |  AdamW (weight decay = 0.05) [cite: 25] |
| **Learning Rate** |  1e-4 [cite: 25] |
| **LR Scheduler** |  ReduceLROnPlateau (factor = 0.5, patience = 20) [cite: 25] |
|  **Precision** | bf16-mixed [cite: 25] |
| **Max Epochs** |  100 (Early Stopping Patience = 50) [cite: 25] |

### Inference & Submission Format
*  **TTA Removed:** Test-Time Augmentation was removed from the inference path to simplify prediction and reduce inference time[cite: 55]. 
*  **Submission Generator:** A dedicated post-processing module converts predicted GeoTIFF masks to the required Column-Major RLE format[cite: 14].
*  **Validation:** The generator includes a built-in RLE round-trip verification (encode -> decode -> compare)[cite: 14].
*  **Empty Mask Handling:** Explicit `0 0` encoding is correctly produced for flood-free patches[cite: 14, 42].

---

## 3. Requirements
* Python 3.9+
* PyTorch + CUDA GPU (bf16-mixed support recommended)
* Key Libraries: `terratorch`, `albumentations`, `rasterio`

## 4. AI Assistance Declaration
 This project made use of Claude (Anthropic, claude-sonnet-4.6) as an AI coding assistant during Phase 2 development[cite: 57].  Claude acted primarily as a technical accelerator for reviewing code, updating loss formulations, refactoring the augmentation pipeline, and implementing the RLE submission generator[cite: 58, 63, 64, 65].  All architectural decisions, training configurations, and scientific methodology remain entirely the work of Team Quantum and Al[cite: 67, 70].
