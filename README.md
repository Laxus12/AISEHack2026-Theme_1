# AISEHack 2026 - Theme 1: Flood Detection
**Team:** Quantum and Al CDAC CINE  

## Overview
This repository contains the pipeline for **3-class flood segmentation** (No Flood, Flood, Water Body) using satellite data   The project evolved significantly during development, pivoting from a complex ensemble model to a highly optimized, single fine-tuned geospatial foundation model 

---

## 1. Initial Approach (3_Class_Flood_Pixel_Segmentation.ipynb)
Our initial development strategy focused on a heavy, complex pipeline aiming for maximum theoretical performance:
* **Architecture:** An ensemble of Prithvi EO v2 600M (60% weight) and 300M (40% weight) models.
* **Inputs:** 8-channel inputs combining optical imagery with SAR and GEE-derived auxiliary data (DEM, slope, rainfall, water maps)
* **Training Strategy:** 5-fold stratified cross-validation with optional pseudo-labeling.
* **Inference:** Heavy Test-Time Augmentation (D4 TTA: 8 orientations × 3 scales for 24 passes per patch) and extensive threshold sweeping.

### Why We Pivoted
The initial approach encountered critical silent data failures and pipeline roadblocks 
* **Data Retrieval Failures:** The 3-day GEE date window for Sentinel-1 resulted in zero overpasses for most patches, silently returning arrays of zeros without throwing errors
* **Overfitting & Generalization:** Fine-tuning massive backbones (600M/300M) on the limited ~40-patch labelled corpus proved highly unstable, particularly when attempting to adapt to the 8-channel input distribution  
* **Pipeline Complexity:** Pseudo-labeling required a clean base model run, which was blocked by the SAR zero-fill failure 

---

## 2. Final Submission Pipeline (composite_focal+diceloss .ipynb)
The final submitted pipeline emphasizes stability, efficient transfer learning, and precise data handling over raw model scale

### Architecture
*  **Model Base:** Prithvi EO v2 Tiny-TL Vision Transformer
*  **Decoder:** UperNet decoder with 128 channels 
*  **Strategy:** **Backbone Freezing.** The pre-trained Prithvi encoder weights are frozen entirely to prevent catastrophic forgetting on the small dataset  The UperNet decoder and segmentation head adapt exclusively to the task 

### Data & Input Processing
*  **Input Features:** 6-channel multi-spectral input (HH, HV, Green, Red, NIR, SWIR) natively sourced from the IBM Flood Dataset Phase 2 
*  **Data Loader:** TerraTorch's `GenericNonGeoSegmentationDataModule` 
*  **Augmentation Suite:** D4 symmetry augmentations (random flips and 90° rotations), ShiftScaleRotate, RandomBrightnessContrast, Gauss Noise, and Coarse Dropout (4-8 rectangular holes of 32-64 px)

### Training Configuration
 To combat the severe class imbalance (~65% No-Flood, ~15% Flood, ~20% Water Body), the training loop employs targeted strategies 
*  **Loss Function:** FocalDice Loss : 12]. 
*  **Inverse-Frequency Weighting:** Class weights are computed analytically from pixel-frequency distribution and applied to the loss function (No-Flood: 0.51, Flood: 2.19, Water Body: 1.71)  

| Parameter | Value |
| :--- | :--- |
| **Batch Size** |  4 
| **Optimizer** |  AdamW (weight decay = 0.05) 
| **Learning Rate** |  1e-4  
| **LR Scheduler** |  ReduceLROnPlateau (factor = 0.5, patience = 20) 
|  **Precision** | bf16-mixed
| **Max Epochs** |  100 (Early Stopping Patience = 50) 

### Inference & Submission Format
*  **TTA Removed:** Test-Time Augmentation was removed from the inference path to simplify prediction and reduce inference time  
*  **Submission Generator:** A dedicated post-processing module converts predicted GeoTIFF masks to the required Column-Major RLE format 
*  **Validation:** The generator includes a built-in RLE round-trip verification (encode -> decode -> compare) 
*  **Empty Mask Handling:** Explicit `0 0` encoding is correctly produced for flood-free patches

---

## 3. Requirements
* Python 3.9+
* PyTorch + CUDA GPU (bf16-mixed support recommended)
* Key Libraries: `terratorch`, `albumentations`, `rasterio`

## 4. AI Assistance Declaration
 This project made use of Claude (Anthropic, claude-sonnet-4.6) as an AI coding assistant during Phase 2 development.  Claude acted primarily as a technical accelerator for reviewing code, updating loss formulations, refactoring the augmentation pipeline, and implementing the RLE submission generator.  All architectural decisions, training configurations, and scientific methodology remain entirely the work of Team Quantum and Al.
