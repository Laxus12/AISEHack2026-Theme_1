# AISEHack2026-Theme_

# 3-Class Flood Segmentation Pipeline

**Prithvi EO v2 (600M + 300M Ensemble) | SAR + GEE | 5-Fold CV | D4 TTA**

---

## Overview

End-to-end pipeline for **3-class flood segmentation** using satellite data. Combines optical imagery with SAR + auxiliary data (DEM, slope, rainfall, water maps) to generate pixel-wise predictions.

---

## Classes

* `0` – No Flood
* `1` – Flood *(primary metric)*
* `2` – Water Body

---

## Pipeline

1. Load satellite TIFs
2. (Optional) Fetch GEE data (DEM, SAR, rainfall, etc.)
3. Merge into 8-channel inputs
4. Train models (Prithvi EO v2 600M & 300M)
5. 5-fold stratified CV
6. Ensemble + TTA (24 passes)
7. Threshold tuning → final predictions

---

## Model

* Backbone: Prithvi EO v2 (600M + 300M)
* Loss: FocalDice
* Input: 8 channels
* Ensemble: 60% (600M) + 40% (300M)

---

## Training

* 5-fold stratified CV
* Heavy augmentation (D4 rotations/flips)
* Class imbalance handled via weighted loss

---

## Inference

* TTA: 8 orientations × 3 scales
* Ensemble averaging
* Threshold sweep to optimize Flood IoU

---

## Metrics

* **Flood IoU (primary)**
* mIoU
* Pixel Accuracy

---

## Submission

* Output: `submission.csv`
* Format: RLE (Flood class only)
* Must include all test IDs

---

## Requirements

* Python ≥ 3.9
* PyTorch + CUDA GPU recommended
* Key libs: `terratorch`, `albumentations`, `rasterio`, `earthengine-api`

---

## Notes

* Optional pseudo-labeling supported
* GEE usage can be disabled
* AI-assisted development with human validation

