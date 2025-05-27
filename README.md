# Dobbe AI - Adaptive Preprocessing Pipeline for IOPA X-ray Images

## Problem Statement

Dobbe AI is building AI-based diagnostic tools using IOPA (Intraoral Periapical) X-rays. These radiographs often vary in brightness, contrast, sharpness, and noise due to different acquisition conditions. A static preprocessing pipeline fails to standardize these variations, reducing model performance.

The goal of this project is to build an adaptive preprocessing pipeline that dynamically adjusts image quality based on the characteristics of the input image. This includes classical image processing techniques and optional ML-based classification.

---

## My Thought Process

### Initial Analysis

I began with a simple exploration of the dataset provided using a notebook (`Initial_Analysis.ipynb`). This helped me:

* Understand the structure and type of images (mostly DICOM)
* Visualize their brightness and contrast distributions
* Identify that some images were noisy or blurry

### Metric-Based Evaluation

To systematically quantify image quality, I implemented metrics for:

* Brightness: Mean intensity
* Contrast: RMS and Michelson contrast
* Sharpness: Laplacian variance and Tenengrad gradient
* Noise: Gaussian blur residual-based standard deviation

### Static Preprocessing

As a baseline, I applied traditional preprocessing steps:

* Histogram equalization
* Unsharp masking (sharpening)
  While these steps worked on some images, they were not robust across diverse quality.

### Adaptive Preprocessing

I then developed an adaptive pipeline that made preprocessing decisions based on quality metrics. Depending on initial conditions:

* Applied CLAHE for low contrast
* Sharpened blurry images
* Denoised images with high noise
* Normalized brightness when over/underexposed

### ML-Based Classification

To make the system more intelligent:

* I auto-labeled images into `good`, `blurry`, `underexposed`, etc. using thresholds
* Trained a RandomForestClassifier to predict these categories based on metrics
* Used predicted labels to route images through appropriate preprocessing

### Data Augmentation

The dataset had only 7 DICOM images, so to improve the classifier:

* I used SRGAN to synthetically generate realistic IOPA-style PNGs
* Added those to the dataset folder (total \~50 images)
* Considered adding more augmentations (brightness/contrast shifts, rotations) for diversity

### Evaluation

To evaluate effectiveness, I computed:

* SSIM and PSNR scores (original vs static, original vs adaptive)
* Observed that adaptive pipeline significantly improved both

### Final Output

* Now, only the adaptive image is saved per input (in the `processed/` folder)
* All metrics and classification results are printed for analysis

---

## Features Implemented

### 1. Input Format Handling

* Reads DICOM (`.dcm`) and JPEG (`.png`) images
* Extracts and visualizes pixel arrays for `.dcm` files using `pydicom`

### 2. Image Quality Metrics Computed

* Brightness: Mean pixel intensity
* Contrast:

  * RMS Contrast (Standard Deviation)
  * Michelson Contrast
* Sharpness:

  * Laplacian Variance
  * Tenengrad (Gradient-based focus measure)
* Noise: Estimated using Gaussian blur residuals

### 3. Static Preprocessing (Baseline)

* Histogram Equalization
* Unsharp Masking (Sharpening Filter)

### 4. Adaptive Preprocessing Pipeline

* CLAHE applied for low contrast images
* Sharpening for blurry images
* Denoising for high noise images
* Normalization for over/underexposed images

### 5. Image Quality Classification (ML-Based)

* Auto-labels image quality based on metric thresholds
* Trains a RandomForestClassifier to classify into categories:

  * `good`, `blurry`, `low_contrast`, `underexposed`, `overexposed`
* Uses predicted label to guide preprocessing

### 6. Evaluation Metrics

* SSIM (Structural Similarity Index)
* PSNR (Peak Signal-to-Noise Ratio)
* Compared: Original vs Static and Original vs Adaptive

### 7. Output

* Only the final adaptively preprocessed image is saved (in `processed/` folder)
* No static or original outputs unless needed for evaluation
* Additionally, side-by-side visual comparisons (original vs adaptive) are saved for all images

---

## Results Snapshot

| Image     | QualityLabel | SSIM (Static) | SSIM (Adaptive) | PSNR (Adaptive) |
| --------- | ------------ | ------------- | --------------- | --------------- |
| DICOM-01  | good         | 0.39          | 0.78            | 32.54           |
| DICOM-02  | good         | 0.25          | 0.76            | 19.44           |
| GAN-Image | blurry       | 0.31          | 0.84            | 33.40           |

Adaptive pipeline consistently outperforms static preprocessing.

---


## Folder Structure

```
project/
├── Image_Generation_GAN.ipynb  
├── Iopa_Preprocessing.py         
├── Initial_Analysis.ipynb        
├── processed/                    
├── sample_images/                
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run the script

python Iopa_Preprocessing.py

### 3. Check outputs

* Processed adaptive images are saved in the `processed/` folder.
* All metrics are printed in the console.

---

## Future Work

* Use a bigger dataset so that the model learns better
* Use GAN in order to improve image quality
* Replace rule-based thresholds with a CNN for quality classification
* Use a U-Net or Autoencoder for image-to-image enhancement
* Integrate with downstream diagnostic AI pipeline

---

## Credits

Developed as part of Dobbe AI’s Data Science Intern Screening Assignment

Author: Anvay Jaykar

Date: May 2025
