# Brain Tumor Classification Using Deep Learning

## Project Overview

Brain tumors require early and accurate diagnosis to improve treatment outcomes. Manual MRI interpretation can be time-consuming and subject to human error, particularly under heavy clinical workloads.

This project develops a Deep Learning-based Brain Tumor Classification system that automatically classifies MRI scans into four categories:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

A custom Convolutional Neural Network (CNN) was built and deployed through a Streamlit web application for real-time predictions.

---

## Business Problem

Radiologists and healthcare professionals often review large volumes of MRI scans. Delayed or incorrect diagnosis can significantly impact patient outcomes.

The objective of this project is to:

- Automate MRI image classification
- Reduce diagnostic workload
- Improve early tumor detection
- Minimize false negatives through recall-focused evaluation
- Provide a simple deployment interface for end users

---

## Dataset Information

### Source

Brain MRI Dataset

### Dataset Statistics

| Metric | Value |
|----------|----------|
| Raw Images | 8,745 |
| Duplicate Images Removed | 1,110 |
| Final Images | 7,635 |
| Classes | 4 |

### Class Distribution

| Class | Images |
|---------|---------|
| Glioma | 3,067 |
| Meningioma | 3,042 |
| Pituitary | 968 |
| No Tumor | 558 |

---

## Data Cleaning

To improve model quality:

- MD5 hashing was used to identify duplicate images
- 1,110 duplicate MRI scans were removed
- Cross-class duplicates were eliminated
- Contradictory labels were prevented

---

## Data Preprocessing

### Image Processing

- Image Resize: 224 × 224
- RGB Conversion
- Dataset-wide Z-Score Normalization

### Normalization Formula

```python
normalized = (image / 255 - mean) / std
```

Benefits:

- Centers data around zero
- Standardizes pixel distribution
- Improves model convergence

---

## Deep Learning Architecture

### Custom CNN Architecture

Input Layer

224 × 224 × 3

Block 1

- Conv2D (32)
- Batch Normalization
- Conv2D (32)
- Max Pooling
- Dropout (0.25)

Block 2

- Conv2D (64)
- Batch Normalization
- Conv2D (64)
- Max Pooling
- Dropout (0.25)

Block 3

- Conv2D (128)
- Batch Normalization
- Conv2D (128)
- Max Pooling
- Dropout (0.40)

Classification Head

- Global Average Pooling
- Dense (256)
- Dropout (0.50)
- Dense (4)
- Softmax Activation

### Total Parameters

249,000 Trainable Parameters

---

## Data Augmentation

The following augmentation techniques were applied:

- Rotation (±20°)
- Width Shift (15%)
- Height Shift (15%)
- Zoom (15%)
- Horizontal Flip
- Brightness Variation (±15%)

---

## Training Strategy

### Class Balancing

```python
class_weight = "balanced"
```

### Callbacks

- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

### Training Summary

| Metric | Value |
|----------|----------|
| Epochs Run | 46 |
| Best Epoch | 36 |
| Validation Accuracy | 93.25% |

---

## Evaluation Philosophy

In healthcare applications, missing a tumor is more dangerous than raising a false alarm.

Therefore, F2-Score was selected as the primary metric because it emphasizes Recall more than Precision.

### F2 Formula

```text
Fβ = (1 + β²) × (Precision × Recall)
     --------------------------------
      β² × Precision + Recall

β = 2
```

Recall receives 4× more weight than Precision.

---

## Model Performance

### Overall Results

| Metric | Score |
|----------|----------|
| Accuracy | 93.31% |
| Weighted F2 Score | 93.29% |
| Validation Dataset | 1,525 Images |

---

## Per-Class Performance

| Class | Precision | Recall | F2 Score |
|---------|---------|---------|---------|
| Glioma | 95% | 93% | 93.74% |
| Meningioma | 94% | 91% | 91.23% |
| No Tumor | 96% | 97% | 97.12% |
| Pituitary | 85% | 99% | 96.19% |

---

## Safety Optimization

To reduce the risk of missed tumors:

- Confidence Threshold = 0.95
- Applied specifically to "No Tumor" predictions
- Model predicts "No Tumor" only when highly confident

This improves clinical safety by reducing false negatives.

---

## Deployment

The model was deployed using Streamlit.

### Prediction Workflow

1. Upload MRI Scan
2. Preprocess Image
3. Normalize Input
4. CNN Prediction
5. Confidence Evaluation
6. Display Diagnosis

---

## Technologies Used

### Programming

- Python

### Deep Learning

- TensorFlow
- Keras

### Data Processing

- NumPy
- Pandas
- OpenCV

### Visualization

- Matplotlib
- Seaborn

### Deployment

- Streamlit

## Key Achievements

- Removed 1,110 duplicate MRI images
- Built a Custom CNN Architecture
- Achieved 93.31% Accuracy
- Achieved 93.29% Weighted F2 Score
- Implemented Recall-Focused Medical Evaluation
- Added Confidence Threshold Safety Mechanism
- Successfully Deployed Using Streamlit

---

## Future Improvements

- K-Fold Cross Validation
- External Dataset Validation
- Grad-CAM Explainability
- Multi-Hospital Generalization Testing
- Model Monitoring and MLOps Pipeline

---

## Limitations

- Single dataset training
- Limited external validation
- No Grad-CAM explainability implementation
- Real-world MRI variability may impact performance

---

## Author

### Sriram B

Data Analyst | Data Scientist | AI Engineer

GitHub: (https://github.com/sriramb0105)

LinkedIn: (https://www.linkedin.com/in/sriram-b-129b003ba/)

---

## Disclaimer

This project is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.
