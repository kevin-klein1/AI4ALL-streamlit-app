#  Breast Cancer Histopathology Classifier
*A simple ResNet-18 model for classifying breast tumor images.*

## Background
This project uses deep learning to classify breast cancer histopathology images as **benign** or **malignant**. The model is built with **ResNet-18**, a lightweight CNN that is effective for medical image tasks. The images come from the **BreaKHis 400X** dataset, which contains microscopic tumor tissue samples.

## Model & Methods
- **Model:** ResNet-18 (fine-tuned from ImageNet)  
- **Data:** BreaKHis 400X (benign vs. malignant)  
- **Steps:** image resizing, normalization, and light data augmentation  
- **Training:** binary classification using cross-entropy loss and the Adam optimizer  

## Purpose
The goal is to show how deep learning can help analyze histopathology slides by:
- Giving quick predictions  
- Supporting consistency in classification  
- Demonstrating AI-assisted screening

## Performance
The model reaches an **accuracy of ~96.15%** on the evaluation dataset.

> *Not intended for clinical diagnosis.*

## How to Use
1. Upload a histopathology image.  
2. Choose the true label.  
3. Click **Run Prediction**.  
4. View the model output and accuracy tracking.

---

