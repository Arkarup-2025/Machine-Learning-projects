# 🧠 Handwritten Digit Recognition using CNN (MNIST Dataset)

This project is part of my **RISE Internship at Tamizhan Skills**, where I built a **Convolutional Neural Network (CNN)** to classify handwritten digits using the **MNIST dataset**.

---

## 📌 Project Overview

The goal is to train a CNN that can recognize digits (0–9) from grayscale 28x28 pixel images with high accuracy.

---

## 📂 Dataset

- **Source**: `tensorflow.keras.datasets.mnist`  
- **Size**: 70,000 images (60,000 for training, 10,000 for testing)  
- Each image is a 28x28 grayscale image of a handwritten digit.

---

## 🔧 Technologies & Libraries Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 🏗️ Model Architecture

```text
Input Layer: 28x28x1 grayscale images
⬇️
Conv2D (32 filters, 3x3, ReLU)
⬇️
MaxPooling2D (2x2)
⬇️
Conv2D (64 filters, 3x3, ReLU)
⬇️
MaxPooling2D (2x2)
⬇️
Flatten
⬇️
Dense (128 units, ReLU)
⬇️
Dropout (30%)
⬇️
Dense (10 units, Softmax)
