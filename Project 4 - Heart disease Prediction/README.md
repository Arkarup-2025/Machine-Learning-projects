# ❤️ Heart Disease Prediction using Logistic Regression

This project predicts whether a person has heart disease or not using a Logistic Regression model based on medical features.

## 📁 Dataset
The dataset used is `heart.csv`, which contains 13 medical attributes and a target label:
- `target` = 1 (has heart disease)
- `target` = 0 (no heart disease)

## 📊 Features Used
Examples include:
- Age
- Sex
- Resting Blood Pressure
- Cholesterol
- Max Heart Rate
- ST depression, etc.

## ⚙️ Libraries Used
- pandas
- numpy
- scikit-learn (for preprocessing, modeling, evaluation)

## 🧠 Model
- **Logistic Regression** with scaled features and increased iterations (`max_iter=1000`).
- Feature scaling is applied using `StandardScaler` to help the model converge.

## 🎯 Results
- Accuracy on Training Data: ~85–90%
- Accuracy on Test Data: ~80–85% (depending on dataset split)

## 🚀 How to Run
1. Clone the repository
2. Make sure you have the required libraries installed:
   ```bash
   pip install numpy pandas scikit-learn
