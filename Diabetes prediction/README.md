
# 🩺 Diabetes Prediction using Machine Learning (SVM)

This project implements a machine learning model to predict whether a person has diabetes based on various medical features using the **Support Vector Machine (SVM)** classifier. It is based on the **Pima Indians Diabetes Dataset**.

---

## 📂 Dataset

The dataset used is `Diabetes.csv`, which contains medical data of female patients of Pima Indian heritage. It includes the following features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0 = Non-diabetic, 1 = Diabetic)

---

## ✅ Project Workflow

1. **Data Collection and Exploration**
   - Load and inspect the dataset
   - Check statistical details and class distribution

2. **Data Preprocessing**
   - Feature-label separation
   - Standardization using `StandardScaler`

3. **Model Building**
   - Split dataset into training and testing sets
   - Train a linear SVM classifier

4. **Model Evaluation**
   - Evaluate accuracy on both training and test data

5. **Prediction System**
   - Build a prediction system to classify new input data

---

## 📊 Libraries Used

- `numpy`
- `pandas`
- `scikit-learn` (`sklearn`)

You can install dependencies using:

```bash
pip install numpy pandas scikit-learn
```

---

## 🚀 How to Run

1. Clone this repository or download the files.
2. Ensure `Diabetes.csv` is in the same directory.
3. Run the script using any Python IDE (e.g., VSCode, Jupyter Notebook, Spyder) or directly via terminal:

```bash
python diabetes_prediction.py
```

---

## 🔍 Sample Input for Prediction

```python
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
```

### Output:
```
The person is diabetic
```

---

## 📈 Accuracy

- **Training Accuracy**: ~ (depends on data)
- **Testing Accuracy**: ~ (depends on data)

You can expect an accuracy of around 75–80% depending on preprocessing and model tuning.

---

## 🛠️ Future Improvements

- Implement cross-validation
- Add GUI using Streamlit or Flask
- Improve performance using ensemble methods (Random Forest, XGBoost)
- Handle class imbalance with SMOTE or class weighting

---

## 🙌 Acknowledgements

- [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Developed using `scikit-learn` and `pandas`

---

## 📜 License

This project is open-source and available under the MIT License.
