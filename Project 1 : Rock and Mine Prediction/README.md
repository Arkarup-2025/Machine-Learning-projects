# Rock vs Mine Prediction using Logistic Regression

This machine learning project classifies sonar signals as either **Rock (R)** or **Mine (M)** using **Logistic Regression**. The model is trained on the [UCI Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)).

## 🔧 Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn

## 📂 Dataset
- 60 features representing sonar signal strength at different angles
- Binary output: **R** = Rock, **M** = Mine
- **Place `Sonar data.csv` in the project directory before running the script**

## 📌 Steps Involved
1. Load and explore the dataset
2. Preprocess features using `StandardScaler`
3. Train a `LogisticRegression` model
4. Evaluate accuracy on training and test sets
5. Predict new instances based on signal values

## ✅ Accuracy
- **Training Accuracy**: ~91%
- **Test Accuracy**: ~76%

## 💡 How to Use

1. Clone the repo:
    ```bash
    git clone https://github.com/Arkarup-2025/rock-vs-mine.git
    cd rock-vs-mine
    ```

2. (Optional) Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the script:
    ```bash
    python sonar_data_prediction.py
    ```

## 📈 Sample Prediction

```python
The object is a Rock
