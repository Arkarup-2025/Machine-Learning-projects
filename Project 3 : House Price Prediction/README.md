
# 🏠 California Housing Price Prediction using Machine Learning (XGBoost)

This project implements a machine learning model to predict California housing prices based on census data using the **XGBoost Regressor**. It is based on the **California Housing Dataset** from `sklearn.datasets`.

---

## 📂 Dataset

The dataset used is fetched via `fetch_california_housing()` from `scikit-learn`. It includes the following features:

- MedInc: Median income in block group  
- HouseAge: Median house age in block group  
- AveRooms: Average number of rooms per household  
- AveBedrms: Average number of bedrooms per household  
- Population: Block group population  
- AveOccup: Average house occupancy  
- Latitude: Block group latitude  
- Longitude: Block group longitude  
- **price**: Median house value (target variable)

---

## ✅ Project Workflow

1. **Data Collection and Exploration**
   - Load dataset using `sklearn.datasets`
   - Convert to Pandas DataFrame and inspect
   - Visualize correlation matrix using heatmap

2. **Data Preprocessing**
   - Check for null values
   - Separate features and target
   - Split dataset into training and testing sets

3. **Model Building**
   - Use `XGBRegressor` to train the model
   - Train on 80% of the data

4. **Model Evaluation**
   - Predict on training and test data
   - Evaluate using R² score and Mean Absolute Error (MAE)
   - Visualize actual vs predicted prices using scatter plot

---

## 📊 Libraries Used

- `numpy`  
- `pandas`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  
- `xgboost`

You can install dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

---

## 🚀 How to Run

1. Clone this repository or copy the code files.
2. Make sure you're connected to the internet (to fetch dataset).
3. Run the script using any Python IDE or terminal:

```bash
python house_price_prediction.py
```

---

## 🔍 Sample Output for Evaluation

```
R squared error :  0.956
Mean Absolute Error :  0.236
```

*Scatter plot of actual vs predicted prices is displayed.*

---

## 📈 Accuracy

- **Training Accuracy (R²)**: ~0.95  
- **Testing Accuracy (R²)**: ~0.79

You can expect high accuracy depending on data preprocessing and model tuning.

---

## 🛠️ Future Improvements

- Try other regression models (Random Forest, Linear Regression)
- Feature engineering for improved performance
- Use grid search for hyperparameter tuning
- Save and deploy the model as a web service

---

## 🙌 Acknowledgements

- [California Housing Dataset - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- Developed using `pandas`, `xgboost`, and `scikit-learn`

---

## 📜 License

This project is open-source and available under the MIT License.

