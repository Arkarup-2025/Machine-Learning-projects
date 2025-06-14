import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load datasets
train_df = pd.read_csv("loan-train.csv")
test_df = pd.read_csv("loan-test.csv")

# Drop Loan_ID
train_df.drop(columns=['Loan_ID'], inplace=True)
test_df.drop(columns=['Loan_ID'], inplace=True)

# Fill missing values in train
fill_values_train = {
    'Gender': train_df['Gender'].mode()[0],
    'Married': train_df['Married'].mode()[0],
    'Dependents': train_df['Dependents'].mode()[0],
    'Self_Employed': train_df['Self_Employed'].mode()[0],
    'LoanAmount': train_df['LoanAmount'].median(),
    'Loan_Amount_Term': train_df['Loan_Amount_Term'].mode()[0],
    'Credit_History': train_df['Credit_History'].mode()[0]
}
train_df = train_df.fillna(fill_values_train)

# Fill missing values in test
fill_values_test = {
    col: (train_df[col].mode()[0] if train_df[col].dtype == 'object' else train_df[col].median())
    for col in test_df.columns if col in train_df.columns
}
test_df = test_df.fillna(fill_values_test)

# Encode categorical variables
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
encoder = LabelEncoder()
for col in categorical_cols:
    if col in train_df.columns:
        train_df[col] = encoder.fit_transform(train_df[col])
    if col in test_df.columns:
        test_df[col] = encoder.transform(test_df[col])

# Encode target variable
train_df['Loan_Status'] = train_df['Loan_Status'].map({'Y': 1, 'N': 0})

# Features and target
X = train_df.drop(columns=['Loan_Status'])
y = train_df['Loan_Status']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_val)

# Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)

# Evaluation
print("Logistic Regression Classification Report:\n", classification_report(y_val, log_preds))
print("Random Forest Classification Report:\n", classification_report(y_val, rf_preds))

# ROC Curve for Logistic Regression
log_probs = log_model.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, log_probs)
roc_auc = roc_auc_score(y_val, log_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()

# Prepare test set for prediction
test_df = test_df.reindex(columns=X.columns, fill_value=0)
test_preds = rf_model.predict(test_df)

# Show sample predictions
print("Sample Test Predictions (0 = Rejected, 1 = Approved):")
print(test_preds[:10])
