import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
#Data Collection and Analysis

diabetes_dataset = pd.read_csv('Diabetes.csv') 
# printing the first 5 rows of the dataset
print(diabetes_dataset.head())

# number of rows and Columns in this dataset
print(diabetes_dataset.shape)

# getting the statistical measures of the data
print(diabetes_dataset.describe())

print(diabetes_dataset['Outcome'].value_counts())

print(diabetes_dataset.groupby('Outcome').mean())

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()

standardized_data = scaler.fit_transform(X)

X = standardized_data
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

#Model Evaluation

#Accuracy Score

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f'Accuracy on training data: {training_data_accuracy * 100:.2f}%')

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f'Accuracy on test data: {test_data_accuracy * 100:.2f}%')

#Making a Predictive System
input_data = (5,166,72,19,175,25.8,0.587,51)

# Create a DataFrame with the same column names
input_data_df = pd.DataFrame([input_data], columns=diabetes_dataset.columns[:-1])

# Standardize the input data
std_data = scaler.transform(input_data_df)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
