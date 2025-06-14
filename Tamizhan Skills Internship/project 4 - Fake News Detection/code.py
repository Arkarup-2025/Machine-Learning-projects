import nltk
import pandas as pd
import numpy as np
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, classification_report


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Load TRUE and FAKE datasets
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# Add labels
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# Combine and shuffle
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Use only the 'text' column
df = df[['text', 'label']]

# Clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    
    # Optional: remove stopwords using sklearn or manual list
    stop_words = set([
        'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'is', 'was', 'were', 
        'this', 'that', 'to', 'from', 'by', 'with', 'for', 'as', 'of', 'it', 'its'
    ])
    filtered = [word for word in tokens if word not in stop_words]
    
    return " ".join(filtered)

df['clean_text'] = df['text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# === Model 1: PassiveAggressiveClassifier ===
print("=== Passive Aggressive Classifier ===")
pac = PassiveAggressiveClassifier(max_iter=1000)
pac.fit(X_train_tfidf, y_train)
y_pred_pac = pac.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred_pac))
print("F1 Score:", f1_score(y_test, y_pred_pac, pos_label='FAKE'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_pac))

# === Model 2: SVM ===
print("\n=== Support Vector Machine (SVM) ===")
svm_clf = svm.LinearSVC()
svm_clf.fit(X_train_tfidf, y_train)
y_pred_svm = svm_clf.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("F1 Score:", f1_score(y_test, y_pred_svm, pos_label='FAKE'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))
