# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# 2️⃣ Load Dataset
df = pd.read_csv('spam_ham_dataset.csv', encoding='latin-1')

# View first few rows to understand structure
print(df.head())

print(df.columns)

df.columns = ['Unnamed: 0', 'label', 'text', 'label_num']

# 3️⃣ Preprocessing Function
def clean_text(text):
    text = str(text).lower()                              # Lowercase
    text = re.sub(r'\d+', '', text)                       # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()                                   # Strip whitespace
    return text

# Apply text cleaning
df['clean_text'] = df['text'].apply(clean_text)

# 4️⃣ Stopword Removal
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

df['clean_text'] = df['clean_text'].apply(remove_stopwords)

# 5️⃣ Vectorization (TF-IDF)
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_text']).toarray()

# 6️⃣ Labels Encoding (spam=1, ham=0)
y = np.where(df['label'].str.lower() == 'spam', 1, 0)

# 7️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8️⃣ Model 1: Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# 9️⃣ Model 2: SVM
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# 10️⃣ Evaluation Function
def evaluate_model(y_test, y_pred, model_name):
    print(f"\n📊 Evaluation for {model_name}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
    print(f"Recall:    {recall_score(y_test, y_pred) * 100:.2f}%")
    print(f"F1 Score:  {f1_score(y_test, y_pred) * 100:.2f}%")

# 11️⃣ Results
evaluate_model(y_test, y_pred_nb, "Naive Bayes")
evaluate_model(y_test, y_pred_svm, "SVM (LinearSVC)")

# 12️⃣ Save Models (Optional for Deployment)
import joblib
# Save SVM model
joblib.dump(svm_model, 'svm_spam_classifier.joblib')
# Save TF-IDF Vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')

print("\n✅ Models saved! Ready for integration into web app.")

