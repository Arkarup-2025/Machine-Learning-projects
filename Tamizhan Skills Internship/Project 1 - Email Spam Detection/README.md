
# 📧 Spam vs Ham Classifier using Naive Bayes & SVM

This project builds a machine learning pipeline to classify SMS messages as **spam** or **ham (not spam)** using Natural Language Processing (NLP) techniques. It includes preprocessing, vectorization, model training (Naive Bayes & SVM), evaluation, and saving models for deployment.

---

## 📂 Dataset

- **File**: `spam_ham_dataset.csv`
- **Columns**: `label`, `text` (along with other unused columns)
- **Labels**: `spam`, `ham`

---

## ✅ Features

- Clean and preprocess SMS text
- Remove stopwords using NLTK
- Convert text into TF-IDF features
- Train two models:
  - Multinomial Naive Bayes
  - Linear SVM (LinearSVC)
- Evaluate with metrics:
  - Accuracy, Precision, Recall, F1-score
- Save models and vectorizer using `joblib`

---

## 🧪 Libraries Used

```python
pandas
numpy
re
string
sklearn
nltk
joblib
```

---

## 🛠️ Project Steps

### 1️⃣ Import Libraries

```python
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
```

---

### 2️⃣ Load Dataset

```python
df = pd.read_csv('spam_ham_dataset.csv', encoding='latin-1')
df.columns = ['Unnamed: 0', 'label', 'text', 'label_num']
```

---

### 3️⃣ Text Cleaning

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)
```

---

### 4️⃣ Stopword Removal

```python
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

df['clean_text'] = df['clean_text'].apply(remove_stopwords)
```

---

### 5️⃣ TF-IDF Vectorization

```python
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_text']).toarray()
```

---

### 6️⃣ Encode Labels

```python
y = np.where(df['label'].str.lower() == 'spam', 1, 0)
```

---

### 7️⃣ Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 8️⃣ Train Models

**Naive Bayes**

```python
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
```

**SVM**

```python
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
```

---

### 9️⃣ Evaluation

```python
def evaluate_model(y_test, y_pred, model_name):
    print(f"\n📊 Evaluation for {model_name}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
    print(f"Recall:    {recall_score(y_test, y_pred) * 100:.2f}%")
    print(f"F1 Score:  {f1_score(y_test, y_pred) * 100:.2f}%")
    
evaluate_model(y_test, y_pred_nb, "Naive Bayes")
evaluate_model(y_test, y_pred_svm, "SVM (LinearSVC)")
```

---

### 🔟 Save Trained Models

```python
import joblib
joblib.dump(svm_model, 'svm_spam_classifier.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
```

---

## 🧾 Sample Output

```
📊 Evaluation for Naive Bayes
Accuracy:  97.36%
Precision: 95.32%
Recall:    91.64%
F1 Score:  93.44%

📊 Evaluation for SVM (LinearSVC)
Accuracy:  98.14%
Precision: 96.87%
Recall:    93.83%
F1 Score:  95.32%

✅ Models saved! Ready for integration into web app.
```

---

## 🚀 How to Run

1. Make sure Python 3.x is installed.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn nltk joblib
   ```
3. Run the Python script in your IDE or terminal.

---

## 🧠 Future Ideas

- Add deep learning models (e.g., LSTM, BERT)
- Add GUI or web app using Flask/Streamlit
- Deploy on Heroku or Hugging Face Spaces

---

## 👨‍💻 Author

**Arkarup Kundu**  
Passionate about AI, ML & NLP | [GitHub](https://github.com/Arkarup-2025)
