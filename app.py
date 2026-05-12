import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import nltk
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -----------------------------
# Download NLTK Data
# -----------------------------

nltk.download('stopwords')

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("dataset/fake_or_real_news.csv")

# -----------------------------
# Rename Columns
# -----------------------------

df.columns = ['Unnamed', 'Title', 'Text', 'Label']

# -----------------------------
# Convert Labels
# -----------------------------

df['Label'] = df['Label'].replace({
    'FAKE': 0,
    'REAL': 1
})

# -----------------------------
# Select Features
# -----------------------------

messages = df['Text']
y = df['Label']

# -----------------------------
# Text Preprocessing
# -----------------------------

ps = PorterStemmer()

corpus = []

for i in range(len(messages)):

    review = re.sub('[^a-zA-Z]', ' ', str(messages[i]))

    review = review.lower()

    review = review.split()

    review = [
        ps.stem(word)
        for word in review
        if word not in stopwords.words('english')
    ]

    review = ' '.join(review)

    corpus.append(review)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(corpus).toarray()

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------

model = LogisticRegression()

model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Custom Prediction
# -----------------------------

news = input("\nEnter News Text:\n")

news = re.sub('[^a-zA-Z]', ' ', news)

news = news.lower()

news = news.split()

news = [
    ps.stem(word)
    for word in news
    if word not in stopwords.words('english')
]

news = ' '.join(news)

news_vector = vectorizer.transform([news]).toarray()

prediction = model.predict(news_vector)

if prediction[0] == 1:
    print("\nPrediction: REAL NEWS")
else:
    print("\nPrediction: FAKE NEWS")