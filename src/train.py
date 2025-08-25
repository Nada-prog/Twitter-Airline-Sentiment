import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Ensure NLTK stopwords are downloaded
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# --- 1. Load Dataset ---

df = pd.read_csv("data/Tweets.csv")

# --- 2. Use only needed columns ---
df = df[["text", "airline_sentiment"]]
df = df.dropna()

# --- 3. Preprocessing Function ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)  # remove numbers
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["cleaned"] = df["text"].apply(clean_text)

# --- 4. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned"], df["airline_sentiment"], test_size=0.2, random_state=42, stratify=df["airline_sentiment"]
)

# --- 5. Vectorize ---
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- 6. Train Model ---
model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

print("✅ Training Done!")
print("Accuracy on train:", model.score(X_train_vec, y_train))
print("Accuracy on test :", model.score(X_test_vec, y_test))

# --- 7. Save Model + Vectorizer ---
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("✅ Model & Vectorizer Saved in models/")
