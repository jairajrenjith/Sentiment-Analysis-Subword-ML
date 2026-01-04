import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("🔹 Loading dataset...")
df = pd.read_csv("data/Sentiment_Analysis.csv")

print("📊 Available columns:", list(df.columns))

# 🔹 Auto-detect columns (same logic as custom classifier)
TEXT_CANDIDATES = ["text", "review", "sentence", "comment"]
LABEL_CANDIDATES = ["label", "sentiment", "polarity", "target"]

text_col = None
label_col = None

for c in TEXT_CANDIDATES:
    if c in df.columns:
        text_col = c
        break

for c in LABEL_CANDIDATES:
    if c in df.columns:
        label_col = c
        break

if text_col is None or label_col is None:
    raise ValueError(
        "❌ Could not auto-detect text/label columns.\n"
        f"Columns found: {list(df.columns)}"
    )

print(f"✅ Using text column: '{text_col}'")
print(f"✅ Using label column: '{label_col}'")
print(f"📊 Dataset loaded with {len(df)} samples")

print("🔹 Loading GloVe embeddings...")

def load_glove_embeddings(path):
    embeddings = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=float)
            embeddings[word] = vector
    return embeddings

glove_path = "embeddings/glove.6B.50d.txt"
embeddings = load_glove_embeddings(glove_path)

EMBED_DIM = len(next(iter(embeddings.values())))
print(f"🧠 GloVe embedding dimension detected: {EMBED_DIM}")

print("🔹 Converting sentences to vectors (GloVe)...")

def sentence_vector(text):
    tokens = str(text).lower().split()
    vecs = [embeddings[t] for t in tokens if t in embeddings]
    return np.mean(vecs, axis=0) if vecs else np.zeros(EMBED_DIM)

X = np.array([sentence_vector(t) for t in df[text_col]])
y = df[label_col].values

print("🔹 Splitting dataset into train & test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🔹 Training Logistic Regression classifier (GloVe)...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("🔹 Making predictions...")
preds = model.predict(X_test)

print("\n✅ CLASSIFICATION RESULTS (GloVe Embeddings)")
print("🎯 Accuracy:", accuracy_score(y_test, preds))
print("\n📄 Detailed Classification Report:")
print(classification_report(y_test, preds))

print("🎉 GloVe-based classification completed successfully!")
