import numpy as np
import pandas as pd
import json

EMBED_DIM = 50
EPOCHS = 5
LR = 0.01

print("🔹 Loading dataset...")
df = pd.read_csv("data/Sentiment_Analysis.csv")
texts = df["text"].astype(str).tolist()

print("🔹 Loading subword vocabulary...")
with open("tokenizer/subword_vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)

word2idx = vocab
idx2word = {v: k for k, v in vocab.items()}

print(f"🔹 Vocabulary size: {len(vocab)}")

# Initialize embeddings
embeddings = np.random.randn(len(vocab), EMBED_DIM)

print("🔹 Training custom embeddings...")

for epoch in range(EPOCHS):
    for text in texts:
        for token in text.lower().split():
            if token in word2idx:
                embeddings[word2idx[token]] += LR
    print(f"Epoch {epoch + 1} completed")

print("🔹 Saving embeddings to file...")

# 🔥 IMPORTANT FIX: encoding="utf-8"
with open("embeddings/custom_embeddings.txt", "w", encoding="utf-8") as f:
    for word, idx in word2idx.items():
        vector = " ".join(map(str, embeddings[idx]))
        f.write(f"{word} {vector}\n")

print("✅ Embeddings saved successfully!")
print("📁 File: embeddings/custom_embeddings.txt")
