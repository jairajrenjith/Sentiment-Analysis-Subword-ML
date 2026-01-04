import pandas as pd
from tokenizer.bpe_tokenizer import BPETokenizer

print("🔹 Loading dataset...")

# Load dataset
df = pd.read_csv("data/Sentiment_Analysis.csv")

# Ensure correct column
if "text" not in df.columns:
    raise ValueError("Dataset must contain a 'text' column")

texts = df["text"].astype(str).tolist()

print(f"🔹 Dataset loaded with {len(texts)} sentences")

# Initialize tokenizer (kept small to be fast)
tokenizer = BPETokenizer(vocab_size=200)

print("🔹 Training BPE tokenizer...")
tokenizer.train(texts)

print("🔹 Saving vocabulary...")
tokenizer.save_vocab("tokenizer/subword_vocab.json")

print("✅ Tokenizer training completed successfully!")
print("📁 Vocabulary saved to tokenizer/subword_vocab.json")

print("\n🔹 Sample Tokenization (10 sentences):\n")

for i in range(min(10, len(texts))):
    print(f"Sentence {i+1}:")
    print(texts[i])
    print("Tokens:")
    print(tokenizer.tokenize(texts[i]))
    print("-" * 60)

print("\n🎉 DONE. You can now proceed to embedding training.")
