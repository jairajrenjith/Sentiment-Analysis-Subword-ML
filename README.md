# Sentiment Analysis Using Machine Learning (Subword Tokenization)

This project implements a complete end-to-end NLP pipeline for sentiment analysis. It includes a custom subword tokenizer (Byte Pair Encoding – BPE), custom-trained word embeddings, and a machine learning classifier, with a comparison against pre-trained GloVe embeddings.

The project is implemented in two ways:
1. Script-based approach using Python files
2. Notebook-based approach using a single Jupyter notebook

---

## Objectives

- Implement a custom subword tokenizer from scratch (BPE)
- Train custom word embeddings
- Perform sentiment classification using a machine learning model
- Evaluate using Accuracy, Precision, Recall, and F1-score
- Compare performance with pre-trained GloVe embeddings

---

## Project Structure

```
Sentiment-Analysis-Subword-ML/
├── data/
│   └── Sentiment_Analysis.csv
│
├── tokenizer/
│   ├── bpe_tokenizer.py
│   └── subword_vocab.json
│
├── embeddings/
│   ├── train_embeddings.py
│   ├── custom_embeddings.txt
│   └── glove.6B.50d.txt
│
├── models/
│   ├── classifier_custom.py
│   └── classifier_glove.py
│
├── evaluation/
│   └── results.txt
│
├── .gitignore
├── run_tokenizer.py
├── run.ipynb
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Create Virtual Environment

  ```bash
  py -m venv venv
  venv\Scripts\activate
  ```

### 2. Install Dependencies

  ```bash
  pip install -r requirements.txt
  ```

---

## GloVe Embeddings Download

Link : https://www.kaggle.com/datasets/watts2/glove6b50dtxt

After downloading, place the file at:

  ```bash
  embeddings/glove.6B.50d.txt
  ```

---

## How to Run (Script-Based – Way 1)

  ```bash
  python run_tokenizer.py
  python embeddings/train_embeddings.py
  python models/classifier_custom.py
  python models/classifier_glove.py
  ```

## How to Run (Notebook-Based – Way 2)

  ```bash
  jupyter notebook run.ipynb
  ```

---

## Results Summary

Model                                | Accuracy  | F1-score (weighted avg) 
-------------------------------------|-----------|-------------------------
Custom Tokenizer + Custom Embeddings | 0.5810625 | 0.58
Pre-trained GloVe Embeddings         | 0.6755625 | 0.68 

Conclusion:
Pre-trained GloVe embeddings outperform custom embeddings due to richer semantic knowledge learned from large external corpora. However, the custom pipeline successfully demonstrates a complete NLP workflow built from scratch.

---

## Notes

- A ConvergenceWarning may appear during Logistic Regression training.
- This is normal for unscaled embedding features.
- The model still produces valid evaluation metrics.

---

## License

This project is intended for educational and academic purposes only.

---

By Jairaj R.
