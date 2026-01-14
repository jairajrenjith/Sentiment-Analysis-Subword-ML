# Sentiment Analysis Using Machine Learning (Subword Tokenization)

This project implements a complete end-to-end NLP pipeline for sentiment analysis. It includes a custom subword tokenizer (Byte Pair Encoding – BPE), custom-trained word embeddings, and a machine learning classifier, with a comparison against pre-trained GloVe embeddings.

The project is implemented by a notebook-based approach using a single Jupyter notebook.

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
│   └── subword_vocab.json
│
├── embeddings/
│   ├── custom_embeddings.txt
│   └── glove.6B.50d.txt
│
├── .gitignore
├── sentiment_analysis.ipynb
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

## How to Run

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
