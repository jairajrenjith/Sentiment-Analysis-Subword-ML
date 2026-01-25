# Sentiment Analysis Using Custom Subword Tokenization & Word Embeddings

This project builds a **complete NLP pipeline from scratch** for sentiment analysis.  

It includes:

- Custom **Subword Tokenizer (BPE)** implementation  
- Training **custom word embeddings**
- Building a **text classification model**
- Comparing performance with **pre-trained embeddings**
- Visualizing learned embeddings

---

## Project Objective

The goal of this project is to understand how modern NLP systems work internally by implementing key components manually instead of relying entirely on pre-built libraries.

We:
1. Built a **Byte Pair Encoding (BPE) subword tokenizer from scratch**
2. Trained **Word2Vec embeddings** on subword tokens
3. Used these embeddings in a **neural network classifier**
4. Compared performance with **pre-trained spaCy embeddings**

---

## Project Structure

```
Sentiment-Analysis-Subword-ML/
│
├── data/
│   ├── Sentiment_Analysis.csv              # Main dataset
│   └── imdb_dataset.csv                    # External IMDB dataset
│
├── notebooks/
│   ├── 00_download_imdb.ipynb              # Downloads IMDB dataset
│   ├── 01_custom_tokenizer.ipynb           # Custom BPE tokenizer
│   ├── 02_train_embeddings.ipynb           # Train Word2Vec embeddings
│   ├── 03_custom_classification.ipynb      # Custom embedding classifier
│   ├── 04_pretrained_comparison.ipynb      # Pretrained embedding comparison
│   └── 05_embedding_visualization.ipynb    # PCA visualization
│
├── outputs/
│   ├── subword_vocab.json                  # Saved tokenizer vocabulary
│   ├── custom_embeddings.vec               # Trained word vectors
│   └── custom_model.pt                     # Trained classification model
│
├── README.md                               # README.md file
└── requirements.txt                        # Project dependencies
```

---

## Installation & Setup

### 1. Clone the Repository

  ```bash
  git clone <repo-link>
  cd Sentiment-Analysis-Subword-ML
  ```


### 2. Create Virtual Environment

  ```bash
  python -m venv venv
  ```

Activate it:

a. Windows

  ```bash
  venv\Scripts\activate
  ```

b. Mac/Linux

  ```bash
  source venv/bin/activate
  ```


### 3. Install Dependencies

  ```bash
  pip install -r requirements.txt
  ```


### 4. Download spaCy Language Model

  ```bash
  python -m spacy download en_core_web_md
  ```


### 5. Register Jupyter Kernel

  ```bash
  python -m ipykernel install --user --name=venv --display-name "Python (venv)"
  ```

Start Jupyter:

  ```bash
  jupyter notebook
  ```

Select kernel: **Python (venv)**

---

## Dataset Setup

### Main Dataset
Place your provided dataset inside:

  ```bash
  data/Sentiment_Analysis.csv
  ```

It should contain text reviews and sentiment labels.

---

### External Dataset (IMDB)

Run the notebook:

  ```bash
  notebooks/00_download_imdb.ipynb
  ```

This downloads the IMDB dataset and saves:

  ```bash
  data/imdb_dataset.csv
  ```

---

## Execution Order

Run notebooks in this exact order:

  ```bash
  00_download_imdb.ipynb
  01_custom_tokenizer.ipynb
  02_train_embeddings.ipynb
  03_custom_classification.ipynb
  04_pretrained_comparison.ipynb
  05_embedding_visualization.ipynb
  ```

---

## Step 1 — Custom Subword Tokenizer

Implemented **Byte Pair Encoding (BPE)** from scratch:
- Learns subword vocabulary
- Tokenizes words into subword units
- Saves vocabulary as `subword_vocab.json`



## Step 2 — Train Custom Word Embeddings

- Tokenized corpus using the custom tokenizer
- Trained **Word2Vec embeddings**
- Saved vectors as:

  ```bash
  outputs/custom_embeddings.vec
  ```



## Step 3 — Text Classification (Custom Embeddings)

- Converted each text into an average of its subword vectors
- Built a **PyTorch Neural Network**
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-score

Model saved as:

  ```bash
  outputs/custom_model.pt
  ```



## Step 4 — Pretrained Embedding Comparison

Used **spaCy's pre-trained embeddings** and trained a Logistic Regression classifier to compare performance.

This helps evaluate the effectiveness of the custom-built pipeline.



## Step 5 — Embedding Visualization

Used **PCA** to reduce embedding dimensions and visualize how words cluster semantically.

---

## Results

| Pipeline | Description |
|---------|-------------|
| Custom Tokenizer + Custom Embeddings | Fully built from scratch |
| Pretrained spaCy Embeddings | Transfer learning baseline |

Evaluation metrics are printed in the respective notebooks.

---

## Technologies Used

- Python
- PyTorch
- Gensim (Word2Vec)
- spaCy
- Scikit-learn
- Matplotlib
- HuggingFace Datasets

---

## Deliverables Included

✔ Custom Subword Tokenizer  
✔ Saved Vocabulary File  
✔ Custom Word Embeddings  
✔ Classification Model  
✔ Evaluation Results  
✔ Pretrained Comparison  
✔ Embedding Visualization  

---

## Conclusion

This project demonstrates a **full NLP workflow from scratch**, providing deeper insight into how modern language models process text at the subword level and learn semantic representations.

---

By Jairaj R.
