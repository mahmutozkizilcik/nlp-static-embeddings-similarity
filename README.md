# Static Word Embeddings for Similarity Analysis and Text Classification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gensim](https://img.shields.io/badge/Gensim-4.3%2B-F7D54E.svg)](https://radimrehurek.com/gensim/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)

An implementation and analytical evaluation of **Static Word Embeddings (`word2vec-google-news-300`)** across lexical similarity analysis, sentence representation vectorization, embedding dimension decomposition, and downstream sentiment classification on the IMDB dataset.

---

## Overview

Static word embeddings map natural language vocabulary into dense vector spaces where semantic similarity corresponds to geometric proximity. This project explores foundational vector space NLP methods across two core modules:

1. **Part I: Semantic Similarity & Dimension Analysis (`starter_part1.py`)**
   * **Contextual Lexical Replacement**: Replaces target tokens in a sentence with their top-ranked cosine-similar alternatives while preserving sentence structure.
   * **Sentence Vectorization**: Aggregates word vectors via unweighted mean pooling to compute fixed-length sentence embeddings.
   * **Sentence Retrieval**: Ranks candidate sentences from a corpus against a query sentence vector using cosine similarity.
   * **Dimension Contribution Decomposition**: Analyzes pairwise word vectors to determine how individual latent dimensions positively or negatively impact semantic similarity.

2. **Part II: Document Classification (`starter_part2.py`)**
   * **Document Vectorization**: Transforms variable-length IMDB movie reviews into 300-dimensional dense document representations via token averaging.
   * **Supervised Sentiment Classification**: Trains and evaluates a linear classifier (`LogisticRegression`) to distinguish positive vs. negative sentiment reviews.

---

## Repository Structure

```text
├── starter_part1.py                 # Core implementation for Part I (Similarity & Dimension Analysis)
├── starter_part2.py                 # Core implementation for Part II (Document Vectorization & Classifier)
├── sample_input_part1.py            # Execution demonstration script for Part I functions
├── sample_input_part2.py            # Execution demonstration script for Part II classification pipeline
├── requirements.txt                 # Project Python package dependencies
├── Assignment4.pdf                  # Official assignment specification document
├── assignment_text.txt              # Plain text transcript of the assignment description
├── subset10000_IMDB_Dataset.csv     # 10,000 sample subset of IMDB sentiment dataset
├── sentences_sample.txt             # Sample candidate sentences for sentence retrieval
├── word_pairs_sample.txt            # Sample word pairs for dimension analysis
├── sample_output_part1.txt          # Expected reference outputs for Part I
└── sample_output_part2.txt          # Expected reference outputs for Part II
```

---

## Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/mahmutozkizilcik/nlp-static-embeddings-similarity.git
cd nlp-static-embeddings-similarity
pip install -r requirements.txt
```

### 2. Pre-trained Model Loading

The framework relies on Google's pre-trained Word2Vec vectors (`word2vec-google-news-300`). On initial execution, `gensim.downloader` will automatically fetch and cache the model (~1.6 GB) to your local environment:

```python
import gensim.downloader
model = gensim.downloader.load("word2vec-google-news-300")
```

### 3. Running Execution Demonstrations

To run the similarity analysis pipeline (Part I):
```bash
python sample_input_part1.py
```

To execute the IMDB sentiment classification pipeline (Part II):
```bash
python sample_input_part2.py
```

---

## Methodology & Highlights

* **Cosine Similarity vs. Dot Product**: Word replacement and sentence ranking utilize normalized cosine similarity $S_c(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$ to ensure invariance to vector magnitude differences.
* **Out-of-Vocabulary (OOV) Handling**: During sentence/document mean pooling, tokens absent from the pre-trained vocabulary are dynamically skipped to maintain representation integrity.
* **Dimensional Interpretability**: The dimension contribution analysis inspects element-wise products $u_i \times v_i$, identifying specific latent axes that align or diverge across word concepts.

---

## Academic Context

Developed as coursework for **AIN442 / BBM497 - Natural Language Processing Practicum & Lab** at Hacettepe University.
