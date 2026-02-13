# Gen AI Foundational Models for NLP & Language Understanding – Project Implementations

This folder contains NLP and foundational language model implementations developed during the  
**IBM AI Engineering Professional Certificate**.

The focus of this module was building practical intuition for text representations, embeddings, 
and early language modeling techniques that underpin modern transformer architectures.

---

## 🧠 Overview

Key areas explored:

- Word embeddings and feature representations (EmbeddingBag, Word2Vec, GloVe)
- N-gram language modeling
- Feed-forward neural language models
- Document classification pipelines
- Embedding visualization using t-SNE

These projects establish the NLP foundations used later in transformer fine-tuning, alignment, 
and retrieval-augmented generation systems.

---

## 📂 Selected Implementations

### 🔹 Document Classification & Embeddings

- `classifying_document.py`  
  Document classifier using EmbeddingBag + softmax trained on the **AG_NEWS** dataset.  
  Achieved **84% test accuracy** across four classes.

---

### 🔹 N-gram & Neural Language Modeling

- `n_gram_analysis_models.py`  
  Monogram, bigram, and trigram analysis exploring linguistic patterns.

- `FNN_LanguageModel.py`  
  Feed-forward neural network predicting next-word probabilities.  
  Compared 2-gram, 4-gram, and 8-gram architectures using perplexity metrics.

<img src="Images/embeddings 4-gram.png" width="220"/>
<img src="Images/perplexity.png" width="220"/>

---

### 🔹 Word Embeddings & Representation Learning

- `Word2VecModels.py`  
  Implemented **CBOW**, **Skip-gram**, and Stanford **GloVe** embeddings.  
  Visualized embedding spaces using t-SNE.

<img src="Images/CBOW_embeddings.png" width="240"/>

- `Word2VecApplications.py`  
  Applied optimized embeddings to AG_NEWS classification  
  (Accuracy: **64.6%** over 10 epochs).

---

## 🔧 Tools & Libraries

Python • PyTorch • NumPy • Matplotlib

---

## 📌 Context

This module builds the NLP foundation within the  
IBM AI Engineering Professional Certificate, supporting later work in 
transformers, LLM alignment, and generative AI systems.
