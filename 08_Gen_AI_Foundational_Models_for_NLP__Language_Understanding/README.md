# Course 8: Gen AI Foundational Models for NLP & Language Understanding

This folder contains coursework and projects completed for the **[Gen AI Foundational Models for NLP & Language Understanding](https://www.coursera.org/learn/gen-ai-foundational-models-for-nlp-and-language-understanding?specialization=ai-engineer)** course, part of the [IBM AI Engineer Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer) on Coursera.

## 🧠 Course Description

This course provides a comprehensive introduction to foundational models in natural language processing (NLP) and language understanding. Learners explore various techniques for text representation and model architectures used in NLP tasks.

By the end of this course, you will be able to:

- Explain how to use one-hot encoding, bag-of-words, embedding, and embedding bags to convert words to features.
- Build and use Word2Vec models for contextual embedding.
- Build and train a simple language model with a neural network.
- Utilize N-gram and sequence-to-sequence models for document classification, text analysis, and sequence transformation.

---

## 📂 Contents: The coding projects I worked on

- `classifying_document.py`: Building a large-scale document classifier with EmbeddingBag layer and softmax output layer, trained on the AG_NEWS dataset with a data loader, visualised using 3D t-SNE. Final test accuracy was 84% over 4 distinct classes.
- `n_gram_analysis_models.py`: Building an N-gram Histogram model in NTLK to uncover word patterns in 90s rap. Utilised monogram, bigram, and trigram analysis of which the latter performs best, but limited compared to more advanced models as we will see next.
- `FNN_LanguageModel.py`: Building a gneral feed forward neural network (FNN) to predict rap words. Compared 2-gram, 4-gram, and 8-gram in terms of their perplexity over training epoch. <br>
<img src="Images/embeddings 4-gram.png" alt="Translation Spanisch<->English.." width="200"/> <br>
<img src="Images/perplexity.png" alt="Translation Spanisch<->English.." width="200"/>
- ``: 

---

## 🔧 Tools and Libraries

- Python
- Jupyter Notebooks
- PyTorch
- NumPy
- Matplotlib

---

## 📌 Certificate Series

This is the eighth course in the [IBM AI Engineer Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer).
