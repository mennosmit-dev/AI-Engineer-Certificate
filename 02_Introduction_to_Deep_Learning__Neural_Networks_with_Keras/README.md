# Introduction to Deep Learning & Neural Networks with Keras – Project Implementations

This folder contains deep learning implementations developed as part of the  
**IBM AI Engineering Professional Certificate**.

The focus of this module was building practical intuition for neural network design, training dynamics, and evaluation using Keras with TensorFlow.

---

## 🧠 Overview

Key areas explored:

- Fully connected neural networks (dense architectures)
- Convolutional neural networks (CNNs)
- Basic transformer architectures
- Regularization techniques (dropout, batch normalization)
- Hyperparameter tuning and model evaluation

These projects establish the deep learning foundation supporting later work in reinforcement learning, transformers, and applied ML systems.

---

## 📂 Selected Implementations

### 🔹 Neural Networks & Regression

- `concrete_compressive_strength.py`  
  Multi-layer dense network for regression (Validation MSE: **25.89**).

- `conventionalnn_mnist`  
  Fully connected network achieving **99.5% accuracy** on MNIST.

---

### 🔹 Convolutional Neural Networks

- `convolutionalnn_mnist`  
  CNN with two convolution + pooling blocks achieving **99.77% accuracy**.
  
- `aviation_damage`  
  Aircraft damage classification using fine-tuned **VGG16** and **ResNet**.  
  Best validation accuracy: **68.8%**.  
  Extended with caption generation using a BLIP transformer model.

<img src="Images/boeing747.png" width="200"/>
<img src="Images/crack.png" width="200"/>
<img src="Images/captionandsummary.png" width="350"/>

---

### 🔹 Transformers

- `transformer_seq2seq_translation`  
  Sequence-to-sequence transformer for English → Spanish translation.  
  Achieved **100% accuracy** on a controlled benchmark dataset.

---

## 🔧 Tools & Libraries

Python • TensorFlow • Keras • NumPy • Matplotlib • Jupyter

---

## 📌 Context

This module forms the deep learning foundation within the  
IBM AI Engineering Professional Certificate and supports later work in reinforcement learning, LLM systems, and production-oriented ML pipelines.
