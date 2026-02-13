# Introduction to Deep Learning & Neural Networks with Keras – Project Implementations

This folder contains deep learning implementations developed as part of the  
**IBM AI Engineering Professional Certificate**.

The focus of this module was building practical intuition for neural network design, training dynamics, and model evaluation using Keras with TensorFlow.

---

## 🧠 Overview

Implemented and experimented with:

- Fully connected neural networks (dense architectures)
- Convolutional neural networks (CNNs)
- Basic transformer architectures
- Regularization techniques (dropout, batch normalization, early stopping)
- Hyperparameter tuning (learning rate, batch size, depth)
- Model evaluation via accuracy, MSE, confusion matrices, and learning curves

These projects strengthen the deep learning foundation supporting my later work in reinforcement learning, transformers, and production ML systems.

---

## 📂 Selected Implementations

### 🔹 Neural Networks & Regression

- `concrete_compressive_strength.py`  
  Multi-layer dense network (ReLU + Adam) for regression.  
  Validation MSE: **25.89** (90/10 train-validation split).

- `conventionalnn_mnist`  
  Fully connected neural network for MNIST digit classification.  
  Accuracy: **99.5%**.

---

### 🔹 Convolutional Neural Networks

- `convolutionalnn_mnist`  
  CNN with two convolution + pooling blocks.  
  Accuracy: **99.77%**.

- `aviation_damage`  
  Aircraft damage classification using fine-tuned **VGG16** and **ResNet**.  
  Best validation accuracy: **68.8%**.  
  Extended with image captioning using a pre-trained BLIP transformer.

---

### 🔹 Transformers

- `transformer_seq2seq_translation`  
  Custom sequence-to-sequence transformer for English → Spanish translation.  
  Achieved 100% accuracy on a controlled benchmark dataset.

---

## 🔧 Tools & Libraries

Python • TensorFlow • Keras • NumPy • Matplotlib • Jupyter

---

## 📌 Context

This module forms the deep learning foundation within the  
IBM AI Engineering Professional Certificate and supports my broader work in reinforcement learning, LLM fine-tuning, and applied ML engineering.
