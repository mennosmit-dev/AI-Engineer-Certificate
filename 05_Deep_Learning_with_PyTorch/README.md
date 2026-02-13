# Deep Learning with PyTorch – Project Implementations

This folder contains advanced deep learning implementations developed during the  
**IBM AI Engineering Professional Certificate**.

The focus of this module was building practical PyTorch workflows for CNNs, 
sequence models, transfer learning, and training optimization techniques.

---

## 🧠 Overview

Key areas explored:

- Convolutional neural networks (CNNs) and feature extraction
- Model initialization strategies and batch normalization
- Regularization techniques (dropout, normalization)
- Training optimization and architectural experimentation
- Applied image classification using real-world datasets

Several smaller lab-style experiments were completed to build intuition around 
PyTorch mechanics and training dynamics; only the more applied implementations 
are highlighted below.

---

## 📂 Selected Implementations

### 🔹 Model Initialization & Training Behavior

- `different_mnist_initalisations.py`  
  Compared initialization strategies on MNIST.  
  Xavier initialization achieved **63% accuracy**, outperforming default (23%) and uniform (12%).

- `normalised_batch_mnist.py`  
  Investigated batch normalization effects (+2% improvement over strong baseline).

---

### 🔹 Convolutional Neural Networks

- `cnn_mnist.py`  
  Two-layer CNN architecture achieving **95% accuracy** on MNIST.

- `cnn_mnist_batch.py`  
  CNN with batch normalization improving accuracy to **98%**.

- `cnn_fashion_mnist.py`  
  Fashion-MNIST classification with performance visualization.

- `cnn_anime.py`  
  Anime character classification using CNN architectures and LeakyReLU activations.  
  Achieved **100% accuracy** on controlled subsets.

---

## 🔧 Tools & Libraries

Python • PyTorch • Torchvision • Torchtext • NumPy • Matplotlib

---

## 📌 Context

This module extends the PyTorch foundation established earlier in the program, 
bridging core neural network concepts with more applied deep learning workflows 
used later in reinforcement learning and transformer-based systems.
