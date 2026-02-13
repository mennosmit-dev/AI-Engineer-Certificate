# Machine Learning with Python – Project Implementations

This folder contains foundational machine learning implementations developed during the  
**IBM AI Engineering Professional Certificate**.

The focus of this module was building practical intuition for classical ML algorithms, 
model evaluation, feature engineering, and pipeline design — forming the statistical 
foundation behind my applied ML and reinforcement learning work.

---

## 🧠 Overview

Implemented a wide range of supervised and unsupervised learning techniques:

- Regression and classification (Logistic Regression, SVM, KNN, Trees, Random Forest, XGBoost)
- Clustering (K-Means, DBSCAN, HDBSCAN)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Regularization (Ridge, Lasso)
- Pipelines with GridSearchCV
- Model evaluation using ROC-AUC, F1-score, confusion matrices, and clustering metrics

---

## 📂 Selected Implementations

### 🔹 Regression & Classification

- `decision_trees.py` – Drug prescription classifier (**98.3% accuracy**)  
- `decision_tree_svm_ccfraud.py` – Credit card fraud detection (**ROC-AUC: 0.986**)  
- `multi_class_classification.py` – Obesity prediction (**76% → 92% accuracy improvement**)

<img src="Images/decision_trees_new.png" width="200"/>
<img src="Images/feature_importance.png" width="200"/>

---

### 🔹 Tree-Based Models & Ensembles

- `random__forests__xgboost.py` – Housing price prediction (XGBoost MSE: **0.2226**)  
- `evaluating_random_forest.py` – Feature importance and diagnostics

---

### 🔹 Clustering & Dimensionality Reduction

- `k-means-customer-seg.py` – Customer segmentation  
- `comparing_dbscan_hdbscan.py` – Density-based clustering comparison  
- `pca.py` – PCA projection (72% variance explained)

<img src="Images/Education_Age_Income.png" width="220"/>
<img src="Images/DBSCAN.png" width="200"/>
<img src="Images/HDBSCAN.png" width="200"/>

---

### 🔹 Model Evaluation & Pipelines

- `ml_pipelines_and_gridsearchcv.py` – Pipeline + hyperparameter tuning  
- `evaluating_classification_models.py` – Breast cancer benchmarking

<img src="Images/confusion.png" width="200"/>
<img src="Images/Hyperparam_Tune.png" width="200"/>

---

### 🔹 Applied Projects

- `practice_project` – Titanic survival prediction  
- `finalproject_ausweather_.py` – Rainfall prediction pipeline (**83% accuracy**)

<img src="Images/weather_drivers.png" width="220"/>

---

## 🔧 Tools & Libraries

Python • Scikit-learn • Pandas • NumPy • Matplotlib • Jupyter

---

## 📌 Context

This module forms the classical ML foundation within the  
IBM AI Engineering Professional Certificate and complements my work in deep learning, reinforcement learning, and production ML systems.
